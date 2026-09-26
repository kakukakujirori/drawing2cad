"""Round instructions shared by stages, independent of graph state channels."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from string import Template
from typing import Literal, cast

from langchain_core.messages import HumanMessage, SystemMessage, merge_message_runs
from langchain_core.messages.content import ContentBlock, create_text_block
from pydantic import BaseModel

from zeroshot.pipeline.messages.artifact import SandboxedArtifact
from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.stages.contracts import ReconstructionSnapshot
from zeroshot.pipeline.stages.interpretation.contracts import DrawingView
from zeroshot.pipeline.stages.tickets.contracts import tickets_assigned_to
from zeroshot.pipeline.stages.types import (
    REASONING_STAGES,
    PipelineStage,
)
from zeroshot.pipeline.workflow.state import ReconstructionState, current_snapshot


@dataclass(frozen=True)
class PromptTemplate:
    path: Path

    def __post_init__(self) -> None:
        if not self.path.is_file():
            raise ValueError(f"prompt not found: {self.path}")

    @property
    def sha256(self) -> str:
        return sha256(self.path.read_bytes()).hexdigest()

    def render(self, **context: str) -> str:
        """Fill the `$name` placeholders, refusing to leave any unfilled."""
        source = self.path.read_text(encoding="utf-8")
        return Template(source).substitute(context).strip()


@dataclass(frozen=True)
class StageInstructions:
    input_artifact: Sequence[DrawingView]
    input_presentation_mode: Literal["path", "image"]
    prompt_context: Mapping[str, str]
    workdir: SandboxWorkdir

    def __post_init__(self) -> None:
        if self.input_presentation_mode not in ("path", "image"):
            raise ValueError(
                f"invalid input_presentation_mode: {self.input_presentation_mode}"
            )

    def build(
        self,
        state: ReconstructionState,
        stage: PipelineStage,
        *,
        include_artifact: bool,
        **extra_context: str,
    ) -> HumanMessage:
        if validation_error := state.get("stage_validation_error"):
            # A re-ask, so the round's instructions already stand in
            # the transcript and only the rejection is new.
            return HumanMessage(
                content_blocks=[
                    create_text_block(
                        f"[{stage.title()} Validation Error]\n"
                        f"Your previous {stage} stage output was rejected. "
                        "Return the corrected complete output using this feedback:\n\n"
                        f"{validation_error}"
                    )
                ]
            )

        snapshot = current_snapshot(state)
        context = {
            **self.prompt_context,
            "current_round": str(snapshot.round),
            "assigned_tickets": _assigned_ticket_ids(snapshot, stage),
            **extra_context,
        }
        stages_dir = Path(__file__).parent.parent
        text_templates = [
            PromptTemplate(stages_dir / "_base" / "prompts" / "coordinate_frames.md"),
            PromptTemplate(stages_dir / stage.value / "prompts" / "round.md"),
        ]
        texts = "\n\n".join(template.render(**context) for template in text_templates)
        instruction = HumanMessage(content_blocks=[create_text_block(texts)])

        if include_artifact:
            (instruction,) = cast(
                list[HumanMessage],
                merge_message_runs([instruction, self.create_artifact_message()]),
            )

        return instruction

    def create_artifact_message(self) -> HumanMessage:
        # host path -> sandbox path
        presented = SandboxedArtifact.of(self.input_artifact, self.workdir)

        lines = ["[Input artifacts]", *presented.listing()]
        blocks: list[ContentBlock] = [create_text_block("\n".join(lines))]

        if self.input_presentation_mode == "image":
            blocks.extend(presented.images())

        return HumanMessage(content_blocks=blocks)


def _assigned_ticket_ids(
    snapshot: ReconstructionSnapshot,
    stage: PipelineStage,
) -> str:
    """Expose evidence image paths even when a stage's jq query omits evidence_renders."""
    if stage not in REASONING_STAGES:
        return "none"
    named = [
        ticket.ticket_id
        + (
            f" (evidence: {', '.join(ticket.evidence_renders)})"
            if ticket.evidence_renders
            else ""
        )
        for ticket in tickets_assigned_to(snapshot.open_tickets, stage)
    ]
    return ", ".join(named) or "none"


def schema_for_prompt(contract: type[BaseModel]) -> str:
    """A contract as JSON Schema, less the titles pydantic derives from field names.

    Serialised the way a provider receives a tool schema: no indent, and `$defs`
    left referenced, so a type cited from several places reads as one.
    """
    schema = contract.model_json_schema()

    def without_titles(node: object) -> object:
        if isinstance(node, dict):
            return {k: without_titles(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [without_titles(v) for v in node]
        return node

    return json.dumps(without_titles(schema))


def build_system_prompt(
    role_path: Path | None,
    context: Mapping[str, str],
    output_schema: type[BaseModel] | None = None,
) -> SystemMessage:
    """Load shared reconstruction context, then role_path when supplied.

    role_path=None omits only the stage role, not the shared context.
    A baseline without reconstruction_path receives only its own role.
    """
    context = dict(context)

    if output_schema is not None:
        context["output_schema"] = schema_for_prompt(output_schema)

    sections_templates = []
    if "reconstruction_path" in context:
        sections_templates.append(
            PromptTemplate(
                Path(__file__).parent / "prompts" / "reconstruction_context.md"
            )
        )
    if role_path is not None:
        sections_templates.append(PromptTemplate(role_path))
    system_texts = "\n\n".join(
        template.render(**context) for template in sections_templates
    )
    return SystemMessage(content_blocks=[create_text_block(system_texts)])
