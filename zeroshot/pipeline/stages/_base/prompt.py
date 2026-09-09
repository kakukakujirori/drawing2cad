"""Round instructions shared by stages, independent of graph state channels."""

from __future__ import annotations
import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from string import Template
from typing import cast

from langchain_core.messages import HumanMessage, SystemMessage, merge_message_runs
from langchain_core.messages.content import create_text_block
from pydantic import BaseModel

from zeroshot.pipeline.messages import (
    ArtifactPresenter,
    InputManifest,
)
from zeroshot.pipeline.messages.contracts.stages import (
    REASONING_STAGES,
    PipelineStage,
)
from zeroshot.pipeline.messages.contracts.reconstruction import ReconstructionSnapshot, tickets_assigned_to
from zeroshot.pipeline.sandbox import SandboxWorkdir
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
    prompt_context: Mapping[str, str]
    artifact_presenter: ArtifactPresenter
    input_manifest: InputManifest
    workdir: SandboxWorkdir

    def build(self, state: ReconstructionState, stage: PipelineStage, *, include_input: bool, **extra_context: str) -> HumanMessage:
        if validation_error := state.get("stage_validation_error"):
            # A re-ask, so the round's terms and guidelines already stand in
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
            PromptTemplate(stages_dir / stage.value / "prompts" / "guidelines.md"),
        ]
        texts = "\n\n".join(template.render(**context) for template in text_templates)
        instruction = HumanMessage(content_blocks=[create_text_block(texts)])

        if include_input:
            # Build on every invocation so message IDs are never reused.
            source = HumanMessage(
                content_blocks=self.artifact_presenter.build_input_message_blocks(
                    manifest=self.input_manifest,
                    workdir=self.workdir,
                )
            )
            (instruction,) = cast(
                list[HumanMessage], merge_message_runs([instruction, source])
            )

        return instruction


def _assigned_ticket_ids(
    snapshot: ReconstructionSnapshot,
    stage: PipelineStage,
) -> str:
    """Name the tickets this stage owns, so it never has to go looking."""
    if stage not in REASONING_STAGES:
        return "none"
    assigned = tickets_assigned_to(snapshot.open_tickets, stage)
    return ", ".join(ticket.ticket_id for ticket in assigned) or "none"


def build_system_prompt(
    prompt_path: Path,
    context: Mapping[str, str],
    output_schema: type[BaseModel] | None = None,
) -> SystemMessage:
    context = dict(context)

    if output_schema is not None:
        context["output_schema"] = json.dumps(
            output_schema.model_json_schema(),
            indent=2,
        )

    sections_templates = [PromptTemplate(prompt_path)]
    if "reconstruction_path" in context:
        stages_dir = Path(__file__).parent.parent
        sections_templates.append(PromptTemplate(stages_dir / "_base" / "prompts" / "reconstruction_history.md"))
    system_texts = "\n\n".join(template.render(**context) for template in sections_templates)
    return SystemMessage(content_blocks=[create_text_block(system_texts)])
