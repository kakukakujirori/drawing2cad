import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from langchain_core.messages.content import create_text_block
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.pregel import Pregel

from zeroshot.pipeline.messages.manifest import read_dxf_frame
from zeroshot.pipeline.stages._base.prompt import (
    StageInstructions,
    build_system_prompt,
    schema_for_prompt,
)
from zeroshot.pipeline.stages.interpretation.contracts import DrawingInterpretation
from zeroshot.pipeline.stages.interpretation.verify import InterpretationVerifier
from zeroshot.pipeline.stages.tickets.contracts import (
    StageReport,
    TicketAnswers,
    tickets_assigned_to,
)
from zeroshot.pipeline.stages.tickets.verify import TicketVerifier
from zeroshot.pipeline.stages.types import PipelineStage
from zeroshot.pipeline.tools.calculate_drawing_scale import (
    create_calculate_drawing_scale_tool,
)
from zeroshot.pipeline.verification.attempts import AttemptStore
from zeroshot.pipeline.workflow._config import _child_graph_config
from zeroshot.pipeline.workflow.lifecycle import interpretation_baseline
from zeroshot.pipeline.workflow.middleware import VerifyOnWriteMiddleware
from zeroshot.pipeline.workflow.state import ReconstructionState, current_snapshot

type CompiledGraph = Pregel[Any, Any, Any, Any]
type AgentBuilder = partial[CompiledGraph]


@dataclass(frozen=True)
class InterpretationStage:
    agent: CompiledGraph
    instructions: StageInstructions
    interpretation_verifier: InterpretationVerifier
    ticket_verifier: TicketVerifier
    middleware: VerifyOnWriteMiddleware
    input_after_compaction: bool
    dxf_context: str | None = None

    def run(self, state: ReconstructionState, config: RunnableConfig) -> dict[str, Any]:
        snapshot = current_snapshot(state)
        if state.get("stage_validation_error") is None:
            self.interpretation_verifier.reset(
                interpretation_baseline(state["reconstruction"])
            )
            self.middleware.reset()
        if not tickets_assigned_to(snapshot.open_tickets, PipelineStage.INTERPRETATION):
            return {
                "stage_submission": TicketAnswers(
                    stage_report=StageReport(
                        concerns={}, dimension_checks=None, unticketed_changes={}
                    ),
                    responses={},
                )
            }
        self.ticket_verifier.reset(state["reconstruction"])
        previous = state.get("interpretation_state") or {}
        instruction = self.instructions.build(
            state,
            PipelineStage.INTERPRETATION,
            include_artifact=(not previous or self.input_after_compaction),
            interpretation_output_path=str(
                self.instructions.workdir.sandbox_bind_dir
                / self.interpretation_verifier.source_filename
            ),
            interpretation_schema=schema_for_prompt(DrawingInterpretation),
        )
        if self.dxf_context is not None and state.get("stage_validation_error") is None:
            instruction.content = [
                *instruction.content_blocks,
                create_text_block(self.dxf_context),
            ]
        result = self.agent.invoke(
            {
                **previous,
                "messages": [*list(previous.get("messages") or []), instruction],
            },
            config=_child_graph_config(config),
        )
        return {
            "interpretation_state": result,
            "stage_submission": result.get("structured_response"),
        }


def _dxf_context(instructions: StageInstructions) -> str | None:
    originals = []
    for view in instructions.input_artifact:
        if Path(view.file).suffix.lower() != ".dxf":
            continue

        file = instructions.workdir.host_to_sandbox_path(view.file)
        originals.append(
            {
                "view": view.name,
                "file": str(file),
                **read_dxf_frame(instructions.workdir.sandbox_to_host_path(file)),
            }
        )
    if not originals:
        return None
    return (
        "[Native DXF coordinates]\n"
        "The original DXFs below are already registered. Keep their names, "
        "files, roles and full-file regions; add dimension readings. "
        "A drawing unit is a millimetre and nothing is moved, so box_uv is "
        "what you read out of the file. box_mm is that file's full extent.\n"
        + json.dumps({"originals": originals})
    )


def create_interpretation_stage(
    builder: AgentBuilder,
    tools: Sequence[BaseTool],
    role_path: Path | None,
    instructions: StageInstructions,
    prompt_context: dict[str, str],
    attempt_store: AttemptStore,
    interpretation_filename: str = "interpretation.json",
    input_after_compaction: bool = False,
) -> InterpretationStage:
    dxf_context = _dxf_context(instructions)
    interpretation_verifier = InterpretationVerifier(
        workdir=instructions.workdir,
        attempt_store=attempt_store,
        input_artifact=instructions.input_artifact,
        source_filename=interpretation_filename,
    )
    ticket_verifier = TicketVerifier(
        lambda: interpretation_verifier.accepted_interpretation
    )
    middleware = VerifyOnWriteMiddleware(
        interpretation_verifier,
        ticket_verifier=ticket_verifier,
        fingerprint=interpretation_verifier.source_digest,
        refusal=(
            "The interpretation is not ready to submit. Correct the current JSON "
            "and referenced files, read the validation and calibration feedback, "
            "and answer only after the current artifact validates."
        ),
        require_feedback_before_submit=True,
    )
    agent = builder(
        tools=[*tools, create_calculate_drawing_scale_tool()],
        system_prompt=build_system_prompt(
            role_path,
            prompt_context | {"max_turns": builder.keywords["max_turns"]},
            TicketAnswers,
        ),
        output_schema=TicketAnswers,
        extra_middleware=[middleware],
    )
    return InterpretationStage(
        agent,
        instructions,
        interpretation_verifier,
        ticket_verifier,
        middleware,
        input_after_compaction,
        dxf_context,
    )
