from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.pregel import Pregel

from zeroshot.pipeline.messages.contracts.reconstruction import (
    DrawingSubmission,
    tickets_assigned_to,
)
from zeroshot.pipeline.stages._base.prompt import StageInstructions, build_system_prompt
from zeroshot.pipeline.stages.types import PipelineStage
from zeroshot.pipeline.tools import create_calculate_drawing_scale_tool
from zeroshot.pipeline.verification import AttemptStore, DrawingVerifier
from zeroshot.pipeline.workflow._config import _child_graph_config
from zeroshot.pipeline.workflow.middleware import VerifyOnWriteMiddleware
from zeroshot.pipeline.workflow.reconstruction import drawing_baseline
from zeroshot.pipeline.workflow.state import ReconstructionState, current_snapshot

type CompiledGraph = Pregel[Any, Any, Any, Any]
type AgentBuilder = partial[CompiledGraph]


@dataclass(frozen=True)
class DrawingStage:
    agent: CompiledGraph
    instructions: StageInstructions
    verifier: DrawingVerifier
    middleware: VerifyOnWriteMiddleware
    input_after_compaction: bool

    def run(self, state: ReconstructionState, config: RunnableConfig) -> dict[str, Any]:
        snapshot = current_snapshot(state)
        if state.get("stage_validation_error") is None:
            self.verifier.reset(drawing_baseline(state["reconstruction"]))
            self.middleware.reset()
        if not tickets_assigned_to(snapshot.open_tickets, PipelineStage.DRAWINGS):
            return {"stage_submission": DrawingSubmission.unchanged()}

        previous = state.get("drawings_state") or {}
        instruction = self.instructions.build(
            state,
            PipelineStage.DRAWINGS,
            include_artifact=(not previous or self.input_after_compaction),
        )
        result = self.agent.invoke(
            {
                **previous,
                "messages": [
                    *list(previous.get("messages") or []),
                    instruction,
                ],
            },
            config=_child_graph_config(config),
        )
        return {
            "drawings_state": result,
            "stage_submission": result.get("structured_response"),
        }


def create_drawing_stage(
    drawing_agent_builder: AgentBuilder,
    tools: Sequence[BaseTool],
    system_prompt_path: Path | None,
    instructions: StageInstructions,
    prompt_context: dict[str, str],
    attempt_store: AttemptStore,
    feedback_presentation_mode: Literal["none", "path", "image"],
    drawing_filename: str = "drawing.json",
    input_after_compaction: bool = False,
) -> DrawingStage:
    if system_prompt_path is None:
        system_prompt_path = Path(__file__).parent / "prompts" / "role.md"

    drawing_verifier = DrawingVerifier(
        workdir=instructions.workdir,
        attempt_store=attempt_store,
        feedback_presentation_mode=feedback_presentation_mode,
        source_filename=drawing_filename,
    )

    drawing_middleware = VerifyOnWriteMiddleware(
        drawing_verifier,
        refusal=(
            "The drawing artifact is not ready to submit. Inspect the rendered "
            "views, correct drawing.json if needed, and submit only after the "
            "current version has been shown back to you and validates."
        ),
        require_feedback_before_submit=True,
    )

    drawings_agent = drawing_agent_builder(
        # Only this stage measures a raster, so only it is offered the fit.
        tools=[*tools, create_calculate_drawing_scale_tool()],
        system_prompt=build_system_prompt(
            system_prompt_path,
            prompt_context | {"max_turns": drawing_agent_builder.keywords["max_turns"]},
            DrawingSubmission,
        ),
        output_schema=DrawingSubmission,
        extra_middleware=[drawing_middleware],
    )
    return DrawingStage(
        agent=drawings_agent,
        instructions=instructions,
        verifier=drawing_verifier,
        middleware=drawing_middleware,
        input_after_compaction=input_after_compaction,
    )
