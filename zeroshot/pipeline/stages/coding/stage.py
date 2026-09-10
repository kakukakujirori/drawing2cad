from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.pregel import Pregel

from zeroshot.pipeline.messages.contracts.reconstruction import CodingSubmission
from zeroshot.pipeline.messages.contracts.drawings import DrawingSource
from zeroshot.pipeline.messages.contracts.stages import PipelineStage
from zeroshot.pipeline.sandbox import SandboxRunner
from zeroshot.pipeline.stages._base.prompt import StageInstructions, build_system_prompt
from zeroshot.pipeline.verification import (
    AttemptStore,
    CadQueryExecutor,
    OutputVerifier,
    StepRenderer,
)
from zeroshot.pipeline.workflow._config import _child_graph_config
from zeroshot.pipeline.workflow.middleware import VerifyOnWriteMiddleware
from zeroshot.pipeline.workflow.state import ReconstructionState, current_snapshot

type CompiledGraph = Pregel[Any, Any, Any, Any]
type AgentBuilder = partial[CompiledGraph]


@dataclass(frozen=True)
class CodingStage:
    agent: CompiledGraph
    instructions: StageInstructions
    verifier: OutputVerifier
    middleware: VerifyOnWriteMiddleware
    input_after_compaction: bool

    def run(self, state: ReconstructionState, config: RunnableConfig) -> dict[str, Any]:
        snapshot = current_snapshot(state)
        if snapshot.last_completed_stage is not PipelineStage.OPERATIONS:
            raise RuntimeError("coding requires integrated operations")
        drawing = cast(DrawingSource, snapshot.drawings)

        if state.get("stage_validation_error") is None:
            self.verifier.reset()
            self.middleware.reset()

        # The verifier redraws the solid in the views the drawing names, and
        # guesses none. Set here because both the build inside the agent and
        # the one at integration belong to this stage of this round.
        self.verifier.views = [sheet.role for sheet in drawing.orthographic()]

        previous = state.get("coding_state") or {}
        messages = [
            *list(previous.get("messages") or []),
            self.instructions.build(
                state,
                PipelineStage.CODING,
                include_artifact=(not previous or self.input_after_compaction),
            ),
        ]
        result = self.agent.invoke(
            {
                **previous,
                "messages": messages,
            },
            config=_child_graph_config(config),
        )
        return {
            "coding_state": result,
            "stage_submission": result.get("structured_response"),
        }


def create_coding_stage(
    coding_agent_builder: AgentBuilder,
    tools: Sequence[BaseTool],
    system_prompt_path: Path | None,
    instructions: StageInstructions,
    prompt_context: dict[str, str],
    attempt_store: AttemptStore,
    sandbox_runner: SandboxRunner,
    feedback_presentation_mode: Literal["none", "path", "image"],
    output_filename: str = "model.py",
    show_intermediate_returns: bool = False,
    input_after_compaction: bool = False,
) -> CodingStage:
    if system_prompt_path is None:
        system_prompt_path = Path(__file__).parent / "prompts" / "role.md"

    executor = CadQueryExecutor(sandbox_runner=sandbox_runner)
    renderer = StepRenderer()
    coding_verifier = OutputVerifier(
        executor=executor,
        workdir=instructions.workdir,
        renderer=renderer,
        feedback_presentation_mode=feedback_presentation_mode,
        attempt_store=attempt_store,
        source_filename=output_filename,
        show_intermediate_returns=show_intermediate_returns,
    )
    coding_middleware = VerifyOnWriteMiddleware(
        coding_verifier,
        refusal=(
            "The current program must produce a verified solid and its "
            "verification feedback must be shown before submission. Read the "
            "feedback, correct model.py, and submit only after verification "
            "succeeds."
        ),
        require_feedback_before_submit=True,
    )
    coding_agent = coding_agent_builder(
        tools=tools,
        system_prompt=build_system_prompt(
            system_prompt_path,
            prompt_context | {"max_turns": coding_agent_builder.keywords["max_turns"]},
            CodingSubmission,
        ),
        output_schema=CodingSubmission,
        extra_middleware=[coding_middleware],
    )
    return CodingStage(
        agent=coding_agent,
        instructions=instructions,
        verifier=coding_verifier,
        middleware=coding_middleware,
        input_after_compaction=input_after_compaction,
    )
