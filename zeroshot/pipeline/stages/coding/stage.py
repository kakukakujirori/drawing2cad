from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, cast

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.pregel import Pregel
from omegaconf import DictConfig, OmegaConf

from zeroshot.pipeline.messages.artifact import ArtifactPresenter
from zeroshot.pipeline.sandbox import SandboxRunner
from zeroshot.pipeline.stages._base.prompt import StageInstructions, build_system_prompt
from zeroshot.pipeline.stages.coding.progress import (
    CodingProgressMiddleware,
)
from zeroshot.pipeline.stages.coding.progress import (
    ProgressOutputVerifier as OutputVerifier,
)
from zeroshot.pipeline.stages.tickets.contracts import TicketAnswers
from zeroshot.pipeline.stages.tickets.verify import TicketVerifier
from zeroshot.pipeline.stages.types import PipelineStage
from zeroshot.pipeline.tools.render_step import create_render_step_tool
from zeroshot.pipeline.verification import (
    AttemptStore,
    CadQueryExecutor,
    DrawingDiffExecutor,
    StepRenderer,
)
from zeroshot.pipeline.workflow._config import _child_graph_config
from zeroshot.pipeline.workflow.middleware import CodingTrialMiddleware
from zeroshot.pipeline.workflow.middleware.fresh_coding import (
    INSTRUCTIONS_NAME,
    FreshCodingMiddleware,
)
from zeroshot.pipeline.workflow.state import ReconstructionState, current_snapshot

type CompiledGraph = Pregel[Any, Any, Any, Any]
type AgentBuilder = partial[CompiledGraph]


@dataclass(frozen=True)
class CodingStage:
    agent: CompiledGraph
    instructions: StageInstructions
    output_verifier: OutputVerifier
    ticket_verifier: TicketVerifier
    middleware: CodingProgressMiddleware
    input_after_compaction: bool

    def run(self, state: ReconstructionState, config: RunnableConfig) -> dict[str, Any]:
        snapshot = current_snapshot(state)
        if snapshot.last_completed_stage is not PipelineStage.OPERATIONS:
            raise RuntimeError("coding requires integrated operations")
        interpretation = snapshot.interpretation
        if interpretation is None:
            raise RuntimeError("coding requires integrated interpretation")

        # The verifier checks the program against this round's operations and
        # redraws the solid in the views the drawing names, guessing none. Set
        # here because both the build inside the agent and the one at
        # integration belong to this stage of this round.
        self.output_verifier.interpretation = interpretation
        self.output_verifier.operations = snapshot.operations
        fresh = isinstance(self.middleware, FreshCodingMiddleware)
        if fresh:
            self.middleware.set_context(state, self.instructions)

        # Start verification against this round's interpretation and operations.
        baseline = []
        if state.get("stage_validation_error") is None:
            self.output_verifier.reset()
            self.middleware.reset()  # NOTE: must be after attributes are assigned
            baseline = self.middleware.baseline_feedback()
        self.ticket_verifier.reset(state["reconstruction"])

        previous = state.get("coding_state") or {}
        instruction = self.instructions.build(
            state,
            PipelineStage.CODING,
            append_inputs=(not previous or self.input_after_compaction),
            dimension_inventory=interpretation.render_dimension_inventory(),
        )
        if fresh:
            instruction = instruction.model_copy(
                update={
                    "name": INSTRUCTIONS_NAME,
                    "additional_kwargs": {
                        **instruction.additional_kwargs,
                        "coding_validation_error": bool(
                            state.get("stage_validation_error")
                        ),
                    },
                }
            )
        messages = [
            *list(previous.get("messages") or []),
            instruction,
            *(
                [
                    self.middleware.checkpoint_message(baseline)
                    if fresh
                    else HumanMessage(content_blocks=baseline)
                ]
                if baseline
                else []
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
    role_path: Path | None,
    instructions: StageInstructions,
    prompt_context: dict[str, str],
    attempt_store: AttemptStore,
    sandbox_runner: SandboxRunner,
    diff_drawer_config: dict[str, Any] | None,
    artifact_presenter: ArtifactPresenter,
    output_filename: str = "model.py",
    input_after_compaction: bool = False,
    fresh_memory: bool = False,
) -> CodingStage:
    if isinstance(diff_drawer_config, DictConfig):
        diff_drawer_config = cast(
            dict[str, Any], OmegaConf.to_container(diff_drawer_config, resolve=True)
        )

    executor = CadQueryExecutor(sandbox_runner=sandbox_runner)
    renderer = StepRenderer()
    diff_drawer = (
        DrawingDiffExecutor(**diff_drawer_config)
        if diff_drawer_config is not None
        else None
    )
    output_verifier = OutputVerifier(
        executor=executor,
        workdir=instructions.workdir,
        renderer=renderer,
        diff_drawer=diff_drawer,
        artifact_presenter=artifact_presenter,
        attempt_store=attempt_store,
        source_filename=output_filename,
    )
    ticket_verifier = TicketVerifier(lambda: output_verifier.accepted_source)
    middleware_type = (
        FreshCodingMiddleware if fresh_memory else CodingProgressMiddleware
    )
    coding_middleware = middleware_type(
        output_verifier,
        ticket_verifier=ticket_verifier,
        fingerprint=output_verifier.source_digest,
    )
    # Resolve frames on each call so a later round uses its updated interpretation.
    coding_tools = [
        *tools,
        create_render_step_tool(
            instructions.workdir,
            renderer,
            drawing_frames=lambda: (
                output_verifier.interpretation.view_frames()
                if output_verifier.interpretation is not None
                else {}
            ),
        ),
    ]
    coding_agent = coding_agent_builder(
        tools=coding_tools,
        system_prompt=build_system_prompt(
            role_path,
            prompt_context | {"max_turns": coding_agent_builder.keywords["max_turns"]},
            TicketAnswers,
        ),
        output_schema=TicketAnswers,
        extra_middleware=[
            coding_middleware,
            CodingTrialMiddleware(
                tool_names=[tool.name for tool in coding_tools],
                max_turns=coding_agent_builder.keywords["max_turns"],
            ),
        ],
    )
    return CodingStage(
        agent=coding_agent,
        instructions=instructions,
        output_verifier=output_verifier,
        ticket_verifier=ticket_verifier,
        middleware=coding_middleware,
        input_after_compaction=input_after_compaction,
    )
