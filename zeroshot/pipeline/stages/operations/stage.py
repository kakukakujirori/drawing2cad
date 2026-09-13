import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.pregel import Pregel

from zeroshot.pipeline.messages.tickets import TicketAnswers, tickets_assigned_to
from zeroshot.pipeline.stages._base.prompt import StageInstructions, build_system_prompt
from zeroshot.pipeline.stages.operations.contracts import OperationPlan
from zeroshot.pipeline.stages.types import PipelineStage
from zeroshot.pipeline.verification.attempts import AttemptStore
from zeroshot.pipeline.verification.verify_operations import OperationPlanVerifier
from zeroshot.pipeline.workflow._config import _child_graph_config
from zeroshot.pipeline.workflow.lifecycle import operations_baseline
from zeroshot.pipeline.workflow.middleware import VerifyOnWriteMiddleware
from zeroshot.pipeline.workflow.state import ReconstructionState, current_snapshot

type CompiledGraph = Pregel[Any, Any, Any, Any]
type AgentBuilder = partial[CompiledGraph]


@dataclass(frozen=True)
class OperationStage:
    agent: CompiledGraph
    instructions: StageInstructions
    verifier: OperationPlanVerifier
    middleware: VerifyOnWriteMiddleware
    input_after_compaction: bool

    def run(self, state: ReconstructionState, config: RunnableConfig) -> dict[str, Any]:
        snapshot = current_snapshot(state)
        interpretation = snapshot.interpretation
        if (
            snapshot.last_completed_stage is not PipelineStage.INTERPRETATION
            or interpretation is None
        ):
            raise RuntimeError("operations requires an integrated interpretation")

        if state.get("stage_validation_error") is None:
            self.verifier.reset(
                operations_baseline(state["reconstruction"]), interpretation
            )
            self.middleware.reset()
        if not tickets_assigned_to(snapshot.open_tickets, PipelineStage.OPERATIONS):
            return {"stage_submission": TicketAnswers(responses=[])}

        previous = state.get("operations_state") or {}
        messages = [
            *list(previous.get("messages") or []),
            self.instructions.build(
                state,
                PipelineStage.OPERATIONS,
                include_artifact=(not previous or self.input_after_compaction),
                operations_output_path=str(
                    self.instructions.workdir.sandbox_bind_dir
                    / self.verifier.source_filename
                ),
                operations_schema=json.dumps(OperationPlan.model_json_schema()),
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
            "operations_state": result,
            "stage_submission": result.get("structured_response"),
        }


def create_operation_stage(
    operation_agent_builder: AgentBuilder,
    tools: Sequence[BaseTool],
    system_prompt_path: Path | None,
    instructions: StageInstructions,
    prompt_context: dict[str, str],
    attempt_store: AttemptStore,
    operations_filename: str = "operations.json",
    input_after_compaction: bool = False,
) -> OperationStage:
    if system_prompt_path is None:
        system_prompt_path = Path(__file__).parent / "prompts" / "role.md"

    verifier = OperationPlanVerifier(
        workdir=instructions.workdir,
        attempt_store=attempt_store,
        source_filename=operations_filename,
    )
    middleware = VerifyOnWriteMiddleware(
        verifier,
        refusal=(
            "The operation plan is not ready to submit. Correct the current JSON "
            "using the validation feedback, and answer only after it validates."
        ),
    )
    agent = operation_agent_builder(
        tools=tools,
        system_prompt=build_system_prompt(
            system_prompt_path,
            prompt_context
            | {"max_turns": operation_agent_builder.keywords["max_turns"]},
            TicketAnswers,
        ),
        output_schema=TicketAnswers,
        extra_middleware=[middleware],
    )
    return OperationStage(
        agent=agent,
        instructions=instructions,
        verifier=verifier,
        middleware=middleware,
        input_after_compaction=input_after_compaction,
    )
