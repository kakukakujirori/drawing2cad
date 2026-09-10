from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.pregel import Pregel

from zeroshot.pipeline.messages.contracts.reconstruction import (
    OperationSubmission,
    tickets_assigned_to,
)
from zeroshot.pipeline.messages.contracts.stages import PipelineStage
from zeroshot.pipeline.stages._base.prompt import StageInstructions, build_system_prompt
from zeroshot.pipeline.workflow._config import _child_graph_config
from zeroshot.pipeline.workflow.state import ReconstructionState, current_snapshot

type CompiledGraph = Pregel[Any, Any, Any, Any]
type AgentBuilder = partial[CompiledGraph]


@dataclass(frozen=True)
class OperationStage:
    agent: CompiledGraph
    instructions: StageInstructions
    input_after_compaction: bool

    def run(self, state: ReconstructionState, config: RunnableConfig) -> dict[str, Any]:
        snapshot = current_snapshot(state)
        if snapshot.last_completed_stage is not PipelineStage.SEMANTICS:
            raise RuntimeError("operations requires integrated semantics")

        if not tickets_assigned_to(snapshot.open_tickets, PipelineStage.OPERATIONS):
            return {"stage_submission": OperationSubmission.unchanged()}

        previous = state.get("operations_state") or {}
        messages = [
            *list(previous.get("messages") or []),
            self.instructions.build(
                state,
                PipelineStage.OPERATIONS,
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
            "operations_state": result,
            "stage_submission": result.get("structured_response"),
        }


def create_operation_stage(
    operation_agent_builder: AgentBuilder,
    tools: Sequence[BaseTool],
    system_prompt_path: Path | None,
    instructions: StageInstructions,
    prompt_context: dict[str, str],
    input_after_compaction: bool,
) -> OperationStage:
    if system_prompt_path is None:
        system_prompt_path = Path(__file__).parent / "prompts" / "role.md"

    agent = operation_agent_builder(
        tools=tools,
        system_prompt=build_system_prompt(
            system_prompt_path,
            prompt_context
            | {"max_turns": operation_agent_builder.keywords["max_turns"]},
            OperationSubmission,
        ),
        output_schema=OperationSubmission,
    )
    return OperationStage(
        agent=agent,
        instructions=instructions,
        input_after_compaction=input_after_compaction,
    )
