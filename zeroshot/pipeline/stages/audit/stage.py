import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.pregel import Pregel

from zeroshot.pipeline.messages.contracts.audit import AuditReport
from zeroshot.pipeline.stages._base.prompt import StageInstructions, build_system_prompt
from zeroshot.pipeline.stages.types import PipelineStage
from zeroshot.pipeline.verification import AttemptStore
from zeroshot.pipeline.verification._run_program import INTERMEDIATE_RETURNS_DIR
from zeroshot.pipeline.workflow._config import _child_graph_config
from zeroshot.pipeline.workflow.state import ReconstructionState, current_snapshot

type CompiledGraph = Pregel[Any, Any, Any, Any]
type AgentBuilder = partial[CompiledGraph]


@dataclass(frozen=True)
class AuditStage:
    agent: CompiledGraph
    instructions: StageInstructions
    attempt_store: AttemptStore

    def run(self, state: ReconstructionState, config: RunnableConfig) -> dict[str, Any]:
        snapshot = current_snapshot(state)
        if snapshot.last_completed_stage is not PipelineStage.CODING:
            raise RuntimeError("audit requires a completed coding snapshot")
        verification = snapshot.verification
        if verification is None:
            raise RuntimeError("audit requires verification")

        attempt_dir = str(
            self.attempt_store.sandbox_attempt_dir(
                snapshot.round, "coding", verification.verification_id
            )
            if verification.verification_id is not None
            else self.attempt_store.sandbox_root
        )
        drawing_attempt = self.attempt_store.latest_sandbox_attempt_dir(
            "drawing", snapshot.round
        )
        previous = state.get("audit_state") or {}
        instruction = self.instructions.build(
            state,
            PipelineStage.AUDIT,
            include_artifact=not previous,
            attempt_dir=attempt_dir,
            drawing_attempt_dir=(
                str(drawing_attempt)
                if drawing_attempt is not None
                else "unavailable in this workspace"
            ),
            ticket_responses=json.dumps(
                [
                    response.model_dump(mode="json")
                    for ticket in snapshot.open_tickets
                    for response in ticket.responses
                ],
                indent=2,
            ),
            intermediate_returns_dir=(
                str(PurePosixPath(attempt_dir) / INTERMEDIATE_RETURNS_DIR)
                if verification.intermediate_returns
                else "unavailable"
            ),
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
            "audit_state": result,
            "audit_report": result.get("structured_response"),
        }


def create_audit_stage(
    audit_agent_builder: AgentBuilder,
    tools: Sequence[BaseTool],
    system_prompt_path: Path | None,
    instructions: StageInstructions,
    prompt_context: dict[str, str],
    attempt_store: AttemptStore,
) -> AuditStage:
    if system_prompt_path is None:
        system_prompt_path = Path(__file__).parent / "prompts" / "role.md"

    audit_agent = audit_agent_builder(
        tools=tools,
        system_prompt=build_system_prompt(
            system_prompt_path,
            prompt_context | {"max_turns": audit_agent_builder.keywords["max_turns"]},
            AuditReport,
        ),
        output_schema=AuditReport,
    )
    return AuditStage(
        agent=audit_agent,
        instructions=instructions,
        attempt_store=attempt_store,
    )
