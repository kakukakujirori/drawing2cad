from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.pregel import Pregel

from zeroshot.pipeline.stages._base.prompt import (
    StageInstructions,
    build_system_prompt,
    schema_for_prompt,
)
from zeroshot.pipeline.stages.audit.contracts import AuditReport, AuditSubmission
from zeroshot.pipeline.stages.audit.verify import AuditVerifier
from zeroshot.pipeline.stages.types import PipelineStage
from zeroshot.pipeline.verification import AttemptStore
from zeroshot.pipeline.verification._run_program import INTERMEDIATE_RETURNS_DIR
from zeroshot.pipeline.workflow._config import _child_graph_config
from zeroshot.pipeline.workflow.middleware import VerifyOnWriteMiddleware
from zeroshot.pipeline.workflow.state import ReconstructionState, current_snapshot

type CompiledGraph = Pregel[Any, Any, Any, Any]
type AgentBuilder = partial[CompiledGraph]


@dataclass(frozen=True)
class AuditStage:
    agent: CompiledGraph
    instructions: StageInstructions
    attempt_store: AttemptStore
    audit_verifier: AuditVerifier
    middleware: VerifyOnWriteMiddleware

    def run(self, state: ReconstructionState, config: RunnableConfig) -> dict[str, Any]:
        snapshot = current_snapshot(state)
        if snapshot.last_completed_stage is not PipelineStage.CODING:
            raise RuntimeError("audit requires a completed coding snapshot")
        verification = snapshot.verification
        if verification is None:
            raise RuntimeError("audit requires verification")
        if state.get("stage_validation_error") is None:
            self.audit_verifier.reset(snapshot)
            self.middleware.reset()

        attempt_dir = str(
            self.attempt_store.sandbox_attempt_dir(
                snapshot.round, "coding", verification.verification_id
            )
            if verification.verification_id is not None
            else self.attempt_store.sandbox_root
        )
        previous = state.get("audit_state") or {}
        instruction = self.instructions.build(
            state,
            PipelineStage.AUDIT,
            include_artifact=not previous,
            audit_output_path=str(
                self.instructions.workdir.sandbox_bind_dir
                / self.audit_verifier.source_filename
            ),
            audit_schema=schema_for_prompt(AuditReport),
            attempt_dir=attempt_dir,
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
            "audit_report": (
                self.audit_verifier.accepted_report
                if isinstance(result.get("structured_response"), AuditSubmission)
                else None
            ),
            "audit_evidence": self.audit_verifier.evidence_crops,
        }


def create_audit_stage(
    audit_agent_builder: AgentBuilder,
    tools: Sequence[BaseTool],
    system_prompt_path: Path | None,
    instructions: StageInstructions,
    prompt_context: dict[str, str],
    attempt_store: AttemptStore,
    audit_filename: str = "audit.json",
) -> AuditStage:
    if system_prompt_path is None:
        system_prompt_path = Path(__file__).parent / "prompts" / "role.md"

    audit_verifier = AuditVerifier(attempt_store, source_filename=audit_filename)
    middleware = VerifyOnWriteMiddleware(
        audit_verifier,
        require_feedback_before_submit=True,
        refusal=f"Correct {audit_filename}, read validation feedback and check the generated evidence before submitting AuditSubmission.",
    )
    audit_agent = audit_agent_builder(
        tools=tools,
        system_prompt=build_system_prompt(
            system_prompt_path,
            prompt_context | {"max_turns": audit_agent_builder.keywords["max_turns"]},
            AuditSubmission,
        ),
        output_schema=AuditSubmission,
        extra_middleware=[middleware],
    )
    return AuditStage(
        agent=audit_agent,
        instructions=instructions,
        attempt_store=attempt_store,
        audit_verifier=audit_verifier,
        middleware=middleware,
    )
