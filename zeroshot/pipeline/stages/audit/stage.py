from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.pregel import Pregel

from zeroshot.pipeline.messages.artifact import ArtifactPresenter
from zeroshot.pipeline.stages._base.prompt import (
    PromptTemplate,
    StageInstructions,
    build_system_prompt,
    schema_for_prompt,
)
from zeroshot.pipeline.stages.audit.contracts import AuditReport, AuditSubmission
from zeroshot.pipeline.stages.audit.verify import AuditVerifier
from zeroshot.pipeline.stages.coding.verify import build_verification_feedback
from zeroshot.pipeline.stages.types import PipelineStage
from zeroshot.pipeline.verification import AttemptStore
from zeroshot.pipeline.workflow._config import _child_graph_config
from zeroshot.pipeline.workflow.evidence import EvidenceMode
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
    artifact_presenter: ArtifactPresenter
    drawing_diff_reviews: str

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
            append_inputs=not previous,
            audit_output_path=str(
                self.instructions.workdir.sandbox_bind_dir
                / self.audit_verifier.source_filename
            ),
            audit_schema=schema_for_prompt(AuditReport),
            attempt_dir=attempt_dir,
            drawing_diff_reviews=self.drawing_diff_reviews,
            feedback=build_verification_feedback(
                verification,
                self.instructions.workdir,
                self.artifact_presenter,
                snapshot.interpretation.view_frames()
                if snapshot.interpretation
                else {},
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
            "audit_evidence": self.audit_verifier.evidence_renders,
        }


def create_audit_stage(
    audit_agent_builder: AgentBuilder,
    tools: Sequence[BaseTool],
    role_path: Path | None,
    instructions: StageInstructions,
    prompt_context: dict[str, str],
    attempt_store: AttemptStore,
    artifact_presenter: ArtifactPresenter,
    audit_filename: str = "audit.json",
    evidence_mode: EvidenceMode = "mark",
    review_drawing_diff_clusters: bool = True,
) -> AuditStage:
    audit_verifier = AuditVerifier(
        attempt_store,
        source_filename=audit_filename,
        evidence_mode=evidence_mode,
        require_drawing_diff_reviews=review_drawing_diff_clusters,
    )
    middleware = VerifyOnWriteMiddleware(audit_verifier)
    audit_agent = audit_agent_builder(
        tools=tools,
        system_prompt=build_system_prompt(
            role_path,
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
        artifact_presenter=artifact_presenter,
        drawing_diff_reviews=(
            PromptTemplate(
                Path(__file__).parent / "prompts/drawing_diff_reviews.md"
            ).render()
            if review_drawing_diff_clusters
            else ""
        ),
    )
