import re
from collections.abc import Mapping
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, StateGraph
from langgraph.pregel import Pregel

from zeroshot.pipeline.messages.artifact import ArtifactPresenter, drawing_for_model
from zeroshot.pipeline.messages.manifest import InputManifest
from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.stages._base.prompt import StageInstructions
from zeroshot.pipeline.stages._base.validate import SubmissionValidationError
from zeroshot.pipeline.stages.audit.contracts import AuditReport, AuditSubmission
from zeroshot.pipeline.stages.coding.progress import DEFAULT_MATCH_MARGIN_PX
from zeroshot.pipeline.stages.coding.validate import CodingOutput
from zeroshot.pipeline.stages.contracts import ReconstructionHistory
from zeroshot.pipeline.stages.stage import stage_factory
from zeroshot.pipeline.stages.tickets.contracts import TicketAnswers
from zeroshot.pipeline.stages.types import PipelineStage, next_stage
from zeroshot.pipeline.stages.validate import validate_submission
from zeroshot.pipeline.tools.load_image import create_load_image_tool
from zeroshot.pipeline.tools.run_shell import create_run_shell_tool
from zeroshot.pipeline.verification import AttemptStore
from zeroshot.pipeline.workflow.components import compact_transcript
from zeroshot.pipeline.workflow.evidence import EvidenceMode
from zeroshot.pipeline.workflow.lifecycle import (
    advance_reconstruction,
    load_reconstruction,
    open_next_round,
    save_reconstruction,
    start_reconstruction,
)
from zeroshot.pipeline.workflow.middleware.output_limit_budget import OutputLimitBudget
from zeroshot.pipeline.workflow.state import ReconstructionState, current_snapshot

type CompiledGraph = Pregel[Any, Any, Any, Any]
type AgentBuilder = partial[CompiledGraph]


def create_reconstruction_graph(
    coding_agent_builder: AgentBuilder,
    audit_agent_builder: AgentBuilder,
    sandbox_runner: SandboxRunner,
    sandbox_workdir: SandboxWorkdir,
    artifact_presenter: ArtifactPresenter,
    input_manifest: InputManifest,
    output_filename: str = "model.py",
    interpretation_filename: str = "interpretation.json",
    verification_dirname: PurePosixPath = PurePosixPath("attempts"),
    reconstruction_history_filename: str = "reconstruction.json",
    max_audit_reject_count: int = 3,
    max_stage_validation_retries: int = 3,
    max_output_limit_failures: int = 3,
    compact_between_stages: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    audit_evidence_mode: EvidenceMode = "mark",
    diff_drawer_config: Mapping[str, Any] | None = None,
    fresh_coder: bool = False,
    coding_trial_reminder: bool = True,
    match_margin_px: float = DEFAULT_MATCH_MARGIN_PX,
    review_drawing_diff_clusters: bool = True,
):
    """Interpret the drawing and implement the part, then verify and audit it."""
    if max_audit_reject_count < 0:
        raise ValueError(f"{max_audit_reject_count=} must be non-negative")
    if max_stage_validation_retries < 0:
        raise ValueError(f"{max_stage_validation_retries=} must be non-negative")

    # Unlike stage validation retries, output limits recur inside a child agent's
    # wrap_model_call before any update reaches the parent workflow state. Share
    # one mutable counter across agents to stop inside that retry loop without
    # threading the count through every stage's input/output. Keep it across rounds.
    output_limit_budget = OutputLimitBudget(max_output_limit_failures)

    # Create tools
    basic_tools = [
        create_run_shell_tool(sandbox_runner, sandbox_workdir),
        create_load_image_tool(sandbox_workdir),
    ]
    history_path = sandbox_workdir.host_bind_dir / reconstruction_history_filename

    def current_round() -> int:
        """Read the round committed before an agent can issue an attempt."""
        return len(load_reconstruction(history_path).snapshots) - 1

    attempt_store = AttemptStore(
        sandbox_workdir,
        round_source=current_round,
        root_dirname=verification_dirname,
    )

    # instantiate agents
    prompt_context = {
        "coding_output_path": str(sandbox_workdir.sandbox_bind_dir / output_filename),
        "interpretation_output_path": str(
            sandbox_workdir.sandbox_bind_dir / interpretation_filename
        ),
        "verification_dir": str(
            sandbox_workdir.sandbox_bind_dir / verification_dirname
        ),
        # TODO: move to read-only directory (e.g., `inputs`)
        "reconstruction_path": str(
            sandbox_workdir.sandbox_bind_dir / reconstruction_history_filename
        ),
    }
    stage_instructions = StageInstructions(
        prompt_context=prompt_context,
        input_artifact=input_manifest.drawing,
        input_presentation_mode=artifact_presenter.input,
        workdir=sandbox_workdir,
    )

    stages_dir = Path(__file__).resolve().parents[1] / "stages"

    # Bind the shared budget into builders; each stage adds its tools and schema.
    coding_stage = stage_factory(PipelineStage.CODING)(
        partial(coding_agent_builder, output_limit_budget=output_limit_budget),
        tools=basic_tools,
        role_path=stages_dir / "coding/prompts/role.md",
        instructions=stage_instructions,
        prompt_context=prompt_context,
        attempt_store=attempt_store,
        sandbox_runner=sandbox_runner,
        artifact_presenter=artifact_presenter,
        diff_drawer_config=diff_drawer_config,
        output_filename=output_filename,
        interpretation_filename=interpretation_filename,
        input_after_compaction=compact_between_stages is not None,
        fresh_memory=fresh_coder,
        coding_trial_reminder=coding_trial_reminder,
        match_margin_px=match_margin_px,
    )
    audit_stage = stage_factory(PipelineStage.AUDIT)(
        partial(audit_agent_builder, output_limit_budget=output_limit_budget),
        tools=basic_tools,
        role_path=stages_dir / "audit/prompts/role.md",
        instructions=stage_instructions,
        prompt_context=prompt_context,
        attempt_store=attempt_store,
        evidence_mode=audit_evidence_mode,
        artifact_presenter=artifact_presenter,
        review_drawing_diff_clusters=review_drawing_diff_clusters,
    )

    def save_history(history: ReconstructionHistory) -> None:
        save_reconstruction(history_path, history)

    # ------------------------------------------------------------------
    # Round initialization and common stage input
    # ------------------------------------------------------------------

    def initialize(state: ReconstructionState) -> dict[str, Any]:
        """Create or adopt and persist the history before any model reads it."""
        history = state.get("reconstruction")
        if history is None:
            run_suffix = re.sub(
                r"[^a-z0-9]+", "_", input_manifest.sample_id.casefold()
            ).strip("_")
            history = start_reconstruction(
                run_id=f"run_{run_suffix or 'sample'}",
                instruction="Reconstruct the input drawing as a CadQuery model.",
                # Addressed the way the model will read them, because the model
                # is what reads and revises this history from here on.
                drawings=drawing_for_model(input_manifest.drawing, sandbox_workdir),
            )
        save_history(history)
        return {
            "reconstruction": history,
            "stage_submission": None,
            "stage_validation_error": None,
            "stage_validation_failure_count": 0,
            "audit_report": None,
            "audit_evidence": {},
        }

    def after_initialize(state: ReconstructionState) -> str:
        """Enter the first unfinished stage of a new or resumed history."""
        snapshot = current_snapshot(state)
        if (
            snapshot.last_completed_stage is PipelineStage.CODING
            and snapshot.round >= max_audit_reject_count
        ):
            return "__end__"
        following = next_stage(snapshot.last_completed_stage)
        if following is None:
            raise RuntimeError("the initial reconstruction has no unfinished stage")
        return following.value

    # ------------------------------------------------------------------
    # Reasoning-stage validation, integration, and routing
    # ------------------------------------------------------------------

    def _validation_failure(
        state: ReconstructionState,
        error: str,
    ) -> dict[str, Any]:
        return {
            "stage_validation_error": error,
            "stage_validation_failure_count": (
                state.get("stage_validation_failure_count", 0) + 1
            ),
        }

    def _rejected_stage_submission(
        state: ReconstructionState,
        error: str,
    ) -> dict[str, Any]:
        return {
            **_validation_failure(state, error),
            "stage_submission": None,
        }

    def _workspace_output() -> CodingOutput:
        """The verified files the coder left: its reading and the build of its program."""
        workspace = coding_stage.workspace
        interpretation = workspace.interpretation.accepted_interpretation
        if interpretation is None:
            raise SubmissionValidationError(
                f"{interpretation_filename} has not passed validation for this submission"
            )
        return CodingOutput(interpretation, workspace.output.verify())

    def integrate_stage_submission(
        state: ReconstructionState,
    ) -> dict[str, Any]:
        """Validate and atomically integrate the pending reasoning output."""
        submission = state.get("stage_submission")
        if not isinstance(submission, TicketAnswers):
            return _rejected_stage_submission(
                state,
                "the reasoning stage did not return its ticket answers",
            )

        reconstruction = state.get("reconstruction")
        if reconstruction is None:
            raise RuntimeError("stage integration requires reconstruction")

        try:
            updated = advance_reconstruction(
                reconstruction, submission, workspace_output=_workspace_output()
            )
        except SubmissionValidationError as error:
            return _rejected_stage_submission(state, str(error))

        save_history(updated)
        return {
            "reconstruction": updated,
            "stage_submission": None,
            "stage_validation_error": None,
            "stage_validation_failure_count": 0,
        }

    def after_stage_integration(state: ReconstructionState) -> str:
        snapshot = current_snapshot(state)
        if state.get("stage_validation_error") is not None:
            if (
                state.get("stage_validation_failure_count", 0)
                <= max_stage_validation_retries
            ):
                return PipelineStage.CODING.value
            return "__end__"

        if snapshot.last_completed_stage is not PipelineStage.CODING:
            raise RuntimeError("successful stage integration did not complete coding")
        if snapshot.round >= max_audit_reject_count:
            return "__end__"
        if compact_between_stages is not None:
            return "coding_handover"
        return PipelineStage.AUDIT.value

    # ------------------------------------------------------------------
    # Audit validation and round transition
    # ------------------------------------------------------------------

    def integrate_audit_report(state: ReconstructionState) -> dict[str, Any]:
        """Validate an audit and atomically open its requested next round."""
        report = state.get("audit_report")
        answer = (state.get("audit_state") or {}).get("structured_response")
        if not isinstance(report, AuditReport) or not isinstance(
            answer, AuditSubmission
        ):
            return _validation_failure(
                state,
                "the auditor did not finish a validated audit report file with AuditSubmission",
            )
        try:
            validate_submission(report, current_snapshot(state))
            if answer.accepted == bool(report.findings):
                raise SubmissionValidationError(
                    "AuditSubmission.accepted must be true exactly when the report's "
                    "findings list is empty. Recheck the report and evidence; correct "
                    "the audit report file or your final decision before submitting again."
                )
        except SubmissionValidationError as error:
            return _validation_failure(state, str(error))

        if (
            not answer.accepted
            and current_snapshot(state).round < max_audit_reject_count
        ):
            reconstruction = state.get("reconstruction")
            if reconstruction is None:
                raise RuntimeError("audit integration requires reconstruction")
            updated = open_next_round(reconstruction, report, state["audit_evidence"])
            save_history(updated)
            return {
                "reconstruction": updated,
                "stage_submission": None,
                "stage_validation_error": None,
                "stage_validation_failure_count": 0,
                "audit_report": None,
                "audit_evidence": {},
            }

        return {
            "stage_validation_error": None,
            "stage_validation_failure_count": 0,
        }

    def after_audit_integration(state: ReconstructionState) -> str:
        if state.get("stage_validation_error") is not None:
            if (
                state.get("stage_validation_failure_count", 0)
                <= max_stage_validation_retries
            ):
                return PipelineStage.AUDIT.value
            return "__end__"

        if current_snapshot(state).last_completed_stage is None:
            return PipelineStage.CODING.value
        return "__end__"

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def coding_handover(
        state: ReconstructionState, config: RunnableConfig
    ) -> dict[str, Any]:
        """Trade the coder's working turns for notes before its next round."""
        assert compact_between_stages is not None
        coding_state = state.get("coding_state") or {}
        thread = compact_transcript(
            list(coding_state.get("messages") or []),
            model=compact_between_stages,
            config=config,
        )
        return {
            "coding_state": {
                **coding_state,
                "messages": thread,
                "reported_message_count": len(thread),
            }
        }

    # Construct a graph
    workflow = StateGraph(state_schema=ReconstructionState)  # type: ignore[type-var]
    workflow.add_node("initialize", initialize)
    workflow.add_node(PipelineStage.CODING.value, coding_stage.run)
    workflow.add_node(PipelineStage.AUDIT.value, audit_stage.run)
    workflow.add_node("integrate_stage_submission", integrate_stage_submission)
    workflow.add_node("integrate_audit_report", integrate_audit_report)
    if compact_between_stages is not None:
        workflow.add_node("coding_handover", coding_handover)
        workflow.add_edge("coding_handover", PipelineStage.AUDIT.value)

    workflow.add_edge(START, "initialize")
    workflow.add_conditional_edges("initialize", after_initialize)
    workflow.add_edge(PipelineStage.CODING.value, "integrate_stage_submission")
    workflow.add_edge(PipelineStage.AUDIT.value, "integrate_audit_report")
    workflow.add_conditional_edges(
        "integrate_stage_submission",
        after_stage_integration,
    )
    workflow.add_conditional_edges(
        "integrate_audit_report",
        after_audit_integration,
    )

    graph = workflow.compile(checkpointer=checkpointer)
    rounds = max_audit_reject_count + 1
    attempts_per_stage = max_stage_validation_retries + 1
    return graph.with_config(
        recursion_limit=len(workflow.nodes) * rounds * attempts_per_stage + 10
    )
