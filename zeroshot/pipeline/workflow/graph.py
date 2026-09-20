import re
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
from zeroshot.pipeline.stages.audit.contracts import AuditReport
from zeroshot.pipeline.stages.contracts import (
    ReconstructionHistory,
    ReconstructionSnapshot,
)
from zeroshot.pipeline.stages.stage import stage_factory
from zeroshot.pipeline.stages.tickets.contracts import (
    TicketAnswers,
    tickets_assigned_to,
)
from zeroshot.pipeline.stages.types import (
    REASONING_STAGES,
    PipelineStage,
    ReasoningStage,
    next_stage,
)
from zeroshot.pipeline.stages.validate import validate_submission
from zeroshot.pipeline.tools.load_image import create_load_image_tool
from zeroshot.pipeline.tools.run_shell import create_run_shell_tool
from zeroshot.pipeline.verification import AttemptStore
from zeroshot.pipeline.workflow.components import compact_transcript
from zeroshot.pipeline.workflow.lifecycle import (
    advance_reconstruction,
    interpretation_baseline,
    load_reconstruction,
    open_next_round,
    operations_baseline,
    save_reconstruction,
    start_reconstruction,
)
from zeroshot.pipeline.workflow.state import (
    ReconstructionState,
    carry_thread,
    current_snapshot,
    lead_transcript,
)

type CompiledGraph = Pregel[Any, Any, Any, Any]
type AgentBuilder = partial[CompiledGraph]


def create_reconstruction_graph(
    interpretation_agent_builder: AgentBuilder,
    operations_agent_builder: AgentBuilder,
    coding_agent_builder: AgentBuilder,
    audit_agent_builder: AgentBuilder,
    sandbox_runner: SandboxRunner,
    sandbox_workdir: SandboxWorkdir,
    artifact_presenter: ArtifactPresenter,
    input_manifest: InputManifest,
    output_filename: str = "model.py",
    interpretation_filename: str = "interpretation.json",
    operations_filename: str = "operations.json",
    verification_dirname: PurePosixPath = PurePosixPath("attempts"),
    reconstruction_history_filename: str = "reconstruction.json",
    max_audit_reject_count: int = 3,
    max_stage_validation_retries: int = 3,
    show_intermediate_returns: bool = True,
    share_thread: bool = False,
    compact_between_stages: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
):
    """Interpret and plan the part, implement it, then verify and audit it."""
    if max_audit_reject_count < 0:
        raise ValueError(f"{max_audit_reject_count=} must be non-negative")
    if max_stage_validation_retries < 0:
        raise ValueError(f"{max_stage_validation_retries=} must be non-negative")
    if compact_between_stages is not None and not share_thread:
        raise ValueError(
            "compact_between_stages needs share_thread: with a transcript per "
            "stage there is no handover at which to compact anything."
        )

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
        "operations_output_path": str(
            sandbox_workdir.sandbox_bind_dir / operations_filename
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
        input_presentation_mode=artifact_presenter.input_mode,
        workdir=sandbox_workdir,
    )

    share_thread_system_prompt = (
        Path(__file__).resolve().parents[1]
        / "stages/_base/prompts/cad_reconstructor.md"
    )

    interpretation_stage = stage_factory(PipelineStage.INTERPRETATION)(
        interpretation_agent_builder,
        tools=basic_tools,
        system_prompt_path=share_thread_system_prompt if share_thread else None,
        instructions=stage_instructions,
        prompt_context=prompt_context,
        attempt_store=attempt_store,
        interpretation_filename=interpretation_filename,
        input_after_compaction=compact_between_stages is not None,
    )
    operation_stage = stage_factory(PipelineStage.OPERATIONS)(
        operations_agent_builder,
        tools=basic_tools,
        system_prompt_path=share_thread_system_prompt if share_thread else None,
        instructions=stage_instructions,
        prompt_context=prompt_context,
        attempt_store=attempt_store,
        operations_filename=operations_filename,
        input_after_compaction=compact_between_stages is not None,
    )
    coding_stage = stage_factory(PipelineStage.CODING)(
        coding_agent_builder,
        tools=basic_tools,
        system_prompt_path=share_thread_system_prompt if share_thread else None,
        instructions=stage_instructions,
        prompt_context=prompt_context,
        attempt_store=attempt_store,
        sandbox_runner=sandbox_runner,
        feedback_presentation_mode=artifact_presenter.feedback_mode,
        output_filename=output_filename,
        show_intermediate_returns=show_intermediate_returns,
        input_after_compaction=compact_between_stages is not None,
    )
    audit_stage = stage_factory(PipelineStage.AUDIT)(
        audit_agent_builder,
        tools=basic_tools,
        system_prompt_path=None,
        instructions=stage_instructions,
        prompt_context=prompt_context,
        attempt_store=attempt_store,
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

    def _workspace_output(
        stage: PipelineStage | None,
        snapshot: ReconstructionSnapshot,
        reconstruction: ReconstructionHistory,
    ) -> Any:
        """The verified artifact the stage produced, or the baseline it kept."""
        tickets = snapshot.open_tickets
        match stage:
            case PipelineStage.INTERPRETATION:
                if not tickets_assigned_to(tickets, PipelineStage.INTERPRETATION):
                    return interpretation_baseline(reconstruction)
                accepted = (
                    interpretation_stage.interpretation_verifier.accepted_interpretation
                )
                filename = interpretation_filename
            case PipelineStage.OPERATIONS:
                if not tickets_assigned_to(tickets, PipelineStage.OPERATIONS):
                    return operations_baseline(reconstruction)
                accepted = operation_stage.operation_verifier.accepted_plan
                filename = operations_filename
            case PipelineStage.CODING:
                return coding_stage.output_verifier.verify()[0]
            case _:
                raise RuntimeError(f"{stage} is not a reasoning stage")
        if accepted is None:
            raise SubmissionValidationError(
                f"{filename} has not passed validation for this submission"
            )
        return accepted

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

        snapshot = current_snapshot(state)
        stage = next_stage(snapshot.last_completed_stage)
        try:
            updated = advance_reconstruction(
                reconstruction,
                submission,
                workspace_output=_workspace_output(stage, snapshot, reconstruction),
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
                retry_stage = next_stage(snapshot.last_completed_stage)
                if retry_stage not in REASONING_STAGES:
                    raise RuntimeError("no reasoning stage is available to retry")
                return retry_stage.value
            return "__end__"

        completed = snapshot.last_completed_stage
        if completed is None:
            raise RuntimeError(
                "successful stage integration did not complete a reasoning stage"
            )
        if (
            completed is PipelineStage.CODING
            and snapshot.round >= max_audit_reject_count
        ):
            return "__end__"
        if share_thread:
            return _handover_node(completed)
        following = next_stage(completed)
        if following is None:
            raise RuntimeError("coding must be followed by audit")
        return following.value

    # ------------------------------------------------------------------
    # Audit validation and round transition
    # ------------------------------------------------------------------

    def integrate_audit_report(state: ReconstructionState) -> dict[str, Any]:
        """Validate an audit and atomically open its requested next round."""
        report = state.get("audit_report")
        if not isinstance(report, AuditReport):
            return _validation_failure(
                state,
                "the auditor did not return an AuditReport",
            )
        try:
            validate_submission(report, current_snapshot(state))
        except SubmissionValidationError as error:
            return _validation_failure(state, str(error))

        if (
            not report.accepted
            and current_snapshot(state).round < max_audit_reject_count
        ):
            reconstruction = state.get("reconstruction")
            if reconstruction is None:
                raise RuntimeError("audit integration requires reconstruction")
            updated = open_next_round(reconstruction, report, sandbox_workdir)
            save_history(updated)
            return {
                "reconstruction": updated,
                "stage_submission": None,
                "stage_validation_error": None,
                "stage_validation_failure_count": 0,
                "audit_report": None,
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
            return PipelineStage.INTERPRETATION.value
        return "__end__"

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    # compaction after stage & carry thread through stages
    def handover(
        state: ReconstructionState, config: RunnableConfig, *, stage: ReasoningStage
    ) -> dict[str, Any]:
        thread = lead_transcript(state, stage)
        if compact_between_stages is not None:
            thread = compact_transcript(
                thread, model=compact_between_stages, config=config
            )
        return carry_thread(state, thread)

    def _handover_node(stage: ReasoningStage) -> str:
        return stage.value + "_handover"

    # Construct a graph
    workflow = StateGraph(state_schema=ReconstructionState)  # type: ignore[type-var]
    workflow.add_node("initialize", initialize)
    workflow.add_node(PipelineStage.INTERPRETATION.value, interpretation_stage.run)
    workflow.add_node(PipelineStage.OPERATIONS.value, operation_stage.run)
    workflow.add_node(PipelineStage.CODING.value, coding_stage.run)
    workflow.add_node(PipelineStage.AUDIT.value, audit_stage.run)
    workflow.add_node("integrate_stage_submission", integrate_stage_submission)
    workflow.add_node("integrate_audit_report", integrate_audit_report)

    if share_thread:
        for stage in REASONING_STAGES:
            workflow.add_node(_handover_node(stage), partial(handover, stage=stage))
            following = next_stage(stage)
            if following is None:
                raise RuntimeError(f"{stage.value} has no successor")
            workflow.add_edge(_handover_node(stage), following.value)

    workflow.add_edge(START, "initialize")
    workflow.add_conditional_edges("initialize", after_initialize)
    for stage in REASONING_STAGES:
        workflow.add_edge(stage.value, "integrate_stage_submission")
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
