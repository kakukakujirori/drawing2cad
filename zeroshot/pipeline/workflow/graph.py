import re
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import PurePosixPath
from typing import Any, Protocol, cast

from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, merge_message_runs
from langchain_core.messages.content import create_text_block
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, StateGraph
from langgraph.pregel import Pregel
from pydantic import BaseModel

from zeroshot.pipeline.messages import (
    ArtifactPresenter,
    InputManifest,
    build_instruction,
)
from zeroshot.pipeline.messages.contracts.audit import AuditReport
from zeroshot.pipeline.messages.contracts.reconstruction import (
    CodingSubmission,
    OperationSubmission,
    ReconstructionSnapshot,
    SemanticSubmission,
    StageSubmission,
    tickets_assigned_to,
)
from zeroshot.pipeline.messages.contracts.stages import (
    REASONING_STAGES,
    PipelineStage,
    ReasoningStage,
    next_stage,
)
from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.tools import (
    create_load_image_tool,
    create_run_shell_tool,
)
from zeroshot.pipeline.verification import (
    CadQueryExecutor,
    OutputVerifier,
    StepRenderer,
)
from zeroshot.pipeline.workflow._config import _child_graph_config
from zeroshot.pipeline.workflow.components import compact_transcript
from zeroshot.pipeline.workflow.middleware import VerifyOnWriteMiddleware
from zeroshot.pipeline.workflow.reconstruction import (
    advance_reconstruction,
    open_next_round,
    save_reconstruction,
    start_reconstruction,
)
from zeroshot.pipeline.workflow.state import (
    ReconstructionState,
    carry_thread,
    lead_transcript,
)
from zeroshot.pipeline.workflow.validate_submission import (
    SubmissionValidationError,
    validate_submission,
)

type CompiledGraph = Pregel[Any, Any, Any, Any]


class AgentBuilder(Protocol):
    def __call__(
        self,
        *,
        tools: Sequence[BaseTool],
        prompt_context: Mapping[str, str],
        output_schema: type[BaseModel] | None = None,
        extra_middleware: Sequence[AgentMiddleware[Any, None, Any]] = (),
    ) -> CompiledGraph: ...


def create_reconstruction_graph(
    semantics_agent_builder: AgentBuilder,
    operations_agent_builder: AgentBuilder,
    coding_agent_builder: AgentBuilder,
    audit_agent_builder: AgentBuilder,
    sandbox_runner: SandboxRunner,
    sandbox_workdir: SandboxWorkdir,
    renderer: StepRenderer,
    artifact_presenter: ArtifactPresenter,
    input_manifest: InputManifest,
    output_filename: str = "model.py",
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
    executor = CadQueryExecutor(sandbox_runner=sandbox_runner)
    verifier = OutputVerifier(
        executor=executor,
        workdir=sandbox_workdir,
        renderer=renderer,
        artifact_presenter=artifact_presenter,
        source_filename=output_filename,
        output_dirname=verification_dirname,
        show_intermediate_returns=show_intermediate_returns,
    )

    # instantiate agents
    prompt_context = {
        "output_path": str(sandbox_workdir.sandbox_bind_dir / output_filename),
        "verification_dir": str(
            sandbox_workdir.sandbox_bind_dir / verification_dirname
        ),
        # TODO: move to read-only directory (e.g., `inputs`)
        "reconstruction_path": str(
            sandbox_workdir.sandbox_bind_dir / reconstruction_history_filename
        ),
    }

    semantics_agent = semantics_agent_builder(
        tools=basic_tools,
        prompt_context=prompt_context,
        output_schema=SemanticSubmission,
    )
    operations_agent = operations_agent_builder(
        tools=basic_tools,
        prompt_context=prompt_context,
        output_schema=OperationSubmission,
    )
    coding_agent = coding_agent_builder(
        tools=basic_tools,
        prompt_context=prompt_context,
        output_schema=CodingSubmission,
        extra_middleware=[VerifyOnWriteMiddleware(verifier)],
    )
    audit_agent = audit_agent_builder(
        tools=basic_tools,
        prompt_context=prompt_context,
        output_schema=AuditReport,
    )

    # prepare inputs (by calling it on-the-fly, prevent message_id duplication)
    def _prepare_input_message():
        return HumanMessage(
            content_blocks=artifact_presenter.build_input_message_blocks(
                manifest=input_manifest,
                workdir=sandbox_workdir,
            )
        )

    def current_snapshot(state: ReconstructionState) -> ReconstructionSnapshot:
        reconstruction = state.get("reconstruction")
        if reconstruction is None:
            raise RuntimeError("reconstruction has not been initialized")
        return reconstruction.snapshots[-1]

    # ------------------------------------------------------------------
    # Round initialization and common stage input
    # ------------------------------------------------------------------

    def initialize(state: ReconstructionState) -> dict[str, Any]:
        """Create or adopt and persist the history before any model reads it."""
        run = state.get("reconstruction")
        if run is None:
            run_suffix = re.sub(
                r"[^a-z0-9]+", "_", input_manifest.sample_id.casefold()
            ).strip("_")
            run = start_reconstruction(
                run_id=f"run_{run_suffix or 'sample'}",
                instruction="Reconstruct the input drawing as a CadQuery model.",
            )
        save_reconstruction(
            sandbox_workdir.host_bind_dir / reconstruction_history_filename,
            run,
        )
        return {
            "reconstruction": run,
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

    def assigned_ticket_ids(
        snapshot: ReconstructionSnapshot,
        stage: PipelineStage,
    ) -> str:
        """Name the tickets this stage owns, so it never has to go looking."""
        if stage not in REASONING_STAGES:
            return "none"
        assigned = tickets_assigned_to(snapshot.open_tickets, stage)
        return ", ".join(ticket.ticket_id for ticket in assigned) or "none"

    def build_stage_instruction(
        state: ReconstructionState,
        stage: PipelineStage,
        *,
        include_input: bool,
        **extra_context: str,
    ) -> HumanMessage:
        """Build the same round/ticket view for every reasoning stage."""
        if validation_error := state.get("stage_validation_error"):
            # A re-ask, so the round's terms and guidelines already stand in
            # the transcript and only the rejection is new.
            return HumanMessage(
                content_blocks=[
                    create_text_block(
                        f"[{stage.value.title()} Validation Error]\n"
                        f"Your previous {stage.value} stage output was rejected. "
                        "Return the corrected complete output using this feedback:\n\n"
                        f"{validation_error}"
                    )
                ]
            )

        snapshot = current_snapshot(state)
        instruction = build_instruction(
            f"{stage.value}/round",
            **prompt_context,
            current_round=str(snapshot.round),
            assigned_tickets=assigned_ticket_ids(snapshot, stage),
            **extra_context,
        )

        if include_input:
            (instruction,) = cast(
                list[HumanMessage],
                merge_message_runs([instruction, _prepare_input_message()]),
            )
        return instruction

    # ------------------------------------------------------------------
    # Reasoning-stage inference
    # ------------------------------------------------------------------

    def run_semantics(state: ReconstructionState, config: RunnableConfig):
        snapshot = current_snapshot(state)
        if not tickets_assigned_to(snapshot.open_tickets, PipelineStage.SEMANTICS):
            return {"stage_submission": SemanticSubmission.unchanged()}

        previous = state.get("semantics_state") or {}
        messages = [
            *list(previous.get("messages") or []),
            build_stage_instruction(
                state,
                PipelineStage.SEMANTICS,
                include_input=(not previous or compact_between_stages is not None),
            ),
        ]
        result = semantics_agent.invoke(
            {
                **previous,
                "messages": messages,
            },
            config=_child_graph_config(config),
        )
        return {
            "semantics_state": result,
            "stage_submission": result.get("structured_response"),
        }

    def run_operations(state: ReconstructionState, config: RunnableConfig):
        snapshot = current_snapshot(state)
        if snapshot.last_completed_stage is not PipelineStage.SEMANTICS:
            raise RuntimeError("operations requires integrated semantics")

        if not tickets_assigned_to(snapshot.open_tickets, PipelineStage.OPERATIONS):
            return {"stage_submission": OperationSubmission.unchanged()}

        previous = state.get("operations_state") or {}
        messages = [
            *list(previous.get("messages") or []),
            build_stage_instruction(
                state,
                PipelineStage.OPERATIONS,
                include_input=(not previous or compact_between_stages is not None),
            ),
        ]
        result = operations_agent.invoke(
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

    def run_coding(state: ReconstructionState, config: RunnableConfig):
        snapshot = current_snapshot(state)
        if snapshot.last_completed_stage is not PipelineStage.OPERATIONS:
            raise RuntimeError("coding requires integrated operations")

        previous = state.get("coding_state") or {}
        messages = [
            *list(previous.get("messages") or []),
            build_stage_instruction(
                state,
                PipelineStage.CODING,
                include_input=(not previous or compact_between_stages is not None),
            ),
        ]
        result = coding_agent.invoke(
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

    def integrate_stage_submission(
        state: ReconstructionState,
    ) -> dict[str, Any]:
        """Validate and atomically integrate the pending reasoning output."""
        submission = state.get("stage_submission")
        if not isinstance(submission, StageSubmission):
            return _rejected_stage_submission(
                state,
                "the reasoning stage did not return a StageSubmission",
            )

        reconstruction = state.get("reconstruction")
        if reconstruction is None:
            raise RuntimeError("stage integration requires reconstruction")

        verification = None
        if current_snapshot(state).last_completed_stage is PipelineStage.OPERATIONS:
            if not isinstance(submission, CodingSubmission):
                return _rejected_stage_submission(
                    state,
                    "coding did not return a CodingSubmission",
                )
            verification, _ = verifier.verify()

        try:
            updated = advance_reconstruction(
                reconstruction,
                submission,
                verification=verification,
            )
        except SubmissionValidationError as error:
            return _rejected_stage_submission(state, str(error))

        save_reconstruction(
            sandbox_workdir.host_bind_dir / reconstruction_history_filename,
            updated,
        )
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

    def run_audit(state: ReconstructionState, config: RunnableConfig):
        snapshot = current_snapshot(state)
        if snapshot.last_completed_stage is not PipelineStage.CODING:
            raise RuntimeError("audit requires a completed coding snapshot")
        verification = snapshot.verification
        if verification is None:
            raise RuntimeError("audit requires verification")

        attempts_dir = sandbox_workdir.sandbox_bind_dir / verification_dirname
        attempt_dir = str(
            attempts_dir / verification.verification_id
            if verification.verification_id is not None
            else attempts_dir
        )
        previous = state.get("audit_state") or {}
        instruction = build_stage_instruction(
            state,
            PipelineStage.AUDIT,
            include_input=not previous,
            attempt_dir=attempt_dir,
        )
        result = audit_agent.invoke(
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
            updated = open_next_round(reconstruction, report)
            save_reconstruction(
                sandbox_workdir.host_bind_dir / reconstruction_history_filename,
                updated,
            )
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
            return PipelineStage.SEMANTICS.value
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
    workflow.add_node(PipelineStage.SEMANTICS.value, run_semantics)
    workflow.add_node(PipelineStage.OPERATIONS.value, run_operations)
    workflow.add_node(PipelineStage.CODING.value, run_coding)
    workflow.add_node(PipelineStage.AUDIT.value, run_audit)
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
