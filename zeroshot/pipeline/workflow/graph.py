import json
import re
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, cast

from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, StateGraph
from langgraph.pregel import Pregel
from pydantic import BaseModel

from zeroshot.pipeline.messages import (
    ArtifactPresenter,
    DrawingSource,
    InputManifest,
    drawing_for_model,
)
from zeroshot.pipeline.messages.contracts.audit import AuditReport
from zeroshot.pipeline.messages.contracts.reconstruction import (
    CodingSubmission,
    DrawingSubmission,
    OperationSubmission,
    ReconstructionRun,
    SemanticSubmission,
    TicketAnswers,
    tickets_assigned_to,
)
from zeroshot.pipeline.messages.contracts.stages import (
    REASONING_STAGES,
    PipelineStage,
    ReasoningStage,
    next_stage,
)
from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.stages._base.prompt import PromptTemplate, StageInstructions, build_system_prompt
from zeroshot.pipeline.tools import (
    create_calculate_drawing_scale_tool,
    create_load_image_tool,
    create_run_shell_tool,
)
from zeroshot.pipeline.verification import (
    AttemptStore,
    CadQueryExecutor,
    DrawingVerifier,
    OutputVerifier,
    StepRenderer,
)
from zeroshot.pipeline.verification._run_program import INTERMEDIATE_RETURNS_DIR
from zeroshot.pipeline.workflow._config import _child_graph_config
from zeroshot.pipeline.workflow.components import compact_transcript
from zeroshot.pipeline.workflow.middleware import VerifyOnWriteMiddleware
from zeroshot.pipeline.workflow.reconstruction import (
    advance_reconstruction,
    drawing_baseline,
    load_reconstruction,
    open_next_round,
    save_reconstruction,
    start_reconstruction,
)
from zeroshot.pipeline.workflow.state import (
    ReconstructionState,
    carry_thread,
    current_snapshot,
    lead_transcript,
)
from zeroshot.pipeline.workflow.validate_submission import (
    SubmissionValidationError,
    validate_submission,
)

type AgentBuilder = partial[CompiledGraph]
type CompiledGraph = Pregel[Any, Any, Any, Any]


def create_reconstruction_graph(
    drawings_agent_builder: AgentBuilder,
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
    drawing_filename: str = "drawing.json",
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
    executor = CadQueryExecutor(sandbox_runner=sandbox_runner)
    drawing_verifier = DrawingVerifier(
        workdir=sandbox_workdir,
        attempt_store=attempt_store,
        artifact_presenter=artifact_presenter,
        source_filename=drawing_filename,
    )
    drawing_middleware = VerifyOnWriteMiddleware(
        drawing_verifier,
        refusal=(
            "The drawing artifact is not ready to submit. Inspect the rendered "
            "views, correct drawing.json if needed, and submit only after the "
            "current version has been shown back to you and validates."
        ),
        require_feedback_before_submit=True,
    )
    coding_verifier = OutputVerifier(
        executor=executor,
        workdir=sandbox_workdir,
        renderer=renderer,
        artifact_presenter=artifact_presenter,
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

    # instantiate agents
    prompt_context = {
        "coding_output_path": str(sandbox_workdir.sandbox_bind_dir / output_filename),
        "drawing_output_path": str(sandbox_workdir.sandbox_bind_dir / drawing_filename),
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
        artifact_presenter=artifact_presenter,
        input_manifest=input_manifest,
        workdir=sandbox_workdir,
    )

    share_thread_system_prompt = Path(__file__).resolve().parents[1] / "stages/_base/prompts/cad_reconstructor.md"

    drawings_agent = drawings_agent_builder(
        # Only this stage measures a raster, so only it is offered the fit.
        tools=[*basic_tools, create_calculate_drawing_scale_tool()],
        system_prompt=build_system_prompt(
            share_thread_system_prompt if share_thread else Path(__file__).resolve().parents[1] / "stages/drawings/prompts/role.md",
            prompt_context | {"max_turns": drawings_agent_builder.keywords["max_turns"]},
            DrawingSubmission,
        ),
        output_schema=DrawingSubmission,
        extra_middleware=[drawing_middleware],
    )
    semantics_agent = semantics_agent_builder(
        tools=basic_tools,
        system_prompt=build_system_prompt(
            share_thread_system_prompt if share_thread else Path(__file__).resolve().parents[1] / "stages/semantics/prompts/role.md",
            prompt_context | {"max_turns": semantics_agent_builder.keywords["max_turns"]},
            SemanticSubmission,
        ),
        output_schema=SemanticSubmission,
    )
    operations_agent = operations_agent_builder(
        tools=basic_tools,
        system_prompt=build_system_prompt(
            share_thread_system_prompt if share_thread else Path(__file__).resolve().parents[1] / "stages/operations/prompts/role.md",
            prompt_context | {"max_turns": operations_agent_builder.keywords["max_turns"]},
            OperationSubmission
        ),
        output_schema=OperationSubmission,
    )
    coding_agent = coding_agent_builder(
        tools=basic_tools,
        system_prompt=build_system_prompt(
            share_thread_system_prompt if share_thread else Path(__file__).resolve().parents[1] / "stages/coding/prompts/role.md",
            prompt_context | {"max_turns": coding_agent_builder.keywords["max_turns"]},
            CodingSubmission,
        ),
        output_schema=CodingSubmission,
        extra_middleware=[coding_middleware],
    )
    audit_agent = audit_agent_builder(
        tools=basic_tools,
        system_prompt=build_system_prompt(
            Path(__file__).resolve().parents[1] / "stages/audit/prompts/role.md",
            prompt_context | {"max_turns": audit_agent_builder.keywords["max_turns"]},
            AuditReport
        ),
        output_schema=AuditReport,
    )


    def save_history(run: ReconstructionRun) -> None:
        save_reconstruction(history_path, run)

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
                # Addressed the way the model will read them, because the model
                # is what reads and revises this history from here on.
                drawings=drawing_for_model(input_manifest.drawing, sandbox_workdir),
            )
        save_history(run)
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

    # ------------------------------------------------------------------
    # Reasoning-stage inference
    # ------------------------------------------------------------------

    def run_drawings(state: ReconstructionState, config: RunnableConfig):
        snapshot = current_snapshot(state)
        if state.get("stage_validation_error") is None:
            drawing_verifier.reset(drawing_baseline(state["reconstruction"]))
            drawing_middleware.reset()
        if not tickets_assigned_to(snapshot.open_tickets, PipelineStage.DRAWINGS):
            return {"stage_submission": DrawingSubmission.unchanged()}

        previous = state.get("drawings_state") or {}
        messages = [
            *list(previous.get("messages") or []),
            stage_instructions.build(
                state,
                PipelineStage.DRAWINGS,
                include_input=(not previous or compact_between_stages is not None),
            ),
        ]
        result = drawings_agent.invoke(
            {
                **previous,
                "messages": messages,
            },
            config=_child_graph_config(config),
        )
        return {
            "drawings_state": result,
            "stage_submission": result.get("structured_response"),
        }

    def run_semantics(state: ReconstructionState, config: RunnableConfig):
        snapshot = current_snapshot(state)
        if snapshot.last_completed_stage is not PipelineStage.DRAWINGS:
            raise RuntimeError("semantics requires an integrated drawing")

        if not tickets_assigned_to(snapshot.open_tickets, PipelineStage.SEMANTICS):
            return {"stage_submission": SemanticSubmission.unchanged()}

        previous = state.get("semantics_state") or {}
        messages = [
            *list(previous.get("messages") or []),
            stage_instructions.build(
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
            stage_instructions.build(
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
        drawing = cast(DrawingSource, snapshot.drawings)

        # The verifier redraws the solid in the views the drawing names, and
        # guesses none. Set here because both the build inside the agent and
        # the one at integration belong to this stage of this round.
        if state.get("stage_validation_error") is None:
            coding_verifier.reset()
            coding_middleware.reset()
        coding_verifier.views = [sheet.role for sheet in drawing.orthographic()]

        previous = state.get("coding_state") or {}
        messages = [
            *list(previous.get("messages") or []),
            stage_instructions.build(
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
        workspace_output = None
        if stage is PipelineStage.DRAWINGS:
            if tickets_assigned_to(snapshot.open_tickets, PipelineStage.DRAWINGS):
                workspace_output = drawing_verifier.accepted_drawing
                if workspace_output is None:
                    return _rejected_stage_submission(
                        state,
                        "drawing.json has not been structurally validated and "
                        "rendered for this submission",
                    )
            else:
                workspace_output = drawing_baseline(reconstruction)
        elif stage is PipelineStage.CODING:
            workspace_output, _ = coding_verifier.verify()

        try:
            updated = advance_reconstruction(
                reconstruction,
                submission,
                workspace_output=workspace_output,
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

    def run_audit(state: ReconstructionState, config: RunnableConfig):
        snapshot = current_snapshot(state)
        if snapshot.last_completed_stage is not PipelineStage.CODING:
            raise RuntimeError("audit requires a completed coding snapshot")
        verification = snapshot.verification
        if verification is None:
            raise RuntimeError("audit requires verification")

        attempt_dir = str(
            attempt_store.sandbox_attempt_dir(
                snapshot.round, "coding", verification.verification_id
            )
            if verification.verification_id is not None
            else attempt_store.sandbox_root
        )
        drawing_attempt = attempt_store.latest_sandbox_attempt_dir(
            "drawing", snapshot.round
        )
        previous = state.get("audit_state") or {}
        instruction = stage_instructions.build(
            state,
            PipelineStage.AUDIT,
            include_input=not previous,
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
            # Asked of the build rather than of the config that enabled it:
            # the report carries this only when returns were actually written,
            # so the layout the auditor is given cannot name a directory the
            # attempt does not have.
            intermediate_returns=PromptTemplate(
                Path(__file__).resolve().parents[1] / "stages/audit/prompts/intermediate_returns.md"
            ).render(returns_dir=INTERMEDIATE_RETURNS_DIR) if bool(verification.intermediate_returns) else ""
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
            return PipelineStage.DRAWINGS.value
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
    workflow.add_node(PipelineStage.DRAWINGS.value, run_drawings)
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
