"""Single-stage baseline: the coder reads the drawing itself; audit findings loop back."""

import json
from collections.abc import Sequence
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any, NotRequired, TypedDict, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, merge_message_runs
from langchain_core.messages.content import create_text_block
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.pregel import Pregel

from zeroshot.pipeline.messages.artifact import ArtifactPresenter
from zeroshot.pipeline.messages.manifest import InputManifest
from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.stages._base.prompt import (
    PromptTemplate,
    StageInstructions,
    build_system_prompt,
)
from zeroshot.pipeline.stages.interpretation.contracts import View
from zeroshot.pipeline.tools.calculate_drawing_scale import (
    create_calculate_drawing_scale_tool,
)
from zeroshot.pipeline.tools.load_image import create_load_image_tool
from zeroshot.pipeline.tools.run_shell import create_run_shell_tool
from zeroshot.pipeline.verification import (
    AttemptStore,
    CadQueryExecutor,
    OutputVerifier,
    StepRenderer,
)
from zeroshot.pipeline.workflow._config import _child_graph_config
from zeroshot.pipeline.workflow.components import compact_transcript
from zeroshot.pipeline.workflow.components.agent import AgentState
from zeroshot.pipeline.workflow.middleware import VerifyOnWriteMiddleware
from zeroshot.pipeline_single.contracts import CodingReport, SingleAuditReport

type AgentBuilder = partial[Pregel[Any, Any, Any, Any]]

_PROMPTS = Path(__file__).parent / "prompts"
_COORDINATE_FRAMES = (
    Path(__file__).parents[1] / "pipeline/stages/_base/prompts/coordinate_frames.md"
)


class SingleState(TypedDict):
    coding_state: NotRequired[AgentState]
    audit_state: NotRequired[AgentState]
    round: NotRequired[int]
    # JSON-ready dicts, so the checkpointer needs no allowlist for this package.
    stage_submission: NotRequired[dict[str, Any] | None]
    audit_report: NotRequired[dict[str, Any] | None]
    audit_findings: NotRequired[list[dict[str, Any]]]
    verification: NotRequired[dict[str, Any]]
    stage_validation_error: NotRequired[str | None]
    stage_validation_failure_count: NotRequired[int]


def _without_answer(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "structured_response"}


def create_single_graph(
    coding_agent_builder: AgentBuilder,
    audit_agent_builder: AgentBuilder,
    compact_between_stages: BaseChatModel,
    sandbox_runner: SandboxRunner,
    sandbox_workdir: SandboxWorkdir,
    artifact_presenter: ArtifactPresenter,
    input_manifest: InputManifest,
    output_filename: str = "model.py",
    verification_dirname: PurePosixPath = PurePosixPath("attempts"),
    projection_views: Sequence[str] = ("front", "top", "right"),
    max_audit_reject_count: int = 1,
    max_stage_validation_retries: int = 3,
    dxf_mm_per_unit: dict[str, float] | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
):
    """Code the part from the drawing, audit it, and recode from the findings."""
    del dxf_mm_per_unit  # read by run_pipeline, not by this graph
    run_shell = create_run_shell_tool(sandbox_runner, sandbox_workdir)
    load_image = create_load_image_tool(sandbox_workdir)

    # The attempt store asks for the round when it issues a directory.
    rounds = [0]
    attempt_store = AttemptStore(
        sandbox_workdir,
        round_source=lambda: rounds[0],
        root_dirname=verification_dirname,
    )
    prompt_context = {
        "coding_output_path": str(sandbox_workdir.sandbox_bind_dir / output_filename),
        "verification_dir": str(
            sandbox_workdir.sandbox_bind_dir / verification_dirname
        ),
    }
    drawing = StageInstructions(
        input_artifact=input_manifest.drawing,
        input_presentation_mode=artifact_presenter.input_mode,
        prompt_context=prompt_context,
        workdir=sandbox_workdir,
    )

    output_verifier = OutputVerifier(
        executor=CadQueryExecutor(sandbox_runner=sandbox_runner),
        workdir=sandbox_workdir,
        renderer=StepRenderer(),
        feedback_presentation_mode=artifact_presenter.feedback_mode,
        attempt_store=attempt_store,
        views=[View(view) for view in projection_views],
        source_filename=output_filename,
        # `ret_` returns name planned operations; there is no plan here.
        show_intermediate_returns=False,
    )
    coding_middleware = VerifyOnWriteMiddleware(
        output_verifier,
        refusal=(
            "The current program must produce a verified solid, and its "
            "verification feedback must be shown before submission. Read the "
            "feedback, correct model.py, and submit only after verification "
            "succeeds."
        ),
        require_feedback_before_submit=True,
    )
    coding_agent = coding_agent_builder(
        tools=[run_shell, load_image, create_calculate_drawing_scale_tool()],
        system_prompt=build_system_prompt(
            _PROMPTS / "coder_role.md", prompt_context, CodingReport
        ),
        output_schema=CodingReport,
        extra_middleware=[coding_middleware],
    )
    audit_agent = audit_agent_builder(
        tools=[run_shell, load_image],
        system_prompt=build_system_prompt(
            _PROMPTS / "auditor_role.md",
            prompt_context | {"max_turns": audit_agent_builder.keywords["max_turns"]},
            SingleAuditReport,
        ),
        output_schema=SingleAuditReport,
    )

    def instruction(
        state: SingleState, name: str, templates: Sequence[Path], **context: str
    ) -> HumanMessage:
        if error := state.get("stage_validation_error"):
            return HumanMessage(
                content_blocks=[
                    create_text_block(
                        f"[{name.title()} Validation Error]\n"
                        f"Your previous {name} stage output was rejected. "
                        "Return the corrected complete output using this "
                        f"feedback:\n\n{error}"
                    )
                ]
            )
        context = (
            prompt_context | {"current_round": str(state.get("round", 0))} | context
        )
        text = "\n\n".join(PromptTemplate(path).render(**context) for path in templates)
        (message,) = merge_message_runs(
            [
                HumanMessage(content_blocks=[create_text_block(text)]),
                drawing.create_artifact_message(),
            ]
        )
        return cast(HumanMessage, message)

    def failure(state: SingleState, error: str) -> dict[str, Any]:
        return {
            "stage_validation_error": error,
            "stage_validation_failure_count": (
                state.get("stage_validation_failure_count", 0) + 1
            ),
        }

    cleared = {"stage_validation_error": None, "stage_validation_failure_count": 0}

    def coding(state: SingleState, config: RunnableConfig) -> dict[str, Any]:
        rounds[0] = state.get("round", 0)
        if state.get("stage_validation_error") is None:
            output_verifier.reset()
            coding_middleware.reset()
        findings = state.get("audit_findings")
        previous = state.get("coding_state") or {}
        message = instruction(
            state,
            "coding",
            [
                _COORDINATE_FRAMES,
                _PROMPTS / "coding_round.md",
                _PROMPTS / "coding_guidelines.md",
            ],
            audit_findings=json.dumps(findings, indent=2) if findings else "none",
        )
        result = coding_agent.invoke(
            {**previous, "messages": [*previous.get("messages", []), message]},
            config=_child_graph_config(config),
        )
        report = result.get("structured_response")
        verification, _ = output_verifier.verify()
        update = {
            "coding_state": _without_answer(result),
            "stage_submission": None if report is None else report.model_dump(),
            "verification": {
                "verification_id": verification.verification_id,
                "status": verification.status.value,
            },
        }
        if report is None:
            return update | failure(state, "the coder did not return its CodingReport")
        return update | cleared

    def after_coding(state: SingleState) -> str:
        if state.get("stage_validation_error") is not None:
            retry = (
                state["stage_validation_failure_count"] <= max_stage_validation_retries
            )
            return "coding" if retry else END
        if state.get("round", 0) >= max_audit_reject_count:
            return END
        return "coding_handover"

    def coding_handover(state: SingleState, config: RunnableConfig) -> dict[str, Any]:
        thread = compact_transcript(
            state["coding_state"]["messages"],
            model=compact_between_stages,
            config=config,
        )
        return {
            "coding_state": {
                **state["coding_state"],
                "messages": thread,
                "reported_message_count": len(thread),
            }
        }

    def audit(state: SingleState, config: RunnableConfig) -> dict[str, Any]:
        verification = state["verification"]
        attempt_dir = (
            attempt_store.sandbox_root
            / f"round_{state.get('round', 0):03d}/coding"
            / verification["verification_id"]
            if verification["verification_id"] is not None
            else attempt_store.sandbox_root
        )
        previous = state.get("audit_state") or {}
        message = instruction(
            state,
            "audit",
            [_PROMPTS / "audit_round.md"],
            attempt_dir=str(attempt_dir),
            verification_status=verification["status"],
            coding_report=json.dumps(state.get("stage_submission"), indent=2),
        )
        result = audit_agent.invoke(
            {**previous, "messages": [*previous.get("messages", []), message]},
            config=_child_graph_config(config),
        )
        report = result.get("structured_response")
        update: dict[str, Any] = {
            "audit_state": _without_answer(result),
            "audit_report": None if report is None else report.model_dump(),
        }
        if report is None:
            return update | failure(state, "the auditor did not return its report")
        if report.accepted:
            return update | cleared
        return (
            update
            | cleared
            | {
                "round": state.get("round", 0) + 1,
                "audit_findings": update["audit_report"]["findings"],
            }
        )

    def after_audit(state: SingleState) -> str:
        if state.get("stage_validation_error") is not None:
            retry = (
                state["stage_validation_failure_count"] <= max_stage_validation_retries
            )
            return "audit" if retry else END
        report = state.get("audit_report") or {}
        return END if report.get("accepted") else "coding"

    workflow = StateGraph(state_schema=SingleState)  # type: ignore[type-var]
    workflow.add_node("coding", coding)
    workflow.add_node("coding_handover", coding_handover)
    workflow.add_node("audit", audit)
    workflow.add_edge(START, "coding")
    workflow.add_conditional_edges("coding", after_coding)
    workflow.add_edge("coding_handover", "audit")
    workflow.add_conditional_edges("audit", after_audit)
    return workflow.compile(checkpointer=checkpointer).with_config(
        recursion_limit=3
        * (max_audit_reject_count + 1)
        * (max_stage_validation_retries + 1)
        + 10
    )
