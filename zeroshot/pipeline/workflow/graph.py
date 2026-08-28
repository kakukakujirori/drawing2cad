import json
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import PurePosixPath
from typing import Any, Literal, Protocol, cast

from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, merge_message_runs
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
from zeroshot.pipeline.messages.contracts import (
    OperationPlan,
    SemanticHypothesis,
    render_hypothesis,
    render_plan,
    render_plan_review,
    review_plan,
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
from zeroshot.pipeline.workflow.state import (
    Audit,
    ReasoningStage,
    ReconstructionState,
    carry_thread,
    lead_transcript,
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


class ReasoningStageBuilder(Protocol):
    def __call__(
        self,
        *,
        tools: Sequence[BaseTool],
        prompt_context: Mapping[str, str],
        proposal_schema: type[BaseModel],
    ) -> CompiledGraph: ...


class FanoutReduceBuilder(Protocol):
    def __call__(
        self,
        *,
        tools: Sequence[BaseTool],
        prompt_context: Mapping[str, str],
        fanout_workdir_prefix: str,
        proposal_schema: type[BaseModel],
    ) -> CompiledGraph: ...


_STAGE_ORDER = ["semantics", "operations", "coding", "audit"]


# audit rerouting
def _invocation_reason(
    state: ReconstructionState,
    current_stage: ReasoningStage,
) -> tuple[Literal["initial", "targeted_redo", "upstream_changed"], str]:
    # The plan read against its hypothesis. Ahead of the audit, because a fault
    # in the plan that is in state now outranks a verdict from a lap earlier.
    # `describes` is what makes that safe: a reading left over from earlier
    # work answers no and falls through, so nobody has to remember to delete it.
    review = state.get("plan_review")
    plan = state.get("operation_plan")
    hypothesis = state.get("semantic_hypothesis")
    faulty = (
        review is not None
        and plan is not None
        and hypothesis is not None
        and review.describes(hypothesis, plan)
        and not review.sound
    )
    if current_stage == "operations" and faulty:
        return "targeted_redo", render_plan_review(review)

    audit = state.get("audit")
    if audit is None:
        return "initial", ""

    restart_from = audit.revise
    rationale = audit.rationale

    if restart_from is None:
        raise RuntimeError(
            f"_invocation_reason called for {current_stage} after audit accept."
        )
    elif restart_from == current_stage:
        return "targeted_redo", rationale
    elif _STAGE_ORDER.index(restart_from) < _STAGE_ORDER.index(current_stage):
        return "upstream_changed", rationale

    raise RuntimeError(
        f"{current_stage!r} cannot run after restart from {restart_from!r}"
    )


def create_reconstruction_graph(
    semantics_agent_builder: FanoutReduceBuilder,
    operations_agent_builder: ReasoningStageBuilder,
    coding_agent_builder: AgentBuilder,
    audit_agent_builder: AgentBuilder,
    sandbox_runner: SandboxRunner,
    sandbox_workdir: SandboxWorkdir,
    renderer: StepRenderer,
    artifact_presenter: ArtifactPresenter,
    input_manifest: InputManifest,
    output_filename: str = "model.py",
    verification_dirname: PurePosixPath = PurePosixPath("attempts"),
    max_audit_reject_count: int = 3,
    share_thread: bool = False,
    compact_between_stages: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
):
    """Interpret and plan the part, implement it, then verify and audit it."""
    if max_audit_reject_count <= 0:
        raise ValueError(f"{max_audit_reject_count=} must be positive")
    if compact_between_stages is not None and not share_thread:
        raise ValueError(
            "compact_between_stages needs share_thread: with a transcript per "
            "stage there is no handover at which to compact anything."
        )

    # Create tools
    executor = CadQueryExecutor(sandbox_runner=sandbox_runner)
    basic_tools = [
        create_run_shell_tool(sandbox_runner, sandbox_workdir),
        create_load_image_tool(sandbox_workdir),
    ]
    verifier = OutputVerifier(
        executor=executor,
        workdir=sandbox_workdir,
        renderer=renderer,
        artifact_presenter=artifact_presenter,
        source_filename=output_filename,
        output_dirname=verification_dirname,
    )

    # instantiate agents
    prompt_context = {
        "output_path": str(sandbox_workdir.sandbox_bind_dir / output_filename),
        "verification_dir": str(
            sandbox_workdir.sandbox_bind_dir / verification_dirname
        ),
    }

    semantics_agent = semantics_agent_builder(
        tools=basic_tools,
        prompt_context=prompt_context,
        proposal_schema=SemanticHypothesis,
        fanout_workdir_prefix=str(
            sandbox_workdir.sandbox_bind_dir / "semantic_hypothesis"
        ),
    )
    operations_agent = operations_agent_builder(
        tools=basic_tools,
        prompt_context=prompt_context,
        proposal_schema=OperationPlan,
    )
    coding_agent = coding_agent_builder(
        tools=basic_tools,
        prompt_context=prompt_context,
        extra_middleware=[VerifyOnWriteMiddleware(verifier)],
    )
    audit_agent = audit_agent_builder(
        tools=basic_tools,
        prompt_context=prompt_context,
        output_schema=Audit,
    )

    # prepare inputs (by calling it on-the-fly, prevent message_id duplication)
    def _prepare_input_message():
        return HumanMessage(
            content_blocks=artifact_presenter.build_input_message_blocks(
                manifest=input_manifest,
                workdir=sandbox_workdir,
            )
        )

    # define nodes
    def run_semantics(state: ReconstructionState, config: RunnableConfig):
        previous = state.get("semantics_state") or {}
        entry_reason, audit_rationale = _invocation_reason(state, "semantics")

        proposer_instruction = build_instruction(
            f"semantics/{entry_reason}",
            **prompt_context,
            rationale=audit_rationale,
        )
        if entry_reason == "initial" or compact_between_stages is not None:
            # `merge_message_runs` merges multiple HumanMessages into one
            (proposer_instruction,) = merge_message_runs(
                [proposer_instruction, _prepare_input_message()]
            )

        # The first call fans out; later audit-driven calls revise through the
        # reducer without re-running the independent proposers.
        result = semantics_agent.invoke(
            {
                **previous,
                "invocation_instruction": proposer_instruction,
            },
            config=_child_graph_config(config),
        )

        return {
            "semantics_state": result,
            "semantic_hypothesis": result.get("proposal"),
        }

    def after_semantics(
        state: ReconstructionState,
    ) -> Literal["operations", "__end__"]:
        return (
            "operations" if state.get("semantic_hypothesis") is not None else "__end__"
        )

    def run_operations(state: ReconstructionState, config: RunnableConfig):
        semantic_hypothesis = state.get("semantic_hypothesis")
        if semantic_hypothesis is None:
            raise RuntimeError("operations requires semantic_hypothesis")
        semantic_hypothesis_json = render_hypothesis(semantic_hypothesis)

        previous = state.get("operations_state") or {}
        entry_reason, audit_rationale = _invocation_reason(state, "operations")

        proposer_instruction = build_instruction(
            f"operations/{entry_reason}",
            **prompt_context,
            semantic_hypothesis=semantic_hypothesis_json,
            rationale=audit_rationale,
        )
        reviewer_instruction = build_instruction(
            "operations/review",
            semantic_hypothesis=semantic_hypothesis_json,
        )

        if entry_reason == "initial" or compact_between_stages is not None:
            # `merge_message_runs` merges multiple HumanMessages into one
            (proposer_instruction,) = merge_message_runs(
                [proposer_instruction, _prepare_input_message()]
            )
            (reviewer_instruction,) = merge_message_runs(
                [reviewer_instruction, _prepare_input_message()]
            )

        # invoke (proposer_reviewer subgraph provides `xxx_entry_instruction`)
        result = operations_agent.invoke(
            {
                **previous,
                "proposer_entry_instruction": proposer_instruction,
                "reviewer_entry_instruction": reviewer_instruction,
            },
            config=_child_graph_config(config),
        )

        return {
            "operations_state": result,
            "operation_plan": result.get("proposal"),
        }

    def check_plan(state: ReconstructionState) -> dict[str, Any]:
        """Read the plan against the hypothesis it was made from.

        The only place holding both, so the only place that can see a feature
        the plan dropped, a feature it invented, a measurement it copied out
        instead of citing, or a citation that stands for nothing.
        """
        hypothesis = state.get("semantic_hypothesis")
        plan = state.get("operation_plan")
        if hypothesis is None or plan is None:
            return {}

        return {"plan_review": review_plan(plan, hypothesis)}

    def after_operations(
        state: ReconstructionState,
    ) -> Literal["coding", "operations", "__end__"]:
        if state.get("operation_plan") is None:
            return "__end__"
        review = state.get("plan_review")
        if review is None or review.sound:
            return "coding"
        return "operations"

    def write_code(state: ReconstructionState, config: RunnableConfig):
        semantic_hypothesis = state.get("semantic_hypothesis")
        operation_plan = state.get("operation_plan")
        if semantic_hypothesis is None or operation_plan is None:
            raise RuntimeError(
                "coding requires both semantic_hypothesis and operation_plan"
            )

        # add messages
        previous = state.get("coding_state") or {}
        messages = list(previous.get("messages") or [])

        entry_reason, audit_rationale = _invocation_reason(state, "coding")
        instruction = build_instruction(
            f"coding/{entry_reason}",
            **prompt_context,
            semantic_hypothesis=render_hypothesis(semantic_hypothesis),
            operation_plan=render_plan(operation_plan, semantic_hypothesis),
            rationale=audit_rationale,
        )

        if entry_reason == "initial" or compact_between_stages is not None:
            (instruction,) = cast(
                list[HumanMessage],
                merge_message_runs([instruction, _prepare_input_message()]),
            )

        messages.append(instruction)

        # invoke
        result = coding_agent.invoke(
            {**previous, "messages": messages},
            config=_child_graph_config(config),
        )
        return {"coding_state": result}

    def verify_final(state: ReconstructionState):
        del state
        # The report alone: the rendered views are feedback for whoever is
        # still editing, and this runs after the coder has stopped.
        report, _ = verifier.verify()
        return {"last_verification": report}

    def after_verification(state: ReconstructionState) -> Literal["audit", "__end__"]:
        """Whether an audit could still send the run back.

        Note that the inequality below includes "=", which means the final audit round
        doesn't run because there is nobody consumes the audit result.
        """
        if state.get("audit_reject_count", 0) >= max_audit_reject_count:
            return "__end__"
        return "audit"

    def audit_output(state: ReconstructionState, config: RunnableConfig):
        """Judge the verified model, and say which stage owns any mismatch."""
        semantic_hypothesis = state.get("semantic_hypothesis")
        operation_plan = state.get("operation_plan")
        verification = state.get("last_verification")
        if semantic_hypothesis is None or operation_plan is None:
            raise RuntimeError("audit requires semantic_hypothesis and operation_plan")
        if verification is None:
            raise RuntimeError("audit requires last_verification")

        attempts_dir = sandbox_workdir.sandbox_bind_dir / verification_dirname
        last_attempt_dir = str(
            attempts_dir / verification.verification_id
            if verification.verification_id is not None
            else attempts_dir
        )

        # add messages
        previous = state.get("audit_state") or {}
        messages = list(previous.get("messages") or [])

        instruction = build_instruction(
            "audit",
            report=json.dumps(verification.serialize(), indent=2),
            attempt_dir=last_attempt_dir,
            output_path=str(sandbox_workdir.sandbox_bind_dir / output_filename),
            semantic_hypothesis=render_hypothesis(semantic_hypothesis),
            operation_plan=render_plan(operation_plan, semantic_hypothesis),
        )

        if not messages:
            # first time, so add input artifact messages
            (instruction,) = cast(
                list[HumanMessage],
                merge_message_runs([instruction, _prepare_input_message()]),
            )

        messages.append(instruction)

        # invoke
        result = audit_agent.invoke(
            {**previous, "messages": messages},
            config=_child_graph_config(config),
        )
        audit = cast(Audit | None, result.get("structured_response"))

        update: dict[str, Any] = {
            "audit_state": result,
            "audit": audit,
        }
        if audit is not None and audit.revise is not None:
            update["audit_reject_count"] = 1

        return update

    def after_audit(
        state: ReconstructionState,
    ) -> Literal["semantics", "operations", "coding", "__end__"]:
        audit = state.get("audit")
        if audit is None or audit.revise is None:
            return "__end__"
        # The budget is spent before the audit runs, not after it: `audit` is
        # only reached while a send-back is still left.
        return audit.revise

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

    def _handover_node(stage: str) -> str:
        return stage + "_handover"

    def _exit(stage: str) -> str:
        """Where a reasoning stage hands over to the rest of the graph."""
        return _handover_node(stage) if share_thread else stage

    # Construct a graph
    workflow = StateGraph(state_schema=ReconstructionState)  # type: ignore[type-var]
    workflow.add_node("semantics", run_semantics)
    workflow.add_node("operations", run_operations)
    workflow.add_node("check_plan", check_plan)
    workflow.add_node("coding", write_code)
    workflow.add_node("verify_final", verify_final)
    workflow.add_node("audit", audit_output)

    if share_thread:
        # broadcast states across all stages
        for stage in ["semantics", "operations", "coding"]:
            workflow.add_node(_handover_node(stage), partial(handover, stage=stage))
            workflow.add_edge(stage, _handover_node(stage))

    workflow.add_edge(START, "semantics")
    workflow.add_conditional_edges(_exit("semantics"), after_semantics)
    workflow.add_edge(_exit("operations"), "check_plan")
    workflow.add_conditional_edges("check_plan", after_operations)
    workflow.add_edge(_exit("coding"), "verify_final")
    workflow.add_conditional_edges("verify_final", after_verification)
    workflow.add_conditional_edges("audit", after_audit)

    graph = workflow.compile(checkpointer=checkpointer)
    # Every send-back is another lap of the graph, and LangGraph's default
    # ceiling is lower than the laps this budget allows.  Counted from the graph
    # itself so that adding a node cannot leave the ceiling behind.
    return graph.with_config(
        recursion_limit=len(workflow.nodes) * (max_audit_reject_count + 2)
    )
