import json
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any, Literal, Protocol, cast

from langchain.agents.middleware import AgentMiddleware
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
from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.tools import (
    OutputVerifier,
    create_load_image_tool,
    create_run_shell_tool,
)
from zeroshot.pipeline.verification import CadQueryExecutor, StepRenderer
from zeroshot.pipeline.workflow._config import _child_graph_config
from zeroshot.pipeline.workflow.middleware import VerifyOnWriteMiddleware
from zeroshot.pipeline.workflow.state import Audit, ReconstructionState

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


class ProposerReviewerBuilder(Protocol):
    def __call__(
        self,
        *,
        tools: Sequence[BaseTool],
        prompt_context: Mapping[str, str],
    ) -> CompiledGraph: ...


_STAGE_ORDER = ["semantics", "operations", "coding", "audit"]


# audit rerouting
def _invocation_reason(
    state: ReconstructionState,
    current_stage: Literal["semantics", "operations", "coding"],
) -> tuple[Literal["initial", "targeted_redo", "upstream_changed"], str]:
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
    semantics_agent_builder: ProposerReviewerBuilder,
    operations_agent_builder: ProposerReviewerBuilder,
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
    checkpointer: BaseCheckpointSaver[Any] | None = None,
):
    """Interpret and plan the part, implement it, then verify and audit it."""
    if max_audit_reject_count <= 0:
        raise ValueError(f"{max_audit_reject_count=} must be positive")

    # Create tools
    executor = CadQueryExecutor(sandbox_runner=sandbox_runner)
    basic_tools = [
        create_run_shell_tool(sandbox_runner, sandbox_workdir),
        create_load_image_tool(sandbox_workdir),
    ]
    # One verifier for the whole graph: the coder's turns and the workflow's
    # own final check number their attempts from the same directory.
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
    )
    operations_agent = operations_agent_builder(
        tools=basic_tools,
        prompt_context=prompt_context,
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
            rationale=audit_rationale,
        )
        reviewer_instruction = build_instruction("semantics/review")

        if entry_reason == "initial":
            # `merge_message_runs` merges multiple HumanMessages into one
            (proposer_instruction,) = merge_message_runs(
                [proposer_instruction, _prepare_input_message()]
            )
            (reviewer_instruction,) = merge_message_runs(
                [reviewer_instruction, _prepare_input_message()]
            )

        # invoke (proposer_reviewer subgraph provides `xxx_entry_instruction`)
        result = semantics_agent.invoke(
            {
                **previous,
                "proposer_entry_instruction": proposer_instruction,
                "reviewer_entry_instruction": reviewer_instruction,
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
        semantic_hypothesis_json = semantic_hypothesis.model_dump_json(indent=2)

        previous = state.get("operations_state") or {}
        entry_reason, audit_rationale = _invocation_reason(state, "operations")

        proposer_instruction = build_instruction(
            f"operations/{entry_reason}",
            semantic_hypothesis=semantic_hypothesis_json,
            rationale=audit_rationale,
        )
        reviewer_instruction = build_instruction(
            "operations/review",
            semantic_hypothesis=semantic_hypothesis_json,
        )

        if entry_reason == "initial":
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

    def after_operations(
        state: ReconstructionState,
    ) -> Literal["coding", "__end__"]:
        return "coding" if state.get("operation_plan") is not None else "__end__"

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
            semantic_hypothesis=semantic_hypothesis.model_dump_json(indent=2),
            operation_plan=operation_plan.model_dump_json(indent=2),
            rationale=audit_rationale,
        )

        if entry_reason == "initial":
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
            semantic_hypothesis=semantic_hypothesis.model_dump_json(indent=2),
            operation_plan=operation_plan.model_dump_json(indent=2),
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
        if state.get("audit_reject_count", 0) > max_audit_reject_count:
            return "__end__"
        return audit.revise

    # Construct a graph
    workflow = StateGraph(state_schema=ReconstructionState)  # type: ignore[type-var]
    workflow.add_node("semantics", run_semantics)
    workflow.add_node("operations", run_operations)
    workflow.add_node("coding", write_code)
    workflow.add_node("verify_final", verify_final)
    workflow.add_node("audit", audit_output)

    workflow.add_edge(START, "semantics")
    workflow.add_conditional_edges("semantics", after_semantics)
    workflow.add_conditional_edges("operations", after_operations)
    workflow.add_edge("coding", "verify_final")
    workflow.add_edge("verify_final", "audit")
    workflow.add_conditional_edges("audit", after_audit)

    graph = workflow.compile(checkpointer=checkpointer)
    # Every send-back is another lap of the graph, and LangGraph's default
    # ceiling is lower than the laps this budget allows.  Counted from the graph
    # itself so that adding a node cannot leave the ceiling behind.
    return graph.with_config(
        recursion_limit=len(workflow.nodes) * (max_audit_reject_count + 2)
    )
