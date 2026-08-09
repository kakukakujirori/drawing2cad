import json
from pathlib import PurePosixPath
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from zeroshot.pipeline.messages import MessageBuilder, build_instruction
from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.tools import (
    create_load_image_tool,
    create_run_shell_tool,
    create_verify_output_tool,
)
from zeroshot.pipeline.verification import CadQueryExecutor, StepRenderer
from zeroshot.pipeline.workflow.agent import AgentFactory, StructuredOutput, run_agent
from zeroshot.pipeline.workflow.proposer_critic import (
    Opening,
    ProposerCriticSpec,
    StageFactory,
    StageRunner,
)
from zeroshot.pipeline.workflow.state import (
    Audit,
    OperationPlan,
    ReconstructionState,
    SemanticHypothesis,
)

# What each stage produces, and where the words asking for it live.  The pairing
# is code because it is not a choice: a plan reviewer reviews plans.
SEMANTIC = ProposerCriticSpec(proposal=SemanticHypothesis, instructions="semantic")
OPERATIONS = ProposerCriticSpec(proposal=OperationPlan, instructions="operations")

# Where the audit sends the run back to, in the order the work is done.  The
# keys are `Audit.decision` values, so a verdict the graph cannot route would
# fail to compile rather than at run time -- LangGraph checks a conditional
# edge's destinations against its nodes.  The order is what says whether a
# verdict landed before or after a given node.
_REDO_TARGETS = {
    "redo_semantics": "semantic_stage",
    "redo_operations": "operations_stage",
    "redo_code": "coder",
}


# The audit runs after every node the audit can send the run back to, so it is
# the last position and never a destination.
_ORDER = (*_REDO_TARGETS, "audit")

_PRODUCER_KEY = "workflow_node"
"""Marks which node a message came from. Written and read only in this module."""


def _produced_by(messages: list[BaseMessage], node: str) -> list[BaseMessage]:
    """Sign a node's own messages before they join the run transcript."""
    for message in messages:
        message.additional_kwargs[_PRODUCER_KEY] = node
    return messages


def _through(messages: list[BaseMessage], node: str) -> list[BaseMessage]:
    """The transcript as far as this node, dropping what came after it.

    A node the audit sent back is being asked to redo work that everything
    downstream was built on, so that work is no longer the run so far -- it is
    a branch the workflow has abandoned.  Left in, it would read as settled:
    the same plan already approved, the same program already written.

    The run's own opening message is signed by nobody and belongs to every node.
    """
    limit = _ORDER.index(node)
    return [
        message
        for message in messages
        if (producer := message.additional_kwargs.get(_PRODUCER_KEY)) is None
        or _ORDER.index(producer) <= limit
    ]


def _opening(state: ReconstructionState, redo: str) -> tuple[Opening, dict[str, str]]:
    """Which instruction a node opens on, and what that instruction needs.

    Three reasons a node runs, and the name says which: it is answering for the
    first time, it is answering for a fault the audit found in its own work, or
    what it worked from has been redone since it ran.  Whether a verdict is
    addressed here is the graph's question, not the node's -- the same work
    sits behind a different set of verdicts in a different workflow -- so it is
    decided once, here, for every kind of node.
    """
    audit = state.get("audit")
    if audit is None or audit.decision == "accept":
        return "propose", {}
    if audit.decision == redo:
        return "revise", {"feedback": audit.feedback}
    order = list(_REDO_TARGETS)
    if order.index(audit.decision) < order.index(redo):
        return "restate", {}
    return "propose", {}


def _stage_node(run_stage: StageRunner, artifact: str, redo: str, **upstream: str):
    """Wrap a stage as a graph node: run it, and say what it settled.

    A stage that produced nothing reports why the run is ending instead, since
    the workflow routes past everything downstream of it.
    """

    def node(state: ReconstructionState):
        settled = {
            name: state[key].model_dump_json(indent=2) for name, key in upstream.items()
        }
        opening, context = _opening(state, redo)
        result = run_stage(
            _through(state["messages"], redo),
            opening=opening,
            **context,
            **settled,
        )
        progress = {
            "messages": _produced_by(result.messages, redo),
            "agent_turns": result.turns,
            "stop_reasons": result.stop_reasons,
        }
        if result.artifact is not None:
            return progress | {artifact: result.artifact}
        return progress

    return node


def _settled(artifact: str, then: str, otherwise: str = "verify_final"):
    """Route on whether the stage before this one produced its artifact."""

    def route(state: ReconstructionState):
        return then if state.get(artifact) is not None else otherwise

    return route


def _audited(max_audit_reject_count: int):
    """Route on the audit's verdict, while the run may still be sent back.

    An auditor that spent its budget without a verdict ends the run: the
    workflow has no judgement to act on, and re-running the same code to ask
    again would cost another audit for the same answer.
    """

    def route(state: ReconstructionState):
        audit = state.get("audit")
        if audit is None or audit.decision == "accept":
            return END
        if state.get("audit_reject_count", 0) > max_audit_reject_count:
            return END
        return _REDO_TARGETS[audit.decision]

    return route


def create_reconstruction_graph(
    semantic_stage: StageFactory,
    operations_stage: StageFactory,
    coder: AgentFactory,
    auditor: AgentFactory,
    sandbox_runner: SandboxRunner,
    sandbox_workdir: SandboxWorkdir,
    renderer: StepRenderer,
    message_builder: MessageBuilder,
    input_message: HumanMessage,
    output_filename: str = "model.py",
    verification_dirname: PurePosixPath = PurePosixPath("attempts"),
    structured_output: StructuredOutput = "provider",
    max_audit_reject_count: int = 3,
    model_retries: int = 5,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
):
    """Read the drawing, agree on what the part is, then write and verify it."""
    # Create and bind tools
    executor = CadQueryExecutor(sandbox_runner=sandbox_runner)
    basic_tools = [
        create_run_shell_tool(sandbox_runner, sandbox_workdir),
        create_load_image_tool(sandbox_workdir),
    ]
    # The coder's own verification: evidence it gathers before it stops, not a
    # judgement the workflow acts on.  Whether the solid is the drawing's solid
    # is a later question, and not this tool's to answer.
    submission_tools = [
        create_verify_output_tool(
            executor=executor,
            workdir=sandbox_workdir,
            renderer=renderer,
            message_builder=message_builder,
            source_filename=output_filename,
            output_dirname=verification_dirname,
            serialize_output=True,
        ),
    ]

    prompt_context = {
        "output_path": str(sandbox_workdir.sandbox_bind_dir / output_filename),
        "verification_dir": str(
            sandbox_workdir.sandbox_bind_dir / verification_dirname
        ),
    }

    stage_environment = {
        "tools": basic_tools,
        "prompt_context": prompt_context,
        "structured_output": structured_output,
        "model_retries": model_retries,
    }
    run_semantic = semantic_stage(spec=SEMANTIC, **stage_environment)
    run_operations = operations_stage(spec=OPERATIONS, **stage_environment)
    agent_coder = coder(
        tools=basic_tools + submission_tools,
        prompt_context=prompt_context,
        model_retries=model_retries,
    )
    # The auditor judges what was actually scored, so it is given no
    # `verify_output` of its own: a build of its own would land in the attempt
    # directory as if it were an answer, and cost turns to reach a solid the
    # workflow already has.
    agent_auditor = auditor(
        tools=basic_tools,
        prompt_context=prompt_context,
        output_schema=Audit,
        structured_output=structured_output,
        model_retries=model_retries,
    )

    def initialize_input(state: ReconstructionState):
        """Open the transcript, once, with what the run offers the model."""
        del state
        return {"messages": [input_message]}

    def write_code(state: ReconstructionState):
        """Implement what the two stages settled, reading how they settled it.

        The coder is a stage with two upstreams and no critic, so it opens the
        same three ways: it implements, it corrects a fault the audit found in
        its program, or it rebuilds against a plan that has changed under it.
        An instruction ignores what it does not use, so the artifacts are
        passed whichever way it opens.
        """
        opening, context = _opening(state, "redo_code")
        run = run_agent(
            agent_coder,
            _through(state["messages"], "redo_code"),
            build_instruction(
                f"code/{opening}",
                hypothesis=state["semantic_hypothesis"].model_dump_json(indent=2),
                plan=state["operation_plan"].model_dump_json(indent=2),
                **context,
            ),
        )
        return {
            "messages": _produced_by(run.messages, "redo_code"),
            "agent_turns": run.turns_by_role,
            "stop_reasons": run.stop_reasons,
        }

    # Postprocess node
    # The final attempt is rendered like any other, so an evaluation pass does
    # not have to special-case the last verification directory.  It returns the
    # report itself, so it needs no MessageBuilder: nothing here reaches a model.
    verify_final_tool = create_verify_output_tool(
        executor=executor,
        workdir=sandbox_workdir,
        renderer=renderer,
        message_builder=None,
        source_filename=output_filename,
        output_dirname=verification_dirname,
        serialize_output=False,
    )

    def verify_final(state: ReconstructionState):
        del state
        return {"last_verification": verify_final_tool.invoke({})}

    def audit_output(state: ReconstructionState):
        """Judge the verified model, and say which stage owns any mismatch."""
        verification = state["last_verification"]
        attempts_dir = sandbox_workdir.sandbox_bind_dir / verification_dirname
        run = run_agent(
            agent_auditor,
            _through(state["messages"], "audit"),
            build_instruction(
                "audit",
                report=json.dumps(verification.serialize(), indent=2),
                attempt_dir=str(
                    attempts_dir / verification.verification_id
                    if verification.verification_id is not None
                    else attempts_dir
                ),
            ),
            Audit,
        )
        progress: dict[str, Any] = {
            "messages": _produced_by(run.messages, "audit"),
            "agent_turns": run.turns_by_role,
            "stop_reasons": run.stop_reasons,
            "audit": run.answer,
        }
        if run.answer is not None and run.answer.decision != "accept":
            progress["audit_reject_count"] = 1
        return progress

    # Construct a graph
    workflow = StateGraph(state_schema=ReconstructionState)  # type: ignore[type-var]
    workflow.add_node("initialize_input", initialize_input)
    workflow.add_node(
        "semantic_stage",
        _stage_node(run_semantic, "semantic_hypothesis", "redo_semantics"),
    )
    workflow.add_node(
        "operations_stage",
        _stage_node(
            run_operations,
            "operation_plan",
            "redo_operations",
            hypothesis="semantic_hypothesis",
        ),
    )
    workflow.add_node("coder", write_code)
    workflow.add_node("verify_final", verify_final)
    workflow.add_node("audit", audit_output)

    workflow.add_edge(START, "initialize_input")
    workflow.add_edge("initialize_input", "semantic_stage")
    workflow.add_conditional_edges(
        "semantic_stage",
        _settled("semantic_hypothesis", "operations_stage"),
        ["operations_stage", "verify_final"],
    )
    workflow.add_conditional_edges(
        "operations_stage",
        _settled("operation_plan", "coder"),
        ["coder", "verify_final"],
    )
    workflow.add_edge("coder", "verify_final")
    # Nothing to audit when the run never reached the coder: the verification
    # reports a missing program, which is not a judgement anyone can act on.
    workflow.add_conditional_edges(
        "verify_final",
        _settled("operation_plan", "audit", END),
        ["audit", END],
    )
    workflow.add_conditional_edges(
        "audit",
        _audited(max_audit_reject_count),
        [*_REDO_TARGETS.values(), END],
    )

    graph = workflow.compile(checkpointer=checkpointer)
    # Every send-back is another lap of the graph, and LangGraph's default
    # ceiling is lower than the laps this budget allows.  Counted from the graph
    # itself so that adding a node cannot leave the ceiling behind.
    return graph.with_config(
        recursion_limit=len(workflow.nodes) * (max_audit_reject_count + 2)
    )


if __name__ == "__main__":
    import io
    import sys
    from functools import partial
    from pathlib import Path
    from typing import cast
    from unittest.mock import Mock

    from langchain_core.runnables import RunnableLambda
    from PIL import Image

    from zeroshot.pipeline.workflow.agent import create_agent
    from zeroshot.pipeline.workflow.proposer_critic import create_proposer_critic_loop

    preview_model_mock = Mock(spec=BaseChatModel)
    preview_model_mock.bind_tools.return_value = RunnableLambda(
        lambda messages: AIMessage(content="")
    )
    preview_model = cast(BaseChatModel, preview_model_mock)

    def preview_agent(role: str) -> AgentFactory:
        return partial(create_agent, role=role, model=preview_model)

    def preview_stage(proposer: str, critic: str) -> StageFactory:
        return partial(
            create_proposer_critic_loop,
            proposer=preview_agent(proposer),
            critic=preview_agent(critic),
            max_revisions=10,
        )

    sandbox_runner = SandboxRunner(
        python_executable=Path(sys.executable),
        default_timeout_s=10,
    )

    message_builder = MessageBuilder(
        access_render3d="path",
        access_render3d_styles=("hlg_perspective",),
        feedback_render3d="path",
        feedback_render3d_styles=("hlg_perspective",),
    )

    with SandboxWorkdir() as workdir:
        graph = create_reconstruction_graph(
            semantic_stage=preview_stage("semantic_hypothesizer", "semantic_reviewer"),
            operations_stage=preview_stage("operation_planner", "operation_reviewer"),
            coder=preview_agent("coder"),
            auditor=preview_agent("output_auditor"),
            sandbox_runner=sandbox_runner,
            sandbox_workdir=workdir,
            renderer=StepRenderer(timeout_s=60),
            message_builder=message_builder,
            input_message=HumanMessage(content="preview"),
        )

        png_data = graph.get_graph().draw_mermaid_png()

    img = Image.open(io.BytesIO(png_data))
    img.show()
