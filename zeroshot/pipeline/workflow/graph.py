from pathlib import PurePosixPath
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
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
from zeroshot.pipeline.workflow.agent import AgentFactory, run_agent
from zeroshot.pipeline.workflow.proposer_critic import (
    ProposerCriticSpec,
    StageFactory,
    StageRunner,
)
from zeroshot.pipeline.workflow.state import (
    OperationPlan,
    ReconstructionState,
    SemanticHypothesis,
)

# What each stage produces, and where the words asking for it live.  The pairing
# is code because it is not a choice: a plan reviewer reviews plans.
SEMANTIC = ProposerCriticSpec(proposal=SemanticHypothesis, instructions="semantic")
OPERATIONS = ProposerCriticSpec(proposal=OperationPlan, instructions="operations")


def _stage_node(run_stage: StageRunner, artifact: str, **upstream: str):
    """Wrap a stage as a graph node: run it, and say what it settled.

    A stage that produced nothing reports why the run is ending instead, since
    the workflow routes past everything downstream of it.
    """

    def node(state: ReconstructionState):
        settled = {
            name: state[key].model_dump_json(indent=2) for name, key in upstream.items()
        }
        result = run_stage(state["messages"], **settled)
        if result.artifact is not None:
            return {"messages": result.messages, artifact: result.artifact}
        return {
            "messages": result.messages,
            "agent_turns": result.turns,
            "stop_reason": result.stop_reason,
        }

    return node


def _settled(artifact: str, then: str):
    """Route on whether the stage before this one produced its artifact."""

    def route(state: ReconstructionState):
        return then if state.get(artifact) is not None else "verify_final"

    return route


def create_reconstruction_graph(
    semantic_stage: StageFactory,
    operations_stage: StageFactory,
    coder: AgentFactory,
    sandbox_runner: SandboxRunner,
    sandbox_workdir: SandboxWorkdir,
    renderer: StepRenderer,
    message_builder: MessageBuilder,
    input_message: HumanMessage,
    output_filename: str = "model.py",
    verification_dirname: PurePosixPath = PurePosixPath("attempts"),
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
        "model_retries": model_retries,
    }
    run_semantic = semantic_stage(spec=SEMANTIC, **stage_environment)
    run_operations = operations_stage(spec=OPERATIONS, **stage_environment)
    agent_coder = coder(
        tools=basic_tools + submission_tools,
        prompt_context=prompt_context,
        model_retries=model_retries,
    )

    def initialize_input(state: ReconstructionState):
        """Open the transcript, once, with what the run offers the model."""
        del state
        return {"messages": [input_message]}

    def write_code(state: ReconstructionState):
        """Implement what the two stages settled, reading how they settled it."""
        run = run_agent(
            agent_coder,
            state["messages"],
            build_instruction(
                "implement",
                hypothesis=state["semantic_hypothesis"].model_dump_json(indent=2),
                plan=state["operation_plan"].model_dump_json(indent=2),
            ),
        )
        return {
            "messages": run.messages,
            "agent_turns": run.turns,
            "stop_reason": run.stop_reason,
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

    # Construct a graph
    workflow = StateGraph(state_schema=ReconstructionState)  # type: ignore[type-var]
    workflow.add_node("initialize_input", initialize_input)
    workflow.add_node(
        "semantic_stage", _stage_node(run_semantic, "semantic_hypothesis")
    )
    workflow.add_node(
        "operations_stage",
        _stage_node(run_operations, "operation_plan", hypothesis="semantic_hypothesis"),
    )
    workflow.add_node("coder", write_code)
    workflow.add_node("verify_final", verify_final)

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
    workflow.add_edge("verify_final", END)

    graph = workflow.compile(checkpointer=checkpointer)
    return graph


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
            structured_output="provider",
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
            sandbox_runner=sandbox_runner,
            sandbox_workdir=workdir,
            renderer=StepRenderer(timeout_s=60),
            message_builder=message_builder,
            input_message=HumanMessage(content="preview"),
        )

        png_data = graph.get_graph().draw_mermaid_png()

    img = Image.open(io.BytesIO(png_data))
    img.show()
