from functools import partial
from pathlib import PurePosixPath
from typing import Any

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from zeroshot.pipeline.messages import MessageBuilder
from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.tools import (
    create_load_image_tool,
    create_run_shell_tool,
    create_verify_output_tool,
)
from zeroshot.pipeline.verification import CadQueryExecutor, StepRenderer
from zeroshot.pipeline.workflow.state import ReconstructionState, StopReason


def create_reconstruction_graph(
    model: BaseChatModel,
    sandbox_runner: SandboxRunner,
    sandbox_workdir: SandboxWorkdir,
    renderer: StepRenderer,
    message_builder: MessageBuilder,
    output_filename: str = "model.py",
    verification_dirname: PurePosixPath = PurePosixPath("attempts"),
    max_agent_turns: int = 30,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
):
    if max_agent_turns < 1:
        raise ValueError("max_agent_turns must be at least 1")

    def should_continue(state: ReconstructionState):
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            raise TypeError("agent node must append an AIMessage before routing")
        if last_message.tool_calls and state["agent_turns"] < max_agent_turns:
            return "tools"
        return "verify_final"

    def call_agent(
        state: ReconstructionState,
        agent: Runnable[LanguageModelInput, AIMessage],
    ):
        response = agent.invoke(state["messages"])
        return {
            "messages": [response],
            "agent_turns": state.get("agent_turns", 0) + 1,
        }

    # Create and bind tools
    executor = CadQueryExecutor(sandbox_runner=sandbox_runner)
    tools = [
        create_run_shell_tool(sandbox_runner, sandbox_workdir),
        create_load_image_tool(sandbox_workdir),
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
    agent = model.bind_tools(tools)
    tool_node = ToolNode(tools, handle_tool_errors=ValueError)

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
        last_message = state["messages"][-1]
        stop_reason = (
            StopReason.BUDGET_EXHAUSTED
            if isinstance(last_message, AIMessage) and last_message.tool_calls
            else StopReason.COMPLETED
        )
        report = verify_final_tool.invoke({})
        return {"last_verification": report, "stop_reason": stop_reason}

    # Construct a graph
    workflow = StateGraph(state_schema=ReconstructionState)  # type: ignore[type-var]
    workflow.add_node("agent", partial(call_agent, agent=agent))
    workflow.add_node("tools", tool_node)
    workflow.add_node("verify_final", verify_final)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, ["tools", "verify_final"])
    workflow.add_edge("tools", "agent")
    workflow.add_edge("verify_final", END)

    graph = workflow.compile(checkpointer=checkpointer)
    return graph


if __name__ == "__main__":
    import io
    import sys
    from pathlib import Path
    from typing import cast
    from unittest.mock import Mock

    from langchain_core.runnables import RunnableLambda
    from PIL import Image

    preview_model_mock = Mock(spec=BaseChatModel)
    preview_model_mock.bind_tools.return_value = RunnableLambda(
        lambda messages: AIMessage(content="")
    )
    preview_model = cast(BaseChatModel, preview_model_mock)

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
            model=preview_model,
            sandbox_runner=sandbox_runner,
            sandbox_workdir=workdir,
            renderer=StepRenderer(timeout_s=60),
            message_builder=message_builder,
        )

        png_data = graph.get_graph().draw_mermaid_png()

    img = Image.open(io.BytesIO(png_data))
    img.show()
