from functools import partial
from pathlib import PurePosixPath
from typing import Any

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode

from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.tools import (
    create_load_image_tool,
    create_run_shell_tool,
    create_verify_output_tool,
)
from zeroshot.pipeline.verification import CadQueryExecutor
from zeroshot.pipeline.workflow.state import ReconstructionState


def create_reconstruction_graph(
    model: BaseChatModel,
    sandbox_runner: SandboxRunner,
    sandbox_workdir: SandboxWorkdir,
    output_filename: str = "model.py",
    verification_dirname: PurePosixPath = PurePosixPath("attempts"),
    checkpointer: BaseCheckpointSaver[Any] | None = None,
):
    def should_continue(state: ReconstructionState):
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            raise TypeError("agent node must append an AIMessage before routing")
        if last_message.tool_calls:
            return "tools"
        return "verify_final"

    def call_agent(
        state: ReconstructionState,
        agent: Runnable[LanguageModelInput, AIMessage],
    ):
        response = agent.invoke(state["messages"])
        return {"messages": [response]}

    # Create and bind tools
    executor = CadQueryExecutor(sandbox_runner=sandbox_runner)
    tools = [
        create_run_shell_tool(sandbox_runner, sandbox_workdir),
        create_load_image_tool(sandbox_workdir),
        create_verify_output_tool(
            executor=executor,
            workdir=sandbox_workdir,
            render_views=False,
            source_filename=output_filename,
            output_dirname=verification_dirname,
            serialize_output=True,
        ),
    ]
    agent = model.bind_tools(tools)
    tool_node = ToolNode(tools)

    # Postprocess node
    verify_final_tool = create_verify_output_tool(
        executor=executor,
        workdir=sandbox_workdir,
        render_views=False,
        source_filename=output_filename,
        output_dirname=verification_dirname,
        serialize_output=False,
    )

    def verify_final(state: ReconstructionState):
        report = verify_final_tool.invoke({})
        return {"last_verification": report}

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

    with SandboxWorkdir() as workdir:
        graph = create_reconstruction_graph(
            model=preview_model,
            sandbox_runner=sandbox_runner,
            sandbox_workdir=workdir,
        )

        png_data = graph.get_graph().draw_mermaid_png()

    img = Image.open(io.BytesIO(png_data))
    img.show()
