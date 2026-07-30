from collections.abc import Sequence

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode

from zeroshot.pipeline.workflow.state import ReconstructionState


def create_reconstruction_graph(
    agent_with_tools: Runnable[LanguageModelInput, AIMessage],
    tools: Sequence[BaseTool],
):
    def should_continue(state: ReconstructionState):
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            raise TypeError("agent node must append an AIMessage before routing")
        if last_message.tool_calls:
            return "tools"
        return END

    def call_agent(state: ReconstructionState):
        response = agent_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    workflow = StateGraph(state_schema=ReconstructionState)  # type: ignore[type-var]
    workflow.add_node("agent", call_agent)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    workflow.add_edge("tools", "agent")

    graph = workflow.compile()
    return graph


if __name__ == "__main__":
    import io
    from langchain_core.runnables import RunnableLambda
    from langchain_core.tools import tool
    from PIL import Image

    @tool
    def preview_tool(command: str) -> str:
        """Preview-only tool used to render the graph structure."""
        return command

    preview_agent = RunnableLambda(lambda messages: AIMessage(content=""))

    graph = create_reconstruction_graph(
        agent_with_tools=preview_agent,
        tools=[preview_tool],
    )

    png_data = graph.get_graph().draw_mermaid_png()

    # Save to file
    out_path = "workflow_graph.png"
    # with open(out_path, "wb") as f:
    #     f.write(png_data)
    # print(f"Graph saved to {out_path}")

    # Open in a separate window
    img = Image.open(io.BytesIO(png_data))
    img.show()
