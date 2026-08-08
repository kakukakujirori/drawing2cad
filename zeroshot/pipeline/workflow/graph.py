from collections.abc import Callable
from pathlib import PurePosixPath
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from zeroshot.pipeline.messages import MessageBuilder
from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.tools import (
    create_load_image_tool,
    create_run_shell_tool,
    create_verify_output_tool,
)
from zeroshot.pipeline.verification import CadQueryExecutor, StepRenderer
from zeroshot.pipeline.workflow.state import ReconstructionState

# A config binds who an agent is -- role, model, budget, output contract -- and
# leaves the environment open.  The graph supplies the rest, because which tools
# a stage gets is a topology decision.
AgentFactory = Callable[..., Any]


def create_reconstruction_graph(
    semantic_hypothesizer: AgentFactory,
    semantic_reviewer: AgentFactory,
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
    """One agent writes a CadQuery program, then the workflow verifies it."""
    # Create and bind tools
    executor = CadQueryExecutor(sandbox_runner=sandbox_runner)
    basic_tools = [
        create_run_shell_tool(sandbox_runner, sandbox_workdir),
        create_load_image_tool(sandbox_workdir),
    ]
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

    # The semantic stage builds on the same tools; it is wired in the next step.
    del semantic_hypothesizer, semantic_reviewer

    prompt_context = {
        "output_path": str(sandbox_workdir.sandbox_bind_dir / output_filename),
        "verification_dir": str(
            sandbox_workdir.sandbox_bind_dir / verification_dirname
        ),
    }

    agent_coder = coder(
        tools=basic_tools + submission_tools,
        prompt_context=prompt_context,
        model_retries=model_retries,
    )

    def agent_loop(state: ReconstructionState):
        """Give the coder its task and adopt what it reports.

        Only the mapping between two state schemas lives here.  The transcript
        deliberately stays in the subgraph: copying it up would record every
        turn twice -- once as the agent produced it, once as this node's update
        -- and an offline reader replaying the log would see the conversation
        doubled.
        """
        del state
        result = agent_coder.invoke({"messages": [input_message]})
        return {
            "agent_turns": result["turns"],
            "stop_reason": result["stop_reason"],
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
    workflow.add_node("coder", agent_loop)
    workflow.add_node("verify_final", verify_final)

    workflow.add_edge(START, "coder")
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

    preview_model_mock = Mock(spec=BaseChatModel)
    preview_model_mock.bind_tools.return_value = RunnableLambda(
        lambda messages: AIMessage(content="")
    )
    preview_model = cast(BaseChatModel, preview_model_mock)

    def preview_agent(role: str) -> AgentFactory:
        return partial(create_agent, role=role, model=preview_model)

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
            semantic_hypothesizer=preview_agent("semantic_hypothesizer"),
            semantic_reviewer=preview_agent("semantic_reviewer"),
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
