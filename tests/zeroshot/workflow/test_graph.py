import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import cast

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableLambda

from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.tools.run_shell import create_run_shell_tool
from zeroshot.pipeline.workflow.graph import create_reconstruction_graph


def test_graph_runs_tools_until_agent_returns_without_tool_calls() -> None:
    sandbox_runner = SandboxRunner(
        python_executable=Path(sys.executable),
        default_timeout_s=10,
    )
    responses: Iterator[AIMessage] = iter(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "run_shell",
                        "args": {"command": "printf 41 > value.txt"},
                        "id": "call-write",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "run_shell",
                        "args": {"command": "cat value.txt"},
                        "id": "call-read",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    agent_inputs: list[list[BaseMessage]] = []

    def scripted_agent(model_input: LanguageModelInput) -> AIMessage:
        assert isinstance(model_input, list)
        assert all(isinstance(message, BaseMessage) for message in model_input)
        agent_inputs.append(cast(list[BaseMessage], model_input))
        return next(responses)

    agent_with_tools: Runnable[LanguageModelInput, AIMessage] = RunnableLambda(
        scripted_agent
    )

    with SandboxWorkdir() as workdir:
        run_shell = create_run_shell_tool(sandbox_runner, workdir)
        graph = create_reconstruction_graph(
            agent_with_tools=agent_with_tools,
            tools=[run_shell],
        )

        result = graph.invoke(
            {"messages": [HumanMessage(content="Create and inspect value.txt")]}
        )

        assert (workdir.host_bind_dir / "value.txt").read_text(encoding="utf-8") == "41"

    messages = result["messages"]
    assert [type(message) for message in messages] == [
        HumanMessage,
        AIMessage,
        ToolMessage,
        AIMessage,
        ToolMessage,
        AIMessage,
    ]

    write_message = messages[2]
    assert isinstance(write_message, ToolMessage)
    assert write_message.tool_call_id == "call-write"
    assert isinstance(write_message.content, str)
    assert json.loads(write_message.content) == {
        "status": "COMPLETED",
        "returncode": 0,
        "stdout": "",
        "stderr": "",
    }

    read_message = messages[4]
    assert isinstance(read_message, ToolMessage)
    assert read_message.tool_call_id == "call-read"
    assert isinstance(read_message.content, str)
    assert json.loads(read_message.content) == {
        "status": "COMPLETED",
        "returncode": 0,
        "stdout": "41",
        "stderr": "",
    }

    assert len(agent_inputs) == 3
    assert len(agent_inputs[0]) == 1
    assert agent_inputs[1][-1] is write_message
    assert agent_inputs[2][-1] is read_message
    assert isinstance(messages[-1], AIMessage)
    assert messages[-1].content == "done"
    assert messages[-1].tool_calls == []
