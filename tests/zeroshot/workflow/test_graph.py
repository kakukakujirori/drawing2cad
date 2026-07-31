import base64
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool, tool
from PIL import Image

from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.tools import VerifyOutputResult
from zeroshot.pipeline.workflow import graph as graph_module
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

    model_mock = Mock(spec=BaseChatModel)
    model_mock.bind_tools.return_value = RunnableLambda(scripted_agent)
    model = cast(BaseChatModel, model_mock)

    with SandboxWorkdir() as workdir:
        graph = create_reconstruction_graph(
            model=model,
            sandbox_runner=sandbox_runner,
            sandbox_workdir=workdir,
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

    model_mock.bind_tools.assert_called_once()
    bound_tools = model_mock.bind_tools.call_args.args[0]
    assert [bound_tool.name for bound_tool in bound_tools] == [
        "run_shell",
        "load_image",
        "verify_output",
    ]

    assert result["last_verification"] == VerifyOutputResult(
        status="REJECTED",
        executor_error="model.py was not found",
    )


def test_graph_repeats_verification_after_model_calls_verify_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
                        "name": "verify_output",
                        "args": {},
                        "id": "call-intermediate-verification",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    model_mock = Mock(spec=BaseChatModel)
    model_mock.bind_tools.return_value = RunnableLambda(lambda _: next(responses))
    model = cast(BaseChatModel, model_mock)

    invocations: list[str] = []
    final_report = VerifyOutputResult(
        verification_id="001",
        status="VERIFIED",
        source="result = object()",
        returncode=0,
    )

    @tool("verify_output")
    def model_verify_output() -> dict[str, str]:
        """Stub the model-facing intermediate verification."""
        invocations.append("model")
        return {"status": "REJECTED"}

    @tool("verify_output")
    def final_verify_output() -> VerifyOutputResult:
        """Stub the workflow-owned final verification."""
        invocations.append("workflow")
        return final_report

    def create_stub_verify_output_tool(
        *args: object,
        serialize_output: bool = True,
        **kwargs: object,
    ) -> BaseTool:
        del args, kwargs
        return model_verify_output if serialize_output else final_verify_output

    monkeypatch.setattr(
        graph_module,
        "create_verify_output_tool",
        create_stub_verify_output_tool,
    )

    with SandboxWorkdir() as workdir:
        graph = create_reconstruction_graph(
            model=model,
            sandbox_runner=sandbox_runner,
            sandbox_workdir=workdir,
        )
        result = graph.invoke(
            {"messages": [HumanMessage(content="Verify the candidate")]}
        )

    assert invocations == ["model", "workflow"]
    assert result["last_verification"] == final_report

    messages = result["messages"]
    assert [type(message) for message in messages] == [
        HumanMessage,
        AIMessage,
        ToolMessage,
        AIMessage,
    ]

    intermediate_result = messages[2]
    assert isinstance(intermediate_result, ToolMessage)
    assert intermediate_result.tool_call_id == "call-intermediate-verification"
    assert json.loads(cast(str, intermediate_result.content)) == {"status": "REJECTED"}


def test_graph_returns_loaded_image_to_agent(tmp_path: Path) -> None:
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
                        "name": "load_image",
                        "args": {"image_path": "/work/view.png"},
                        "id": "call-load-image",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="image inspected"),
        ]
    )
    agent_inputs: list[list[BaseMessage]] = []

    def scripted_agent(model_input: LanguageModelInput) -> AIMessage:
        assert isinstance(model_input, list)
        agent_inputs.append(cast(list[BaseMessage], model_input))
        return next(responses)

    model_mock = Mock(spec=BaseChatModel)
    model_mock.bind_tools.return_value = RunnableLambda(scripted_agent)
    model = cast(BaseChatModel, model_mock)

    with SandboxWorkdir(host_bind_dir=tmp_path) as workdir:
        image_path = workdir.host_bind_dir / "view.png"
        Image.new("RGB", (3, 2), color=(10, 20, 30)).save(image_path)
        expected_image = image_path.read_bytes()

        graph = create_reconstruction_graph(
            model=model,
            sandbox_runner=sandbox_runner,
            sandbox_workdir=workdir,
        )
        result = graph.invoke(
            {"messages": [HumanMessage(content="Inspect /work/view.png")]}
        )

    assert len(agent_inputs) == 2
    image_result = agent_inputs[1][-1]
    assert isinstance(image_result, ToolMessage)
    assert image_result.tool_call_id == "call-load-image"
    assert isinstance(image_result.content, list)
    assert len(image_result.content) == 1
    image_block = image_result.content[0]
    assert image_block["type"] == "image"
    assert image_block["mime_type"] == "image/png"
    assert base64.b64decode(image_block["base64"]) == expected_image
    assert result["last_verification"].status == "REJECTED"
