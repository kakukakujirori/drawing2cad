import base64
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import pytest
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool, tool
from PIL import Image

from zeroshot.pipeline.messages import MessageBuilder
from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.tools import ToolFeedbackError, VerifyOutputResult
from zeroshot.pipeline.verification import StepRenderer
from zeroshot.pipeline.workflow import StopReason
from zeroshot.pipeline.workflow import graph as graph_module
from zeroshot.pipeline.workflow.graph import create_reconstruction_graph


def _renderer() -> StepRenderer:
    """A real renderer: these graph tests never reach a verified STEP, so it is
    only here to satisfy the wiring, not to render anything."""
    return StepRenderer(timeout_s=60.0)


def _message_builder() -> MessageBuilder:
    return MessageBuilder(
        access_render3d="none",
        access_render3d_styles=(),
        feedback_render3d="none",
        feedback_render3d_styles=(),
    )


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
            renderer=_renderer(),
            message_builder=_message_builder(),
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
            renderer=_renderer(),
            message_builder=_message_builder(),
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


def test_graph_returns_image_results_to_agent(tmp_path: Path) -> None:
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
                        "args": {"image_path": "/work/missing.png"},
                        "id": "call-load-missing-image",
                        "type": "tool_call",
                    }
                ],
            ),
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
            renderer=_renderer(),
            message_builder=_message_builder(),
        )
        result = graph.invoke(
            {"messages": [HumanMessage(content="Inspect /work/view.png")]}
        )

    assert len(agent_inputs) == 3
    error_result = agent_inputs[1][-1]
    assert isinstance(error_result, ToolMessage)
    assert error_result.tool_call_id == "call-load-missing-image"
    assert error_result.status == "error"
    assert "Cannot access image: /work/missing.png" in error_result.text

    image_result = agent_inputs[2][-1]
    assert isinstance(image_result, ToolMessage)
    assert image_result.tool_call_id == "call-load-image"
    assert isinstance(image_result.content, list)
    assert len(image_result.content) == 1
    image_block = image_result.content[0]
    assert image_block["type"] == "image"
    assert image_block["mime_type"] == "image/png"
    assert base64.b64decode(image_block["base64"]) == expected_image
    assert result["last_verification"].status == "REJECTED"


def _relentless_agent() -> BaseChatModel:
    """An agent that always asks for another tool call and never finishes."""

    turns = {"n": 0}

    def call(_: LanguageModelInput) -> AIMessage:
        turns["n"] += 1
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "run_shell",
                    "args": {"command": "true"},
                    "id": f"call-{turns['n']}",
                    "type": "tool_call",
                }
            ],
        )

    model_mock = Mock(spec=BaseChatModel)
    model_mock.bind_tools.return_value = RunnableLambda(call)
    return cast(BaseChatModel, model_mock)


def _graph_with_budget(workdir: SandboxWorkdir, model: BaseChatModel, budget: int):
    return create_reconstruction_graph(
        model=model,
        sandbox_runner=SandboxRunner(
            python_executable=Path(sys.executable), default_timeout_s=10
        ),
        sandbox_workdir=workdir,
        renderer=_renderer(),
        message_builder=_message_builder(),
        max_agent_turns=budget,
    )


def test_an_exhausted_budget_still_submits() -> None:
    """LangGraph's own ceiling raises, which loses the run's final verification.

    The budget has to route to `verify_final` instead, so a model that never
    declares itself finished still produces a submission.
    """

    with SandboxWorkdir() as workdir:
        graph = _graph_with_budget(workdir, _relentless_agent(), budget=3)
        result = graph.invoke({"messages": [HumanMessage(content="go")]})

    assert result["agent_turns"] == 3
    assert result["stop_reason"] is StopReason.BUDGET_EXHAUSTED
    assert result["last_verification"] is not None


def test_a_finished_agent_is_not_recorded_as_cut_off() -> None:
    responses: Iterator[AIMessage] = iter([AIMessage(content="done")])
    model_mock = Mock(spec=BaseChatModel)
    model_mock.bind_tools.return_value = RunnableLambda(lambda _: next(responses))

    with SandboxWorkdir() as workdir:
        graph = _graph_with_budget(workdir, cast(BaseChatModel, model_mock), budget=3)
        result = graph.invoke({"messages": [HumanMessage(content="go")]})

    assert result["agent_turns"] == 1
    assert result["stop_reason"] is StopReason.COMPLETED


def test_the_budget_is_spent_on_tool_calls_not_super_steps() -> None:
    """The unit is agent turns, so adding a graph node cannot change it."""

    with SandboxWorkdir() as workdir:
        graph = _graph_with_budget(workdir, _relentless_agent(), budget=7)
        result = graph.invoke({"messages": [HumanMessage(content="go")]})

    assert result["agent_turns"] == 7


def test_a_budget_below_one_is_refused() -> None:
    with SandboxWorkdir() as workdir, pytest.raises(ValueError):
        _graph_with_budget(workdir, _relentless_agent(), budget=0)


def _recording_graph(workdir: SandboxWorkdir, budget: int = 3, **options: Any):
    """A graph over a relentless agent, plus the message lists it was called
    with. `options` is left unset so a test can pin a default."""

    seen: list[list[BaseMessage]] = []
    turns = {"n": 0}

    def call(model_input: LanguageModelInput) -> AIMessage:
        assert isinstance(model_input, list)
        seen.append(cast(list[BaseMessage], model_input))
        turns["n"] += 1
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "verify_output",
                    "args": {},
                    "id": f"call-{turns['n']}",
                    "type": "tool_call",
                }
            ],
        )

    model_mock = Mock(spec=BaseChatModel)
    model_mock.bind_tools.return_value = RunnableLambda(call)
    graph = create_reconstruction_graph(
        model=cast(BaseChatModel, model_mock),
        sandbox_runner=SandboxRunner(
            python_executable=Path(sys.executable), default_timeout_s=10
        ),
        sandbox_workdir=workdir,
        renderer=_renderer(),
        message_builder=_message_builder(),
        max_agent_turns=budget,
        **options,
    )
    return graph, seen


def test_the_agent_is_told_where_it_is_in_its_budget() -> None:
    """It cannot infer the turn count from the transcript on its own."""

    with SandboxWorkdir() as workdir:
        graph, seen = _recording_graph(workdir, announce_turn_budget=True)
        graph.invoke({"messages": [HumanMessage(content="go")]})

    assert [call[-1].text for call in seen] == [
        f"[turn {turn}/3; candidates submitted: {turn - 1}]" for turn in (1, 2, 3)
    ]


def test_every_budget_notice_stays_in_the_transcript() -> None:
    """Dropping the earlier ones would leave the agent its position but not its
    rate: at `turn 3/3` it is the `turn 1/3` above that says how fast it got
    there."""

    with SandboxWorkdir() as workdir:
        graph, seen = _recording_graph(workdir, announce_turn_budget=True)
        result = graph.invoke({"messages": [HumanMessage(content="go")]})

    assert [
        sum(m.text.startswith("[turn ") for m in call if isinstance(m, HumanMessage))
        for call in seen
    ] == [1, 2, 3]
    assert [
        message.text
        for message in result["messages"]
        if isinstance(message, HumanMessage) and message.text.startswith("[turn ")
    ] == [f"[turn {turn}/3; candidates submitted: {turn - 1}]" for turn in (1, 2, 3)]


def test_the_budget_stays_hidden_unless_asked_for() -> None:
    """The recorded baselines ran without it, so it cannot be on by default.

    Nothing here passes `announce_turn_budget`, which is what pins the default.
    """

    with SandboxWorkdir() as workdir:
        graph, seen = _recording_graph(workdir)
        graph.invoke({"messages": [HumanMessage(content="go")]})

    assert not [m for call in seen for m in call if m.text.startswith("[turn ")]


def _agent_calling(tool_name: str, args: dict[str, object]) -> BaseChatModel:
    def call(_: LanguageModelInput) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": tool_name,
                    "args": args,
                    "id": "call-once",
                    "type": "tool_call",
                }
            ],
        )

    model_mock = Mock(spec=BaseChatModel)
    model_mock.bind_tools.return_value = RunnableLambda(call)
    return cast(BaseChatModel, model_mock)


def _graph_with_a_broken_load_image(
    workdir: SandboxWorkdir, error: Exception, monkeypatch, max_agent_turns: int = 30
):
    @tool("load_image")
    def broken_load_image(image_path: str) -> str:
        """Load an image from a file path."""
        del image_path
        raise error

    monkeypatch.setattr(
        graph_module, "create_load_image_tool", lambda _workdir: broken_load_image
    )
    return create_reconstruction_graph(
        model=_agent_calling("load_image", {"image_path": "/work/view.png"}),
        sandbox_runner=SandboxRunner(
            python_executable=Path(sys.executable), default_timeout_s=10
        ),
        sandbox_workdir=workdir,
        renderer=_renderer(),
        message_builder=_message_builder(),
        max_agent_turns=max_agent_turns,
    )


def test_a_defect_in_a_tool_ends_the_run(monkeypatch) -> None:
    """A stray ValueError is a bug here, not advice for the model.

    Routing on it would have the model politely work around our own defects
    while the run reports success, so only ToolFeedbackError is forwarded.
    """
    with SandboxWorkdir() as workdir:
        graph = _graph_with_a_broken_load_image(
            workdir, ValueError("internal invariant broken"), monkeypatch
        )
        with pytest.raises(ValueError, match="internal invariant broken"):
            graph.invoke({"messages": [HumanMessage(content="go")]})


def test_an_error_raised_for_the_model_is_forwarded_as_feedback(monkeypatch) -> None:
    with SandboxWorkdir() as workdir:
        graph = _graph_with_a_broken_load_image(
            workdir,
            ToolFeedbackError("that is not an image"),
            monkeypatch,
            max_agent_turns=2,
        )
        result = graph.invoke({"messages": [HumanMessage(content="go")]})

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_messages
    assert tool_messages[0].status == "error"
    assert "that is not an image" in tool_messages[0].text
