import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import pytest
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool, tool

from zeroshot.pipeline.messages import MessageBuilder
from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.tools import VerifyOutputResult
from zeroshot.pipeline.verification import StepRenderer
from zeroshot.pipeline.workflow import AgentSpec, StopReason
from zeroshot.pipeline.workflow import graph as graph_module
from zeroshot.pipeline.workflow.graph import create_reconstruction_graph


def _renderer() -> StepRenderer:
    return StepRenderer(timeout_s=60.0)


def _message_builder() -> MessageBuilder:
    return MessageBuilder(
        access_render3d="none",
        access_render3d_styles=(),
        feedback_render3d="none",
        feedback_render3d_styles=(),
    )


def _agent(model: BaseChatModel, **overrides: Any) -> AgentSpec:
    return AgentSpec(role="coder", model=model, **overrides)


def _model(
    responses: Iterator[AIMessage],
    seen: list[list[BaseMessage]] | None = None,
) -> tuple[BaseChatModel, Mock]:
    def call(model_input: LanguageModelInput) -> AIMessage:
        if seen is not None:
            assert isinstance(model_input, list)
            seen.append(cast(list[BaseMessage], model_input))
        return next(responses)

    model_mock = Mock(spec=BaseChatModel)
    model_mock.bind_tools.return_value = RunnableLambda(call)
    return cast(BaseChatModel, model_mock), model_mock


def _graph(
    workdir: SandboxWorkdir,
    agent: AgentSpec,
    input_message: HumanMessage,
    **overrides: Any,
):
    return create_reconstruction_graph(
        agent=agent,
        sandbox_runner=SandboxRunner(
            python_executable=Path(sys.executable), default_timeout_s=10
        ),
        sandbox_workdir=workdir,
        renderer=_renderer(),
        message_builder=_message_builder(),
        input_message=input_message,
        **overrides,
    )


def test_graph_wires_the_coder_and_adopts_its_result() -> None:
    seen: list[list[BaseMessage]] = []
    model, model_mock = _model(iter([AIMessage(content="done")]), seen)
    input_message = HumanMessage(content="Create the model")

    with SandboxWorkdir() as workdir:
        graph = _graph(
            workdir,
            _agent(model, announce_turn_budget=False),
            input_message,
        )
        result = graph.invoke({"messages": []})

    assert len(seen) == 1
    assert [type(message) for message in seen[0]] == [SystemMessage, HumanMessage]
    assert seen[0][1] is input_message
    model_mock.bind_tools.assert_called_once()
    assert [tool.name for tool in model_mock.bind_tools.call_args.args[0]] == [
        "run_shell",
        "load_image",
        "verify_output",
    ]
    assert result["agent_turns"] == 1
    assert result["stop_reason"] is StopReason.COMPLETED
    assert result["last_verification"] == VerifyOutputResult(
        status="REJECTED",
        executor_error="model.py was not found",
    )


def test_graph_repeats_verification_after_the_coder_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _ = _model(
        iter(
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
    )
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
        graph = _graph(
            workdir,
            _agent(model, announce_turn_budget=False),
            HumanMessage(content="Verify the candidate"),
        )
        result = graph.invoke({"messages": []})

    assert invocations == ["model", "workflow"]
    assert result["last_verification"] == final_report


def test_graph_runs_final_verification_after_agent_budget_exhaustion() -> None:
    turns = {"count": 0}

    def call(_: LanguageModelInput) -> AIMessage:
        turns["count"] += 1
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "run_shell",
                    "args": {"command": "true"},
                    "id": f"call-{turns['count']}",
                    "type": "tool_call",
                }
            ],
        )

    model_mock = Mock(spec=BaseChatModel)
    model_mock.bind_tools.return_value = RunnableLambda(call)
    model = cast(BaseChatModel, model_mock)

    with SandboxWorkdir() as workdir:
        graph = _graph(
            workdir,
            _agent(model, max_turns=3, announce_turn_budget=False),
            HumanMessage(content="go"),
        )
        result = graph.invoke({"messages": []})

    assert result["agent_turns"] == 3
    assert result["stop_reason"] is StopReason.BUDGET_EXHAUSTED
    assert result["last_verification"] == VerifyOutputResult(
        status="REJECTED",
        executor_error="model.py was not found",
    )
