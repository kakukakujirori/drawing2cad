import sys
from functools import partial
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool

from tests.zeroshot.chat_models import ScriptedChatModel, tool_call
from zeroshot.pipeline.messages import MessageBuilder
from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.tools import VerifyOutputResult
from zeroshot.pipeline.verification import StepRenderer
from zeroshot.pipeline.workflow import StopReason, create_agent
from zeroshot.pipeline.workflow import graph as graph_module
from zeroshot.pipeline.workflow.graph import AgentFactory, create_reconstruction_graph


def _renderer() -> StepRenderer:
    return StepRenderer(timeout_s=60.0)


def _message_builder() -> MessageBuilder:
    return MessageBuilder(
        access_render3d="none",
        access_render3d_styles=(),
        feedback_render3d="none",
        feedback_render3d_styles=(),
    )


def _unused_model() -> ScriptedChatModel:
    """A stage this graph does not run yet still has to be built."""
    return ScriptedChatModel(responses=())


def _agent(role: str, model: BaseChatModel, **overrides: Any) -> AgentFactory:
    return partial(create_agent, role=role, model=model, **overrides)


def _graph(
    workdir: SandboxWorkdir,
    coder: AgentFactory,
    input_message: HumanMessage,
    **overrides: Any,
):
    return create_reconstruction_graph(
        semantic_hypothesizer=_agent("semantic_hypothesizer", _unused_model()),
        semantic_reviewer=_agent("semantic_reviewer", _unused_model()),
        coder=coder,
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
    model = ScriptedChatModel(responses=(AIMessage(content="done"),))
    input_message = HumanMessage(content="Create the model")

    with SandboxWorkdir() as workdir:
        graph = _graph(
            workdir,
            _agent("coder", model, announce_turn_budget=False),
            input_message,
        )
        result = graph.invoke({"messages": []})

    seen = model.received_messages
    assert len(seen) == 1
    assert [type(message) for message in seen[0]] == [SystemMessage, HumanMessage]
    assert seen[0][1].text == input_message.text
    assert model.bound_tool_names == ("run_shell", "load_image", "verify_output")
    assert result["agent_turns"] == 1
    assert result["stop_reason"] is StopReason.COMPLETED
    assert result["last_verification"] == VerifyOutputResult(
        status="REJECTED",
        executor_error="model.py was not found",
    )


def test_graph_repeats_verification_after_the_coder_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = ScriptedChatModel(
        responses=(
            tool_call("verify_output", {}, "call-intermediate-verification"),
            AIMessage(content="done"),
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
            _agent("coder", model, announce_turn_budget=False),
            HumanMessage(content="Verify the candidate"),
        )
        result = graph.invoke({"messages": []})

    assert invocations == ["model", "workflow"]
    assert result["last_verification"] == final_report


def test_graph_runs_final_verification_after_agent_budget_exhaustion() -> None:
    model = ScriptedChatModel(
        responses=tuple(
            tool_call("run_shell", {"command": "true"}, f"call-{turn}")
            for turn in range(1, 5)
        )
    )

    with SandboxWorkdir() as workdir:
        graph = _graph(
            workdir,
            _agent("coder", model, max_turns=3, announce_turn_budget=False),
            HumanMessage(content="go"),
        )
        result = graph.invoke({"messages": []})

    assert result["agent_turns"] == 3
    assert result["stop_reason"] is StopReason.BUDGET_EXHAUSTED
    assert result["last_verification"] == VerifyOutputResult(
        status="REJECTED",
        executor_error="model.py was not found",
    )
