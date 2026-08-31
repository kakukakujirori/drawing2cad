import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field

from tests.zeroshot.chat_models import ScriptedChatModel, tool_call
from zeroshot.pipeline.workflow import StopReason
from zeroshot.pipeline.workflow.components.fanout_reduce import (
    create_fanout_reduce_graph,
)


class _Proposal(BaseModel):
    """A contract of the test's own, so these tests exercise the template
    rather than whichever schema the semantics stage happens to use."""

    model_config = ConfigDict(extra="forbid")
    proposal: list[str] = Field(..., description="List of proposals.")
    rationale: str = Field(..., description="Rationale for the proposal.")


def _answer(*items: str) -> AIMessage:
    return AIMessage(
        content=json.dumps({"proposal": list(items), "rationale": "the views agree"})
    )


def _proposal(*items: str) -> _Proposal:
    return _Proposal(
        proposal=list(items),
        rationale="the views agree",
    )


def _graph(
    models: list[ScriptedChatModel],
    commands: list[str],
    *,
    max_turns: int = 3,
):
    @tool("run_shell")
    def run_shell(command: str) -> dict[str, Any]:
        """Record a test shell command as successful."""
        commands.append(command)
        return {"returncode": 0, "stdout": "", "stderr": ""}

    return create_fanout_reduce_graph(
        proposer_role="semantic_hypothesizer",
        proposer_models=models,
        tools=[run_shell],
        fanout_workdir_prefix="/work/semantics",
        proposal_schema=_Proposal,
        response_format_strategy="provider",
        max_proposer_turns=max_turns,
        max_reducer_turns=max_turns,
        announce_turns=False,
        checkpointer=False,
    )


def test_initial_call_fans_out_and_reduces() -> None:
    commands: list[str] = []
    head = ScriptedChatModel(
        responses=(
            _answer("plate"),
            _answer("plate", "through hole"),
        )
    )
    peer = ScriptedChatModel(responses=(_answer("through hole"),))

    result = _graph([head, peer], commands).invoke(
        {"invocation_instruction": HumanMessage(content="analyze the drawing")}
    )

    assert result["proposal"] == _proposal("plate", "through hole")
    assert result["invocation_instruction"] is None
    assert commands == ["mkdir -p -- /work/semantics_0 /work/semantics_1"]
    assert len(head.received_messages) == 2
    assert len(peer.received_messages) == 1
    reduction_instruction = head.received_messages[1][-1].text
    assert "through hole" in reduction_instruction
    assert _proposal("through hole").model_dump_json() in reduction_instruction
    assert (
        _proposal("through hole").model_dump_json(indent=2) not in reduction_instruction
    )


def test_reentry_revises_the_adopted_proposal_without_fanning_out_again() -> None:
    commands: list[str] = []
    head = ScriptedChatModel(
        responses=(
            _answer("plate"),
            _answer("plate", "through hole"),
            _answer("bracket", "through hole"),
        )
    )
    peer = ScriptedChatModel(responses=(_answer("through hole"),))
    graph = _graph([head, peer], commands)
    first = graph.invoke(
        {"invocation_instruction": HumanMessage(content="analyze the drawing")}
    )

    second = graph.invoke(
        {
            **first,
            "invocation_instruction": HumanMessage(
                content="The audit shows that the body is a bracket."
            ),
        }
    )

    assert second["proposal"] == _proposal("bracket", "through hole")
    assert commands == ["mkdir -p -- /work/semantics_0 /work/semantics_1"]
    assert len(peer.received_messages) == 1
    assert len(head.received_messages) == 3
    reentry = head.received_messages[-1]
    assert any(
        isinstance(message, HumanMessage)
        and "[Revision Context]" in message.text
        and "through hole" in message.text
        for message in reentry
    )


def test_peer_fallback_is_explicit_when_the_reducer_later_reenters() -> None:
    commands: list[str] = []
    head = ScriptedChatModel(
        responses=(
            tool_call("run_shell", {"command": "inspect"}, "inspect"),
            tool_call("run_shell", {"command": "compare"}, "compare"),
            _answer("revised peer one"),
        )
    )
    peer_one = ScriptedChatModel(responses=(_answer("peer one"),))
    peer_two = ScriptedChatModel(responses=(_answer("peer two"),))
    graph = _graph([head, peer_one, peer_two], commands, max_turns=1)

    first = graph.invoke(
        {"invocation_instruction": HumanMessage(content="analyze the drawing")}
    )
    assert first["proposal"] == _proposal("peer one")
    assert first["reducer_state"]["stop_reason"] is StopReason.BUDGET_EXHAUSTED

    graph.invoke(
        {
            **first,
            "invocation_instruction": HumanMessage(content="revise the answer"),
        }
    )

    reentry = head.received_messages[-1]
    revision_context = next(
        message.text
        for message in reentry
        if isinstance(message, HumanMessage)
        and message.text.startswith("[Revision Context]")
    )
    assert "peer one" in revision_context
    assert "peer two" not in revision_context


def test_the_template_renders_whatever_schema_it_was_built_for() -> None:
    """The merge prompt has to show a peer's answer without knowing its shape.
    A schema with none of `_Proposal`'s field names still has to reach the
    reducer intact, or the template has quietly re-acquired a contract."""

    class _Other(BaseModel):
        model_config = ConfigDict(extra="forbid")
        finding: str = Field(..., description="What was found.")

    def _found(text: str) -> AIMessage:
        return AIMessage(content=json.dumps({"finding": text}))

    commands: list[str] = []
    head = ScriptedChatModel(
        responses=(_found("a plate"), _found("a plate and a bore"))
    )
    peer = ScriptedChatModel(responses=(_found("a bore"),))

    @tool("run_shell")
    def run_shell(command: str) -> dict[str, Any]:
        """Record a test shell command as successful."""
        commands.append(command)
        return {"returncode": 0, "stdout": "", "stderr": ""}

    graph = create_fanout_reduce_graph(
        proposer_role="semantic_hypothesizer",
        proposer_models=[head, peer],
        tools=[run_shell],
        fanout_workdir_prefix="/work/semantics",
        proposal_schema=_Other,
        announce_turns=False,
    )
    result = graph.invoke(
        {"invocation_instruction": HumanMessage(content="propose semantics")}
    )

    assert result["proposal"] == _Other(finding="a plate and a bore")
    assert "a bore" in head.received_messages[-1][-1].text


def test_the_template_refuses_a_proposal_schema_that_is_not_a_model() -> None:
    @tool("run_shell")
    def run_shell(command: str) -> dict[str, Any]:
        """Record a test shell command as successful."""
        return {"returncode": 0, "stdout": "", "stderr": ""}

    with pytest.raises(TypeError, match="pydantic model"):
        create_fanout_reduce_graph(
            proposer_role="semantic_hypothesizer",
            proposer_models=[ScriptedChatModel(responses=()) for _ in range(2)],
            tools=[run_shell],
            fanout_workdir_prefix="/work/semantics",
            proposal_schema=dict,  # type: ignore[arg-type]
        )
