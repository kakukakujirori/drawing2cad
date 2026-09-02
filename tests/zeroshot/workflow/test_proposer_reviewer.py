import json
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

from tests.zeroshot.chat_models import ScriptedChatModel, tool_call
from zeroshot.pipeline.workflow import StopReason
from zeroshot.pipeline.workflow.components.proposer_reviewer import (
    Proposal,
    Review,
    create_proposer_reviewer_loop,
)

PROMPT_CONTEXT = {"output_path": "/work/model.py", "verification_dir": "/work/attempts"}


@tool("echo")
def echo(value: str) -> str:
    """Return the supplied value."""
    return value


def _hypothesis(*semantics: str) -> AIMessage:
    return AIMessage(
        content=json.dumps(
            {"proposal": list(semantics), "rationale": "the views agree"}
        )
    )


def _proposed(*semantics: str) -> Proposal:
    return Proposal(proposal=list(semantics), rationale="the views agree")


def _review(accept: bool, rationale: str) -> AIMessage:
    return AIMessage(content=json.dumps({"accept": accept, "rationale": rationale}))


def _loop(
    proposer: BaseChatModel,
    reviewer: BaseChatModel,
    max_revisions: int = 2,
    **options: Any,
):
    return create_proposer_reviewer_loop(
        proposer_role="semantic_hypothesizer",
        proposer_model=proposer,
        reviewer_role="semantic_reviewer",
        reviewer_model=reviewer,
        tools=(echo,),
        prompt_context=PROMPT_CONTEXT,
        response_format_strategy="provider",
        max_revisions=max_revisions,
        announce_turns=False,
        checkpointer=False,
        **options,
    )


def _entry(text: str = "Here is the drawing") -> dict[str, HumanMessage]:
    return {
        "proposer_entry_instruction": HumanMessage(content=text, id=f"proposer-{text}"),
        "reviewer_entry_instruction": HumanMessage(content=text, id=f"reviewer-{text}"),
    }


def _texts(messages: list[BaseMessage]) -> list[str]:
    return [message.text for message in messages]


def test_an_accepted_proposal_is_the_stage_artifact() -> None:
    proposer = ScriptedChatModel(responses=(_hypothesis("a flanged boss"),))
    reviewer = ScriptedChatModel(responses=(_review(True, "matches all views"),))

    state = _loop(proposer, reviewer).invoke(_entry())

    assert state["proposal"] == _proposed("a flanged boss")
    assert state["review"] == Review(accept=True, rationale="matches all views")
    assert state["revision_count"] == 1
    assert (
        state["proposer_state"]["current_turn"],
        state["proposer_state"]["total_turns"],
        state["proposer_state"]["stop_reason"],
    ) == (1, 1, StopReason.COMPLETED)
    assert (
        state["reviewer_state"]["current_turn"],
        state["reviewer_state"]["total_turns"],
        state["reviewer_state"]["stop_reason"],
    ) == (1, 1, StopReason.COMPLETED)


def test_a_rejected_proposal_comes_back_with_the_reviewers_rationale() -> None:
    proposer = ScriptedChatModel(
        responses=(_hypothesis("a plate"), _hypothesis("a plate", "a through hole"))
    )
    reviewer = ScriptedChatModel(
        responses=(
            _review(False, "the hole in the top view is unaccounted for"),
            _review(True, "now complete"),
        )
    )

    state = _loop(proposer, reviewer).invoke(_entry())

    assert state["proposal"] == _proposed("a plate", "a through hole")
    assert state["revision_count"] == 2
    assert "the hole in the top view is unaccounted for" in (
        proposer.received_messages[1][-1].text
    )
    # Each revision has a fresh per-revision budget, while lifetime usage is kept.
    assert (
        state["proposer_state"]["current_turn"],
        state["proposer_state"]["total_turns"],
    ) == (1, 2)
    assert (
        state["reviewer_state"]["current_turn"],
        state["reviewer_state"]["total_turns"],
    ) == (1, 2)


def test_the_reviewer_sees_the_proposal_but_not_the_proposers_transcript() -> None:
    proposer = ScriptedChatModel(
        responses=(_hypothesis("a plate"), _hypothesis("a plate", "a hole"))
    )
    reviewer = ScriptedChatModel(
        responses=(
            _review(False, "check the top view"),
            _review(True, "fine"),
        )
    )

    state = _loop(proposer, reviewer).invoke(_entry())

    first_review, second_review = reviewer.received_messages
    assert [type(message) for message in first_review] == [
        SystemMessage,
        HumanMessage,
        HumanMessage,
    ]
    assert "a plate" in first_review[-1].text
    assert "the views agree" in first_review[-1].text
    # The second review retains the reviewer's own decision, not the proposer's AI turn.
    assert "check the top view" in "".join(_texts(second_review))
    assert not any(
        message is proposer.received_messages[0][-1] for message in second_review
    )
    assert state["proposal"] == _proposed("a plate", "a hole")


def test_a_role_is_told_no_schema_but_its_own() -> None:
    """The reviewer owes a verdict, not a proposal, and a schema it does not
    owe would only invite it to answer in the wrong shape."""
    proposer = ScriptedChatModel(responses=(_hypothesis("a plate"),))
    reviewer = ScriptedChatModel(responses=(_review(True, "fine"),))

    _loop(proposer, reviewer).invoke(_entry())

    proposer_prompt = proposer.received_messages[0][0].text
    reviewer_prompt = reviewer.received_messages[0][0].text
    assert json.dumps(Review.model_json_schema(), indent=2) in reviewer_prompt
    assert json.dumps(Proposal.model_json_schema(), indent=2) not in reviewer_prompt
    # The proposer is held to its shape by the response format it is called
    # with, which `ProviderStrategy` sends with the request, so its role does
    # not carry a copy of the schema for the stages sharing that role to read.
    assert json.dumps(Proposal.model_json_schema(), indent=2) not in proposer_prompt


def test_revision_limit_counts_reviews_and_keeps_the_latest_proposal() -> None:
    proposer = ScriptedChatModel(
        responses=(_hypothesis("first"), _hypothesis("second"), _hypothesis("unused"))
    )
    reviewer = ScriptedChatModel(
        responses=(
            _review(False, "still incomplete"),
            _review(False, "still incomplete"),
            _review(False, "unused"),
        )
    )

    state = _loop(proposer, reviewer, max_revisions=2).invoke(_entry())

    assert len(proposer.received_messages) == 2
    assert len(reviewer.received_messages) == 2
    assert state["revision_count"] == 2
    assert state["review"].accept is False
    assert state["proposal"] == _proposed("second")


def test_a_proposer_that_runs_out_of_turns_is_not_reviewed() -> None:
    proposer = ScriptedChatModel(
        responses=tuple(
            tool_call("echo", {"value": "looking"}, f"call-{turn}") for turn in range(4)
        )
    )
    reviewer = ScriptedChatModel(responses=())

    state = _loop(
        proposer,
        reviewer,
        max_proposer_turns_per_revision=2,
    ).invoke(_entry())

    assert state["proposal"] is None
    assert state["proposer_state"]["stop_reason"] == StopReason.BUDGET_EXHAUSTED
    assert state["proposer_state"]["current_turn"] == 2
    assert state["proposer_state"]["total_turns"] == 2
    assert reviewer.received_messages == []


def test_a_proposal_that_breaks_its_contract_ends_the_loop() -> None:
    """The proposer has already spent its corrections on that answer, so asking
    the same node again would only repeat them."""
    proposer = ScriptedChatModel(
        responses=(AIMessage(content='{"proposal": "a plate"}'),)
    )
    reviewer = ScriptedChatModel(responses=())

    result = _loop(proposer, reviewer, model_retries=0).invoke(_entry())

    assert result.get("proposal") is None
    assert reviewer.received_messages == []


def test_stage_reentry_resets_per_invocation_counts_but_keeps_agent_memory() -> None:
    proposer = ScriptedChatModel(
        responses=(_hypothesis("first"), _hypothesis("revised"))
    )
    reviewer = ScriptedChatModel(
        responses=(_review(True, "fine"), _review(True, "also fine"))
    )
    loop = _loop(proposer, reviewer)

    first = loop.invoke(_entry("initial"))
    second = loop.invoke({**first, **_entry("audit redo")})

    assert second["revision_count"] == 1
    assert (
        second["proposer_state"]["current_turn"],
        second["proposer_state"]["total_turns"],
    ) == (1, 2)
    assert (
        second["reviewer_state"]["current_turn"],
        second["reviewer_state"]["total_turns"],
    ) == (1, 2)
    assert "initial" in _texts(second["proposer_state"]["messages"])
    assert "audit redo" in _texts(second["proposer_state"]["messages"])


def test_the_loop_requires_a_non_negative_revision_limit() -> None:
    proposer = ScriptedChatModel(responses=())
    reviewer = ScriptedChatModel(responses=())

    with pytest.raises(ValueError, match="must be non-negative"):
        _loop(proposer, reviewer, max_revisions=-1)


def test_no_revision_budget_makes_the_first_proposal_the_answer() -> None:
    """Zero is the setting that turns the stage into a single ask.

    Two models arguing over the same evidence is the loop's whole cost, so
    switching it off has to leave a proposal behind rather than nothing.
    """
    proposer = ScriptedChatModel(responses=(_hypothesis("only"),))
    reviewer = ScriptedChatModel(responses=())

    state = _loop(proposer, reviewer, max_revisions=0).invoke(_entry())

    assert state["proposal"] == _proposed("only")
    assert reviewer.received_messages == []
    assert state.get("revision_count", 0) == 0


def test_a_stage_with_no_review_budget_may_keep_its_revision_count() -> None:
    """Entry rejects a count that is already spent, and zero is never spent.

    A loop configured for no reviews starts every entry at the limit, so the
    guard would refuse the one proposal the stage exists to produce.
    """
    proposer = ScriptedChatModel(responses=(_hypothesis("first"), _hypothesis("again")))
    reviewer = ScriptedChatModel(responses=())
    loop = _loop(
        proposer,
        reviewer,
        max_revisions=0,
        reset_revision_count_when_reentrant=False,
    )

    first = loop.invoke(_entry("initial"))
    second = loop.invoke({**first, **_entry("audit redo")})

    assert second["proposal"] == _proposed("again")


def test_reentering_a_stage_that_spent_its_revisions_is_refused() -> None:
    """Without a reset the loop would re-enter with nothing left to do."""
    proposer = ScriptedChatModel(responses=(_hypothesis("first"),))
    reviewer = ScriptedChatModel(responses=(_review(False, "incomplete"),))
    loop = _loop(
        proposer,
        reviewer,
        max_revisions=1,
        reset_revision_count_when_reentrant=False,
    )

    first = loop.invoke(_entry("initial"))
    assert first["revision_count"] == 1

    with pytest.raises(ValueError, match="Revision count reached max revisions"):
        loop.invoke({**first, **_entry("audit redo")})
