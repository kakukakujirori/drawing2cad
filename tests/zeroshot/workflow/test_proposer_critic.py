import json
from functools import partial
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from tests.zeroshot.chat_models import ScriptedChatModel, tool_call
from zeroshot.pipeline.workflow import create_agent
from zeroshot.pipeline.workflow.proposer_critic import (
    ProposerCriticSpec,
    create_proposer_critic_loop,
)
from zeroshot.pipeline.workflow.state import Review, SemanticHypothesis, StopReason

SEMANTIC = ProposerCriticSpec(proposal=SemanticHypothesis, instructions="semantic")

PROMPT_CONTEXT = {"output_path": "/work/model.py", "verification_dir": "/work/attempts"}


def _hypothesis(*semantics: str) -> AIMessage:
    return AIMessage(content=json.dumps({"semantics": list(semantics)}))


def _review(decision: str, feedback: str = "") -> AIMessage:
    return AIMessage(content=json.dumps({"decision": decision, "feedback": feedback}))


def _loop(
    proposer: BaseChatModel,
    critic: BaseChatModel,
    max_revisions: int = 2,
    **agent_options: Any,
):
    options = {"announce_turn_budget": False, **agent_options}
    return create_proposer_critic_loop(
        SEMANTIC,
        partial(create_agent, role="semantic_hypothesizer", model=proposer, **options),
        partial(create_agent, role="semantic_reviewer", model=critic, **options),
        max_revisions=max_revisions,
        structured_output="provider",
        tools=[],
        prompt_context=PROMPT_CONTEXT,
    )


def _drawing() -> list[BaseMessage]:
    return [HumanMessage(content="Here is the drawing", id="run-input")]


def _texts(messages: list[BaseMessage]) -> list[str]:
    return [message.text for message in messages]


def test_an_accepted_proposal_is_the_stages_artifact() -> None:
    proposer = ScriptedChatModel(responses=(_hypothesis("a flanged boss"),))
    critic = ScriptedChatModel(responses=(_review("accept", "matches all views"),))

    stage = _loop(proposer, critic)(_drawing())

    assert stage.artifact == SemanticHypothesis(semantics=["a flanged boss"])
    assert stage.stop_reasons == {
        "semantic_hypothesizer": StopReason.COMPLETED,
        "semantic_reviewer": StopReason.COMPLETED,
    }
    # One model call each, and the stage reports what it cost altogether.
    assert stage.turns == {"semantic_hypothesizer": 1, "semantic_reviewer": 1}
    # Both roles' turns come back, and the run's own input does not: the
    # workflow already holds that.
    assert [type(message) for message in stage.messages] == [
        HumanMessage,
        AIMessage,
        HumanMessage,
        AIMessage,
    ]


def test_a_rejected_proposal_comes_back_with_the_feedback() -> None:
    proposer = ScriptedChatModel(
        responses=(_hypothesis("a plate"), _hypothesis("a plate", "a through hole"))
    )
    critic = ScriptedChatModel(
        responses=(
            _review("revise", "the hole in the top view is unaccounted for"),
            _review("accept", "now complete"),
        )
    )

    stage = _loop(proposer, critic)(_drawing())

    assert stage.artifact == SemanticHypothesis(semantics=["a plate", "a through hole"])
    second_attempt = proposer.received_messages[1]
    assert "the hole in the top view is unaccounted for" in second_attempt[-1].text


def test_the_critic_judges_the_proposal_and_not_the_reasoning_behind_it() -> None:
    """Reading how the proposal was arrived at is how a critic talks itself
    into accepting it."""
    proposer = ScriptedChatModel(
        responses=(
            tool_call("run_shell", {"command": "true"}, "call-look"),
            _hypothesis("a plate"),
            _hypothesis("a plate"),
        )
    )
    critic = ScriptedChatModel(
        responses=(_review("revise", "check the top view"), _review("accept", "fine")),
    )

    stage = _loop(proposer, critic)(_drawing())

    first_review, second_review = critic.received_messages
    # The drawing, then the proposal to judge. Nothing the proposer said.
    assert [type(message) for message in first_review] == [
        SystemMessage,
        HumanMessage,
        HumanMessage,
    ]
    assert "a plate" in first_review[-1].text
    assert not [m for m in first_review if isinstance(m, ToolMessage)]
    # On the second round the critic sees its own previous review, so it can
    # tell whether what it asked for was done.
    assert "check the top view" in "".join(_texts(second_review))
    assert stage.artifact == SemanticHypothesis(semantics=["a plate"])


def test_each_role_is_told_the_schema_it_owes_from_one_definition() -> None:
    proposer = ScriptedChatModel(responses=(_hypothesis("a plate"),))
    critic = ScriptedChatModel(responses=(_review("accept", "fine"),))

    _loop(proposer, critic)(_drawing())

    proposer_prompt = proposer.received_messages[0][0].text
    critic_prompt = critic.received_messages[0][0].text
    assert (
        json.dumps(SemanticHypothesis.model_json_schema(), indent=2) in proposer_prompt
    )
    assert json.dumps(Review.model_json_schema(), indent=2) in critic_prompt
    # The prompts no longer carry a hand-written copy to drift from.
    assert '"semantics"' not in critic_prompt


def test_the_loop_stops_at_its_revision_limit_and_keeps_the_latest_proposal() -> None:
    proposer = ScriptedChatModel(
        responses=tuple(_hypothesis("a plate") for _ in range(6))
    )
    critic = ScriptedChatModel(
        responses=tuple(_review("revise", "still incomplete") for _ in range(6))
    )

    stage = _loop(proposer, critic, max_revisions=2)(_drawing())

    assert len(proposer.received_messages) == 3
    assert len(critic.received_messages) == 3
    assert stage.artifact == SemanticHypothesis(semantics=["a plate"])
    # Three rounds of two agents, counted over the whole stage.
    # Three rounds each, accumulated rather than replaced.
    assert stage.turns == {"semantic_hypothesizer": 3, "semantic_reviewer": 3}


def test_a_proposer_that_ran_out_of_turns_leaves_no_artifact() -> None:
    proposer = ScriptedChatModel(
        responses=tuple(
            tool_call("run_shell", {"command": "true"}, f"call-{turn}")
            for turn in range(4)
        )
    )
    critic = ScriptedChatModel(responses=())

    stage = _loop(proposer, critic, max_turns=2)(_drawing())

    assert stage.artifact is None
    # The critic never ran, so the proposer's is the only reason there is.
    assert stage.stop_reasons == {"semantic_hypothesizer": StopReason.BUDGET_EXHAUSTED}
    assert critic.received_messages == []


def test_a_proposal_that_breaks_its_contract_stops_the_run() -> None:
    proposer = ScriptedChatModel(
        responses=(AIMessage(content='{"semantics": "a plate"}'),)
    )
    critic = ScriptedChatModel(responses=())

    with pytest.raises(Exception, match="SemanticHypothesis"):
        _loop(proposer, critic)(_drawing())


def test_the_loop_rejects_a_negative_revision_limit() -> None:
    proposer = ScriptedChatModel(responses=())
    critic = ScriptedChatModel(responses=())

    with pytest.raises(ValueError, match="max_revisions must not be negative"):
        _loop(proposer, critic, max_revisions=-1)
