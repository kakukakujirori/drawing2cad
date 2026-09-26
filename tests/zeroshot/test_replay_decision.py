import pytest
from langchain_core.messages import AIMessage, HumanMessage

from zeroshot.replay_decision import apply_edits


def _human(text: str) -> HumanMessage:
    return HumanMessage(content=[{"type": "text", "text": text}])


def test_an_edit_replaces_its_one_match_in_human_text_only() -> None:
    history = [_human("- Read the census."), AIMessage(content="- Read the census.")]

    edited = apply_edits(history, [{"old": "the census.", "new": "the census twice."}])

    assert edited[0].text == "- Read the census twice."
    assert edited[1].text == "- Read the census."
    assert history[0].text == "- Read the census."


@pytest.mark.parametrize("texts", [["no match"], ["census", "census"]])
def test_an_edit_that_does_not_match_exactly_once_is_refused(texts: list[str]) -> None:
    with pytest.raises(ValueError, match="not 1"):
        apply_edits([_human(text) for text in texts], [{"old": "census", "new": "x"}])


def test_an_edit_may_expect_a_repeated_match() -> None:
    history = [_human("feedback"), _human("feedback")]

    edited = apply_edits(history, [{"old": "feedback", "new": "note", "count": 2}])

    assert [message.text for message in edited] == ["note", "note"]
