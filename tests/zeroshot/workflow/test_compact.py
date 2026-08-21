"""What `compact_transcript` trades away, and what it must not."""

import pytest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from tests.zeroshot.chat_models import ScriptedChatModel, tool_call
from zeroshot.pipeline.workflow.components.compact import (
    COMPACTION_INSTRUCTION,
    SUMMARY_PREAMBLE,
    compact_transcript,
)


def _worked_transcript() -> list[AnyMessage]:
    """An opening that carries the drawing, then the turns spent reading it."""
    return [
        HumanMessage(content="[Input DXF path: /work/inputs/techdraw.dxf]"),
        tool_call("run_shell", {"command": "ezdxf draw techdraw.dxf"}, "call-0"),
        ToolMessage(content="wrote techdraw.png", tool_call_id="call-0"),
        AIMessage(content="the front view is 100mm wide"),
    ]


def _notes(text: str = "measured 100mm across the front view") -> ScriptedChatModel:
    return ScriptedChatModel(responses=(AIMessage(content=text),))


def test_everything_is_traded_away_by_default() -> None:
    """Which message is irreplaceable, and whether one is, belongs to whoever
    knows what the conversation was for."""
    compacted = compact_transcript(_worked_transcript(), model=_notes())

    assert len(compacted) == 1
    assert compacted[0].text == SUMMARY_PREAMBLE.format(
        notes="measured 100mm across the front view"
    )


def test_the_notes_come_back_saying_they_stand_in_for_turns_that_are_gone() -> None:
    """Handed a bare summary, a model reads its own notes as something it was
    told; and without delimiters, the framing reads as part of them."""
    (summary,) = compact_transcript(_worked_transcript(), model=_notes("NOTES"))

    assert "replaced" in summary.text
    assert summary.text.rstrip().endswith("<summary>\nNOTES\n</summary>")
    # Nothing offers the turns back, because nothing kept them.
    assert "saved to" not in summary.text


def test_a_caller_that_knows_its_opening_is_irreplaceable_keeps_it() -> None:
    transcript = _worked_transcript()

    compacted = compact_transcript(transcript, model=_notes(), keep_head=1)

    assert compacted[0] is transcript[0]
    assert len(compacted) == 2
    assert "<summary>" in compacted[1].text


def test_a_caller_that_means_to_go_on_working_keeps_the_turns_it_needs() -> None:
    transcript = _worked_transcript()

    compacted = compact_transcript(transcript, model=_notes(), keep_tail=1)

    assert compacted[-1] is transcript[-1]
    assert len(compacted) == 2
    assert "<summary>" in compacted[0].text


def test_a_kept_head_takes_the_answers_to_a_call_it_would_end_on() -> None:
    """A request without its reply is rejected by the provider, so the head
    that asked has to keep what answered it."""
    transcript = _worked_transcript()
    assert transcript[1].tool_calls and isinstance(transcript[2], ToolMessage)

    compacted = compact_transcript(transcript, model=_notes(), keep_head=2)

    assert compacted[:3] == transcript[:3]
    assert "<summary>" in compacted[3].text


def test_a_kept_tail_does_not_open_on_a_reply_whose_request_is_gone() -> None:
    transcript = _worked_transcript()

    compacted = compact_transcript(transcript, model=_notes(), keep_tail=2)

    assert not isinstance(compacted[1], ToolMessage)
    assert compacted[1:] == transcript[3:]


def test_the_model_is_shown_the_conversation_it_is_asked_to_read_back() -> None:
    """The notes are written by whoever holds the context, not by a reader of
    a rendering of it, so the real messages go to the model."""
    model = _notes()
    transcript = _worked_transcript()

    compact_transcript(transcript, model=model)

    (asked,) = model.received_messages
    assert asked[: len(transcript)] == transcript
    assert asked[-1].text == COMPACTION_INSTRUCTION


def test_the_kept_turns_are_not_sent_to_be_summarised() -> None:
    """They survive verbatim, so summarising them would only spend tokens."""
    model = _notes()
    transcript = _worked_transcript()

    compact_transcript(transcript, model=model, keep_tail=1)

    (asked,) = model.received_messages
    assert asked[:-1] == transcript[:-1]


def test_a_transcript_with_nothing_to_trade_is_left_alone() -> None:
    model = _notes()
    transcript = _worked_transcript()

    assert compact_transcript([], model=model) == []
    assert compact_transcript(transcript, model=model, keep_head=99) == transcript
    assert compact_transcript(transcript, model=model, keep_tail=99) == transcript
    assert model.received_messages == []


def test_a_negative_bound_is_refused() -> None:
    for bounds in ({"keep_head": -1}, {"keep_tail": -1}):
        with pytest.raises(ValueError, match="must not be negative"):
            compact_transcript(_worked_transcript(), model=_notes(), **bounds)
