"""What `compact_transcript` trades away, and what it must not."""

from contextlib import nullcontext
from unittest.mock import Mock

import httpx
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from openai import APIError, APIStatusError, LengthFinishReasonError
from openai.types.chat import ChatCompletion

from tests.zeroshot.chat_models import ScriptedChatModel, tool_call
from zeroshot.pipeline.event_logging.projections import RunEvent, RunEventTransformer
from zeroshot.pipeline.workflow.components.compact import (
    COMPACTION_INSTRUCTION,
    SUMMARY_PREAMBLE,
    InvalidCompactionSummary,
    compact_transcript,
)
from zeroshot.pipeline.workflow.middleware import connection_retry


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
    assert isinstance(transcript[1], AIMessage)
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


@pytest.mark.parametrize(
    ("keep_head", "keep_tail", "max_retries"), [(-1, 0, 0), (0, -1, 0), (0, 0, -1)]
)
def test_a_negative_bound_is_refused(
    keep_head: int, keep_tail: int, max_retries: int
) -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        compact_transcript(
            _worked_transcript(),
            model=_notes(),
            keep_head=keep_head,
            keep_tail=keep_tail,
            max_retries=max_retries,
        )


def _status_error(status: int) -> APIStatusError:
    return APIStatusError(
        "provider error",
        response=httpx.Response(
            status, request=httpx.Request("POST", "https://example.invalid/responses")
        ),
        body=None,
    )


@pytest.fixture
def retry_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    sleeps: list[float] = []
    monkeypatch.setattr(connection_retry.time, "sleep", sleeps.append)
    monkeypatch.setattr(connection_retry.random, "uniform", lambda low, high: 0.0)
    return sleeps


@pytest.mark.parametrize(
    "error",
    [
        httpx.RemoteProtocolError("stream dropped"),
        httpx.ReadTimeout("stream stalled"),
        APIError(
            "backend overloaded",
            httpx.Request("POST", "https://example.invalid/responses"),
            body=None,
        ),
        _status_error(429),
        _status_error(503),
        ValueError("OpenRouter API returned an error during streaming: x (code: 502)"),
    ],
)
def test_transport_retries_preserve_every_request_and_config(
    error: Exception, retry_sleeps: list[float]
) -> None:
    model = Mock(spec=BaseChatModel)
    model.invoke.side_effect = [error, error, AIMessage(content="NOTES")]
    transcript = _worked_transcript()
    before = [message.model_dump() for message in transcript]
    config = {"tags": ["compaction-test"]}

    result = compact_transcript(
        transcript, model=model, config=config, keep_head=1, keep_tail=1, max_retries=2
    )

    calls = model.invoke.call_args_list
    assert len(calls) == 3
    assert calls[0] == calls[1] == calls[2]
    assert calls[0].args[0][:-1] == transcript[:-1]
    assert calls[0].kwargs["config"] is config
    assert retry_sleeps == [2.0, 4.0]
    assert result[0] is transcript[0] and result[-1] is transcript[-1]
    assert result[1].text == SUMMARY_PREAMBLE.format(notes="NOTES")
    assert [message.model_dump() for message in transcript] == before


@pytest.mark.parametrize("error", [_status_error(400), ValueError("invalid request")])
def test_permanent_errors_are_not_retried(
    error: Exception, retry_sleeps: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    model = Mock(spec=BaseChatModel)
    model.invoke.side_effect = error
    with pytest.raises(type(error)) as raised:
        compact_transcript(_worked_transcript(), model=model)
    assert raised.value is error
    assert model.invoke.call_count == 1
    assert retry_sleeps == []
    assert "model_retry" in caplog.text and "'retrying': False" in caplog.text


@pytest.mark.parametrize(
    "rejected",
    [
        AIMessage(content="  "),
        AIMessage(content=[{"type": "reasoning", "reasoning": "thinking only"}]),
        AIMessage(
            content="partial notes", response_metadata={"finish_reason": "length"}
        ),
        AIMessage(content="partial notes", response_metadata={"status": "incomplete"}),
        AIMessage(
            content="partial notes",
            response_metadata={"finish_reason": "content_filter"},
        ),
        AIMessage(
            content="notes",
            tool_calls=[{"name": "run_shell", "args": {}, "id": "bad-call"}],
        ),
        LengthFinishReasonError(
            completion=ChatCompletion.model_validate(
                {
                    "id": "cut-off",
                    "created": 0,
                    "model": "test",
                    "object": "chat.completion",
                    "choices": [],
                }
            )
        ),
    ],
)
def test_rejected_summaries_are_corrected_using_the_original_context(
    rejected: AIMessage | Exception, retry_sleeps: list[float]
) -> None:
    model = Mock(spec=BaseChatModel)
    model.invoke.side_effect = [rejected, AIMessage(content="COMPLETE NOTES")]
    transcript = _worked_transcript()

    result = compact_transcript(transcript, model=model, max_retries=1)

    first, second = model.invoke.call_args_list
    assert first.args[0][:-1] == second.args[0][:-1] == transcript
    correction = second.args[0][-1].text
    assert COMPACTION_INSTRUCTION in correction
    assert "shorter, complete notes" in correction
    assert "JSON" not in correction
    assert result[0].text == SUMMARY_PREAMBLE.format(notes="COMPLETE NOTES")
    assert retry_sleeps == []


@pytest.mark.parametrize("last_is_transport", [True, False])
def test_exhaustion_counts_transport_and_summary_failures_separately(
    last_is_transport: bool, retry_sleeps: list[float]
) -> None:
    transport = httpx.ReadTimeout("stream stalled")
    empty = AIMessage(content="")
    model = Mock(spec=BaseChatModel)
    model.invoke.side_effect = [
        transport,
        empty,
        transport if last_is_transport else empty,
    ]
    transcript = _worked_transcript()
    before = [message.model_dump() for message in transcript]

    with pytest.raises(
        httpx.ReadTimeout if last_is_transport else InvalidCompactionSummary
    ):
        compact_transcript(transcript, model=model, max_retries=1)

    assert model.invoke.call_count == 3
    assert retry_sleeps == [2.0]
    assert [message.model_dump() for message in transcript] == before


@pytest.mark.parametrize("succeeds", [True, False])
def test_compaction_retries_reach_the_existing_event_log(
    succeeds: bool, retry_sleeps: list[float]
) -> None:
    model = Mock(spec=BaseChatModel)
    model.invoke.side_effect = [
        _status_error(429),
        AIMessage(content="NOTES") if succeeds else _status_error(429),
    ]

    graph = StateGraph(dict)  # pyrefly: ignore[bad-specialization]
    graph.add_node(
        "compact",
        lambda state: {
            "messages": compact_transcript(
                state["messages"], model=model, max_retries=1
            )
        },
    )
    graph.add_edge(START, "compact")
    graph.add_edge("compact", END)
    events: list[RunEvent] = []
    transformer = RunEventTransformer(sink=events.append)
    transformer.init()

    with nullcontext() if succeeds else pytest.raises(APIStatusError):
        for chunk in graph.compile().stream(
            {"messages": _worked_transcript()}, stream_mode="custom"
        ):
            transformer.process(  # type: ignore[arg-type]
                {
                    "type": "event",
                    "method": "custom",
                    "params": {"timestamp": 1000, "namespace": [], "data": chunk},
                }
            )

    assert len(events) == (1 if succeeds else 2)
    for attempt, event in enumerate(events, start=1):
        assert event["event"] == "model_retry"
        report = event["data"]
        assert report["role"] == "compaction"
        assert (report["attempt"], report["max_attempts"]) == (attempt, 2)
        assert report["status_code"] == 429
        assert report["retrying"] is (attempt == 1)
        assert report["request_adjusted"] is False
    assert retry_sleeps == [2.0]
