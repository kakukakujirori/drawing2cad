"""Message Compaction"""

import re
from collections.abc import Sequence
from enum import StrEnum
from functools import partial
from typing import assert_never

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.messages.content import create_text_block
from langchain_core.runnables import RunnableConfig
from openai import LengthFinishReasonError

from zeroshot.pipeline.workflow.middleware.connection_retry import (
    ModelConnectionRetry,
    report_model_retry,
)


class InvalidCompactionSummary(ValueError):
    """A summary that cannot safely replace the original conversation."""


def compaction_model(
    model: BaseChatModel, max_tokens: int | None = None
) -> BaseChatModel:
    """The model that writes the notes, its output capped at `max_tokens`.

    A model whose own max_tokens is lower or unset keeps it; None leaves the
    model as it is.
    """
    if max_tokens is None:
        return model
    if max_tokens < 1:
        raise ValueError(f"{max_tokens=} must be positive")
    if "max_tokens" not in type(model).model_fields:
        raise ValueError(f"{type(model).__name__} has no max_tokens to cap")
    own = getattr(model, "max_tokens", None)
    if own is None or own <= max_tokens:
        return model
    return model.model_copy(update={"max_tokens": max_tokens})


COMPACTION_INSTRUCTION = """\
You are about to continue this same conversation with everything above it
replaced by what you write now. Write the notes your later self will need and
would otherwise have to work out again from nothing.

Answer every section. Where a section has nothing to report, write "None".

## OBJECTIVE
What is being asked of you overall, and what would count as having done it.

## FINDINGS
What you observed and measured: the values and identifiers you arrived at, and
the images, files or commands that showed them. Separate direct observations
and measurements from untested predictions. Do not list rejected alternatives
here. A rejection is a judgment, and it can be wrong.

## WORK DONE
The commands you ran, the files you read or wrote, and what came back. Enough
that you do not repeat an action whose result you already have.

## OPEN QUESTIONS
What is still undecided, and what would settle it. List each rejected
alternative here with the views, lines or numbers that decided against it and
the artifact paths. Mark untested alternatives as untested.

These are notes to yourself, not a report to a reader. Answer with the notes
alone: no preamble, and no remarks about the act of summarising.
"""

SUMMARY_PREAMBLE = """\
You are in the middle of a conversation whose earlier turns have been replaced
by the notes below.  Nothing else of them is left in what you can see, so work
from the notes as you would have worked from the turns they stand for.

<summary>
{notes}
</summary>
"""


_SECTIONS = ("## OBJECTIVE", "## FINDINGS", "## WORK DONE", "## OPEN QUESTIONS")


def _has_every_section(notes: str) -> bool:
    """Whether every section asked for heads a line, however it is marked up."""
    return all(
        re.search(rf"(?im)^[#*\s]*{re.escape(section.removeprefix('## '))}\b", notes)
        for section in _SECTIONS
    )


def _extract_summary_from_reasoning(message: AIMessage) -> str:
    """Notes delivered in the reasoning channel, the reply left empty.

    Taken only when the reasoning is the notes alone: it opens with the first
    section, holds it once, and has every section. Thinking before a draft, or
    a second draft, is no summary.
    """
    blocks = message.content if isinstance(message.content, list) else []
    reasoning = "".join(
        block.get("reasoning", "")
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "reasoning"
    ) or str(message.additional_kwargs.get("reasoning_content") or "")
    notes = reasoning.strip()
    if not notes.startswith(_SECTIONS[0]) or notes.count(_SECTIONS[0]) != 1:
        return ""
    if any(section not in notes for section in _SECTIONS[1:]):
        return ""
    return notes


def _past_answers_to_earlier_calls(transcript: Sequence[AnyMessage], index: int) -> int:
    """Move `index` past `ToolMessage`s answering a call made before it.

    A cut that ends a run on an `AIMessage` asking for tools, or opens one on
    the `ToolMessage` answering it, leaves a request without its reply or a
    reply without its request; providers reject both.  The answers sit directly
    after the call, so taking them settles either case.

    An index of 0 has nothing before it to have made a call, so it stays put --
    a caller keeping no head is not owed messages it did not ask to keep.
    """
    while 0 < index < len(transcript) and isinstance(transcript[index], ToolMessage):
        index += 1
    return index


def compact_transcript(
    transcript: Sequence[AnyMessage],
    *,
    model: BaseChatModel,
    config: RunnableConfig | None = None,
    keep_head: int = 0,
    keep_tail: int = 0,
    max_retries: int = 5,
) -> list[AnyMessage]:
    """Compact transcript messages into a summary while preserving initial inputs.

    Retains the first `keep_head` and last `keep_tail` messages verbatim and
    compresses the intermediate reasoning and tool calls between them into a
    summary.  Both default to nothing: which messages are irreplaceable -- the
    drawing a run opened with, the turns a caller means to go on working from
    -- is known to the caller and not here.  Either end is extended when it
    would otherwise cut a tool call away from its answer.

    The summary is wrapped as a `HumanMessage` to serve as context/input rather
    than ungrounded past model output. Transport failures and rejected summaries
    each get up to `max_retries` retries, counted separately as in agent calls.
    Exhaustion raises without changing the supplied transcript.
    """
    if keep_head < 0:
        raise ValueError(f"{keep_head=} must not be negative")
    if keep_tail < 0:
        raise ValueError(f"{keep_tail=} must not be negative")
    if max_retries < 0:
        raise ValueError(f"{max_retries=} must not be negative")

    head_end = _past_answers_to_earlier_calls(
        transcript, min(keep_head, len(transcript))
    )
    tail_start = _past_answers_to_earlier_calls(
        transcript, max(head_end, len(transcript) - keep_tail)
    )
    if tail_start <= head_end:
        return list(transcript)

    instruction = COMPACTION_INSTRUCTION
    transport = ModelConnectionRetry(max_retries, role="compaction")
    rejected = 0
    while True:
        try:
            notes = transport.invoke(
                partial(
                    model.invoke,
                    [*transcript[:tail_start], HumanMessage(content=instruction)],
                    config=config,
                )
            )
            summary = notes.text.strip() or _extract_summary_from_reasoning(notes)
            metadata = notes.response_metadata
            if metadata.get("finish_reason") in {
                "length",
                "content_filter",
            } or metadata.get("status") in {"incomplete", "failed", "cancelled"}:
                raise InvalidCompactionSummary(
                    "Compaction returned an incomplete summary "
                    f"(finish_reason={metadata.get('finish_reason')}, "
                    f"status={metadata.get('status')})"
                )
            if not summary:
                raise InvalidCompactionSummary("Compaction returned an empty summary")
            if not _has_every_section(summary):
                raise InvalidCompactionSummary(
                    "Compaction returned notes without every section"
                )
            if notes.tool_calls or notes.invalid_tool_calls:
                raise InvalidCompactionSummary(
                    "Compaction returned tool calls, not notes"
                )
        except (InvalidCompactionSummary, LengthFinishReasonError) as error:
            retrying = rejected < max_retries
            report_model_retry(
                error,
                role="compaction",
                attempt=rejected,
                max_retries=max_retries,
                retrying=retrying,
                adjusted=True,
            )
            if not retrying:
                raise
            rejected += 1
            # Replace the instruction without accumulating partial summaries
            # or correction prompts. The original context stays intact.
            instruction = (
                COMPACTION_INSTRUCTION
                + "\nYour previous summary was empty or incomplete, or used tools. "
                "Return shorter, complete notes covering every section. "
                "Write the notes now; do not call tools."
            )
        else:
            break
    return [
        *transcript[:head_end],
        HumanMessage(
            content_blocks=[create_text_block(SUMMARY_PREAMBLE.format(notes=summary))]
        ),
        *transcript[tail_start:],
    ]


# ------------------------------------------------------------------
# When an agent whose turns ran out is re-asked from notes
# ------------------------------------------------------------------


class RetryCompaction(StrEnum):
    """When to compact an agent's transcript before re-asking it once its turns ran out."""

    NEVER = "never"
    # The first time in a round: later retries are short and keep their context.
    ONCE = "once"
    EVERY = "every"

    def applies(self, *, exhausted: bool, compacted_this_round: bool) -> bool:
        """Whether a retry starts from notes rather than the whole transcript."""
        if not exhausted:
            return False
        match self:
            case RetryCompaction.NEVER:
                return False
            case RetryCompaction.ONCE:
                return not compacted_this_round
            case RetryCompaction.EVERY:
                return True
            case _:
                assert_never(self)
