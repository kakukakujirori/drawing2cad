"""Message Compaction"""

from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage
from langchain_core.messages.content import create_text_block
from langchain_core.runnables import RunnableConfig

COMPACTION_INSTRUCTION = """\
You are about to continue this same conversation with everything above it
replaced by what you write now. Write the notes your later self will need and
would otherwise have to work out again from nothing.

Answer every section. Where a section has nothing to report, write "None".

## OBJECTIVE
What is being asked of you overall, and what would count as having done it.

## FINDINGS
What you established, and the evidence that settled it: the measurements,
values and identifiers you arrived at, and the observations they came from.
Include the conclusions you considered and rejected, and why, so that you do
not reopen them.

## WORK DONE
The commands you ran, the files you read or wrote, and what came back. Enough
that you do not repeat an action whose result you already have.

## OPEN QUESTIONS
What is still undecided, and what would settle it.

## NEXT STEPS
What to do next.

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
) -> list[AnyMessage]:
    """Compact transcript messages into a summary while preserving initial inputs.

    Retains the first `keep_head` and last `keep_tail` messages verbatim and
    compresses the intermediate reasoning and tool calls between them into a
    summary.  Both default to nothing: which messages are irreplaceable -- the
    drawing a run opened with, the turns a caller means to go on working from
    -- is known to the caller and not here.  Either end is extended when it
    would otherwise cut a tool call away from its answer.

    The summary is wrapped as a `HumanMessage` to serve as context/input rather
    than ungrounded past model output.
    """
    if keep_head < 0:
        raise ValueError(f"{keep_head=} must not be negative")
    if keep_tail < 0:
        raise ValueError(f"{keep_tail=} must not be negative")

    head_end = _past_answers_to_earlier_calls(
        transcript, min(keep_head, len(transcript))
    )
    tail_start = _past_answers_to_earlier_calls(
        transcript, max(head_end, len(transcript) - keep_tail)
    )
    if tail_start <= head_end:
        return list(transcript)

    notes = model.invoke(
        [
            *transcript[:tail_start],
            HumanMessage(content_blocks=[create_text_block(COMPACTION_INSTRUCTION)]),
        ],
        config=config,
    )
    return [
        *transcript[:head_end],
        HumanMessage(
            content_blocks=[
                create_text_block(SUMMARY_PREAMBLE.format(notes=notes.text.strip()))
            ]
        ),
        *transcript[tail_start:],
    ]
