"""Agent middleware for answer correction, backed by shared transport retries."""

import json
from collections.abc import Awaitable, Callable, Collection
from functools import partial
from typing import Any, NamedTuple, override

from langchain.agents import AgentState as _AgentState
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.structured_output import (
    MultipleStructuredOutputsError,
    StructuredOutputError,
    StructuredOutputValidationError,
    ToolStrategy,
)
from langchain_core.messages import AIMessage, HumanMessage
from openai import LengthFinishReasonError
from pydantic import ValidationError

from zeroshot.pipeline.stages._base.error_locations import answer_errors

from .connection_retry import ModelConnectionRetry, report_model_retry


class UnansweredModelCall(Exception):
    """A model call that came back with nothing the agent loop can act on."""


class TextWithoutToolCall(UnansweredModelCall):
    """Text without a tool call under ToolStrategy, which ends the agent unanswered.

    `ChatOpenRouterSingleReasoning.bind_tools` lets models skip tools, so this happens.
    """


class MalformedToolCall(UnansweredModelCall):
    """A call whose arguments did not parse, so LangChain dropped it.

    The provider reports the turn as finished, so the work exists and only its
    arguments are unreadable. Without this it reads as an empty turn and the
    author is told to write less, which costs the work it has already done.
    """


class ModelCallRetryMiddleware(AgentMiddleware[_AgentState[Any], None, Any]):
    """Retry incomplete model calls, and transport failures beside them.

    The two are counted apart. An answer this middleware rejects -- truncated,
    unparseable, or empty -- is evidence about the model, and the retry that
    follows carries feedback the next attempt is meant to act on. A dropped
    stream or a gateway timeout is evidence about nothing: the model never
    answered, the request goes back unchanged, and spending one of the
    corrections on it would take away a chance the model never got to use.

    An answer failure it runs out of corrections for ends the turn rather than
    the run. The graph already knows what to do with a stage that produced no
    answer -- it re-asks that stage, and stops the round when the re-asks run
    out -- and raising here reached none of that: one GLM audit put a
    `"type": "object"` from the schema text into every finding, and six
    identical corrections later the sample died with everything it had built
    unscored. Empty, truncated and unparseable all end the turn the same way,
    on a message saying how many attempts it took and what the last problem
    was.

    Every attempt it gives up on is reported. A retry is otherwise invisible:
    it leaves no turn, no message and no event, so the only trace is an extra
    HTTP request in a log that also shows requests a single attempt made. What
    an abandoned attempt does leave behind is real -- a model stream nobody
    will ever finish -- so a run has to be able to say when one happened.
    """

    def __init__(
        self,
        max_retries: int,
        role: str = "",
        max_transport_retries: int | None = None,
    ) -> None:
        super().__init__()
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if max_transport_retries is not None and max_transport_retries < 0:
            raise ValueError("max_transport_retries must be >= 0")
        self.max_retries = max_retries
        self.max_transport_retries = (
            max_retries if max_transport_retries is None else max_transport_retries
        )
        self.role = role

    def _report(
        self,
        request: ModelRequest[None],
        attempt: int,
        error: Exception,
        *,
        retrying: bool,
        response: ModelResponse[Any] | None = None,
    ) -> None:
        """Say which attempt failed, why, and what happens next."""
        details: dict[str, object] = {}
        if isinstance(error, StructuredOutputError):
            # The retry request carries this response only in memory. Preserve
            # the rejected raw output so a contract failure is reproducible.
            details["failed_response"] = _extract_text_or_tool_args(error.ai_message)
            details["generation_id"] = error.ai_message.response_metadata.get("id")
        if isinstance(error, UnansweredModelCall) and response is not None:
            details.update(_unanswered_diagnosis(response))
        report_model_retry(
            error,
            role=self.role,
            attempt=attempt,
            max_retries=self.max_retries,
            retrying=retrying,
            adjusted=True,
            details=details,
            stream_writer=request.runtime.stream_writer,
        )

    @staticmethod
    def _retry_length_limited_request(
        request: ModelRequest[None],
    ) -> ModelRequest[None]:
        return request.override(
            messages=[
                *request.messages,
                HumanMessage(content=_answer_wording(request).too_long),
            ]
        )

    @staticmethod
    def _retry_structured_output_request(
        request: ModelRequest[None],
        error: StructuredOutputError,
    ) -> ModelRequest[None]:
        # Replaying an AI message whose tool calls have no results makes the
        # next request invalid, so carry back only a plain-text answer. An
        # answer that came as a tool call goes back inside the correction
        # instead, where it is text and needs no result to match it.
        rejected = error.ai_message
        replay = [rejected] if rejected.text.strip() and not rejected.tool_calls else []
        return request.override(
            messages=[
                *request.messages,
                *replay,
                HumanMessage(content=_correction_text(error, request)),
            ]
        )

    @staticmethod
    def _retry_unanswered_request(
        request: ModelRequest[None],
        response: ModelResponse[Any],
        error: UnansweredModelCall,
    ) -> ModelRequest[None]:
        if isinstance(error, TextWithoutToolCall):
            # Replayed so the correction has the turn it refers to.
            return request.override(
                messages=[
                    *request.messages,
                    *(m for m in response.result if isinstance(m, AIMessage)),
                    HumanMessage(
                        content=_CALL_A_TOOL.format(
                            submit=_answer_wording(request).submit
                        )
                    ),
                ]
            )
        if isinstance(error, MalformedToolCall):
            unparsed = _unparsed_calls(response)
            content = _REWRITE_CALL.format(
                names=", ".join(sorted({str(call.get("name")) for call in unparsed})),
                reason="; ".join(sorted({str(call.get("error")) for call in unparsed})),
            )
        else:
            content = (
                _CUT_OFF_CALL
                if _visible_output_tokens(response) >= _CUT_OFF_CALL_TOKENS
                else _THOUGHT_TOO_LONG
            )
        return request.override(
            messages=[*request.messages, HumanMessage(content=content)]
        )

    @override
    def wrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        current_request = request
        rejected = 0
        transport = ModelConnectionRetry(
            self.max_transport_retries, self.role, request.runtime.stream_writer
        )
        while True:
            try:
                response = transport.invoke(partial(handler, current_request))
            except (LengthFinishReasonError, StructuredOutputError) as error:
                retrying = rejected < self.max_retries
                self._report(current_request, rejected, error, retrying=retrying)
                if not retrying:
                    return _gave_up_answering(current_request, rejected + 1, error)
                rejected += 1
                current_request = (
                    self._retry_length_limited_request(current_request)
                    if isinstance(error, LengthFinishReasonError)
                    else self._retry_structured_output_request(current_request, error)
                )
            else:
                unanswered = _unanswered(current_request, response)
                if unanswered is None:
                    return response
                retrying = rejected < self.max_retries
                self._report(
                    current_request,
                    rejected,
                    unanswered,
                    retrying=retrying,
                    response=response,
                )
                if not retrying:
                    return _gave_up_answering(current_request, rejected + 1, unanswered)
                rejected += 1
                current_request = (
                    self._retry_length_limited_request(current_request)
                    if _answering_ran_out_of_output(current_request, response)
                    else self._retry_unanswered_request(
                        current_request, response, unanswered
                    )
                )

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        current_request = request
        rejected = 0
        transport = ModelConnectionRetry(
            self.max_transport_retries, self.role, request.runtime.stream_writer
        )
        while True:
            try:
                response = await transport.ainvoke(partial(handler, current_request))
            except (LengthFinishReasonError, StructuredOutputError) as error:
                retrying = rejected < self.max_retries
                self._report(current_request, rejected, error, retrying=retrying)
                if not retrying:
                    return _gave_up_answering(current_request, rejected + 1, error)
                rejected += 1
                current_request = (
                    self._retry_length_limited_request(current_request)
                    if isinstance(error, LengthFinishReasonError)
                    else self._retry_structured_output_request(current_request, error)
                )
            else:
                unanswered = _unanswered(current_request, response)
                if unanswered is None:
                    return response
                retrying = rejected < self.max_retries
                self._report(
                    current_request,
                    rejected,
                    unanswered,
                    retrying=retrying,
                    response=response,
                )
                if not retrying:
                    return _gave_up_answering(current_request, rejected + 1, unanswered)
                rejected += 1
                current_request = (
                    self._retry_length_limited_request(current_request)
                    if _answering_ran_out_of_output(current_request, response)
                    else self._retry_unanswered_request(
                        current_request, response, unanswered
                    )
                )


class _AnswerWording(NamedTuple):
    """How corrections ask for an answer, in the channel the strategy parses."""

    correct: str
    too_long: str
    asked_again: str
    submit: str


_JSON_WORDING = _AnswerWording(
    correct=(
        "return corrected raw JSON that matches the required schema, with no "
        "explanation or Markdown outside it"
    ),
    too_long=(
        "Your response reached the output-token limit and could not be parsed. "
        "Do not call tools. Return only concise raw JSON that matches the "
        "required schema, with no explanation, analysis, or Markdown outside "
        "it. Keep every string value short enough to complete the entire JSON "
        "object."
    ),
    asked_again="answer with the corrected JSON and nothing else",
    submit="return your answer as raw JSON",
)


def _answer_wording(request: ModelRequest[None]) -> _AnswerWording:
    """Under ToolStrategy the answer is a tool call; asking for JSON gets text."""
    if not isinstance(request.response_format, ToolStrategy):
        return _JSON_WORDING
    tool = " or ".join(spec.name for spec in request.response_format.schema_specs)
    return _AnswerWording(
        correct=f"call {tool} again with corrected arguments",
        too_long=(
            "Your answer reached the output-token limit and was cut off. Call "
            f"{tool} again with concise arguments, keeping every string value "
            "short enough to complete the call."
        ),
        asked_again=f"call {tool} with the corrected answer and nothing else",
        submit=f"call {tool} to submit your answer",
    )


_GAVE_UP = (
    "After {attempts} attempts your answer still could not be read as the "
    "required structured output. The last problem was: {error}. This stage is "
    "now unanswered; if you are asked again, {asked_again}."
)


def _gave_up_answering(
    request: ModelRequest[None], attempts: int, error: Exception
) -> ModelResponse[Any]:
    """End the turn with no answer, saying why.

    Plain text and no tool calls: this message is kept in the stage's
    transcript and replayed on the re-ask, and a tool call with no result
    would make that request invalid.
    """
    return ModelResponse(
        result=[
            AIMessage(
                content=_GAVE_UP.format(
                    attempts=attempts,
                    error=str(error)[:500],
                    asked_again=_answer_wording(request).asked_again,
                )
            )
        ],
        structured_response=None,
    )


_UNANSWERED = "the model returned no tool call, no text and no structured output"
_TEXT_WITHOUT_CALL = "the model returned text without a tool call"
_MALFORMED = "the arguments of {names} did not parse: {reason}"
_REWRITE_CALL = (
    "Your last call to {names} could not be read: its arguments were not valid "
    "JSON ({reason}), so nothing ran. Send the same call again with the "
    "arguments as one well-formed JSON object. The analysis you have already "
    "done is above; keep it."
)
_CALL_A_TOOL = (
    "Your last turn was text without a tool call, so nothing ran and no answer "
    "was submitted. Call a tool to keep working, or {submit}."
)
_THOUGHT_TOO_LONG = (
    "Your last turn spent its whole output budget on thinking and came back "
    "empty. You have been thinking a long time, so answer now. The analysis "
    "you have already done is above; build on it rather than starting over."
)
# Output beyond reasoning that long was a tool call the limit cut off.
_CUT_OFF_CALL_TOKENS = 1000
_CUT_OFF_CALL = (
    "Your last turn reached the output-token limit while writing a tool call, "
    "so the call was cut off and never ran; nothing was written. Keep each "
    "tool call short: write a large file in several smaller pieces, and think "
    "less before writing. The analysis you have already done is above."
)


def _visible_output_tokens(response: ModelResponse[Any]) -> int:
    last = response.result[-1] if response.result else None
    usage = getattr(last, "usage_metadata", None) or {}
    reasoning = (usage.get("output_token_details") or {}).get("reasoning") or 0
    return (usage.get("output_tokens") or 0) - reasoning


def _unparsed_calls(response: ModelResponse[Any]) -> list[dict[str, Any]]:
    """The calls whose arguments did not parse, which LangChain leaves here."""
    return [
        call
        for message in response.result
        for call in getattr(message, "invalid_tool_calls", None) or ()
    ]


def _unanswered(
    request: ModelRequest[None], response: ModelResponse[Any]
) -> UnansweredModelCall | None:
    """Why this response would end the agent loop without an answer, or None.

    Reasoning alone would, and so would text under ToolStrategy.
    """
    if response.structured_response is not None or any(
        getattr(message, "tool_calls", None) for message in response.result
    ):
        return None
    # Before the empty check: an unparsed call carries no text and no tool call,
    # so it reads as silence while the provider reports the turn as finished.
    if unparsed := _unparsed_calls(response):
        return MalformedToolCall(
            _MALFORMED.format(
                names=", ".join(sorted({str(call.get("name")) for call in unparsed})),
                reason="; ".join(sorted({str(call.get("error")) for call in unparsed})),
            )
        )
    if not any(message.text.strip() for message in response.result):
        return UnansweredModelCall(_UNANSWERED)
    if isinstance(request.response_format, ToolStrategy):
        return TextWithoutToolCall(_TEXT_WITHOUT_CALL)
    return None


def _extract_text_or_tool_args(message: AIMessage) -> str:
    """What the model actually answered with, whichever channel carried it.

    Under `ToolStrategy` the answer is a tool call and `text` is empty, so
    recording the text alone preserves nothing of a contract failure.
    """
    if message.tool_calls:
        return json.dumps(
            [
                {"name": call.get("name"), "args": call.get("args")}
                for call in message.tool_calls
            ],
            ensure_ascii=False,
            default=str,
        )
    return message.text


def _finish_reason(response: ModelResponse[Any]) -> str | None:
    last = response.result[-1] if response.result else None
    metadata = getattr(last, "response_metadata", None) or {}
    return metadata.get("finish_reason")


def _answering_ran_out_of_output(
    request: ModelRequest[None],
    response: ModelResponse[Any],
) -> bool:
    """Whether an answer, rather than a working turn, hit the output limit.

    The OpenAI client raises `LengthFinishReasonError` for a truncated answer,
    and the correction it earns -- call no tools, keep the JSON short enough to
    close -- is written for one. A backend reporting the same thing in
    `finish_reason` deserves it too, but only where it means the same thing: on
    a turn that still has tools the model was working, not answering, and
    telling it to stop calling them reads as an instruction to give up. One
    coding stage did exactly that, and submitted "not yet implemented" on its
    first turn.
    """
    return _finish_reason(response) == "length" and not request.tools


def _unanswered_diagnosis(response: ModelResponse[Any]) -> dict[str, object]:
    """Say what the backend reported about a turn that gave no answer.

    An empty completion is the same event whether the model ran out of output
    budget mid-thought, was cut off upstream, or simply stopped -- and the
    three want different fixes. The backend distinguishes them and this is the
    only place that sees its report.
    """
    last = response.result[-1] if response.result else None
    if last is None:
        return {}
    metadata = getattr(last, "response_metadata", None) or {}
    usage = getattr(last, "usage_metadata", None) or {}
    return {
        "finish_reason": _finish_reason(response),
        # OpenRouter streams leave `provider` null; GET /api/v1/generation?id= names it.
        "generation_id": metadata.get("id"),
        "output_tokens": usage.get("output_tokens"),
        "reasoning_tokens": (usage.get("output_token_details") or {}).get("reasoning"),
    }


def _correction_text(error: StructuredOutputError, request: ModelRequest[None]) -> str:
    """Tell the model what was wrong with its answer, in terms it can act on."""
    if isinstance(error, MultipleStructuredOutputsError):
        return (
            f"You returned {len(error.tool_names)} structured responses "
            f"({', '.join(error.tool_names)}) when exactly one is expected. "
            "Return one call that carries the whole answer."
        )
    if isinstance(error, StructuredOutputValidationError):
        return (
            f"Your previous response was not a valid {error.tool_name} "
            f"structured output. Validation error: {_validation_problem(error)}."
            + _rejected_arguments(error.ai_message, error.tool_name, error.source)
            + " You may continue using tools if you need more "
            "information. When you are ready to answer, "
            f"{_answer_wording(request).correct}."
        )
    return f"Your previous response was rejected: {error}. Answer again."


def _validation_problem(error: StructuredOutputValidationError) -> str:
    """Pydantic errors by JSON path; any other rejection as it was raised."""
    cause = error.source.__cause__
    answer = _call_arguments(error.ai_message, error.tool_name)
    if answer is None:
        try:
            answer = json.loads(error.ai_message.text)
        except ValueError:
            pass
    if not isinstance(cause, ValidationError) or answer is None:
        return str(error.source)
    return "\n".join(answer_errors(cause, answer))


def _call_arguments(message: AIMessage, tool_name: str) -> object:
    """The arguments of the call that carried the answer, if it came as one."""
    return next(
        (call["args"] for call in message.tool_calls if call["name"] == tool_name),
        None,
    )


# What a rejected answer may spend on being shown back to its author. A
# hypothesis runs to tens of thousands of characters, and replaying one whole
# would cost more than the turn it is meant to save.
_REJECTED_BUDGET = 1200


def _rejected_arguments(
    message: AIMessage, tool_name: str, source: BaseException | None = None
) -> str:
    """The answer the model sent, when the model cannot otherwise see it.

    A tool-call answer is never replayed as a message -- a tool call with no
    result invalidates the next request -- so without this the model is told
    only that its answer was wrong. It sent `rationale: ""` six times running
    against a validator that said what the rule was but not which member broke
    it, and nothing in the exchange could have told it which of the three it
    had filled.

    Large arguments come back as a shape, keeping in full only the entries the
    error names: which member was wrong is what a rejected answer has to show,
    and a whole report restated is the same information at fifty times the
    price -- one auditor, shown a shape alone, rewrote its report from memory
    and broke a different part of it each time.
    """
    arguments = _call_arguments(message, tool_name)
    if not isinstance(arguments, dict):
        return ""

    rendered = json.dumps(arguments, ensure_ascii=False, default=str)
    if len(rendered) > _REJECTED_BUDGET:
        cited = _cited_entries(source)
        rendered = json.dumps(
            {
                key: _outline(value, {index for name, index in cited if name == key})
                for key, value in arguments.items()
            },
            ensure_ascii=False,
            default=str,
        )
    return f" You sent: {rendered}."


def _cited_entries(source: BaseException | None) -> set[tuple[str, int]]:
    """The list entries the validation error names, such as ("findings", 4)."""
    cause = source.__cause__ if source is not None else None
    if not isinstance(cause, ValidationError):
        return set()
    return {
        (str(loc[0]), loc[1])
        for error in cause.errors()
        if len(loc := error["loc"]) > 1 and isinstance(loc[1], int)
    }


def _outline(value: Any, shown: Collection[int] = ()) -> Any:
    """One member of a rejected answer, keeping in full the entries the error names."""
    if isinstance(value, str) and len(value) > 80:
        return f"<{len(value)} characters>"
    if isinstance(value, list) and shown:
        # The model rewrote whole reports from memory when it could not see the entry.
        return [item if i in shown else _outline(item) for i, item in enumerate(value)]
    if isinstance(value, list):
        return f"<{len(value)} entries>" if value else []
    if isinstance(value, dict):
        return f"<{len(value)} keys>" if value else {}
    return value
