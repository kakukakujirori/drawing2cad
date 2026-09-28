"""Report what a turn's writes produced, and gate unverified answers."""

from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol, override

from langchain.agents import AgentState as _AgentState
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.content import ContentBlock, create_text_block
from langgraph.runtime import Runtime

_UNREAD = (
    "Answer refused: the latest verification report has not been shown to you "
    "yet. Read it below, then answer again."
)
_ANSWER_WITH_TOOLS_REFUSED = (
    "Answer refused: you called other tools in the same turn as the answer. "
    "Read what they return, then answer in a turn with no other tool calls."
)


class ArtifactVerifier(Protocol):
    """The file to watch and the verification to run after it changes.

    Structural so this module never imports a CAD kernel, and a test can answer
    it with a counter.
    """

    @property
    def source_path(self) -> Path: ...

    @property
    def blockers(self) -> list[str]:
        """Why the file as it is now cannot be the answer, by its latest verification.

        Read only when the model answers; empty once the file can be the answer.
        """
        ...

    def feedback(self) -> list[ContentBlock]: ...


class AnswerVerifier(Protocol):
    """Why the stage's answer contradicts its round; nothing when it stands."""

    def feedback(self, answer: Any, /) -> list[ContentBlock]: ...


class VerifyOnWriteMiddleware(AgentMiddleware[_AgentState[Any], None, Any]):
    """Verify a watched artifact whenever a turn changed it.

    Placed on the path from the tools node back to the model, so a turn that
    rewrote the file four times in parallel is built once, on the state it
    ended in.

    It also gates the stage's answer. A model whose answer schema is bound as a
    tool can call it the way it calls any other tool, part-way through the work;
    only a build standing between that call and the end of the stage stops a
    unverified artifact becoming the stage's result. What either verifier
    reports is handed back to the model; this middleware judges nothing itself.
    """

    def __init__(
        self,
        verifier: ArtifactVerifier,
        *,
        ticket_verifier: AnswerVerifier | None = None,
        require_feedback_before_submit: bool = False,
        fingerprint: Callable[[], str | None] | None = None,
    ) -> None:
        super().__init__()
        self.verifier = verifier
        self.ticket_verifier = ticket_verifier
        self.fingerprint = fingerprint
        self.require_feedback_before_submit = require_feedback_before_submit
        # What was on disk at construction is not this agent's work, so
        # `before_model` stays quiet about it. The gate keeps its own mark,
        # because a program nobody built must never pass for one that builds.
        self._last_seen = self._digest()
        self._last_built: str | None = None
        self._last_report: list[ContentBlock] = []

    def reset(self) -> None:
        """Accept the current source as a machine-provided, unbuilt baseline."""
        self._last_seen = self._digest()
        self._last_built = None
        self._last_report = []

    def _digest(self) -> str | None:
        if self.fingerprint is not None:
            return self.fingerprint()
        # By content, not timestamp: the agent reads the program far more often
        # than it writes it, and a build must not follow a `cat`.
        path = self.verifier.source_path
        return sha256(path.read_bytes()).hexdigest() if path.is_file() else None

    def _build(self) -> list[ContentBlock]:
        self._last_report = self.verifier.feedback()
        # Validation may fill derived fields in the watched artifact.
        self._last_seen = self._last_built = self._digest()
        return self._last_report

    @override
    def before_model(
        self, state: _AgentState[Any], runtime: Runtime[None]
    ) -> dict[str, Any] | None:
        del state, runtime
        if self._digest() == self._last_seen:
            return None
        return {"messages": [HumanMessage(content_blocks=self._build())]}

    @override
    def wrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Any,
    ) -> ModelResponse[Any]:
        response = handler(request)
        if response.structured_response is None:
            return response
        if _has_unrun_calls(response):
            # Unrun tools may still change the artifact. A HumanMessage here
            # would split their results, so only the acknowledgement changes.
            return _refused(response, _ANSWER_WITH_TOOLS_REFUSED)

        reasons = self._artifact_reasons()
        # Integration checks a text answer's tickets, and re-asks the stage.
        if self.ticket_verifier is not None and _answered_by_tool(response):
            reasons += self.ticket_verifier.feedback(response.structured_response)
        if not reasons:
            return response

        refused = _refused(response, "Submission refused; see the message below.")
        return ModelResponse(
            result=[*refused.result, HumanMessage(content_blocks=reasons)],
            structured_response=None,
        )

    def _artifact_reasons(self) -> list[ContentBlock]:
        """Why the artifact is not ready to submit; nothing when it is.

        The reasons come first. A report the model already read is not repeated,
        so a repeated answer costs one short message, and the reason stays in view.
        """
        # A build of exactly this content already stands, and the model read it.
        reported = self._digest() == self._last_built
        report = self._last_report if reported else self._build()
        if not self.verifier.blockers and (
            reported or not self.require_feedback_before_submit
        ):
            return []
        # A missing file matches "never built" (both None) but has no report yet.
        if not report:
            report, reported = self._build(), False
        if not (blockers := self.verifier.blockers):
            # It passes, but was first built just now, so its report is unseen;
            # e.g. a file kept unchanged from the last round.
            return [create_text_block(_UNREAD), *report]
        name = self.verifier.source_path.name
        refusal = create_text_block(
            f"Answer refused. It stays refused until {name} changes:\n"
            + "".join(f"- {blocker}\n" for blocker in blockers)
            + f"Correct {name} based on its latest verification report."
        )
        return [refusal] if reported else [refusal, *report]


def _has_unrun_calls(response: ModelResponse[Any]) -> bool:
    """Whether the answer came with tool calls that have not run yet."""
    answered = {
        message.tool_call_id
        for message in response.result
        if isinstance(message, ToolMessage)
    }
    return any(
        call["id"] not in answered
        for message in response.result
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    )


def _answered_by_tool(response: ModelResponse[Any]) -> bool:
    """Whether the answer came as a tool call rather than as text."""
    return any(isinstance(message, ToolMessage) for message in response.result)


def _refused(response: ModelResponse[Any], text: str) -> ModelResponse[Any]:
    """Replace the answer's acknowledgement so no message says it was received.

    LangChain's tools-to-model edge ends on a structured tool's name, even
    without structured_response, so the name is cleared too.
    """
    return ModelResponse(
        result=[
            message.model_copy(
                update={"content": text, "status": "error", "name": None}
            )
            if isinstance(message, ToolMessage)
            else message
            for message in response.result
        ],
        structured_response=None,
    )
