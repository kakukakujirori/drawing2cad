"""Report what a turn's writes produced, and gate unverified answers."""

from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol, override

from langchain.agents import AgentState as _AgentState
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage
from langchain_core.messages.content import ContentBlock, create_text_block
from langgraph.runtime import Runtime

_SUBMISSION_REFUSED = (
    "This artifact has not passed verification, so it is not ready to submit. "
    "Submitting ends the stage and hands the current artifact on as the answer. "
    "Correct it and trigger verification again; answer only once the current "
    "artifact is confirmed."
)


class ArtifactVerifier(Protocol):
    """The file to watch and the verification to run after it changes.

    Structural so this module never imports a CAD kernel, and a test can answer
    it with a counter.
    """

    @property
    def source_path(self) -> Path: ...

    @property
    def confirmed(self) -> bool: ...

    def feedback(self) -> list[ContentBlock]: ...


class VerifyOnWriteMiddleware(AgentMiddleware[_AgentState[Any], None, Any]):
    """Verify a watched artifact whenever a turn changed it.

    Placed on the path from the tools node back to the model, so a turn that
    rewrote the file four times in parallel is built once, on the state it
    ended in.

    It also gates the stage's answer. A model whose answer schema is bound as a
    tool can call it the way it calls any other tool, part-way through the work;
    only a build standing between that call and the end of the stage stops a
    unverified artifact becoming the stage's result.
    """

    def __init__(
        self,
        verifier: ArtifactVerifier,
        *,
        refusal: str = _SUBMISSION_REFUSED,
        require_feedback_before_submit: bool = False,
    ) -> None:
        super().__init__()
        self.verifier = verifier
        self.refusal = refusal
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
        # By content, not timestamp: the agent reads the program far more often
        # than it writes it, and a build must not follow a `cat`.
        path = self.verifier.source_path
        return sha256(path.read_bytes()).hexdigest() if path.is_file() else None

    def _build(self) -> list[ContentBlock]:
        self._last_seen = self._last_built = self._digest()
        self._last_report = self.verifier.feedback()
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

        # Build the exact content being submitted, unless that is already the
        # build standing: an answer may arrive in the same turn as a write.
        # Either way the refusal repeats the build's own output, so what failed
        # is stated with the refusal rather than a turn behind it.
        built_before_call = self._digest() == self._last_built
        if built_before_call:
            blocks = self._last_report
        else:
            blocks = self._build()
        if self.verifier.confirmed and (
            built_before_call or not self.require_feedback_before_submit
        ):
            return response

        return ModelResponse(
            result=[
                *response.result,
                HumanMessage(content_blocks=[*blocks, create_text_block(self.refusal)]),
            ],
            structured_response=None,
        )
