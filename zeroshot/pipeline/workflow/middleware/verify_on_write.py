"""Report what a turn's writes built, without the agent having to ask."""

from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol, override

from langchain.agents import AgentState as _AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.messages.content import ContentBlock
from langgraph.runtime import Runtime


class ProgramVerifier(Protocol):
    """The file to watch, whose writes were the machine's, and the act.

    Structural so this module never imports a CAD kernel, and a test can answer
    it with a counter.
    """

    @property
    def source_path(self) -> Path: ...

    @property
    def own_write(self) -> str | None: ...

    def feedback(self) -> list[ContentBlock]: ...


class VerifyOnWriteMiddleware(AgentMiddleware[_AgentState[Any], None, Any]):
    """Build the program whenever a turn changed it, and report back for free.

    Placed on the path from the tools node back to the model, so a turn that
    rewrote the file four times in parallel is built once, on the state it
    ended in.
    """

    def __init__(self, verifier: ProgramVerifier) -> None:
        super().__init__()
        self.verifier = verifier
        self._last_seen = self._digest()

    def _digest(self) -> str | None:
        # By content, not timestamp: the agent reads the program far more often
        # than it writes it, and a build must not follow a `cat`.
        path = self.verifier.source_path
        return sha256(path.read_bytes()).hexdigest() if path.is_file() else None

    @override
    def before_model(
        self, state: _AgentState[Any], runtime: Runtime[None]
    ) -> dict[str, Any] | None:
        del state, runtime
        digest = self._digest()
        if digest == self._last_seen:
            return None
        self._last_seen = digest
        # At the first turn of each coding stage, the output file doesn't contain
        # any new code. Therefore, the verifier doesn't have anything to report.
        if digest is not None and digest == self.verifier.own_write:
            return None
        return {"messages": [HumanMessage(content_blocks=self.verifier.feedback())]}
