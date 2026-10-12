"""Question an audit answer that keeps a dismissal the input drawing does not support."""

from typing import Any, override

from langchain.agents import AgentState as _AgentState
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage
from langchain_core.messages.content import ContentBlock, create_text_block

from zeroshot.pipeline.stages.audit.verify import AuditVerifier, dismissal_check_text
from zeroshot.pipeline.workflow.middleware.verify_on_write import refused_answer

_REFUSED_ONCE = (
    "Answer refused once: audit.json keeps the dismissals below. Check each one "
    "against the input drawing as the dismissal check says, and correct "
    "audit.json if it needs it. Then answer again; these dismissals do not "
    "stop an answer a second time."
)


class DismissalCheckMiddleware(AgentMiddleware[_AgentState[Any], None, Any]):
    """Refuse an answer once for each dismissal it keeps that was not yet questioned.

    Placed outside VerifyOnWriteMiddleware, so it sees only answers whose
    audit.json was verified and reported.
    """

    def __init__(self, verifier: AuditVerifier) -> None:
        super().__init__()
        self.verifier = verifier
        self.reset()

    def reset(self) -> None:
        self._questioned: set[str] = set()

    @override
    def wrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Any,
    ) -> ModelResponse[Any]:
        response = handler(request)
        if response.structured_response is None:
            return response
        new = [
            key for key in self.verifier.dismissed_extras if key not in self._questioned
        ]
        if not new:
            return response
        self._questioned.update(new)
        refused = refused_answer(response, "Submission refused; see the message below.")
        reasons: list[ContentBlock] = [
            create_text_block(_REFUSED_ONCE),
            create_text_block(dismissal_check_text(new)),
        ]
        return ModelResponse(
            result=[*refused.result, HumanMessage(content_blocks=reasons)],
            structured_response=None,
        )
