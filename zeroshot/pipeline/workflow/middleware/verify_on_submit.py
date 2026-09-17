"""Check a structured answer against the invocation context before it is committed."""

from collections.abc import Callable
from typing import Any, override

from langchain.agents import AgentState as _AgentState
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.structured_output import StructuredOutputValidationError
from langchain_core.messages import AIMessage

from zeroshot.pipeline.stages._base.validate import SubmissionValidationError


class VerifyOnSubmitMiddleware(AgentMiddleware[_AgentState[Any], Any, Any]):
    """Hand an answer that contradicts the context back to the structured-output retry.

    Neither the rejected answer nor its acknowledgement enters the transcript.
    """

    def __init__(self, validate: Callable[[Any, Any], None]) -> None:
        super().__init__()
        self.validate = validate

    @override
    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        context = request.runtime.context
        if context is None:
            raise RuntimeError(
                "invoke the agent with the context its answer is checked against"
            )
        response = handler(request)
        answer = response.structured_response
        if answer is None:
            return response
        try:
            self.validate(answer, context)
        except SubmissionValidationError as error:
            message = next(m for m in response.result if isinstance(m, AIMessage))
            raise StructuredOutputValidationError(
                type(answer).__name__, error, message
            ) from error
        return response
