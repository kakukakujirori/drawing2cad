from typing import cast

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from zeroshot.pipeline.workflow import SemanticHypothesis
from zeroshot.pipeline.workflow.state import (
    ReconstructionState,
    SemanticHypothesisReview,
)


def test_semantic_hypothesis_validates_json() -> None:
    hypothesis = SemanticHypothesis.model_validate_json(
        '{"semantics":["cylindrical boss","through hole"]}'
    )

    assert hypothesis.semantics == ["cylindrical boss", "through hole"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"semantics": ["hole"], "evidence": "visible in the front view"},
    ],
)
def test_semantic_hypothesis_rejects_schema_violations(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        SemanticHypothesis.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"decision": "reject", "feedback": "wrong feature"},
        {"decision": "accept"},
        {"decision": "accept", "feedback": "", "confidence": 0.9},
    ],
)
def test_semantic_hypothesis_review_rejects_schema_violations(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        SemanticHypothesisReview.model_validate(payload)


@pytest.mark.parametrize("feedback", ["", "   "])
def test_semantic_hypothesis_review_requires_feedback_for_revision(
    feedback: str,
) -> None:
    with pytest.raises(ValidationError, match="feedback"):
        SemanticHypothesisReview(decision="revise", feedback=feedback)


def test_semantic_hypothesis_review_allows_an_accept_without_feedback() -> None:
    review = SemanticHypothesisReview(decision="accept", feedback="")

    assert review.decision == "accept"


def test_reconstruction_state_checkpoints_a_semantic_hypothesis() -> None:
    hypothesis = SemanticHypothesis(semantics=["flange", "blind hole"])

    def store_hypothesis(_: ReconstructionState) -> dict[str, SemanticHypothesis]:
        return {"semantic_hypothesis": hypothesis}

    workflow = StateGraph(state_schema=ReconstructionState)  # type: ignore[type-var]
    workflow.add_node("store_hypothesis", store_hypothesis)
    workflow.add_edge(START, "store_hypothesis")
    workflow.add_edge("store_hypothesis", END)

    serde = JsonPlusSerializer(
        allowed_msgpack_modules=(
            ("zeroshot.pipeline.workflow.state", "SemanticHypothesis"),
        )
    )
    graph = workflow.compile(checkpointer=InMemorySaver(serde=serde))
    config = {"configurable": {"thread_id": "semantic-hypothesis-round-trip"}}

    result = graph.invoke(ReconstructionState(messages=[]), config)
    restored = graph.get_state(config).values["semantic_hypothesis"]

    assert result["semantic_hypothesis"] == hypothesis
    assert isinstance(restored, SemanticHypothesis)
    assert cast(SemanticHypothesis, restored) == hypothesis
