import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError

from zeroshot.pipeline.tools import VerifyOutputResult
from zeroshot.pipeline.workflow import CUSTOM_STATE_TYPES, SemanticHypothesis
from zeroshot.pipeline.workflow.state import (
    OperationPlan,
    ReconstructionState,
    Review,
    StopReason,
)


@pytest.mark.parametrize("contract", [SemanticHypothesis, OperationPlan, Review])
def test_a_contract_carries_no_prose_beyond_its_field_descriptions(
    contract: type[BaseModel],
) -> None:
    """A class docstring becomes the schema's `description`, and the schema is
    sent to the model twice over -- as `$output_schema` in the prompt and as the
    provider's own output contract. Only `Field(description=...)` is written to
    be read by a model, so nothing else may end up there."""
    schema = contract.model_json_schema()

    assert "description" not in schema
    assert set(schema["properties"]) == set(contract.model_fields)


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
        Review.model_validate(payload)


@pytest.mark.parametrize("feedback", ["", "   "])
def test_semantic_hypothesis_review_requires_feedback_for_revision(
    feedback: str,
) -> None:
    with pytest.raises(ValidationError, match="feedback"):
        Review(decision="revise", feedback=feedback)


def test_semantic_hypothesis_review_allows_an_accept_without_feedback() -> None:
    review = Review(decision="accept", feedback="")

    assert review.decision == "accept"


_ARTIFACTS: dict[str, object] = {
    "semantic_hypothesis": SemanticHypothesis(semantics=["flange", "blind hole"]),
    "operation_plan": OperationPlan(operations=["Extrude the outline 25 mm along +z"]),
    "stop_reason": StopReason.BUDGET_EXHAUSTED,
    "last_verification": VerifyOutputResult(
        verification_id="v1",
        status="SUCCEEDED",
        source="result = cq.Workplane()",
        returncode=0,
    ),
}


def test_every_state_artifact_survives_a_checkpoint() -> None:
    """A class the checkpointer was not told about loads back as a plain dict,
    which still passes the `is not None` checks the graph routes on."""
    assert {type(value) for value in _ARTIFACTS.values()} == set(CUSTOM_STATE_TYPES)

    def store(_: ReconstructionState) -> dict[str, object]:
        return dict(_ARTIFACTS)

    workflow = StateGraph(state_schema=ReconstructionState)  # type: ignore[type-var]
    workflow.add_node("store", store)
    workflow.add_edge(START, "store")
    workflow.add_edge("store", END)

    serde = JsonPlusSerializer(allowed_msgpack_modules=list(CUSTOM_STATE_TYPES))
    graph = workflow.compile(checkpointer=InMemorySaver(serde=serde))
    config = {"configurable": {"thread_id": "artifact-round-trip"}}

    graph.invoke(ReconstructionState(messages=[]), config)
    restored = graph.get_state(config).values

    for field, value in _ARTIFACTS.items():
        assert type(restored[field]) is type(value), field
        assert restored[field] == value
