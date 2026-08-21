import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError

from zeroshot.pipeline.tools import VerifyOutputResult
from zeroshot.pipeline.workflow import (
    CUSTOM_STATE_TYPES,
    FanoutReduceProposal,
    Proposal,
)
from zeroshot.pipeline.workflow.agent import StopReason
from zeroshot.pipeline.workflow.proposer_reviewer import Review
from zeroshot.pipeline.workflow.state import Audit, ReconstructionState


@pytest.mark.parametrize("contract", [Proposal, FanoutReduceProposal, Review, Audit])
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


def test_a_proposal_validates_json() -> None:
    proposal = Proposal.model_validate_json(
        '{"proposal":["cylindrical boss","through hole"],"rationale":"both are turned"}'
    )

    assert proposal.proposal == ["cylindrical boss", "through hole"]
    assert proposal.rationale == "both are turned"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"proposal": ["hole"]},
        {"rationale": "visible in the front view"},
        {
            "proposal": ["hole"],
            "rationale": "visible in the front view",
            "evidence": "front view",
        },
    ],
)
def test_a_proposal_rejects_schema_violations(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Proposal.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"accept": False},
        {"rationale": "wrong feature"},
        {"accept": True, "rationale": "correct", "confidence": 0.9},
    ],
)
def test_a_review_rejects_schema_violations(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Review.model_validate(payload)


@pytest.mark.parametrize("rationale", ["", "   "])
def test_a_review_requires_rationale_for_revision(rationale: str) -> None:
    with pytest.raises(ValidationError, match="rationale"):
        Review(accept=False, rationale=rationale)


def test_a_review_allows_an_accept_without_rationale() -> None:
    review = Review(accept=True, rationale="")

    assert review.accept is True


_ARTIFACTS: dict[str, object] = {
    "semantics_state": {
        "invocation_instruction": None,
        "branch_proposals": {
            "propose_0": FanoutReduceProposal(
                proposal=["flange", "blind hole"],
                rationale="the flange carries the bolt circle",
            )
        },
        "proposal": FanoutReduceProposal(
            proposal=["flange", "blind hole"],
            rationale="the flange carries the bolt circle",
        ),
        "reducer_state": {
            "messages": [HumanMessage(content="propose semantics")],
            "current_turn": 1,
            "total_turns": 1,
            "stop_reason": StopReason.COMPLETED,
        },
    },
    "operations_state": {
        "revision_count": 1,
        "proposer_entry_instruction": HumanMessage(content="propose operations"),
        "proposal": Proposal(
            proposal=["Extrude the outline 25 mm along +z"],
            rationale="one extrude reaches the stated height",
        ),
        "proposer_state": {
            "messages": [HumanMessage(content="propose operations")],
            "current_turn": 1,
            "total_turns": 1,
            "stop_reason": StopReason.COMPLETED,
        },
        "reviewer_entry_instruction": HumanMessage(content="review semantics"),
        "review": Review(accept=True, rationale="the views agree"),
        "reviewer_state": {
            "messages": [HumanMessage(content="review semantics")],
            "current_turn": 1,
            "total_turns": 1,
            "stop_reason": StopReason.COMPLETED,
        },
    },
    "coding_state": {
        "messages": [HumanMessage(content="write code")],
        "current_turn": 3,
        "total_turns": 3,
        "stop_reason": StopReason.BUDGET_EXHAUSTED,
    },
    "semantic_hypothesis": FanoutReduceProposal(
        proposal=["flange", "blind hole"],
        rationale="the flange carries the bolt circle",
    ),
    "operation_plan": Proposal(
        proposal=["Extrude the outline 25 mm along +z"],
        rationale="one extrude reaches the stated height",
    ),
    "audit": Audit(revise="operations", rationale="the boss is missing"),
    "last_verification": VerifyOutputResult(
        verification_id="v1",
        status="SUCCEEDED",
        source="result = cq.Workplane()",
        returncode=0,
    ),
}


def test_custom_state_types_include_nested_runtime_values() -> None:
    assert set(CUSTOM_STATE_TYPES) == {
        Proposal,
        FanoutReduceProposal,
        Review,
        Audit,
        StopReason,
        VerifyOutputResult,
    }


def test_every_state_artifact_survives_a_checkpoint() -> None:
    """A class the checkpointer was not told about loads back as a plain dict,
    which still passes the `is not None` checks the graph routes on."""

    def types(value: object):
        if isinstance(value, dict):
            for held in value.values():
                yield from types(held)
        else:
            yield type(value)

    assert {found for value in _ARTIFACTS.values() for found in types(value)} >= set(
        CUSTOM_STATE_TYPES
    )

    def store(_: ReconstructionState) -> dict[str, object]:
        return dict(_ARTIFACTS)

    workflow = StateGraph(state_schema=ReconstructionState)  # type: ignore[type-var]
    workflow.add_node("store", store)
    workflow.add_edge(START, "store")
    workflow.add_edge("store", END)

    serde = JsonPlusSerializer(allowed_msgpack_modules=list(CUSTOM_STATE_TYPES))
    graph = workflow.compile(checkpointer=InMemorySaver(serde=serde))
    config = {"configurable": {"thread_id": "artifact-round-trip"}}

    graph.invoke(ReconstructionState(), config)
    restored = graph.get_state(config).values

    for field, value in _ARTIFACTS.items():
        assert type(restored[field]) is type(value), field
        assert restored[field] == value

    semantics_state = restored["semantics_state"]
    assert type(semantics_state["proposal"]) is FanoutReduceProposal
    assert type(semantics_state["branch_proposals"]["propose_0"]) is (
        FanoutReduceProposal
    )
    assert type(semantics_state["reducer_state"]["stop_reason"]) is StopReason

    operations_state = restored["operations_state"]
    assert type(operations_state["proposal"]) is Proposal
    assert type(operations_state["review"]) is Review
