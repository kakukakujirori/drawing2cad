from typing import cast

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError

from tests.zeroshot.contracts import feature, geometry, hypothesis
from zeroshot.pipeline.messages.contracts import (
    Axis,
    ClaimSource,
    DrawnEntity,
    EdgeStyle,
    FeatureGeometry,
    GeometryKind,
    Operation,
    OperationPlan,
    OperationVerb,
    Parameter,
    ParameterName,
    PlanReview,
    SemanticFeature,
    SemanticHypothesis,
    View,
    ViewEvidence,
    fingerprint,
)
from zeroshot.pipeline.tools import VerifyOutputResult
from zeroshot.pipeline.workflow import (
    CUSTOM_STATE_TYPES,
    Proposal,
)
from zeroshot.pipeline.workflow.components.agent import StopReason
from zeroshot.pipeline.workflow.components.proposer_reviewer import Review
from zeroshot.pipeline.workflow.state import (
    Audit,
    ReconstructionState,
    carry_thread,
    lead_transcript,
)


@pytest.mark.parametrize(
    "contract",
    [
        Proposal,
        SemanticHypothesis,
        SemanticFeature,
        FeatureGeometry,
        Parameter,
        ViewEvidence,
        Review,
        Audit,
    ],
)
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


# A plane is measured by nothing, so a hypothesis of bare planes never builds a
# `Dimension` and the checkpoint check below would pass without covering it.
_A_HYPOTHESIS = hypothesis(
    proposal=[
        feature(1, "flange"),
        feature(2, "blind hole", geometry=[geometry("torus")]),
    ]
)

_A_PLAN = OperationPlan(
    proposal=[
        Operation(
            name="op_base",
            verb=OperationVerb.EXTRUDE,
            detail="Extrude the outline 25 mm along +z",
            depends_on=[],
            semantics=[1],
        )
    ],
    rationale="one extrude reaches the stated height",
)

_ARTIFACTS: dict[str, object] = {
    "semantics_state": {
        "invocation_instruction": None,
        "branch_proposals": {"propose_0": _A_HYPOTHESIS},
        "proposal": _A_HYPOTHESIS,
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
        "proposal": _A_PLAN,
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
    "semantic_hypothesis": _A_HYPOTHESIS,
    "operation_plan": _A_PLAN,
    "plan_review": PlanReview(
        uncovered=[2],
        unknown=[],
        transcribed={},
        unresolved={},
        of_hypothesis=fingerprint(_A_HYPOTHESIS),
        of_plan=fingerprint(_A_PLAN),
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
        Operation,
        OperationPlan,
        OperationVerb,
        PlanReview,
        SemanticHypothesis,
        SemanticFeature,
        FeatureGeometry,
        Parameter,
        ParameterName,
        Axis,
        ViewEvidence,
        # The contract's enums ride in state too. An enum missing from the
        # allowlist restores as a bare string, which still compares equal and
        # so fails nowhere until something asks it for `.value`.
        View,
        DrawnEntity,
        EdgeStyle,
        GeometryKind,
        ClaimSource,
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
        elif isinstance(value, list | tuple):
            for held in value:
                yield from types(held)
        elif isinstance(value, BaseModel):
            yield type(value)
            for held in dict(value).values():
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
    assert type(semantics_state["proposal"]) is SemanticHypothesis
    assert type(semantics_state["branch_proposals"]["propose_0"]) is SemanticHypothesis
    assert type(semantics_state["reducer_state"]["stop_reason"]) is StopReason

    operations_state = restored["operations_state"]
    assert type(operations_state["proposal"]) is OperationPlan
    assert type(operations_state["review"]) is Review


def _threaded_state(**stages: object) -> ReconstructionState:
    """A run part-way through, each stage holding what is its own."""
    read = [HumanMessage(content="read the views")]
    state: dict[str, object] = {
        "semantics_state": {
            "branch_proposals": {"propose_0": None},
            "reducer_state": {"messages": read},
        },
        "operations_state": {
            "revision_count": 2,
            "proposer_state": {"messages": read},
        },
        "coding_state": {"current_turn": 4},
        "audit_state": {"messages": [HumanMessage(content="judge it")]},
    }
    return cast(ReconstructionState, state | stages)


@pytest.mark.parametrize(
    ("stage", "stage_state"),
    [
        ("semantics", {"semantics_state": {"reducer_state": {"messages": ["it"]}}}),
        ("operations", {"operations_state": {"proposer_state": {"messages": ["it"]}}}),
        ("coding", {"coding_state": {"messages": ["it"]}}),
    ],
)
def test_the_thread_is_taken_from_whichever_agent_carried_it(
    stage: str, stage_state: dict[str, object]
) -> None:
    """Each template continues a different one of its agents, so where the
    finished thread sits differs by stage."""
    state = _threaded_state(**stage_state)

    update = carry_thread(state, lead_transcript(state, stage))  # type: ignore[arg-type]

    assert update["coding_state"]["messages"] == ["it"]


def test_the_thread_reaches_every_reasoning_stage_but_not_the_audit() -> None:
    """The audit judges from the outside; a thread it took part in would leave
    it marking its own work."""
    state = _threaded_state(coding_state={"messages": ["wrote the model"]})

    update = carry_thread(state, lead_transcript(state, "coding"))

    assert set(update) == {"semantics_state", "operations_state", "coding_state"}
    assert update["semantics_state"]["reducer_state"]["messages"] == ["wrote the model"]
    assert update["operations_state"]["proposer_state"]["messages"] == [
        "wrote the model"
    ]


def test_the_stage_that_wrote_the_thread_is_given_back_what_it_wrote() -> None:
    """Handing it its own transcript changes nothing, and costs one special
    case less than leaving it out."""
    state = _threaded_state(coding_state={"messages": ["wrote the model"]})

    update = carry_thread(state, lead_transcript(state, "coding"))

    assert update["coding_state"]["messages"] == ["wrote the model"]


def test_what_a_stage_holds_besides_its_messages_survives_the_thread() -> None:
    """Branch proposals and revision counts belong to their stage, not to the
    thread that happens to be passing through it."""
    update = carry_thread(
        _threaded_state(), lead_transcript(_threaded_state(), "coding")
    )

    assert update["semantics_state"]["branch_proposals"] == {"propose_0": None}
    assert update["operations_state"]["revision_count"] == 2


def test_the_prompt_log_is_told_where_the_inherited_thread_ends() -> None:
    """Without the watermark a stage reports the transcript it was handed as
    the prompt it was given."""
    state = _threaded_state(
        semantics_state={"reducer_state": {"messages": ["one", "two"]}}
    )

    update = carry_thread(state, lead_transcript(state, "semantics"))

    assert update["coding_state"]["reported_message_count"] == 2
    assert update["coding_state"]["current_turn"] == 4


def test_a_stage_that_has_not_run_is_seeded_all_the_same() -> None:
    update = carry_thread(ReconstructionState(), [])

    assert update["semantics_state"] == {
        "reducer_state": {"messages": [], "reported_message_count": 0}
    }
