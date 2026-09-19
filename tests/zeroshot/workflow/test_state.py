from dataclasses import fields, is_dataclass
from typing import cast

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from tests.zeroshot.contracts import interpretation, view
from zeroshot.pipeline.stages.audit.contracts import (
    AuditFinding,
    AuditReport,
    CausalHop,
    ConcernReview,
    RevisionRequest,
    StageOutputRef,
    TicketReview,
)
from zeroshot.pipeline.stages.coding.verify import VerifyOutputResult
from zeroshot.pipeline.stages.contracts import (
    ReconstructionHistory,
    ReconstructionSnapshot,
)
from zeroshot.pipeline.stages.interpretation.contracts import (
    Dimension as InterpretedDimension,
)
from zeroshot.pipeline.stages.interpretation.contracts import (
    DrawingInterpretation,
    DrawingView,
    Region,
    SemanticFeature,
    View,
)
from zeroshot.pipeline.stages.interpretation.contracts import (
    View as InterpretedView,
)
from zeroshot.pipeline.stages.operations.contracts import (
    Operation,
    OperationPlan,
    OperationVerb,
)
from zeroshot.pipeline.stages.tickets.contracts import (
    BootstrapWork,
    StageReport,
    Ticket,
    TicketAnswers,
    TicketResponse,
)
from zeroshot.pipeline.stages.types import PipelineStage, ReasoningStage
from zeroshot.pipeline.verification import ExecutionStatus
from zeroshot.pipeline.workflow import (
    CUSTOM_STATE_TYPES,
)
from zeroshot.pipeline.workflow.components.agent import StopReason
from zeroshot.pipeline.workflow.state import (
    ReconstructionState,
    carry_thread,
    lead_transcript,
)


@pytest.mark.parametrize(
    "contract",
    [
        DrawingInterpretation,
        SemanticFeature,
        DrawingView,
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


_A_INTERPRETATION = interpretation("flange", "blind hole")
_A_INTERPRETATION.views[0].dimensions = [
    InterpretedDimension(
        name="dim_bore",
        kind="diameter",
        text="5",
        nominal_value=5.0,
        measured_length=10.0,
        region=Region(view="view_front", box_px=(0, 0, 10, 10)),
        quantity=1,
        note=None,
    )
]
_A_INTERPRETATION.features[1].dimension_refs = ["dim_bore"]
_DIMENSION_CHECKS = {"dim_bore": "Unconfirmed: ret_base omits the interpreted bore."}

_A_PLAN = OperationPlan(
    proposal=[
        Operation(
            name="op_base",
            verb=OperationVerb.EXTRUDE,
            detail="Extrude the outline 25 mm along +z",
            semantics=["sem_feature_1"],
        )
    ],
    rationale="one extrude reaches the stated height",
)

_VERIFICATION = VerifyOutputResult(
    verification_id="v1",
    status=ExecutionStatus.VERIFIED,
    source="ret_base = object()\nresult = ret_base\n",
    returncode=0,
)

_AUDIT_REPORT = AuditReport(
    concern_reviews={
        "coding.concern_boss": ConcernReview(
            finding_name="find_missing_boss",
            disposition="The missing-boss finding covers it.",
        )
    },
    accepted=False,
    ticket_reviews={
        "ticket_missing_boss": TicketReview(
            summary="The boss is still absent from the current solid.",
            solved=False,
        )
    },
    findings=[
        AuditFinding(
            name="find_missing_boss",
            observation="the boss is missing",
            evidence=["attempts/v1/techdraw.dxf"],
            related_ticket_ids=["ticket_missing_boss"],
            backtrace=[
                CausalHop(
                    effect=StageOutputRef(stage=PipelineStage.CODING, name="ret_base"),
                    cause=StageOutputRef(
                        stage=PipelineStage.OPERATIONS, name="op_base"
                    ),
                    rationale="the return implements the base operation",
                )
            ],
            revision_request=RevisionRequest(
                action="modify",
                targets=[
                    StageOutputRef(stage=PipelineStage.OPERATIONS, name="op_base")
                ],
                instruction="add the omitted boss operation",
                proposed_names=[],
            ),
        )
    ],
)

# A page and the view cut from it, so a crop and a printed figure are both in
# the state this checkpoints.
_A_DRAWING = [
    view("full_page", name="view_page", file="inputs/page.png"),
    view(
        "front",
        region=Region(view="view_page", box_uv=(0.0, 0.0, 10.0, 10.0)),
        dimensions=[
            InterpretedDimension(
                name="dim_width",
                kind="linear",
                text="10",
                nominal_value=10.0,
                measured_length=None,
                region=Region(view="view_page", box_uv=(0.0, 0.0, 10.0, 10.0)),
                quantity=1,
                note=None,
            )
        ],
    ),
]
_RECONSTRUCTION = ReconstructionHistory(
    run_id="run_test",
    input_drawings=_A_DRAWING,
    snapshots=[
        ReconstructionSnapshot(
            open_tickets=[
                Ticket(
                    ticket_id="ticket_initial",
                    subject=BootstrapWork(instruction="reconstruct the drawing"),
                    assigned_stages=[
                        PipelineStage.INTERPRETATION,
                        PipelineStage.OPERATIONS,
                        PipelineStage.CODING,
                    ],
                    responses=[
                        TicketResponse(
                            ticket_id="ticket_initial",
                            stage=PipelineStage.INTERPRETATION,
                            summary="established sem_feature_1 and sem_feature_2",
                        ),
                        TicketResponse(
                            ticket_id="ticket_initial",
                            stage=PipelineStage.OPERATIONS,
                            summary="established op_base",
                        ),
                        TicketResponse(
                            ticket_id="ticket_initial",
                            stage=PipelineStage.CODING,
                            summary="implemented ret_base and result",
                        ),
                    ],
                )
            ],
            round=0,
            last_completed_stage=PipelineStage.CODING,
            interpretation=_A_INTERPRETATION,
            operations=_A_PLAN,
            program_source=_VERIFICATION.source,
            verification=_VERIFICATION,
            stage_reports={
                PipelineStage.CODING: StageReport(
                    concerns={"concern_boss": "Check the boss."},
                    dimension_checks=_DIMENSION_CHECKS,
                )
            },
        )
    ],
)

_INTERPRETATION_SUBMISSION = TicketAnswers(
    responses={"ticket_initial": "established sem_feature_1 and sem_feature_2"},
)
_OPERATION_SUBMISSION = TicketAnswers(
    responses={"ticket_initial": "established op_base"},
)
_CODING_SUBMISSION = TicketAnswers(
    stage_report=StageReport(dimension_checks=_DIMENSION_CHECKS),
    responses={"ticket_initial": "implemented ret_base and result"},
)

_ARTIFACTS: dict[str, object] = {
    "interpretation_state": {
        "messages": [HumanMessage(content="interpret the drawing")],
        "structured_response": _INTERPRETATION_SUBMISSION,
        "current_turn": 1,
        "total_turns": 1,
        "stop_reason": StopReason.COMPLETED,
    },
    "operations_state": {
        "messages": [HumanMessage(content="propose operations")],
        "structured_response": _OPERATION_SUBMISSION,
        "current_turn": 1,
        "total_turns": 1,
        "stop_reason": StopReason.COMPLETED,
    },
    "coding_state": {
        "messages": [HumanMessage(content="write code")],
        "structured_response": _CODING_SUBMISSION,
        "current_turn": 3,
        "total_turns": 3,
        "stop_reason": StopReason.BUDGET_EXHAUSTED,
    },
    "stage_submission": _CODING_SUBMISSION,
    "stage_validation_error": None,
    "stage_validation_failure_count": 0,
    "reconstruction": _RECONSTRUCTION,
    "audit_report": _AUDIT_REPORT,
}


def test_custom_state_types_include_nested_runtime_values() -> None:
    assert set(CUSTOM_STATE_TYPES) == {
        Operation,
        OperationPlan,
        OperationVerb,
        DrawingInterpretation,
        SemanticFeature,
        # The contract's enums ride in state too. An enum missing from the
        # allowlist restores as a bare string, which still compares equal and
        # so fails nowhere until something asks it for `.value`.
        View,
        ExecutionStatus,
        StopReason,
        VerifyOutputResult,
        AuditFinding,
        AuditReport,
        ConcernReview,
        TicketReview,
        CausalHop,
        RevisionRequest,
        StageOutputRef,
        BootstrapWork,
        ReconstructionHistory,
        ReconstructionSnapshot,
        Ticket,
        TicketResponse,
        PipelineStage,
        DrawingView,
        Region,
        InterpretedDimension,
        InterpretedView,
        TicketAnswers,
        StageReport,
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
        elif is_dataclass(value) and not isinstance(value, type):
            yield type(value)
            for field in fields(value):
                yield from types(getattr(value, field.name))
        else:
            yield type(value)

    assert {found for value in _ARTIFACTS.values() for found in types(value)} >= set(
        CUSTOM_STATE_TYPES
    )

    def store(_: ReconstructionState) -> ReconstructionState:
        return cast(ReconstructionState, dict(_ARTIFACTS))

    workflow = StateGraph(state_schema=ReconstructionState)  # type: ignore[type-var]
    workflow.add_node("store", store)  # pyrefly: ignore [bad-argument-type]
    workflow.add_edge(START, "store")
    workflow.add_edge("store", END)

    serde = JsonPlusSerializer(allowed_msgpack_modules=list(CUSTOM_STATE_TYPES))
    graph = workflow.compile(checkpointer=InMemorySaver(serde=serde))
    config = {"configurable": {"thread_id": "artifact-round-trip"}}

    graph.invoke(ReconstructionState(), config)
    restored = graph.get_state(config).values
    reviews = restored["audit_report"].ticket_reviews
    assert type(reviews["ticket_missing_boss"]) is TicketReview
    assert (
        type(
            restored["reconstruction"].snapshots[-1].stage_reports[PipelineStage.CODING]
        )
        is StageReport
    )

    for field, value in _ARTIFACTS.items():
        assert type(restored[field]) is type(value), field
        assert restored[field] == value

    interpretation_state = restored["interpretation_state"]
    assert type(interpretation_state["structured_response"]) is TicketAnswers
    assert type(interpretation_state["stop_reason"]) is StopReason

    operations_state = restored["operations_state"]
    assert type(operations_state["structured_response"]) is TicketAnswers


def _threaded_state(**stages: object) -> ReconstructionState:
    """A run part-way through, each stage holding what is its own."""
    read = [HumanMessage(content="read the views")]
    state: dict[str, object] = {
        "interpretation_state": {"messages": read, "current_turn": 3},
        "operations_state": {
            "messages": read,
            "current_turn": 2,
        },
        "coding_state": {"current_turn": 4},
        "audit_state": {"messages": [HumanMessage(content="judge it")]},
    }
    return cast(ReconstructionState, state | stages)


@pytest.mark.parametrize(
    ("stage", "stage_state"),
    [
        (PipelineStage.INTERPRETATION, {"interpretation_state": {"messages": ["it"]}}),
        (PipelineStage.OPERATIONS, {"operations_state": {"messages": ["it"]}}),
        (PipelineStage.CODING, {"coding_state": {"messages": ["it"]}}),
    ],
)
def test_the_thread_is_taken_from_whichever_agent_carried_it(
    stage: ReasoningStage, stage_state: dict[str, object]
) -> None:
    """The thread a stage finished is read from that stage's own channel."""
    state = _threaded_state(**stage_state)

    update = carry_thread(state, lead_transcript(state, stage))

    assert update["coding_state"]["messages"] == ["it"]


def test_the_thread_reaches_every_reasoning_stage_but_not_the_audit() -> None:
    """The audit judges from the outside; a thread it took part in would leave
    it marking its own work."""
    state = _threaded_state(coding_state={"messages": ["wrote the model"]})

    update = carry_thread(state, lead_transcript(state, PipelineStage.CODING))

    assert set(update) == {
        "interpretation_state",
        "operations_state",
        "coding_state",
    }
    assert update["interpretation_state"]["messages"] == ["wrote the model"]
    assert update["operations_state"]["messages"] == ["wrote the model"]


def test_the_stage_that_wrote_the_thread_is_given_back_what_it_wrote() -> None:
    """Handing it its own transcript changes nothing, and costs one special
    case less than leaving it out."""
    state = _threaded_state(coding_state={"messages": ["wrote the model"]})

    update = carry_thread(state, lead_transcript(state, PipelineStage.CODING))

    assert update["coding_state"]["messages"] == ["wrote the model"]


def test_what_a_stage_holds_besides_its_messages_survives_the_thread() -> None:
    """Turn counts belong to their stage, not to the thread that happens to be
    passing through it."""
    update = carry_thread(
        _threaded_state(),
        lead_transcript(_threaded_state(), PipelineStage.CODING),
    )

    assert update["interpretation_state"]["current_turn"] == 3
    assert update["operations_state"]["current_turn"] == 2


def test_the_prompt_log_is_told_where_the_inherited_thread_ends() -> None:
    """Without the watermark a stage reports the transcript it was handed as
    the prompt it was given."""
    state = _threaded_state(interpretation_state={"messages": ["one", "two"]})

    update = carry_thread(state, lead_transcript(state, PipelineStage.INTERPRETATION))

    assert update["coding_state"]["reported_message_count"] == 2
    assert update["coding_state"]["current_turn"] == 4


def test_a_stage_that_has_not_run_is_seeded_all_the_same() -> None:
    update = carry_thread(ReconstructionState(), [])

    assert update["interpretation_state"] == {
        "messages": [],
        "reported_message_count": 0,
    }
