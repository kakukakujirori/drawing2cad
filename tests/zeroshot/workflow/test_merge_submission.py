"""Operation revisions merge by stable name; interpreted artifacts are file outputs."""

import pytest
from pydantic import ValidationError

from tests.zeroshot.contracts import interpretation, interpreted_feature, replacing
from zeroshot.pipeline.messages.tickets import BootstrapWork, Ticket, TicketResponse
from zeroshot.pipeline.stages._base.merge import merge_lists
from zeroshot.pipeline.stages._base.validate import SubmissionValidationError
from zeroshot.pipeline.stages.coding.submission import CodingSubmission
from zeroshot.pipeline.stages.contracts import ReconstructionSnapshot
from zeroshot.pipeline.stages.interpretation.submission import InterpretationSubmission
from zeroshot.pipeline.stages.merge import merge_submission
from zeroshot.pipeline.stages.operations.contracts import Operation, OperationPlan
from zeroshot.pipeline.stages.operations.submission import OperationSubmission
from zeroshot.pipeline.stages.types import REASONING_STAGES, PipelineStage
from zeroshot.pipeline.verification import ExecutionStatus, VerifyOutputResult


def _operation(name="op_base_plate", **overrides):
    return Operation.model_validate(
        {
            "name": name,
            "verb": "extrude",
            "detail": "extrude the plate",
            "depends_on": [],
            "semantics": ["sem_base_plate"],
        }
        | overrides
    )


def _plan():
    return OperationPlan(
        proposal=[
            _operation(),
            _operation(
                "op_main_bore",
                verb="hole",
                detail="bore through the plate",
                depends_on=["op_base_plate"],
                semantics=["sem_main_bore"],
            ),
        ],
        rationale="the bore follows the plate",
    )


def _snapshot(completed=True):
    return ReconstructionSnapshot(
        open_tickets=[
            Ticket(
                ticket_id="ticket_initial",
                subject=BootstrapWork(instruction="Reconstruct the part."),
                assigned_stages=list(REASONING_STAGES),
                responses=[
                    TicketResponse(
                        ticket_id="ticket_initial",
                        stage=stage,
                        summary=f"Answered during {stage}.",
                    )
                    for stage in REASONING_STAGES
                ]
                if completed
                else [],
            )
        ],
        round=0,
        last_completed_stage=PipelineStage.CODING if completed else None,
        interpretation=interpretation(
            features=[
                interpreted_feature("sem_base_plate", "plate"),
                interpreted_feature("sem_main_bore", "bore"),
            ]
        )
        if completed
        else None,
        operations=_plan() if completed else None,
        program_source="result = None\n" if completed else None,
        verification=VerifyOutputResult(status=ExecutionStatus.VERIFIED, returncode=0)
        if completed
        else None,
    )


def _submission(**overrides):
    return OperationSubmission.model_validate(
        {"edits": [], "deleted": [], "rationale": None, "responses": []} | overrides
    )


def _merge(submission, previous=None):
    return merge_submission(
        submission,
        previous if previous is not None else _snapshot(),
        PipelineStage.OPERATIONS,
    )


def test_a_first_round_builds_the_whole_artifact_from_its_edits():
    assert _merge(_submission(**replacing(_plan())), _snapshot(False)) == _plan()


def test_an_untouched_member_survives_a_revision():
    revised = _operation(detail="extrude the chamfered plate")
    merged = _merge(_submission(edits=[revised]))
    assert merged.proposal == [revised, _plan().proposal[1]]


def test_an_edit_replaces_the_complete_member_without_mutating_history():
    previous = _snapshot()
    original = previous.model_dump_json()
    revised = _operation(
        "op_main_bore",
        verb="cut",
        detail="wider bore",
        depends_on=[],
        semantics=["sem_main_bore"],
    )
    merged = _merge(_submission(edits=[revised]), previous)
    assert merged.proposal[1] == revised
    assert merged.proposal[1].depends_on == []
    assert previous.model_dump_json() == original


def test_a_new_member_is_appended_and_an_edited_one_keeps_its_place():
    added = _operation("op_fillet", verb="fillet", depends_on=["op_main_bore"])
    edited = _operation(detail="revised plate")
    assert _merge(_submission(edits=[added, edited])).proposal == [
        edited,
        _plan().proposal[1],
        added,
    ]


def test_deletion_and_complete_replacement_remove_old_links():
    bore = _operation(
        "op_main_bore", verb="hole", depends_on=[], semantics=["sem_main_bore"]
    )
    assert _merge(_submission(edits=[bore], deleted=["op_base_plate"])).proposal == [
        bore
    ]


def test_a_null_rationale_keeps_the_preceding_one():
    assert _merge(_submission()).rationale == _plan().rationale


def test_a_first_round_must_state_a_rationale():
    with pytest.raises(SubmissionValidationError, match="no rationale to keep"):
        _merge(_submission(), _snapshot(False))


def test_a_deletion_must_name_an_existing_member():
    with pytest.raises(SubmissionValidationError, match="absent.*op_absent"):
        _merge(_submission(deleted=["op_absent"]))


def test_a_stage_must_submit_its_own_kind_of_revision():
    with pytest.raises(
        SubmissionValidationError, match="operations must submit an OperationSubmission"
    ):
        _merge(InterpretationSubmission(responses=[]))


def test_invalid_merged_dependencies_raise_a_retryable_error_without_mutation():
    previous = _snapshot()
    original = previous.model_dump_json()
    with pytest.raises(
        SubmissionValidationError, match="depends on op_base_plate"
    ) as caught:
        _merge(_submission(deleted=["op_base_plate"]), previous)
    assert isinstance(caught.value.__cause__, ValidationError)
    assert previous.model_dump_json() == original


def test_merge_lists_accepts_iterators_and_preserves_replacement_order():
    old = _plan().proposal
    edited, added = _operation("op_main_bore"), _operation("op_fillet")
    assert merge_lists(iter(old), iter([added, edited])) == [old[0], edited, added]


def test_merge_lists_asserts_on_internal_edit_delete_conflicts():
    edited = _operation()
    with pytest.raises(AssertionError, match="cannot edit and delete"):
        merge_lists([], iter([edited]), [edited.name])


@pytest.mark.parametrize(
    "stage,submission",
    [
        (PipelineStage.INTERPRETATION, InterpretationSubmission(responses=[])),
        (PipelineStage.CODING, CodingSubmission(responses=[])),
    ],
)
def test_workspace_outputs_are_not_merged(stage, submission):
    with pytest.raises(SubmissionValidationError, match="workspace output"):
        merge_submission(submission, _snapshot(), stage)
