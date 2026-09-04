"""Merging a stage's revision onto the artifact of the preceding round."""

import pytest

from tests.zeroshot.contracts import evidence, feature, geometry, replacing, unchanged
from zeroshot.pipeline.messages.contracts import (
    Operation,
    OperationPlan,
    OperationVerb,
    PipelineStage,
    SemanticHypothesis,
)
from zeroshot.pipeline.messages.contracts.reconstruction import (
    BootstrapWork,
    CodingSubmission,
    OperationSubmission,
    ReconstructionSnapshot,
    SemanticSubmission,
    Ticket,
    TicketResponse,
)
from zeroshot.pipeline.verification import ExecutionStatus, VerifyOutputResult
from zeroshot.pipeline.workflow.merge_submission import merge_submission
from zeroshot.pipeline.workflow.validate_submission import SubmissionValidationError


def _responses(stage: PipelineStage) -> list[TicketResponse]:
    return [
        TicketResponse(
            ticket_id="ticket_initial",
            stage=stage,  # type: ignore[arg-type]
            summary=f"Answered during {stage}.",
        )
    ]


def _semantics() -> SemanticSubmission:
    return SemanticSubmission(
        edits=[],
        deleted=[],
        rationale=None,
        responses=_responses(PipelineStage.SEMANTICS),
    )


def _operations() -> OperationSubmission:
    return OperationSubmission(
        edits=[],
        deleted=[],
        rationale=None,
        responses=_responses(PipelineStage.OPERATIONS),
    )


def _preceding(
    semantics: SemanticHypothesis | None = None,
    operations: OperationPlan | None = None,
) -> ReconstructionSnapshot:
    """The completed snapshot whose artifacts a revision round starts from."""
    stages = (PipelineStage.SEMANTICS, PipelineStage.OPERATIONS, PipelineStage.CODING)
    return ReconstructionSnapshot(
        open_tickets=[
            Ticket(
                ticket_id="ticket_initial",
                subject=BootstrapWork(instruction="Reconstruct the part."),
                assigned_stages=list(stages),  # type: ignore[arg-type]
                responses=[
                    TicketResponse(
                        ticket_id="ticket_initial",
                        stage=stage,  # type: ignore[arg-type]
                        summary=f"Answered during {stage}.",
                    )
                    for stage in stages
                ],
            )
        ],
        round=0,
        last_completed_stage=PipelineStage.CODING,  # type: ignore[arg-type]
        semantics=semantics if semantics is not None else _hypothesis(),
        operations=operations if operations is not None else _plan(),
        program_source="result = None\n",
        verification=VerifyOutputResult(status=ExecutionStatus.VERIFIED, returncode=0),
    )


def _merged_semantics(
    submission: SemanticSubmission,
    previous: ReconstructionSnapshot | None,
) -> SemanticHypothesis:
    merged = merge_submission(submission, previous, PipelineStage.SEMANTICS)
    assert isinstance(merged, SemanticHypothesis)
    return merged


def _bore() -> object:
    return feature(
        "sem_main_bore",
        "the bore through the plate",
        geometry=[geometry("cylinder", name="geo_cylinder")],
        evidence=["ev_front_circle"],
    )


def _hypothesis() -> SemanticHypothesis:
    return SemanticHypothesis(
        evidence=[
            evidence("line", name="ev_line"),
            evidence("circle", name="ev_front_circle"),
            # Cited by nothing, so it is the one a revision may drop.
            evidence("arc", name="ev_spare"),
        ],
        proposal=[feature("sem_base_plate", "the plate"), _bore()],  # type: ignore[list-item]
        rationale="the views agree",
    )


def _plan(detail: str = "extrude the plate") -> OperationPlan:
    return OperationPlan(
        proposal=[
            Operation(
                name="op_base_plate",
                verb=OperationVerb.EXTRUDE,
                detail=detail,
                depends_on=[],
                semantics=["sem_base_plate"],
            ),
            Operation(
                name="op_main_bore",
                verb=OperationVerb.HOLE,
                detail="bore through the plate",
                depends_on=["op_base_plate"],
                semantics=["sem_main_bore"],
            ),
        ],
        rationale="the bore follows the plate",
    )


def test_a_first_round_builds_the_whole_artifact_from_its_edits() -> None:
    previous = _hypothesis()
    submission = SemanticSubmission(
        **replacing(previous),
        responses=_responses(PipelineStage.SEMANTICS),
    )

    assert _merged_semantics(submission, None) == previous


def test_an_untouched_member_survives_a_revision() -> None:
    submission = _semantics().model_copy(
        update={"edits": [feature("sem_base_plate", "the plate, now chamfered")]}
    )

    merged = _merged_semantics(submission, _preceding())

    assert [f.name for f in merged.proposal] == ["sem_base_plate", "sem_main_bore"]
    assert merged.proposal[0].description == "the plate, now chamfered"
    assert merged.proposal[1] == _hypothesis().proposal[1]


def test_an_edited_feature_keeps_the_geo_members_it_leaves_out() -> None:
    wider = geometry("cylinder", name="geo_cylinder", radius=9.0)
    submission = _semantics().model_copy(
        update={
            "edits": [
                feature(
                    "sem_main_bore",
                    "the bore through the plate",
                    geometry=[wider],
                    evidence=["ev_front_circle"],
                )
            ]
        }
    )

    revised = _merged_semantics(submission, _preceding()).proposal[1]

    assert revised.geometry == [wider]


def test_a_feature_that_is_given_at_all_gives_the_whole_of_its_citation() -> None:
    """Evidence are cited rather than carried, so there is no member to merge:
    an omission means "cites nothing", which the hypothesis then refuses."""
    submission = _semantics().model_copy(
        update={
            "edits": [
                feature(
                    "sem_main_bore",
                    "the bore through the plate",
                    geometry=[geometry("cylinder", name="geo_cylinder")],
                    evidence=[],
                )
            ]
        }
    )

    with pytest.raises(SubmissionValidationError, match="cites no evidence"):
        merge_submission(submission, _preceding(), PipelineStage.SEMANTICS)


def test_an_entry_the_revision_leaves_out_keeps_what_it_had() -> None:
    corrected = evidence("circle", name="ev_front_circle", radius=9.0)
    submission = _semantics().model_copy(update={"evidence": [corrected]})

    merged = _merged_semantics(submission, _preceding())

    assert [entry.name for entry in merged.evidence] == [
        "ev_line",
        "ev_front_circle",
        "ev_spare",
    ]
    assert merged.evidence[1] == corrected


def test_an_entry_is_deleted_by_its_own_name() -> None:
    """A bare ev_ name addresses an entry, where a bare sem_ name addresses a
    feature; the two collections share one `deleted` list."""
    submission = _semantics().model_copy(update={"deleted": ["ev_spare"]})

    merged = _merged_semantics(submission, _preceding())

    assert [entry.name for entry in merged.evidence] == [
        "ev_line",
        "ev_front_circle",
    ]


def test_a_new_member_is_appended_and_an_edited_one_keeps_its_place() -> None:
    submission = _semantics().model_copy(
        update={
            "edits": [
                feature("sem_top_fillet", "the rounded top edge"),
                feature("sem_base_plate", "the plate, revised"),
            ]
        }
    )

    merged = _merged_semantics(submission, _preceding())

    assert [f.name for f in merged.proposal] == [
        "sem_base_plate",
        "sem_main_bore",
        "sem_top_fillet",
    ]


def test_deleting_a_whole_feature_and_one_member_of_another() -> None:
    submission = _semantics().model_copy(
        update={"deleted": ["sem_base_plate", "sem_main_bore.geo_cylinder"]}
    )

    merged = _merged_semantics(submission, _preceding())

    assert [f.name for f in merged.proposal] == ["sem_main_bore"]
    assert merged.proposal[0].geometry == []
    assert merged.proposal[0].evidence == ["ev_front_circle"]


def test_a_null_rationale_keeps_the_preceding_one() -> None:
    assert _merged_semantics(_semantics(), _preceding()).rationale == "the views agree"


def test_a_first_round_must_state_a_rationale() -> None:
    with pytest.raises(SubmissionValidationError, match="no rationale to keep"):
        merge_submission(_semantics(), None, PipelineStage.SEMANTICS)


@pytest.mark.parametrize(
    ("deleted", "message"),
    [
        (["sem_absent"], "no such feature"),
        (["sem_main_bore.geo_absent"], "has no member"),
        (["sem_absent.geo_cylinder"], "not an address"),
        (["sem_main_bore.geo_cylinder.radius"], "not an address"),
    ],
)
def test_a_deletion_must_address_something_the_artifact_holds(
    deleted: list[str],
    message: str,
) -> None:
    submission = _semantics().model_copy(update={"deleted": deleted})

    with pytest.raises(SubmissionValidationError, match=message):
        merge_submission(submission, _preceding(), PipelineStage.SEMANTICS)


def test_an_operation_is_deleted_by_its_own_name_alone() -> None:
    submission = _operations().model_copy(update={"deleted": ["op_main_bore.detail"]})

    with pytest.raises(SubmissionValidationError, match="delete it by its own op_"):
        merge_submission(submission, _preceding(), PipelineStage.OPERATIONS)


def test_a_revision_that_deletes_a_cited_entry_is_rejected() -> None:
    """Dropping an entry two features rest on is not a local edit."""
    submission = _semantics().model_copy(update={"deleted": ["ev_front_circle"]})

    with pytest.raises(SubmissionValidationError, match="does not read"):
        merge_submission(submission, _preceding(), PipelineStage.SEMANTICS)


def test_a_stage_must_submit_its_own_kind_of_revision() -> None:
    with pytest.raises(SubmissionValidationError, match="SemanticSubmission"):
        merge_submission(_operations(), _preceding(), PipelineStage.SEMANTICS)


def test_operations_replace_by_name_and_keep_their_place() -> None:
    revised = Operation(
        name="op_base_plate",
        verb=OperationVerb.EXTRUDE,
        detail="extrude the plate 30 mm",
        depends_on=[],
        semantics=["sem_base_plate"],
    )
    submission = _operations().model_copy(update={"edits": [revised]})

    merged = merge_submission(submission, _preceding(), PipelineStage.OPERATIONS)

    assert isinstance(merged, OperationPlan)
    assert merged.proposal[0] == revised
    assert merged.proposal[1] == _plan().proposal[1]
    assert merged.rationale == "the bore follows the plate"


def test_coding_merges_to_nothing() -> None:
    coding = CodingSubmission(**unchanged(), responses=_responses(PipelineStage.CODING))

    assert merge_submission(coding, _preceding(), PipelineStage.CODING) is None
