import pytest
from pydantic import BaseModel, ValidationError

from zeroshot.pipeline.stages.audit.contracts import (
    AuditFinding,
    AuditRegion,
    AuditReport,
    AuditSubmission,
    ConcernReview,
    TicketReview,
)
from zeroshot.pipeline.stages.tickets.contracts import StageReport, TicketAnswers


def _region(file: str) -> AuditRegion:
    return AuditRegion(file=file, box=(0, 0, 10, 10))


def finding(
    name: str = "find_wrong_bore",
    *,
    cause: str = "interpretation",
    targets: list[str] | None = None,
    related_ticket_ids: list[str] | None = None,
) -> AuditFinding:
    return AuditFinding(
        name=name,
        observation="The reconstructed bore is too wide.",
        evidence=[_region("render_3d/hlg_front.png")],
        cause=cause,  # type: ignore[arg-type]
        targets=["sem_bore"] if targets is None else targets,
        revision_request="The bore is 12 mm in the drawing and 16 mm in the build.",
        related_ticket_ids=related_ticket_ids or [],
    )


@pytest.mark.parametrize("targets", [["sem_bore"], ["view_front", "datum"]])
def test_an_interpretation_cause_names_existing_style_members(targets) -> None:
    assert finding(targets=targets).targets == targets


def test_an_interpretation_cause_needs_a_target() -> None:
    with pytest.raises(ValidationError, match="at least one target"):
        finding(targets=[])


def test_a_coding_cause_may_have_no_target() -> None:
    assert finding(cause="coding", targets=[]).targets == []


@pytest.mark.parametrize(
    ("targets", "message"),
    [
        (["bore"], "targets must be"),
        (["find_bore"], "targets must be"),
        (["sem_bore", "sem_bore"], "duplicates"),
    ],
)
def test_targets_are_distinct_member_names(targets, message) -> None:
    with pytest.raises(ValidationError, match=message):
        finding(targets=targets)


@pytest.mark.parametrize("cause", ["audit", "operations", None])
def test_the_cause_is_interpretation_or_coding(cause) -> None:
    with pytest.raises(ValidationError):
        finding(cause=cause)


@pytest.mark.parametrize("invalid_name", ["finding_legacy_bore", "issue_wrong_bore"])
def test_find_names_are_canonical_and_other_prefixes_are_rejected(
    invalid_name: str,
) -> None:
    assert finding().name == "find_wrong_bore"

    with pytest.raises(ValidationError, match="find_\\.\\.\\."):
        finding(name=invalid_name)


@pytest.mark.parametrize(
    ("evidence", "message"),
    [
        ([], "must not be empty"),
        ([_region("front.png"), _region("front.png")], "duplicates"),
    ],
)
def test_a_finding_requires_distinct_evidence_regions(
    evidence: list[AuditRegion], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        AuditFinding(
            name="find_wrong_bore",
            observation="The bore is wrong.",
            evidence=evidence,
            cause="coding",
            targets=[],
            revision_request="The bore is wrong.",
            related_ticket_ids=[],
        )


def test_the_decision_belongs_only_to_the_final_submission() -> None:
    report = AuditReport(concern_reviews={}, ticket_reviews={}, findings=[])
    assert "accepted" not in report.model_dump()
    assert "accepted" not in AuditReport.model_json_schema()["properties"]
    assert AuditSubmission(accepted=False).model_dump() == {"accepted": False}
    with pytest.raises(ValidationError):
        AuditSubmission.model_validate({"reviewed": True})


def test_finding_names_are_unique_within_a_report() -> None:
    with pytest.raises(ValidationError, match="finding names must be unique"):
        AuditReport(
            concern_reviews={},
            ticket_reviews={},
            findings=[finding(), finding()],
        )


def _object_schemas(node: object) -> list[dict]:
    """Every model in the schema. A keyed map is an object too, but its keys are
    ticket IDs and concern names, so it is open by design."""
    if isinstance(node, dict):
        found = [node] if node.get("type") == "object" and "properties" in node else []
        for held in node.values():
            found.extend(_object_schemas(held))
        return found
    if isinstance(node, list):
        return [found for held in node for found in _object_schemas(held)]
    return []


@pytest.mark.parametrize(
    "contract",
    [
        AuditFinding,
        AuditReport,
        AuditSubmission,
        AuditRegion,
        TicketReview,
        ConcernReview,
        TicketAnswers,
        StageReport,
    ],
)
def test_requested_answer_schemas_are_closed_and_require_every_field(
    contract: type[BaseModel],
) -> None:
    for schema in _object_schemas(contract.model_json_schema()):
        assert set(schema.get("required", [])) == set(schema.get("properties", {}))
        assert schema.get("additionalProperties") is False


@pytest.mark.parametrize(
    "value",
    [
        finding(),
        _region("front.png"),
        TicketReview(summary="Checked.", solved=True),
        ConcernReview(finding_name=None, disposition="No correction needed."),
        AuditReport(ticket_reviews={}, concern_reviews={}, findings=[]),
    ],
)
def test_audit_models_ignore_extra_labels_but_still_require_each_field(value):
    contract = type(value)
    payload = {
        **value.model_dump(),
        "type": "unrequested label",
        "extra_note": "unused",
    }
    assert contract.model_validate(payload) == value
    for name in contract.model_fields:
        misspelled = dict(payload)
        misspelled[name + "_typo"] = misspelled.pop(name)
        with pytest.raises(ValidationError) as raised:
            contract.model_validate(misspelled)
        assert any(
            e["type"] == "missing" and e["loc"] == (name,)
            for e in raised.value.errors()
        )


@pytest.mark.parametrize(
    "change",
    [
        {"summary": "  "},
        {"solved": "unknown"},
    ],
)
def test_ticket_review_requires_a_nonblank_check_and_boolean_decision(change) -> None:
    with pytest.raises(ValidationError):
        TicketReview.model_validate(
            {
                "ticket_id": "ticket_bore",
                "summary": "The bore is restored.",
                "solved": True,
            }
            | change
        )


def test_related_ticket_ids_are_unique_within_each_finding() -> None:
    with pytest.raises(ValidationError, match="related_ticket_ids.*duplicates"):
        finding(related_ticket_ids=["ticket_bore", "ticket_bore"])


@pytest.mark.parametrize(
    ("solved", "related", "valid"),
    [
        ([], [], True),
        ([True, True], [], True),
        ([False, True], ["ticket_0"], True),
        ([False, False], ["ticket_0", "ticket_1"], True),
        ([False], [], False),
        ([True], ["ticket_0"], False),
        ([False], ["ticket_0", "ticket_other"], False),
        ([], ["ticket_other"], False),
    ],
)
def test_unsolved_reviews_match_exactly_the_tickets_in_current_findings(
    solved: list[bool], related: list[str], valid: bool
) -> None:
    values = {
        "concern_reviews": {},
        "ticket_reviews": {
            f"ticket_{index}": TicketReview(summary="Checked the bore.", solved=value)
            for index, value in enumerate(solved)
        },
        "findings": [finding(related_ticket_ids=related)],
    }
    if valid:
        report = AuditReport(**values)
        assert report.findings  # New defects remain even when old tickets are solved.
    else:
        with pytest.raises(ValidationError, match="unsolved ticket IDs") as caught:
            AuditReport(**values)
        assert any(ticket in str(caught.value) for ticket in [*related, "ticket_0"])


def test_one_unsolved_ticket_can_require_several_current_findings() -> None:
    report = AuditReport(
        concern_reviews={},
        ticket_reviews={
            "ticket_bore": TicketReview(summary="Two defects remain.", solved=False)
        },
        findings=[
            finding(name, related_ticket_ids=["ticket_bore"])
            for name in ("find_wrong_bore", "find_wrong_boss")
        ],
    )
    assert len(report.findings) == 2


def test_an_empty_review_list_must_still_be_given() -> None:
    payload = finding().model_dump()
    del payload["related_ticket_ids"]
    with pytest.raises(ValidationError, match="related_ticket_ids\n  Field required"):
        AuditFinding.model_validate(payload)
    with pytest.raises(ValidationError, match="ticket_reviews\n  Field required"):
        AuditReport.model_validate({"accepted": True, "findings": []})


def test_a_concern_needs_a_disposition_in_words() -> None:
    with pytest.raises(ValidationError, match="disposition must not be blank"):
        ConcernReview(finding_name=None, disposition="  ")


def test_a_concern_cannot_be_escalated_to_a_finding_the_report_lacks() -> None:
    with pytest.raises(ValidationError, match="does not hold: find_absent"):
        AuditReport(
            concern_reviews={
                "coding.concern_bore": ConcernReview(
                    finding_name="find_absent",
                    disposition="Escalated as a finding.",
                )
            },
            ticket_reviews={},
            findings=[finding()],
        )


@pytest.mark.parametrize(
    "file", ["input.png", "projection/front.PNG", "render_3d/iso.jpg"]
)
def test_raster_boxes_require_json_integers(file):
    region = AuditRegion(file=file, box=(0, 1, 10, 20))
    assert AuditRegion.model_validate_json(region.model_dump_json()) == region
    assert all(type(edge) is int for edge in region.box)
    for edge in [0.5, 0.0, "0", False]:
        with pytest.raises(ValidationError):
            AuditRegion(file=file, box=(edge, 1, 10, 20))


def test_dxf_boxes_allow_fractional_and_negative_millimetres():
    region = AuditRegion(file="projection/front.DXF", box=(-5.5, 0, 10.25, 20))
    assert region.box == (-5.5, 0, 10.25, 20)
    assert AuditRegion.model_validate_json(region.model_dump_json()) == region
    for box in [(0, 0, 10), (0, 0, 10, 20, 30), (0, 0, float("inf"), 20)]:
        with pytest.raises(ValidationError):
            AuditRegion(file="front.dxf", box=box)
