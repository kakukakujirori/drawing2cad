from itertools import pairwise

import pytest
from pydantic import BaseModel, ValidationError

from zeroshot.pipeline.stages.audit.contracts import (
    AuditFinding,
    AuditRegion,
    AuditReport,
    AuditSubmission,
    CausalHop,
    ConcernReview,
    RevisionRequest,
    StageOutputRef,
    TicketReview,
)
from zeroshot.pipeline.stages.tickets.contracts import StageReport, TicketAnswers


def ref(stage: str, name: str | None) -> StageOutputRef:
    return StageOutputRef(stage=stage, name=name)  # type: ignore[arg-type]


def request(
    action: str = "modify",
    *,
    targets: list[StageOutputRef] | None = None,
    proposed_names: list[str] | None = None,
) -> RevisionRequest:
    return RevisionRequest(
        action=action,  # type: ignore[arg-type]
        targets=targets or [ref("interpretation", "sem_bore")],
        instruction="Correct the bore interpretation.",
        proposed_names=proposed_names or [],
    )


def backtrace() -> list[CausalHop]:
    return [
        CausalHop(
            effect=ref("coding", "ret_bore"),
            cause=ref("operations", "op_bore"),
            rationale="The code result implements this operation.",
        ),
        CausalHop(
            effect=ref("operations", "op_bore"),
            cause=ref("interpretation", "sem_bore"),
            rationale="The operation implements this semantic feature.",
        ),
    ]


def _region(file: str) -> AuditRegion:
    return AuditRegion(file=file, box=(0, 0, 10, 10))


def finding(
    name: str = "find_wrong_bore",
    *,
    hops: list[CausalHop] | None = None,
    revision_request: RevisionRequest | None = None,
    related_ticket_ids: list[str] | None = None,
) -> AuditFinding:
    return AuditFinding(
        name=name,
        observation="The reconstructed bore is too wide.",
        evidence=[_region("render_3d/hlg_front.png")],
        backtrace=backtrace() if hops is None else hops,
        revision_request=revision_request or request(),
        related_ticket_ids=related_ticket_ids or [],
    )


@pytest.mark.parametrize(
    ("stage", "name"),
    [
        ("interpretation", "ev_front_line"),
        ("interpretation", "sheet_front"),
        ("interpretation", "op_bore"),
        ("operations", "sem_bore"),
        ("coding", "op_bore"),
        ("coding", "part"),
        ("coding", "result"),
    ],
)
def test_a_stage_reference_rejects_a_name_owned_by_another_stage(
    stage: str, name: str
) -> None:
    with pytest.raises(ValidationError, match="valid member name"):
        ref(stage, name)


@pytest.mark.parametrize(
    ("action", "targets", "proposed_names"),
    [
        ("add", [ref("interpretation", None)], ["sem_bore"]),
        ("modify", [ref("interpretation", None)], []),
        ("modify", [ref("interpretation", "sem_bore")], []),
        (
            "modify",
            [ref("interpretation", "sem_bore"), ref("interpretation", "sem_hole")],
            [],
        ),
        ("delete", [ref("operations", "op_bore")], []),
        (
            "delete",
            [ref("operations", "op_bore"), ref("operations", "op_hole")],
            [],
        ),
        (
            "split",
            [ref("operations", "op_hole")],
            ["op_bore", "op_counterbore"],
        ),
        (
            "merge",
            [ref("interpretation", "sem_hole"), ref("interpretation", "sem_bore")],
            ["sem_stepped_bore"],
        ),
        ("rename", [ref("operations", "op_hole")], ["op_bore"]),
        ("modify", [ref("coding", "ret_hole")], []),
        ("modify", [ref("coding", None)], []),
    ],
)
def test_each_revision_action_accepts_its_defined_shape(
    action: str,
    targets: list[StageOutputRef],
    proposed_names: list[str],
) -> None:
    request(action, targets=targets, proposed_names=proposed_names)


@pytest.mark.parametrize(
    ("action", "targets", "proposed_names", "message"),
    [
        ("add", [ref("interpretation", "sem_bore")], ["sem_hole"], "whole-stage"),
        ("modify", [ref("interpretation", "sem_bore")], ["sem_hole"], "does not"),
        (
            "modify",
            [ref("interpretation", None), ref("interpretation", "sem_bore")],
            [],
            "not both",
        ),
        ("delete", [ref("operations", None)], [], "named target"),
        (
            "delete",
            [ref("operations", None), ref("operations", "op_bore")],
            [],
            "named target",
        ),
        ("split", [ref("operations", "op_hole")], ["op_bore"], "at least two"),
        (
            "merge",
            [ref("interpretation", "sem_hole")],
            ["sem_bore"],
            "at least two",
        ),
        ("rename", [ref("operations", "op_hole")], [], "exactly one proposed"),
    ],
)
def test_each_revision_action_rejects_an_invalid_shape(
    action: str,
    targets: list[StageOutputRef],
    proposed_names: list[str],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        request(action, targets=targets, proposed_names=proposed_names)


def test_a_revision_request_cannot_cross_stage_boundaries() -> None:
    with pytest.raises(ValidationError, match="same stage"):
        request(
            "merge",
            targets=[
                ref("interpretation", "sem_bore"),
                ref("operations", "op_bore"),
            ],
            proposed_names=["sem_merged"],
        )


def test_a_proposed_name_belongs_to_the_target_stage() -> None:
    with pytest.raises(ValidationError, match="invalid proposed names"):
        request(
            "split",
            targets=[ref("operations", "op_hole")],
            proposed_names=["op_bore", "sem_counterbore"],
        )


@pytest.mark.parametrize(
    ("action", "targets", "proposed_names"),
    [
        ("add", [None], ["ret_bore"]),
        ("delete", ["ret_bore"], []),
        ("split", ["ret_bore"], ["ret_through", "ret_counterbore"]),
        ("merge", ["ret_through", "ret_counterbore"], ["ret_bore"]),
        ("rename", ["ret_bore"], ["ret_through"]),
    ],
)
def test_coding_cannot_change_identities_owned_by_the_operation_plan(
    action: str, targets: list[str | None], proposed_names: list[str]
) -> None:
    with pytest.raises(ValidationError, match="coding accepts only modify"):
        request(
            action,
            targets=[ref("coding", name) for name in targets],
            proposed_names=proposed_names,
        )


@pytest.mark.parametrize(
    ("stage", "name"),
    [("interpretation", "sem_bore"), ("operations", "op_bore")],
)
def test_rename_must_change_the_identity(stage: str, name: str) -> None:
    with pytest.raises(ValidationError, match="different proposed name"):
        request("rename", targets=[ref(stage, name)], proposed_names=[name])


@pytest.mark.parametrize(
    ("action", "targets", "proposed_names"),
    [
        ("split", ["op_bore"], ["op_bore", "op_counterbore"]),
        ("merge", ["op_bore", "op_counterbore"], ["op_bore"]),
    ],
)
def test_split_and_merge_can_retain_a_target_identity(
    action: str, targets: list[str], proposed_names: list[str]
) -> None:
    request(
        action,
        targets=[ref("operations", name) for name in targets],
        proposed_names=proposed_names,
    )


def path(*members: tuple[str, str | None]) -> list[CausalHop]:
    return [
        CausalHop(
            effect=ref(*effect),
            cause=ref(*cause),
            rationale="The cause explains the mismatch.",
        )
        for effect, cause in pairwise(members)
    ]


@pytest.mark.parametrize(
    ("effect", "cause"),
    [
        (("coding", "ret_bore"), ("interpretation", "sem_bore")),
        (("coding", None), ("interpretation", None)),
        (("interpretation", "sem_bore"), ("operations", "op_bore")),
        (("operations", None), ("coding", None)),
    ],
)
def test_hops_stay_within_a_stage_or_move_to_the_adjacent_upstream_stage(
    effect: tuple[str, str | None], cause: tuple[str, str | None]
) -> None:
    with pytest.raises(ValidationError, match="adjacent upstream stage"):
        path(effect, cause)


def test_a_backtrace_can_follow_a_feature_to_its_dimension_and_source_view() -> None:
    hops = path(
        ("coding", "ret_bore"),
        ("operations", "op_bore"),
        ("interpretation", "sem_bore"),
        ("interpretation", "dim_diameter"),
        ("interpretation", "view_front"),
    )
    finding(hops=hops, revision_request=request(targets=[hops[-1].cause]))


def test_a_backtrace_can_take_one_named_hop_within_each_prefix() -> None:
    hops = path(
        ("coding", "ret_final"),
        ("coding", "ret_bore"),
        ("operations", "op_bore"),
        ("operations", "op_base"),
        ("interpretation", "sem_bore"),
        ("interpretation", "sem_base"),
        ("interpretation", "dim_diameter"),
        ("interpretation", "dim_width"),
        ("interpretation", "view_front"),
        ("interpretation", "view_page"),
    )
    finding(hops=hops, revision_request=request(targets=[hops[-1].cause]))


@pytest.mark.parametrize(
    ("stage", "prefix"),
    [
        ("coding", "ret"),
        ("operations", "op"),
        ("interpretation", "sem"),
        ("interpretation", "dim"),
        ("interpretation", "view"),
    ],
)
def test_a_backtrace_cannot_walk_repeatedly_within_one_named_prefix(
    stage: str, prefix: str
) -> None:
    hops = path(*[(stage, f"{prefix}_{name}") for name in ("final", "middle", "root")])
    with pytest.raises(ValidationError, match=rf"{prefix}_ prefix more than once"):
        finding(hops=hops, revision_request=request(targets=[hops[-1].cause]))


@pytest.mark.parametrize(
    ("stage", "prefix"),
    [("coding", "ret"), ("operations", "op"), ("interpretation", "sem")],
)
def test_whole_stage_references_do_not_count_as_named_prefix_hops(
    stage: str, prefix: str
) -> None:
    hops = path((stage, None), (stage, f"{prefix}_final"), (stage, f"{prefix}_root"))
    finding(hops=hops, revision_request=request(targets=[hops[-1].cause]))


@pytest.mark.parametrize(
    "members",
    [
        [("coding", None), ("coding", "ret_bore"), ("coding", None)],
        [
            ("interpretation", "sem_bore"),
            ("interpretation", "dim_diameter"),
            ("interpretation", "view_front"),
            ("interpretation", "sem_bore"),
        ],
    ],
)
def test_a_backtrace_cannot_revisit_an_output(
    members: list[tuple[str, str | None]],
) -> None:
    hops = path(*members)
    with pytest.raises(ValidationError, match="cycle"):
        finding(hops=hops, revision_request=request(targets=[hops[-1].cause]))


def test_a_backtrace_must_be_contiguous() -> None:
    broken = backtrace()
    broken[1] = broken[1].model_copy(update={"effect": ref("operations", "op_other")})

    with pytest.raises(ValidationError, match="next hop"):
        finding(hops=broken)


def test_a_backtrace_must_end_at_a_revision_target() -> None:
    with pytest.raises(ValidationError, match="final causal cause"):
        finding(revision_request=request(targets=[ref("interpretation", "sem_other")]))


def test_an_empty_backtrace_is_valid_when_the_finding_is_already_at_its_root() -> None:
    finding(hops=[])
    finding(hops=[], revision_request=request(targets=[ref("coding", None)]))


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
            backtrace=backtrace(),
            revision_request=request(),
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


def test_separate_findings_cannot_propose_the_same_identity() -> None:
    first = finding(
        hops=[],
        revision_request=request(
            "add", targets=[ref("interpretation", None)], proposed_names=["sem_new"]
        ),
    )
    second = finding(
        "find_rename_bore",
        hops=[],
        revision_request=request("rename", proposed_names=["sem_new"]),
    )
    with pytest.raises(ValidationError, match="unique across findings"):
        AuditReport(
            concern_reviews={},
            ticket_reviews={},
            findings=[first, second],
        )


def test_separate_findings_can_propose_distinct_identities() -> None:
    AuditReport(
        concern_reviews={},
        ticket_reviews={},
        findings=[
            finding(
                f"find_add_{name}",
                hops=[],
                revision_request=request(
                    "add",
                    targets=[ref("interpretation", None)],
                    proposed_names=[f"sem_{name}"],
                ),
            )
            for name in ("bore", "boss")
        ],
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
        StageOutputRef,
        RevisionRequest,
        CausalHop,
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
        ref("coding", "ret_bore"),
        request(),
        backtrace()[0],
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


@pytest.mark.parametrize("name", ["view_front", "dim_diameter", "sem_bore"])
def test_interpretation_members_support_direct_add_without_a_backtrace(
    name: str,
) -> None:
    finding(
        hops=[],
        revision_request=request(
            "add", targets=[ref("interpretation", None)], proposed_names=[name]
        ),
    )
    ref("interpretation", name)


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
