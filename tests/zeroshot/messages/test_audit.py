import pytest
from pydantic import BaseModel, ValidationError

from zeroshot.pipeline.messages.contracts.audit import (
    AuditFinding,
    AuditReport,
    Backtrace,
    CausalHop,
    RevisionRequest,
    StageOutputRef,
)


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
        targets=targets or [ref("semantics", "sem_bore")],
        instruction="Correct the bore interpretation.",
        proposed_names=proposed_names or [],
    )


def backtrace(
    *,
    revision_request: RevisionRequest | None = None,
) -> Backtrace:
    return Backtrace(
        hops=[
            CausalHop(
                effect=ref("coding", "ret_bore"),
                cause=ref("operations", "op_bore"),
                rationale="The code result implements this operation.",
            ),
            CausalHop(
                effect=ref("operations", "op_bore"),
                cause=ref("semantics", "sem_bore"),
                rationale="The operation implements this semantic feature.",
            ),
        ],
        revision_request=revision_request or request(),
    )


def finding(name: str = "finding_wrong_bore") -> AuditFinding:
    return AuditFinding(
        name=name,
        observation="The reconstructed bore is too wide.",
        evidence=["render_3d/hlg_front.png", "sem_bore.geo_cylinder.radius"],
        backtraces=[backtrace()],
    )


@pytest.mark.parametrize(
    ("stage", "name"),
    [
        ("semantics", "op_bore"),
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
        ("add", [ref("semantics", None)], ["sem_bore"]),
        ("modify", [ref("semantics", None)], []),
        ("modify", [ref("semantics", "sem_bore")], []),
        ("delete", [ref("operations", "op_bore")], []),
        (
            "split",
            [ref("operations", "op_hole")],
            ["op_bore", "op_counterbore"],
        ),
        (
            "merge",
            [ref("semantics", "sem_hole"), ref("semantics", "sem_bore")],
            ["sem_stepped_bore"],
        ),
        ("rename", [ref("coding", "ret_hole")], ["ret_bore"]),
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
        ("add", [ref("semantics", "sem_bore")], ["sem_hole"], "whole-stage"),
        ("modify", [ref("semantics", "sem_bore")], ["sem_hole"], "does not"),
        ("delete", [ref("operations", None)], [], "named target"),
        ("split", [ref("operations", "op_hole")], ["op_bore"], "at least two"),
        (
            "merge",
            [ref("semantics", "sem_hole")],
            ["sem_bore"],
            "at least two",
        ),
        ("rename", [ref("coding", "ret_hole")], [], "exactly one proposed"),
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
                ref("semantics", "sem_bore"),
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


def test_a_backtrace_must_be_contiguous() -> None:
    trace = backtrace()
    trace.hops[1] = trace.hops[1].model_copy(
        update={"effect": ref("coding", "ret_other")}
    )

    with pytest.raises(ValidationError, match="next hop"):
        Backtrace.model_validate(trace.model_dump())


def test_a_backtrace_must_end_at_a_revision_target() -> None:
    wrong_request = request(targets=[ref("semantics", "sem_other")])
    with pytest.raises(ValidationError, match="final causal cause"):
        backtrace(revision_request=wrong_request)


def test_an_empty_backtrace_is_valid_when_the_finding_is_already_at_its_root() -> None:
    Backtrace(hops=[], revision_request=request())
    Backtrace(
        hops=[],
        revision_request=request(targets=[ref("coding", None)]),
    )


@pytest.mark.parametrize(
    ("evidence", "message"),
    [
        ([], "must not be empty"),
        (["  "], "must not be blank"),
        (["same", "same"], "duplicates"),
    ],
)
def test_a_finding_requires_distinct_evidence_locators(
    evidence: list[str], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        AuditFinding(
            name="finding_wrong_bore",
            observation="The bore is wrong.",
            evidence=evidence,
            backtraces=[backtrace()],
        )


def test_an_accepted_report_has_no_findings() -> None:
    AuditReport(accepted=True, findings=[])
    with pytest.raises(ValidationError, match="accepted must be true"):
        AuditReport(accepted=True, findings=[finding()])


def test_a_rejected_report_has_at_least_one_finding() -> None:
    AuditReport(accepted=False, findings=[finding()])
    with pytest.raises(ValidationError, match="accepted must be true"):
        AuditReport(accepted=False, findings=[])


def test_finding_names_are_unique_within_a_report() -> None:
    with pytest.raises(ValidationError, match="finding names must be unique"):
        AuditReport(accepted=False, findings=[finding(), finding()])


def _object_schemas(node: object) -> list[dict]:
    if isinstance(node, dict):
        found = [node] if node.get("type") == "object" else []
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
        Backtrace,
        AuditFinding,
        AuditReport,
    ],
)
def test_audit_contracts_are_closed_and_every_property_is_required(
    contract: type[BaseModel],
) -> None:
    for schema in _object_schemas(contract.model_json_schema()):
        assert set(schema.get("required", [])) == set(schema.get("properties", {}))
        assert schema.get("additionalProperties") is False
