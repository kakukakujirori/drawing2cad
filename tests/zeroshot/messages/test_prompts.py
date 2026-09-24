import json
import re
from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock

import pytest

from tests.zeroshot.prompt_paths import ROLE_PATHS, STAGES_DIR
from tests.zeroshot.workflow.test_reconstruction_workflow import (
    _completed_run,
    _ref,
    _report,
)
from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.stages._base.prompt import (
    PromptTemplate,
    StageInstructions,
    build_system_prompt,
)
from zeroshot.pipeline.stages.audit.contracts import AuditReport
from zeroshot.pipeline.stages.coding.stage import CodingStage
from zeroshot.pipeline.stages.interpretation.contracts import (
    TOWARD_VIEWER,
    DrawingInterpretation,
    DrawingView,
    Region,
    View,
    cross_axis,
)
from zeroshot.pipeline.stages.operations.contracts import Operation, OperationPlan
from zeroshot.pipeline.stages.types import PipelineStage
from zeroshot.pipeline.verification._run_program import INTERMEDIATE_RETURNS_DIR
from zeroshot.pipeline.verification.render.project import STANDARD_VIEW_FRAMES
from zeroshot.pipeline.workflow.lifecycle import (
    open_next_round,
    start_reconstruction,
)
from zeroshot.pipeline.workflow.state import ReconstructionState


def _write(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


# What the graph supplies to every instruction, whichever stage asked for it.
_AN_UNREAD_PAGE = [
    DrawingView(
        name="view_page",
        role="full_page",
        file="/work/inputs/drawing.dxf",
        region=Region(view="view_page", box_uv=(0.0, 0.0, 420.0, 297.0)),
        dimensions=[],
    )
]


# Stable run paths; round and ticket ownership come from state at build time.
_RUN_PATHS = {
    "coding_output_path": "/work/model.py",
    "audit_output_path": "/work/audit.json",
    "audit_schema": json.dumps(AuditReport.model_json_schema()),
    "interpretation_output_path": "/work/interpretation.json",
    "interpretation_schema": json.dumps(DrawingInterpretation.model_json_schema()),
    "operations_output_path": "/work/operations.json",
    "operations_schema": json.dumps(OperationPlan.model_json_schema()),
    "verification_dir": "/work/attempts",
    "reconstruction_path": "/work/reconstruction.json",
    "dimension_inventory": "[]",
    "drawing_diff_summary": "No automatic drawing comparison was recorded.",
}


@pytest.fixture
def instructions(tmp_path: Path) -> StageInstructions:
    input_path = _write(tmp_path / "drawing.dxf", "0\nEOF\n")
    source = [_AN_UNREAD_PAGE[0].model_copy(update={"file": str(input_path)})]
    return StageInstructions(
        prompt_context=_RUN_PATHS,
        input_presentation_mode="path",
        input_artifact=source,
        workdir=SandboxWorkdir(host_bind_dir=tmp_path),
    )


@pytest.fixture
def state() -> ReconstructionState:
    return {
        "reconstruction": start_reconstruction(
            "run_prompt", "Reconstruct the part.", _AN_UNREAD_PAGE
        )
    }


@pytest.fixture
def render_stage(
    instructions: StageInstructions, state: ReconstructionState
) -> Callable[..., str]:
    def render(stage: str, **context: str) -> str:
        return instructions.build(
            state,
            PipelineStage(stage),
            include_artifact=False,
            **{
                "attempt_dir": "/work/attempts/001",
                "intermediate_returns_dir": "unavailable",
                **context,
            },
        ).text

    return render


def _guidelines(stage: str) -> str:
    return PromptTemplate(STAGES_DIR / stage / "prompts/guidelines.md").render(
        **_RUN_PATHS
    )


def test_a_prompt_path_survives_a_change_of_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    render_stage: Callable[..., str],
) -> None:
    prompt = PromptTemplate(ROLE_PATHS["coder"])
    expected = prompt.render()
    expected_instruction = render_stage("coding")
    monkeypatch.chdir(tmp_path)

    assert prompt.render() == expected
    assert render_stage("coding") == expected_instruction


def test_a_reused_builder_reads_the_latest_round_and_ticket_ownership(
    instructions: StageInstructions,
    state: ReconstructionState,
) -> None:
    first = instructions.build(state, PipelineStage.CODING, include_artifact=False).text
    assert "round 0" in first
    assert "Tickets assigned to coding this round: ticket_initial" in first

    state["reconstruction"] = open_next_round(
        _completed_run(), _report(target=_ref("coding", "ret_hole"))
    )
    coding = instructions.build(
        state, PipelineStage.CODING, include_artifact=False
    ).text
    interpreted = instructions.build(
        state, PipelineStage.INTERPRETATION, include_artifact=False
    ).text

    assert "round 1" in coding
    assert "Tickets assigned to coding this round: ticket_001_shape_mismatch" in coding
    assert "Assigned tickets: none" in interpreted
    assert "ticket_initial" not in coding + interpreted


@pytest.mark.parametrize("stage", ["interpretation", "operations", "coding"])
def test_assigned_evidence_paths_reach_the_round_instruction(
    instructions: StageInstructions,
    state: ReconstructionState,
    stage: str,
) -> None:
    state["reconstruction"] = open_next_round(
        _completed_run(), _report(target=_ref(stage, None))
    )
    ticket = state["reconstruction"].snapshots[-1].open_tickets[0]
    ticket.evidence_crops = [
        "/work/tickets/evidence_0.png",
        "/work/tickets/evidence_1.png",
    ]

    for recipient in ("interpretation", "operations", "coding"):
        text = instructions.build(
            state, PipelineStage(recipient), include_artifact=False
        ).text
        for path in ticket.evidence_crops:
            assert (path in text) == (
                PipelineStage(recipient) in ticket.assigned_stages
            )
        if recipient == stage:
            assert "(evidence: " in text


@pytest.mark.parametrize("stage", list(PipelineStage))
def test_a_validation_retry_needs_only_the_error_from_state(
    instructions: StageInstructions,
    stage: PipelineStage,
) -> None:
    message = instructions.build(
        {"stage_validation_error": "Unknown reference: sem_missing"},
        stage,
        include_artifact=True,
    )

    assert f"[{stage.title()} Validation Error]" in message.text
    assert "Unknown reference: sem_missing" in message.text
    assert "corrected complete output" in message.text
    assert "[Input artifacts]" not in message.text
    assert "## Coordinate frames" not in message.text


def test_input_is_attached_only_when_requested_and_fresh_on_each_build(
    instructions: StageInstructions,
    state: ReconstructionState,
) -> None:
    plain = instructions.build(
        state, PipelineStage.INTERPRETATION, include_artifact=False
    )
    attached = instructions.build(
        state, PipelineStage.INTERPRETATION, include_artifact=True
    )
    another = instructions.build(
        state, PipelineStage.INTERPRETATION, include_artifact=True
    )

    assert "[Input artifacts]" not in plain.text
    assert attached.text.startswith(plain.text)
    assert attached.text.count("[Input artifacts]") == 1
    assert "- view_page (full_page): /work/drawing.dxf" in attached.text
    assert str(instructions.workdir.host_bind_dir) not in attached.text
    assert attached.text == another.text
    assert attached is not another
    first_ids = {block["id"] for block in attached.content_blocks if "id" in block}
    second_ids = {block["id"] for block in another.content_blocks if "id" in block}
    assert first_ids and second_ids
    assert first_ids.isdisjoint(second_ids)


def test_interpretation_prompt_exposes_the_runtime_schema(
    render_stage: Callable[..., str],
) -> None:
    prompt = render_stage("interpretation")
    examples = re.findall(r"```json\n(.*?)\n```", prompt, re.DOTALL)
    assert len(examples) == 1
    schema = json.loads(examples[0])
    assert schema == DrawingInterpretation.model_json_schema()
    assert "parameters" in schema["$defs"]["SemanticFeature"]["properties"]
    assert "Region" in schema["$defs"]
    assert "geometry" not in schema["$defs"]["SemanticFeature"]["properties"]


@pytest.mark.parametrize("stage", list(PipelineStage))
def test_stage_instructions_resolve_all_template_placeholders(
    stage: PipelineStage,
    render_stage: Callable[..., str],
) -> None:
    rendered = render_stage(stage.value)

    assert rendered
    for schema in ("interpretation_schema", "operations_schema", "audit_schema"):
        rendered = rendered.replace(_RUN_PATHS[schema], "")
    assert not re.search(r"\$[a-zA-Z_][a-zA-Z_0-9]*|\$\{", rendered)


def test_reconstruction_guide_keeps_only_the_working_contract() -> None:
    guide = PromptTemplate(
        STAGES_DIR / "_base/prompts/reconstruction_history.md"
    ).render(**_RUN_PATHS)

    for field in (
        "input_drawings",
        "snapshots",
        "open_tickets",
        "interpretation",
        "operations",
        "program_source",
        "verification",
        "responses",
        "stage_reports",
    ):
        assert field in guide
    assert "exactly one response per assigned ticket" not in guide
    assert len(guide.split()) < 450


def test_the_system_prompt_explains_selective_history_navigation() -> None:
    rendered = build_system_prompt(ROLE_PATHS["coder"], _RUN_PATHS).text
    guide = PromptTemplate(
        STAGES_DIR / "_base/prompts/reconstruction_history.md"
    ).render(**_RUN_PATHS)

    assert guide in rendered
    assert "interpretation -> operations -> coding + verification -> audit" in rendered
    assert "never edit or print the whole file" in rendered
    assert ".snapshots[-2]" in rendered

    assert "jq -c" in rendered
    assert "members by name" in rendered
    assert len(rendered.split()) < 500


def test_round_instructions_do_not_repeat_the_reconstruction_guide(
    render_stage: Callable[..., str],
) -> None:
    rendered = render_stage("interpretation")

    assert "## Reconstruction history" not in rendered
    assert "ReconstructionHistory" not in rendered


def test_audit_explains_how_to_report_a_missing_semantic_feature(
    render_stage: Callable[..., str],
) -> None:
    rendered = render_stage("audit")

    assert "leave the `backtrace` empty" in rendered
    assert "whole interpretation stage (`name: null`)" in rendered
    assert "proposing one or more stable `sem_...` names" in rendered


def test_the_returns_section_says_what_the_directory_is_for(
    render_stage: Callable[..., str],
) -> None:
    """The layout line alone does not say which `ret_` a defect belongs to."""
    returns_dir = f"/work/attempts/001/{INTERMEDIATE_RETURNS_DIR}"
    section = render_stage("audit", intermediate_returns_dir=returns_dir)

    assert INTERMEDIATE_RETURNS_DIR in section
    assert "ret_" in section
    assert "what the plan meant it to" in section


def test_the_audit_reads_the_attempt_directory_the_build_actually_wrote(
    render_stage: Callable[..., str],
) -> None:
    """The per-operation views are what localise a defect to one `ret_...`, and
    the auditor only looks in a directory it was told about."""
    returns_dir = f"/work/attempts/001/{INTERMEDIATE_RETURNS_DIR}"
    rendered = render_stage(
        "audit",
        attempt_dir="/work/attempts/001",
        intermediate_returns_dir=returns_dir,
    )

    assert "/work/attempts/001" in rendered
    assert "Latest interpretation verification" not in rendered
    assert "_interpretation_raw" not in rendered
    assert "_interpretation_validation_log" not in rendered
    assert f"Recorded directory: {returns_dir}" in rendered


def test_audit_reads_ticket_bodies_from_history_without_echoing_them(
    render_stage: Callable[..., str],
) -> None:
    rendered = render_stage("audit", ticket_responses="BODY_MUST_NOT_BE_ECHOED")

    assert "open_tickets" in rendered
    assert "subjects and stage responses" in rendered
    assert ".snapshots[-1]" in rendered
    assert "/work/reconstruction.json" in rendered
    assert "BODY_MUST_NOT_BE_ECHOED" not in rendered


def test_auditor_keeps_result_out_of_the_backtrace_graph(
    render_stage: Callable[..., str],
) -> None:
    rendered = render_stage("audit")

    assert "`result` is the terminal export, not a causal member" in rendered
    assert "whole coding stage with `name: null`" in rendered


def test_placeholders_are_filled_from_the_context(
    render_stage: Callable[..., str],
) -> None:
    """The run's paths reach the guidelines the coding instruction carries."""
    rendered = render_stage("coding")

    assert "/work/model.py" in rendered
    assert "/work/attempts" in rendered
    assert "$coding_output_path" not in rendered
    assert "$verification_dir" not in rendered


def test_the_coding_round_carries_the_history_and_result_contract(
    render_stage: Callable[..., str],
) -> None:
    rendered = render_stage("coding")

    assert "`ret_` variable by replacing its `op_` prefix" in rendered
    assert "# ----" not in rendered
    assert "Lxx-Lyy" not in rendered


def test_the_auditor_is_told_the_walk_rule_the_pipeline_would_reject_it_for(
    render_stage: Callable[..., str],
) -> None:
    """Mechanical walk rules reach the model through the report schema."""
    rendered = render_stage("audit")
    assert "at most one named-to-named hop within each prefix" in rendered
    assert "sem_ -> dim_ -> view_" in rendered


def test_the_auditor_role_does_not_repeat_the_api_contract() -> None:
    rendered = build_system_prompt(
        ROLE_PATHS["output_auditor"],
        {**_RUN_PATHS, "max_turns": "10"},
        AuditReport,
    ).text

    assert "`AuditReport`" not in rendered
    assert json.dumps(AuditReport.model_json_schema(), indent=2) not in rendered
    assert "$output_schema" not in rendered


def test_the_auditor_reviews_every_open_ticket_including_bootstrap_work(
    render_stage: Callable[..., str],
) -> None:
    rendered = render_stage("audit")
    guide = build_system_prompt(ROLE_PATHS["output_auditor"], _RUN_PATHS).text

    assert "one `ticket_reviews` entry per ticket" in rendered
    assert "Round 0 has one ticket" in guide
    assert "Cover every unsolved ticket" in rendered
    assert "root may have changed" in rendered


def test_the_operations_round_uses_json_for_the_plan_and_answer_for_tickets(
    render_stage: Callable[..., str],
) -> None:
    rendered = render_stage("operations")

    assert "/work/operations.json" in rendered
    assert "`TicketAnswers`" in rendered
    assert "contents when you answer become this round's plan" in rendered
    assert "`edits`" not in rendered
    assert "`deleted`" not in rendered
    assert "deliverable" not in rendered


def test_the_coding_round_keeps_code_in_the_workspace_and_reports_concerns(
    render_stage: Callable[..., str],
) -> None:
    """Coding revises the workspace, so its answer has no revision members to
    name -- and a member a stage cannot fill is one it can get wrong."""
    rendered = render_stage("coding")

    assert "`edits`" not in rendered
    assert "`deleted`" not in rendered
    assert "`rationale`" not in rendered
    assert (
        "ticket responses, one `stage_report.concerns` entry per remaining "
        "concern those responses do not explain" in rendered
    )
    assert "`dimension_checks`" in rendered
    assert "pipeline captures it through verification" in rendered


def test_audit_can_read_concerns_from_both_ticket_summaries_and_stage_reports(
    render_stage: Callable[..., str],
) -> None:
    prompt = build_system_prompt(
        ROLE_PATHS["output_auditor"],
        {**_RUN_PATHS, "max_turns": "10"},
        AuditReport,
    ).text

    assert "upstream blockers or provisional interpretations" in prompt
    assert "one `concern_...` entry each" in prompt
    assert "further unresolved issues" in prompt
    assert "The auditor reviews each" in prompt
    instruction = render_stage("audit")
    assert "ticket summaries, `concerns` and `unticketed_changes`" in instruction
    assert "placement does not determine" in instruction
    assert "missing reports in old snapshots" in prompt


def test_the_audit_names_concerns_the_way_its_contract_does(
    render_stage: Callable[..., str],
) -> None:
    """The round prompt and the schema must agree on where an ID comes from."""
    rendered = render_stage("audit")
    schema = json.dumps(AuditReport.model_json_schema())

    assert "`<reporting_stage>.<concern_id>`" in rendered
    assert "keyed by <reporting_stage>.<concern_id> as they appear" in schema
    assert "round prompt" not in schema


def test_coding_receives_all_dimension_readings_even_when_the_plan_omits_them(
    instructions: StageInstructions,
    state: ReconstructionState,
) -> None:
    from tests.zeroshot.workflow.test_resolve_submission import interpretation
    from tests.zeroshot.workflow.test_validate_submission import _snapshot

    held = interpretation()
    dimension = held.views[0].dimensions[0]
    held.views[0].dimensions.extend(
        [
            dimension.model_copy(update={"name": "dim_duplicate_value"}),
            dimension.model_copy(
                update={"name": "dim_unreadable", "nominal_value": None}
            ),
        ]
    )
    state["reconstruction"].snapshots[0] = _snapshot(
        PipelineStage.OPERATIONS, held=held
    )
    agent = Mock()
    agent.invoke.return_value = {}
    stage = CodingStage(
        agent=agent,
        instructions=instructions,
        output_verifier=Mock(),
        ticket_verifier=Mock(),
        middleware=Mock(),
        input_after_compaction=False,
    )

    def check_baseline_context():
        assert stage.output_verifier.interpretation is held

    stage.middleware.reset.side_effect = check_baseline_context
    stage.run(state, {})
    instruction = agent.invoke.call_args.args[0]["messages"][-1].text
    (inventory,) = re.findall(r"```json\n(.*?)\n```", instruction, re.DOTALL)
    readings = json.loads(inventory)

    assert [item["name"] for item in readings] == [
        dimension.name,
        "dim_duplicate_value",
        "dim_unreadable",
    ]
    assert readings[0]["nominal_value"] == readings[1]["nominal_value"]
    assert readings[2]["nominal_value"] is None
    assert set(readings[0]) == {"name", "text", "nominal_value", "kind", "quantity"}


def test_the_interpretation_round_uses_json_for_the_artifact_and_answer_for_tickets(
    render_stage: Callable[..., str],
) -> None:
    rendered = render_stage("interpretation")
    assert "/work/interpretation.json" in rendered
    assert "contents when you answer become this round's interpretation" in rendered
    assert "TicketAnswers" in rendered
    assert "current artifact validates" in rendered
    assert "`edits`" not in rendered


@pytest.mark.parametrize("stage", ["interpretation", "operations", "coding", "audit"])
def test_every_reasoning_round_carries_that_stage_s_guidelines(
    render_stage: Callable[..., str], stage: str
) -> None:
    guidelines = _guidelines(stage)

    rendered = render_stage(stage)
    assert guidelines in rendered


@pytest.mark.parametrize(
    "role",
    [
        "drawing_interpreter",
        "operation_planner",
        "coder",
        "cad_reconstructor",
        "output_auditor",
    ],
)
def test_a_proposer_role_says_who_it_is_and_leaves_the_rest_to_the_instruction(
    role: str,
) -> None:
    """What a stage must do belongs to the stage's instruction: a role holding
    all three would put the coding contract in front of a model that is still
    reading the drawing."""
    body = PromptTemplate(ROLE_PATHS[role]).path.read_text(encoding="utf-8")

    assert "$" not in body
    assert body.strip()
    assert "result` variable" not in body
    assert "try-except" not in body


def test_the_merged_role_renders_the_same_text_for_every_stage_that_shares_it() -> None:
    """One thread of thought needs one system prompt: the stages differ in
    turn budget and answer contract, so a prompt that took either as a
    placeholder would come out different for each of them."""
    body = PromptTemplate(ROLE_PATHS["cad_reconstructor"]).path.read_text(
        encoding="utf-8"
    )

    assert "$output_schema" not in body
    assert "$max_turns" not in body

    stage_contexts = [
        {
            "coding_output_path": "/work/model.py",
            "audit_output_path": "/work/audit.json",
            "audit_schema": json.dumps(AuditReport.model_json_schema()),
            "verification_dir": "/work/attempts",
        },
        # What `create_agent` adds for a stage that answers structurally, and
        # what it adds for the coder, which does not.
        {
            "coding_output_path": "/work/model.py",
            "audit_output_path": "/work/audit.json",
            "audit_schema": json.dumps(AuditReport.model_json_schema()),
            "verification_dir": "/work/attempts",
            "output_schema": "SENTINEL_SCHEMA",
            "max_turns": "20",
        },
        {
            "coding_output_path": "/work/model.py",
            "audit_output_path": "/work/audit.json",
            "audit_schema": json.dumps(AuditReport.model_json_schema()),
            "verification_dir": "/work/attempts",
            "max_turns": "10",
        },
    ]
    rendered = {
        PromptTemplate(ROLE_PATHS["cad_reconstructor"]).render(**context)
        for context in stage_contexts
    }

    assert len(rendered) == 1
    assert "$" not in rendered.pop()


def test_a_missing_value_is_refused(tmp_path: Path) -> None:
    """`substitute`, not `safe_substitute`: a value we forgot to pass must not
    reach the model as the literal `$verification_dir`."""
    prompt = PromptTemplate(_write(tmp_path / "p.md", "$here and $there"))

    with pytest.raises(KeyError):
        prompt.render(here="only one")


def test_an_unused_value_is_ignored(tmp_path: Path) -> None:
    """One context serves every stage, so a prompt may use none of it."""
    prompt = PromptTemplate(_write(tmp_path / "p.md", "no placeholders"))

    assert prompt.render(coding_output_path="/work/model.py") == "no placeholders"


def test_braces_survive_rendering(tmp_path: Path) -> None:
    """Why `$name` and not `{name}`: prompts carry CadQuery snippets."""
    body = 'result = cq.Workplane().box(**{"length": 1})'
    prompt = PromptTemplate(_write(tmp_path / "p.md", body))

    assert prompt.render() == body


def test_surrounding_whitespace_does_not_reach_the_model(tmp_path: Path) -> None:
    """Whether a file ends in a newline is an editor's decision, and must not
    silently change the bytes the model is sent."""
    bare = PromptTemplate(_write(tmp_path / "bare.md", "instructions"))
    padded = PromptTemplate(_write(tmp_path / "padded.md", "\ninstructions\n\n"))

    assert bare.render() == padded.render() == "instructions"


def test_an_unknown_prompt_is_refused_before_the_run() -> None:
    with pytest.raises(ValueError, match="prompt not found"):
        PromptTemplate(Path("no_such_prompt"))


def test_the_digest_follows_the_file(tmp_path: Path) -> None:
    """The prompt is the experiment's main variable, so a run's audit trail
    needs a way to say which text it used."""
    path = _write(tmp_path / "p.md", "first")
    prompt = PromptTemplate(path)
    before = prompt.sha256

    _write(path, "second")

    assert prompt.sha256 != before


def test_interpreter_uses_localized_evidence_and_checks_cross_view_ambiguities() -> (
    None
):
    guidelines = _guidelines("interpretation")
    assert "not a trace of every drawing primitive" in guidelines
    assert "hidden lines and matching projections" in guidelines
    assert "numeric sizes and model positions in parameters" in guidelines
    assert "convert pixel measurements with the scale" in guidelines
    assert "report the affected parameter and the choice you made" in guidelines


def test_interpretation_prioritises_a_verified_draft_and_source_pixel_measurements(
    render_stage: Callable[..., str],
) -> None:
    rendered = render_stage("interpretation")
    assert "save a provisional artifact" in rendered
    # Ordered by the pass it follows, not by a turn number picked in advance.
    assert "save a provisional artifact by turn" not in rendered
    assert "Reserve turns to read the automatic validation" in rendered
    assert "current artifact validates" in rendered
    assert "top left, x right, y down" in rendered
    assert "native pixels, not a resized display" in rendered
    assert "validation derives them" in rendered


@pytest.mark.parametrize("stage", list(PipelineStage))
def test_every_stage_receives_one_copy_of_the_fixed_coordinate_convention(
    stage: PipelineStage,
    render_stage: Callable[..., str],
) -> None:
    rendered = render_stage(stage.value)
    convention = PromptTemplate(
        STAGES_DIR / "_base/prompts/coordinate_frames.md"
    ).render()

    assert rendered.count(convention) == 1
    assert "directions, not a shared origin" in rendered


def test_coordinate_markdown_agrees_with_the_projection_contract() -> None:
    convention = PromptTemplate(
        STAGES_DIR / "_base/prompts/coordinate_frames.md"
    ).render()
    rows = re.findall(
        r"^\| (\w+) \| ([+-][XYZ]) \| ([+-][XYZ]) \| ([+-][XYZ]) \|$",
        convention,
        re.MULTILINE,
    )
    actual = {
        View(view.lower()): tuple(axis.lower() for axis in axes) for view, *axes in rows
    }
    assert set(actual) == set(TOWARD_VIEWER)
    for view, (u_axis, v_axis, out) in actual.items():
        assert out == TOWARD_VIEWER[view], view
        # The printed columns must be a frame the contract would accept.
        assert cross_axis(u_axis, v_axis) == out, view
        # The standard projections are drawn in the frames the table teaches.
        assert STANDARD_VIEW_FRAMES[view] == (u_axis, v_axis), view


def test_the_plan_the_prompt_asks_for_is_the_one_the_schema_takes() -> None:
    guidelines = _guidelines("operations")

    assert "build order" in guidelines
    assert "`semantics`" in guidelines
    assert "`verb`" in guidelines
    assert set(Operation.model_fields) == {
        "name",
        "verb",
        "detail",
        "semantics",
    }


def test_the_coder_is_told_to_build_in_list_order() -> None:
    assert "Build the operations in list order" in _guidelines("coding")


def test_downstream_prompts_use_interpreted_features_and_preserve_the_datum() -> None:
    for stage in ("operations", "coding"):
        guidelines = _guidelines(stage)
        assert "sem_main_bore.radius" in guidelines
        assert "sem_main_bore.center" in guidelines
        assert "datum" in guidelines
        assert "null means unknown, never zero" in guidelines.lower()
        assert "ev_" not in guidelines
        assert "geo_" not in guidelines
