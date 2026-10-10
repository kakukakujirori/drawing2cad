import json
import re
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.zeroshot.prompt_paths import (
    RECONSTRUCTION_CONTEXT_PATH,
    ROLE_PATHS,
    STAGES_DIR,
)
from tests.zeroshot.workflow.test_reconstruction_workflow import (
    _completed_run,
    _report,
)
from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.stages._base.prompt import (
    PromptTemplate,
    StageInstructions,
    build_system_prompt,
)
from zeroshot.pipeline.stages.audit.contracts import AuditReport
from zeroshot.pipeline.stages.interpretation.contracts import (
    TOWARD_VIEWER,
    DrawingInterpretation,
    DrawingView,
    Region,
    View,
    cross_axis,
)
from zeroshot.pipeline.stages.types import PipelineStage
from zeroshot.pipeline.verification.render.orthographic import STANDARD_VIEW_FRAMES
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
    "verification_dir": "/work/attempts",
    "reconstruction_path": "/work/reconstruction.json",
}
_DRAWING_DIFF_REVIEWS = PromptTemplate(
    STAGES_DIR / "audit/prompts/drawing_diff_reviews.md"
).render()


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
            append_inputs=False,
            **{
                "attempt_dir": "/work/attempts/001",
                "drawing_diff_reviews": _DRAWING_DIFF_REVIEWS,
                **context,
            },
        ).text

    return render


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
    first = instructions.build(state, PipelineStage.CODING, append_inputs=False).text
    assert "round 0" in first
    assert "Tickets assigned to coding this round: ticket_initial" in first

    state["reconstruction"] = open_next_round(_completed_run(), _report("coding", []))
    coding = instructions.build(state, PipelineStage.CODING, append_inputs=False).text

    assert "round 1" in coding
    assert "Tickets assigned to coding this round: ticket_001_shape_mismatch" in coding
    assert "ticket_initial" not in coding


@pytest.mark.parametrize("cause", ["interpretation", "coding"])
def test_ticket_evidence_paths_reach_the_round_instruction(
    instructions: StageInstructions,
    state: ReconstructionState,
    cause: str,
) -> None:
    state["reconstruction"] = open_next_round(
        _completed_run(), _report(cause, ["sem_feature_1"])
    )
    ticket = state["reconstruction"].snapshots[-1].open_tickets[0]
    ticket.evidence_renders = [
        "/work/tickets/evidence_0.png",
        "/work/tickets/evidence_1.png",
    ]

    text = instructions.build(state, PipelineStage.CODING, append_inputs=False).text
    assert "(evidence: " + ", ".join(ticket.evidence_renders) + ")" in text


@pytest.mark.parametrize("stage", list(PipelineStage))
def test_a_validation_retry_needs_only_the_error_from_state(
    instructions: StageInstructions,
    stage: PipelineStage,
) -> None:
    message = instructions.build(
        {"stage_validation_error": "Unknown reference: sem_missing"},
        stage,
        append_inputs=True,
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
    plain = instructions.build(state, PipelineStage.CODING, append_inputs=False)
    attached = instructions.build(state, PipelineStage.CODING, append_inputs=True)
    another = instructions.build(state, PipelineStage.CODING, append_inputs=True)

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
    prompt = render_stage("coding")
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
    for schema in ("interpretation_schema", "audit_schema"):
        rendered = rendered.replace(_RUN_PATHS[schema], "")
    assert not re.search(r"\$[a-zA-Z_][a-zA-Z_0-9]*|\$\{", rendered)


def test_reconstruction_guide_keeps_only_the_working_contract() -> None:
    guide = PromptTemplate(RECONSTRUCTION_CONTEXT_PATH).render(**_RUN_PATHS)

    for field in (
        "input_drawings",
        "snapshots",
        "open_tickets",
        "interpretation",
        "program_source",
        "verification",
        "responses",
        "stage_reports",
    ):
        assert field in guide
    assert "exactly one response per assigned ticket" not in guide
    assert len(guide.split()) < 650


def test_the_system_prompt_explains_selective_history_navigation() -> None:
    rendered = build_system_prompt(ROLE_PATHS["coder"], _RUN_PATHS).text
    guide = PromptTemplate(RECONSTRUCTION_CONTEXT_PATH).render(**_RUN_PATHS)

    assert guide in rendered
    assert rendered.startswith(guide + "\n\n")
    assert rendered.endswith(ROLE_PATHS["coder"].read_text().strip())
    assert "Never edit it or print the whole file" in rendered
    assert ".snapshots[-2]" in rendered

    assert "jq -c" in rendered
    assert "members by name" in rendered
    assert len(rendered.split()) < 700


def test_round_instructions_do_not_repeat_the_reconstruction_guide(
    render_stage: Callable[..., str],
) -> None:
    rendered = render_stage("coding")

    assert "## Reconstruction history" not in rendered
    assert "ReconstructionHistory" not in rendered


def test_audit_explains_how_to_report_a_missing_semantic_feature(
    render_stage: Callable[..., str],
) -> None:
    rendered = render_stage("audit")

    assert "name the `view_...` where the drawing shows it" in rendered


def test_the_audit_reads_the_attempt_directory_the_build_actually_wrote(
    render_stage: Callable[..., str],
) -> None:
    """The auditor only looks in a directory it was told about."""
    rendered = render_stage(
        "audit",
        attempt_dir="/work/attempts/001",
    )

    assert "/work/attempts/001" in rendered
    assert "Latest interpretation verification" not in rendered
    assert "_interpretation_raw" not in rendered
    assert "_interpretation_validation_log" not in rendered


def test_audit_reads_ticket_bodies_from_history_without_echoing_them(
    render_stage: Callable[..., str],
) -> None:
    rendered = render_stage("audit", ticket_responses="BODY_MUST_NOT_BE_ECHOED")

    assert "open_tickets" in rendered
    assert "subjects and stage responses" in rendered
    assert ".snapshots[-1]" in rendered
    assert "/work/reconstruction.json" in rendered
    assert "BODY_MUST_NOT_BE_ECHOED" not in rendered


def test_auditor_tells_a_misreading_from_a_misbuild(
    render_stage: Callable[..., str],
) -> None:
    rendered = render_stage("audit")

    assert "the program builds that error faithfully" in rendered
    assert "`coding` when the interpretation is right" in rendered


def test_placeholders_are_filled_from_the_context(
    render_stage: Callable[..., str],
) -> None:
    """The run's paths reach the composed coding instruction."""
    rendered = render_stage("coding")

    assert "/work/model.py" in rendered
    assert "/work/attempts" in rendered
    assert "$coding_output_path" not in rendered
    assert "$verification_dir" not in rendered


def test_the_coding_round_carries_the_history_and_result_contract(
    render_stage: Callable[..., str],
) -> None:
    rendered = render_stage("coding")

    assert "Store the final completed CadQuery solid in `result`" in rendered
    assert "# ----" not in rendered
    assert "Lxx-Lyy" not in rendered


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

    assert "one `ticket_reviews` entry per open ticket" in rendered
    assert "Round 0 has one ticket" in guide
    assert "Cover every unsolved ticket" in rendered
    assert "it may differ from the old ticket's" in rendered


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
        "ticket responses and one `stage_report.concerns` entry per remaining "
        "concern those responses do not explain" in rendered
    )
    assert "dimension_checks" not in rendered
    assert "pipeline captures both files through verification" in rendered


def test_audit_can_read_concerns_from_both_ticket_summaries_and_stage_reports(
    render_stage: Callable[..., str],
) -> None:
    prompt = build_system_prompt(
        ROLE_PATHS["output_auditor"],
        {**_RUN_PATHS, "max_turns": "10"},
        AuditReport,
    ).text

    assert "provisional interpretations and what could not be resolved" in prompt
    assert "one `concern_...` entry each" in prompt
    assert "further unresolved issues" in prompt
    assert "The auditor reviews each" in prompt
    instruction = render_stage("audit")
    assert "Ticket responses and `concerns` are claims, not proof" in instruction


def test_the_audit_names_concerns_the_way_its_contract_does(
    render_stage: Callable[..., str],
) -> None:
    """The round prompt and the schema must agree on where an ID comes from."""
    rendered = render_stage("audit")
    schema = json.dumps(AuditReport.model_json_schema())

    assert "`<reporting_stage>.<concern_id>`" in rendered
    assert "Key a concern by <reporting_stage>.<concern_id> as it appears" in schema
    assert "`drawing_diff.*` item" in rendered
    assert "drawing_diff item as listed: drawing_diff.view_top.1" in schema
    assert "round prompt" not in schema


def test_the_interpretation_round_uses_json_for_the_artifact_and_answer_for_tickets(
    render_stage: Callable[..., str],
) -> None:
    rendered = render_stage("coding")
    assert "- `/work/interpretation.json`: your reading of the drawing." in rendered
    assert "Write the interpretation first." in rendered
    assert "builds the program only after the interpretation validates" in rendered
    assert "TicketAnswers" in rendered
    assert "`edits`" not in rendered


@pytest.mark.parametrize("stage", ["coding", "audit"])
def test_every_round_has_explicit_instruction_sections(
    render_stage: Callable[..., str], stage: str
) -> None:
    rendered = render_stage(stage)
    assert re.findall(r"^### (.+)$", rendered, re.MULTILINE)[-4:] == [
        "Task and inputs",
        "Artifact contract",
        "Work cycle",
        "Submission",
    ]


@pytest.mark.parametrize(
    "role",
    ["coder", "output_auditor"],
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


def test_shared_system_is_context_only_regardless_of_stage_budget_and_schema() -> None:
    guide = PromptTemplate(RECONSTRUCTION_CONTEXT_PATH).render(**_RUN_PATHS)
    for turns in (10, 20, 30):
        rendered = build_system_prompt(
            None,
            _RUN_PATHS | {"max_turns": str(turns), "output_schema": "SENTINEL_SCHEMA"},
        ).text
        assert rendered == guide
        assert "$" not in rendered


def test_a_role_without_a_record_receives_no_reconstruction_context() -> None:
    role = ROLE_PATHS["coder"]
    assert build_system_prompt(role, {}).text == role.read_text().strip()


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


def test_interpreter_uses_localized_evidence_and_checks_cross_view_ambiguities(
    render_stage: Callable[..., str],
) -> None:
    instructions = render_stage("coding")
    assert "not a trace of every drawing primitive" in instructions
    assert "hidden lines and matching projections" in instructions
    assert "numeric sizes and model positions in parameters" in instructions
    assert "Convert the pixels to millimetres with the scale" in instructions
    assert "take the most likely value from your measurements" in instructions


def test_interpretation_prioritises_a_verified_draft_and_source_pixel_measurements(
    render_stage: Callable[..., str],
) -> None:
    rendered = render_stage("coding")
    assert "current feature descriptions with numeric parameters" in rendered
    # Ordered by the pass it follows, not by a turn number picked in advance.
    assert "After inspecting all views once" in rendered
    assert "Reserve turns to read the automatic validation" in rendered
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


def test_the_coding_prompt_uses_interpreted_features_and_preserves_the_datum(
    render_stage: Callable[..., str],
) -> None:
    instructions = render_stage("coding")
    assert "sem_main_bore.radius" in instructions
    assert "dim_bore_diameter.nominal_value" in instructions
    assert "datum" in instructions
    assert re.search(r"null means unknown, (?:never|not) zero", instructions.lower())
    assert "ev_" not in instructions
    assert "geo_" not in instructions


@pytest.mark.parametrize("role", list(ROLE_PATHS))
def test_coding_instructions_are_not_injected_into_system_prompts(role: str) -> None:
    system = build_system_prompt(ROLE_PATHS[role], _RUN_PATHS).text

    guide = PromptTemplate(RECONSTRUCTION_CONTEXT_PATH).render(**_RUN_PATHS)
    assert system == guide + "\n\n" + ROLE_PATHS[role].read_text().strip()
    assert "calculate_drawing_scale" not in system
    assert "## Evidence policy" not in system
    assert "correct the interpretation as well as" not in system.lower()
    assert "stroke thickness" not in system.lower()


@pytest.mark.parametrize("stage", list(PipelineStage))
def test_geometry_correction_policy_belongs_to_the_coding_round(
    render_stage: Callable[..., str], stage: PipelineStage
) -> None:
    instruction = render_stage(stage.value)
    assert ("correct the interpretation as well as the program" in instruction) == (
        stage is PipelineStage.CODING
    )


def test_coder_tests_predictions_and_inspects_the_latest_candidate(
    render_stage: Callable[..., str],
) -> None:
    instruction = render_stage("coding")

    assert "Do not believe in your 3D reasoning blindly" in instruction
    assert "do not reject it based only on your reasoning" in instruction
    assert "inspect the latest deciding views and overall scores" in instruction
    assert "before keeping or undoing the change" in instruction
    assert "If the expected change is absent" in instruction
    assert "Read the latest verification feedback before concluding" in instruction
    assert (
        "revise the geometry or coordinate hypothesis before another trial"
        in instruction
    )
    assert (
        "Scratch files are not automatically executed, verified or submitted"
        in instruction
    )
    assert "Compare feature extent, placement and connections" in instruction
    assert "Treat stroke thickness as a drawing convention" in instruction
    assert (
        "Keep untested predictions and unavailable views explicitly unverified"
        in instruction
    )
