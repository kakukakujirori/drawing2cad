import json
import re
from collections.abc import Callable
from pathlib import Path

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
from zeroshot.pipeline.stages.drawings.contracts import (
    VIEW_FRAME,
    DrawingEvidence,
    DrawingSheet,
    DrawingSource,
    DrawnEntity,
)
from zeroshot.pipeline.stages.operations.contracts import Operation
from zeroshot.pipeline.stages.semantics.contracts import GeometryKind
from zeroshot.pipeline.stages.types import PipelineStage
from zeroshot.pipeline.verification._run_program import INTERMEDIATE_RETURNS_DIR
from zeroshot.pipeline.workflow.lifecycle import (
    open_next_round,
    start_reconstruction,
)
from zeroshot.pipeline.workflow.state import ReconstructionState


def _write(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


# What the graph supplies to every instruction, whichever stage asked for it.
_AN_UNREAD_PAGE = DrawingSource(
    sheets=[
        DrawingSheet(
            name="sheet_page",
            role="full_page",
            label=None,
            crop_of=None,
            scale=1.0,
            file="/work/inputs/drawing.dxf",
            evidence=[],
            dimensions=[],
        )
    ]
)


# Stable run paths; round and ticket ownership come from state at build time.
_RUN_PATHS = {
    "coding_output_path": "/work/model.py",
    "drawing_output_path": "/work/drawing.json",
    "verification_dir": "/work/attempts",
    "reconstruction_path": "/work/reconstruction.json",
}


@pytest.fixture
def instructions(tmp_path: Path) -> StageInstructions:
    input_path = _write(tmp_path / "drawing.dxf", "0\nEOF\n")
    source = DrawingSource(
        sheets=[_AN_UNREAD_PAGE.sheets[0].model_copy(update={"file": str(input_path)})]
    )
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
                "drawing_attempt_dir": "/work/attempts/round_000/drawing/003",
                "ticket_responses": "[]",
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
    semantics = instructions.build(
        state, PipelineStage.SEMANTICS, include_artifact=False
    ).text

    assert "round 1" in coding
    assert "Tickets assigned to coding this round: ticket_001_shape_mismatch" in coding
    assert "Tickets assigned to semantics this round: none" in semantics
    assert "ticket_initial" not in coding + semantics


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
    plain = instructions.build(state, PipelineStage.DRAWINGS, include_artifact=False)
    attached = instructions.build(state, PipelineStage.DRAWINGS, include_artifact=True)
    another = instructions.build(state, PipelineStage.DRAWINGS, include_artifact=True)

    assert "[Input artifacts]" not in plain.text
    assert attached.text.startswith(plain.text)
    assert attached.text.count("[Input artifacts]") == 1
    assert "- sheet_page (full_page): /work/drawing.dxf" in attached.text
    assert str(instructions.workdir.host_bind_dir) not in attached.text
    assert attached.text == another.text
    assert attached is not another
    first_ids = {block["id"] for block in attached.content_blocks if "id" in block}
    second_ids = {block["id"] for block in another.content_blocks if "id" in block}
    assert first_ids and second_ids
    assert first_ids.isdisjoint(second_ids)


def test_drawing_prompt_evidence_example_matches_the_runtime_contract(
    render_stage: Callable[..., str],
) -> None:
    prompt = render_stage("drawings")
    examples = re.findall(r"```json\n(.*?)\n```", prompt, re.DOTALL)
    assert len(examples) == 1
    evidence = DrawingEvidence.model_validate(json.loads(examples[0]))
    assert evidence.entity is DrawnEntity.LINE
    assert [p.name for p in evidence.parameters] == ["start", "end"]


@pytest.mark.parametrize("stage", list(PipelineStage))
def test_stage_instructions_resolve_all_template_placeholders(
    stage: PipelineStage,
    render_stage: Callable[..., str],
) -> None:
    rendered = render_stage(stage.value)

    assert rendered
    assert not re.search(r"\$[a-zA-Z_][a-zA-Z_0-9]*|\$\{", rendered)


def test_reconstruction_guide_keeps_only_the_working_contract() -> None:
    guide = PromptTemplate(
        STAGES_DIR / "_base/prompts/reconstruction_history.md"
    ).render(**_RUN_PATHS)

    for field in (
        "input_drawings",
        "snapshots",
        "open_tickets",
        "drawings",
        "semantics",
        "operations",
        "program_source",
        "verification",
        "responses",
    ):
        assert field in guide
    assert "structured submission by itself" in guide
    assert len(guide.split()) < 220


def test_the_system_prompt_explains_selective_history_navigation() -> None:
    rendered = build_system_prompt(ROLE_PATHS["coder"], _RUN_PATHS).text
    guide = PromptTemplate(
        STAGES_DIR / "_base/prompts/reconstruction_history.md"
    ).render(**_RUN_PATHS)

    assert guide in rendered
    assert (
        "drawings -> semantics -> operations -> coding + verification -> audit"
        in rendered
    )
    assert "never edit or print the whole file" in rendered
    assert ".snapshots[-2]" in rendered

    assert "jq -c" in rendered
    assert "members by name" in rendered
    assert len(rendered.split()) < 350


def test_round_instructions_do_not_repeat_the_reconstruction_guide(
    render_stage: Callable[..., str],
) -> None:
    rendered = render_stage("semantics")

    assert "## Reconstruction history" not in rendered
    assert "ReconstructionRun" not in rendered


def test_audit_explains_how_to_report_a_missing_semantic_feature(
    render_stage: Callable[..., str],
) -> None:
    rendered = render_stage(
        "audit",
        attempt_dir="/work/attempts/001",
        intermediate_returns_dir="unavailable",
        drawing_attempt_dir="/work/attempts/round_000/drawing/003",
        ticket_responses="[]",
    )

    assert "leave the `backtrace` empty" in rendered
    assert "whole semantics stage (`name: null`)" in rendered
    assert "propose one or more stable `sem_...` names" in rendered


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
        drawing_attempt_dir="/work/attempts/round_000/drawing/003",
        ticket_responses="[]",
    )

    assert "/work/attempts/001" in rendered
    assert "/work/attempts/round_000/drawing/003" in rendered
    assert f"Recorded directory: {returns_dir}" in rendered


def test_auditor_keeps_result_out_of_the_backtrace_graph() -> None:
    rendered = build_system_prompt(
        ROLE_PATHS["output_auditor"],
        {**_RUN_PATHS, "max_turns": "10"},
        AuditReport,
    ).text

    assert "`result` is the terminal export and is not a backtrace node" in rendered
    assert "whole coding output with `name: null`" in rendered


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

    assert "ret_<operation name without op_>" in rendered
    assert "# ----" not in rendered
    assert "Lxx-Lyy" not in rendered


def test_the_auditor_is_told_the_walk_rule_the_pipeline_would_reject_it_for() -> None:
    """A validator-only rule costs a retry the auditor cannot learn from."""
    rendered = PromptTemplate(ROLE_PATHS["output_auditor"]).render(
        output_schema="{}", max_turns="10"
    )

    assert "at most one hop inside any one stage" in rendered


def test_the_auditor_role_renders_its_contract() -> None:
    rendered = PromptTemplate(ROLE_PATHS["output_auditor"]).render(
        output_schema="SENTINEL_SCHEMA", max_turns="10"
    )

    assert "SENTINEL_SCHEMA" in rendered
    assert "$output_schema" not in rendered


@pytest.mark.parametrize("stage", ["semantics", "operations"])
def test_a_round_asks_for_a_revision_rather_than_a_whole_artifact(
    render_stage: Callable[..., str], stage: str
) -> None:
    rendered = render_stage(stage)

    assert "`edits`" in rendered
    assert "`deleted`" in rendered
    assert "deliverable" not in rendered


def test_the_coding_round_asks_only_for_ticket_responses(
    render_stage: Callable[..., str],
) -> None:
    """Coding revises the workspace, so its answer has no revision members to
    name -- and a member a stage cannot fill is one it can get wrong."""
    rendered = render_stage("coding")

    assert "`edits`" not in rendered
    assert "`deleted`" not in rendered
    assert "`rationale`" not in rendered
    assert "ticket responses and nothing else" in rendered


def test_the_drawing_round_uses_json_for_the_artifact_and_answer_for_tickets(
    render_stage: Callable[..., str],
) -> None:
    rendered = render_stage("drawings")

    assert "/work/drawing.json" in rendered
    assert "schema-valid working draft" in rendered
    assert "inspect the generated" in rendered
    assert "latest verified file" in rendered
    assert "substitute transport" in rendered


@pytest.mark.parametrize("stage", ["drawings", "semantics", "operations", "coding"])
def test_every_reasoning_round_carries_that_stage_s_guidelines(
    render_stage: Callable[..., str], stage: str
) -> None:
    guidelines = _guidelines(stage)

    rendered = render_stage(stage)
    assert guidelines in rendered


@pytest.mark.parametrize(
    "role",
    ["semantic_hypothesizer", "operation_planner", "coder", "cad_reconstructor"],
)
def test_a_proposer_role_says_who_it_is_and_leaves_the_rest_to_the_instruction(
    role: str,
) -> None:
    """What a stage must do belongs to the stage's instruction: a role holding
    all three would put the coding contract in front of a model that is still
    reading the drawing."""
    body = PromptTemplate(ROLE_PATHS[role]).path.read_text(encoding="utf-8")

    assert "$" not in body
    assert "run_shell" in body
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
            "verification_dir": "/work/attempts",
        },
        # What `create_agent` adds for a stage that answers structurally, and
        # what it adds for the coder, which does not.
        {
            "coding_output_path": "/work/model.py",
            "verification_dir": "/work/attempts",
            "output_schema": "SENTINEL_SCHEMA",
            "max_turns": "20",
        },
        {
            "coding_output_path": "/work/model.py",
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


def test_the_drawing_guidelines_describe_how_the_views_are_actually_separated() -> None:
    """The guidelines used to claim the three views sit on their own DXF
    layers. They do not: across all twenty sample drawings every entity is on
    layer `0`, and what does distinguish an edge is its linetype. The stage
    spent turns rediscovering that on every run, and separating the views is
    the drawing stage's job now."""
    guidelines = _guidelines("drawings")

    assert "hidden" in guidelines.lower()
    assert "linetype" in guidelines.lower()
    assert "layer `0`" in guidelines
    assert "does not separate views" in guidelines


def test_the_drawing_round_prioritises_an_early_verified_file(
    render_stage: Callable[..., str],
) -> None:
    rendered = render_stage("drawings")

    assert "Before half the turn budget" in rendered
    assert "never a substitute transport" in rendered
    assert '"$defs"' not in rendered


def test_the_drawing_guidelines_fix_the_raster_uv_origin_at_a_pixel_corner() -> None:
    guidelines = _guidelines("drawings")

    assert "lower-left corner of the bottom-left pixel" in guidelines
    assert "not at that pixel's centre" in guidelines
    assert "(c, h - r - 1)" in guidelines
    assert "(c + 0.5, h - r - 0.5)" in guidelines


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
        view.lower(): tuple(axis.lower() for axis in axes) for view, *axes in rows
    }
    expected = {view.value: axes for view, axes in VIEW_FRAME.items()}
    assert actual == expected


def test_the_plan_the_prompt_asks_for_is_the_one_the_schema_takes() -> None:
    """This test was the other way round while Phase 1 was measured: the DAG
    format was held out of the prompt so that the measurement saw the
    structured hypothesis and nothing else. The schema now carries it, so the
    guard becomes its opposite -- the prompt must name the two fields the
    contract will refuse a plan without."""
    guidelines = _guidelines("operations")

    assert "`depends_on`" in guidelines
    assert "`semantics`" in guidelines
    assert "`verb`" in guidelines
    assert set(Operation.model_fields) == {
        "name",
        "verb",
        "detail",
        "depends_on",
        "semantics",
    }


def test_the_coder_is_told_to_follow_the_operation_dag() -> None:
    guidelines = _guidelines("coding")

    assert "`depends_on`" in guidelines
    assert "JSON list order is not the build order" in guidelines


def test_a_stage_that_builds_in_3d_is_not_told_to_look_for_a_2d_entity() -> None:
    """`spline` is a `DrawnEntity`, seen in a view; the kind a `geometry` entry
    can hold is `bspline_curve` or `bspline_surface`. The coding guidelines
    named `spline`, so the coder was told to watch for a kind that cannot
    appear -- the drift the contract's two vocabularies invite."""
    flat_only = {member.value for member in DrawnEntity} - {
        member.value for member in GeometryKind
    }
    assert flat_only == {"spline", "polyline"}

    for stage in ("operations", "coding"):
        quoted = set(re.findall(r"`([a-z_]+)`", _guidelines(stage)))
        assert not quoted & flat_only, stage
