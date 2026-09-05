import re
from pathlib import Path

import pytest

from zeroshot.pipeline.messages import (
    PromptTemplate,
    instruction_section,
    instruction_text,
    system_prompt_text,
)
from zeroshot.pipeline.messages.contracts import (
    VIEW_FRAME,
    DrawingSheet,
    DrawingSource,
    DrawnEntity,
    GeometryKind,
    Operation,
)
from zeroshot.pipeline.messages.contracts.audit import AuditReport
from zeroshot.pipeline.messages.contracts.reconstruction import (
    ReconstructionRun,
    ReconstructionSnapshot,
    Ticket,
    TicketResponse,
)
from zeroshot.pipeline.verification._run_program import INTERMEDIATE_RETURNS_DIR


def _write(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


# What the graph supplies to every instruction, whichever stage asked for it.
_RUN_PATHS = {
    "output_path": "/work/model.py",
    "verification_dir": "/work/attempts",
    "reconstruction_path": "/work/reconstruction.json",
}


_AN_UNREAD_PAGE = DrawingSource(
    sheets=[
        DrawingSheet(
            name="sheet_page",
            role="unknown",
            label=None,
            derived_from=None,
            file="/work/inputs/drawing.dxf",
            origin=None,
            evidence=[],
            dimensions=[],
        )
    ]
)


def _round_context(**varying: str) -> dict[str, str]:
    """What the graph gives every round instruction, plus what a test varies."""
    return {
        **_RUN_PATHS,
        "assigned_tickets": "ticket_initial",
        "intermediate_returns": "",
        "view_frame": _AN_UNREAD_PAGE.frame_sentence(),
        **varying,
    }


def _guidelines(stage: str) -> str:
    """A stage's guidelines rendered the way `instruction_text` renders them.

    The graph supplies `$view_frame` on top of the run's paths, so a test that
    renders the file directly has to supply it too or it is not looking at the
    text the stage was actually given.
    """
    return PromptTemplate(f"instructions/{stage}/guidelines").render(
        **_RUN_PATHS, view_frame=_AN_UNREAD_PAGE.frame_sentence()
    )


def test_a_packaged_prompt_is_addressed_by_name() -> None:
    """Names, not paths: Hydra moves the process into the job's output
    directory before a run, so a relative path in a config would not resolve."""
    prompt = PromptTemplate("roles/coder")

    assert prompt.path.name == "coder.md"
    assert prompt.path.is_file()


@pytest.mark.parametrize(
    ("name", "context"),
    [
        ("semantics/round", {"current_round": "0", "assigned_tickets": "t_a"}),
        ("operations/round", {"current_round": "0", "assigned_tickets": "t_a"}),
        ("coding/round", {"current_round": "0", "assigned_tickets": "t_a"}),
        (
            "audit/round",
            {
                "current_round": "0",
                "attempt_dir": "/work/attempts/000",
                "intermediate_returns": "",
            },
        ),
    ],
)
def test_stage_instruction_prompts_match_the_invocation_reasons(
    name: str,
    context: dict[str, str],
) -> None:
    rendered = instruction_text(name, **{**_RUN_PATHS, **context})

    assert rendered
    assert "$" not in rendered


def test_reconstruction_guide_tracks_the_durable_contract() -> None:
    guide = PromptTemplate("roles/reconstruction_history").render(**_RUN_PATHS)

    for contract in (ReconstructionRun, ReconstructionSnapshot, Ticket, TicketResponse):
        for field in contract.model_fields:
            assert f"`{field}`" in guide


def test_the_shared_role_explains_selective_history_navigation() -> None:
    rendered = system_prompt_text("cad_reconstructor", _RUN_PATHS)
    guide = PromptTemplate("roles/reconstruction_history").render(**_RUN_PATHS)

    assert guide in rendered
    assert "semantics -> operations -> coding + verification -> audit" in rendered
    assert "ReconstructionRun" in rendered
    assert "Do not print the whole history file" in rendered
    assert "never `cat` it" in rendered
    assert ".snapshots[-2]" in rendered
    assert "diff -u" in rendered

    # Index of names first, then one member by name: no recipe dumps an artifact.
    assert "jq -c" in rendered
    assert "[.geometry[].name]" in rendered
    assert 'select(.name == "sem_main_bore")' in rendered
    assert "'.snapshots[-1].semantics'" not in rendered
    assert "'.snapshots[-1].operations'" not in rendered


def test_round_instructions_do_not_repeat_the_reconstruction_guide() -> None:
    rendered = instruction_text("semantics/round", **_round_context(current_round="1"))

    assert "## Reconstruction history" not in rendered
    assert "ReconstructionRun" not in rendered


def test_audit_explains_how_to_report_a_missing_semantic_feature() -> None:
    rendered = instruction_text(
        "audit/round",
        **_RUN_PATHS,
        current_round="1",
        attempt_dir="/work/attempts/001",
        intermediate_returns="",
    )

    assert "leave the `backtrace` empty" in rendered
    assert "whole semantics stage (`name: null`)" in rendered
    assert "propose one or more stable `sem_...` names" in rendered


def test_the_returns_section_says_what_the_directory_is_for() -> None:
    """The layout line alone does not say which `ret_` a defect belongs to."""
    section = instruction_section(
        "audit/intermediate_returns",
        True,
        returns_dir=INTERMEDIATE_RETURNS_DIR,
    )

    assert INTERMEDIATE_RETURNS_DIR in section
    assert "ret_" in section
    assert "what the plan meant it to" in section


def test_a_build_that_wrote_no_returns_leaves_the_section_out() -> None:
    assert instruction_section("audit/intermediate_returns", False) == ""


def test_the_audit_reads_the_attempt_directory_the_build_actually_wrote() -> None:
    """The per-operation views are what localise a defect to one `ret_...`, and
    the auditor only looks in a directory it was told about."""
    section = instruction_section(
        "audit/intermediate_returns",
        True,
        returns_dir=INTERMEDIATE_RETURNS_DIR,
    )
    rendered = instruction_text(
        "audit/round",
        **_RUN_PATHS,
        current_round="1",
        attempt_dir="/work/attempts/001",
        intermediate_returns=section,
    )

    assert "/work/attempts/001" in rendered
    assert section in rendered


def test_auditor_keeps_result_out_of_the_backtrace_graph() -> None:
    rendered = system_prompt_text(
        "output_auditor",
        {**_RUN_PATHS, "max_turns": "10"},
        AuditReport,
    )

    assert "`result` is the terminal export and is not a backtrace node" in rendered
    assert "whole coding output with `name: null`" in rendered


def test_placeholders_are_filled_from_the_context() -> None:
    """The run's paths reach the guidelines the coding instruction carries."""
    rendered = instruction_text("coding/round", **_round_context(current_round="3"))

    assert "/work/model.py" in rendered
    assert "/work/attempts" in rendered
    assert "$output_path" not in rendered
    assert "$verification_dir" not in rendered


def test_the_coding_round_carries_the_history_and_result_contract() -> None:
    rendered = instruction_text("coding/round", **_round_context(current_round="2"))

    assert "ret_<operation name without op_>" in rendered
    assert "# ----" not in rendered
    assert "Lxx-Lyy" not in rendered


def test_the_auditor_is_told_the_walk_rule_the_pipeline_would_reject_it_for() -> None:
    """A validator-only rule costs a retry the auditor cannot learn from."""
    rendered = PromptTemplate("roles/output_auditor").render(
        output_schema="{}", max_turns="10"
    )

    assert "at most one hop inside any one stage" in rendered


@pytest.mark.parametrize(
    "role",
    [
        "roles/semantic_reviewer",
        "roles/operation_reviewer",
        "roles/output_auditor",
    ],
)
def test_structured_output_roles_render_their_contract(role: str) -> None:
    rendered = PromptTemplate(role).render(
        output_schema="SENTINEL_SCHEMA", max_turns="10"
    )

    assert "SENTINEL_SCHEMA" in rendered
    assert "$output_schema" not in rendered


@pytest.mark.parametrize("stage", ["semantics", "operations"])
def test_a_round_asks_for_a_revision_rather_than_a_whole_artifact(stage: str) -> None:
    rendered = instruction_text(f"{stage}/round", **_round_context(current_round="1"))

    assert "`edits`" in rendered
    assert "`deleted`" in rendered
    assert "deliverable" not in rendered


def test_the_coding_round_asks_only_for_ticket_responses() -> None:
    """Coding revises the workspace, so its answer has no revision members to
    name -- and a member a stage cannot fill is one it can get wrong."""
    rendered = instruction_text("coding/round", **_round_context(current_round="1"))

    assert "`edits`" not in rendered
    assert "`deleted`" not in rendered
    assert "`rationale`" not in rendered
    assert "ticket responses and nothing else" in rendered


@pytest.mark.parametrize("stage", ["semantics", "operations", "coding"])
def test_every_reasoning_round_carries_that_stage_s_guidelines(stage: str) -> None:
    guidelines = _guidelines(stage)

    rendered = instruction_text(f"{stage}/round", **_round_context(current_round="1"))
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
    body = PromptTemplate(f"roles/{role}").path.read_text(encoding="utf-8")

    assert "$" not in body
    assert "run_shell" in body
    assert "result` variable" not in body
    assert "try-except" not in body


def test_the_merged_role_renders_the_same_text_for_every_stage_that_shares_it() -> None:
    """One thread of thought needs one system prompt: the stages differ in
    turn budget and answer contract, so a prompt that took either as a
    placeholder would come out different for each of them."""
    body = PromptTemplate("roles/cad_reconstructor").path.read_text(encoding="utf-8")

    assert "$output_schema" not in body
    assert "$max_turns" not in body

    stage_contexts = [
        {"output_path": "/work/model.py", "verification_dir": "/work/attempts"},
        # What `create_agent` adds for a stage that answers structurally, and
        # what it adds for the coder, which does not.
        {
            "output_path": "/work/model.py",
            "verification_dir": "/work/attempts",
            "output_schema": "SENTINEL_SCHEMA",
            "max_turns": "20",
        },
        {
            "output_path": "/work/model.py",
            "verification_dir": "/work/attempts",
            "max_turns": "10",
        },
    ]
    rendered = {
        PromptTemplate("roles/cad_reconstructor").render(**context)
        for context in stage_contexts
    }

    assert len(rendered) == 1
    assert "$" not in rendered.pop()


def test_a_missing_value_is_refused(tmp_path: Path) -> None:
    """`substitute`, not `safe_substitute`: a value we forgot to pass must not
    reach the model as the literal `$verification_dir`."""
    prompt = PromptTemplate(str(_write(tmp_path / "p.md", "$here and $there")))

    with pytest.raises(KeyError):
        prompt.render(here="only one")


def test_an_unused_value_is_ignored(tmp_path: Path) -> None:
    """One context serves every stage, so a prompt may use none of it."""
    prompt = PromptTemplate(str(_write(tmp_path / "p.md", "no placeholders")))

    assert prompt.render(output_path="/work/model.py") == "no placeholders"


def test_braces_survive_rendering(tmp_path: Path) -> None:
    """Why `$name` and not `{name}`: prompts carry CadQuery snippets."""
    body = 'result = cq.Workplane().box(**{"length": 1})'
    prompt = PromptTemplate(str(_write(tmp_path / "p.md", body)))

    assert prompt.render() == body


def test_surrounding_whitespace_does_not_reach_the_model(tmp_path: Path) -> None:
    """Whether a file ends in a newline is an editor's decision, and must not
    silently change the bytes the model is sent."""
    bare = PromptTemplate(str(_write(tmp_path / "bare.md", "instructions")))
    padded = PromptTemplate(str(_write(tmp_path / "padded.md", "\ninstructions\n\n")))

    assert bare.render() == padded.render() == "instructions"


def test_an_unknown_prompt_is_refused_before_the_run() -> None:
    with pytest.raises(ValueError, match="prompt not found"):
        PromptTemplate("no_such_prompt")


def test_the_digest_follows_the_file(tmp_path: Path) -> None:
    """The prompt is the experiment's main variable, so a run's audit trail
    needs a way to say which text it used."""
    path = _write(tmp_path / "p.md", "first")
    prompt = PromptTemplate(str(path))
    before = prompt.sha256

    _write(path, "second")

    assert prompt.sha256 != before


def test_the_semantics_guidelines_describe_how_the_views_are_actually_separated() -> (
    None
):
    """The guidelines used to claim the three views sit on their own DXF
    layers. They do not: across all twenty sample drawings every entity is on
    layer `0`, and what does distinguish an edge is its linetype. The stage
    spent turns rediscovering that on every run."""
    guidelines = _guidelines("semantics")

    assert "HIDDEN" in guidelines
    assert "linetype" in guidelines.lower()
    assert "layer `0`" in guidelines
    assert "not separated by layer" in guidelines


@pytest.mark.parametrize(
    "name",
    [
        "semantics/round",
        "operations/round",
        "coding/round",
        "audit/round",
    ],
)
def test_every_stage_that_handles_a_coordinate_is_given_the_frame(name: str) -> None:
    """Semantics reports sheet coordinates tagged with a view; every stage after
    it reads them back. A stage left without the frame does not fail -- it
    guesses, and `top` sheet-up is model `-z`, so it guesses wrong in silence.

    The sentence is rendered from `VIEW_FRAME`, so this asserts that it reaches
    the stage rather than that a copy of it is still correct."""
    rendered = instruction_text(
        name,
        **_round_context(current_round="0", attempt_dir="/work/a"),
    )

    for view, (right, up, _) in VIEW_FRAME.items():
        assert f"{view.value.capitalize()} is right={right}, up={up}" in rendered


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
