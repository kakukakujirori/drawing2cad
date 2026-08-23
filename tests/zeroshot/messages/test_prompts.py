import re
from pathlib import Path

import pytest

from zeroshot.pipeline.messages import PromptTemplate, instruction_text
from zeroshot.pipeline.messages.contracts import (
    VIEW_FRAME,
    DrawnEntity,
    GeometryKind,
    Operation,
    view_frame_sentence,
)


def _write(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


# What the graph supplies to every instruction, whichever stage asked for it.
_RUN_PATHS = {"output_path": "/work/model.py", "verification_dir": "/work/attempts"}


def _guidelines(stage: str) -> str:
    """A stage's guidelines rendered the way `instruction_text` renders them.

    It injects `$view_frame` on top of the run's paths, so a test that renders
    the file directly has to supply it too or it is not looking at the text the
    stage was actually given.
    """
    return PromptTemplate(f"instructions/{stage}/guidelines").render(
        **_RUN_PATHS, view_frame=view_frame_sentence()
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
        ("semantics/initial", {}),
        ("semantics/targeted_redo", {"rationale": "wrong feature"}),
        ("semantics/review", {}),
        (
            "operations/initial",
            {"semantic_hypothesis": "a bracket"},
        ),
        (
            "operations/targeted_redo",
            {"rationale": "missing hole", "semantic_hypothesis": "a bracket"},
        ),
        (
            "operations/upstream_changed",
            {
                "rationale": "wrong base feature",
                "semantic_hypothesis": "a revised bracket",
            },
        ),
        (
            "operations/review",
            {"semantic_hypothesis": "a bracket"},
        ),
        (
            "coding/initial",
            {
                "semantic_hypothesis": "a bracket",
                "operation_plan": "extrude the base",
            },
        ),
        (
            "coding/targeted_redo",
            {
                "rationale": "the hole is misplaced",
                "semantic_hypothesis": "a bracket",
                "operation_plan": "extrude the base",
            },
        ),
        (
            "coding/upstream_changed",
            {
                "rationale": "wrong base feature",
                "semantic_hypothesis": "a revised bracket",
                "operation_plan": "rebuild the base",
            },
        ),
        (
            "audit",
            {
                "semantic_hypothesis": "a bracket",
                "operation_plan": "extrude the base",
                "output_path": "/work/model.py",
                "report": "VERIFIED",
                "attempt_dir": "/work/attempts/000",
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


def test_placeholders_are_filled_from_the_context() -> None:
    """The run's paths reach the guidelines the coding instruction carries."""
    rendered = instruction_text(
        "coding/initial",
        **_RUN_PATHS,
        semantic_hypothesis="a bracket",
        operation_plan="extrude the base",
    )

    assert "/work/model.py" in rendered
    assert "/work/attempts" in rendered
    assert "$output_path" not in rendered
    assert "$verification_dir" not in rendered


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


# The ways a stage can be entered, and what each of them needs filled in.
_STAGE_ENTRIES = {
    "semantics": ("initial", "targeted_redo"),
    "operations": ("initial", "targeted_redo", "upstream_changed"),
    "coding": ("initial", "targeted_redo", "upstream_changed"),
}
_ENTRY_CONTEXT = {
    "rationale": "the hole is misplaced",
    "semantic_hypothesis": "a bracket",
    "operation_plan": "extrude the base",
}


@pytest.mark.parametrize("stage", list(_STAGE_ENTRIES))
def test_every_way_into_a_stage_carries_that_stage_s_guidelines(stage: str) -> None:
    """A redo enters on its own instruction, and a compacted thread keeps none
    of the opening one, so the contracts cannot ride on first entry alone."""
    guidelines = _guidelines(stage)

    for entry in _STAGE_ENTRIES[stage]:
        rendered = instruction_text(f"{stage}/{entry}", **_RUN_PATHS, **_ENTRY_CONTEXT)
        assert guidelines in rendered, entry


def test_a_review_is_not_given_the_guidelines_of_what_it_reviews() -> None:
    """The reviewer is a different agent with a different job; the proposer's
    standing advice is not addressed to it."""
    guidelines = _guidelines("operations")

    rendered = instruction_text(
        "operations/review", **_RUN_PATHS, semantic_hypothesis="a bracket"
    )

    assert guidelines not in rendered


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
    "name", ["semantics/initial", "operations/initial", "coding/initial", "audit"]
)
def test_every_stage_that_handles_a_coordinate_is_given_the_frame(name: str) -> None:
    """Semantics reports sheet coordinates tagged with a view; every stage after
    it reads them back. A stage left without the frame does not fail -- it
    guesses, and `top` sheet-up is model `-z`, so it guesses wrong in silence.

    The sentence is rendered from `VIEW_FRAME`, so this asserts that it reaches
    the stage rather than that a copy of it is still correct."""
    rendered = instruction_text(
        name, **_RUN_PATHS, **_ENTRY_CONTEXT, report="VERIFIED", attempt_dir="/work/a"
    )

    for view, (right, up, _) in VIEW_FRAME.items():
        assert f"{view.value.capitalize()} is right={right}, up={up}" in rendered


def test_the_frame_reaches_a_redo_as_well_as_a_first_entry() -> None:
    """A redo enters on its own instruction, and the coordinates it is handed
    are the same ones."""
    rendered = instruction_text(
        "operations/targeted_redo", **_RUN_PATHS, **_ENTRY_CONTEXT
    )

    assert "Top is right=+x, up=-z" in rendered


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
        "id",
        "verb",
        "detail",
        "depends_on",
        "semantics",
    }


def test_the_coder_is_told_the_plan_is_already_ordered() -> None:
    """The order is derived, not written, so a coder that reordered it would be
    undoing the one thing the graph was taken for."""
    assert "already in build order" in _guidelines("coding")


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


def test_the_plan_redo_does_not_credit_the_step_that_found_the_fault() -> None:
    """Two things send the plan back: the audit, and the coverage check that
    runs before any audit does. One instruction serves both, so it must name
    neither -- it opened with "The audit found", which would have had the
    planner answering a verdict nobody had reached."""
    body = PromptTemplate("instructions/operations/targeted_redo").path.read_text(
        encoding="utf-8"
    )

    assert "audit" not in body.lower()
    assert "$rationale" in body
