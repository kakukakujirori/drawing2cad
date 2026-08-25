"""The coder's file cut into sections and preserved across plan revisions.

The layout exists so that a plan revised in one place costs the coder one
localized edit rather than the whole part. The marker changes when its
instruction does, while the old code remains available in a reported range.
"""

from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

import pytest

from tests.zeroshot.contracts import feature, geometry, hypothesis
from zeroshot.pipeline.messages.contracts import (
    Operation,
    OperationPlan,
    OperationVerb,
    SemanticHypothesis,
)
from zeroshot.pipeline.verification.program_outline import (
    MARKER,
    ProgramOutline,
    render_outline_update,
    render_section_review,
    review_sections,
    section_variable,
    update_program_outline,
)


def op(
    name: str,
    *,
    needs: Sequence[str] = (),
    builds: Sequence[int | str] = (),
    detail: str = "",
    verb: OperationVerb = OperationVerb.EXTRUDE,
) -> Operation:
    return Operation(
        name=name,
        verb=verb,
        detail=detail or f"do {name}",
        depends_on=list(needs),
        semantics=[
            f"sem_feature_{held}" if isinstance(held, int) else held for held in builds
        ],
    )


def plan(*operations: Operation) -> OperationPlan:
    return OperationPlan(proposal=list(operations), rationale="because")


def held(radius: float = 5.0) -> SemanticHypothesis:
    return hypothesis(
        proposal=[
            feature(1, "plate", geometry=[geometry("plane", axis="z")]),
            feature(
                4, "bore", geometry=[geometry("cylinder", radius=radius, height=2.0)]
            ),
        ]
    )


def fill(source: str, name: str, code: str) -> str:
    """The file as it stands once the coder has written `code` under `name`."""
    written: list[str] = []
    inside = False
    for line in source.splitlines():
        found = MARKER.match(line)
        if found:
            if inside:
                written.append(code)
            inside = found[1] == name
        elif inside and not line.startswith("# ---- "):
            written.append(code)
            inside = False
        written.append(line)
    if inside:
        written.append(code)
    return "\n".join(written) + "\n"


def body(source: str, name: str) -> str:
    """Whatever code stands under `name`, with the machine's lines dropped."""
    written: list[str] = []
    inside = False
    for line in source.splitlines():
        if found := MARKER.match(line):
            inside = found[1] == name
        elif line.startswith("# ---- result"):
            inside = False
        elif inside and not line.startswith("# ---- "):
            written.append(line)
    return "\n".join(written).strip()


def markers(source: str) -> list[str]:
    return [found[1] for line in source.splitlines() if (found := MARKER.match(line))]


def test_the_sections_follow_the_build_order_not_the_writing_order() -> None:
    """The order the file is written in is the order it runs in, so this is
    what makes the plan's dependencies bind: code that reached forward to a
    step below it would raise a NameError rather than build the wrong part."""
    written, _ = update_program_outline(
        plan(op("op_bore", needs=["op_base"]), op("op_base")), held()
    )

    assert markers(written) == ["op_base", "op_bore"]


def test_a_step_that_did_not_move_keeps_its_code() -> None:
    """The whole point of the layout. A plan revised in one place must not cost
    the coder the fourteen steps that were already right."""
    two = plan(op("op_base", builds=[1]), op("op_bore", needs=["op_base"], builds=[4]))
    coded = fill(
        update_program_outline(two, held())[0],
        "op_base",
        "ret_base = cq.Workplane('XY').box(1)",
    )
    coded = fill(coded, "op_bore", "ret_bore = ret_base.hole(10)")

    revised, update = update_program_outline(
        plan(
            op("op_base", builds=[1]),
            op("op_bore", needs=["op_base"], builds=[4], detail="deeper"),
        ),
        held(),
        coded,
    )

    assert body(revised, "op_base") == "ret_base = cq.Workplane('XY').box(1)"
    assert [change.name for change in update.changed] == ["op_bore"]


def test_a_step_that_moved_keeps_its_code_and_reports_the_edit_range() -> None:
    """The marker changes, but the old implementation remains as local context."""
    one = plan(op("op_base", builds=[1], detail="extrude 25 mm"))
    coded = fill(
        update_program_outline(one, held())[0], "op_base", "ret_base = 'old'"
    )

    revised, update = update_program_outline(
        plan(op("op_base", builds=[1], detail="extrude 30 mm")), held(), coded
    )

    assert body(revised, "op_base") == "ret_base = 'old'"
    assert update.changed[0].first_line == 4
    assert update.changed[0].last_line == 4
    rendered = render_outline_update(update, "/work/model.py")
    assert "/work/model.py:L4-L4" in rendered
    assert "extrude 25 mm" in rendered
    assert "extrude 30 mm" in rendered


def test_a_measurement_that_moved_in_the_hypothesis_marks_the_step_changed() -> None:
    """The step's own wording is not the whole of what it was asked to do. A
    plan citing a semantic parameter says the same words after the radius changes, and
    the code written under it is answering the old number."""
    one = plan(
        op(
            "op_bore",
            builds=[4],
            detail="Cut a hole of sem_feature_4.geo_cylinder.radius",
        )
    )
    coded = fill(
        update_program_outline(one, held())[0],
        "op_bore",
        "ret_bore = ret_base.hole(10)",
    )

    revised, update = update_program_outline(one, held(radius=9.0), coded)

    assert [change.name for change in update.changed] == ["op_bore"]
    assert body(revised, "op_bore") == "ret_bore = ret_base.hole(10)"


def test_changing_a_section_that_was_never_written_is_not_reported() -> None:
    """An empty section has no old implementation to localize for editing."""
    seeded, _ = update_program_outline(
        plan(op("op_base", detail="extrude 25 mm")), held()
    )

    _, update = update_program_outline(
        plan(op("op_base", detail="extrude 30 mm")), held(), seeded
    )

    assert update.changed == []
    assert render_outline_update(update, "/work/model.py") == ""


def test_a_step_the_plan_no_longer_holds_is_dropped_and_named() -> None:
    """Dropped because the file is the plan made buildable and the step is not
    in the plan; named because it is the coder's work being thrown away."""
    two = plan(op("op_base"), op("op_spare"))
    coded = fill(update_program_outline(two, held())[0], "op_spare", "part = 'work'")

    revised, update = update_program_outline(plan(op("op_base")), held(), coded)

    assert update.removed == ["op_spare"]
    assert "op_spare" not in revised
    assert "op_spare" in render_outline_update(update, "/work/model.py")


def test_the_preamble_survives_every_rewrite() -> None:
    """The imports are the coder's, and they are not any one step's."""
    seeded, _ = update_program_outline(plan(op("op_base")), held())
    coded = seeded.replace(
        "import cadquery as cq", "import cadquery as cq\nimport math"
    )

    revised, _ = update_program_outline(
        plan(op("op_base"), op("op_bore")), held(), coded
    )

    assert "import math" in revised


def test_code_under_a_damaged_marker_is_not_thrown_away() -> None:
    """A marker the coder has edited no longer names a section, so the code
    below it belongs to the section above and travels with it. The review says
    the marker is gone; nothing says the work is."""
    two = plan(op("op_base", builds=[1]), op("op_bore", needs=["op_base"], builds=[4]))
    coded = fill(update_program_outline(two, held())[0], "op_base", "part = 'kept'")
    damaged = coded.replace("# ---- op_bore ", "# op_bore ")

    revised, _ = update_program_outline(two, held(), damaged)

    assert "part = 'kept'" in revised
    assert render_section_review(review_sections(damaged, two)).count("op_bore") == 2


def test_an_unwritten_section_is_progress_and_not_a_fault() -> None:
    two = plan(op("op_base"), op("op_bore", needs=["op_base"]))
    seeded, _ = update_program_outline(two, held())

    review = review_sections(seeded, two)

    assert review.empty == ["op_base", "op_bore"]
    assert review.nothing_written
    assert "op_base, op_bore still to write" in render_section_review(review)


def test_a_file_with_something_in_it_is_worth_building() -> None:
    """`nothing_written` is what spares a sandbox run on a file of markers. A
    part half written is not that: the coder is told to write early and read
    what comes back."""
    two = plan(op("op_base"), op("op_bore", needs=["op_base"]))
    coded = fill(update_program_outline(two, held())[0], "op_base", "part = 1")

    review = review_sections(coded, two)

    assert not review.nothing_written
    assert "sections 1/2" in render_section_review(review)


def test_a_file_written_over_wholesale_is_still_built() -> None:
    """The markers are gone, so nothing can be attributed -- but something is
    there, and something is judged by building it. Refusing here would deny a
    working part its verification over a missing comment."""
    one = plan(op("op_base"))

    review = review_sections("result = 1\n", one)

    assert not review.nothing_written
    assert review.missing == ["op_base"]
    assert "marker for op_base is gone" in render_section_review(review)


def test_a_comment_is_not_code() -> None:
    """The coder is asked to leave a note where a step defeated it. A note is
    not the step, and the section is still to write."""
    one = plan(op("op_base"))
    coded = fill(
        update_program_outline(one, held())[0],
        "op_base",
        "# could not work out the profile",
    )

    assert review_sections(coded, one).empty == ["op_base"]


def test_sections_put_back_in_the_wrong_order_are_reported() -> None:
    """The order is what makes the dependencies bind, and it is the one thing
    the machine cannot repair without throwing away code."""
    two = plan(op("op_base"), op("op_bore", needs=["op_base"]))
    seeded, _ = update_program_outline(two, held())
    _, first, second, tail = seeded.split("\n\n")
    swapped = "import cadquery as cq\n\n" + second + "\n" + first + "\n" + tail

    assert review_sections(swapped, two).out_of_order
    assert "build order" in render_section_review(review_sections(swapped, two))


def test_a_file_that_does_not_compile_still_reports_progress() -> None:
    """Read line by line rather than parsed. The coder leaves a bracket open
    and needs to be told what it built, not to have the report crash too."""
    one = plan(op("op_base"))
    coded = fill(
        update_program_outline(one, held())[0], "op_base", "part = cq.Workplane('XY'"
    )

    assert review_sections(coded, one).empty == []


def test_a_step_whose_wording_holds_a_percent_sign_is_written_out() -> None:
    """The heading is the plan's own prose, and a plan is free to say "50% of
    the width". Nothing here may treat that text as a format string."""
    one = plan(op("op_base", detail="Extrude to 50% of the width"))

    written, _ = update_program_outline(one, held())

    assert "50% of the width" in written
    assert review_sections(written, one).missing == []


@pytest.mark.parametrize("times", [2, 3])
def test_laying_a_file_out_again_changes_nothing(times: int) -> None:
    """The layout runs on every entry to the coding stage, including the ones
    that changed nothing. A file that moved on its own would clear sections
    nobody revised."""
    two = plan(op("op_base"), op("op_bore", needs=["op_base"]))
    written = fill(update_program_outline(two, held())[0], "op_base", "part = 1")

    for _ in range(times):
        written, update = update_program_outline(two, held(), written)
        assert render_outline_update(update, "/work/model.py") == ""

    assert body(written, "op_base") == "part = 1"


def test_the_epilogue_outlives_the_step_that_happens_to_be_last() -> None:
    """`result` is what the operations came to, not one of them. Riding on the
    last section, it would be cleared by a revision to a step it has nothing to
    do with, and the program would lose its answer without being told."""
    one = plan(op("op_base", detail="extrude 25 mm"))
    seeded, _ = update_program_outline(one, held())
    coded = fill(seeded, "op_base", "part = 1").replace(
        "# ---- result (not a step; kept across every revision)",
        "# ---- result (not a step; kept across every revision)\nresult = part",
    )

    revised, update = update_program_outline(
        plan(op("op_base", detail="extrude 30 mm")), held(), coded
    )

    assert [change.name for change in update.changed] == ["op_base"]
    assert "result = part" in revised
    assert body(revised, "op_base") == "part = 1"


def test_the_epilogue_is_not_a_section() -> None:
    """It answers to no step, so it is neither expected, missing nor empty --
    and nothing under it may be read as the last step's code."""
    one = plan(op("op_base"))
    written, _ = update_program_outline(one, held())

    assert review_sections(written, one).expected == ["op_base"]
    assert body(written, "op_base") == ""


def test_a_long_instruction_remains_one_machine_owned_line() -> None:
    """Long instructions may be ungainly, but never become fake boundaries."""
    one = plan(
        op(
            "op_base",
            detail="Extrude the nut-capture outline 25 mm along +z to form the "
            "lower-eye base plate of the carrier, which the bore goes through",
        )
    )
    written, _ = update_program_outline(one, held())

    owned = [line for line in written.splitlines() if line.startswith("# ---- op_")]
    assert len(owned) == 1
    assert "lower-eye base plate" in owned[0]
    assert "-> ret_base" in owned[0]


def test_a_marker_the_coder_rewrote_preserves_and_localizes_that_section() -> None:
    """The marker is the whole record of what the code under it answers. One
    the coder has reworded no longer says what the plan says, and code kept
    under it would be code answering an instruction nobody issued."""
    one = plan(op("op_base", detail="extrude 25 mm"))
    coded = fill(update_program_outline(one, held())[0], "op_base", "part = 1")
    reworded = coded.replace("extrude 25 mm", "extrude 30 mm")

    revised, update = update_program_outline(one, held(), reworded)

    assert [change.name for change in update.changed] == ["op_base"]
    assert body(revised, "op_base") == "part = 1"


def test_a_legacy_multiline_marker_migrates_without_deleting_code() -> None:
    """Phase 3 files get the new boundary format without losing their work."""
    one = plan(op("op_base", builds=[1], detail="extrude 25 mm"))
    legacy = (
        "import cadquery as cq\n\n"
        "# ---- op_base extrude (needs nothing; builds sem_feature_1)\n"
        "# ---- extrude 25 mm\n"
        "ret_base = cq.Workplane('XY').box(1)\n\n"
        "# ---- result (not a step; kept across every revision)\n"
        "result = ret_base\n"
    )

    revised, update = update_program_outline(one, held(), legacy)

    assert body(revised, "op_base") == "ret_base = cq.Workplane('XY').box(1)"
    assert [change.name for change in update.changed] == ["op_base"]
    assert "extrude 25 mm" in update.changed[0].previous_instruction
    assert len([line for line in revised.splitlines() if MARKER.match(line)]) == 1


def test_section_variable_uses_the_operation_name_without_duplicate_prefix() -> None:
    assert section_variable("op_base_plate") == "ret_base_plate"


def test_an_outline_nobody_prepared_is_read_against_nothing(tmp_path) -> None:
    """The standalone verification tool is handed a program somebody wrote by
    hand. There is no plan it was meant to follow, so there is no reading to
    give -- as opposed to a reading that says nothing has been written."""
    path = tmp_path / "model.py"
    outline = ProgramOutline(path)
    assert outline.review() is None
    assert outline.own_write is None

    path.write_text("result = 1\n", encoding="utf-8")
    assert outline.review() is None


def test_preparing_an_outline_records_what_it_wrote(tmp_path) -> None:
    """The digest is offered so that whoever watches the file for a turn's work
    can tell this write from the coder's, and it has to move with the plan: a
    revision changes the file, and a watcher still holding the old digest would
    take the new one for somebody's turn."""
    path = tmp_path / "model.py"
    outline = ProgramOutline(path)

    outline.prepare(plan(op("op_base", detail="extrude 25 mm")), held())
    first = outline.own_write
    assert first is not None
    assert outline.review() is not None

    outline.prepare(plan(op("op_base", detail="extrude 30 mm")), held())

    assert outline.own_write != first
    assert outline.own_write == sha256(path.read_bytes()).hexdigest()


def test_rejects_a_path_that_is_not_absolute() -> None:
    """The outline holds a host path, and the same program is named by a
    sandbox path everywhere the model sees it. A relative one says which side
    it belongs to only by where it is read."""
    with pytest.raises(ValueError, match="must be an absolute path"):
        ProgramOutline(Path("model.py"))
