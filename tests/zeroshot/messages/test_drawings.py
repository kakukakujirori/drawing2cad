"""The drawing contract: what a sample may be handed in as, and what it may not.

The cases are the input shapes the pipeline has to survive -- one DXF sheet,
one PNG sheet, a sheet per view, a mixture of the two, a pictorial, and a raw
sheet kept alongside the views it was cut from.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from zeroshot.pipeline.messages.contracts.drawings import (
    ORTHOGRAPHIC_VIEWS,
    VIEW_FRAME,
    ClaimSource,
    Dimension,
    DrawingEvidence,
    DrawingSheet,
    DrawingSource,
    Parameter,
    View,
)


def _write(path: Path, content: bytes = b"data") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _circle(name: str = "ev_circle", source: str = "given") -> DrawingEvidence:
    return DrawingEvidence(
        name=name,
        view=View.FRONT,
        entity="circle",  # type: ignore[arg-type]
        edge_style="visible",  # type: ignore[arg-type]
        source=source,  # type: ignore[arg-type]
        parameters=[
            Parameter(name="center", values=[10.0, 20.0]),  # type: ignore[arg-type]
            Parameter(name="radius", values=[5.0]),  # type: ignore[arg-type]
        ],
    )


def test_an_unsplit_sheet_alone_is_a_drawing(tmp_path: Path) -> None:
    """`unknown` is the role of a sheet nobody has separated into views."""
    source = DrawingSource(
        sheets=[DrawingSheet(role=View.UNKNOWN, file=_write(tmp_path / "sheet.png"))]
    )

    assert [sheet.role for sheet in source.active_sheets()] == [View.UNKNOWN]
    assert source.orthographic() == []
    assert source.paths() == [tmp_path / "sheet.png"]


def test_views_may_be_split_across_formats(tmp_path: Path) -> None:
    source = DrawingSource(
        sheets=[
            DrawingSheet(role=View.FRONT, file=_write(tmp_path / "front.dxf")),
            DrawingSheet(role=View.TOP, file=_write(tmp_path / "top.png")),
        ]
    )

    assert source.views() == (View.FRONT, View.TOP)


def test_an_unsplit_sheet_and_its_views_may_be_held_at_once(tmp_path: Path) -> None:
    """The split does not consume the sheet it was made from.

    A split that went wrong has to stay pointable-at, so the unsplit sheet
    stays in the list and only `active_sheets()` decides what a reasoning
    stage reads.
    """
    source = DrawingSource(
        sheets=[
            DrawingSheet(role=View.UNKNOWN, file=_write(tmp_path / "sheet.png")),
            DrawingSheet(role=View.FRONT, file=_write(tmp_path / "front.png")),
        ]
    )

    assert [sheet.role for sheet in source.active_sheets()] == [View.FRONT]
    assert len(source.paths()) == 2


def test_two_unsplit_sheets_are_a_drawing_of_two_pages(tmp_path: Path) -> None:
    """A drawing set runs to several sheets, and none of them is the one."""
    source = DrawingSource(
        sheets=[
            DrawingSheet(
                role=View.UNKNOWN, label="p1", file=_write(tmp_path / "1.png")
            ),
            DrawingSheet(
                role=View.UNKNOWN, label="p2", file=_write(tmp_path / "2.png")
            ),
        ]
    )

    assert len(source.active_sheets()) == 2


def test_two_sheets_may_not_answer_to_one_name(tmp_path: Path) -> None:
    """The name is how a sheet is announced and staged, and a label may be the
    same word as another sheet's role -- so the check is on the name."""
    image = _write(tmp_path / "iso.png")

    with pytest.raises(ValidationError, match="more than one sheet named a"):
        DrawingSource(
            sheets=[
                DrawingSheet(role=View.ISOMETRIC, label="a", file=image),
                DrawingSheet(role=View.ISOMETRIC, label="a", file=image),
            ]
        )

    with pytest.raises(ValidationError, match="more than one sheet named front"):
        DrawingSource(
            sheets=[
                DrawingSheet(role=View.FRONT, file=image),
                DrawingSheet(role=View.ISOMETRIC, label="front", file=image),
            ]
        )


def test_a_drawing_needs_at_least_one_sheet() -> None:
    with pytest.raises(ValidationError, match="at least one sheet"):
        DrawingSource()


def test_a_view_may_not_be_given_twice(tmp_path: Path) -> None:
    front = _write(tmp_path / "front.png")

    with pytest.raises(ValidationError, match="more than one front view"):
        DrawingSource(
            sheets=[
                DrawingSheet(role=View.FRONT, file=front),
                DrawingSheet(role=View.FRONT, file=front),
            ]
        )


def test_pictorials_may_repeat(tmp_path: Path) -> None:
    """Two isometrics are a drawing convention, not a mistake -- but they need
    a label each, because the role alone no longer tells them apart."""
    image = _write(tmp_path / "iso.png")

    source = DrawingSource(
        sheets=[
            DrawingSheet(role=View.ISOMETRIC, label="from above", file=image),
            DrawingSheet(role=View.ISOMETRIC, label="from below", file=image),
        ]
    )

    assert len(source.sheets) == 2
    assert source.orthographic() == []


@pytest.mark.parametrize("suffix", [".txt", ".step", ".pdf"])
def test_a_sheet_must_be_a_drawing(tmp_path: Path, suffix: str) -> None:
    with pytest.raises(ValidationError, match="is not a drawing"):
        DrawingSheet(role=View.FRONT, file=_write(tmp_path / f"sheet{suffix}"))


def test_a_detail_must_name_a_view_the_drawing_holds(tmp_path: Path) -> None:
    image = _write(tmp_path / "sheet.png")

    with pytest.raises(ValidationError, match="which this drawing does not hold"):
        DrawingSource(
            sheets=[DrawingSheet(role=View.SECTION, detail_of=View.FRONT, file=image)]
        )


def test_only_orthographic_sheets_take_part_in_a_comparison(tmp_path: Path) -> None:
    """A detail is at its own magnification; a section's hatching answers to
    nothing in the parent's outline."""
    image = _write(tmp_path / "sheet.png")
    source = DrawingSource(
        sheets=[
            DrawingSheet(role=View.FRONT, file=image),
            DrawingSheet(role=View.TOP, file=image),
            DrawingSheet(role=View.DETAIL, detail_of=View.FRONT, file=image),
            DrawingSheet(role=View.SECTION, detail_of=View.TOP, file=image),
            DrawingSheet(role=View.ISOMETRIC, file=image),
        ]
    )

    assert [sheet.role for sheet in source.orthographic()] == [View.FRONT, View.TOP]
    assert View.DETAIL not in VIEW_FRAME
    assert View.SECTION not in VIEW_FRAME
    assert View.FRONT in VIEW_FRAME


def test_a_sheet_is_a_file_until_something_analyses_it(tmp_path: Path) -> None:
    """What the sheet draws and at what scale are an analyser's answers, and
    nothing answers them yet: the evidence a run makes live in the hypothesis."""
    sheet = DrawingSheet(role=View.FRONT, file=_write(tmp_path / "front.png"))

    assert set(DrawingSheet.model_fields) == {"role", "label", "detail_of", "file"}
    assert sheet.frame is not None


def test_a_split_drawing_is_told_only_the_frames_it_holds(tmp_path: Path) -> None:
    """Front and top, and no axis belonging to the four views it does not hold."""
    image = _write(tmp_path / "sheet.png")
    split = DrawingSource(
        sheets=[
            DrawingSheet(role=View.FRONT, file=image),
            DrawingSheet(role=View.TOP, file=image),
        ]
    )

    sentence = split.frame_sentence()

    assert "Front is right=+x, up=+y" in sentence
    assert "Top is right=+x, up=-z" in sentence
    assert "Back" not in sentence and "Left" not in sentence
    assert sentence.count(";") == 1


def test_a_drawing_nobody_split_is_told_all_six_frames(tmp_path: Path) -> None:
    """A live run sent the semantics stage "The axes are not yours to choose.
    ." -- no sheet was orthographic, and no wanted view rendered to nothing."""
    unsplit = DrawingSource(
        sheets=[
            DrawingSheet(
                role=View.UNKNOWN, label="drawing", file=_write(tmp_path / "d.dxf")
            ),
            DrawingSheet(
                role=View.PERSPECTIVE, label="hlg", file=_write(tmp_path / "p.png")
            ),
        ]
    )

    sentence = unsplit.frame_sentence()

    assert sentence.count(";") == 5
    for view in ORTHOGRAPHIC_VIEWS:
        assert view.value.capitalize() in sentence


def test_a_given_entry_must_state_the_numbers_the_input_gave_it() -> None:
    """'given' says the input handed the numbers over, so an empty one is a
    contradiction rather than an entry cited for what it proves."""
    with pytest.raises(ValidationError, match="sourced 'given' but states none"):
        DrawingEvidence(
            name="ev_bore",
            view=View.FRONT,
            entity="circle",  # type: ignore[arg-type]
            edge_style="visible",  # type: ignore[arg-type]
            source=ClaimSource.GIVEN,
            parameters=[],
        )


def test_an_entry_cited_for_what_it_proves_takes_no_numbers() -> None:
    proof = DrawingEvidence(
        name="ev_through",
        view=View.TOP,
        entity="line",  # type: ignore[arg-type]
        edge_style="hidden",  # type: ignore[arg-type]
        source=ClaimSource.DERIVED,
        parameters=[],
    )

    assert proof.parameters == []


def test_an_entry_states_every_number_its_entity_takes_or_none() -> None:
    with pytest.raises(ValidationError, match="radius"):
        DrawingEvidence(
            name="ev_bore",
            view=View.FRONT,
            entity="circle",  # type: ignore[arg-type]
            edge_style="visible",  # type: ignore[arg-type]
            source=ClaimSource.DERIVED,
            parameters=[
                Parameter(name="center", values=[0.0, 0.0]),  # type: ignore[arg-type]
            ],
        )
    assert _circle(source="derived").parameters


def test_a_printed_figure_keeps_what_the_sheet_says_beyond_its_size() -> None:
    """`4X ... THRU` decides four holes and a through hole. A contract that
    kept only the number would make one blind hole of them."""
    figure = Dimension(
        name="dim_mounting_holes",
        kind="diameter",  # type: ignore[arg-type]
        text="4X 12 THRU",
        nominal=12.0,
        quantity=4,
        note="THRU",
        targets=["ev_front_hole_left"],
    )

    assert figure.quantity == 4
    assert figure.note == "THRU"


def test_a_figure_measuring_no_linework_is_still_a_figure() -> None:
    """A dimension whose target could not be found is evidence that something
    was printed, and dropping it would lose that."""
    figure = Dimension(
        name="dim_overall_width",
        kind="linear",  # type: ignore[arg-type]
        text="100",
        nominal=100.0,
    )

    assert figure.targets == []
    assert figure.quantity == 1


def test_a_figure_takes_a_dim_name() -> None:
    with pytest.raises(ValidationError, match="not a usable dim name"):
        Dimension(
            name="ev_overall_width",
            kind="linear",  # type: ignore[arg-type]
            text="100",
            nominal=100.0,
        )


def test_a_sheet_is_a_file(tmp_path: Path) -> None:
    """A sheet with nothing behind it would be skipped everywhere it is read,
    and a drawing would come out of the message a view short in silence."""
    with pytest.raises(ValidationError, match="file"):
        DrawingSheet(role=View.FRONT)  # type: ignore[call-arg]
