from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from tests.zeroshot.verification.test_interpretation_validation import _page_layout
from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.stages.interpretation.contracts import (
    DrawingInterpretation,
    ProjectionMethod,
)
from zeroshot.pipeline.stages.interpretation.pictorial_placement import (
    pictorials_in_projected_places,
)


def _with_pictorial(
    workdir: Path,
    pictorial: tuple,
    drawn: tuple,
    *others: dict,
    transparent: bool = False,
) -> DrawingInterpretation:
    """`_page_layout` with front drawn as a box, plus an isometric view at `pictorial`
    whose drawing is the box `drawn`."""
    data = _page_layout(workdir, *others).model_dump()
    with Image.open(workdir / "page.png") as page:
        draw = ImageDraw.Draw(page)
        draw.rectangle((200, 1600, 1200, 2600), outline="black", width=4)
        draw.rectangle(drawn, outline="black", width=4)
        if transparent:  # black ink on a clear background
            page = Image.merge(
                "RGBA",
                (
                    *Image.new("RGB", page.size).split(),
                    page.convert("L").point(lambda v: 255 - v),
                ),
            )
        page.save(workdir / "page.png")
    data["views"].append(
        {
            "name": "view_iso",
            "role": "isometric",
            "file": "iso.png",
            "region": {"view": "view_page", "box_px": pictorial},
            "dimensions": [],
        }
    )
    return DrawingInterpretation.model_validate(data)


def _questions(interpretation: DrawingInterpretation, workdir: Path) -> list[str]:
    return pictorials_in_projected_places(
        interpretation,
        workdir=SandboxWorkdir(workdir),
        projection_method=ProjectionMethod.THIRD_ANGLE,
    )


def test_a_pictorial_drawn_as_a_missing_view_is_questioned(tmp_path: Path) -> None:
    """Right of front, with front's top and bottom, where no right view is."""
    interpretation = _with_pictorial(
        tmp_path, (1500, 1400, 2300, 2800), (1600, 1600, 2200, 2600)
    )

    (question,) = _questions(interpretation, tmp_path)
    assert question.startswith("Highly suspicious: view_iso is registered as isometric")
    assert "to the right of view_front" in question
    assert "third-angle projection draws the right view" in question
    assert "and no right view is registered" in question
    assert "view_iso should be the right view" in question
    assert (
        "compare feature positions and dimension spans along the axes shared "
        "with the other source views" in question
    )
    assert "it counts toward the match score, so the score may rise" in question


@pytest.mark.parametrize(
    ("pictorial", "drawn", "others", "transparent"),
    [
        # A right view is registered, so the place is not empty.
        (
            (1500, 1400, 2300, 2800),
            (1600, 1600, 2200, 2600),
            [{"role": "right", "box_px": (1300, 1400, 1450, 2800)}],
            False,
        ),
        # Its drawing ends nowhere near front's top or bottom.
        ((1500, 1400, 2300, 2800), (1600, 1450, 2200, 2100), [], False),
        # The same, with the clear background left out of the drawing.
        ((1500, 1400, 2300, 2800), (1600, 1450, 2200, 2100), [], True),
        # Up and to the right of front, the place pictorials take.
        ((1500, 100, 2300, 1300), (1600, 300, 2200, 1100), [], False),
    ],
    ids=["place_taken", "not_lined_up", "not_lined_up_transparent", "diagonal"],
)
def test_a_pictorial_apart_from_the_projected_places_is_not_questioned(
    tmp_path: Path,
    pictorial: tuple,
    drawn: tuple,
    others: list[dict],
    transparent: bool,
) -> None:
    interpretation = _with_pictorial(
        tmp_path, pictorial, drawn, *others, transparent=transparent
    )

    assert not _questions(interpretation, tmp_path)
