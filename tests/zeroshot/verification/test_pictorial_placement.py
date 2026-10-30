from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from tests.zeroshot.contracts import UNTURNED
from tests.zeroshot.verification.test_interpretation_validation import _page_layout
from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.stages.interpretation.contracts import (
    ORTHOGRAPHIC_VIEWS,
    DrawingInterpretation,
    ProjectionMethod,
    View,
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


@pytest.mark.parametrize(
    "role", [View.ISOMETRIC, View.PERSPECTIVE, View.DETAIL, View.SECTION, View.UNKNOWN]
)
@pytest.mark.parametrize(
    "method, expected",
    [(ProjectionMethod.THIRD_ANGLE, "right"), (ProjectionMethod.FIRST_ANGLE, "left")],
)
def test_a_pictorial_drawn_as_a_missing_view_is_questioned(
    tmp_path: Path, role, method, expected
) -> None:
    """Right of front, with front's top and bottom, where no right view is."""
    interpretation = _with_pictorial(
        tmp_path, (1500, 1400, 2300, 2800), (1600, 1600, 2200, 2600)
    )

    interpretation.views[-1].role = role
    (question,) = pictorials_in_projected_places(
        interpretation, workdir=SandboxWorkdir(tmp_path), projection_method=method
    )
    assert f"view_iso is registered as {role.value}" in question
    assert "to the right of view_front" in question
    assert (
        f"{method.value.replace('_', '-')} projection draws the {expected} view"
        in question
    )
    assert f"and no {expected} view is registered" in question
    assert question.startswith("Highly suspicious:")
    assert f"view_iso should be the {expected} view" in question
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
@pytest.mark.parametrize(
    "role", [View.ISOMETRIC, View.PERSPECTIVE, View.DETAIL, View.SECTION, View.UNKNOWN]
)
def test_a_pictorial_apart_from_the_projected_places_is_not_questioned(
    tmp_path: Path,
    pictorial: tuple,
    drawn: tuple,
    others: list[dict],
    transparent: bool,
    role: View,
) -> None:
    interpretation = _with_pictorial(
        tmp_path, pictorial, drawn, *others, transparent=transparent
    )
    interpretation.views[-1].role = role

    assert not _questions(interpretation, tmp_path)


def test_a_top_view_below_front_in_first_angle_is_questioned(tmp_path):
    interpretation = _with_pictorial(
        tmp_path, (100, 2700, 1300, 2990), (200, 2750, 1200, 2980)
    )
    (question,) = pictorials_in_projected_places(
        interpretation,
        workdir=SandboxWorkdir(tmp_path),
        projection_method=ProjectionMethod.FIRST_ANGLE,
    )
    assert "view_iso should be the top view" in question


@pytest.mark.parametrize("role", [View.DETAIL, View.SECTION])
@pytest.mark.parametrize("scale", [0.05, 0.2])
def test_a_different_scale_does_not_suppress_the_placement_warning(tmp_path, role, scale):
    interpretation = _with_pictorial(
        tmp_path, (1500, 1400, 2300, 2800), (1600, 1600, 2200, 2600)
    )
    interpretation.views[1].scale = 0.1
    interpretation.views[-1].role = role
    interpretation.views[-1].scale = scale
    (question,) = _questions(interpretation, tmp_path)
    assert "view_iso should be the right view" in question


@pytest.mark.parametrize("role", [View.FULL_PAGE, *ORTHOGRAPHIC_VIEWS])
def test_pages_and_registered_orthographic_roles_are_not_candidates(tmp_path, role):
    interpretation = _with_pictorial(
        tmp_path, (1500, 1400, 2300, 2800), (1600, 1600, 2200, 2600)
    )
    interpretation.views[-1].role = role
    if role in ORTHOGRAPHIC_VIEWS:
        interpretation.views[-1].u_axis, interpretation.views[-1].v_axis = UNTURNED[
            role
        ]
    assert not _questions(interpretation, tmp_path)
