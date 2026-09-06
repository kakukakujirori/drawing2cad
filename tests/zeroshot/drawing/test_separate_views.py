"""How an unsplit page separates into three views, and when it refuses to."""

import pytest

from zeroshot.pipeline.drawing.separate_views import ViewSplitError, extent, place_views
from zeroshot.pipeline.messages.contracts.drawings import (
    DrawingEvidence,
    DrawingSheet,
    DrawingSource,
    DrawnEntity,
    View,
)
from zeroshot.pipeline.messages.contracts.parameters import Parameter

# Three boxes in the third-angle L: top sits above front and shares its
# horizontal extent, right sits beside front and shares its vertical one.
FRONT = (0.0, 0.0, 40.0, 30.0)
TOP = (0.0, 50.0, 40.0, 80.0)
RIGHT = (60.0, 0.0, 90.0, 30.0)


def entry(name, entity=DrawnEntity.LINE, **parameters):
    return DrawingEvidence(
        name=name,
        entity=entity,
        edge_style="visible",
        source=[],
        parameters=[
            Parameter(name=key, values=values) for key, values in parameters.items()
        ],
    )


def outline(prefix, box):
    """The four edges of one box, as four lines."""
    x0, y0, x1, y1 = box
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return [
        entry(
            f"ev_{prefix}_{index}",
            start=list(corner),
            end=list(corners[(index + 1) % 4]),
        )
        for index, corner in enumerate(corners)
    ]


def page(entries):
    return DrawingSource(
        sheets=[
            DrawingSheet(
                name="sheet_page",
                role=View.FULL_PAGE,
                label=None,
                crop_of=None,
                scale=1.0,
                file="page.dxf",
                evidence=entries,
                dimensions=[],
            )
        ]
    )


def three_views(extra=()):
    return page(
        [
            *outline("front", FRONT),
            *outline("top", TOP),
            *outline("right", RIGHT),
            *extra,
        ]
    )


def test_a_three_view_page_separates_into_the_three_views(tmp_path):
    placement = place_views(three_views(), tmp_path)

    by_role = {sheet.role: sheet for sheet in placement.drawing.sheets}
    assert set(by_role) == {View.FULL_PAGE, View.FRONT, View.TOP, View.RIGHT}
    assert placement.alignment_error == pytest.approx(0.0)
    assert by_role[View.FULL_PAGE].evidence == []
    for role in (View.FRONT, View.TOP, View.RIGHT):
        assert len(by_role[role].evidence) == 4
        assert by_role[role].crop_of is not None
        assert by_role[role].crop_of.sheet == "sheet_page"
        assert by_role[role].file == str(tmp_path / f"sheet_{role.value}.dxf")


def test_every_view_says_which_region_of_the_page_it_covers(tmp_path):
    placement = place_views(three_views(), tmp_path)

    cuts = {sheet.role: sheet.crop_of for sheet in placement.drawing.sheets[1:]}
    for role, box in ((View.FRONT, FRONT), (View.TOP, TOP), (View.RIGHT, RIGHT)):
        assert cuts[role] is not None
        assert cuts[role].box == pytest.approx(list(box))


def test_a_view_is_read_from_its_own_corner_rather_than_the_page_s(tmp_path):
    """A crop is measured in its own frame, so the page offset is taken off."""
    placement = place_views(three_views(), tmp_path)

    by_role = {sheet.role: sheet for sheet in placement.drawing.sheets}
    starts = [
        parameter.values
        for entry in by_role[View.TOP].evidence
        for parameter in entry.parameters
        if parameter.name.value == "start"
    ]
    assert min(value[0] for value in starts) == pytest.approx(0.0)
    assert min(value[1] for value in starts) == pytest.approx(0.0)


def test_the_views_that_share_an_extent_are_checked_against_each_other(tmp_path):
    shifted = (2.0, 50.0, 42.0, 80.0)
    misaligned = page(
        [*outline("front", FRONT), *outline("top", shifted), *outline("right", RIGHT)]
    )

    with pytest.raises(ViewSplitError, match="shared extents"):
        place_views(misaligned, tmp_path)


def test_a_page_whose_views_are_not_separated_is_refused(tmp_path):
    touching = page(
        [*outline("front", FRONT), *outline("top", (0.0, 30.0, 40.0, 60.0))]
    )

    with pytest.raises(ViewSplitError, match="gap of at least"):
        place_views(touching, tmp_path)


def test_a_page_with_a_fourth_populated_quadrant_is_refused(tmp_path):
    crowded = three_views(outline("fourth", (60.0, 50.0, 90.0, 80.0)))

    with pytest.raises(ViewSplitError):
        place_views(crowded, tmp_path)


def test_an_entity_reaching_across_two_views_closes_the_gap_between_them(tmp_path):
    """Nothing is filed under a guess: the page stops being a three-view page."""
    spanning = three_views([entry("ev_spanning", start=[10.0, 10.0], end=[70.0, 10.0])])

    with pytest.raises(ViewSplitError, match="gap of at least"):
        place_views(spanning, tmp_path)


def test_the_page_keeps_its_file_and_gives_up_its_linework(tmp_path):
    placement = place_views(three_views(), tmp_path)

    page_sheet = placement.drawing.sheets[0]
    assert page_sheet.name == "sheet_page"
    assert page_sheet.file == "page.dxf"
    assert page_sheet.evidence == []


def test_a_page_this_drawing_does_not_hold_is_refused(tmp_path):
    with pytest.raises(ViewSplitError, match="not a sheet"):
        place_views(three_views(), tmp_path, page="sheet_missing")


def test_a_circle_reaches_a_radius_past_its_centre():
    assert extent(
        entry("ev_c", DrawnEntity.CIRCLE, center=[50.0, 50.0], radius=[20.0])
    ) == (30.0, 30.0, 70.0, 70.0)


@pytest.mark.parametrize(
    ("start", "end", "box"),
    [
        # A quarter turn reaches only as far as its own two ends.
        ([10.0, 0.0], [0.0, 10.0], (0.0, 0.0, 10.0, 10.0)),
        # One that sweeps past due west reaches the full radius that way.
        ([0.0, 10.0], [0.0, -10.0], (-10.0, -10.0, 0.0, 10.0)),
    ],
)
def test_an_arc_reaches_its_ends_and_any_quarter_it_sweeps_past(start, end, box):
    reach = extent(
        entry(
            "ev_a",
            DrawnEntity.ARC,
            center=[0.0, 0.0],
            radius=[10.0],
            start=start,
            end=end,
        )
    )

    assert reach == pytest.approx(box, abs=1e-9)


def test_a_view_of_concentric_circles_still_occupies_its_area(tmp_path):
    """The case a centre-only reach collapsed to a point."""
    rings = [
        entry(
            f"ev_ring_{index}", DrawnEntity.CIRCLE, center=[20.0, 15.0], radius=[radius]
        )
        for index, radius in enumerate((5.0, 10.0, 15.0))
    ]

    placement = place_views(
        page(
            [
                *outline("front", FRONT),
                *outline("top", TOP),
                *rings,
                *outline("right", RIGHT),
            ]
        ),
        tmp_path,
    )

    front = next(s for s in placement.drawing.sheets if s.role is View.FRONT)
    assert len(front.evidence) == 7
