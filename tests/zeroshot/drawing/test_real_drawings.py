"""The drawing reader against the real technical drawings, all twenty of them.

The counts here are what `drawing_contract_1a.md` measured. A change to them
is a change to what the drawing stage is given, not a test to be adjusted.
"""

from collections import Counter
from pathlib import Path

import pytest

from zeroshot.pipeline.drawing import export_drawing, place_views, read_drawing

TECHDRAW = Path("data/test_vlm/techdraw/dxf")

# Every entity the twenty drawings hold, and the centre marks among them that
# the contract has no shape for. Each sits exactly on a circle's centre, so
# the circle already states what it marks.
TRANSCRIBED = 3283
CENTRE_MARKS = 49

pytestmark = pytest.mark.skipif(
    not TECHDRAW.is_dir(), reason=f"{TECHDRAW} is not present"
)


def drawings() -> list[Path]:
    return sorted(TECHDRAW.glob("*.dxf"))


def signature(drawing) -> list[tuple]:
    """Every entity as its kind, its linework and its numbers."""
    return [
        (
            entry.entity.value,
            entry.edge_style.value,
            sorted(
                (parameter.name.value, tuple(round(v, 6) for v in parameter.values))
                for parameter in entry.parameters
            ),
        )
        for entry in drawing.evidence()
    ]


def test_every_drawing_is_transcribed_but_for_its_centre_marks():
    transcribed = 0
    skipped: Counter[str] = Counter()

    for path in drawings():
        reading = read_drawing(path)
        transcribed += len(reading.sheet.evidence)
        skipped.update(reading.skipped)

    assert transcribed == TRANSCRIBED
    assert sum(skipped.values()) == CENTRE_MARKS
    assert all(label.startswith("INSERT SW_CENTERMARK") for label in skipped)


def test_every_drawing_separates_into_three_aligned_views():
    for path in drawings():
        placement = place_views(read_drawing(path).drawing)

        roles = [sheet.role.value for sheet in placement.drawing.sheets[1:]]
        assert sorted(roles) == ["front", "right", "top"], path.stem
        # The views share the extents that carry x and y, so the corner each
        # of them places is the same physical point.
        assert placement.alignment_error == pytest.approx(0.0, abs=1e-9), path.stem
        assert all(sheet.evidence for sheet in placement.drawing.sheets[1:]), path.stem


def test_a_separated_drawing_survives_being_written_out_and_read_back(tmp_path):
    for path in drawings():
        placed = place_views(read_drawing(path).drawing).drawing

        written = export_drawing(placed, tmp_path / f"{path.stem}.dxf")

        assert signature(read_drawing(written).drawing) == signature(placed), path.stem
