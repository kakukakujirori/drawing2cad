"""What a DXF becomes as drawing evidence, and what it becomes on the way back."""

import math

import ezdxf
import pytest

from zeroshot.pipeline.drawing import export_drawing, export_sheet, read_drawing
from zeroshot.pipeline.messages.contracts.drawings import (
    DrawnEntity,
    EdgeStyle,
    View,
)


def written(tmp_path, draw, name="sheet.dxf", setup=True):
    """A DXF holding whatever `draw` puts in its modelspace."""
    document = ezdxf.new("R2010", setup=setup)
    document.layers.add("hidden_layer", linetype="DASHED")
    draw(document.modelspace(), document)
    path = tmp_path / name
    document.saveas(path)
    return path


def values_of(entry):
    return {parameter.name.value: parameter.values for parameter in entry.parameters}


def only_entry(tmp_path, draw):
    reading = read_drawing(written(tmp_path, draw))
    assert reading.skipped == {}
    (entry,) = reading.sheet.evidence
    return entry


def test_a_page_is_read_as_one_unsplit_sheet(tmp_path):
    path = written(tmp_path, lambda space, _: space.add_line((0, 0), (10, 0)))

    reading = read_drawing(path)

    assert len(reading.drawing.sheets) == 1
    assert reading.sheet.role is View.UNKNOWN
    assert reading.sheet.origin is None
    assert reading.sheet.derived_from is None
    assert reading.sheet.file == str(path)
    assert reading.sheet.dimensions == []


def test_a_line_keeps_both_of_its_ends(tmp_path):
    entry = only_entry(tmp_path, lambda space, _: space.add_line((1, 2), (3, 4)))

    assert entry.entity is DrawnEntity.LINE
    assert values_of(entry) == {"start": [1.0, 2.0], "end": [3.0, 4.0]}


def test_a_circle_keeps_its_centre_and_radius(tmp_path):
    entry = only_entry(tmp_path, lambda space, _: space.add_circle((5, 6), 2.5))

    assert entry.entity is DrawnEntity.CIRCLE
    assert values_of(entry) == {"center": [5.0, 6.0], "radius": [2.5]}


@pytest.mark.parametrize(
    ("start_angle", "end_angle", "start", "end"),
    [
        (0, 90, [10.0, 0.0], [0.0, 10.0]),
        (90, 180, [0.0, 10.0], [-10.0, 0.0]),
        # An arc that runs through zero keeps running through it as two points.
        (350, 10, [9.8481, -1.7365], [9.8481, 1.7365]),
    ],
)
def test_an_arc_states_its_ends_as_points_on_itself(
    tmp_path, start_angle, end_angle, start, end
):
    entry = only_entry(
        tmp_path,
        lambda space, _: space.add_arc((0, 0), 10, start_angle, end_angle),
    )

    assert entry.entity is DrawnEntity.ARC
    read = values_of(entry)
    assert read["center"] == [0.0, 0.0]
    assert read["radius"] == [10.0]
    assert read["start"] == pytest.approx(start, abs=1e-4)
    assert read["end"] == pytest.approx(end, abs=1e-4)


def test_a_part_ellipse_keeps_the_part_it_draws(tmp_path):
    entry = only_entry(
        tmp_path,
        lambda space, _: space.add_ellipse(
            (0, 0), major_axis=(10, 0), ratio=0.5, start_param=0, end_param=math.pi / 2
        ),
    )

    assert entry.entity is DrawnEntity.ELLIPSE
    read = values_of(entry)
    assert read["center"] == [0.0, 0.0]
    assert read["major_axis"] == [10.0, 0.0]
    assert read["minor_radius"] == pytest.approx([5.0])
    assert read["start"] == pytest.approx([10.0, 0.0], abs=1e-9)
    assert read["end"] == pytest.approx([0.0, 5.0], abs=1e-9)


def test_a_whole_ellipse_gives_the_same_point_twice(tmp_path):
    entry = only_entry(
        tmp_path,
        lambda space, _: space.add_ellipse((0, 0), major_axis=(4, 0), ratio=0.5),
    )

    read = values_of(entry)
    assert read["start"] == pytest.approx(read["end"], abs=1e-9)


def test_a_spline_keeps_its_control_polygon_and_knots(tmp_path):
    def draw(space, _):
        spline = space.add_spline(degree=3)
        spline.control_points = [(0, 0), (1, 2), (2, 2), (3, 0)]
        spline.knots = [0, 0, 0, 0, 1, 1, 1, 1]

    entry = only_entry(tmp_path, draw)

    assert entry.entity is DrawnEntity.SPLINE
    assert values_of(entry) == {
        "control_points": [0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 3.0, 0.0],
        "degree": [3.0],
        "knots": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    }


def test_a_polyline_keeps_its_vertices(tmp_path):
    entry = only_entry(
        tmp_path,
        lambda space, _: space.add_lwpolyline([(0, 0), (1, 2), (2, 0)]),
    )

    assert entry.entity is DrawnEntity.POLYLINE
    assert values_of(entry) == {"vertices": [0.0, 0.0, 1.0, 2.0, 2.0, 0.0]}


@pytest.mark.parametrize(
    ("linetype", "expected"),
    [
        ("CONTINUOUS", EdgeStyle.VISIBLE),
        ("DASHED", EdgeStyle.HIDDEN),
        ("CENTER", EdgeStyle.CENTERLINE),
        ("PHANTOM", EdgeStyle.PHANTOM),
    ],
)
def test_a_linetype_written_on_the_entity_becomes_its_edge_style(
    tmp_path, linetype, expected
):
    entry = only_entry(
        tmp_path,
        lambda space, _: space.add_line(
            (0, 0), (1, 1), dxfattribs={"linetype": linetype}
        ),
    )

    assert entry.edge_style is expected


def test_an_entity_deferring_to_its_layer_takes_the_layer_s_linetype(tmp_path):
    entry = only_entry(
        tmp_path,
        lambda space, _: space.add_line(
            (0, 0), (1, 1), dxfattribs={"layer": "hidden_layer", "linetype": "BYLAYER"}
        ),
    )

    assert entry.edge_style is EdgeStyle.HIDDEN


def test_an_entity_is_named_for_its_handle_and_keeps_that_name(tmp_path):
    path = written(tmp_path, lambda space, _: space.add_line((0, 0), (1, 1)))

    first = read_drawing(path).sheet.evidence
    again = read_drawing(path).sheet.evidence

    assert [entry.name for entry in first] == [entry.name for entry in again]
    assert first[0].name.startswith("ev_")


def test_a_block_reference_is_reported_rather_than_transcribed(tmp_path):
    def draw(space, document):
        block = document.blocks.new("CENTREMARK")
        block.add_line((-1, 0), (1, 0))
        space.add_blockref("CENTREMARK", (5, 5))
        space.add_line((0, 0), (1, 1))

    reading = read_drawing(written(tmp_path, draw))

    assert reading.skipped == {"INSERT CENTREMARK": 1}
    assert len(reading.sheet.evidence) == 1


def test_an_entity_the_contract_has_no_shape_for_is_reported(tmp_path):
    def draw(space, _):
        space.add_text("A", dxfattribs={"insert": (0, 0)})
        space.add_line((0, 0), (1, 1))

    reading = read_drawing(written(tmp_path, draw))

    assert reading.skipped == {"TEXT": 1}
    assert len(reading.sheet.evidence) == 1


def test_an_entity_off_the_drawing_plane_is_reported_rather_than_flattened(tmp_path):
    def draw(space, _):
        space.add_line((0, 0, 5), (1, 1, 5))
        space.add_line((0, 0), (1, 1))

    reading = read_drawing(written(tmp_path, draw))

    assert reading.skipped == {"LINE off the drawing plane": 1}
    assert len(reading.sheet.evidence) == 1


def test_an_entity_drawn_against_a_turned_axis_is_reported(tmp_path):
    def draw(space, _):
        space.add_circle((0, 0), 1, dxfattribs={"extrusion": (0, 0, -1)})
        space.add_line((0, 0), (1, 1))

    reading = read_drawing(written(tmp_path, draw))

    assert reading.skipped == {"CIRCLE off the drawing plane": 1}
    assert len(reading.sheet.evidence) == 1


def _every_entity(space, _):
    space.add_line((1, 2), (3, 4), dxfattribs={"linetype": "DASHED"})
    space.add_circle((5, 6), 2.5)
    space.add_arc((0, 0), 10, 350, 10)
    space.add_ellipse(
        (2, 3), major_axis=(6, 2), ratio=0.4, start_param=0.3, end_param=2.1
    )
    space.add_lwpolyline([(0, 0), (1, 2), (2, 0)])
    spline = space.add_spline(degree=3)
    spline.control_points = [(0, 0), (1, 2), (2, 2), (3, 0)]
    spline.knots = [0, 0, 0, 0, 1, 1, 1, 1]


def test_writing_a_sheet_out_and_reading_it_back_changes_no_number(tmp_path):
    first = read_drawing(written(tmp_path, _every_entity)).drawing

    again = read_drawing(export_sheet(first.sheets[0], tmp_path / "out.dxf")).drawing

    assert [entry.entity for entry in again.evidence()] == [
        entry.entity for entry in first.evidence()
    ]
    assert [entry.edge_style for entry in again.evidence()] == [
        entry.edge_style for entry in first.evidence()
    ]
    for before, after in zip(first.evidence(), again.evidence()):
        written_out, read_back = values_of(before), values_of(after)
        assert read_back.keys() == written_out.keys()
        for name, numbers in written_out.items():
            assert read_back[name] == pytest.approx(numbers, abs=1e-9)


def test_a_written_drawing_puts_each_sheet_on_a_layer_of_its_own(tmp_path):
    reading = read_drawing(written(tmp_path, _every_entity))
    front = reading.sheet.model_copy(
        update={
            "name": "sheet_front",
            "role": View.FRONT,
            "derived_from": None,
            "origin": [0.0, 0.0],
        }
    )

    path = export_drawing(
        reading.drawing.model_copy(update={"sheets": [front]}), tmp_path / "d.dxf"
    )

    document = ezdxf.readfile(path)
    assert {entity.dxf.layer for entity in document.modelspace()} == {"sheet_front"}


def test_a_sheet_that_drew_nothing_writes_no_layer(tmp_path):
    reading = read_drawing(
        written(tmp_path, lambda space, _: space.add_line((0, 0), (1, 1)))
    )
    empty = reading.sheet.model_copy(update={"name": "sheet_blank", "evidence": []})

    path = export_drawing(
        reading.drawing.model_copy(update={"sheets": [reading.sheet, empty]}),
        tmp_path / "d.dxf",
    )

    assert "sheet_blank" not in ezdxf.readfile(path).layers
