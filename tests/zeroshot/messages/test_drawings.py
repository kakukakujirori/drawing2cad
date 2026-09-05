"""What the drawing contract accepts, refuses, and preserves through JSON."""

from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from zeroshot.pipeline.messages.contracts.drawings import (
    ORTHOGRAPHIC_VIEWS,
    Dimension,
    DrawingEvidence,
    DrawingSheet,
    DrawingSource,
    EdgeStyle,
    drawing_paths,
    edge_style_for_linetype,
    sheet_name,
)


def evidence(name="ev_edge", entity="line", source=(), **parameters):
    return {
        "name": name,
        "entity": entity,
        "edge_style": "visible",
        "source": list(source),
        "parameters": [
            {"name": key, "values": values}
            for key, values in (
                parameters or {"start": [0, 0], "end": [10, 10]}
            ).items()
        ],
    }


def sheet(name="sheet_page", role="unknown", **fields):
    return {
        "name": name,
        "role": role,
        "label": None,
        "derived_from": None,
        "file": "/work/inputs/drawing.png",
        "origin": None,
        "evidence": [],
        "dimensions": [],
        **fields,
    }


def dimension(**fields):
    return {
        "name": "dim_length",
        "kind": "linear",
        "text": "10",
        "nominal": 10,
        "quantity": 1,
        "note": None,
        **fields,
    }


@pytest.mark.parametrize(
    "contract", [DrawingSource, DrawingSheet, DrawingEvidence, Dimension]
)
def test_a_contract_carries_no_prose_beyond_its_field_descriptions(contract):
    schema = contract.model_json_schema()

    assert "description" not in schema
    assert set(schema["properties"]) == set(contract.model_fields)


@pytest.mark.parametrize(
    ("entity", "parameters"),
    [
        ("line", {"start": [1, 2], "end": [3, 4]}),
        ("circle", {"center": [5, 6], "radius": [2]}),
        (
            "arc",
            {"center": [5, 6], "radius": [2], "start": [7, 6], "end": [3, 6]},
        ),
        (
            "ellipse",
            {
                "center": [0, 0],
                "major_axis": [3, 4],
                "minor_radius": [2],
                "start": [3, 4],
                "end": [-1.6, 1.2],
            },
        ),
        (
            "spline",
            {
                "control_points": [0, 0, 1, 2, 2, 2, 3, 0],
                "degree": [3],
                "knots": [0, 0, 0, 0, 1, 1, 1, 1],
            },
        ),
        ("polyline", {"vertices": [0, 0, 1, 2, 2, 0]}),
    ],
)
def test_primitive_json_round_trips_without_losing_a_number(entity, parameters):
    entry = DrawingEvidence.model_validate(evidence(entity=entity, **parameters))

    assert DrawingEvidence.model_validate_json(entry.model_dump_json()) == entry


def test_source_round_trip_includes_crop_ancestry_and_dimension_references():
    value = {
        "sheets": [
            sheet(),
            sheet(
                "sheet_front",
                "front",
                derived_from="sheet_page",
                file="/work/crop.png",
                origin=[33, 40],
                evidence=[evidence(source=["dim_length"])],
                dimensions=[dimension()],
            ),
        ],
        "rationale": "The front view was read from the page.",
    }
    new = DrawingSource.model_validate(value)
    assert DrawingSource.model_validate_json(new.model_dump_json()) == new
    assert new.cited_names() == {"ev_edge"}


@pytest.mark.parametrize(
    "case",
    [
        "duplicate_parameter",
        "missing_parameter",
        "unknown_parameter",
        "wrong_arity",
        "empty_points",
        "zero_axis",
        "off_ellipse",
    ],
)
def test_a_parameter_set_that_does_not_describe_the_entity_is_refused(case):
    value = evidence()
    if case == "duplicate_parameter":
        value["parameters"].append(deepcopy(value["parameters"][0]))
    elif case == "missing_parameter":
        value["parameters"].pop()
    elif case == "unknown_parameter":
        value["parameters"].append({"name": "radius", "values": [1]})
    elif case == "wrong_arity":
        value["parameters"][0]["values"] = [1]
    elif case == "empty_points":
        value = evidence(entity="polyline", vertices=[])
    else:
        value = evidence(
            entity="ellipse",
            center=[0, 0],
            major_axis=[5, 0],
            minor_radius=[2],
            start=[5, 0],
            end=[0, 2],
        )
        field = "major_axis" if case == "zero_axis" else "start"
        next(p for p in value["parameters"] if p["name"] == field)["values"] = [0, 0]
    with pytest.raises(ValidationError):
        DrawingEvidence.model_validate(value)


@pytest.mark.parametrize(
    ("offset", "accepted"), [(0, True), (0.03, True), (0.08, False)]
)
def test_a_rotated_full_ellipse_holds_its_endpoint_tolerance(offset, accepted):
    endpoint = [3 * (1 + offset), 4 * (1 + offset)]
    value = evidence(
        entity="ellipse",
        center=[0, 0],
        major_axis=[3, 4],
        minor_radius=[2],
        start=endpoint,
        end=endpoint,
    )
    if accepted:
        DrawingEvidence.model_validate(value)
    else:
        with pytest.raises(ValidationError):
            DrawingEvidence.model_validate(value)


@pytest.mark.parametrize(
    "case",
    [
        "no_source",
        "bad_file",
        "absent_figure",
        "bad_name",
        "self_parent",
        "duplicate_evidence",
        "missing_origin",
        "bad_origin",
        "pictorial_origin",
    ],
)
def test_a_sheet_that_cannot_be_placed_or_read_is_refused(case):
    value = sheet()
    changes = {
        "no_source": {"file": None},
        "absent_figure": {"evidence": [evidence(source=["dim_length"])]},
        "bad_file": {"file": "/work/model.step"},
        "bad_name": {"name": "sheet_BAD"},
        "self_parent": {"derived_from": "sheet_page"},
        "duplicate_evidence": {"evidence": [evidence(), evidence()]},
        "missing_origin": {"role": "front", "evidence": [evidence()]},
        "bad_origin": {"role": "front", "evidence": [evidence()], "origin": [0]},
        "pictorial_origin": {"role": "perspective", "origin": [0, 0]},
    }
    value.update(changes[case])
    with pytest.raises(ValidationError):
        DrawingSheet.model_validate(value)


@pytest.mark.parametrize(
    "case",
    [
        "empty",
        "duplicate_sheet",
        "duplicate_dimension",
        "duplicate_role",
        "duplicate_evidence",
        "missing_ancestor",
        "cycle",
    ],
)
def test_a_drawing_whose_names_or_ancestry_break_is_refused(case):
    a, b = sheet("sheet_a"), sheet("sheet_b")
    if case == "duplicate_sheet":
        b["name"] = a["name"]
    elif case == "duplicate_dimension":
        a["dimensions"], b["dimensions"] = [dimension()], [dimension()]
    elif case == "duplicate_role":
        a["role"] = b["role"] = "front"
    elif case == "duplicate_evidence":
        a["evidence"], b["evidence"] = [evidence()], [evidence()]
    elif case == "missing_ancestor":
        b["derived_from"] = "sheet_missing"
    elif case == "cycle":
        a["derived_from"], b["derived_from"] = "sheet_b", "sheet_a"
    value = {"sheets": [] if case == "empty" else [a, b]}
    with pytest.raises(ValidationError):
        DrawingSource.model_validate(value)


def test_zero_quantity_is_still_rejected():
    with pytest.raises(ValidationError):
        Dimension.model_validate(dimension(quantity=0))


def test_a_spline_control_polygon_does_not_reject_a_separate_view():
    # This cubic reaches y=15; its control polygon reaches y=20.
    spline = evidence(
        "ev_spline",
        "spline",
        control_points=[0, 0, 0, 20, 10, 20, 10, 0],
        degree=[3],
        knots=[0, 0, 0, 0, 1, 1, 1, 1],
    )
    value = {
        "sheets": [
            sheet(),
            sheet(
                "sheet_front",
                "front",
                derived_from="sheet_page",
                file=None,
                origin=[0, 0],
                evidence=[spline],
            ),
            sheet(
                "sheet_top",
                "top",
                derived_from="sheet_page",
                file=None,
                origin=[0, 18],
                evidence=[evidence("ev_top", start=[2, 18], end=[8, 19])],
            ),
        ]
    }
    assert len(DrawingSource.model_validate(value).sheets) == 3


@pytest.mark.parametrize(
    ("linetype", "expected"),
    [
        ("Continuous", EdgeStyle.VISIBLE),
        ("HIDDEN", EdgeStyle.HIDDEN),
        ("CENTERX2", EdgeStyle.CENTERLINE),
        ("DOT2", EdgeStyle.HIDDEN),
        ("PHANTOM", EdgeStyle.PHANTOM),
        ("BYLAYER", EdgeStyle.OTHER),
        ("BYBLOCK", EdgeStyle.OTHER),
        ("unknown", EdgeStyle.OTHER),
    ],
)
def test_a_linetype_maps_to_one_edge_style(linetype, expected):
    assert edge_style_for_linetype(linetype) is expected


def test_staging_and_prompt_helpers_name_input_files_and_frames():
    value = {
        "sheets": [
            sheet(file=Path("/work/drawing.png")),
            sheet("sheet_perspective", "perspective", file="/work/perspective.png"),
        ]
    }

    both = DrawingSource.model_validate(value)
    assert drawing_paths(both) == [
        Path("/work/drawing.png"),
        Path("/work/perspective.png"),
    ]
    # No view is settled yet, so every frame a later stage might need is offered.
    assert both.frame_sentence().count(";") == len(ORTHOGRAPHIC_VIEWS) - 1

    front = DrawingSource.model_validate({"sheets": [sheet("sheet_front", "front")]})
    assert front.frame_sentence() == "Front is right=+x, up=+y"
    assert sheet_name("style-a") == "sheet_style_a"
