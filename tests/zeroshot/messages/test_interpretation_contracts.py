"""Local validation for file-backed interpretation sheets and evidence."""

import json
from copy import deepcopy

import pytest
from pydantic import ValidationError

from zeroshot.pipeline.stages.interpretation.contracts import (
    ORTHOGRAPHIC_VIEWS,
    VIEW_FRAME,
    DimensionSummary,
    DrawingInterpretation,
    Region,
    View,
)


def pin_interpretation() -> dict:
    return {
        "datum": "mm; X right, Y rearward, Z up; origin at the lower front left.",
        "views": [
            {
                "name": "view_front",
                "role": "front",
                "file": "front.png",
                "region": {"view": "view_front", "box_px": [370, 845, 1165, 1330]},
                "dimensions": [
                    {
                        "name": "dim_pin_diameter",
                        "kind": "diameter",
                        "text": "Ø4.2",
                        "nominal_value": 4.2,
                        "quantity": 1,
                        "note": None,
                        "region": {
                            "view": "view_front",
                            "box_px": [450, 750, 630, 1015],
                        },
                    }
                ],
            }
        ],
        "features": [
            {
                "name": "sem_pin_upper_left",
                "description": "Solid cylinder pointing toward the open front from the rear inner wall. base_center is the attachment-face center; axis is the protrusion direction.",
                "parameters": {
                    "base_center": [29, None, 43.9],
                    "axis": [0, -1, 0],
                    "diameter": 4.2,
                    "length": None,
                },
                "evidence": [{"view": "view_front", "box_px": [560, 944, 642, 1011]}],
                "dimension_refs": ["dim_pin_diameter"],
            }
        ],
    }


def test_pin_roundtrip_preserves_unknowns_zero_and_optional_metadata() -> None:
    interpretation = DrawingInterpretation.model_validate(pin_interpretation())
    assert interpretation.features[0].parameters == {
        "base_center": [29, None, 43.9],
        "axis": [0, -1, 0],
        "diameter": 4.2,
        "length": None,
    }
    assert interpretation.questions == []
    sheet = interpretation.views[0]
    assert sheet.image_size is None and sheet.scale is None
    assert sheet.region.view == "view_front"
    assert sheet.dimensions[0].measured_length is None
    assert (
        DrawingInterpretation.model_validate_json(interpretation.model_dump_json())
        == interpretation
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_numbers_are_rejected_in_parameters_regions_and_measurements(
    value: float,
) -> None:
    for parameters in [{"radius": value}, {"origin": [0, None, value]}]:
        data = pin_interpretation()
        data["features"][0]["parameters"] = parameters
        with pytest.raises(ValidationError, match="finite number"):
            DrawingInterpretation.model_validate(data)
    for field in ["box_px", "box_uv"]:
        with pytest.raises(ValidationError, match="finite number"):
            Region(view="view_front", **{field: (0, 0, value, 1)})
    data = pin_interpretation()
    data["views"][0]["dimensions"][0]["measured_length"] = value
    with pytest.raises(ValidationError):
        DrawingInterpretation.model_validate(data)


@pytest.mark.parametrize(
    "box", [(2, 0, 1, 3), (0, 2, 3, 1), (1, 0, 1, 3), (0, 1, 3, 1)]
)
@pytest.mark.parametrize("field", ["box_px", "box_uv"])
def test_region_requires_positive_width_and_height(box: tuple, field: str) -> None:
    with pytest.raises(ValidationError):
        Region(view="view_front", **{field: box})


def test_region_requires_a_sheet_and_nonnegative_box_and_is_immutable() -> None:
    with pytest.raises(ValidationError):
        Region(box_px=(0, 0, 10, 20))
    with pytest.raises(ValidationError, match="at least one"):
        Region(view="view_front")
    for field in ["box_px", "box_uv"]:
        with pytest.raises(ValidationError):
            Region(view="view_front", **{field: (-20, -10, -5, -1)})
    region = Region(view="view_front", box_px=(0, 0, 10, 20))
    with pytest.raises(ValidationError, match="frozen"):
        region.box_uv = (0, 0, 1, 2)


@pytest.mark.parametrize("target", ["interpretation", "sheet", "region", "dimension"])
def test_removed_source_crop_and_legacy_fields_are_rejected(target: str) -> None:
    data = pin_interpretation()
    if target == "interpretation":
        data["sheets"] = []
    elif target == "sheet":
        data["views"][0]["crop_of"] = data["features"][0]["evidence"][0]
    elif target == "region":
        data["features"][0]["evidence"][0]["source"] = "source_png"
    else:
        data["views"][0]["dimensions"][0]["measured_px"] = 42
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        DrawingInterpretation.model_validate(data)


@pytest.mark.parametrize(
    ("collection", "name"),
    [
        ("views", "sheet_front"),
        ("views", "view_"),
        ("features", "pin"),
        ("features", "sem_Upper"),
    ],
)
def test_names_are_stable_prefixed_addresses(collection: str, name: str) -> None:
    data = pin_interpretation()
    data[collection][0]["name"] = name
    with pytest.raises(ValidationError, match="usable"):
        DrawingInterpretation.model_validate(data)


@pytest.mark.parametrize("collection", ["views", "features"])
def test_named_entries_are_unique(collection: str) -> None:
    data = pin_interpretation()
    data[collection].append(deepcopy(data[collection][0]))
    with pytest.raises(ValidationError, match="duplicate names"):
        DrawingInterpretation.model_validate(data)


def test_feature_requires_evidence_and_unknown_view_error_names_its_location() -> None:
    data = pin_interpretation()
    data["features"][0]["evidence"] = []
    with pytest.raises(ValidationError):
        DrawingInterpretation.model_validate(data)
    data = pin_interpretation()
    data["features"][0]["evidence"][0]["view"] = "view_missing"
    with pytest.raises(ValidationError) as error:
        DrawingInterpretation.model_validate(data)
    message = str(error.value)
    assert all(
        text in message
        for text in ["sem_pin_upper_left", "evidence[0]", "view_missing"]
    )


def test_dimension_names_and_referenced_views_are_checked() -> None:
    data = pin_interpretation()
    second = deepcopy(data["views"][0])
    second.update(name="view_top", file="top.png")
    second["dimensions"][0]["region"]["view"] = "view_top"
    data["views"].append(second)
    with pytest.raises(ValidationError, match="duplicate names in dimensions"):
        DrawingInterpretation.model_validate(data)
    for refs in [["dim_missing"], ["dim_pin_diameter", "dim_pin_diameter"]]:
        data = pin_interpretation()
        data["features"][0]["dimension_refs"] = refs
        with pytest.raises(ValidationError):
            DrawingInterpretation.model_validate(data)
    data = pin_interpretation()
    data["views"][0]["dimensions"][0]["region"]["view"] = "view_top"
    with pytest.raises(ValidationError):
        DrawingInterpretation.model_validate(data)


def test_angular_dimensions_do_not_accept_measured_lengths() -> None:
    data = pin_interpretation()
    data["views"][0]["dimensions"][0].update(kind="angular", measured_length=30)
    with pytest.raises(ValidationError, match="angular"):
        DrawingInterpretation.model_validate(data)


def test_schema_describes_fields_and_preserves_file_backed_views() -> None:
    schema = DrawingInterpretation.model_json_schema()
    assert set(schema["required"]) == {"datum", "views", "features"}
    view_schema = schema["$defs"]["DrawingView"]
    assert set(view_schema["properties"]) == {
        "name",
        "role",
        "file",
        "region",
        "dimensions",
        "image_size",
        "scale",
    }
    assert {"file", "region"} <= set(view_schema["required"])
    assert set(schema["$defs"]["Region"]["properties"]) == {"view", "box_px", "box_uv"}
    assert "view" in schema["$defs"]["Region"]["required"]
    for held in [schema, *schema["$defs"].values()]:
        if held.get("type") != "object":
            continue
        assert held["additionalProperties"] is False
        for name, field in held["properties"].items():
            assert field.get("description"), (held["title"], name)


def test_every_orthographic_view_has_a_frame() -> None:
    """Sections, details, pictorials, and an unsplit page establish no global axes."""
    assert set(VIEW_FRAME) == set(ORTHOGRAPHIC_VIEWS)
    assert set(VIEW_FRAME) < set(View)
    axes = {axis for frame in VIEW_FRAME.values() for axis in frame}
    assert axes <= {"+x", "-x", "+y", "-y", "+z", "-z"}
    for view, frame in VIEW_FRAME.items():
        assert len({axis.lstrip("+-") for axis in frame}) == 3, view


def test_region_matches_bounds() -> None:
    raster = Region(view="view_front", box_px=(0, 0, 100, 200))
    assert raster.matches_bounds(raster)
    assert raster.matches_bounds(Region(view="view_front", box_px=(0, 0, 100, 200)))
    # Different view name
    assert not raster.matches_bounds(Region(view="view_other", box_px=(0, 0, 100, 200)))
    # Different pixel bounds
    assert not raster.matches_bounds(Region(view="view_front", box_px=(0, 0, 100, 150)))

    dxf = Region(view="view_front", box_uv=(0.0, 0.0, 50.0, 80.0))
    assert dxf.matches_bounds(dxf)
    # Within tolerance
    assert dxf.matches_bounds(
        Region(view="view_front", box_uv=(0.0, 0.0, 50.0 + 1e-8, 80.0 - 1e-8))
    )
    # Outside tolerance
    assert not dxf.matches_bounds(
        Region(view="view_front", box_uv=(0.0, 0.0, 50.0 + 1e-5, 80.0))
    )

    # Disagreeing kind (raster vs DXF)
    assert not raster.matches_bounds(dxf)
    assert not dxf.matches_bounds(raster)


def test_interpretation_all_dimensions_and_inventory() -> None:
    interpretation = DrawingInterpretation.model_validate(pin_interpretation())

    assert len(interpretation.all_dimensions) == 1
    dimension = interpretation.all_dimensions[0]
    assert dimension.name == "dim_pin_diameter"
    assert dimension.kind == "diameter"
    assert dimension.text == "Ø4.2"
    assert dimension.nominal_value == 4.2
    assert dimension.quantity == 1

    inventory = interpretation.dimension_inventory()
    assert len(inventory) == 1
    summary = inventory[0]
    assert isinstance(summary, DimensionSummary)
    assert summary.name == "dim_pin_diameter"
    assert summary.text == "Ø4.2"
    assert summary.nominal_value == 4.2
    assert summary.kind == "diameter"
    assert summary.quantity == 1

    rendered = interpretation.render_dimension_inventory()
    assert json.loads(rendered) == [
        {
            "name": "dim_pin_diameter",
            "text": "Ø4.2",
            "nominal_value": 4.2,
            "kind": "diameter",
            "quantity": 1,
        }
    ]
