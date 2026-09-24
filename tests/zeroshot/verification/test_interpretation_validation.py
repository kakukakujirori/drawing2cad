"""Validate real sheet files, calibration and evidence without model calls."""

from copy import deepcopy
from pathlib import Path

import ezdxf
import pytest
from PIL import Image

from tests.zeroshot.contracts import UNTURNED
from tests.zeroshot.messages.test_interpretation_contracts import pin_interpretation
from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.stages._base.validate import LocatedError
from zeroshot.pipeline.stages.interpretation.contracts import (
    DrawingInterpretation,
    View,
)
from zeroshot.pipeline.stages.interpretation.validate import validate_interpretation

MEASUREMENTS = [(4.2, 42), (10, 100), (20, 200)]


def raster_case(workdir: Path, measurements=MEASUREMENTS):
    Image.new("RGB", (1200, 1400), "white").save(workdir / "front.png")
    data = pin_interpretation()
    template = data["views"][0]["dimensions"][0]
    dimensions = []
    for index, (nominal, measured) in enumerate(measurements):
        dimension = deepcopy(template)
        dimension.update(
            name="dim_pin_diameter" if index == 0 else f"dim_length_{index}",
            kind="diameter" if index == 0 else "linear",
            nominal_value=nominal,
            measured_length=measured,
            text=str(nominal),
        )
        dimensions.append(dimension)
    data["views"][0]["dimensions"] = dimensions
    if not dimensions:
        data["features"][0]["dimension_refs"] = []
    return DrawingInterpretation.model_validate(data)


def test_raster_consensus_enriches_a_copy_and_revalidates_idempotently(
    tmp_path: Path,
) -> None:
    original = raster_case(tmp_path, [*MEASUREMENTS, (30, 900)])
    before = original.model_dump()
    contents = (tmp_path / "front.png").read_bytes()
    accepted, diagnostics = validate_interpretation(
        original, workdir=SandboxWorkdir(tmp_path)
    )
    sheet = accepted.views[0]
    assert accepted is not original and original.model_dump() == before
    assert (tmp_path / "front.png").read_bytes() == contents
    assert sheet.image_size == (1200, 1400)
    assert sheet.scale == pytest.approx(0.1)
    assert sheet.region.box_uv == pytest.approx((37, 7, 116.5, 55.5))
    assert diagnostics["view_front"]["status"] == "ok"
    assert diagnostics["view_front"]["inliers"] == [
        "dim_pin_diameter",
        "dim_length_1",
        "dim_length_2",
    ]
    assert diagnostics["view_front"]["outliers"] == ["dim_length_3"]
    assert sheet.dimensions[0].region.box_uv == pytest.approx((45, 38.5, 63, 65))
    assert accepted.features[0].evidence[0].box_uv == pytest.approx(
        (56, 38.9, 64.2, 45.6)
    )
    revalidated, _ = validate_interpretation(accepted, workdir=SandboxWorkdir(tmp_path))
    assert revalidated == accepted


@pytest.mark.parametrize("file", ["front.png", "/work/front.png", "./front.png"])
def test_different_roles_cannot_share_a_png(tmp_path: Path, file: str) -> None:
    data = raster_case(tmp_path).model_dump()
    top = deepcopy(data["views"][0])
    top.update(name="view_top", role="top", file=file, v_axis="+y")
    top["region"].update(view="view_top", box_px=(50, 50, 350, 250))
    for index, dimension in enumerate(top["dimensions"]):
        dimension["name"] = f"dim_top_{index}"
        dimension["measured_length"] /= 2
        dimension["region"]["view"] = "view_top"
    data["views"].append(top)
    data["features"][0]["evidence"].append(
        {"view": "view_top", "box_px": (100, 100, 300, 200)}
    )
    with pytest.raises(
        LocatedError, match="different roles must use different files"
    ) as error:
        validate_interpretation(
            DrawingInterpretation.model_validate(data), workdir=SandboxWorkdir(tmp_path)
        )
    assert error.value.found[0][0] == ("views", 1, "file")
    assert "view_top (top)" in str(error.value)
    assert "view_front (front)" in str(error.value)


def test_repeated_role_may_still_reference_the_same_file(tmp_path: Path) -> None:
    submitted = raster_case(tmp_path)
    submitted.views.append(
        submitted.views[0].model_copy(update={"name": "view_detail", "dimensions": []})
    )
    accepted, _ = validate_interpretation(submitted, workdir=SandboxWorkdir(tmp_path))
    assert len(accepted.views) == 2


def test_full_page_and_separate_files_use_referenced_regions_and_own_measurements(
    tmp_path: Path,
) -> None:
    data = raster_case(tmp_path).model_dump()
    Image.new("RGB", (2400, 3000), "white").save(tmp_path / "original.png")
    Image.new("RGB", (600, 800), "white").save(tmp_path / "top.png")
    contents = (tmp_path / "original.png").read_bytes()
    front = data["views"][0]
    front["region"].update(view="view_full_page", box_px=(100, 200, 1300, 1600))
    top = deepcopy(front)
    top.update(name="view_top", role="top", file="/work/top.png", v_axis="+y")
    top["region"]["box_px"] = (1500, 100, 2100, 900)
    for index, dimension in enumerate(top["dimensions"]):
        dimension["name"] = f"dim_top_{index}"
        dimension["measured_length"] /= 2
        # The callout is on the original page, but length was measured on top.png.
        dimension["region"].update(
            view="view_full_page", box_px=(1800, 700, 2200, 1000)
        )
    page_dimensions = deepcopy(front["dimensions"])
    for index, dimension in enumerate(page_dimensions):
        dimension["name"] = f"dim_page_{index}"
        dimension["measured_length"] *= 2
        dimension["region"]["view"] = "view_full_page"
    full_page = {
        "name": "view_full_page",
        "role": "full_page",
        "file": "original.png",
        "region": {"view": "view_full_page", "box_px": (0, 0, 2400, 3000)},
        "dimensions": page_dimensions,
    }
    data["views"] = [full_page, front, top]
    data["features"][0]["evidence"].extend(
        [
            {"view": "view_top", "box_px": (100, 100, 300, 200)},
            {"view": "view_full_page", "box_px": (100, 200, 200, 300)},
        ]
    )
    submitted = DrawingInterpretation.model_validate(data)
    before = submitted.model_dump()
    accepted, diagnostics = validate_interpretation(
        submitted, workdir=SandboxWorkdir(tmp_path)
    )
    whole, front, top = accepted.views
    assert [view.image_size for view in accepted.views] == [
        (2400, 3000),
        (1200, 1400),
        (600, 800),
    ]
    assert [view.scale for view in accepted.views] == pytest.approx([0.05, 0.1, 0.2])
    assert diagnostics["view_full_page"]["status"] == "ok"
    assert whole.region.box_uv == pytest.approx((0, 0, 120, 150))
    assert front.region.box_uv == pytest.approx((5, 70, 65, 140))
    assert top.region.box_uv == pytest.approx((75, 105, 105, 145))
    assert top.dimensions[0].region.box_uv == pytest.approx((90, 100, 110, 115))
    evidence = accepted.features[0].evidence
    assert evidence[0].box_uv == pytest.approx((56, 38.9, 64.2, 45.6))
    assert evidence[1].box_uv == pytest.approx((20, 120, 60, 140))
    assert evidence[2].box_uv == pytest.approx((5, 135, 10, 140))
    assert (
        submitted.model_dump() == before
        and (tmp_path / "original.png").read_bytes() == contents
    )


def test_submitted_scale_and_uv_are_replaced_by_calculated_values(
    tmp_path: Path,
) -> None:
    data = raster_case(tmp_path).model_dump()
    data["views"][0]["scale"] = 0.25
    data["features"][0]["evidence"][0]["box_uv"] = (0, 0, 1, 1)
    accepted, _ = validate_interpretation(
        DrawingInterpretation.model_validate(data), workdir=SandboxWorkdir(tmp_path)
    )
    assert accepted.views[0].scale == pytest.approx(0.1)
    assert accepted.features[0].evidence[0].box_uv == pytest.approx(
        (56, 38.9, 64.2, 45.6)
    )


def test_a_corrected_measurement_recalibrates_an_enriched_file(tmp_path: Path) -> None:
    accepted, _ = validate_interpretation(
        raster_case(tmp_path), workdir=SandboxWorkdir(tmp_path)
    )
    data = accepted.model_dump()
    for dimension in data["views"][0]["dimensions"]:
        dimension["measured_length"] *= 2
    revised, _ = validate_interpretation(
        DrawingInterpretation.model_validate(data), workdir=SandboxWorkdir(tmp_path)
    )
    assert revised.views[0].scale == pytest.approx(0.05)
    assert revised.features[0].evidence[0].box_uv == pytest.approx(
        (28, 19.45, 32.1, 22.8)
    )


@pytest.mark.parametrize("measurements", [[], [(4.2, 42)], [(4.2, 42), (10, 200)]])
def test_unconfirmed_calibration_does_not_publish_scale_or_uv(
    tmp_path: Path, measurements: list
) -> None:
    original = raster_case(tmp_path, measurements)
    accepted, diagnostics = validate_interpretation(
        original, workdir=SandboxWorkdir(tmp_path)
    )
    assert diagnostics["view_front"]["status"] != "ok"
    assert accepted.views[0].scale is None
    assert accepted.features[0].evidence[0].box_uv is None
    assert accepted.views[0].image_size == (1200, 1400)
    data = original.model_dump()
    data["views"][0]["scale"] = 0.1
    data["features"][0]["evidence"][0]["box_uv"] = (56, 38.9, 64.2, 45.6)
    stale, _ = validate_interpretation(
        DrawingInterpretation.model_validate(data), workdir=SandboxWorkdir(tmp_path)
    )
    assert stale.views[0].scale is None
    assert stale.features[0].evidence[0].box_uv is None


def test_calibration_ignores_angles_and_unreadable_or_zero_figures(
    tmp_path: Path,
) -> None:
    data = raster_case(tmp_path).model_dump()
    dimensions = data["views"][0]["dimensions"]
    for name, kind, nominal, measured in [
        ("angle", "angular", 90, None),
        ("unreadable", "linear", None, 100),
        ("zero", "linear", 0, 100),
    ]:
        dimensions.append(
            dimensions[0]
            | {
                "name": f"dim_{name}",
                "kind": kind,
                "nominal_value": nominal,
                "measured_length": measured,
            }
        )
    accepted, diagnostics = validate_interpretation(
        DrawingInterpretation.model_validate(data), workdir=SandboxWorkdir(tmp_path)
    )
    assert accepted.views[0].scale == pytest.approx(0.1)
    assert len(diagnostics["view_front"]["measurements"]) == 3


def _with_dimensions(tmp_path: Path, extra) -> DrawingInterpretation:
    data = raster_case(tmp_path).model_dump()
    dimensions = data["views"][0]["dimensions"]
    for name, kind, nominal, measured in extra:
        dimensions.append(
            dimensions[0]
            | {
                "name": f"dim_{name}",
                "kind": kind,
                "nominal_value": nominal,
                "measured_length": measured,
            }
        )
    return DrawingInterpretation.model_validate(data)


def test_a_readable_linear_figure_must_carry_its_measurement(tmp_path: Path) -> None:
    submitted = _with_dimensions(tmp_path, [("slot_width", "linear", 10, None)])
    with pytest.raises(ValueError, match="dim_slot_width"):
        validate_interpretation(submitted, workdir=SandboxWorkdir(tmp_path))


def test_arcs_and_unreadable_lengths_may_stay_unmeasured(tmp_path: Path) -> None:
    submitted = _with_dimensions(
        tmp_path,
        [
            ("fillet", "radius", 16, None),
            ("bore", "diameter", 8, None),
            ("illegible", "linear", None, None),
        ],
    )
    accepted, diagnostics = validate_interpretation(
        submitted, workdir=SandboxWorkdir(tmp_path)
    )
    assert accepted.views[0].scale == pytest.approx(0.1)
    assert len(diagnostics["view_front"]["measurements"]) == 3


@pytest.mark.parametrize("defect", ["image_size", "pixel_bounds", "uv_only"])
def test_raster_rejects_inconsistent_metadata_and_regions(
    tmp_path: Path, defect: str
) -> None:
    data = raster_case(tmp_path).model_dump()
    if defect == "image_size":
        data["views"][0]["image_size"] = (1201, 1400)
    elif defect == "pixel_bounds":
        data["features"][0]["evidence"][0]["box_px"] = (0, 0, 1201, 1400)
    else:
        data["features"][0]["evidence"][0].update(box_px=None, box_uv=(0, 0, 1, 1))
    with pytest.raises(ValueError):
        validate_interpretation(
            DrawingInterpretation.model_validate(data), workdir=SandboxWorkdir(tmp_path)
        )


@pytest.mark.parametrize(
    "defect",
    ["missing", "directory", "outside_relative", "outside_absolute", "outside_symlink"],
)
def test_sheet_file_must_be_an_existing_regular_file_inside_workdir(
    tmp_path: Path, defect: str
) -> None:
    data = raster_case(tmp_path).model_dump()
    if defect == "missing":
        file = "missing.png"
    elif defect == "directory":
        file = "."
    else:
        outside = tmp_path.parent / (tmp_path.name + "_outside.png")
        Image.new("RGB", (10, 10), "white").save(outside)
        if defect == "outside_relative":
            file = "../" + outside.name
        elif defect == "outside_absolute":
            file = str(outside)
        else:
            (tmp_path / "linked.png").symlink_to(outside)
            file = "linked.png"
    data["views"][0]["file"] = file
    with pytest.raises((ValueError, OSError)):
        validate_interpretation(
            DrawingInterpretation.model_validate(data), workdir=SandboxWorkdir(tmp_path)
        )


def dxf_case(workdir: Path):
    doc = ezdxf.new()
    # Drawn away from the origin, so a region has to cite the file's own frame.
    doc.modelspace().add_lwpolyline(
        [(10, 20), (111.6, 20), (111.6, 96.2), (10, 96.2)], close=True
    )
    doc.saveas(workdir / "front.dxf")
    region = {"view": "view_front", "box_uv": [10.0, 20.0, 111.6, 96.2]}
    return DrawingInterpretation.model_validate(
        {
            "datum": "mm; lower-left input-file point defines the planar reference.",
            "views": [
                {
                    "name": "view_front",
                    "role": "front",
                    "file": "front.dxf",
                    "region": region,
                    "u_axis": "+x",
                    "v_axis": "+z",
                    "dimensions": [
                        {
                            "name": "dim_width",
                            "kind": "linear",
                            "text": "101.6",
                            "nominal_value": 101.6,
                            "measured_length": 101.6,
                            "quantity": 1,
                            "note": None,
                            "region": region,
                        }
                    ],
                }
            ],
            "features": [
                {
                    "name": "sem_plate",
                    "description": "Rectangular plate, thickness unknown.",
                    "parameters": {"width": 101.6, "height": 76.2, "thickness": None},
                    "dimension_refs": ["dim_width"],
                    "evidence": [region],
                }
            ],
        }
    )


def test_native_dxf_preserves_native_measurements_and_normalized_uv_without_mutation(
    tmp_path: Path,
) -> None:
    original = dxf_case(tmp_path)
    before = original.model_dump()
    contents = (tmp_path / "front.dxf").read_bytes()
    accepted, diagnostics = validate_interpretation(
        original, workdir=SandboxWorkdir(tmp_path)
    )
    assert diagnostics["view_front"]["status"] == "native_dxf"
    assert diagnostics["view_front"]["box_mm"] == pytest.approx(
        (10.0, 20.0, 111.6, 96.2)
    )
    sheet = accepted.views[0]
    assert sheet.image_size is None and sheet.scale is None
    # A drawing unit is a millimetre, so the drawn length is the printed one.
    assert sheet.dimensions[0].measured_length == 101.6
    assert accepted.features[0].evidence[0].box_uv == pytest.approx(
        (10.0, 20.0, 111.6, 96.2)
    )
    assert (
        original.model_dump() == before
        and (tmp_path / "front.dxf").read_bytes() == contents
    )
    revalidated, _ = validate_interpretation(accepted, workdir=SandboxWorkdir(tmp_path))
    assert revalidated == accepted


@pytest.mark.parametrize("defect", ["image_size", "scale", "pixel_box", "uv_bounds"])
def test_native_dxf_rejects_raster_metadata_and_out_of_bounds_uv(
    tmp_path: Path, defect: str
) -> None:
    data = dxf_case(tmp_path).model_dump()
    if defect == "image_size":
        data["views"][0]["image_size"] = (1200, 1400)
    elif defect == "scale":
        data["views"][0]["scale"] = 0.1
    elif defect == "pixel_box":
        data["features"][0]["evidence"][0]["box_px"] = (0, 0, 100, 100)
    else:
        data["features"][0]["evidence"][0]["box_uv"] = (10.0, 20.0, 200.0, 96.2)
    with pytest.raises(ValueError):
        validate_interpretation(
            DrawingInterpretation.model_validate(data),
            workdir=SandboxWorkdir(tmp_path),
        )


@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings:UserWarning")
def test_validation_uses_revalidated_numeric_types_without_mutating_input(
    tmp_path: Path,
) -> None:
    original = raster_case(tmp_path)
    pixel_strings = ("560", "944", "642", "1011")
    region = (
        original.features[0].evidence[0].model_copy(update={"box_px": pixel_strings})
    )
    feature = original.features[0].model_copy(update={"evidence": [region]})
    submitted = original.model_copy(update={"features": [feature]})
    accepted, _ = validate_interpretation(submitted, workdir=SandboxWorkdir(tmp_path))
    assert accepted.features[0].evidence[0].box_px == (560, 944, 642, 1011)
    assert accepted.features[0].evidence[0].box_uv == pytest.approx(
        (56, 38.9, 64.2, 45.6)
    )
    assert submitted.features[0].evidence[0].box_px == pixel_strings


@pytest.mark.parametrize("kind", ["linear", "radius", "diameter"])
def test_image_diagonal_limits_linear_measurements_only(
    tmp_path: Path, kind: str
) -> None:
    data = raster_case(tmp_path, [(10, 1000), (20, 2000), (30, 3000)]).model_dump()
    Image.new("RGB", (100, 100), "white").save(tmp_path / "front.png")
    data["features"][0]["evidence"][0]["box_px"] = (0, 0, 100, 100)
    data["views"][0]["region"]["box_px"] = (0, 0, 100, 100)
    for dimension in data["views"][0]["dimensions"]:
        dimension["kind"] = kind
        dimension["region"]["box_px"] = (0, 0, 100, 100)
    submitted = DrawingInterpretation.model_validate(data)
    if kind == "linear":
        with pytest.raises(ValueError, match="diagonal"):
            validate_interpretation(submitted, workdir=SandboxWorkdir(tmp_path))
    else:
        accepted, _ = validate_interpretation(
            submitted, workdir=SandboxWorkdir(tmp_path)
        )
        assert accepted.views[0].scale == pytest.approx(0.01)


def _page_layout(workdir: Path, *others: dict) -> DrawingInterpretation:
    """A page carrying the front view at its lower left, plus `others`."""
    Image.new("RGB", (2400, 3000), "white").save(workdir / "page.png")
    data = raster_case(workdir).model_dump()
    front = data["views"][0]
    front["region"].update(view="view_page", box_px=(100, 1800, 1300, 2800))
    views = [
        {
            "name": "view_page",
            "role": "full_page",
            "file": "page.png",
            "region": {"view": "view_page", "box_px": (0, 0, 2400, 3000)},
            "dimensions": [],
        },
        front,
    ]
    for other in others:
        role = other["role"]
        Image.new("RGB", (1200, 1400), "white").save(workdir / f"{role}.png")
        crop = deepcopy(front)
        crop.update(
            name=f"view_{role}",
            file=f"{role}.png",
            dimensions=[],
            u_axis=UNTURNED[View(role)][0],
            v_axis=UNTURNED[View(role)][1],
            **other,
        )
        crop["region"] = {"view": "view_page", "box_px": other["box_px"]}
        del crop["box_px"]
        views.append(crop)
    data["views"] = views
    return DrawingInterpretation.model_validate(data)


@pytest.mark.parametrize(
    ("role", "box_px"),
    [
        ("top", (100, 300, 1300, 1300)),
        ("right", (1500, 1800, 2300, 2800)),
        # The arrangement ditech 005 draws: a right view beside the top row.
        ("right", (1500, 300, 2300, 1300)),
    ],
)
def test_a_view_off_its_row_or_column_is_still_placed_correctly(
    tmp_path: Path, role: str, box_px: tuple
) -> None:
    accepted, _ = validate_interpretation(
        _page_layout(tmp_path, {"role": role, "box_px": box_px}),
        workdir=SandboxWorkdir(tmp_path),
    )
    assert [view.role for view in accepted.views] == [
        View.FULL_PAGE,
        View.FRONT,
        View(role),
    ]


@pytest.mark.parametrize(
    ("role", "box_px"),
    [
        ("top", (100, 2850, 1300, 2950)),
        ("bottom", (100, 300, 1300, 1300)),
        ("right", (0, 1800, 400, 2800)),
        ("left", (1500, 1800, 2300, 2800)),
    ],
)
def test_a_role_drawn_on_the_far_side_of_front_is_refused(
    tmp_path: Path, role: str, box_px: tuple
) -> None:
    with pytest.raises(LocatedError, match="far side of view_front"):
        validate_interpretation(
            _page_layout(tmp_path, {"role": role, "box_px": box_px}),
            workdir=SandboxWorkdir(tmp_path),
        )


def test_a_page_without_a_front_view_is_placed_against_nothing(tmp_path: Path) -> None:
    """ditech 005 reads as top, bottom and an end view: no view plays front."""
    layout = _page_layout(
        tmp_path,
        {"role": "top", "box_px": (100, 300, 1300, 1300)},
        {"role": "right", "box_px": (1500, 300, 2300, 1300)},
    )
    data = layout.model_dump()
    front = data["views"].pop(1)
    data["views"][1]["dimensions"] = front["dimensions"]
    for dimension in front["dimensions"]:
        dimension["region"]["view"] = "view_top"
    for feature in data["features"]:
        for region in feature["evidence"]:
            region["view"] = "view_top"

    accepted, _ = validate_interpretation(
        DrawingInterpretation.model_validate(data), workdir=SandboxWorkdir(tmp_path)
    )

    assert [view.role for view in accepted.views] == [
        View.FULL_PAGE,
        View.TOP,
        View.RIGHT,
    ]
