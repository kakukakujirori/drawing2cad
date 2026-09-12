"""Validate real sheet files, calibration and evidence without model calls."""

from copy import deepcopy
from pathlib import Path

import ezdxf
import pytest
from PIL import Image

from tests.zeroshot.messages.test_interpretation_contracts import pin_interpretation
from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.stages.interpretation.contracts import DrawingInterpretation
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


def test_shared_png_views_keep_independent_scales(tmp_path: Path) -> None:
    data = raster_case(tmp_path).model_dump()
    top = deepcopy(data["views"][0])
    top.update(name="view_top", role="top")
    top["region"].update(view="view_top", box_px=(50, 50, 350, 250))
    for index, dimension in enumerate(top["dimensions"]):
        dimension["name"] = f"dim_top_{index}"
        dimension["measured_length"] /= 2
        dimension["region"]["view"] = "view_top"
    data["views"].append(top)
    data["features"][0]["evidence"].append(
        {"view": "view_top", "box_px": (100, 100, 300, 200)}
    )
    accepted, _ = validate_interpretation(
        DrawingInterpretation.model_validate(data), workdir=SandboxWorkdir(tmp_path)
    )
    assert accepted.views[0].file == accepted.views[1].file
    assert [view.image_size for view in accepted.views] == [(1200, 1400), (1200, 1400)]
    assert [view.scale for view in accepted.views] == pytest.approx([0.1, 0.2])
    assert accepted.views[1].region.box_uv == pytest.approx((10, 230, 70, 270))
    assert accepted.features[0].evidence[1].box_uv == pytest.approx((20, 240, 60, 260))


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
    top.update(name="view_top", role="top", file="/work/top.png")
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


def test_uv_within_one_pixel_is_replaced_by_calculated_coordinates(
    tmp_path: Path,
) -> None:
    data = raster_case(tmp_path).model_dump()
    data["features"][0]["evidence"][0]["box_uv"] = (56.05, 38.85, 64.25, 45.55)
    accepted, _ = validate_interpretation(
        DrawingInterpretation.model_validate(data), workdir=SandboxWorkdir(tmp_path)
    )
    assert accepted.features[0].evidence[0].box_uv == pytest.approx(
        (56, 38.9, 64.2, 45.6)
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
    data["features"][0]["evidence"][0]["box_uv"] = (56, 38.9, 64.2, 45.6)
    with pytest.raises(ValueError):
        validate_interpretation(
            DrawingInterpretation.model_validate(data), workdir=SandboxWorkdir(tmp_path)
        )


def test_calibration_ignores_angles_unreadable_zero_and_unmeasured_figures(
    tmp_path: Path,
) -> None:
    data = raster_case(tmp_path).model_dump()
    dimensions = data["views"][0]["dimensions"]
    for name, kind, nominal, measured in [
        ("angle", "angular", 90, None),
        ("unreadable", "linear", None, 100),
        ("zero", "linear", 0, 100),
        ("unmeasured", "linear", 10, None),
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


@pytest.mark.parametrize(
    "defect", ["image_size", "scale", "uv", "pixel_bounds", "uv_only"]
)
def test_raster_rejects_inconsistent_metadata_and_regions(
    tmp_path: Path, defect: str
) -> None:
    data = raster_case(tmp_path).model_dump()
    if defect == "image_size":
        data["views"][0]["image_size"] = (1201, 1400)
    elif defect == "scale":
        data["views"][0]["scale"] = 0.25
    elif defect == "uv":
        data["features"][0]["evidence"][0]["box_uv"] = (0, 0, 1, 1)
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
    doc.units = 1  # Inches: physical conversion is explicit sheet metadata.
    doc.modelspace().add_lwpolyline(
        [(10, 20), (14, 20), (14, 23), (10, 23)], close=True
    )
    doc.saveas(workdir / "front.dxf")
    region = {"view": "view_front", "box_uv": [0, 0, 101.6, 76.2]}
    return DrawingInterpretation.model_validate(
        {
            "datum": "mm; lower-left input-file point defines the planar reference.",
            "views": [
                {
                    "name": "view_front",
                    "role": "front",
                    "file": "front.dxf",
                    "region": region,
                    "dimensions": [
                        {
                            "name": "dim_width",
                            "kind": "linear",
                            "text": "101.6",
                            "nominal_value": 101.6,
                            "measured_length": 4,
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
        original,
        workdir=SandboxWorkdir(tmp_path),
        dxf_mm_per_unit={"view_front": 25.4},
    )
    assert diagnostics["view_front"]["status"] == "native_dxf"
    assert diagnostics["view_front"]["origin_native"] == pytest.approx((10, 20))
    assert diagnostics["view_front"]["mm_per_unit"] == 25.4
    assert diagnostics["view_front"]["size_mm"] == pytest.approx((101.6, 76.2))
    sheet = accepted.views[0]
    assert sheet.image_size is None and sheet.scale is None
    assert sheet.dimensions[0].measured_length == 4
    assert accepted.features[0].evidence[0].box_uv == pytest.approx((0, 0, 101.6, 76.2))
    assert (
        original.model_dump() == before
        and (tmp_path / "front.dxf").read_bytes() == contents
    )
    revalidated, _ = validate_interpretation(
        accepted,
        workdir=SandboxWorkdir(tmp_path),
        dxf_mm_per_unit={"view_front": 25.4},
    )
    assert revalidated == accepted
    with pytest.raises(ValueError):
        validate_interpretation(original, workdir=SandboxWorkdir(tmp_path))


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
        data["features"][0]["evidence"][0]["box_uv"] = (0, 0, 120, 76.2)
    with pytest.raises(ValueError):
        validate_interpretation(
            DrawingInterpretation.model_validate(data),
            workdir=SandboxWorkdir(tmp_path),
            dxf_mm_per_unit={"view_front": 25.4},
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
