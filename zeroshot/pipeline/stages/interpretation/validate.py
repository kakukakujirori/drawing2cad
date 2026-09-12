"""Read sheet files and calibrate the Regions in a submitted interpretation."""

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import ezdxf
from ezdxf import bbox
from PIL import Image

from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.stages.interpretation.contracts import DrawingInterpretation
from zeroshot.pipeline.tools.calculate_drawing_scale import calculate_drawing_scale


def read_dxf_frame(path: Path, mm_per_unit: float) -> dict[str, Any]:
    """Input metadata to expose before asking a model for native-DXF regions.

    A DXF unit need not be a millimetre. The input adapter must supply its
    physical conversion, accounting for the dataset's drawing-scale convention.
    For a native point (x, y), u = (x - origin_x) * mm_per_unit, and likewise v.
    The source file itself is preserved; its entity order is irrelevant.
    """
    if not math.isfinite(mm_per_unit) or mm_per_unit <= 0:
        raise ValueError("DXF mm_per_unit must be finite and positive")
    bounds = bbox.extents(ezdxf.readfile(path).modelspace())
    if not bounds.has_data:
        raise ValueError(f"{path}: DXF has no measurable sheet bounds")
    if not all(
        math.isfinite(value)
        for point in (bounds.extmin, bounds.extmax)
        for value in point
    ):
        raise ValueError(f"{path}: DXF bounds must be finite")
    if abs(bounds.extmin.z) > 1e-6 or abs(bounds.extmax.z) > 1e-6:
        raise ValueError(f"{path}: expected a planar XY drawing")
    size = (bounds.size.x * mm_per_unit, bounds.size.y * mm_per_unit)
    if any(not math.isfinite(value) or value <= 0 for value in size):
        raise ValueError(
            f"{path}: DXF must have positive finite sheet width and height"
        )
    return {
        "origin_native": (bounds.extmin.x, bounds.extmin.y),
        "mm_per_unit": mm_per_unit,
        "size_mm": size,
    }


def _region(
    region: dict[str, Any],
    subject: str,
    size: tuple[float, float],
    scale: float | None,
    *,
    dxf: bool,
) -> dict[str, Any]:
    px, uv = region["box_px"], region["box_uv"]
    if dxf:
        if px is not None or uv is None:
            raise ValueError(f"{subject}: DXF requires box_uv and forbids box_px")
        box = uv
    else:
        if px is None:
            raise ValueError(f"{subject}: raster regions require box_px")
        box = px
    if box[2] > size[0] + 1e-7 or box[3] > size[1] + 1e-7:
        raise ValueError(f"{subject}: region exceeds referenced file bounds {size}")

    if not dxf:
        expected = None
        if scale is not None:
            x0, y0, x1, y1 = px
            expected = (
                x0 * scale,
                (size[1] - y1) * scale,
                x1 * scale,
                (size[1] - y0) * scale,
            )
        if uv is not None:
            if expected is None:
                raise ValueError(
                    f"{subject}: cannot validate box_uv without scale consensus"
                )
            # Allow one source pixel of rounding, then store the exact conversion.
            if any(
                not math.isclose(a, b, rel_tol=0, abs_tol=scale)
                for a, b in zip(uv, expected)
            ):
                raise ValueError(f"{subject}: box_uv disagrees with box_px")
        uv = expected
    return {**region, "box_uv": uv}


def validate_interpretation(
    interpretation: DrawingInterpretation,
    *,
    workdir: SandboxWorkdir,
    dxf_mm_per_unit: Mapping[str, float] | None = None,
) -> tuple[DrawingInterpretation, dict[str, dict[str, Any]]]:
    """Validate and enrich without mutating model output or the input files.

    Raster submissions normally leave image_size, scale and box_uv null.
    Already enriched values are accepted only if they still agree with the file
    and measurements. No consensus (including one measurement) leaves scale/UV
    null; return diagnostics to the interpreter instead of guessing a scale.

    Native DXF uses normalized, sheet-relative UV in mm, never preview pixels.
    dxf_mm_per_unit is authoritative input metadata, not an LLM-submitted field.
    Keys are DrawingView names. DXF measured_length uses native drawing units
    and is retained; it is not passed to RANSAC with pixel tolerances.
    Supply 1.0 only when the dataset guarantees 1:1 millimetre drawing units.
    Use read_dxf_frame to expose the origin and conversion at input preparation.
    """
    interpretation = DrawingInterpretation.model_validate(interpretation.model_dump())
    data = interpretation.model_dump()
    sizes: dict[tuple[Path, float | None], tuple[float, float]] = {}
    dxf_frames: dict[tuple[Path, float | None], dict[str, Any]] = {}
    reports: dict[str, dict[str, Any]] = {}
    contexts: dict[str, tuple[tuple[float, float], float | None, bool]] = {}

    for view, output in zip(interpretation.views, data["views"]):
        path = workdir.sandbox_to_host_path(view.file)
        if path.is_symlink() or not path.resolve().is_relative_to(
            workdir.host_bind_dir.resolve()
        ):
            raise ValueError(
                f"{view.name}.file must stay inside the workspace without symlinks"
            )
        if not path.is_file():
            raise ValueError(f"{view.name}.file is not a regular file: {view.file}")
        path = path.resolve()
        dxf = path.suffix.lower() == ".dxf"
        factor = None
        if dxf:
            factor = (dxf_mm_per_unit or {}).get(view.name)
            if factor is None:
                raise ValueError(
                    f"{view.name}: supply explicit DXF mm_per_unit input metadata"
                )
        key = (path, factor)
        if key not in sizes:
            if dxf:
                dxf_frames[key] = read_dxf_frame(path, factor)
                sizes[key] = dxf_frames[key]["size_mm"]
            else:
                with Image.open(path) as image:
                    sizes[key] = image.size
                    image.verify()
        size = sizes[key]
        if dxf:
            if view.image_size is not None or view.scale is not None:
                raise ValueError(
                    f"{view.name}: native DXF cannot have image_size or pixel scale"
                )
            scale = None
            reports[view.name] = {"status": "native_dxf", **dxf_frames[key]}
        else:
            if view.image_size is not None and view.image_size != size:
                raise ValueError(
                    f"{view.name}: image_size disagrees with the source file"
                )
            for dim in view.dimensions:
                # A fitted radius of a partial arc can exceed the image bounds;
                # a directly measured linear segment cannot.
                if (
                    dim.kind == "linear"
                    and dim.measured_length is not None
                    and dim.measured_length > math.hypot(*size)
                ):
                    raise ValueError(
                        f"{dim.name}: measured_length exceeds sheet image diagonal"
                    )
            report = calculate_drawing_scale(
                [
                    {
                        "name": dim.name,
                        "nominal": dim.nominal_value,
                        "measured": dim.measured_length,
                    }
                    for dim in view.dimensions
                    if dim.kind != "angular"
                    and dim.nominal_value is not None
                    and dim.nominal_value > 0
                    and dim.measured_length is not None
                ]
            )
            scale = report["scale"] if report["status"] == "ok" else None
            if view.scale is not None and (
                scale is None
                or not math.isclose(view.scale, scale, rel_tol=1e-9, abs_tol=0)
            ):
                raise ValueError(
                    f"{view.name}: scale disagrees with dimension calibration"
                )
            reports[view.name] = report
            output.update(image_size=size, scale=scale)
        contexts[view.name] = (size, scale, dxf)

    for output in data["views"]:
        region = output["region"]
        size, scale, dxf = contexts[region["view"]]
        output["region"] = _region(
            region, f"{output['name']}.region", size, scale, dxf=dxf
        )
        for dim in output["dimensions"]:
            size, scale, dxf = contexts[dim["region"]["view"]]
            dim["region"] = _region(
                dim["region"], f"{dim['name']}.region", size, scale, dxf=dxf
            )

    for feature, output in zip(interpretation.features, data["features"]):
        output["evidence"] = []
        for index, region in enumerate(feature.evidence):
            size, scale, dxf = contexts[region.view]
            output["evidence"].append(
                _region(
                    region.model_dump(),
                    f"{feature.name}.evidence[{index}]",
                    size,
                    scale,
                    dxf=dxf,
                )
            )
    return DrawingInterpretation.model_validate(data), reports
