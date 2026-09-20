"""Read sheet files and calibrate the Regions in a submitted interpretation."""

import math
from pathlib import Path
from typing import Any

from PIL import Image

from zeroshot.pipeline.messages.manifest import read_dxf_frame
from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.stages._base.validate import KeyLocation, LocatedError
from zeroshot.pipeline.stages.interpretation.contracts import (
    UNDECIDED,
    DrawingInterpretation,
    DrawingView,
    Region,
    View,
)
from zeroshot.pipeline.tools.calculate_drawing_scale import calculate_drawing_scale


def _region(
    region: dict[str, Any],
    subject: str,
    location: KeyLocation,
    sheet: tuple[float, float, float, float],
    scale: float | None,
    *,
    dxf: bool,
) -> dict[str, Any]:
    px, uv = region["box_px"], region["box_uv"]
    if dxf:
        if px is not None or uv is None:
            raise LocatedError.at(
                location, f"{subject}: DXF requires box_uv and forbids box_px"
            )
        box = uv
    else:
        if px is None:
            raise LocatedError.at(location, f"{subject}: raster regions require box_px")
        box = px
    if any(
        edge < bound - 1e-7 for edge, bound in zip(box[:2], sheet[:2], strict=True)
    ) or any(
        edge > bound + 1e-7 for edge, bound in zip(box[2:], sheet[2:], strict=True)
    ):
        # Naming the view it is measured against: the box is usually the whole
        # page's coordinates left on a region that cites a crop.
        raise LocatedError.at(
            location,
            f"{subject}: region {tuple(box)} lies outside {region['view']}, "
            f"which spans {tuple(round(edge, 3) for edge in sheet)}",
        )

    if not dxf:
        # Derived from box_px and the current scale; a submitted value may be stale.
        uv = None
        if scale is not None:
            x0, y0, x1, y1 = px
            uv = (
                x0 * scale,
                (sheet[3] - y1) * scale,
                x1 * scale,
                (sheet[3] - y0) * scale,
            )
    return {**region, "box_uv": uv}


def _require_third_angle_placement(views: list[DrawingView]) -> None:
    """Refuse wrong configurations such as top view being placed below the front view, etc."""

    def _page_centre(region: Region) -> tuple[float, float]:
        """The region's middle on its parent sheet, x rightwards and y upwards."""
        if region.box_px is not None:  # pixel y runs down the page, UV runs up
            x0, y0, x1, y1 = region.box_px
            return ((x0 + x1) / 2, -(y0 + y1) / 2)
        if region.box_uv is not None:
            x0, y0, x1, y1 = region.box_uv
            return ((x0 + x1) / 2, (y0 + y1) / 2)
        raise ValueError(f"{region.view}: region has neither box_px nor box_uv")

    front = next((view for view in views if view.role is View.FRONT), None)
    if front is None:
        return
    fx, fy = _page_centre(front.region)
    for index, view in enumerate(views):
        step = {
            View.TOP: (0, 1),
            View.BOTTOM: (0, -1),
            View.RIGHT: (1, 0),
            View.LEFT: (-1, 0),
        }.get(view.role)
        if step is None or view.region.view != front.region.view:
            continue
        x, y = _page_centre(view.region)
        if (x - fx) * step[0] + (y - fy) * step[1] < 0:
            raise LocatedError.at(
                ("views", index, "role"),
                f"{view.name} is drawn on the far side of {front.name} from where a "
                f"{view.role.value} view belongs; re-read the roles or the arrangement",
            )


def validate_interpretation(
    interpretation: DrawingInterpretation,
    *,
    workdir: SandboxWorkdir,
) -> tuple[DrawingInterpretation, dict[str, dict[str, Any]]]:
    """Validate and enrich without mutating model output or the input files.

    Raster submissions normally leave image_size, scale and box_uv null.
    Scale and box_uv are recalculated on every call, since a measurement fix
    changes them. A stored image_size must still match its file, which catches
    a resized sheet. No consensus (including one measurement) leaves scale/UV
    null; return diagnostics to the interpreter instead of guessing a scale.

    A DXF keeps its own millimetre coordinates, never preview pixels, so its
    measured_length is already a length and is retained rather than passed to
    RANSAC with pixel tolerances.
    """
    interpretation = DrawingInterpretation.model_validate(interpretation.model_dump())
    if UNDECIDED in interpretation.datum:
        raise LocatedError.at(
            ("datum",),
            f"datum still holds {UNDECIDED}: state the model frame, and "
            "report any doubt as a concern in your stage report",
        )
    # A readable printed length is what calibrates a file, so its measurement
    # is required; an unreadable one cannot calibrate and stays optional.
    if unmeasured := [
        (("views", v, "dimensions", d), dim.name)
        for v, view in enumerate(interpretation.views)
        for d, dim in enumerate(view.dimensions)
        if dim.kind == "linear"
        and dim.nominal_value is not None
        and dim.nominal_value > 0
        and dim.measured_length is None
    ]:
        raise LocatedError.at(
            unmeasured[0][0],
            "measure every linear dimension whose printed value you read, in "
            f"its own view file's units: {', '.join(n for _, n in unmeasured)}",
        )
    _require_third_angle_placement(interpretation.views)
    data = interpretation.model_dump()
    # The bounds a region must fall inside: a DXF's own extent, a raster's pixels.
    sheets: dict[Path, tuple[float, float, float, float]] = {}
    dxf_frames: dict[Path, dict[str, Any]] = {}
    reports: dict[str, dict[str, Any]] = {}
    contexts: dict[
        str, tuple[tuple[float, float, float, float], float | None, bool]
    ] = {}

    for index, (view, output) in enumerate(zip(interpretation.views, data["views"])):
        location: KeyLocation = ("views", index)
        path = workdir.sandbox_to_host_path(view.file)
        if path.is_symlink() or not path.resolve().is_relative_to(
            workdir.host_bind_dir.resolve()
        ):
            raise LocatedError.at(
                (*location, "file"),
                f"{view.name}.file must stay inside the workspace without symlinks",
            )
        if not path.is_file():
            raise LocatedError.at(
                (*location, "file"),
                f"{view.name}.file is not a regular file: {view.file}",
            )
        path = path.resolve()
        dxf = path.suffix.lower() == ".dxf"
        if path not in sheets:
            if dxf:
                dxf_frames[path] = read_dxf_frame(path)
                sheets[path] = dxf_frames[path]["box_mm"]
            else:
                with Image.open(path) as image:
                    sheets[path] = (0, 0, *image.size)
                    image.verify()
        sheet = sheets[path]
        size = (sheet[2] - sheet[0], sheet[3] - sheet[1])
        if dxf:
            if view.image_size is not None or view.scale is not None:
                raise LocatedError.at(
                    (*location, "image_size"),
                    f"{view.name}: native DXF cannot have image_size or pixel scale",
                )
            scale = None
            reports[view.name] = {"status": "native_dxf", **dxf_frames[path]}
        else:
            if view.image_size is not None and view.image_size != size:
                raise LocatedError.at(
                    (*location, "image_size"),
                    f"{view.name}: image_size disagrees with the source file",
                )
            for position, dim in enumerate(view.dimensions):
                # A fitted radius of a partial arc can exceed the image bounds;
                # a directly measured linear segment cannot.
                if (
                    dim.kind == "linear"
                    and dim.measured_length is not None
                    and dim.measured_length > math.hypot(*size)
                ):
                    raise LocatedError.at(
                        (*location, "dimensions", position, "measured_length"),
                        f"{dim.name}: measured_length exceeds sheet image diagonal",
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
            # Recalculated on every write; a submitted scale is ignored.
            scale = report["scale"] if report["status"] == "ok" else None
            reports[view.name] = report
            output.update(image_size=size, scale=scale)
        contexts[view.name] = (sheet, scale, dxf)

    for index, output in enumerate(data["views"]):
        region = output["region"]
        sheet, scale, dxf = contexts[region["view"]]
        output["region"] = _region(
            region,
            f"{output['name']}.region",
            ("views", index, "region"),
            sheet,
            scale,
            dxf=dxf,
        )
        for position, dim in enumerate(output["dimensions"]):
            sheet, scale, dxf = contexts[dim["region"]["view"]]
            dim["region"] = _region(
                dim["region"],
                f"{dim['name']}.region",
                ("views", index, "dimensions", position, "region"),
                sheet,
                scale,
                dxf=dxf,
            )

    for index, (feature, output) in enumerate(
        zip(interpretation.features, data["features"])
    ):
        output["evidence"] = []
        for position, region in enumerate(feature.evidence):
            sheet, scale, dxf = contexts[region.view]
            output["evidence"].append(
                _region(
                    region.model_dump(),
                    f"{feature.name}.evidence[{position}]",
                    ("features", index, "evidence", position),
                    sheet,
                    scale,
                    dxf=dxf,
                )
            )
    return DrawingInterpretation.model_validate(data), reports
