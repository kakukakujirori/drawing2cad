"""Read sheet files and calibrate the Regions in a submitted interpretation."""

import math
from dataclasses import dataclass, replace
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

type _Box = tuple[float, ...]


@dataclass(frozen=True)
class _Sheet:
    """The frame one view's coordinates are read in."""

    bounds: _Box  # x0, y0, x1, y1 in the file's own units
    scale: float | None  # millimetres per pixel; a DXF is already in millimetres
    dxf: bool

    @property
    def size(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bounds
        return (x1 - x0, y1 - y0)

    @property
    def wanted_box(self) -> str:
        return (
            "DXF requires box_uv and forbids box_px"
            if self.dxf
            else "raster regions require box_px"
        )

    def measured_box(self, region: dict[str, Any]) -> _Box | None:
        """The region's box in this file's own coordinates, if it gave one."""
        if not self.dxf:
            return region["box_px"]
        return None if region["box_px"] is not None else region["box_uv"]

    def holds(self, box: _Box) -> bool:
        x0, y0, x1, y1 = self.bounds
        return (
            box[0] >= x0 - 1e-7
            and box[1] >= y0 - 1e-7
            and box[2] <= x1 + 1e-7
            and box[3] <= y1 + 1e-7
        )

    def millimetres(self, box: _Box) -> _Box | None:
        """The box in mm: a DXF already is one, a raster needs a fitted scale."""
        if self.dxf:
            return box
        if self.scale is None:
            return None
        x0, y0, x1, y1 = box
        height = self.bounds[3]
        return (
            x0 * self.scale,
            (height - y1) * self.scale,
            x1 * self.scale,
            (height - y0) * self.scale,
        )


def _region(
    region: dict[str, Any],
    subject: str,
    location: KeyLocation,
    sheet: _Sheet,
) -> dict[str, Any]:
    """The region with box_uv recalculated, refused if it is not on its sheet."""
    box = sheet.measured_box(region)
    if box is None:
        raise LocatedError.at(location, f"{subject}: {sheet.wanted_box}")
    if not sheet.holds(box):
        # Naming the view it is measured against: the box is usually the whole
        # page's coordinates left on a region that cites a crop.
        raise LocatedError.at(
            location,
            f"{subject}: region {tuple(box)} lies outside {region['view']}, "
            f"which spans {tuple(round(edge, 3) for edge in sheet.bounds)}",
        )
    # Derived from the sheet on every call; a submitted box_uv may be stale.
    return {**region, "box_uv": sheet.millimetres(box)}


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


def _require_a_readable_submission(interpretation: DrawingInterpretation) -> None:
    """Refuse a submission no file can calibrate or lay out."""
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


def _calibrate_sheets(
    interpretation: DrawingInterpretation,
    data: dict[str, Any],
    workdir: SandboxWorkdir,
) -> tuple[dict[str, _Sheet], dict[str, dict[str, Any]]]:
    """Open each view's file, fit a raster's scale, and write both back.

    Returns the frame each view is read in, and the diagnostics for the model.
    """
    # One file may carry several views, and is opened once for all of them.
    bounds: dict[Path, _Box] = {}
    reports: dict[str, dict[str, Any]] = {}
    sheets: dict[str, _Sheet] = {}

    for index, (view, output) in enumerate(zip(interpretation.views, data["views"])):
        location: KeyLocation = ("views", index)
        path = _readable_path(view, location, workdir)
        if path not in bounds:
            bounds[path] = _bounds_of(path)
        sheet = _Sheet(bounds[path], scale=None, dxf=path.suffix.lower() == ".dxf")
        if sheet.dxf:
            if view.image_size is not None or view.scale is not None:
                raise LocatedError.at(
                    (*location, "image_size"),
                    f"{view.name}: native DXF cannot have image_size or pixel scale",
                )
            reports[view.name] = {"status": "native_dxf", "box_mm": sheet.bounds}
        else:
            sheet, reports[view.name] = _calibrated_raster(
                view, output, sheet, location
            )
        sheets[view.name] = sheet
    return sheets, reports


def _readable_path(
    view: DrawingView, location: KeyLocation, workdir: SandboxWorkdir
) -> Path:
    """The view's file, refused if it leaves the workspace or is not a file."""
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
    return path.resolve()


def _bounds_of(path: Path) -> _Box:
    """The span of the file's own coordinates: a DXF's extents, a raster's pixels."""
    if path.suffix.lower() == ".dxf":
        return read_dxf_frame(path)["box_mm"]
    with Image.open(path) as image:
        bounds = (0, 0, *image.size)
        image.verify()
    return bounds


def _calibrated_raster(
    view: DrawingView,
    output: dict[str, Any],
    sheet: _Sheet,
    location: KeyLocation,
) -> tuple[_Sheet, dict[str, Any]]:
    """Fit millimetres per pixel from the printed lengths, and record it."""
    size = sheet.size
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
    output.update(image_size=size, scale=scale)
    return replace(sheet, scale=scale), report


def _calibrate_regions(
    interpretation: DrawingInterpretation,
    data: dict[str, Any],
    sheets: dict[str, _Sheet],
) -> None:
    """Recalculate every region in the artifact against the sheet it cites."""
    for index, output in enumerate(data["views"]):
        region = output["region"]
        output["region"] = _region(
            region,
            f"{output['name']}.region",
            ("views", index, "region"),
            sheets[region["view"]],
        )
        for position, dim in enumerate(output["dimensions"]):
            dim["region"] = _region(
                dim["region"],
                f"{dim['name']}.region",
                ("views", index, "dimensions", position, "region"),
                sheets[dim["region"]["view"]],
            )

    for index, (feature, output) in enumerate(
        zip(interpretation.features, data["features"])
    ):
        output["evidence"] = []
        for position, region in enumerate(feature.evidence):
            output["evidence"].append(
                _region(
                    region.model_dump(),
                    f"{feature.name}.evidence[{position}]",
                    ("features", index, "evidence", position),
                    sheets[region.view],
                )
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
    _require_a_readable_submission(interpretation)
    data = interpretation.model_dump()
    sheets, reports = _calibrate_sheets(interpretation, data, workdir)
    _calibrate_regions(interpretation, data, sheets)
    return DrawingInterpretation.model_validate(data), reports
