"""Read sheet files and calibrate the Regions in a submitted interpretation."""

import math
import re
from dataclasses import dataclass, replace
from itertools import combinations, product
from pathlib import Path
from statistics import median
from typing import Any

from PIL import Image

from zeroshot.pipeline.messages.manifest import read_dxf_frame
from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.stages._base.validate import KeyLocation, LocatedError
from zeroshot.pipeline.stages.interpretation.contracts import (
    ORTHOGRAPHIC_VIEWS,
    PICTORIAL_VIEWS,
    UNDECIDED,
    VIEW_PLACEMENT,
    DrawingInterpretation,
    DrawingView,
    ProjectionMethod,
    Region,
    View,
)
from zeroshot.pipeline.tools.calculate_drawing_scale import calculate_drawing_scale

type _Box = tuple[float, ...]

# Provisional annotation limit; calibrated separately from CAD difference scores.
MAX_DIMENSION_REGION_FRACTION = 0.25
# Relative spread within which views' scales count as one, as in RANSAC.
SCALE_AGREEMENT = 0.02
# A number in a callout, with either a decimal point or a decimal comma.
_PRINTED_NUMBER = re.compile(r"\d+(?:[.,]\d+)?|[.,]\d+")
# A length off a printed number by a power of ten, or by 25.4 either way,
# reads that number in other units.
_UNIT_FACTORS = (1.0, 25.4, 1 / 25.4)
_UNIT_FACTOR_TOLERANCE = 0.01


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


_SIDE_OF = {
    (0, 1): "above",
    (0, -1): "below",
    (1, 0): "to the right of",
    (-1, 0): "to the left of",
}


def _page_centre(region: Region) -> tuple[float, float]:
    """The region's middle on its parent sheet, x rightwards and y upwards."""
    if region.box_px is not None:  # pixel y runs down the page, UV runs up
        x0, y0, x1, y1 = region.box_px
        return ((x0 + x1) / 2, -(y0 + y1) / 2)
    if region.box_uv is not None:
        x0, y0, x1, y1 = region.box_uv
        return ((x0 + x1) / 2, (y0 + y1) / 2)
    raise ValueError(f"{region.view}: region has neither box_px nor box_uv")


def _require_conventional_placement(
    views: list[DrawingView], method: ProjectionMethod
) -> None:
    """Refuse a view drawn where the projection method puts another role.

    A third-angle top view below front, or a first-angle one above it, is a
    misread role: the mirrored reading projects just as well.
    """
    front = next((view for view in views if view.role is View.FRONT), None)
    if front is None:
        return
    fx, fy = _page_centre(front.region)
    for index, view in enumerate(views):
        step = VIEW_PLACEMENT[method].get(view.role)
        if step is None or view.region.view != front.region.view:
            continue
        x, y = _page_centre(view.region)
        if (x - fx) * step[0] + (y - fy) * step[1] < 0:
            raise LocatedError.at(
                ("views", index, "role"),
                f"{view.name} sits {_SIDE_OF[(-step[0], -step[1])]} {front.name}, "
                f"but {method.replace('_', '-')} projection draws the "
                f"{view.role.value} view {_SIDE_OF[step]} front; "
                "re-read the roles or the arrangement, "
                "and if you change the u/v-axis assignment, update any inconsistencies in "
                "the affected feature descriptions, XYZ positions, direction parameters, "
                "and evidence as well.",
            )


def _printed_numbers(text: str) -> set[float]:
    """Every number a callout prints; a comma is a decimal comma or groups thousands."""
    numbers = set()
    for token in _PRINTED_NUMBER.findall(text):
        numbers |= {float(token.replace(",", ".")), float(token.replace(",", ""))}
    return numbers


def _in_other_units(value: float, printed: set[float]) -> bool:
    """Whether a length is a printed number times a unit factor."""
    tolerance = math.log10(1 + _UNIT_FACTOR_TOLERANCE)
    for number, factor in product(printed - {0.0}, _UNIT_FACTORS):
        exponent = math.log10(value / number / factor)
        if abs(exponent - round(exponent)) <= tolerance and (
            factor != 1.0 or round(exponent) != 0
        ):
            return True
    return False


def _require_printed_values(interpretation: DrawingInterpretation) -> None:
    """Refuse a nominal_value that is not a number its text prints."""
    misread = []
    for v, view in enumerate(interpretation.views):
        for d, dim in enumerate(view.dimensions):
            printed = _printed_numbers(dim.text)
            if dim.nominal_value is not None and not any(
                math.isclose(dim.nominal_value, number, rel_tol=1e-6, abs_tol=1e-9)
                for number in printed
            ):
                location = ("views", v, "dimensions", d, "nominal_value")
                misread.append((location, dim, printed))
    if not misread:
        return
    readings = "; ".join(
        f"{dim.name} text {dim.text!r}, nominal_value {dim.nominal_value:g}"
        for _, dim, _ in misread
    )
    message = (
        f"nominal_value is the number its text prints, but these differ: {readings}. "
        "Read each callout again, and record its text exactly as printed and "
        "that number as nominal_value."
    )
    if any(
        dim.kind != "angular"
        and dim.nominal_value
        and _in_other_units(dim.nominal_value, printed)
        for _, dim, printed in misread
    ):
        message += (
            " Printed lengths are millimetres exactly as printed, whatever their "
            "decimals or the part's apparent size: never convert or rescale them, "
            "and size the part from them."
        )
    raise LocatedError.at(misread[0][0], message)


def _require_dimensions_on_orthographic_views(views: list[DrawingView]) -> None:
    """A pictorial fixes no axes, so a view that carries dimensions is not one."""
    for index, view in enumerate(views):
        if view.role in PICTORIAL_VIEWS and view.dimensions:
            names = ", ".join(dim.name for dim in view.dimensions)
            raise LocatedError.at(
                ("views", index, "role"),
                f"{view.name} carries printed dimensions ({names}). A view with "
                f"printed dimensions is an orthographic view: if {view.name} is "
                "one, give it the role it shows (front, back, top, bottom, left or "
                "right) with its u_axis and v_axis; if a dimension measures "
                "geometry drawn in another orthographic view, register it under "
                "that view.",
            )


def _require_a_readable_submission(
    interpretation: DrawingInterpretation, method: ProjectionMethod
) -> None:
    """Refuse a submission no file can calibrate or lay out."""
    if UNDECIDED in interpretation.datum:
        raise LocatedError.at(
            ("datum",),
            f"datum still holds {UNDECIDED}: state the model frame, and "
            "report any doubt as a concern in your stage report",
        )
    _require_printed_values(interpretation)
    _require_dimensions_on_orthographic_views(interpretation.views)
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
            f"its own view file's units: {', '.join(n for _, n in unmeasured)}. "
            "If you cannot locate the entity that annotation measures, "
            "remove it from dimensions and report it in concerns.",
        )
    _require_conventional_placement(interpretation.views, method)


def _require_distinct_view_files(
    views: list[DrawingView], paths: dict[str, Path]
) -> None:
    """Different roles need separate files."""
    for (_, first), (j, second) in combinations(enumerate(views), 2):
        if paths[first.name] == paths[second.name] and first.role is not second.role:
            raise LocatedError.at(
                ("views", j, "file"),
                f"{second.name} ({second.role.value}) shares a file with "
                f"{first.name} ({first.role.value}); different roles must use different files. "
                "Save separate crops and update their evidence coordinates.",
            )


def _require_raster_crops(views: list[DrawingView], paths: dict[str, Path]) -> None:
    """A derived raster must be the unmodified, native-size crop it declares."""
    for index, view in enumerate(views):
        region = view.region
        path, parent_path = paths[view.name], paths[region.view]
        if region.view == view.name or ".dxf" in {
            path.suffix.lower(),
            parent_path.suffix.lower(),
        }:
            continue
        # Region validation has already checked the parent's pixel bounds.
        assert region.box_px is not None
        x0, y0, x1, y1 = region.box_px
        with Image.open(parent_path) as parent, Image.open(path) as crop:
            if crop.size != (x1 - x0, y1 - y0) or (
                crop.convert("RGBA").tobytes()
                != parent.crop(region.box_px).convert("RGBA").tobytes()
            ):
                raise LocatedError.at(
                    ("views", index, "file"),
                    f"{view.name}.file must exactly match the unmodified 1:1 crop of "
                    f"{region.view} at region.box_px={region.box_px}. "
                    "Save the crop without resizing, rotating or preprocessing; "
                    "keep observation-only crops and zooms in separate files. "
                    "If the registered crop changes, update its Region and measurements.",
                )


def _calibrate_sheets(
    interpretation: DrawingInterpretation,
    data: dict[str, Any],
    paths: dict[str, Path],
) -> tuple[dict[str, _Sheet], dict[str, dict[str, Any]]]:
    """Open each view's file, fit a raster's scale, and write both back.

    Returns the frame each view is read in, and the diagnostics for the model.
    """
    # Full-page aliases and repeated roles may share a file, which is opened once.
    bounds: dict[Path, _Box] = {}
    reports: dict[str, dict[str, Any]] = {}
    sheets: dict[str, _Sheet] = {}

    for index, (view, output) in enumerate(zip(interpretation.views, data["views"])):
        location: KeyLocation = ("views", index)
        path = paths[view.name]
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
    _share_page_scale(interpretation, data, sheets, reports)
    return sheets, reports


def _share_page_scale(
    interpretation: DrawingInterpretation,
    data: dict[str, Any],
    sheets: dict[str, _Sheet],
    reports: dict[str, dict[str, Any]],
) -> None:
    """Give a page's orthographic crops without a scale the one the others agree on.

    Detail and section views may be enlarged, so they neither give nor take one.
    """
    outputs = {output["name"]: output for output in data["views"]}
    for page in interpretation.views:
        if page.region.view != page.name or sheets[page.name].dxf:
            continue
        family = [page] + [
            view
            for view in interpretation.views
            if view.region.view == page.name
            and view.name != page.name
            and view.role in ORTHOGRAPHIC_VIEWS
        ]
        known = {
            view.name: sheets[view.name].scale
            for view in family
            if sheets[view.name].scale is not None
        }
        if not known or max(known.values()) / min(known.values()) - 1 > SCALE_AGREEMENT:
            continue
        scale = median(known.values())
        for view in family:
            report = reports[view.name]
            # One printed length cannot fit a scale, but a disagreeing one
            # may mark an enlarged view.
            single = report.get("scale")
            if (
                view.name in known
                or report["status"] != "insufficient_evidence"
                or (single is not None and abs(single / scale - 1) > SCALE_AGREEMENT)
            ):
                continue
            sheets[view.name] = replace(sheets[view.name], scale=scale)
            outputs[view.name]["scale"] = scale
            reports[view.name] = {
                **report,
                "scale": scale,
                "scale_source": "shared with " + ", ".join(known),
            }


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
            # Native-size crops share their parent's pixel units. Use the
            # containing view's area even when the annotation cites that parent.
            sheet = sheets[output["name"]]
            box = dim["region"]["box_px"]
            if not sheet.dxf and box is not None:
                x0, y0, x1, y1 = box
                if (x1 - x0) * (y1 - y0) >= (
                    MAX_DIMENSION_REGION_FRACTION * math.prod(sheet.size)
                ):
                    raise LocatedError.at(
                        ("views", index, "dimensions", position, "region", "box_px"),
                        f"{dim['name']}.region must cover less than "
                        f"{MAX_DIMENSION_REGION_FRACTION:.0%} of {output['name']}'s "
                        "image area. Tighten the box around this dimension's "
                        "text, symbols and dimension/extension/leader lines; "
                        "do not expand it to enclose the dimensioned feature.",
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
    projection_method: ProjectionMethod,
) -> tuple[DrawingInterpretation, dict[str, dict[str, Any]]]:
    """Validate and enrich without mutating model output or the input files.

    Raster submissions normally leave image_size, scale and box_uv null.
    Scale and box_uv are recalculated on every call, since a measurement fix
    changes them. A stored image_size must still match its file, which catches
    a resized sheet. Derived rasters must also match their declared parent crop
    pixel for pixel, including on the first submission. An orthographic crop
    without its own consensus takes the scale its page and sibling crops agree on.
    No consensus (including one measurement) leaves scale/UV null; return
    diagnostics to the interpreter instead of guessing a scale.

    A DXF keeps its own millimetre coordinates, never preview pixels, so its
    measured_length is already a length and is retained rather than passed to
    RANSAC with pixel tolerances.
    """
    interpretation = DrawingInterpretation.model_validate(interpretation.model_dump())
    _require_a_readable_submission(interpretation, projection_method)
    paths = {
        view.name: _readable_path(view, ("views", index), workdir)
        for index, view in enumerate(interpretation.views)
    }
    _require_distinct_view_files(interpretation.views, paths)
    data = interpretation.model_dump()
    sheets, reports = _calibrate_sheets(interpretation, data, paths)
    _calibrate_regions(interpretation, data, sheets)
    _require_raster_crops(interpretation.views, paths)
    return DrawingInterpretation.model_validate(data), reports
