"""Techdraw orchestrator: STEP -> 3-view SVG + DXF + PDF (ECCV/SolidWorks look).

Public entry point::

    generate_techdraw(step_path, paths, cfg=None) -> dict

Pure and synchronous: no multiprocessing, no timeouts, no import-time work.
Batch isolation / timeouts are handled by render_dataset.py.

Pipeline: load STEP -> OCC HLR project three third-angle views -> typed 2D
primitives -> scale + centre layout (sheet-mm) -> center marks -> write DXF
(sheet-mm) + SVG (points, GT transform) + PDF (cairosvg of the SVG).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from OCC.Core.STEPControl import STEPControl_Reader

from src.render.centermarks import all_marks
from src.render.config import SHEET_H_MM, SHEET_W_MM, TechdrawPaths
from src.render.hlr import project
from src.render.layout import Layout, build_layout
from src.render.writers.dxf_writer import write_dxf
from src.render.writers.pdf_writer import write_pdf
from src.render.writers.svg_writer import write_svg


# Third-angle view frames (view_dir, up_dir), verified by raster-IoU vs GT.
FRONT = ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0))  # look -Z, up +Y  -> screen (X, Y)
TOP = ((0.0, 1.0, 0.0), (0.0, 0.0, -1.0))  # look -Y, up -Z  -> screen (X, Z)
RIGHT = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))  # look -X, up +Y  -> screen (-Z, Y)


@dataclass
class TechdrawConfig:
    deflection: float = 0.02  # b-spline discretisation (model units)
    include_smooth: bool = False  # tangent/smooth (Rg1) edges; GT suppresses them
    center_marks: bool = True


# A placed view narrower than this (sheet-mm, either axis) carries no recoverable
# shape and violates the manifest's positive-area bbox contract. It only ever
# fires on pathological inputs: an empty HLR projection (0 edges, e.g. an OCC
# axis-degeneracy the tilt fallback could not recover) or a knife-edge view of a
# near-zero-thickness plate that collapses to a single line.
MIN_VIEW_EXTENT_MM = 0.05
# Every view is laid out to fit the A4 sheet; a bbox this far outside it means the
# projection ran away (observed on corrupt STEPs whose model AABB is astronomical,
# so OCC HLR emits outline curves at 1e5+ mm). Reject rather than record garbage.
SHEET_MARGIN_MM = 10.0


class DegenerateDrawingError(RuntimeError):
    """A view could not be rendered into a valid positive-area, on-sheet bbox."""


def _validate_layout(layout: Layout) -> None:
    """Fail the part if any placed view bbox is degenerate or off-sheet.

    ``generate_techdraw`` runs one part per isolated process; raising here makes
    ``render_dataset`` record ``ok=False`` so the sample is dropped from the
    manifest instead of poisoning it with an invalid bbox that the data loader's
    metadata validator would later reject at train time.
    """
    for v in layout.views:
        x_min, y_min, x_max, y_max = v.bbox
        values = (x_min, y_min, x_max, y_max)
        if not all(math.isfinite(c) for c in values):
            raise DegenerateDrawingError(
                f"view {v.name!r} bbox is not finite: {values}"
            )
        width = x_max - x_min
        height = y_max - y_min
        if width < MIN_VIEW_EXTENT_MM or height < MIN_VIEW_EXTENT_MM:
            raise DegenerateDrawingError(
                f"view {v.name!r} has near-zero extent "
                f"(w={width:.4f}, h={height:.4f} mm): {values}"
            )
        if (
            x_min < -SHEET_MARGIN_MM
            or y_min < -SHEET_MARGIN_MM
            or x_max > SHEET_W_MM + SHEET_MARGIN_MM
            or y_max > SHEET_H_MM + SHEET_MARGIN_MM
        ):
            raise DegenerateDrawingError(
                f"view {v.name!r} bbox runs off the sheet "
                f"(sheet {SHEET_W_MM}x{SHEET_H_MM} mm): {values}"
            )


def _load_shape(step_path: Path):
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != 1:  # IFSelect_RetDone
        raise RuntimeError(f"STEP read failed ({status}): {step_path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape is None or shape.IsNull():
        raise RuntimeError(f"empty shape: {step_path}")
    return shape


def _diag(shape) -> float:
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib

    b = Bnd_Box()
    brepbndlib.Add(shape, b)
    xmin, ymin, zmin, xmax, ymax, zmax = b.Get()
    dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
    return max((dx * dx + dy * dy + dz * dz) ** 0.5, 1e-6)


def _project_nonempty(
    shape, view_dir, up_dir, *, deflection, include_smooth, merge_tol
):
    """Project one view, guarding against the OCC HLR exact-axis degeneracy.

    ``HLRBRep_Algo`` occasionally returns *zero* edges when the view direction is
    exactly axis-aligned and the shape has faces exactly perpendicular / parallel
    to it (observed on a thin disc for the +X / right view: 0 edges).  A tiny tilt
    of the frame breaks the degeneracy and recovers the full projection; the sub-
    milliradian rotation is far below the drawing's calibration tolerance and is
    only ever applied to views that would otherwise be empty, so non-degenerate
    views are bit-for-bit unchanged.  This keeps the renderer at 3 views always.
    """
    vp = project(
        shape,
        view_dir,
        up_dir,
        deflection=deflection,
        include_smooth=include_smooth,
        merge_tol=merge_tol,
    )
    if vp.visible.count() + vp.hidden.count() > 0:
        return vp
    for eps in (1e-3, 3e-3, 1e-2):
        vd = (view_dir[0] + eps, view_dir[1] + eps, view_dir[2] + eps)
        vp = project(
            shape,
            vd,
            up_dir,
            deflection=deflection,
            include_smooth=include_smooth,
            merge_tol=merge_tol,
        )
        if vp.visible.count() + vp.hidden.count() > 0:
            return vp
    return vp


def generate_techdraw(
    step_path: Path, paths: TechdrawPaths, cfg: TechdrawConfig | None = None
) -> dict:
    step_path = Path(step_path)
    cfg = cfg or TechdrawConfig()

    shape = _load_shape(step_path)

    # scale deflection / merge tolerance to the model size (unit-agnostic)
    diag = _diag(shape)
    defl = cfg.deflection * diag
    mtol = 5e-4 * diag

    front = _project_nonempty(
        shape,
        *FRONT,
        deflection=defl,
        include_smooth=cfg.include_smooth,
        merge_tol=mtol,
    )
    top = _project_nonempty(
        shape, *TOP, deflection=defl, include_smooth=cfg.include_smooth, merge_tol=mtol
    )
    right = _project_nonempty(
        shape,
        *RIGHT,
        deflection=defl,
        include_smooth=cfg.include_smooth,
        merge_tol=mtol,
    )

    layout = build_layout(front, top, right)
    # Reject degenerate / runaway projections before writing any output so the
    # part is recorded ok=False and dropped, never emitting an invalid bbox.
    _validate_layout(layout)
    marks = all_marks(layout.views) if cfg.center_marks else []

    for p in (paths.svg, paths.dxf, paths.pdf):
        Path(p).parent.mkdir(parents=True, exist_ok=True)

    svg_text = write_svg(Path(paths.svg), layout.views, marks)
    write_dxf(Path(paths.dxf), layout.views, marks)
    write_pdf(Path(paths.pdf), svg_text)

    def _counts(v):
        return {
            "visible": v.visible.count(),
            "hidden": v.hidden.count(),
        }

    info = {
        "scale": layout.scale,
        "bbox_format": "xyxy",
        "bbox_coordinate_system": {
            "unit": "mm",
            "origin": "sheet_bottom_left",
            "x_axis": "right",
            "y_axis": "up",
        },
        "cluster_bbox": [round(x, 3) for x in layout.cluster_bbox],
        "views": {
            v.name: {"bbox": [round(x, 3) for x in v.bbox], **_counts(v)}
            for v in layout.views
        },
        "n_center_marks": len(marks),
    }
    return info


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import sys

    import rootutils

    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    from src.render.config import techdraw_paths

    step = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/td_out")
    tp = techdraw_paths(out, step.stem)
    print(generate_techdraw(step, tp))
