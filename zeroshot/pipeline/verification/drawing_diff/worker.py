"""Child process for drawing comparison: align pairs and save diff images."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..run_drawing_diff import DrawingDiffReport
from .align import AlignmentResult, Backend, Model, align, read_rgb
from .diff import DiffResult, compute_diff


def run_worker(
    pairs: Sequence[tuple[Path, Path]],
    connection: Connection,
    *,
    backend: Backend = "directional_chamfer",
    model: Model = "similarity",
    alignment_options: Mapping[str, Any] | None = None,
    distance_clip_px: float | None = 12.0,
    drawing_scales: Sequence[float | None] | None = None,
) -> None:
    """Initialize a runtime once and process the supplied image pairs in order."""
    try:
        runtime = None
        if backend == "match_anything":
            from .match_anything import load_runtime

            runtime = load_runtime()

        for idx, (drawing_path, projection_path) in enumerate(pairs):
            try:
                report = run_align_diff_save(
                    drawing_path=drawing_path,
                    projection_path=projection_path,
                    backend=backend,
                    model=model,
                    alignment_options=alignment_options,
                    distance_clip_px=distance_clip_px,
                    drawing_mm_per_pixel=drawing_scales[idx]
                    if drawing_scales is not None
                    else None,
                    runtime=runtime,
                )
            except Exception as error:  # noqa: BLE001
                report = DrawingDiffReport(
                    drawing_path=drawing_path,
                    projection_path=projection_path,
                    error=f"{type(error).__name__}: {error}",
                )

            connection.send((idx, report))

    finally:
        connection.close()


def run_align_diff_save(
    drawing_path: Path,
    projection_path: Path,
    *,
    backend: Backend = "directional_chamfer",
    model: Model = "similarity",
    alignment_options: Mapping[str, Any] | None = None,
    distance_clip_px: float | None = 12.0,
    drawing_mm_per_pixel: float | None = None,
    runtime: Any = None,
) -> DrawingDiffReport:
    """Align one pair, save available diff images, and return their paths."""

    drawing = read_rgb(drawing_path)
    projection = read_rgb(projection_path)
    alignment, diff = _align_and_compare(
        drawing,
        projection,
        projection_path,
        backend=backend,
        model=model,
        alignment_options=alignment_options,
        distance_clip_px=distance_clip_px,
        drawing_mm_per_pixel=drawing_mm_per_pixel,
        runtime=runtime,
    )

    paths = {}
    if diff.overlay is not None:
        path = projection_path.with_name(f"{projection_path.stem}_overlay.png")
        Image.fromarray(diff.overlay).save(path)
        paths["overlay_path"] = path
    if diff.unmatched is not None:
        path = projection_path.with_name(f"{projection_path.stem}_unmatched.png")
        Image.fromarray(diff.unmatched).save(path)
        paths["unmatched_path"] = path

    return DrawingDiffReport(
        drawing_path=drawing_path,
        projection_path=projection_path,
        alignment=alignment,
        stats=diff.stats,
        warnings=diff.warnings,
        paths=paths,
    )


def _align_and_compare(
    drawing: np.ndarray,
    projection: np.ndarray,
    projection_path: Path,
    *,
    backend: Backend,
    model: Model,
    alignment_options: Mapping[str, Any] | None,
    distance_clip_px: float | None,
    drawing_mm_per_pixel: float | None,
    runtime: Any,
) -> tuple[AlignmentResult, DiffResult]:
    """Calibrate and compare images, then map their error boxes to CAD UV."""

    options = dict(alignment_options or {})
    scale_tolerance = options.pop("scale_tolerance", 0.01)
    if not 0 < scale_tolerance < 1:
        raise ValueError("scale_tolerance must be in (0, 1)")
    projection_to_uv = None
    modelspace = None
    uv_mapping = {"status": "unavailable", "reason": "projection DXF unavailable"}
    dxf_path = projection_path.with_suffix(".dxf")
    if dxf_path.is_file():
        import ezdxf

        from ..render.export_dxf import DEFAULT_PNG_MARGIN_RATIO, png_bounds

        try:
            modelspace = ezdxf.readfile(dxf_path).modelspace()
            x0, y0, x1, y1 = png_bounds(
                modelspace,
                margin_ratio=DEFAULT_PNG_MARGIN_RATIO,
            )
            if not all(map(math.isfinite, (x0, y0, x1, y1))) or x1 <= x0 or y1 <= y0:
                raise ValueError("invalid projection bounds")
            h, w = projection.shape[:2]
            # PNG boundaries, including renderer margins; +V is up, pixel y is down.
            projection_to_uv = np.array(
                [[(x1 - x0) / w, 0, x0], [0, -(y1 - y0) / h, y1], [0, 0, 1.0]]
            )
            uv_mapping["reason"] = "drawing-to-projection alignment unavailable"
        except (OSError, ValueError, ezdxf.DXFError) as error:
            modelspace = None
            uv_mapping["reason"] = (
                f"projection DXF unreadable or invalid ({type(error).__name__})"
            )

    calibration: dict[str, Any] = {
        "status": "unavailable",
        "reason": "input scale unavailable",
    }
    if drawing_mm_per_pixel is not None:
        if not math.isfinite(drawing_mm_per_pixel) or drawing_mm_per_pixel <= 0:
            raise ValueError("drawing_mm_per_pixel must be positive and finite")
        if backend != "directional_chamfer" or model != "similarity":
            calibration["reason"] = (
                "scale constraint requires directional_chamfer similarity"
            )
        elif projection_to_uv is None:
            calibration["reason"] = uv_mapping["reason"]
        else:
            source_scale = float((projection_to_uv[0, 0] - projection_to_uv[1, 1]) / 2)
            pixel_scale = source_scale / drawing_mm_per_pixel
            relative_scale = (
                pixel_scale * max(projection.shape[:2]) / max(drawing.shape[:2])
            )
            options["relative_scale"] = (
                relative_scale * (1 - scale_tolerance),
                relative_scale * (1 + scale_tolerance),
            )
            calibration = {
                "status": "applied",
                "expected_pixel_scale": pixel_scale,
            }

    alignment = align(
        drawing,
        projection,
        backend=backend,
        model=model,
        options=options,
        runtime=runtime,
    )
    alignment.diagnostics["scale_calibration"] = calibration
    diff = compute_diff(
        drawing,
        projection,
        alignment,
        distance_clip_px=distance_clip_px,
        projection_modelspace=modelspace,
        projection_to_uv=projection_to_uv,
    )

    if projection_to_uv is not None:
        diff.stats["projection_to_model_uv"] = projection_to_uv.tolist()
        if alignment.H_drawing_to_projection is not None:
            # Public H and cluster boxes both use pixel boundaries, not OpenCV centres.
            transform = projection_to_uv @ np.array(alignment.H_drawing_to_projection)
            diff.stats["drawing_to_model_uv"] = transform.tolist()
            uv_mapping = {
                "status": "ok"
                if alignment.status == "ok" and calibration["status"] == "applied"
                else "provisional"
            }
            for item in diff.stats.get("unmatched", []):
                x0, y0, x1, y1 = item["box_px"]
                corners = np.array([[x0, y0, 1], [x1, y0, 1], [x0, y1, 1], [x1, y1, 1]])
                mapped = corners @ transform.T
                uv = mapped[:, :2] / mapped[:, 2:]
                item["box_model_uv"] = [
                    *uv.min(axis=0).tolist(),
                    *uv.max(axis=0).tolist(),
                ]
    diff.stats["model_uv_mapping"] = uv_mapping
    return alignment, diff
