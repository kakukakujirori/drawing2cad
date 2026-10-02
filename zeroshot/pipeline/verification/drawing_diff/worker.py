"""Child process for drawing comparison: align pairs and save diff images."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

from PIL import Image

from ..run_drawing_diff import DrawingDiffReport
from .align import Backend, Model, align, read_rgb
from .diff import compute_diff


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

    options = dict(alignment_options or {})
    scale_tolerance = options.pop("scale_tolerance", 0.01)
    if not 0 < scale_tolerance < 1:
        raise ValueError("scale_tolerance must be in (0, 1)")
    calibration: dict[str, Any] = {
        "status": "unavailable",
        "reason": "input scale unavailable",
    }
    if drawing_mm_per_pixel is not None:
        if not math.isfinite(drawing_mm_per_pixel) or drawing_mm_per_pixel <= 0:
            raise ValueError("drawing_mm_per_pixel must be positive and finite")
        dxf_path = projection_path.with_suffix(".dxf")
        if backend != "directional_chamfer" or model != "similarity":
            calibration["reason"] = (
                "scale constraint requires directional_chamfer similarity"
            )
        elif not dxf_path.is_file():
            calibration["reason"] = "projection DXF unavailable"
        else:
            import ezdxf

            from ..render.export_dxf import DEFAULT_PNG_MARGIN_RATIO, png_bounds

            x0, y0, x1, y1 = png_bounds(
                ezdxf.readfile(dxf_path).modelspace(),
                margin_ratio=DEFAULT_PNG_MARGIN_RATIO,
            )
            h, w = projection.shape[:2]
            source_scale = ((x1 - x0) / w + (y1 - y0) / h) / 2
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
