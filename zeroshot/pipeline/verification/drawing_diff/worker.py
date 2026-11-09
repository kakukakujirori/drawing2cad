"""Child process for drawing comparison: align pairs and save diff images."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..run_drawing_diff import DrawingDiffReport
from .align import AlignmentResult, Backend, Model, align, read_rgb
from .diff import DiffResult, compute_diff
from .ocr import read_text_lines
from .tangent_display import projection_without_tangents, sheet_omits_tangents
from .text_marks import text_regions


@dataclass(frozen=True)
class AlignedView:
    """A projection registered to its drawing, with all a comparison needs."""

    drawing: np.ndarray
    non_tangent_rgb: np.ndarray
    tangent_mask: np.ndarray | None
    alignment: AlignmentResult
    modelspace: Any
    projection_to_uv: np.ndarray | None
    calibration: dict[str, Any]
    uv_mapping: dict[str, Any]


def run_worker(
    pairs: Sequence[tuple[Path, Path]],
    connection: Connection,
    *,
    backend: Backend = "directional_chamfer",
    model: Model = "similarity",
    alignment_options: Mapping[str, Any] | None = None,
    distance_clip_px: float | None = 12.0,
    drawing_scales: Sequence[float | None] | None = None,
    auto_hide_tangents: bool = False,
) -> None:
    """Initialize a runtime once and process the supplied image pairs in order."""
    try:
        runtime = None
        if backend == "match_anything":
            from .match_anything import load_runtime

            runtime = load_runtime()

        scales = drawing_scales if drawing_scales is not None else [None] * len(pairs)
        # Lazy, so each pair is finished and sent before the next one starts.
        views = (
            _align_pair(
                pair,
                backend=backend,
                model=model,
                alignment_options=alignment_options,
                drawing_mm_per_pixel=scale,
                runtime=runtime,
            )
            for pair, scale in zip(pairs, scales, strict=True)
        )

        hide_tangents = False
        if auto_hide_tangents:
            # The sheet's decision needs all of its views before any is compared.
            views = list(views)
            hide_tangents = sheet_omits_tangents(
                view for view in views if isinstance(view, AlignedView)
            )

        for idx, (pair, view) in enumerate(zip(pairs, views, strict=True)):
            report = _compare_pair(
                pair,
                view,
                distance_clip_px=distance_clip_px,
                hide_tangents=hide_tangents,
            )
            connection.send((idx, report))

    finally:
        connection.close()


def _failed(pair: tuple[Path, Path], error: Exception) -> DrawingDiffReport:
    return DrawingDiffReport(
        drawing_path=pair[0],
        projection_path=pair[1],
        error=f"{type(error).__name__}: {error}",
    )


def _align_pair(
    pair: tuple[Path, Path],
    *,
    backend: Backend,
    model: Model,
    alignment_options: Mapping[str, Any] | None,
    drawing_mm_per_pixel: float | None,
    runtime: Any,
) -> AlignedView | DrawingDiffReport:
    """Register one pair, or report why it could not be."""
    try:
        return _align_view(
            read_rgb(pair[0]),
            read_rgb(pair[1]),
            pair[1],
            backend=backend,
            model=model,
            alignment_options=alignment_options,
            drawing_mm_per_pixel=drawing_mm_per_pixel,
            runtime=runtime,
        )
    except Exception as error:  # noqa: BLE001
        return _failed(pair, error)


def _compare_pair(
    pair: tuple[Path, Path],
    view: AlignedView | DrawingDiffReport,
    *,
    distance_clip_px: float | None,
    hide_tangents: bool,
) -> DrawingDiffReport:
    """Compare a registered pair and save its images, or report why it could not be."""
    if isinstance(view, DrawingDiffReport):
        return view
    try:
        if hide_tangents and view.tangent_mask is not None:
            Image.fromarray(projection_without_tangents(view.modelspace)).save(pair[1])
        diff = _compare_view(
            view, distance_clip_px=distance_clip_px, hide_tangents=hide_tangents
        )
        return _save_diff(*pair, view.alignment, diff)
    except Exception as error:  # noqa: BLE001
        return _failed(pair, error)


def _save_diff(
    drawing_path: Path,
    projection_path: Path,
    alignment: AlignmentResult,
    diff: DiffResult,
) -> DrawingDiffReport:
    """Save the available diff images next to the projection and report their paths."""
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


def _align_view(
    drawing: np.ndarray,
    projection: np.ndarray,
    projection_path: Path,
    *,
    backend: Backend,
    model: Model,
    alignment_options: Mapping[str, Any] | None,
    drawing_mm_per_pixel: float | None,
    runtime: Any,
) -> AlignedView:
    """Read the projection's DXF and register the drawing to its required lines."""

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

    non_tangent_rgb, tangent_mask = projection, None
    if modelspace is not None and any(
        entity.dxf.layer == "TANGENT" for entity in modelspace
    ):
        from ezdxf.addons.drawing.config import ColorPolicy

        from ..render.export_dxf import create_drawing_figure

        non_tangent_figure = create_drawing_figure(
            modelspace,
            filter_func=lambda entity: entity.dxf.layer != "TANGENT",
            color_policy=ColorPolicy.BLACK,
        )
        non_tangent_figure.canvas.draw()
        non_tangent_rgb = np.asarray(non_tangent_figure.canvas.buffer_rgba())[
            ..., :3
        ].copy()
        if non_tangent_rgb.shape != projection.shape:
            raise ValueError("projection PNG does not match the DXF raster grid")
        tangent_figure = create_drawing_figure(
            modelspace,
            filter_func=lambda entity: entity.dxf.layer == "TANGENT",
            color_policy=ColorPolicy.BLACK,
        )
        tangent_figure.canvas.draw()
        # Keep the antialiased stroke too; its display gray never sets the mask.
        tangent_mask = np.asarray(tangent_figure.canvas.buffer_rgba())[..., 0] < 255
    alignment = align(
        drawing,
        non_tangent_rgb,
        backend=backend,
        model=model,
        options=options,
        runtime=runtime,
    )
    alignment.diagnostics["scale_calibration"] = calibration
    return AlignedView(
        drawing,
        non_tangent_rgb,
        tangent_mask,
        alignment,
        modelspace,
        projection_to_uv,
        calibration,
        uv_mapping,
    )


def _compare_view(
    view: AlignedView,
    *,
    distance_clip_px: float | None,
    hide_tangents: bool = False,
) -> DiffResult:
    """Compare a registered view, then map its error boxes to CAD UV."""
    alignment, calibration = view.alignment, view.calibration
    modelspace, projection_to_uv = view.modelspace, view.projection_to_uv
    uv_mapping = view.uv_mapping
    diff = compute_diff(
        view.drawing,
        view.non_tangent_rgb,
        alignment,
        tangent_ink=view.tangent_mask,
        text_regions=text_regions(
            read_text_lines(view.drawing), view.drawing.shape[:2]
        ),
        distance_clip_px=distance_clip_px,
        projection_modelspace=modelspace,
        projection_to_uv=projection_to_uv,
        hide_tangents=hide_tangents,
    )
    if modelspace is None:
        diff = replace(
            diff,
            warnings=diff.warnings
            + (
                "Projection DXF unavailable; tangent boundaries cannot be distinguished from required lines.",
            ),
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
    return diff
