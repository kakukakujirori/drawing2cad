"""Full-resolution drawing residuals in the output projection's pixel frame."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .align import AlignmentResult, _validate_rgb, opencv_transform
from .image_ops import distance_map, foreground_mask


@dataclass(frozen=True)
class DiffResult:
    overlay: np.ndarray | None
    residual: np.ndarray | None
    stats: dict[str, Any]
    warnings: tuple[str, ...]


def _distance_summary(distances: np.ndarray) -> dict[str, Any]:
    """Keep raw distance diagnostics without making a CAD correctness decision."""
    return {
        "count": distances.size,
        "median_px": float(np.median(distances)) if distances.size else None,
        "p95_px": float(np.percentile(distances, 95)) if distances.size else None,
        "max_px": float(distances.max()) if distances.size else None,
    }


def _measure_distances(
    aligned_ink: np.ndarray, output_ink: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Measure reciprocal line distances and distinguish unknown input coverage."""
    observed = output_ink & valid
    outside_count = int((output_ink & ~valid).sum())
    output_count = int(output_ink.sum())
    # Only output lines within the warped input canvas have a measured residual.
    output_distances = distance_map(aligned_ink)[observed]
    input_distances = distance_map(output_ink)[aligned_ink]
    stats = {
        "comparison_status": "ok" if output_distances.size else "uncertain",
        "valid_canvas_fraction": float(valid.mean()),
        "output_ink_count": output_count,
        "outside_count": outside_count,
        "outside_fraction": float(outside_count / output_count),
        "output_to_input": _distance_summary(output_distances),
        "input_to_output": _distance_summary(input_distances),
    }
    return output_distances, input_distances, stats


def _normalized(distances: np.ndarray, clip: float | None) -> np.ndarray:
    """Handle perfect agreement (zero range) and an empty observation explicitly."""
    return np.clip(distances / clip, 0, 1) if clip else np.zeros_like(distances)


def _colors(distances: np.ndarray, clip: float | None) -> np.ndarray:
    """Map distances through OpenCV's 256-color JET table, returning RGB pixels."""
    if not distances.size:
        return np.empty((0, 3), dtype=np.uint8)
    indices = np.rint(_normalized(distances, clip) * 255).astype(np.uint8)
    # OpenCV returns BGR; image arrays elsewhere in this module use RGB.
    return cv2.applyColorMap(indices[:, None], cv2.COLORMAP_JET)[:, 0, ::-1]


def _warp_drawing(
    drawing_gray: np.ndarray, matrix: np.ndarray, output_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Warp the line pixels, observed canvas, and gray background together."""
    height, width = output_shape
    # Nearest-neighbor warping preserves binary line membership and observation.
    masks = (foreground_mask(drawing_gray), np.ones(drawing_gray.shape, dtype=bool))
    aligned_ink, valid = (
        cv2.warpPerspective(
            mask.astype(np.uint8),
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        ).astype(bool)
        for mask in masks
    )
    if not aligned_ink.any():
        raise ValueError("no aligned input ink falls inside the output canvas")
    # Only the display background is interpolated, leaving distance masks binary.
    aligned_gray = cv2.warpPerspective(
        drawing_gray,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderValue=255,
    )
    return aligned_ink, valid, aligned_gray


def _render_overlay(
    aligned_gray: np.ndarray,
    observed: np.ndarray,
    outside: np.ndarray,
    distances: np.ndarray,
    clip: float | None,
) -> np.ndarray:
    """Draw measured output lines in color over the pale aligned input drawing."""
    gray = np.rint(225 + aligned_gray.astype(float) * (30 / 255)).astype(np.uint8)
    overlay = np.repeat(gray[..., None], 3, axis=2)
    overlay[observed] = _colors(distances, clip)
    overlay[outside] = 128  # Unknown input coverage is neutral, not a large error.
    return overlay


def _render_residual(
    aligned_ink: np.ndarray, distances: np.ndarray, clip: float | None
) -> np.ndarray:
    """Darken input lines absent from the output; agreement remains pale gray."""
    residual = np.full(aligned_ink.shape, 255, dtype=np.uint8)
    residual[aligned_ink] = np.rint(225 - 225 * _normalized(distances, clip)).astype(
        np.uint8
    )
    return residual


def compute_diff(
    drawing_rgb: np.ndarray,
    projection_rgb: np.ndarray,
    alignment: AlignmentResult,
    *,
    distance_clip_px: float | None = 12.0,
) -> DiffResult:
    """Color every observed output ink pixel by distance to aligned input ink.

    JET maps zero distance to dark blue and ``distance_clip_px`` or above to
    dark red. None uses each direction's observed maximum independently;
    perfect agreement stays dark blue or pale gray.
    Unsupported output pixels are medium gray, excluded from distance statistics.
    The reverse residual is grayscale: larger missing-input distances are darker.
    Distances are raw output pixels, independent of optimizer sampling/loss.

    Invalid arguments raise ValueError. Unusable image content or a failed
    alignment returns the reason in warnings.
    """
    # Invalid input arrays and display scales are caller errors.
    _validate_rgb(drawing_rgb, "drawing_rgb")
    _validate_rgb(projection_rgb, "projection_rgb")
    if distance_clip_px is not None and (
        not math.isfinite(distance_clip_px) or distance_clip_px <= 0
    ):
        raise ValueError("distance_clip_px must be finite and positive, or None")
    # Record the display range separately from raw distances and alignment status.
    stats: dict[str, Any] = {
        "comparison_status": "failed",
        "provisional": alignment.status != "ok",
        "distance_unit": "projection_px",
        "color_normalization": "image_max" if distance_clip_px is None else "fixed",
        "distance_clip_px": distance_clip_px,
        "red_distance_px": distance_clip_px,
        "black_distance_px": distance_clip_px,
        "unobserved_color_rgb": [128, 128, 128],
    }
    warnings = list(alignment.warnings)
    try:
        if alignment.status == "failed" or alignment.H_drawing_to_projection is None:
            raise ValueError("no usable drawing-to-projection alignment")
        # Map input line membership and input coverage into the output pixel frame.
        drawing_gray = cv2.cvtColor(drawing_rgb, cv2.COLOR_RGB2GRAY)
        projection_gray = cv2.cvtColor(projection_rgb, cv2.COLOR_RGB2GRAY)
        output_ink = foreground_mask(projection_gray)
        matrix = opencv_transform(alignment.H_drawing_to_projection)
        aligned_ink, valid, aligned_gray = _warp_drawing(
            drawing_gray, matrix, projection_gray.shape
        )
        # Measure both directions; output pixels outside input coverage are unknown.
        output_distances, input_distances, measured = _measure_distances(
            aligned_ink, output_ink, valid
        )
        stats.update(measured)
        # Automatic display scales are per direction; unavailable maxima stay null.
        if distance_clip_px is None:
            stats["red_distance_px"] = stats["output_to_input"]["max_px"]
            stats["black_distance_px"] = stats["input_to_output"]["max_px"]
        overlay = _render_overlay(
            aligned_gray,
            output_ink & valid,
            output_ink & ~valid,
            output_distances,
            stats["red_distance_px"],
        )
        residual = _render_residual(
            aligned_ink, input_distances, stats["black_distance_px"]
        )
        # Warn about missing observation coverage, not a global residual threshold.
        if stats["outside_count"]:
            warnings.append(
                f"{stats['outside_fraction']:.1%} of output ink is outside the observed input "
                "region; shown in neutral gray and excluded from output distance statistics."
            )
        if not output_distances.size:
            warnings.append("No output ink is inside the observed input region.")
        return DiffResult(overlay, residual, stats, tuple(warnings))
    except (ValueError, cv2.error) as exc:
        # An unusable alignment or line drawing produces no residual images.
        warnings.append(f"Drawing residuals unavailable: {exc}")
        return DiffResult(None, None, stats, tuple(warnings))
