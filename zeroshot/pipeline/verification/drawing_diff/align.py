"""Align a drawing to a CAD projection without writing files or loading models.

Inputs are original-resolution uint8 RGB arrays. Returned transforms map drawing
to projection in pixel-boundary coordinates: pixel (x, y) has its centre at
(x + .5, y + .5). Use ``opencv_transform`` before passing a matrix to OpenCV.

``ok`` means a supported transform was found, not that the CAD is correct.
Backend diagnostics retain their explicitly named units; they are not the
full-image residuals that a later comparison/visualisation computes.

Each backend provides validate_options(model, options) and
estimate_transform(..., model, options, runtime, diagnostics). The latter returns
(integer-centre drawing→projection matrix, status, warnings), updating diagnostics
in place so a failure can retain the evidence collected before it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image

type Backend = Literal["directional_chamfer", "match_anything"]
type Model = Literal["similarity", "affine", "homography"]

_MODELS = ("similarity", "affine", "homography")
_PIXEL_OFFSET = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
_INDEX_OFFSET = np.array([[1.0, 0.0, -0.5], [0.0, 1.0, -0.5], [0.0, 0.0, 1.0]])


@dataclass(frozen=True)
class AlignmentResult:
    backend: Backend
    model: Model
    status: Literal["ok", "uncertain", "failed"]
    H_drawing_to_projection: list[list[float]] | None
    diagnostics: dict[str, Any]
    warnings: tuple[str, ...] = ()


def read_rgb(path: str | Path) -> np.ndarray:
    """Composite transparency on white, preserving the file's pixel frame."""
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, "white")
        return np.array(Image.alpha_composite(white, rgba).convert("RGB"))


def pixel_boundary_transform(matrix: np.ndarray) -> np.ndarray:
    """Convert an integer-centre transform to the Region pixel-boundary frame."""
    return _PIXEL_OFFSET @ np.asarray(matrix, dtype=float) @ _INDEX_OFFSET


def opencv_transform(matrix: np.ndarray | list[list[float]]) -> np.ndarray:
    """Convert a public transform back to OpenCV's integer-centre frame."""
    return _INDEX_OFFSET @ np.asarray(matrix, dtype=float) @ _PIXEL_OFFSET


def _validate_rgb(image: np.ndarray, name: str) -> None:
    """Enforce the shared original-resolution RGB array contract."""
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or min(image.shape[:2]) < 2
    ):
        raise ValueError(f"{name} must be a nonempty uint8 RGB image (H, W, 3)")


def _checked_transform(matrix: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Reject singular transforms and projective poles across the input image."""
    # Normalize homogeneous scale and require a finite inverse.
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("alignment did not produce a finite 3x3 transform")
    if matrix[2, 2] == 0:
        raise ValueError("alignment has a projective pole at the image origin")
    matrix = matrix / matrix[2, 2]
    if not np.isfinite(np.linalg.inv(matrix)).all():
        raise ValueError("alignment transform is not invertible")
    # Inspect the whole input rectangle, not only the points used by the backend.
    height, width = shape[:2]
    corners = np.array([[0, 0, 1], [width, 0, 1], [0, height, 1], [width, height, 1]])
    mapped = corners @ matrix.T
    # The denominator is linear, so checking corner signs covers the rectangle.
    if np.any(mapped[:, 2] <= 1e-10):
        raise ValueError("alignment has a projective pole in the input image")
    if not np.isfinite(mapped[:, :2] / mapped[:, 2:]).all():
        raise ValueError("alignment maps image corners to non-finite coordinates")
    return matrix


def align(
    drawing_rgb: np.ndarray,
    projection_rgb: np.ndarray,
    *,
    backend: Backend = "directional_chamfer",
    model: Model = "similarity",
    options: Mapping[str, Any] | None = None,
    runtime: Any = None,
) -> AlignmentResult:
    """Estimate an input-drawing → output-projection transform.

    ``options`` overrides the selected backend's settings: Chamfer Config fields
    (except model), or ransac_reproj_threshold/ransac_confidence/ransac_max_iter.
    MatchAnything requires an explicitly prepared ``load_runtime(...)`` result;
    this function never downloads resources or constructs a learned model.

    Bad arguments raise ValueError/TypeError. Blank images, insufficient matches,
    runtime/inference and numerical failures return status=failed. Both backends
    require the image/scientific dependencies installed by this project.
    """
    if backend not in ("directional_chamfer", "match_anything"):
        raise ValueError(f"unknown alignment backend: {backend}")
    if model not in _MODELS:
        raise ValueError(f"unknown alignment model: {model}")
    _validate_rgb(drawing_rgb, "drawing_rgb")
    _validate_rgb(projection_rgb, "projection_rgb")
    # Select once; each backend owns option validation and transform estimation.
    if backend == "directional_chamfer":
        from . import directional_chamfer as implementation
    elif backend == "match_anything":
        from . import match_anything as implementation
    else:
        raise NotImplementedError(f"Unknown alignment backend: {backend}")
    settings = implementation.validate_options(model, dict(options or {}))

    # Keep backend diagnostics, including partial progress when estimation fails.
    started = perf_counter()
    diagnostics: dict[str, Any] = {"options": settings}
    warnings: list[str] = []
    status: Literal["ok", "uncertain", "failed"] = "ok"
    try:
        # Reject blank images before running optimization or learned inference.
        drawing_gray = cv2.cvtColor(drawing_rgb, cv2.COLOR_RGB2GRAY)
        projection_gray = cv2.cvtColor(projection_rgb, cv2.COLOR_RGB2GRAY)
        if np.ptp(drawing_gray) == 0 or np.ptp(projection_gray) == 0:
            raise ValueError("uniform/blank image has no usable lines")
        # Both backends return drawing→projection in OpenCV integer-centre coordinates.
        matrix, status, warnings = implementation.estimate_transform(
            drawing_rgb,
            projection_rgb,
            model=model,
            options=settings,
            runtime=runtime,
            diagnostics=diagnostics,
        )
        # Publish the Region coordinate frame and reject poles/singular transforms.
        matrix = _checked_transform(pixel_boundary_transform(matrix), drawing_rgb.shape)
        # A mirrored candidate is usable for inspection, but needs an explicit warning.
        if np.linalg.det(matrix) < 0:
            status = "uncertain"
            warnings.append(
                "Transform reverses image orientation; inspect view axes or mirroring."
            )
    except (
        ValueError,
        RuntimeError,
        ImportError,
        np.linalg.LinAlgError,
        cv2.error,
    ) as error:
        # Numerical/inference failures remain reportable and never carry a false matrix.
        matrix = None
        status = "failed"
        diagnostics["error"] = f"{type(error).__name__}: {error}"
    # Return arrays and metadata only; residual calculation and file I/O live elsewhere.
    diagnostics["seconds"] = perf_counter() - started
    return AlignmentResult(
        backend,
        model,
        status,
        matrix.tolist() if matrix is not None else None,
        diagnostics,
        tuple(warnings),
    )
