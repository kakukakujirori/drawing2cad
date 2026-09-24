"""Backend-independent operations on dark line drawings with light backgrounds."""

from __future__ import annotations

import cv2
import numpy as np


def foreground_mask(gray: np.ndarray) -> np.ndarray:
    """Return True at dark line pixels, preserving every foreground component.

    Inputs are normally nearly binary; Otsu thresholding defensively handles
    antialiasing and gray pixels. This selects pixels, not a crop or an annotation
    removal region. Blank images and reversed foreground polarity are rejected.
    """
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("Expected a 2D uint8 grayscale image")
    if np.ptp(gray) == 0:
        raise ValueError("Uniform/blank image has no usable lines")
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    mask = binary > 0
    if np.count_nonzero(mask) < 8 or mask.mean() > 0.65:
        raise ValueError("No usable line drawing, or foreground polarity is wrong")
    return mask


def distance_map(foreground: np.ndarray) -> np.ndarray:
    """Return each pixel's Euclidean distance to the nearest foreground pixel."""
    if not np.any(foreground):
        return np.full(foreground.shape, np.hypot(*foreground.shape), dtype=np.float64)
    # OpenCV measures distance to zero, so selected foreground pixels become zero.
    return cv2.distanceTransform(
        (~foreground).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    ).astype(np.float64)
