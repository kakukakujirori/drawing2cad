"""Backend-independent operations on dark line drawings with light backgrounds."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.morphology import skeletonize


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


# Two-tone drawings print the part dark and its annotation (dimension, extension
# and leader lines, arrows, text) in light gray. Antialiasing alone spreads one
# tone's line centres over less than this, measured on low-resolution scans.
TONE_GAP = 100
# Each tone must be a real population, not a few stray pixels.
MIN_TONE_SHARE = 0.05
# Light ink this close to a part line belongs to it: a pale stretch of the line.
PART_REACH_PX = 3


def annotation_mask(gray: np.ndarray, ink: np.ndarray) -> np.ndarray:
    """Ink in a separate lighter tone than the part's lines; empty if single-tone.

    Each pixel takes the darkest value around it, so a dark line's antialiased
    rim stays with its line. Where the line centres do not form two clearly
    separate tones, every line is treated as the part's.
    """
    core = cv2.erode(gray, np.ones((3, 3), np.uint8))
    centres = core[skeletonize(ink)]
    none = np.zeros_like(ink, dtype=bool)
    if centres.size < 2 or np.ptp(centres) == 0:
        return none
    threshold, _ = cv2.threshold(
        centres.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    dark, light = centres[centres <= threshold], centres[centres > threshold]
    if (
        min(dark.size, light.size) < MIN_TONE_SHARE * centres.size
        or light.mean() - dark.mean() < TONE_GAP
    ):
        return none
    # Blur can pale a part line where it curves; that must not open its outline.
    part = (ink & (core <= threshold)).astype(np.uint8)
    near_part = cv2.dilate(
        part, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * PART_REACH_PX + 1,) * 2)
    )
    return ink & (core > threshold) & ~near_part.astype(bool)


def distance_map(foreground: np.ndarray) -> np.ndarray:
    """Return each pixel's Euclidean distance to the nearest foreground pixel."""
    if not np.any(foreground):
        return np.full(foreground.shape, np.hypot(*foreground.shape), dtype=np.float64)
    # OpenCV measures distance to zero, so selected foreground pixels become zero.
    return cv2.distanceTransform(
        (~foreground).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    ).astype(np.float64)


# Below this share of the filled part surviving the opening, the part is thin-walled.
MIN_KEPT_AREA = 0.5


def drawing_area(ink: np.ndarray) -> np.ndarray:
    """The largest filled input part, excluding thin annotations; empty if absent.

    Empty too when the part's own walls are as thin as the annotations, as in
    sheet metal seen edge-on, since the opening then erases the part itself.
    """
    # Do not close input gaps: that can join dimension lines to the part.
    filled = binary_fill_holes(ink).astype(np.uint8)
    area = cv2.morphologyEx(
        filled, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(area)
    before = cv2.connectedComponentsWithStats(filled)[2][1:, cv2.CC_STAT_AREA]
    if count == 1 or stats[1:, cv2.CC_STAT_AREA].max() < MIN_KEPT_AREA * before.max():
        return np.zeros_like(ink, dtype=bool)
    # ponytail: one main connected part per view; assemblies need per-part masks.
    return labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
