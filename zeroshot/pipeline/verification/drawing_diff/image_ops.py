"""Backend-independent operations on dark line drawings with light backgrounds."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.filters import threshold_otsu
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
    near_part = cv2.dilate(part, disk(PART_REACH_PX))
    return ink & (core > threshold) & ~near_part.astype(bool)


def disk(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)


# Gaps a rendered outline may leave, closed before its area is filled.
# Tuned on outputs/silhouette_20260927.
CLOSE_PX = 3


def filled_area(ink: np.ndarray) -> np.ndarray:
    """The area inside the outermost lines."""
    closed = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_CLOSE, disk(CLOSE_PX))
    return binary_fill_holes(closed)


def external_boundary(ink: np.ndarray) -> np.ndarray:
    """The outer edge of the area a line drawing encloses."""
    area = filled_area(ink).astype(np.uint8)
    return area.astype(bool) & ~cv2.erode(area, np.ones((3, 3), np.uint8)).astype(bool)


# Drafting draws a part's outlines thick and its other lines thin: dimension,
# extension and leader lines, and also hidden and centre lines. ISO 128 sets
# thick at twice thin, but blur and low resolution widen thin lines, so a
# drawing counts as using both widths from this ratio on.
WIDTH_RATIO = 1.3
# Strokes this far from every line of the registered part are not its outline.
AWAY_PX = 12
# Stroke centres needed on each side to tell their widths.
MIN_WIDTH_SAMPLES = 50


def thin_mask(gray: np.ndarray, ink: np.ndarray, part: np.ndarray) -> np.ndarray:
    """The ink drawn in thin lines, when the drawing draws its outlines thicker.

    Thin lines are every line but the part's outlines: dimension, extension and
    leader lines, and hidden and centre lines too. part is the registered CAD's
    ink on this drawing. It tells which width the outlines take here: strokes on
    its outer edge are outlines, and strokes far from all its lines are not.
    Otsu's threshold on stroke width splits thin from thick. The result is empty
    unless the outlines are clearly the wider: most of their strokes lie above
    that threshold, most strokes away from the part lie below it, and the two
    widths are WIDTH_RATIO apart.

    A stroke's width is its ink summed across the stroke. Unlike a count of
    binary pixels, it stays the same wherever a thin line falls on the pixels.
    """
    centres = skeletonize(ink)
    darkness = (255 - gray.astype(np.float32)) / 255
    width = cv2.boxFilter(darkness, -1, (7, 7), normalize=False) / 7
    outline = centres & (distance_map(external_boundary(part)) <= 2)
    away = centres & (distance_map(skeletonize(part)) > AWAY_PX)
    if min(outline.sum(), away.sum()) < MIN_WIDTH_SAMPLES:
        return np.zeros_like(ink, dtype=bool)
    level = threshold_otsu(width[centres])
    on, off = width[outline], width[away]
    # Most outline strokes lie above the split and most others below it.
    if (on >= level).mean() <= 0.5 or (off >= level).mean() >= 0.5:
        return np.zeros_like(ink, dtype=bool)
    if np.median(on) < WIDTH_RATIO * np.median(off):
        return np.zeros_like(ink, dtype=bool)
    # Each stroke pixel takes the width of the centre lines within the outlines'
    # half-width; ink farther from any, like a filled arrowhead, is no outline.
    reach = max(2, int(np.ceil(np.median(distance_map(~ink)[outline]))))
    square = np.ones((2 * reach + 1,) * 2, np.uint8)
    beside = cv2.dilate(np.where(centres, width, 0), square)
    return ink & (beside < level)


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
# The opening that drops thin annotations from the filled part.
OPEN_PX = 3
# A silhouette filled inside a broken outline leaves more of the outline than
# this outside it. Measured on saved builds: whole silhouettes leave up to 0.22,
# partial ones from 0.47.
MAX_OUTLINE_OUTSIDE = 0.35


def drawing_area(ink: np.ndarray) -> np.ndarray:
    """The largest filled input part, excluding thin annotations; empty if absent.

    Empty too when the part's own walls are as thin as the annotations, as in
    sheet metal seen edge-on, since the opening then erases the part itself.
    """
    # Do not close input gaps: that can join dimension lines to the part.
    filled = binary_fill_holes(ink).astype(np.uint8)
    area = cv2.morphologyEx(filled, cv2.MORPH_OPEN, disk(OPEN_PX))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(area)
    before = cv2.connectedComponentsWithStats(filled)[2][1:, cv2.CC_STAT_AREA]
    if count == 1 or stats[1:, cv2.CC_STAT_AREA].max() < MIN_KEPT_AREA * before.max():
        return np.zeros_like(ink, dtype=bool)
    # ponytail: one main connected part per view; assemblies need per-part masks.
    return labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])


def beside_part(ink: np.ndarray, outline: np.ndarray) -> np.ndarray:
    """Ink outside the part's filled silhouette, like dimensions; empty if unfilled.

    Every line of the part lies within its silhouette. The silhouette is filled
    from outline, the ink not drawn thin. If more than MAX_OUTLINE_OUTSIDE of
    that ink lies outside it yet joins it, the outline was broken and the fill
    partial; then only ink that never joins the silhouette, like dimension
    text, is returned.
    """
    area = drawing_area(outline)
    if not area.any():
        return np.zeros_like(ink, dtype=bool)
    # Undo the opening, which can shave the outline itself.
    near = cv2.dilate(area.astype(np.uint8), disk(OPEN_PX)).astype(bool)
    outside = ink & ~near
    _, labels = cv2.connectedComponents(ink.astype(np.uint8), connectivity=8)
    joined = np.isin(labels, np.unique(labels[ink & near]))
    if (outline & outside & joined).sum() > MAX_OUTLINE_OUTSIDE * outline.sum():
        return outside & ~joined
    return outside
