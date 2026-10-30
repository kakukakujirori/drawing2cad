"""Backend-independent operations on dark line drawings with light backgrounds."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, gaussian_filter1d
from scipy.signal import find_peaks
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


@dataclass(frozen=True)
class DrawingMasks:
    """Annotation to exclude from fitting, and outlines for silhouette comparison."""

    annotation: np.ndarray
    outline: np.ndarray


def drawing_masks(
    gray: np.ndarray, ink: np.ndarray, part: np.ndarray | None = None
) -> DrawingMasks:
    """Separate annotation and silhouettes using one tone/width decision.

    Fitting keeps hidden lines and separate thin or open views. The silhouette
    also excludes thin hidden/centre lines; registered CAD can confirm its width.
    """
    light = _light_annotations(gray, ink)
    if light.any():
        # Once tone separates annotation, width can distinguish two real part edges.
        return DrawingMasks(light, ink & ~light)
    thin = thin_mask(gray, ink)
    annotation = _external_annotations(ink, thin)
    cleaned = ink & ~annotation
    if part is not None:
        thin = thin_mask(gray, cleaned, part)
    return DrawingMasks(annotation, cleaned & ~thin)


def _light_annotations(gray: np.ndarray, ink: np.ndarray) -> np.ndarray:
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
# Stroke centres needed at each width to tell them apart.
MIN_WIDTH_SAMPLES = 50


def stroke_widths(gray: np.ndarray, ink: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stroke centres and their ink-integral widths, without nearby parallel lines.

    Take the shortest crossing in eight directions. A gap ends a crossing;
    a square window would count neighbouring strokes and dense text as width.
    """
    centres = skeletonize(ink)
    darkness = (255 - gray.astype(np.float32)) / 255
    ys, xs = np.nonzero(centres)
    width = np.zeros(gray.shape, np.float32)
    # shortcut: widths above 24px saturate; widen profiles for high-resolution scans.
    steps = np.arange(-12, 12.5, 0.5, dtype=np.float32)
    mid = len(steps) // 2
    # Batches stay below OpenCV remap's 32767-row limit, even on large sheets.
    for start in range(0, len(xs), 16000):
        x, y = xs[start : start + 16000], ys[start : start + 16000]
        measured = np.full(len(x), np.inf, np.float32)
        for angle in np.arange(8) * np.pi / 8:
            values = cv2.remap(
                darkness,
                (x[:, None] + np.cos(angle) * steps).astype(np.float32),
                (y[:, None] + np.sin(angle) * steps).astype(np.float32),
                cv2.INTER_LINEAR,
            )
            left = np.cumprod(values[:, :mid][:, ::-1] > 0.05, axis=1)[:, ::-1]
            right = np.cumprod(values[:, mid + 1 :] > 0.05, axis=1)
            crossing = (
                values[:, mid]
                + (values[:, :mid] * left).sum(axis=1)
                + (values[:, mid + 1 :] * right).sum(axis=1)
            ) * 0.5
            measured = np.minimum(measured, crossing)
        width[y, x] = measured
    return centres, width


def thin_mask(
    gray: np.ndarray, ink: np.ndarray, part: np.ndarray | None = None
) -> np.ndarray:
    """Thin ink when two distinct stroke widths exist; includes hidden lines.

    Width modes, rather than a split of all samples, keep text and crossings
    from raising the threshold above real outlines. Optional registered CAD
    confirms that its outer boundary predominantly uses the thicker width.
    Ink far from CAD can be missing geometry or text, not necessarily thin ink.
    """
    none = np.zeros_like(ink, dtype=bool)
    centres, width = stroke_widths(gray, ink)
    values = width[centres]
    if values.size < 2 * MIN_WIDTH_SAMPLES:
        return none
    # Log bins give the same relative tolerance to narrow and wide strokes.
    bins = np.arange(-0.7, 3.3, 0.05)
    hist, _ = np.histogram(np.log(np.maximum(values, 0.1)), bins=bins)
    smooth = gaussian_filter1d(hist.astype(float), 1.3)
    peaks, _ = find_peaks(smooth, prominence=max(2, smooth.max() * 0.08), distance=5)
    modes = [
        np.exp((bins[i] + bins[i + 1]) / 2)
        for i in peaks
        if hist[max(0, i - 2) : i + 3].sum()
        >= max(MIN_WIDTH_SAMPLES, 0.05 * values.size)
    ]
    pairs = [(a, b) for a, b in pairwise(modes) if b >= WIDTH_RATIO * a]
    if not pairs:
        return none
    narrow, wide = pairs[0]
    level = np.sqrt(narrow * wide)
    if part is not None:
        outline = centres & (distance_map(external_boundary(part)) <= 2)
        if outline.sum() < MIN_WIDTH_SAMPLES or (width[outline] >= level).mean() <= 0.5:
            return none
    reach = max(2, int(np.ceil(wide / 2)))
    beside = cv2.dilate(width, np.ones((2 * reach + 1,) * 2, np.uint8))
    # Filled crossings can extend beyond the sampled centres; unmeasured ink stays.
    thin = ink & (beside > 0) & (beside < level)
    kept = ink & ~thin
    # A closed pinhole or a filled stroke alone is no part interior.
    holes = binary_fill_holes(kept) & ~kept
    areas = cv2.connectedComponentsWithStats(holes.astype(np.uint8), connectivity=4)[2][
        1:, cv2.CC_STAT_AREA
    ]
    if not areas.size or areas.max() < np.pi * (wide / 2) ** 2:
        return none
    # Measure the filled silhouette so dense internal edges do not shrink it.
    if distance_map(~drawing_area(kept)).max() < wide:
        return none
    return thin


def _external_annotations(ink: np.ndarray, thin: np.ndarray) -> np.ndarray:
    """Thin exterior annotation attached to closed outlines.

    Thin lines inside a part can be hidden geometry. Only mark thin ink
    outside a fillable outline it joins; keep open and separate thin views.
    """
    annotation = np.zeros_like(ink, dtype=bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8))
    for label in range(1, count):
        x, y, w, h, size = stats[label]
        if size < MIN_WIDTH_SAMPLES:
            continue
        box = np.s_[y : y + h, x : x + w]
        component = labels[box] == label
        outline = component & ~thin[box]
        if not (binary_fill_holes(outline) & ~outline).any():
            continue
        outside = beside_part(component, outline)
        # Keep thick fragments of other views even when one component joins them.
        annotation[box] |= outside & thin[box]
    # A disconnected hole can still lie inside another component's outline.
    return annotation & ~binary_fill_holes(ink & ~thin)


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
