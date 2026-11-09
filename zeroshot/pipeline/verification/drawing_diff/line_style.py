"""Recognize input line styles and compare them with corresponding DXF curves."""

from itertools import product

import cv2
import numpy as np
from ezdxf.layouts import Modelspace
from scipy.spatial import cKDTree
from skimage.morphology import thin

from .image_ops import WIDTH_RATIO, distance_map
from .unmatched import Unmatched, drawing_curves


def input_hidden_curves(
    input_rgb: np.ndarray, part_ink: np.ndarray
) -> list[tuple[np.ndarray, str]]:
    """Recognize straight hidden boundaries across their drawing's dash gaps.

    Only repeated dashes establish a curve. Local evidence covers its own
    interval, not a whole Hough segment that may continue through solid ink.
    shortcut: curved hidden boundaries need a separate curve recognizer.
    """
    segments = cv2.HoughLinesP(
        thin(part_ink).astype(np.uint8),
        rho=1,
        theta=np.pi / 720,
        threshold=20,
        minLineLength=50,
        maxLineGap=8,
    )
    if segments is None:
        return []
    ink = part_ink.astype(np.float32)
    darkness = 1 - cv2.cvtColor(input_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    curves = []
    for segment in segments.reshape(-1, 2, 2):
        start, end = segment.astype(float)
        length = np.linalg.norm(end - start)
        xy = np.linspace(start, end, int(np.ceil(length)) + 1)
        step = length / (len(xy) - 1)
        normal = ((end - start) / length)[::-1] * (-1, 1)
        probes = xy[None] + np.arange(-1, 1.01, 0.5)[:, None, None] * normal
        support = (
            cv2.remap(
                ink,
                probes[..., 0].astype(np.float32),
                probes[..., 1].astype(np.float32),
                cv2.INTER_NEAREST,
            ).max(axis=0)
            > 0
        )
        strength, _ = _ink_profiles(darkness, xy, normal)
        if _style(support, step, False, strength) == "hidden":
            curves.append((np.array([start, end]), "hidden"))
            continue
        # Crossings can disturb regularity outside an otherwise clear dash run.
        accepted = np.zeros(len(xy), bool)
        for i in range(0, len(xy) - 63, 32):
            section = slice(i, i + 64)
            if _style(support[section], step, False, strength[section]) == "hidden":
                accepted[section] = True
        edges = np.flatnonzero(np.diff(np.r_[False, accepted, False]))
        for first, stop in zip(edges[::2], edges[1::2], strict=True):
            curves.append((xy[[first, stop - 1]], "hidden"))
    return curves


def find_line_style_mismatches(
    output_modelspace: Modelspace,
    output_model_to_input: np.ndarray,
    input_rgb: np.ndarray,
    input_comparison_ink: np.ndarray,
    input_annotation: np.ndarray | None = None,
) -> tuple[list[Unmatched], dict]:
    """Unique mismatching curve portions, plus classified and unknown line lengths.

    Local translation tolerates small registration errors. It minimizes distance
    to ink, independently of the expected style, without widening dash gaps.
    output_model_to_input maps output DXF model coordinates to input pixels.
    input_rgb retains pale dash gaps lost in binarization; input_comparison_ink
    contains only strokes selected by the caller's annotation/silhouette filtering.
    """
    output_curves = list(_curves(output_modelspace, output_model_to_input))
    states = np.zeros(len(output_curves), np.uint8)
    coverages = np.zeros(len(output_curves))
    output_curve_trees = [cKDTree(xy) for xy, _, _, _ in output_curves]
    input_ink_distance = distance_map(input_comparison_ink).astype(np.float32)
    input_ink_pixels = input_comparison_ink.astype(np.float32)
    input_gray = cv2.cvtColor(input_rgb, cv2.COLOR_RGB2GRAY)
    input_darkness = (255 - input_gray.astype(np.float32)) / 255
    height, width = input_comparison_ink.shape
    output_visible_ink = np.zeros(input_comparison_ink.shape, np.uint8)
    for xy, style, _, _ in output_curves:
        if style in {"visible", "tangent"}:
            cv2.polylines(
                output_visible_ink, [np.rint(xy).astype(np.int32)], False, 1, 1
            )
    output_visible_distance = distance_map(output_visible_ink.astype(bool))
    input_annotation_distance = (
        distance_map(input_annotation).astype(np.float32)
        if input_annotation is not None and input_annotation.any()
        else None
    )
    offsets = np.array(list(product(np.arange(-2, 2.01, 0.5), repeat=2)))
    stats = {
        "status": "ok",
        "checked_curves": 0,
        "unknown_curves": 0,
        "mismatch_count": 0,
    }
    for index, (xy, output_style, step, closed) in enumerate(output_curves):
        if output_style == "tangent":
            continue
        stats["unknown_curves"] += 1
        # shortcut: a fragment inside one dash can look solid; use wider stroke context if needed.
        if len(xy) < 24:
            continue
        if not (
            (xy >= 5).all()
            and (xy[:, 0] < width - 5).all()
            and (xy[:, 1] < height - 5).all()
        ):
            continue
        pixels = np.rint(xy).astype(int)
        # Partially removed text beside a part edge can masquerade as regular dashes.
        if (
            input_annotation_distance is not None
            and (input_annotation_distance[pixels[:, 1], pixels[:, 0]] <= 3).mean()
            > 0.2
        ):
            continue
        # A visible or optional boundary can explain a solid stroke over a hidden edge.
        if (
            output_style == "hidden"
            and (output_visible_distance[pixels[:, 1], pixels[:, 0]] < 3).mean() > 0.2
        ):
            continue
        shifted = xy[None] + offsets[:, None]
        distances = cv2.remap(
            input_ink_distance,
            shifted[..., 0].astype(np.float32),
            shifted[..., 1].astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=1000,
        )
        # Fit only the supported part; gaps must not pull a dashed curve off its ink.
        count = max(1, int(len(xy) * 0.6))
        fit = np.partition(distances, count - 1, axis=1)[:, :count].mean(axis=1)
        best = np.argmin(fit + np.linalg.norm(offsets, axis=1) * 0.01)
        if fit[best] > 0.6:
            continue
        direction = (
            np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)
            if closed
            else np.gradient(xy, axis=0)
        )
        normal = direction[:, ::-1] * (-1, 1)
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        # Tolerate normal displacement without filling gaps along the curve.
        probes = shifted[best][None] + np.arange(-1, 1.01, 0.5)[:, None, None] * normal
        support = (
            cv2.remap(
                input_ink_pixels,
                probes[..., 0].astype(np.float32),
                probes[..., 1].astype(np.float32),
                cv2.INTER_NEAREST,
            ).max(axis=0)
            > 0
        )
        support &= (
            distances[best] <= 1.0
        )  # Sampling must not widen positional tolerance.
        strength, widths = _ink_profiles(input_darkness, shifted[best], normal)
        input_style = _style(support, step, closed, strength)
        if input_style is None:
            continue
        if (
            input_style == "visible"
            and input_annotation_distance is not None
            and _continues_annotation(
                shifted[best], input_annotation_distance, input_darkness, widths
            )
        ):
            continue
        if input_style != output_style:
            # Nearby distinct edges can fill each other's gaps; coincident duplicates cannot.
            ambiguous = False
            for other_index, (other, _, _, _) in enumerate(output_curves):
                if other_index == index or (
                    (other.max(axis=0) < xy.min(axis=0) - 3).any()
                    or (other.min(axis=0) > xy.max(axis=0) + 3).any()
                ):
                    continue
                separation = output_curve_trees[other_index].query(xy)[0]
                if ((separation > 1.25) & (separation <= 3)).mean() > 0.2:
                    ambiguous = True
                    break
            if ambiguous:
                continue
        stats["unknown_curves"] -= 1
        stats["checked_curves"] += 1
        states[index] = 1 if input_style == output_style else 2
        coverages[index] = support.mean()
    groups, lengths = _summarize(
        output_curves, states, coverages, output_curve_trees, input_comparison_ink.shape
    )
    stats["mismatch_count"] = len(groups)
    stats.update(lengths)
    return groups, stats



def _runs(support: np.ndarray, step: float, closed: bool):
    ends = np.r_[0, np.flatnonzero(np.diff(support)) + 1, len(support)]
    lengths = np.diff(ends) * step
    values = support[ends[:-1]]
    if closed and len(lengths) > 1 and values[0] == values[-1]:
        lengths[0] += lengths[-1]
        lengths, values = lengths[:-1], values[:-1]
    return lengths, values


def _style(
    support: np.ndarray, step: float, closed: bool, strength: np.ndarray
) -> str | None:
    """Require continuity or repeated dashes; sparse/broken ink alone is unknown."""
    lengths, values = _runs(support, step, closed)
    gaps = lengths[~values]
    full_gaps = gaps if closed else lengths[1:-1][~values[1:-1]]
    if (
        support.mean() >= 0.9
        and (not gaps.size or gaps.max() <= 2)
        and len(full_gaps) < 3
        and (
            closed
            or not len(full_gaps)
            or np.all(strength[~support] >= np.median(strength) - 0.1)
        )
    ):
        # Pale dash gaps can survive as ink after global binarization.
        padded = np.pad(support, 1, mode="wrap" if closed else "edge")
        reliable = padded[:-2] & support & padded[2:]
        # A tolerated raster hole and its antialiased rim are not new dash evidence.
        runs, pale = _runs(
            (strength < np.median(strength) - 0.1) & reliable, step, closed
        )
        if not closed:
            runs, pale = runs[1:-1], pale[1:-1]
        dips, ink = runs[pale], runs[~pale]
        # Isolated tone variations do not establish a repeated dash pattern.
        if (
            np.count_nonzero(dips >= 2 * step) >= 2
            and len(ink)
            and np.median(dips) >= 2 * step
            and np.median(ink) >= 1.5 * np.median(dips)
        ):
            return None
        return "visible"
    # Repeated short gaps may be fine dashes, not isolated raster holes.
    if len(full_gaps) >= 3 and full_gaps.max() <= 2:
        return None
    # A one-sample raster hole must not split a dash into two different lengths.
    padded = np.pad(support, 1, mode="wrap" if closed else "edge")
    lengths, values = _runs(support | (padded[:-2] & padded[2:]), step, closed)
    # Open curves can cut through the first/last dash; only full runs set regularity.
    if not closed:
        lengths, values = lengths[1:-1], values[1:-1]
    ink, gaps = lengths[values], lengths[~values]
    if not 0.4 <= support.mean() <= 0.85 or min(len(ink), len(gaps)) < 3:
        return None
    # Long-short centre lines and one missing arc are not hidden-line evidence.
    if all(runs.min() >= 2 and runs.max() <= 2 * runs.min() for runs in (ink, gaps)):
        return "hidden"
    return None


def _ink_profiles(darkness, xy, normal):
    """Darkest central ink and integrated stroke width along a sampled curve."""
    offsets = np.arange(-3, 3.01, 0.5)
    cross_section = xy[None] + offsets[:, None, None] * normal
    values = cv2.remap(
        darkness,
        cross_section[..., 0].astype(np.float32),
        cross_section[..., 1].astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mid = len(offsets) // 2
    # Stop at white gaps so a neighbouring dashed line cannot modulate a solid stroke.
    left = np.cumprod(values[:mid][::-1] > 0.05, axis=0)[::-1]
    right = np.cumprod(values[mid + 1 :] > 0.05, axis=0)
    widths = 0.5 * (
        values[mid]
        + (values[:mid] * left).sum(axis=0)
        + (values[mid + 1 :] * right).sum(axis=0)
    )
    return values[np.abs(offsets) <= 1].max(axis=0), widths


def _continues_annotation(xy, annotation_distance, darkness, stroke_widths):
    """A same-width annotation continuation can hide gaps without proving geometry."""
    direction = xy[-1] - xy[0]
    length = np.linalg.norm(direction)
    if length < 1:
        return False
    direction /= length
    normal = direction[::-1] * (-1, 1)
    if np.abs((xy - xy[0]) @ normal).max() > 0.5:
        return False
    # Follow connected ink; a fixed endpoint radius misses long internal extensions.
    reach = (
        np.arange(1, min(16000, int(np.hypot(*darkness.shape)) + 1))[:, None]
        * direction
    )
    beyond = np.concatenate([xy[0] - reach, xy[-1] + reach])
    nearby = (
        cv2.remap(
            annotation_distance,
            beyond[None, :, 0].astype(np.float32),
            beyond[None, :, 1].astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=1000,
        )[0]
        <= 1
    )
    strength, widths = _ink_profiles(darkness, beyond, normal)
    # Three white samples break continuity; short raster holes do not.
    blank = np.pad((strength < 0.1).reshape(2, -1), ((0, 0), (2, 0)))
    broken = blank[:, :-2] & blank[:, 1:-1] & blank[:, 2:]
    nearby &= ~np.maximum.accumulate(broken, axis=1).ravel()
    if nearby.sum() < 8:
        return False
    # Hidden dashes can thicken the stroke; its narrow stretches reveal the annotation.
    narrow = np.convolve(stroke_widths, np.ones(3) / 3, mode="valid").min()
    return narrow < WIDTH_RATIO * np.median(widths[nearby])


def _curves(modelspace: Modelspace, transform: np.ndarray):
    for xy, style in drawing_curves(modelspace, transform, tolerance_px=0.25):
        along = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        # shortcut: long curves need segmentation to fit within OpenCV's remap limit.
        if along[-1] > 16000:
            continue
        closed = np.linalg.norm(xy[0] - xy[-1]) < 0.1
        samples = np.linspace(
            0, along[-1], max(2, int(np.ceil(along[-1]))), endpoint=not closed
        )
        yield (
            np.column_stack([np.interp(samples, along, axis) for axis in xy.T]),
            style,
            float(samples[1] - samples[0]),
            closed,
        )


def _summarize(curves, states, coverages, trees, shape):
    """Use the same unique curve portions for mismatch groups and line lengths."""
    lengths = np.zeros(3)  # unknown, classified, mismatched
    bounds = np.array([(xy.min(axis=0), xy.max(axis=0)) for xy, _, _, _ in curves])
    curve_lengths = [step * (len(xy) - (not closed)) for xy, _, step, closed in curves]
    measured = np.zeros(len(curves), bool)
    reported = np.zeros(shape, bool)
    groups = []
    # Known classifications own overlaps before unknowns; longer curves own fragments.
    order = sorted(
        range(len(curves)),
        key=lambda i: (states[i], curve_lengths[i], tuple(bounds[i].ravel())),
        reverse=True,
    )
    for index in order:
        xy, style, step, closed = curves[index]
        if style == "tangent":
            continue
        weights = np.full(len(xy), step)
        if not closed:
            weights[[0, -1]] *= 0.5
        covered = np.zeros(len(xy), bool)
        nearby = measured & (bounds[:, 0] <= bounds[index, 1] + 0.75).all(axis=1)
        nearby &= (bounds[:, 1] >= bounds[index, 0] - 0.75).all(axis=1)
        # shortcut: subpixel-near curves count once; use exact unions if their separation matters.
        for other in np.flatnonzero(nearby):
            covered |= trees[other].query(xy, distance_upper_bound=0.75)[0] < 0.75
        inside = ((xy >= 0) & (xy <= (shape[1] - 1, shape[0] - 1))).all(axis=1)
        unique = inside & ~covered
        lengths[states[index]] += weights[unique].sum()
        measured[index] = True
        if states[index] != 2:
            continue
        pixels = np.unique(np.rint(xy[unique]).astype(int), axis=0)
        pixels = pixels[~reported[pixels[:, 1], pixels[:, 0]]]
        if len(pixels):
            groups.append(
                Unmatched(
                    "line_style",
                    pixels,
                    input_style="visible" if style == "hidden" else "hidden",
                    output_style=style,
                    coverage=float(coverages[index]),
                )
            )
            reported[pixels[:, 1], pixels[:, 0]] = True
    return groups, {
        "mismatch_length_px": float(lengths[2]),
        "checked_length_px": float(lengths[1] + lengths[2]),
        "unknown_length_px": float(lengths[0]),
    }
