"""Fit straight lines and circles to the strokes inside boxes of a drawing image."""

import math
from dataclasses import dataclass, replace
from functools import lru_cache
from inspect import cleandoc
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from langchain_core.tools import BaseTool, tool
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares

from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.tools.errors import ToolFeedbackError
from zeroshot.pipeline.tools.load_image import resolve_image_path

type Kind = Literal["line", "circle"]

MAX_ITEMS = 12
MAX_STROKES = 4
MAX_BOX_AREA = 1_500_000
# A stroke centre supports a curve within this distance and normal angle.
BAND_PX = 1.0
NORMAL_COS = math.cos(math.radians(25))
# Acceptance: tuned on clean renders, scans and hand sketches.
MAX_RMS_PX = 0.25
MAX_RADIUS_SPREAD = 0.05
MIN_COVERAGE = 0.4
# Solid strokes keep 85% or more even where others cross; dashes keep less.
DASHED_BELOW = 0.83
MIN_BOW_PX = 1.5
# Trim an arc's end only while it fits this many times worse than the arc's mean.
END_MISFIT = 2.0
# A circle whose widest gap is narrower than this is reported as full.
OPEN_ARC_DEG = 20
# A line drifting less than this end to end is reported as horizontal or vertical.
STRAIGHT_PX = 0.5
# The box cuts a stroke's end when a step this long past it leaves the box.
EDGE_PX = 2.0
DRAWS = 20_000
COUNTED = 600
SCORED = 50


@dataclass(frozen=True)
class Sheet:
    gray: np.ndarray
    ink: np.ndarray
    # Shorter of the horizontal and vertical ink runs through each pixel.
    run_width: np.ndarray
    threshold: float
    line_width: float


@dataclass(frozen=True)
class Points:
    """Sub-pixel stroke centres, their normal angles and stroke widths."""

    xy: np.ndarray
    normal: np.ndarray
    width: np.ndarray

    def take(self, index: np.ndarray) -> "Points":
        return Points(self.xy[index], self.normal[index], self.width[index])


@dataclass(frozen=True)
class Circle:
    cx: float
    cy: float
    r: float

    def residual(self, xy: np.ndarray) -> np.ndarray:
        return np.hypot(xy[:, 0] - self.cx, xy[:, 1] - self.cy) - self.r

    def normal(self, xy: np.ndarray) -> np.ndarray:
        return np.arctan2(xy[:, 1] - self.cy, xy[:, 0] - self.cx)

    def angle(self, xy: np.ndarray) -> np.ndarray:
        """Counterclockwise from +x with +y up, in [0, 2 pi)."""
        return np.mod(np.arctan2(self.cy - xy[:, 1], xy[:, 0] - self.cx), 2 * np.pi)

    def point(self, angle: float) -> tuple[float, float]:
        return self.cx + self.r * math.cos(angle), self.cy - self.r * math.sin(angle)


@dataclass(frozen=True)
class Line:
    x: float
    y: float
    ux: float
    uy: float

    def residual(self, xy: np.ndarray) -> np.ndarray:
        return (xy[:, 0] - self.x) * -self.uy + (xy[:, 1] - self.y) * self.ux

    def normal(self, xy: np.ndarray) -> np.ndarray:
        return np.full(len(xy), math.atan2(self.ux, -self.uy))

    def along(self, xy: np.ndarray) -> np.ndarray:
        return (xy[:, 0] - self.x) * self.ux + (xy[:, 1] - self.y) * self.uy

    def point(self, t: float) -> tuple[float, float]:
        return self.x + t * self.ux, self.y + t * self.uy


@dataclass(frozen=True)
class Stroke:
    model: Circle | Line
    # Where the ink lies along the model: radians ccw for a circle, px for a line.
    start: float
    end: float
    support: float
    coverage: float
    rms: float
    radius_sd: float = 0.0
    centre_sd: float = 0.0


@dataclass(frozen=True)
class Finding:
    strokes: list[Stroke]
    reason: str | None = None
    more: bool = False


def load_sheet(path: Path) -> Sheet:
    stat = path.stat()
    return _load_sheet(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=2)
def _load_sheet(path: str, mtime_ns: int, size: int) -> Sheet:
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
    paper = Image.new("RGBA", rgba.size, "white")
    gray = np.asarray(Image.alpha_composite(paper, rgba).convert("L"))
    if np.median(gray) < 128:
        gray = 255 - gray
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = gray < threshold
    run_width = np.minimum(_run_lengths(ink), _run_lengths(ink.T).T)
    widths = run_width[ink & (run_width <= 12)]
    return Sheet(
        gray=gray.astype(np.float32),
        ink=ink,
        run_width=run_width,
        threshold=float(threshold),
        line_width=float(np.percentile(widths, 90)) if widths.size else 1.0,
    )


def _run_lengths(ink: np.ndarray) -> np.ndarray:
    """Length of the horizontal ink run through each pixel."""
    edges = np.diff(np.pad(ink, ((0, 0), (1, 1))).astype(np.int8), axis=1).ravel()
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    runs = np.zeros(ink.shape, np.int32)
    runs[ink] = np.repeat(lengths, lengths)
    return runs


def centre_points(sheet: Sheet, box: tuple[int, int, int, int]) -> Points:
    """Steger's ridge points: where the across-stroke darkness peaks, sub-pixel."""
    sigma = max(1.0, sheet.line_width / (2 * math.sqrt(3)))
    pad = math.ceil(4 * sigma)
    x0, y0, x1, y1 = box
    height, width = sheet.gray.shape
    left, top = max(x0 - pad, 0), max(y0 - pad, 0)
    right, bottom = min(x1 + pad, width), min(y1 + pad, height)
    dark = (255 - sheet.gray[top:bottom, left:right]) / 255

    def derivative(dy: int, dx: int) -> np.ndarray:
        return gaussian_filter(dark, sigma, order=(dy, dx))

    fx, fy = derivative(0, 1), derivative(1, 0)
    fxx, fxy, fyy = derivative(0, 2), derivative(1, 1), derivative(2, 0)
    # The normal is the direction of the most negative second derivative.
    normal = 0.5 * np.arctan2(2 * fxy, fxx - fyy) + np.pi / 2
    nx, ny = np.cos(normal), np.sin(normal)
    root = np.hypot((fxx - fyy) / 2, fxy)
    across, along = (fxx + fyy) / 2 - root, (fxx + fyy) / 2 + root
    with np.errstate(divide="ignore", invalid="ignore"):
        offset = -(fx * nx + fy * ny) / across
    dx, dy = offset * nx, offset * ny
    rows, cols = np.mgrid[top:bottom, left:right]
    ridge = (
        sheet.ink[top:bottom, left:right]
        & (across < 0)
        & (np.abs(across) > np.abs(along))
    )
    shared_x, dx = _shared_peaks(ridge, dx, dy, axis=1)
    shared_y, dy = _shared_peaks(ridge, dy, dx, axis=0)
    keep = (
        ridge
        & ((np.abs(dx) <= 0.5) & (np.abs(dy) <= 0.5) | shared_x | shared_y)
        & (cols >= x0)
        & (cols < x1)
        & (rows >= y0)
        & (rows < y1)
    )
    xy = np.c_[cols[keep] + 0.5 + dx[keep], rows[keep] + 0.5 + dy[keep]]
    return Points(xy, normal[keep], sheet.run_width[rows[keep], cols[keep]])


def _shared_peaks(
    ridge: np.ndarray, toward: np.ndarray, aside: np.ndarray, axis: int
) -> tuple[np.ndarray, np.ndarray]:
    """Find peaks midway between two pixels, where Newton's step overshoots from both.

    Marks the first pixel of each such pair and moves its offset to the mean of
    the pair's estimates, so a two-pixel stroke on the pixel grid keeps its centre.
    """
    level = ridge & (np.abs(aside) <= 0.5) & (np.abs(toward) <= 1)
    first = tuple(slice(None, -1) if a == axis else slice(None) for a in (0, 1))
    second = tuple(slice(1, None) if a == axis else slice(None) for a in (0, 1))
    shared = np.zeros_like(ridge)
    shared[first] = (
        level[first] & (toward[first] > 0.5) & level[second] & (toward[second] < -0.5)
    )
    moved = toward.copy()
    moved[first] = np.where(
        shared[first], (toward[first] + 1 + toward[second]) / 2, toward[first]
    )
    return shared, moved


def _aligned(model_normal: np.ndarray, point_normal: np.ndarray) -> np.ndarray:
    return np.abs(np.cos(model_normal - point_normal)) >= NORMAL_COS


def _inliers(model: Circle | Line, points: Points) -> np.ndarray:
    return (np.abs(model.residual(points.xy)) <= BAND_PX) & _aligned(
        model.normal(points.xy), points.normal
    )


def _positions(model: Circle | Line, xy: np.ndarray) -> tuple[np.ndarray, float]:
    """Length along the model from its first ink, and where that first ink is."""
    if isinstance(model, Line):
        t = model.along(xy)
        origin = float(t.min())
        return t - origin, origin
    angle = np.sort(model.angle(xy))
    widest = int(np.argmax(np.diff(np.r_[angle, angle[0] + 2 * np.pi])))
    origin = float(angle[(widest + 1) % len(angle)])
    return np.mod(model.angle(xy) - origin, 2 * np.pi) * model.r, origin


@dataclass(frozen=True)
class _Scale:
    """Lengths in px that follow the sheet's line width."""

    line_width: float

    @property
    def join(self) -> float:
        """Gaps up to this join pieces; runs shorter than this are crossings."""
        return max(6.0, 2.5 * self.line_width)

    @property
    def min_support(self) -> float:
        return max(12.0, 3 * self.line_width)


def _inked(position: np.ndarray, scale: _Scale) -> tuple[list[tuple[int, int]], float]:
    """Inked runs along a model as (first, last) px, and the inked length.

    Bins are 2 px: centres along a slanted stroke lie up to 1.4 px apart.
    """
    bins = np.unique(np.floor(position / 2).astype(np.int64))
    cuts = np.flatnonzero(np.diff(bins) > scale.join / 2)
    starts, ends = np.r_[0, cuts + 1], np.r_[cuts, len(bins) - 1]
    keep = 2 * (bins[ends] - bins[starts] + 1) >= scale.join
    runs = [
        (int(2 * bins[a]), int(2 * bins[b] + 1))
        for a, b in zip(starts[keep], ends[keep])
    ]
    return runs, float(2 * (ends[keep] - starts[keep] + 1).sum())


def _score(model: Circle | Line, points: Points, scale: _Scale) -> float:
    inliers = _inliers(model, points)
    if inliers.sum() < 2:
        return 0.0
    runs, support = _inked(_positions(model, points.xy[inliers])[0], scale)
    if not runs:
        return 0.0
    if (
        isinstance(model, Circle)
        and _bow(model, (runs[-1][1] - runs[0][0] + 1) / model.r) < MIN_BOW_PX
    ):
        return 0.0  # straight ink: leave it to a line
    return support


def _closes(circle: Circle, inked_span: float, scale: _Scale) -> bool:
    """A full circle whose widest gap is a dash gap or a crossing."""
    gap = 2 * math.pi * circle.r - inked_span
    return gap <= scale.join or gap <= circle.r * math.radians(OPEN_ARC_DEG)


def _bow(circle: Circle, sweep: float) -> float:
    """How far the arc bulges from its chord, in px."""
    return circle.r * (1 - math.cos(min(sweep, math.pi) / 2))


def _hypotheses(
    points: Points, kind: Kind, diagonal: float, rng: np.random.Generator
) -> list[Circle | Line]:
    xy, normal = points.xy, points.normal
    if kind == "line":
        i, j = rng.integers(0, len(xy), (2, DRAWS))
        d = xy[j] - xy[i]
        length = np.hypot(d[:, 0], d[:, 1])
        u = d / np.maximum(length, 1e-9)[:, None]
        line_normal = np.arctan2(u[:, 0], -u[:, 1])
        ok = (
            (length >= 3)
            & _aligned(line_normal, normal[i])
            & _aligned(line_normal, normal[j])
        )
        return [Line(*xy[i[k]], *u[k]) for k in np.flatnonzero(ok)]
    s = rng.integers(0, len(xy), (DRAWS, 3))
    (x1, y1), (x2, y2), (x3, y3) = (xy[s[:, k]].T for k in range(3))
    a = x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2
    q1, q2, q3 = x1**2 + y1**2, x2**2 + y2**2, x3**2 + y3**2
    with np.errstate(divide="ignore", invalid="ignore"):
        cx = (q1 * (y2 - y3) + q2 * (y3 - y1) + q3 * (y1 - y2)) / (2 * a)
        cy = (q1 * (x3 - x2) + q2 * (x1 - x3) + q3 * (x2 - x1)) / (2 * a)
    r = np.hypot(x1 - cx, y1 - cy)
    ok = np.isfinite(r) & (r >= 1.5) & (r <= 6 * diagonal)
    for k in range(3):
        px, py = xy[s[:, k]].T
        ok &= _aligned(np.arctan2(py - cy, px - cx), normal[s[:, k]])
    return [Circle(cx[k], cy[k], r[k]) for k in np.flatnonzero(ok)]


def _best(
    hypotheses: list[Circle | Line], points: Points, scale: _Scale
) -> tuple[Circle | Line | None, float]:
    """Rank by aligned inlier count, then score the leaders by inked length."""
    counted = hypotheses[:COUNTED]
    counts = [int(_inliers(model, points).sum()) for model in counted]
    leaders = [counted[k] for k in np.argsort(counts)[::-1][:SCORED]]
    scored = [(_score(model, points, scale), model) for model in leaders]
    score, model = max(scored, key=lambda item: item[0], default=(0.0, None))
    return model, score


def _refit(model: Circle | Line, points: Points) -> Circle | Line:
    for _ in range(3):
        inliers = _inliers(model, points)
        if inliers.sum() < 5:
            return model
        xy = points.xy[inliers]
        if isinstance(model, Line):
            centre = xy.mean(axis=0)
            _, _, vt = np.linalg.svd(xy - centre, full_matrices=False)
            u = vt[0] if vt[0] @ (model.ux, model.uy) >= 0 else -vt[0]
            model = Line(*centre, *u)
        else:
            model = Circle(*_fit_circle(xy, model).x)
    return model


def _fit_circle(xy: np.ndarray, guess: Circle):
    return least_squares(
        lambda v: np.hypot(xy[:, 0] - v[0], xy[:, 1] - v[1]) - v[2],
        (guess.cx, guess.cy, guess.r),
        loss="soft_l1",
        f_scale=0.3,
    )


def _trim_arc(circle: Circle, xy: np.ndarray) -> tuple[Circle, np.ndarray]:
    """Drop the worse end while the misfit is systematic rather than noise.

    An arc between tangent lines otherwise fits a circle through all three.
    """
    position, _ = _positions(circle, xy)
    order = np.argsort(position)
    lo, hi = 0, len(order)
    while hi - lo > 8:
        fit = _fit_circle(xy[order[lo:hi]], circle)
        circle, residual = Circle(*fit.x), fit.fun
        rms = math.sqrt(np.mean(residual**2))
        noise = np.std(np.diff(residual)) / math.sqrt(2)
        if rms <= 0.08 or rms <= 2 * noise:
            break
        k = max(1, (hi - lo) // 20)
        head, tail = np.abs(residual[:k]).mean(), np.abs(residual[-k:]).mean()
        # Misfit spread along the whole arc is noise, not a tangent line to cut.
        if max(head, tail) < END_MISFIT * np.abs(residual).mean():
            break
        if head > tail:
            lo += k
        else:
            hi -= k
    return circle, order[lo:hi]


def _measure(model: Circle | Line, xy: np.ndarray, scale: _Scale) -> Stroke | None:
    position, origin = _positions(model, xy)
    runs, support = _inked(position, scale)
    if not runs:
        return None
    first, last = runs[0][0], runs[-1][1] + 1
    unit = model.r if isinstance(model, Circle) else 1.0
    if isinstance(model, Circle) and _closes(model, last - first, scale):
        last = first + 2 * math.pi * model.r
    residual = model.residual(xy)
    stroke = Stroke(
        model=model,
        start=origin + first / unit,
        end=origin + last / unit,
        support=support,
        coverage=support / (last - first),
        rms=math.sqrt(np.mean(residual**2)),
    )
    if isinstance(model, Line):
        return stroke
    # Neighbouring centres share noise: count one sample per line width.
    dx, dy = xy[:, 0] - model.cx, xy[:, 1] - model.cy
    rho = np.hypot(dx, dy)
    jacobian = np.c_[-dx / rho, -dy / rho, -np.ones(len(xy))]
    sigma2 = max(stroke.rms, 0.1) ** 2 * len(xy) / max(support / scale.line_width, 3)
    covariance = np.linalg.pinv(jacobian.T @ jacobian) * sigma2
    return replace(
        stroke,
        radius_sd=math.sqrt(covariance[2, 2]),
        centre_sd=math.sqrt(max(covariance[0, 0], covariance[1, 1])),
    )


_NO_INK = "no ink in the box."
_NO_STROKE = "no {kind} with {min_support:.0f} px of ink along it."
_SCATTERED = "the ink on the best {kind} is scattered pieces ({coverage:.0%} of it)."
_NOT_ONE = (
    "the ink does not lie on one {kind} (rms {rms:.2f} px). The box may hold "
    "an arc with straight ends or a curve that is not circular; narrow it to "
    "one segment."
)
_FLAT = (
    f"the arc bows under {MIN_BOW_PX:g} px, too little to tell from a line. "
    "Widen the box along it."
)
_SHORT = (
    "the arc is too short to fix its radius (R {r:.1f} ±{sd:.1f} px). "
    "Widen the box along the arc."
)
_WIDE = (
    "the ink is {width:.1f} px wide against {line_width:.1f} px lines, so "
    "strokes overlap and a fit would fall between them."
)
_STRAIGHT = f"no circle: the strokes here are straight to within {MIN_BOW_PX:g} px."


def _reject(stroke: Stroke, points: Points, kind: Kind, sheet: Sheet) -> str | None:
    if stroke.coverage < MIN_COVERAGE:
        return _SCATTERED.format(kind=kind, coverage=stroke.coverage)
    width = float(np.median(points.width))
    if width > 1.6 * sheet.line_width + 1:
        return _WIDE.format(width=width, line_width=sheet.line_width)
    if stroke.rms > MAX_RMS_PX:
        return _NOT_ONE.format(kind=kind, rms=stroke.rms)
    model = stroke.model
    if isinstance(model, Circle):
        if _bow(model, stroke.end - stroke.start) < MIN_BOW_PX:
            return _FLAT
        if stroke.radius_sd > MAX_RADIUS_SPREAD * model.r:
            return _SHORT.format(r=model.r, sd=stroke.radius_sd)
    return None


def _more(
    pool: Points, kind: Kind, diagonal: float, rng: np.random.Generator, scale: _Scale
) -> bool:
    """Whether another stroke beyond the cap would qualify."""
    if len(pool.xy) < 10:
        return False
    _, score = _best(_hypotheses(pool, kind, diagonal, rng), pool, scale)
    return score >= scale.min_support


def find_strokes(sheet: Sheet, box: tuple[int, int, int, int], kind: Kind) -> Finding:
    points = centre_points(sheet, box)
    if not len(points.xy):
        return Finding([], _NO_INK)
    scale = _Scale(sheet.line_width)
    rng = np.random.default_rng(0)
    diagonal = math.hypot(box[2] - box[0], box[3] - box[1])
    alive = np.ones(len(points.xy), bool)
    strokes: list[Stroke] = []
    reasons: list[str] = []
    while len(strokes) < MAX_STROKES and alive.sum() >= 10:
        index = np.flatnonzero(alive)
        pool = points.take(index)
        model, score = _best(_hypotheses(pool, kind, diagonal, rng), pool, scale)
        if model is None or score < scale.min_support:
            break
        model = _refit(model, pool)
        inliers = np.flatnonzero(_inliers(model, pool))
        if isinstance(model, Circle) and len(inliers) > 10:
            model, kept = _trim_arc(model, pool.xy[inliers])
            inliers = inliers[kept]
        # Claim this stroke's centres, aligned ones only, so crossings stay.
        claimed = _inliers(model, pool)
        claimed[inliers] = True
        if not claimed.any():
            break
        alive[index[claimed]] = False
        stroke = _measure(model, pool.xy[inliers], scale)
        if stroke is None or stroke.support < scale.min_support:
            continue
        reason = _reject(stroke, pool.take(inliers), kind, sheet)
        if reason is None:
            strokes.append(stroke)
        else:
            reasons.append(reason)
    if strokes:
        rest = points.take(np.flatnonzero(alive))
        full = len(strokes) == MAX_STROKES
        return Finding(strokes, more=full and _more(rest, kind, diagonal, rng, scale))
    # The strongest candidate comes first; a flat one means the ink is straight.
    if kind == "circle" and (not reasons or reasons[0] == _FLAT):
        lines = find_strokes(sheet, box, "line")
        if lines.strokes:
            return Finding(lines.strokes, _STRAIGHT, lines.more)
    return Finding(
        [],
        reasons[0]
        if reasons
        else _NO_STROKE.format(kind=kind, min_support=scale.min_support),
    )


_HEADER = (
    "{image}: {width}x{height} px; ink is gray below {threshold:.0f}; "
    "lines about {line_width:.0f} px wide.{scale}"
)
_SCALE = " Millimetres at {mm_per_px:g} mm/px (100 mm = {px_per_100:.1f} px)."
_ITEM = "[{index}] {kind} in box ({x0}, {y0}, {x1}, {y1}):"
_CIRCLE = (
    "  #{n} centre ({cx:.2f}, {cy:.2f}) ±{centre_sd:.2f}, "
    "R {r:.2f} ±{radius_sd:.2f} px{mm}; {extent}{cut}; {ink}, rms {rms:.2f} px"
)
_CIRCLE_MM = " = R {r:.3f} ±{sd:.3f} mm (Ø {d:.3f})"
_FULL = "full circle"
_ARC = (
    "arc {start:.0f}° to {end:.0f}° ccw, from ({x0:.1f}, {y0:.1f}) "
    "through ({xm:.1f}, {ym:.1f}) to ({x1:.1f}, {y1:.1f})"
)
_HORIZONTAL = (
    "  #{n} horizontal line y = {y:.2f}, x {a:.1f} to {b:.1f} "
    "({length:.1f} px{mm}){cut}; {ink}, rms {rms:.2f} px"
)
_VERTICAL = (
    "  #{n} vertical line x = {x:.2f}, y {a:.1f} to {b:.1f} "
    "({length:.1f} px{mm}){cut}; {ink}, rms {rms:.2f} px"
)
_SLANTED = (
    "  #{n} line ({x0:.2f}, {y0:.2f}) to ({x1:.2f}, {y1:.2f}), angle {angle:.2f}°, "
    "{length:.1f} px{mm}{cut}; {ink}, rms {rms:.2f} px"
)
_CUT_BOTH = "; the box cuts both ends"
_CUT_ONE = "; the box cuts the end at ({x:.1f}, {y:.1f})"
_LENGTH_MM = " = {mm:.3f} mm"
_SOLID = "solid"
_BROKEN = "dashed or broken: ink on {share:.0%} of it"
_SPACING = "  spacing #{a} to #{b}: {px:.2f} px{mm}"
_NOT_MEASURED = "  not measured: {reason}"
_NOTE = "  {reason}"
_MORE = "  more strokes remain in this box; narrow it to the one you need."


def _mm(px: float, mm_per_px: float | None) -> str:
    return "" if mm_per_px is None else _LENGTH_MM.format(mm=px * mm_per_px)


def _ink(stroke: Stroke) -> str:
    if stroke.coverage >= DASHED_BELOW:
        return _SOLID
    return _BROKEN.format(share=stroke.coverage)


def _line_angle(line: Line) -> float:
    return math.degrees(math.atan2(-line.uy, line.ux)) % 180


def _full(stroke: Stroke) -> bool:
    circle = isinstance(stroke.model, Circle)
    return circle and stroke.end - stroke.start >= 2 * math.pi - 1e-9


def _cut(stroke: Stroke, box: tuple[int, int, int, int]) -> str:
    """Name the ends where the stroke runs out of the box, so may go on."""
    if _full(stroke):
        return ""
    model = stroke.model
    step = EDGE_PX / model.r if isinstance(model, Circle) else EDGE_PX
    left, top, right, bottom = box

    def leaves(t: float) -> bool:
        x, y = model.point(t)
        return not (left <= x <= right and top <= y <= bottom)

    ends = [
        end
        for end, past in (
            (stroke.start, stroke.start - step),
            (stroke.end, stroke.end + step),
        )
        if leaves(past)
    ]
    if len(ends) == 2:
        return _CUT_BOTH
    if ends:
        x, y = model.point(ends[0])
        return _CUT_ONE.format(x=x, y=y)
    return ""


def _describe(
    n: int, stroke: Stroke, box: tuple[int, int, int, int], mm_per_px: float | None
) -> str:
    model = stroke.model
    common = {"n": n, "ink": _ink(stroke), "rms": stroke.rms, "cut": _cut(stroke, box)}
    if isinstance(model, Circle):
        (x0, y0), (x1, y1) = model.point(stroke.start), model.point(stroke.end)
        xm, ym = model.point((stroke.start + stroke.end) / 2)
        extent = (
            _FULL
            if _full(stroke)
            else _ARC.format(
                start=math.degrees(stroke.start) % 360,
                end=math.degrees(stroke.end) % 360,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                xm=xm,
                ym=ym,
            )
        )
        mm = (
            ""
            if mm_per_px is None
            else _CIRCLE_MM.format(
                r=model.r * mm_per_px,
                sd=stroke.radius_sd * mm_per_px,
                d=2 * model.r * mm_per_px,
            )
        )
        return _CIRCLE.format(
            cx=model.cx,
            cy=model.cy,
            centre_sd=stroke.centre_sd,
            r=model.r,
            radius_sd=stroke.radius_sd,
            mm=mm,
            extent=extent,
            **common,
        )
    (x0, y0), (x1, y1) = model.point(stroke.start), model.point(stroke.end)
    length = stroke.end - stroke.start
    angle = _line_angle(model)
    common |= {"length": length, "mm": _mm(length, mm_per_px)}
    if abs(y1 - y0) < STRAIGHT_PX:
        a, b = sorted((x0, x1))
        return _HORIZONTAL.format(y=(y0 + y1) / 2, a=a, b=b, **common)
    if abs(x1 - x0) < STRAIGHT_PX:
        a, b = sorted((y0, y1))
        return _VERTICAL.format(x=(x0 + x1) / 2, a=a, b=b, **common)
    return _SLANTED.format(x0=x0, y0=y0, x1=x1, y1=y1, angle=angle, **common)


def _spacings(strokes: list[Stroke], mm_per_px: float | None) -> list[str]:
    lines = []
    for i, a in enumerate(strokes):
        for j, b in enumerate(strokes[i + 1 :], start=i + 1):
            if not (isinstance(a.model, Line) and isinstance(b.model, Line)):
                continue
            difference = abs(_line_angle(a.model) - _line_angle(b.model))
            if min(difference, 180 - difference) >= 1:
                continue
            mid = np.array([b.model.point((b.start + b.end) / 2)])
            px = abs(float(a.model.residual(mid)[0]))
            lines.append(
                _SPACING.format(a=i + 1, b=j + 1, px=px, mm=_mm(px, mm_per_px))
            )
    return lines


def _pixel_box(box: list[float], sheet: Sheet) -> tuple[int, int, int, int]:
    height, width = sheet.gray.shape
    x0, y0 = max(math.floor(box[0]), 0), max(math.floor(box[1]), 0)
    x1, y1 = min(math.ceil(box[2]), width), min(math.ceil(box[3]), height)
    return x0, y0, x1, y1


_OUTSIDE = "the box holds no pixels of this {width}x{height} image."
_TOO_LARGE = (
    "the box covers {area:,} px; keep it under {limit:,} px around one feature."
)


def fit_report(
    image: str, path: Path, items: list["FitItem"], mm_per_px: float | None
) -> str:
    try:
        sheet = load_sheet(path)
    except (UnidentifiedImageError, OSError):
        raise ToolFeedbackError(
            f"Not a raster image: {image}. Read DXF entities with ezdxf."
        ) from None
    height, width = sheet.gray.shape
    lines = [
        _HEADER.format(
            image=image,
            width=width,
            height=height,
            threshold=sheet.threshold,
            line_width=sheet.line_width,
            scale=""
            if mm_per_px is None
            else _SCALE.format(mm_per_px=mm_per_px, px_per_100=100 / mm_per_px),
        )
    ]
    for index, item in enumerate(items, start=1):
        box = _pixel_box(item.box_px, sheet)
        x0, y0, x1, y1 = box
        lines.append(
            _ITEM.format(index=index, kind=item.kind, x0=x0, y0=y0, x1=x1, y1=y1)
        )
        if x1 <= x0 or y1 <= y0:
            lines.append(
                _NOT_MEASURED.format(reason=_OUTSIDE.format(width=width, height=height))
            )
            continue
        area = (x1 - x0) * (y1 - y0)
        if area > MAX_BOX_AREA:
            lines.append(
                _NOT_MEASURED.format(
                    reason=_TOO_LARGE.format(area=area, limit=MAX_BOX_AREA)
                )
            )
            continue
        finding = find_strokes(sheet, box, item.kind)
        if finding.reason is not None:
            template = _NOTE if finding.strokes else _NOT_MEASURED
            lines.append(template.format(reason=finding.reason))
        lines += [
            _describe(n, stroke, box, mm_per_px)
            for n, stroke in enumerate(finding.strokes, start=1)
        ]
        lines += _spacings(finding.strokes, mm_per_px)
        if finding.more:
            lines.append(_MORE)
    return "\n".join(lines)


class FitItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    box_px: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="x0, y0, x1, y1 around the stroke in the image's pixels.",
    )
    kind: Kind = Field(..., description="line, or circle for holes and arcs.")


class FitEdgeInput(BaseModel):
    image_path: str = Field(
        ..., description="An input drawing or a 1:1 crop of it, not a resized copy."
    )
    items: list[FitItem] = Field(
        ..., min_length=1, max_length=MAX_ITEMS, description="The boxes to fit."
    )
    mm_per_px: float | None = Field(
        default=None,
        gt=0,
        description=(
            "The view's scale from calculate_drawing_scale or the validated "
            "interpretation. Adds millimetres to sizes and spacings."
        ),
    )


# The report for a synthetic drawing; a test keeps its format in step with the code.
EXAMPLE_REPORT = cleandoc(
    """
    /work/inputs/view_drawing.png: 900x700 px; ink is gray below 141; lines about 4 px wide. Millimetres at 0.25 mm/px (100 mm = 400.0 px).
    [1] circle in box (300, 200, 420, 320):
      #1 centre (360.00, 260.00) ±0.02, R 39.95 ±0.01 px = R 9.988 ±0.003 mm (Ø 19.976); full circle; solid, rms 0.02 px
      #2 centre (360.00, 259.99) ±0.03, R 23.96 ±0.02 px = R 5.990 ±0.005 mm (Ø 11.979); full circle; dashed or broken: ink on 68% of it, rms 0.09 px
    [2] circle in box (120, 400, 200, 480):
      #1 centre (149.79, 450.20) ±0.20, R 30.21 ±0.22 px = R 7.552 ±0.054 mm (Ø 15.105); arc 1° to 104° ccw, from (180.0, 449.5) through (168.2, 426.2) to (142.6, 420.9); solid, rms 0.07 px
    [3] line in box (90, 95, 290, 145):
      #1 horizontal line y = 129.22, x 89.5 to 289.5 (200.0 px = 50.000 mm); the box cuts both ends; solid, rms 0.00 px
      #2 horizontal line y = 110.22, x 89.5 to 289.5 (200.0 px = 50.000 mm); the box cuts both ends; solid, rms 0.00 px
      spacing #1 to #2: 19.00 px = 4.750 mm
    """
)

_DESCRIPTION = cleandoc(
    """Fit straight lines or circles/arcs to the strokes inside boxes of a drawing image.

    Use it instead of scanning pixels with run_shell, on an input drawing or a
    1:1 crop of it. Send all boxes for one image in one call, up to {max_items}.

    It fits stroke centre lines by random sampling, so crossing leaders and text
    do not pull the fit, and it cuts an arc back where tangent lines continue it.
    Each box reports up to {max_strokes} strokes, longest ink first. A stroke's
    centre is the drawn edge.

    Positions are the file's pixels, x right and y down. Angles are as seen on
    the page: counterclockwise, 0° pointing right and 90° up. Millimetres from
    a datum pixel (xd, yd): u = (x - xd) * mm_per_px, v = (yd - y) * mm_per_px.

    Example result with mm_per_px 0.25:
    {example}

    - ± is one standard deviation of the fit, not of a wrong pick; a printed
      size within two ± agrees.
    - rms is the centres' distance from the fit: 0.02 to 0.15 px when clean.
    - "dashed or broken" means hidden or centre lines, or crossings; gaps under
      3 px read as solid.
    - "spacing" is the distance between parallel lines, such as a wall or a
      dimension's extension lines.
    - Arc ends at a tangent joint are approximate.
    - If the stroke you mean is missing or more strokes remain, narrow the box.
    """
).format(max_items=MAX_ITEMS, max_strokes=MAX_STROKES, example=EXAMPLE_REPORT)


def create_fit_edge_tool(workdir: SandboxWorkdir) -> BaseTool:
    @tool("fit_edge", args_schema=FitEdgeInput, description=_DESCRIPTION)
    def fit_edge(
        image_path: str, items: list[FitItem], mm_per_px: float | None = None
    ) -> str:
        path = resolve_image_path(workdir, image_path)
        return fit_report(image_path, path, items, mm_per_px)

    return fit_edge


def save_skeleton(image: Path, target: Path) -> None:
    """Store the sub-pixel skeleton of every stroke in a drawing, sorted by y."""
    sheet = load_sheet(image)
    height, width = sheet.gray.shape
    points = centre_points(sheet, (0, 0, width, height))
    order = np.lexsort((points.xy[:, 0], points.xy[:, 1]))
    x, y = points.xy[order].T
    # The normal is in image axes (y down); the stroke runs across it.
    angle = np.degrees(-points.normal[order] - np.pi / 2) % 180
    np.savez_compressed(
        target,
        x=x.astype(np.float32),
        y=y.astype(np.float32),
        angle=angle.astype(np.float32),
    )
