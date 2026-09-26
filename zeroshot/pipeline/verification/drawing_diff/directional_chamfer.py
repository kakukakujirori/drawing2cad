"""Robust CAD-line registration, mapping CAD pixels -> drawing pixels.

CPU-only, no learned features, no annotation removal. A grid search over
similarity scale and rotation, exhaustive in translation, is followed by local
similarity/affine/homography refinement. This is heuristic optimization, NOT a
certificate of global optimality.

Distances and robust scales are in original DRAWING pixels. Ink is True.
Direction banks approximate directional Chamfer; this is not a reimplementation
of the acceleration scheme in the original FDCM paper.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import minimize
from scipy.spatial import cKDTree

if __package__:
    from .image_ops import distance_map, foreground_mask
else:  # Preserve direct-file CLI execution alongside package imports.
    from image_ops import distance_map, foreground_mask


@dataclass
class Config:
    model: Literal["similarity", "affine", "homography"] = "similarity"
    loss: Literal["welsch", "cauchy", "squared"] = "welsch"
    # Object extent / drawing's longest image dimension, not pixel scale.
    relative_scale: tuple[float, float] = (0.45, 0.90)
    rotation_degrees: float = 10.0
    tau: float = 3.0
    orientation_weight: float = 3.0  # drawing pixels / radian; 0 disables it
    direction_bins: int = 12
    balance_power: float = 0.0  # 0: equal point weights; 1: equal direction bins
    pyramid: tuple[float, ...] = (0.5, 1.0)
    # Grid of the global search: log scale and degrees.
    scale_step: float = 0.02
    rotation_step: float = 1.0
    top_k: int = 4
    regularization: float = 0.01
    anisotropy_limit: float = 0.06  # log singular stretch parameter, approximately
    shear_limit: float = 0.10
    perspective_limit: float = 0.15  # normalized source coordinates

    def validate(self) -> None:
        self.relative_scale = tuple(self.relative_scale)
        self.pyramid = tuple(self.pyramid)
        if self.model not in ("similarity", "affine", "homography"):
            raise ValueError("Unsupported model")
        if self.loss not in ("welsch", "cauchy", "squared"):
            raise ValueError("Unsupported loss")
        if not 0 < self.relative_scale[0] < self.relative_scale[1]:
            raise ValueError("relative_scale must be positive and increasing")
        if self.tau <= 0 or self.orientation_weight < 0 or self.regularization < 0:
            raise ValueError("Invalid loss/regularization scale")
        if self.direction_bins < 4 or not 0 <= self.balance_power <= 1:
            raise ValueError("Invalid orientation settings")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.scale_step <= 0 or self.rotation_step <= 0:
            raise ValueError("Grid steps must be positive")
        if (
            not self.pyramid
            or self.pyramid[-1] != 1.0
            or any(q <= 0 or q > 1 for q in self.pyramid)
        ):
            raise ValueError("pyramid must contain positive factors <= 1 and end at 1")
        if tuple(sorted(self.pyramid)) != self.pyramid:
            raise ValueError("pyramid must be ascending")
        if not 0 < self.rotation_degrees <= 180:
            raise ValueError("rotation_degrees must be in (0, 180]")
        if min(self.anisotropy_limit, self.shear_limit, self.perspective_limit) <= 0:
            raise ValueError("Transform limits must be positive")
        if self.perspective_limit >= 0.8:
            raise ValueError(
                "perspective_limit must be < 0.8 to avoid projective poles"
            )


def validate_options(model: str, options: dict[str, Any]) -> dict[str, Any]:
    """Validate common-API overrides and return the effective Chamfer settings."""
    # Keep all defaults and search constraints in the optimizer's existing Config.
    try:
        config = Config(model=model, **options)
    except TypeError as error:
        raise ValueError(f"invalid Chamfer options: {error}") from error
    config.validate()
    settings = asdict(config)
    # Reject NaN/Inf even in sequence settings before entering native optimization.
    for value in settings.values():
        for number in value if isinstance(value, (list, tuple)) else (value,):
            if isinstance(number, (int, float)) and not math.isfinite(number):
                raise ValueError("alignment settings must be finite")
    return settings


def estimate_transform(
    drawing_rgb: np.ndarray,
    projection_rgb: np.ndarray,
    *,
    model: str,
    options: dict[str, Any],
    runtime: Any,
    diagnostics: dict[str, Any],
) -> tuple[np.ndarray, Literal["ok", "uncertain"], list[str]]:
    """Return (drawing→projection integer-centre H, status, warnings).

    Called by align() with validated options; runtime is unused by this CPU backend.
    Diagnostics are updated in place so the caller can report numerical failures.
    """
    # Preserve CAD→drawing optimization to avoid fitting annotation lines to CAD.
    drawing_gray = cv2.cvtColor(drawing_rgb, cv2.COLOR_RGB2GRAY)
    projection_gray = cv2.cvtColor(projection_rgb, cv2.COLOR_RGB2GRAY)
    config = Config(**(options | {"model": model}))
    raw = register(projection_gray, drawing_gray, config)
    diagnostics["chamfer"] = raw
    diagnostics["chamfer_distance_frame"] = "drawing pixels"
    # Adapt the optimizer's direction here; the common layer sees only drawing→CAD.
    matrix = np.linalg.inv(np.asarray(raw["H_source_to_drawing"], dtype=float))
    warnings = list(raw.get("warnings", ()))
    status: Literal["ok", "uncertain"] = "ok"
    # Budget exhaustion leaves an inspectable candidate, with uncertain convergence.
    if not any(run["success"] for run in raw["optimizer_runs"]):
        status = "uncertain"
        warnings.append(
            "Chamfer search exhausted its budget without reported convergence."
        )
    return matrix, status, warnings


def read_gray(path: str | Path) -> np.ndarray:
    """Read grayscale, RGB or RGBA; composite transparency onto white."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.dtype != np.uint8:
        raise ValueError("Please supply an 8-bit image")
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        alpha = image[..., 3:4].astype(np.float64) / 255.0
        image = np.rint(image[..., :3] * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def features(ink: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stroke pixels and unsigned local tangent angles, in [0, pi)."""
    a = ink.astype(np.float64)
    gx = gaussian_filter(a, 1.0, order=(0, 1))
    gy = gaussian_filter(a, 1.0, order=(1, 0))
    xx = gaussian_filter(gx * gx, 1.5)
    yy = gaussian_filter(gy * gy, 1.5)
    xy = gaussian_filter(gx * gy, 1.5)
    angle = (0.5 * np.arctan2(2 * xy, xx - yy) + np.pi / 2) % np.pi
    y, x = np.nonzero(ink)
    return np.column_stack((x, y)).astype(np.float64), angle[y, x]


class DistanceBank:
    """Conservative point-rasterized pyramid: thin strokes are not averaged away."""

    def __init__(
        self,
        shape: tuple[int, int],
        points: np.ndarray,
        angles: np.ndarray,
        factor: float,
        cfg: Config,
    ):
        self.factor = factor
        self.k = cfg.direction_bins if cfg.orientation_weight > 0 else 1
        h, w = np.ceil((np.array(shape) - 1) * factor).astype(int) + 1
        xy = np.rint(points * factor).astype(int)
        self.maps = np.empty((self.k, h, w), dtype=np.float64)
        bins = np.rint(angles * self.k / np.pi).astype(int) % self.k
        for b in range(self.k):
            m = np.zeros((h, w), dtype=bool)
            chosen = xy[bins == b]
            m[chosen[:, 1], chosen[:, 0]] = True
            self.maps[b] = distance_map(m) / factor
        if self.k > 1:
            raw = self.maps.copy()
            for b in range(self.k):
                diff = np.abs(np.arange(self.k) - b)
                diff = np.minimum(diff, self.k - diff) * np.pi / self.k
                self.maps[b] = np.min(
                    raw + cfg.orientation_weight * diff[:, None, None], axis=0
                )
        self.shape = shape

    def sample(self, xy: np.ndarray, angle: np.ndarray) -> np.ndarray:
        x, y = (xy * self.factor).T
        # Clamp only for sampling, then ADD an outside penalty. Never silently
        # drop off-image points or pad the distance field with zeros.
        h, w = self.maps.shape[1:]
        xc = np.clip(x, 0, w - 1)
        yc = np.clip(y, 0, h - 1)
        if self.k == 1:
            d = map_coordinates(self.maps[0], [yc, xc], order=1, prefilter=False)
        else:
            f = (angle % np.pi) * self.k / np.pi
            b = np.floor(f).astype(int)
            alpha = f - b
            d0 = map_coordinates(self.maps, [b, yc, xc], order=1, prefilter=False)
            d1 = map_coordinates(
                self.maps, [(b + 1) % self.k, yc, xc], order=1, prefilter=False
            )
            d = (1 - alpha) * d0 + alpha * d1
        oh, ow = self.shape
        outside = np.hypot(
            xy[:, 0] - np.clip(xy[:, 0], 0, ow - 1),
            xy[:, 1] - np.clip(xy[:, 1], 0, oh - 1),
        )
        return d + outside


def apply_homography(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    z = np.column_stack((xy, np.ones(len(xy)))) @ H.T
    if np.any(np.abs(z[:, 2]) < 1e-10):
        raise ValueError("Homography has a pole on evaluated points")
    return z[:, :2] / z[:, 2:3]


class Objective:
    def __init__(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        cfg: Config,
        source_visibility: np.ndarray | None = None,
    ):
        self.cfg = cfg
        self.src_shape, self.dst_shape = src.shape, dst.shape
        self.Ls, self.Lt = float(max(src.shape)), float(max(dst.shape))
        self.center = (np.array(src.shape[::-1]) - 1) / 2
        p, angle = features(src)
        if source_visibility is not None:
            if source_visibility.shape != src.shape:
                raise ValueError("source_visibility has the wrong shape")
            valid = source_visibility[p[:, 1].astype(int), p[:, 0].astype(int)] > 0
            p, angle = p[valid], angle[valid]
        if len(p) < 8:
            raise ValueError("Too few visible source pixels")
        self.visible_points = p.copy()
        self.points = p
        self.u = (p - self.center) / self.Ls
        self.tangents = np.column_stack((np.cos(angle), np.sin(angle)))
        bins = (
            np.rint(angle * cfg.direction_bins / np.pi).astype(int) % cfg.direction_bins
        )
        weights = np.maximum(
            np.bincount(bins, minlength=cfg.direction_bins)[bins], 1
        ) ** (-cfg.balance_power)
        self.weights = weights / weights.sum()
        target_points, target_angles = features(dst)
        self.banks = [
            DistanceBank(dst.shape, target_points, target_angles, q, cfg)
            for q in cfg.pyramid
        ]
        h, w = src.shape
        self.corners = np.array(
            [[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]]
        )

    @staticmethod
    def unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = np.zeros(8)
        p[: len(params)] = params
        tx, ty, ls, theta, a, shear, px, py = p
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        U = np.array([[np.exp(a), shear], [0.0, np.exp(-a)]])
        return np.exp(ls) * (R @ U), np.array([tx, ty]), np.array([px, py])

    def transform(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        A, t, proj = self.unpack(params)
        z = 1 + self.u @ proj
        Au = self.u @ A.T
        v = self.tangents @ A.T
        # Derivative of A*u/(1+p.u), applied to each source tangent.
        tv = (v * z[:, None] - Au * (self.tangents @ proj)[:, None]) / (z * z)[:, None]
        xy = self.Lt * (Au / z[:, None] + t)
        return xy, np.arctan2(tv[:, 1], tv[:, 0]) % np.pi

    def matrix(self, params: np.ndarray) -> np.ndarray:
        A, t, proj = self.unpack(params)
        Hn = np.eye(3)
        Hn[:2, :2] = A + np.outer(t, proj)
        Hn[:2, 2] = t
        Hn[2, :2] = proj
        Ns = np.array(
            [
                [1 / self.Ls, 0, -self.center[0] / self.Ls],
                [0, 1 / self.Ls, -self.center[1] / self.Ls],
                [0, 0, 1],
            ]
        )
        H = np.diag([self.Lt, self.Lt, 1.0]) @ Hn @ Ns
        return H / H[2, 2]

    def __call__(self, params: np.ndarray, level: int = -1) -> float:
        xy, angle = self.transform(params)
        d = self.banks[level].sample(xy, angle)
        z = (d / self.cfg.tau) ** 2
        if self.cfg.loss == "welsch":
            rho = -np.expm1(-0.5 * z)  # bounded, smooth 0..1
        elif self.cfg.loss == "cauchy":
            rho = np.log1p(z)
        else:
            rho = z
        score = float(self.weights @ rho)
        if len(params) > 4:
            limits = np.array(
                [
                    self.cfg.anisotropy_limit,
                    self.cfg.shear_limit,
                    self.cfg.perspective_limit,
                    self.cfg.perspective_limit,
                ]
            )
            score += self.cfg.regularization * np.sum(
                (params[4:] / limits[: len(params) - 4]) ** 2
            )
        return float(score)

    def bounds(self, n: int) -> list[tuple[float, float]]:
        cfg = self.cfg
        h, w = self.dst_shape
        bounds = [
            (0, (w - 1) / self.Lt),
            (0, (h - 1) / self.Lt),
            tuple(np.log(cfg.relative_scale)),
            tuple(np.deg2rad([-cfg.rotation_degrees, cfg.rotation_degrees])),
            (-cfg.anisotropy_limit, cfg.anisotropy_limit),
            (-cfg.shear_limit, cfg.shear_limit),
            (-cfg.perspective_limit, cfg.perspective_limit),
            (-cfg.perspective_limit, cfg.perspective_limit),
        ]
        return bounds[:n]

    def diverse(
        self, candidates: list[np.ndarray], level: int, k: int
    ) -> list[np.ndarray]:
        kept: list[np.ndarray] = []
        geometries: list[np.ndarray] = []
        for p in sorted(candidates, key=lambda q: self(q, level)):
            g = apply_homography(self.matrix(p), self.corners)
            if all(
                np.sqrt(np.mean(np.sum((g - old) ** 2, axis=1))) > 2.0
                for old in geometries
            ):
                kept.append(p.copy())
                geometries.append(g)
            if len(kept) >= k:
                break
        return kept

    def refine(self, p: np.ndarray, level: int) -> tuple[np.ndarray, bool]:
        """Improve p locally, and say whether the search converged.

        Bounded windows keep it from jumping to an unrelated repeated line.
        """
        radius = np.array(
            [
                0.045,
                0.045,
                0.10,
                np.deg2rad(3),
                self.cfg.anisotropy_limit,
                self.cfg.shear_limit,
                self.cfg.perspective_limit,
                self.cfg.perspective_limit,
            ]
        )[: len(p)]
        bounds = [
            (max(lo, v - r), min(hi, v + r))
            for (lo, hi), v, r in zip(self.bounds(len(p)), p, radius)
        ]
        fun = lambda v: self(v, level)
        result = minimize(
            fun,
            p,
            method="Nelder-Mead",
            bounds=bounds,
            options={
                "maxfev": 1500 if len(p) == 4 else 3000,
                "xatol": 1e-6,
                "fatol": 1e-7,
            },
        )
        better = np.isfinite(result.fun) and result.fun < fun(p)
        return (result.x if better else p.copy()), bool(result.success)


# The global search runs on a coarse copy, where tau widens with the pixel.
_COARSE_FACTOR = 0.25
_COARSE_KEPT = 32


def grid_search(objective: Objective, target: np.ndarray) -> list[np.ndarray]:
    """The best translation of each grid scale and rotation, best first.

    Translation is exhaustive by correlation, so the result is deterministic.
    """
    cfg, q = objective.cfg, _COARSE_FACTOR
    points, angles = features(target)
    distance = DistanceBank(
        target.shape, points, angles, q, replace(cfg, orientation_weight=0.0)
    ).maps[0]
    loss = (-np.expm1(-0.5 * (distance * q / cfg.tau) ** 2)).astype(np.float32)
    h, w = target.shape
    u = (objective.visible_points - objective.center) / objective.Ls
    low, high = np.log(cfg.relative_scale)
    limit = cfg.rotation_degrees
    found = []
    for log_scale in np.arange(low, high + 1e-9, cfg.scale_step):
        for degrees in np.arange(-limit, limit + 1e-9, cfg.rotation_step):
            theta = np.deg2rad(degrees)
            c, s = np.cos(theta), np.sin(theta)
            offsets = q * objective.Lt * np.exp(log_scale) * (u @ [[c, s], [-s, c]])
            corner = np.floor(offsets.min(axis=0))
            cells = np.rint(offsets - corner).astype(int)
            template = np.zeros(cells.max(axis=0)[::-1] + 1, np.float32)
            np.add.at(template, (cells[:, 1], cells[:, 0]), 1.0 / len(cells))
            th, tw = template.shape
            # Placing the template off the drawing costs the maximum loss.
            padded = cv2.copyMakeBorder(
                loss, th, th, tw, tw, cv2.BORDER_CONSTANT, value=1.0
            )
            score = cv2.matchTemplate(padded, template, cv2.TM_CCORR)
            # Template corner (x, y) puts the part centre at q*Lt*t below.
            ys, xs = np.mgrid[: score.shape[0], : score.shape[1]]
            tx = (xs - tw - corner[0]) / (q * objective.Lt)
            ty = (ys - th - corner[1]) / (q * objective.Lt)
            inside = (
                (tx >= 0)
                & (tx <= (w - 1) / objective.Lt)
                & (ty >= 0)
                & (ty <= (h - 1) / objective.Lt)
            )
            score[~inside] = np.inf
            y, x = np.unravel_index(np.argmin(score), score.shape)
            found.append(
                (score[y, x], np.array([tx[y, x], ty[y, x], log_scale, theta]))
            )
    found.sort(key=lambda item: item[0])
    return [params for _, params in found[:_COARSE_KEPT]]


def register(
    source_gray: np.ndarray,
    drawing_gray: np.ndarray,
    cfg: Config | None = None,
    source_visibility: np.ndarray | None = None,
) -> dict:
    """Return H, hypotheses and diagnostics. source_visibility is known, not inferred.

    H always maps original source pixel coordinates into original drawing pixels.
    source_visibility is an optional same-size boolean/0-1 mask (True = observable).
    A small score is NOT a probability that the CAD model is correct.
    """
    cfg = cfg or Config()
    cfg.validate()
    start = time.perf_counter()
    source, target = foreground_mask(source_gray), foreground_mask(drawing_gray)
    objective = Objective(source, target, cfg, source_visibility)
    candidates = objective.diverse(grid_search(objective, target), 0, cfg.top_k)
    converged: list[bool] = []
    for level in range(len(cfg.pyramid)):
        refined = [objective.refine(p, level) for p in candidates]
        candidates = objective.diverse([p for p, _ in refined], level, cfg.top_k)
        converged = [ok for _, ok in refined]
    n = {"similarity": 4, "affine": 6, "homography": 8}[cfg.model]
    if n > 4:
        refined = [objective.refine(np.pad(p, (0, n - 4)), -1) for p in candidates]
        candidates = [p for p, _ in refined]
        converged = [ok for _, ok in refined]
    candidates.sort(key=objective)
    p = candidates[0]
    H = objective.matrix(p)
    xy, _ = objective.transform(p)
    target_dt = distance_map(target)
    geometric_distance = geometric_samples(target_dt, xy)
    inside = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] <= target.shape[1] - 1)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] <= target.shape[0] - 1)
    )
    return {
        "H_source_to_drawing": H.tolist(),
        "parameters_normalized": p.tolist(),
        "loss": objective(p),
        "median_geometric_distance_px": float(np.median(geometric_distance)),
        "fraction_within_2px": float(np.mean(geometric_distance <= 2)),
        "outside_fraction": float(1 - inside.mean()),
        "source_visibility_applied": source_visibility is not None,
        # Keep diagnostics checkpointable: msgpack cannot encode NumPy scalars.
        "visible_source_ink_fraction": float(
            len(objective.visible_points) / np.count_nonzero(source)
        ),
        "seconds": time.perf_counter() - start,
        "config": asdict(cfg),
        "optimizer_runs": [{"success": ok} for ok in converged],
        "hypotheses": [
            {"loss": objective(v), "H": objective.matrix(v).tolist()}
            for v in candidates
        ],
        # General alignment caveats belong in the shared feedback legend, once per batch.
        "warnings": [],
    }


def geometric_samples(dt: np.ndarray, xy: np.ndarray) -> np.ndarray:
    h, w = dt.shape
    xc = np.clip(xy[:, 0], 0, w - 1)
    yc = np.clip(xy[:, 1], 0, h - 1)
    return map_coordinates(dt, [yc, xc], order=1, prefilter=False) + np.hypot(
        xy[:, 0] - xc, xy[:, 1] - yc
    )


def rasterize(
    source_ink: np.ndarray, H: np.ndarray, shape: tuple[int, int], supersample: int = 4
) -> np.ndarray:
    """Display-only: normalize stroke width to 1 drawing pixel, preserve thin lines."""
    h, w = shape
    canvas = np.full((h * supersample, w * supersample), 255, np.uint8)
    contours, _ = cv2.findContours(
        source_ink.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )
    for contour in contours:
        points = apply_homography(H, contour[:, 0, :])
        points = np.clip(points * supersample * 16, -(2**29), 2**29).astype(np.int32)
        cv2.polylines(
            canvas,
            [points],
            True,
            0,
            thickness=supersample,
            lineType=cv2.LINE_AA,
            shift=4,
        )
    return cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)


def save_outputs(
    result: dict, source_gray: np.ndarray, target_gray: np.ndarray, output: str | Path
) -> None:
    """Save raw (unrobustified) two-way residuals, and grayscale review panels."""
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    H = np.array(result["H_source_to_drawing"])
    source, target = foreground_mask(source_gray), foreground_mask(target_gray)
    ps = np.column_stack(np.nonzero(source)[::-1]).astype(float)
    pt = np.column_stack(np.nonzero(target)[::-1]).astype(float)
    warped_points = apply_homography(H, ps)
    fwd = geometric_samples(distance_map(target), warped_points)
    # Exact nearest distance to TRANSFORMED source samples, in drawing pixels.
    # Unlike warping a precomputed DT, this is valid for affine/projective H too.
    reverse = cKDTree(warped_points).query(pt, workers=1)[0]
    residual_map = np.full(target.shape, np.nan, np.float32)
    residual_map[pt[:, 1].astype(int), pt[:, 0].astype(int)] = reverse
    np.savez_compressed(
        out / "residuals.npz",
        source_xy=ps,
        warped_source_xy=warped_points,
        cad_to_drawing_distance_px=fwd,
        drawing_xy=pt,
        drawing_to_cad_distance_px=reverse,
        drawing_residual_map=residual_map,
        H=H,
    )
    aligned = rasterize(source, H, target.shape)
    # Light-gray input, black registered CAD; no artificial CAD deformation.
    background = np.rint(160 + target_gray.astype(float) * (95 / 255)).astype(np.uint8)
    overlay = np.minimum(background, aligned)
    cv2.imwrite(str(out / "aligned.png"), aligned)
    cv2.imwrite(str(out / "overlay.png"), overlay)
    native = cv2.warpPerspective(
        source_gray, H, target.shape[::-1], flags=cv2.INTER_LINEAR, borderValue=255
    )
    cv2.imwrite(str(out / "aligned_native.png"), native)
    reverse_image = np.full(target.shape, 255, np.uint8)
    reverse_image[pt[:, 1].astype(int), pt[:, 0].astype(int)] = np.uint8(
        225 - 225 * np.clip(reverse / 12, 0, 1)
    )
    cv2.imwrite(str(out / "drawing_residual.png"), reverse_image)
    # Side-by-side rather than alpha blending alone, so coincident/absent edges
    # can be distinguished. This presentation is not used in optimization.
    titles = [
        "Input (dimensions retained)",
        "Registered CAD",
        "Overlay: CAD black / input gray",
        "Input -> CAD residual (dark = far)",
    ]
    panels = [target_gray, aligned, overlay, reverse_image]
    enlarged = [
        cv2.resize(im, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        for im in panels
    ]
    ph, pw = enlarged[0].shape
    board = np.full((ph + 52, 4 * (pw + 16) + 16), 255, np.uint8)
    for i, (im, title) in enumerate(zip(enlarged, titles)):
        x = 16 + i * (pw + 16)
        board[44 : 44 + ph, x : x + pw] = im
        cv2.putText(
            board, title, (x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, 0, 1, cv2.LINE_AA
        )
    cv2.imwrite(str(out / "comparison.png"), board)
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="CAD render")
    parser.add_argument("--target", required=True, help="Drawing / scan")
    parser.add_argument("--out", default="alignment_result")
    parser.add_argument(
        "--model", choices=["similarity", "affine", "homography"], default="similarity"
    )
    parser.add_argument(
        "--loss", choices=["welsch", "cauchy", "squared"], default="welsch"
    )
    parser.add_argument(
        "--relative-scale",
        type=float,
        nargs=2,
        default=[0.45, 0.9],
        metavar=("MIN", "MAX"),
    )
    parser.add_argument("--rotation-degrees", type=float, default=10.0)
    parser.add_argument("--tau", type=float, default=3.0)
    parser.add_argument("--orientation-weight", type=float, default=3.0)
    parser.add_argument("--balance-power", type=float, default=0.0)
    parser.add_argument(
        "--source-visibility",
        help="Known visibility: white=use, black=ignore, same size as CAD",
    )
    args = parser.parse_args()
    cfg = Config(
        model=args.model,
        loss=args.loss,
        relative_scale=tuple(args.relative_scale),
        rotation_degrees=args.rotation_degrees,
        tau=args.tau,
        orientation_weight=args.orientation_weight,
        balance_power=args.balance_power,
    )
    source, target = read_gray(args.source), read_gray(args.target)
    visibility = (
        read_gray(args.source_visibility) > 127 if args.source_visibility else None
    )
    result = register(source, target, cfg, visibility)
    save_outputs(result, source, target, args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
