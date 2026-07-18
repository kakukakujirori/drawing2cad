"""Calibration / verification for the render3d pipeline.

Renders a handful of GT train STEPs with candidate camera parameters and
compares against the aligned GT PNGs (visual side-by-side + a blurred-IoU
proxy score used only to prune the search space -- final selection is by eye
across MULTIPLE parts at once, since any single part can have symmetries that
make the proxy metric ambiguous).

Usage:
    python src/render/calibrate_render3d.py search      # coarse+fine camera search
    python src/render/calibrate_render3d.py compare      # render calibrated cfg vs GT
    python src/render/calibrate_render3d.py smoke        # smoke-test on real-scale STEPs
"""

from __future__ import annotations

import math
import multiprocessing as mp
import sys
from pathlib import Path

import rootutils

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

from src.render.config import render3d_paths  # noqa: E402
from src.render.render3d import (  # noqa: E402
    Render3dConfig,
    _build_camera,
    _fit_transform,
    _hlr_project,
    _load_shape,
    generate_render3d,
)

GT_DIR = ROOT / "data" / "eccv2026-cad-challenge-data" / "train"
CAL_STEMS = ["000000", "000123", "004000"]
OUT_DIR = ROOT / "experiments" / "render3d_calibration"

RES = 200
BLUR_SIGMA = 2.5


# --------------------------------------------------------------------------
# Blurred-occupancy similarity (search proxy only)
# --------------------------------------------------------------------------


def gt_occupancy(png_path: Path) -> np.ndarray:
    img = Image.open(png_path).convert("L").resize((RES, int(RES * 1000 / 1400)))
    arr = np.array(img)
    occ = (arr < 250).astype(np.float32)
    return gaussian_filter(occ, BLUR_SIGMA)


def candidate_occupancy(visible, hidden, canvas_w=1400, canvas_h=1000, margin_frac=0.08, res=RES):
    all_lines = visible + hidden
    if not all_lines:
        return None
    tf = _fit_transform(all_lines, canvas_w, canvas_h, margin_frac)
    h = int(res * canvas_h / canvas_w)
    img = Image.new("L", (res, h), 0)
    draw = ImageDraw.Draw(img)
    sx = res / canvas_w
    sy = h / canvas_h
    for line in all_lines:
        pts = [(x * sx, y * sy) for x, y in (tf(px, py) for px, py in line)]
        if len(pts) >= 2:
            draw.line(pts, fill=255, width=1)
    occ = (np.array(img) > 0).astype(np.float32)
    return gaussian_filter(occ, BLUR_SIGMA)


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def score_cfg(stem: str, cfg: Render3dConfig) -> float:
    step = GT_DIR / "target_step" / f"{stem}.step"
    gt_png = GT_DIR / "render_3d" / "hlg_perspective" / f"{stem}.png"
    gt_occ = gt_occupancy(gt_png)
    shape = _load_shape(step)
    camera = _build_camera(shape, cfg)
    vis, hid = _hlr_project(shape, camera, cfg, deflection_scale=8.0)
    occ = candidate_occupancy(vis, hid, cfg.canvas_w, cfg.canvas_h, cfg.margin_frac)
    if occ is None:
        return 0.0
    return ncc(gt_occ, occ)


def score_cfg_multi(stems: list[str], cfg: Render3dConfig) -> float:
    return sum(score_cfg(s, cfg) for s in stems) / len(stems)


# --------------------------------------------------------------------------
# Search: enumerate SolidWorks-standard candidate families (world Y-up,
# confirmed by src/render/techdraw.py's IoU-verified front/top/right frames),
# each combined with a handful of roll values, scored against ALL calibration
# parts jointly (kills the single-part symmetry degeneracy).
# --------------------------------------------------------------------------


def octant_dirs():
    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                yield (sx, sy, sz)


def run_search():
    print(f"scoring octant x roll candidates jointly over {CAL_STEMS}")
    best = None
    results = []
    for ed in octant_dirs():
        for roll in range(0, 360, 30):
            cfg = Render3dConfig(eye_dir=ed, world_up=(0.0, 1.0, 0.0),
                                  dist_factor=8.0, focus_factor=0.15, roll_deg=float(roll))
            s = score_cfg_multi(CAL_STEMS, cfg)
            results.append((s, ed, roll))
            if best is None or s > best[0]:
                best = (s, ed, roll)
                print("new best", best)
    results.sort(reverse=True)
    print("top 10:")
    for r in results[:10]:
        print(r)
    return results


# --------------------------------------------------------------------------
# Isolated pipeline run (OCC HLR can hang in native code on rare parts; the
# shipped generate_render3d() is pure/synchronous by design, so calibration
# loops that run it over many/unknown STEPs must add their own timeout guard).
# --------------------------------------------------------------------------


def _worker(step, paths, cfg, q):
    try:
        info = generate_render3d(Path(step), paths, cfg)
        q.put(("ok", info))
    except Exception as exc:  # noqa: BLE001
        q.put(("err", f"{type(exc).__name__}: {exc}"))


def run_isolated(step, paths, cfg, timeout=60):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(str(step), paths, cfg, q), daemon=True)
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join(5)
        if p.is_alive():
            p.kill()
            p.join()
        return "timeout", None
    try:
        status, payload = q.get_nowait()
    except Exception:  # noqa: BLE001
        return "crash", None
    return status, payload


def render_compare(cfg: Render3dConfig, stems=CAL_STEMS, out_dir: Path = OUT_DIR, timeout=60):
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        step = GT_DIR / "target_step" / f"{stem}.step"
        paths = render3d_paths(out_dir, stem)
        status, info = run_isolated(step, paths, cfg, timeout=timeout)
        print(stem, status, info)
        if status != "ok":
            continue
        gt = Image.open(GT_DIR / "render_3d" / "hlg_perspective" / f"{stem}.png")
        ours = Image.open(paths.hlg) if paths.hlg.exists() else Image.new("RGB", (1400, 1000), "white")
        sheet = Image.new("RGB", (1400 * 2, 1000), "white")
        sheet.paste(gt, (0, 0))
        sheet.paste(ours, (1400, 0))
        sheet.save(out_dir / f"compare_hlg_{stem}.png")

        gt_s = Image.open(GT_DIR / "render_3d" / "transparent_shaded_edges_perspective" / f"{stem}.png")
        ours_s = Image.open(paths.shaded) if paths.shaded.exists() else Image.new("RGB", (1400, 1000), "white")
        sheet_s = Image.new("RGB", (1400 * 2, 1000), "white")
        sheet_s.paste(gt_s, (0, 0))
        sheet_s.paste(ours_s, (1400, 0))
        sheet_s.save(out_dir / f"compare_shaded_{stem}.png")

        gt_t = Image.open(GT_DIR / "render_3d" / "hlg_translucent_faces_perspective" / f"{stem}.png")
        ours_t = Image.open(paths.hlg_translucent) if paths.hlg_translucent.exists() else Image.new("RGB", (1400, 1000), "white")
        sheet_t = Image.new("RGB", (1400 * 2, 1000), "white")
        sheet_t.paste(gt_t, (0, 0))
        sheet_t.paste(ours_t, (1400, 0))
        sheet_t.save(out_dir / f"compare_translucent_{stem}.png")


def run_smoke(real_dir: Path, n: int = 3, timeout: float = 120.0):
    steps = sorted(real_dir.glob("*.step"))[:n]
    out_dir = OUT_DIR / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = Render3dConfig()
    for step in steps:
        paths = render3d_paths(out_dir, step.stem)
        status, info = run_isolated(step, paths, cfg, timeout=timeout)
        print(step.stem, status, info)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "compare"
    if mode == "search":
        run_search()
    elif mode == "compare":
        render_compare(Render3dConfig())
    elif mode == "smoke":
        real_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "experiments" / "stage_z2c_train"
        run_smoke(real_dir)
    else:
        print(f"unknown mode {mode}")
