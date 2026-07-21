"""Calibration / verification for the techdraw pipeline.

Runs generate_techdraw on train GT STEPs (each in an isolated, timeout-guarded
subprocess -- OCC HLR can hang in native code) and compares the output against
the aligned GT drawing:

  * DXF entity-type / linetype count deltas
  * view-arrangement correctness (which projected view lands where)
  * per-view similarity-normalised raster IoU (GT STEPs are scale-normalised, so
    absolute overlay is impossible -- comparison is per-view up to scale+shift)
  * side-by-side comparison PNGs (ours over GT)

Also smoke-tests real (original-scale, mm) STEP inputs.

Usage:
    python src/data/render/calibrate_techdraw.py [--n 24] [--stems 000000,...] \
        [--out DIR] [--timeout 60] [--real-dir DIR --real-n 3]
"""

from __future__ import annotations

import argparse
import io
import math
import multiprocessing as mp
import random
from collections import Counter
from pathlib import Path

import rootutils

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import cairosvg  # noqa: E402
import ezdxf  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from scipy.ndimage import binary_dilation  # noqa: E402

from src.data.render.config import techdraw_paths  # noqa: E402

GT_DIR = ROOT / "data" / "eccv2026-cad-challenge-data" / "train"


# --------------------------------------------------------------------------
# GT DXF parsing / clustering / raster IoU
# --------------------------------------------------------------------------


def dxf_polylines(path, layer="0"):
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()
    out = []
    for e in msp:
        if str(e.dxf.get("layer", "")) != layer:
            continue
        lt = str(e.dxf.get("linetype", "BYLAYER"))
        t = e.dxftype()
        try:
            if t == "LINE":
                out.append(
                    ([(e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)], lt)
                )
            elif t == "CIRCLE":
                c, r = e.dxf.center, e.dxf.radius
                out.append(
                    (
                        [
                            (
                                c.x + r * math.cos(2 * math.pi * i / 64),
                                c.y + r * math.sin(2 * math.pi * i / 64),
                            )
                            for i in range(65)
                        ],
                        lt,
                    )
                )
            elif t == "ARC":
                c, r = e.dxf.center, e.dxf.radius
                a0, a1 = math.radians(e.dxf.start_angle), math.radians(e.dxf.end_angle)
                if a1 < a0:
                    a1 += 2 * math.pi
                n = max(4, int((a1 - a0) / (2 * math.pi) * 64))
                out.append(
                    (
                        [
                            (
                                c.x + r * math.cos(a0 + (a1 - a0) * i / n),
                                c.y + r * math.sin(a0 + (a1 - a0) * i / n),
                            )
                            for i in range(n + 1)
                        ],
                        lt,
                    )
                )
            elif t == "ELLIPSE":
                out.append(([(p.x, p.y) for p in e.flattening(0.02)], lt))
            elif t == "SPLINE":
                out.append(([(p[0], p[1]) for p in e.flattening(0.02)], lt))
            elif t == "LWPOLYLINE":
                out.append(([(p[0], p[1]) for p in e.get_points()], lt))
        except Exception:  # noqa: BLE001
            continue
    return out


def _bbox(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def cluster_views(entities, gap=12.0):
    n = len(entities)
    if n == 0:
        return []
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    boxes = [_bbox(pts) for pts, _ in entities]
    for i in range(n):
        for j in range(i + 1, n):
            ax0, ay0, ax1, ay1 = boxes[i]
            bx0, by0, bx1, by1 = boxes[j]
            dx = max(0, max(ax0 - bx1, bx0 - ax1))
            dy = max(0, max(ay0 - by1, by0 - ay1))
            if dx <= gap and dy <= gap:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(entities[i])
    clusters = list(groups.values())

    def extent(cl):
        allpts = [p for pts, _ in cl for p in pts]
        x0, y0, x1, y1 = _bbox(allpts)
        return (x1 - x0) * (y1 - y0) + len(cl)

    clusters.sort(key=extent, reverse=True)
    return clusters


def cluster_bbox(cluster):
    return _bbox([p for pts, _ in cluster for p in pts])


def label_views(clusters):
    """Return {front,top,right: cluster} for the 3 largest clusters, or None."""
    cl = clusters[:3]
    if len(cl) < 3:
        return None
    bbs = [cluster_bbox(c) for c in cl]
    ccs = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in bbs]
    mi = min(range(3), key=lambda i: ccs[i][0] + ccs[i][1])
    mx, my = ccs[mi]
    others = [i for i in range(3) if i != mi]
    ai = min(others, key=lambda i: abs(ccs[i][0] - mx))
    ri = [i for i in others if i != ai][0]
    return {"front": cl[mi], "top": cl[ai], "right": cl[ri]}


def rasterize(entities, size=320, pad=0.06, bbox=None):
    if bbox is None:
        bbox = _bbox([p for pts, _ in entities for p in pts])
    x0, y0, x1, y1 = bbox
    w = max(x1 - x0, 1e-6)
    h = max(y1 - y0, 1e-6)
    scale = (1 - 2 * pad) * size / max(w, h)
    ox = (size - w * scale) / 2 - x0 * scale
    oy = (size - h * scale) / 2 - y0 * scale
    img = Image.new("1", (size, size), 0)
    d = ImageDraw.Draw(img)
    for pts, _ in entities:
        sp = [(p[0] * scale + ox, size - (p[1] * scale + oy)) for p in pts]
        if len(sp) >= 2:
            d.line(sp, fill=1, width=1)
    return np.array(img, dtype=bool)


def iou(a, b, tol=2):
    ad = binary_dilation(a, iterations=tol)
    bd = binary_dilation(b, iterations=tol)
    return np.logical_and(ad, bd).sum() / max(np.logical_or(ad, bd).sum(), 1)


def dxf_counts(path):
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()
    t, lt = Counter(), Counter()
    for e in msp:
        t[e.dxftype()] += 1
        lt[str(e.dxf.get("linetype", "BYLAYER"))] += 1
    return dict(t), dict(lt)


# --------------------------------------------------------------------------
# Isolated pipeline run (timeout-guarded)
# --------------------------------------------------------------------------


def _worker(step, out_dir, stem, q):
    import rootutils as _r

    _r.setup_root(str(Path(__file__)), indicator=".project-root", pythonpath=True)
    from src.data.render.techdraw import generate_techdraw
    from src.data.render.config import techdraw_paths as _tp

    try:
        info = generate_techdraw(Path(step), _tp(Path(out_dir), stem))
        q.put(("ok", info))
    except Exception as exc:  # noqa: BLE001
        q.put(("err", f"{type(exc).__name__}: {exc}"))


def run_isolated(step, out_dir, stem, timeout):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(
        target=_worker, args=(str(step), str(out_dir), stem, q), daemon=True
    )
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


def svg_gray(path, w=1200, h=849):
    png = cairosvg.svg2png(
        url=str(path), output_width=w, output_height=h, background_color="white"
    )
    return np.array(Image.open(io.BytesIO(png)).convert("L")) < 128


def compare_part(stem, out_dir, timeout):
    step = GT_DIR / "target_step" / f"{stem}.step"
    gt_svg = GT_DIR / "techdraw" / "svg" / f"{stem}.svg"
    gt_dxf = GT_DIR / "techdraw" / "dxf" / f"{stem}.dxf"
    tp = techdraw_paths(Path(out_dir), stem)

    status, info = run_isolated(step, out_dir, stem, timeout)
    if status != "ok":
        return {"stem": stem, "status": status}

    ot, olt = dxf_counts(tp.dxf)
    gt_t, glt = dxf_counts(gt_dxf)

    our_ents = dxf_polylines(str(tp.dxf))
    gt_ents = dxf_polylines(str(gt_dxf))
    oc = label_views(cluster_views(our_ents))
    gc = label_views(cluster_views(gt_ents))
    pv = {}
    arrangement_ok = oc is not None and gc is not None
    if arrangement_ok:
        for k in ("front", "top", "right"):
            pv[k] = round(float(iou(rasterize(oc[k]), rasterize(gc[k]))), 3)

    # side-by-side PNG (ours top, GT bottom), full sheet
    our_img = svg_gray(tp.svg)
    gt_img = svg_gray(gt_svg)
    a = Image.fromarray((~our_img * 255).astype("uint8"))
    b = Image.fromarray((~gt_img * 255).astype("uint8"))
    comp = Image.new("L", (a.width, a.height * 2 + 8), 255)
    comp.paste(a, (0, 0))
    comp.paste(b, (0, a.height + 8))
    cp = Path(out_dir) / f"compare_{stem}.png"
    comp.save(cp)

    return {
        "stem": stem,
        "status": "ok",
        "scale": info["scale"],
        "n_marks": info["n_center_marks"],
        "arrangement_ok": arrangement_ok,
        "perview_iou": pv,
        "our_types": ot,
        "gt_types": gt_t,
        "our_lt": olt,
        "gt_lt": glt,
        "compare_png": str(cp),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--stems", type=str, default="")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/tmp/claude-1001/-home-ryotaro-my-works-drawing2cad/"
            "dcd1cb76-d8cd-436c-b767-6583954aaf3a/scratchpad/"
            "techdraw_agent/calib"
        ),
    )
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--real-dir", type=Path, default=None)
    ap.add_argument("--real-n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    if args.stems:
        stems = args.stems.split(",")
    else:
        all_stems = sorted(p.stem for p in (GT_DIR / "target_step").glob("*.step"))
        random.Random(args.seed).shuffle(all_stems)
        stems = ["000000", "000123", "004000"] + all_stems[: args.n - 3]

    results = []
    ious = []
    print(f"== Calibrating on {len(stems)} GT parts ==")
    for stem in stems:
        r = compare_part(stem, args.out, args.timeout)
        results.append(r)
        if r["status"] != "ok":
            print(f"{stem}: {r['status']}")
            continue
        pv = r["perview_iou"]
        ious += list(pv.values())
        dmarks = r["n_marks"] - r["gt_types"].get("INSERT", 0)
        print(
            f"{stem}: iou={pv} scale={r['scale']} marks={r['n_marks']}"
            f"(GT {r['gt_types'].get('INSERT', 0)}) arrange={r['arrangement_ok']}"
            f"{dmarks=}"
        )

    ok = [r for r in results if r["status"] == "ok"]
    print("\n== Summary ==")
    print(f"ran={len(results)} ok={len(ok)} failed={len(results) - len(ok)}")
    if ious:
        arr = np.array(ious)
        print(
            f"per-view IoU  n={len(arr)}  mean={arr.mean():.3f}  median={np.median(arr):.3f}"
            f"  p10={np.percentile(arr, 10):.3f}  min={arr.min():.3f}"
        )
    arr_ok = sum(r["arrangement_ok"] for r in ok)
    print(f"arrangement correct: {arr_ok}/{len(ok)}")

    # entity count aggregate deltas
    def agg(key, sub):
        return sum(r[key].get(sub, 0) for r in ok)

    for et in ("LINE", "CIRCLE", "ARC", "SPLINE", "LWPOLYLINE", "ELLIPSE", "INSERT"):
        print(f"  {et:11s} ours={agg('our_types', et):5d} gt={agg('gt_types', et):5d}")
    for lt in ("Continuous", "HIDDEN"):
        print(f"  lt {lt:10s} ours={agg('our_lt', lt):5d} gt={agg('gt_lt', lt):5d}")

    # real-input smoke test
    if args.real_dir:
        print("\n== Real-input smoke test ==")
        reals = sorted(args.real_dir.glob("*.step"))[: args.real_n]
        for step in reals:
            status, info = run_isolated(
                step, args.out / "real", step.stem, args.timeout
            )
            print(
                f"{step.name}: {status} "
                f"{ {k: info[k] for k in ('scale', 'n_center_marks')} if status == 'ok' else '' }"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
