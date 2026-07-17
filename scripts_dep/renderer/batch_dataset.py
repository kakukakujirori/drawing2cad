#!/usr/bin/env python3
# batch_dataset.py -- end-to-end synthetic-drawing dataset over a dir of seed STEP solids.
#   stage 1: render_dataset.py --step-dir -> per-part <name>.svg + <name>.graph.json  (needs FreeCAD)
#   stage 2: cairosvg rasterize -> <name>.png ; scan-noise aug -> <name>.scan.png (affine into graph)
#   -> manifest.jsonl ; (PNG, graph.json) pairs consumed directly by scripts/detector/circlenet.py
#
# Single-env driver (drawing2cad env): runs stage 1 as a subprocess of THIS python.
#
# Usage:
#   python batch_dataset.py <step_dir> <out_dir> [--width 1800] [--no-scan] [--n N]
import argparse
import glob
import itertools
import json
import os
import subprocess
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from pipeline import scan_augment   # main() is __main__-guarded; importing is safe


def stage1(step_dir: str, out_dir: str, width: int, n: int, workers: int = 0, timeout: int = 60):
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    # render_dataset.py imports scripts.renderer.* — make the repo root importable
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    logp = os.path.join(out_dir, "render_dataset.log")
    # each part renders in its own forked process (see render_dataset.py's
    # _run_batch) so a TechDraw HLR hang on one bad shape can be killed instead
    # of freezing the whole run; 0 = auto (leave a couple cores free)
    if workers <= 0:
        workers = max(1, (os.cpu_count() or 4) - 2)
    cmd = [sys.executable, os.path.join(root, "scripts", "renderer", "render_dataset.py"),
           "--step-dir", step_dir, "--out-dir", out_dir, "--width", str(width), "--log", logp,
           "--workers", str(workers), "--timeout", str(timeout)]
    if n and n > 0:
        cmd += ["--limit", str(n)]
    r = subprocess.run(cmd, env=env, check=False)
    log = open(logp).read() if os.path.exists(logp) else ""
    nok = log.count("\nOK ") + (1 if log.startswith("OK ") else 0)
    nskip = log.count("\nSKIP ") + (1 if log.startswith("SKIP ") else 0)
    print("[stage1] render_dataset: OK=%d SKIP=%d rc=%d" % (nok, nskip, r.returncode), flush=True)
    if r.returncode != 0 or (nok == 0 and nskip == 0):
        print("[stage1] WARNING: renderer produced nothing — check %s" % logp, flush=True)


def rasterize(svg_path: str, png_path: str, width: int):
    import cairosvg
    cairosvg.svg2png(url=svg_path, write_to=png_path, output_width=width, background_color="white")


def _atomic_write_json(path, obj):
    # write to a temp file + fsync + os.replace so a killed/crashed worker (or a
    # transient disk hiccup) can never leave a truncated graph.json at `path` --
    # we hit exactly this once (an "OK"-logged part whose graph.json was truncated
    # with no corresponding error), which then crashes stage2's json.load below.
    tmp = "%s.tmp.%d" % (path, os.getpid())
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _stage2_worker(gp, out_dir, width, do_scan):
    """Rasterize one drawing (+ its scan-augmented version). Pure cairosvg/PIL/numpy
    CPU work with no native-hang risk (unlike stage1's FreeCAD calls), so a plain
    process pool is enough -- no per-task timeout/kill needed. Already-rasterized
    parts are skipped (idempotent) so a killed/resumed stage2 run doesn't redo work.
    Returns ("OK", name, manifest_row) / ("FAIL", name, error) / None (no svg yet)."""
    name = os.path.basename(gp)[:-len(".graph.json")]
    svg = os.path.join(out_dir, name + ".svg")
    if not os.path.exists(svg):
        return None
    png = os.path.join(out_dir, name + ".png")
    try:
        g = json.load(open(gp))
    except Exception as e:
        # a bad file here must not take down the whole pool (ProcessPoolExecutor.map
        # re-raises on the parent otherwise, aborting every still-pending part too)
        return ("FAIL", name, "corrupt graph.json: %s -- re-render this part via "
                               "render_dataset.py --step <its .step file>" % e)
    # graph px coords assume sheet.width_px; rasterize at THAT width so
    # labels stay aligned even if --width differs across invocations
    gw = (g.get("sheet") or {}).get("width_px") or width
    scanp = os.path.join(out_dir, name + ".scan.png")
    if not (os.path.exists(png) and (not do_scan or os.path.exists(scanp))):
        try:
            rasterize(svg, png, gw)
        except Exception as e:
            return ("FAIL", name, str(e))
        if do_scan:
            try:
                # stable per-part seed (hash() is salted per process — not reproducible)
                aff = scan_augment(png, scanp, seed=zlib.crc32(name.encode()) & 0xffff)
            except Exception as e:
                print("  scan FAIL", name, e)
            else:
                g.setdefault("source", {})["scan_affine"] = aff
                _atomic_write_json(gp, g)
    row = {"part_id": name, "png": png, "graph": gp}
    if do_scan and os.path.exists(scanp):
        row["scan_png"] = scanp
    return ("OK", name, row)


def stage2(out_dir: str, width: int, do_scan: bool, workers: int = 0):
    graphs = sorted(glob.glob(os.path.join(out_dir, "*.graph.json")))
    if workers <= 0:
        workers = max(1, (os.cpu_count() or 4) - 2)
    manifest = open(os.path.join(out_dir, "manifest.jsonl"), "w")
    nok = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        jobs = ex.map(_stage2_worker, graphs, itertools.repeat(out_dir),
                      itertools.repeat(width), itertools.repeat(do_scan))
        for res in tqdm(jobs, total=len(graphs), desc="stage2"):
            if res is None:
                continue
            kind, name, payload = res
            if kind == "FAIL":
                print("  raster FAIL", name, payload); continue
            manifest.write(json.dumps(payload) + "\n")
            nok += 1
    manifest.close()
    print("[stage2] rasterized %d drawings -> %s" % (nok, out_dir), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step_dir", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--no-scan", action="store_true")
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--workers", type=int, default=0,
                        help="parallel worker processes for stage1 (FreeCAD) and stage2 "
                             "(rasterize/scan-aug) (0 = auto: cpu_count-2)")
    parser.add_argument("--timeout", type=int, default=60,
                        help="per-part wall-clock timeout in seconds for stage1")
    parser.add_argument("--resume", action="store_true", help="resume from previous run")
    args = parser.parse_args()

    if os.path.isdir(args.out_dir) and not args.resume:
        raise FileExistsError(f"{args.out_dir=} already exists. Use --resume or remove it first.")
    os.makedirs(args.out_dir, exist_ok=True)
    if not args.skip_stage1:
        stage1(args.step_dir, args.out_dir, args.width, args.n, args.workers, args.timeout)
    stage2(args.out_dir, args.width, not args.no_scan, args.workers)
