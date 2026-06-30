#!/usr/bin/env python3
# batch_dataset.py -- end-to-end synthetic-drawing dataset over a dir of seed STEP solids.
#   stage 1: render_dataset.py (RF_BATCH) -> per-part <name>.svg + <name>.graph.json  (needs FreeCAD)
#   stage 2: cairosvg rasterize -> <name>.png ; scan-noise aug -> <name>.scan.png (affine into graph)
#   -> manifest.jsonl ; (PNG, graph.json) pairs consumed directly by scripts/detector/circlenet.py
#
# Single-env driver (drawing2cad env): runs stage 1 as a subprocess of THIS python.
# render_dataset.py imports the conda-forge FreeCAD via the `freecad` shim, so no
# freecadcmd and no PYTHONPATH are needed.
#
# Usage:
#   python batch_dataset.py <step_dir> <out_dir> [--width 1800] [--no-scan] [--n N]
import os, sys, glob, json, subprocess, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pipeline import scan_augment   # main() is __main__-guarded; importing is safe


def stage1(step_dir, out_dir, width, n):
    env = dict(os.environ)
    env["RF_BATCH"] = "1"
    env["RF_STEPDIR"] = step_dir
    env["RF_OUTDIR"] = out_dir
    env["RF_WIDTH"] = str(width)
    env["RF_LOG"] = os.path.join(out_dir, "render_dataset.log")
    env["PYTHONUNBUFFERED"] = "1"
    if n and n > 0:
        env["RF_LIMIT"] = str(n)
    subprocess.run([sys.executable, os.path.join(HERE, "render_dataset.py")], env=env, check=False)
    log = open(env["RF_LOG"]).read() if os.path.exists(env["RF_LOG"]) else ""
    nok = log.count("\nOK ") + (1 if log.startswith("OK ") else 0)
    nskip = log.count("SKIP ")
    print("[stage1] render_dataset: OK=%d SKIP=%d" % (nok, nskip), flush=True)


def rasterize(svg_path, png_path, width):
    import cairosvg
    cairosvg.svg2png(url=svg_path, write_to=png_path, output_width=width, background_color="white")


def stage2(out_dir, width, do_scan):
    graphs = sorted(glob.glob(os.path.join(out_dir, "*.graph.json")))
    manifest = open(os.path.join(out_dir, "manifest.jsonl"), "w")
    nok = 0
    for gp in graphs:
        name = os.path.basename(gp)[:-len(".graph.json")]
        svg = os.path.join(out_dir, name + ".svg")
        if not os.path.exists(svg):
            continue
        png = os.path.join(out_dir, name + ".png")
        try:
            rasterize(svg, png, width)
        except Exception as e:
            print("  raster FAIL", name, e); continue
        row = {"part_id": name, "png": png, "graph": gp}
        if do_scan:
            scanp = os.path.join(out_dir, name + ".scan.png")
            aff = scan_augment(png, scanp, seed=hash(name) & 0xffff)
            g = json.load(open(gp)); g.setdefault("source", {})["scan_affine"] = aff
            json.dump(g, open(gp, "w"), indent=1)
            row["scan_png"] = scanp
        manifest.write(json.dumps(row) + "\n")
        nok += 1
    manifest.close()
    print("[stage2] rasterized %d drawings -> %s" % (nok, out_dir), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--width", type=int, default=1800)
    ap.add_argument("--no-scan", action="store_true")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--skip-stage1", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if not a.skip_stage1:
        stage1(a.step_dir, a.out_dir, a.width, a.n)
    stage2(a.out_dir, a.width, not a.no_scan)
