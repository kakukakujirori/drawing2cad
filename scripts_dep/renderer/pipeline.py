#!/usr/bin/env python3
# pipeline.py -- orchestrates the dataset generation (runs in the venv, NOT freecadcmd).
#   1) calls freecadcmd render_dataset.py to emit <name>.svg + <name>.graph.json
#   2) rasterizes <name>.png (cairosvg)
#   3) scan-augments -> <name>.scan.png (+ records affine into graph.json)
#   4) verifies GT against the clean PNG -> <name>.verify.png
#   5) writes manifest.jsonl
#
# Usage:  ./venv/bin/python pipeline.py
import os, sys, json, subprocess, math, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Machine-specific locations come from env vars (no hardcoded paths).
#   FREECADCMD  : path to the freecadcmd binary (default: on PATH)
#   STEP_DIR    : dir holding the seed *.step files (default: cwd)
#   D2C_OUT     : output dataset dir (default: ./dataset)
OUT = os.environ.get("D2C_OUT", "dataset")
FREECAD = os.environ.get("FREECADCMD", "freecadcmd")
RENDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "render_dataset.py")
WIDTH = int(os.environ.get("RF_WIDTH", "1800"))

_STEP_DIR = os.environ.get("STEP_DIR", ".")
JOBS = [(os.path.join(_STEP_DIR, "Bracket.step"), "Bracket"),
        (os.path.join(_STEP_DIR, "Flange.step"), "Flange")]


def run_freecad():
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    logp = os.path.join(OUT, "render_dataset.log")
    cmd = [FREECAD, RENDER, "--step-dir", _STEP_DIR, "--out-dir", OUT,
           "--width", str(WIDTH), "--log", logp]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    log = open(logp).read() if os.path.exists(logp) else ""
    return r.returncode, log


def rasterize(svg_path, png_path, width=WIDTH):
    import cairosvg
    cairosvg.svg2png(url=svg_path, write_to=png_path, output_width=width, background_color="white")


# ---------------------------------------------------------------------------
#  scan-noise augmentation
# ---------------------------------------------------------------------------

def scan_augment(png_path, out_path, seed=0):
    """Turn a clean vector PNG into a realistic scanned drawing.
    Returns the applied affine {rotation_deg, tx, ty, scale} (maps clean px -> scanned px)."""
    rng = random.Random(seed)
    img = Image.open(png_path).convert("L")   # grayscale
    W, H = img.size

    # 1) paper tone: very light grey background instead of pure white
    paper = rng.randint(238, 250)
    bg = Image.new("L", (W, H), paper)
    # darken lines slightly (ink not pure black) + line-width jitter via slight blur then threshold-ish
    arr = np.asarray(img).astype(np.float32)
    # map white->paper, black-> ink grey
    ink = rng.randint(18, 48)
    arr = ink + (arr / 255.0) * (paper - ink)

    # 2) line-width jitter: small morphological-ish via min-filter on darks (random)
    img2 = Image.fromarray(arr.clip(0, 255).astype(np.uint8), "L")
    if rng.random() < 0.6:
        img2 = img2.filter(ImageFilter.MinFilter(3))  # thicken dark strokes a touch

    # 3) affine: small rotation + translation + scale (simulate scanner misalignment)
    rot = rng.uniform(-1.6, 1.6)
    sc = rng.uniform(0.992, 1.008)
    tx = rng.uniform(-6, 6); ty = rng.uniform(-6, 6)
    # PIL rotate about center; record the composite affine (clean_px -> scan_px)
    img3 = img2.rotate(rot, resample=Image.BICUBIC, fillcolor=paper, center=(W / 2, H / 2))
    if abs(sc - 1.0) > 1e-4:
        nw, nh = int(W * sc), int(H * sc)
        img3 = img3.resize((nw, nh), Image.BICUBIC)
        canvas = Image.new("L", (W, H), paper)
        canvas.paste(img3, (int((W - nw) / 2), int((H - nh) / 2)))
        img3 = canvas
    if tx or ty:
        img3 = img3.transform((W, H), Image.AFFINE, (1, 0, -tx, 0, 1, -ty),
                              resample=Image.BICUBIC, fillcolor=paper)

    # 4) blur (scan softness)
    img3 = img3.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 0.9)))

    # 5) gaussian + speckle noise
    a = np.asarray(img3).astype(np.float32)
    a += np.random.default_rng(seed).normal(0, rng.uniform(3, 7), a.shape)
    # speckle: a few darker specks
    mask = np.random.default_rng(seed + 1).random(a.shape) < 0.0008
    a[mask] = np.random.default_rng(seed + 2).integers(20, 90, mask.sum())
    a = a.clip(0, 255).astype(np.uint8)
    img4 = Image.fromarray(a, "L")

    # 6) mild contrast variation + JPEG artifacts
    if rng.random() < 0.7:
        tmp = out_path + ".tmp.jpg"
        img4.convert("RGB").save(tmp, "JPEG", quality=rng.randint(55, 80))
        img4 = Image.open(tmp).convert("L")
        os.remove(tmp)
    img4.save(out_path)

    return {"rotation_deg": round(rot, 4), "scale": round(sc, 5),
            "tx": round(tx, 3), "ty": round(ty, 3),
            "center_px": [W / 2, H / 2],
            "note": "scan_px = R(rot)·S(scale) about center, then +(tx,ty); apply to clean-px GT coords to map onto scan.png"}


def apply_affine(pt, aff):
    """Map a clean-px point onto the scanned image using the recorded affine."""
    cx, cy = aff["center_px"]; rot = math.radians(aff["rotation_deg"]); sc = aff["scale"]
    x, y = pt[0] - cx, pt[1] - cy
    # rotate (PIL rotate by +rot is counter-clockwise visually => forward map uses -rot)
    ca, sa = math.cos(-rot), math.sin(-rot)
    xr = (x * ca - y * sa) * sc + cx + aff["tx"]
    yr = (x * sa + y * ca) * sc + cy + aff["ty"]
    return [round(xr, 2), round(yr, 2)]


# ---------------------------------------------------------------------------
#  verifier: overlay GT onto the clean PNG
# ---------------------------------------------------------------------------

def verify(png_path, graph_path, out_path):
    g = json.load(open(graph_path))
    img = Image.open(png_path).convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        fontb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = fontb = ImageFont.load_default()

    COL_VIS = (0, 150, 0, 255)
    COL_HID = (200, 120, 0, 255)
    COL_CIRC = (0, 90, 255, 255)
    COL_DIM = (220, 0, 0, 255)
    prim_by_id = {}

    for gv in g["views"]:
        for p in gv["primitives"]:
            prim_by_id[p["id"]] = p
            col = COL_VIS if p.get("line_role", p.get("visibility")) == "visible" else COL_HID
            if p["type"] in ("circle", "arc"):
                cx, cy = p["center"]; r = p["r_px"]
                d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=COL_CIRC, width=2)
                d.line([cx - 4, cy, cx + 4, cy], fill=COL_CIRC, width=1)
                d.line([cx, cy - 4, cx, cy + 4], fill=COL_CIRC, width=1)
            else:
                bb = p["bbox_px"]
                d.rectangle(bb, outline=col, width=1)

    # dimensions: draw text anchor + line to each ref's bbox center
    for dim in g.get("annotations", g.get("dimensions", [])):
        tx, ty = dim["text_px"]
        d.ellipse([tx - 5, ty - 5, tx + 5, ty + 5], outline=COL_DIM, width=2)
        lbl = "%s=%s" % (dim["id"], dim["value"])
        d.text((tx + 6, ty - 8), lbl, fill=COL_DIM, font=font)
        for ref in dim["refs"]:
            p = prim_by_id.get(ref)
            if not p:
                continue
            bb = p["bbox_px"]; mx = (bb[0] + bb[2]) / 2; my = (bb[1] + bb[3]) / 2
            d.line([tx, ty, mx, my], fill=(220, 0, 0, 120), width=1)

    # legend
    leg = ["GT OVERLAY  green=visible prim bbox  orange=hidden  blue=circle  red=dim anchor+refs"]
    d.rectangle([8, 8, 8 + 760, 34], fill=(255, 255, 255, 220))
    d.text((12, 10), leg[0], fill=(0, 0, 0, 255), font=fontb)
    img.save(out_path)


# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT, exist_ok=True)
    rc, log = run_freecad()
    print("freecad rc=", rc)
    print(log.strip())
    if "FAIL" in log:
        print("RENDER FAILED — aborting"); sys.exit(1)

    manifest = open(os.path.join(OUT, "manifest.jsonl"), "w")
    summary = []
    for step, name in JOBS:
        svg = os.path.join(OUT, name + ".svg")
        graph_path = os.path.join(OUT, name + ".graph.json")
        if not os.path.exists(graph_path):
            print("MISSING graph for", name); continue
        clean = os.path.join(OUT, name + ".png")
        scan = os.path.join(OUT, name + ".scan.png")
        verifyp = os.path.join(OUT, name + ".verify.png")

        rasterize(svg, clean)
        import zlib
        aff = scan_augment(clean, scan, seed=zlib.crc32(name.encode()) & 0xffff)

        # record affine into graph.json so labels map onto scan.png
        g = json.load(open(graph_path))
        g.setdefault("source", {})["scan_affine"] = aff
        json.dump(g, open(graph_path, "w"), indent=1)

        verify(clean, graph_path, verifyp)

        counts = g.get("_counts") or _counts(g)
        row = {"part_id": name, "step": step,
               "png": clean, "scan_png": scan, "graph": graph_path, "verify": verifyp,
               "bbox_3d": (g.get("world") or {}).get("bbox_3d") or g.get("bbox_3d"),
               "scale": g["sheet"]["scale"],
               "counts": counts, "scan_affine": aff}
        manifest.write(json.dumps(row) + "\n")
        summary.append((name, counts))
        print("DONE", name, counts)
    manifest.close()
    print("\nMANIFEST:", os.path.join(OUT, "manifest.jsonl"))
    for nm, c in summary:
        print(" ", nm, c)


def _counts(g):
    from collections import Counter
    _anns = g.get("annotations", g.get("dimensions", []))
    _vis = lambda p: p.get("line_role", p.get("visibility"))
    nv = sum(1 for gv in g["views"] for p in gv["primitives"] if _vis(p) == "visible")
    nh = sum(1 for gv in g["views"] for p in gv["primitives"] if _vis(p) == "hidden")
    by_type = Counter(d.get("kind", d.get("type")) for d in _anns)
    nrefless = sum(1 for d in _anns if not d["refs"])
    return {"prim_visible": nv, "prim_hidden": nh, "dims_total": len(_anns),
            "dims_by_type": dict(by_type), "dims_without_refs": nrefless,
            "features": len(g.get("features", g.get("correspondences", [])))}


if __name__ == "__main__":
    main()
