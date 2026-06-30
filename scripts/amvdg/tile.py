#!/usr/bin/env python3
"""Crop-tile preprocessing for REAL high-res engineering drawings (e.g. CADGenBench
A2 sheets, 3072x2118). A whole-sheet pass downscales dimension text to illegibility
and overwhelms primitive detection; tiling at NATIVE resolution restores legibility.

Pipeline role:  image -> tiles (native res, overlapping) -> [extractor per tile,
local px coords] -> merge_graphs() -> single Tier-1 graph in GLOBAL image px.

This module is dependency-light (PIL only) and extractor-agnostic — the same tiles
feed a VLM or a dedicated OCR/CV stack. Coordinate remap + geometric dedup at tile
overlaps live here so the consumer just hands back per-tile graphs.

CLI:  python crop_tile.py IMAGE OUTDIR [--tile 1024] [--overlap 0.18]
      -> writes OUTDIR/tile_XX.png + OUTDIR/tiles.json (manifest w/ global offsets)
"""
import argparse, json, math, os
from PIL import Image

# keep tiles under the vision long-edge cap (~2576) so they are NOT downsampled
DEFAULT_TILE = 1024
DEFAULT_OVERLAP = 0.18


def make_tiles(W, H, tile=DEFAULT_TILE, overlap=DEFAULT_OVERLAP):
    """Grid of overlapping tiles covering WxH. Returns list of dicts with global box."""
    stride = max(1, int(round(tile * (1.0 - overlap))))
    def starts(extent):
        if extent <= tile:
            return [0]
        n = math.ceil((extent - tile) / stride) + 1
        # distribute so the last tile ends exactly at `extent`
        return [min(int(round(i * (extent - tile) / (n - 1))), extent - tile) for i in range(n)]
    xs, ys = starts(W), starts(H)
    tiles = []
    k = 0
    for y0 in ys:
        for x0 in xs:
            tiles.append({"i": k, "x0": x0, "y0": y0,
                          "x1": min(x0 + tile, W), "y1": min(y0 + tile, H)})
            k += 1
    return tiles


def tile_image(path, outdir, tile=DEFAULT_TILE, overlap=DEFAULT_OVERLAP):
    im = Image.open(path).convert("RGB")
    W, H = im.size
    os.makedirs(outdir, exist_ok=True)
    tiles = make_tiles(W, H, tile, overlap)
    for t in tiles:
        crop = im.crop((t["x0"], t["y0"], t["x1"], t["y1"]))
        fn = os.path.join(outdir, "tile_%02d.png" % t["i"])
        crop.save(fn)
        t["file"] = fn
    manifest = {"source": path, "image_size_px": [W, H], "tile": tile,
                "overlap": overlap, "tiles": tiles}
    json.dump(manifest, open(os.path.join(outdir, "tiles.json"), "w"), indent=1)
    return manifest


# ---------------------------------------------------------------------------
# merge per-tile graphs (local px) -> one global graph (global px), dedup overlaps
# ---------------------------------------------------------------------------
def _shift_prim(p, dx, dy):
    q = dict(p)
    if p["type"] == "line":
        q["p1"] = [p["p1"][0] + dx, p["p1"][1] + dy]
        q["p2"] = [p["p2"][0] + dx, p["p2"][1] + dy]
    else:  # circle / arc
        q["center"] = [p["center"][0] + dx, p["center"][1] + dy]
    return q


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _same_curve(a, b, ctol=20.0, rrel=0.2):
    return (_dist(a["center"], b["center"]) < ctol
            and abs(a.get("r_px", 0) - b.get("r_px", 0)) < max(8.0, rrel * a.get("r_px", 0)))


def _same_line(a, b, tol=20.0):
    fwd = _dist(a["p1"], b["p1"]) < tol and _dist(a["p2"], b["p2"]) < tol
    rev = _dist(a["p1"], b["p2"]) < tol and _dist(a["p2"], b["p1"]) < tol
    return fwd or rev


def merge_graphs(tiles, graphs, image_size_px):
    """tiles: list from make_tiles (same order as graphs). graphs[i] is the Tier-1
    graph extracted from tile i (LOCAL px). Returns one merged global-px graph with
    overlap duplicates collapsed and dim refs remapped to surviving primitive ids."""
    kept = []            # canonical global primitives
    idmap = {}           # per-tile global-id -> canonical kept id
    for t, g in zip(tiles, graphs):
        if not g:
            continue
        dx, dy = t["x0"], t["y0"]
        for v in g.get("views", []):
            for p in v.get("primitives", []):
                gp = _shift_prim(p, dx, dy)
                gid = "t%d_%s" % (t["i"], p.get("id", "?"))
                gp["id"] = gid
                dup = None
                for kp in kept:
                    if kp["type"] in ("circle", "arc") and gp["type"] in ("circle", "arc"):
                        if _same_curve(kp, gp):
                            dup = kp; break
                    elif kp["type"] == "line" and gp["type"] == "line":
                        if _same_line(kp, gp):
                            dup = kp; break
                if dup is not None:
                    idmap[gid] = dup["id"]
                else:
                    idmap[gid] = gid
                    kept.append(gp)

    # dims: remap refs to canonical ids, dedup by (type, value, ref-set)
    dims = []
    seen = set()
    for t, g in zip(tiles, graphs):
        if not g:
            continue
        for dim in g.get("dimensions", []):
            refs = [idmap.get("t%d_%s" % (t["i"], r)) for r in (dim.get("refs") or [])]
            refs = [r for r in refs if r]
            key = (dim.get("type"), round(float(dim.get("value", 0)), 1), frozenset(refs))
            if key in seen:
                continue
            seen.add(key)
            q = dict(dim); q["refs"] = refs
            q["id"] = "D%d" % (len(dims) + 1)
            dims.append(q)

    return {"image_size_px": image_size_px,
            "views": [{"name": "merged", "primitives": kept}],
            "dimensions": dims}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("outdir")
    ap.add_argument("--tile", type=int, default=DEFAULT_TILE)
    ap.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    a = ap.parse_args()
    m = tile_image(a.image, a.outdir, a.tile, a.overlap)
    print("%s %dx%d -> %d tiles (%dpx, %.0f%% overlap) in %s"
          % (os.path.basename(a.image), *m["image_size_px"], len(m["tiles"]),
             m["tile"], m["overlap"] * 100, a.outdir))
    for t in m["tiles"]:
        print("  tile %02d  global [%d,%d]-[%d,%d]" % (t["i"], t["x0"], t["y0"], t["x1"], t["y1"]))


if __name__ == "__main__":
    main()
