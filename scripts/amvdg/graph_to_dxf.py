#!/usr/bin/env python3
"""Intra-view graph (JSON) -> 2D CAD (DXF). The deterministic last step of the 2D->2D leg:
the model predicts the graph; this turns it into an openable 2D-CAD file. Proves the
output IS 2D CAD data, not just JSON.

Geometry on layers VISIBLE (continuous) / HIDDEN (dashed); dimension callouts as TEXT
on layer DIM. Image px (y-down) are flipped to DXF y-up. Arcs lack stored start/end
angles in the intra-view schema, so they export as full circles on layer ARC (approx).

Usage:  python graph_to_dxf.py GRAPH.json OUT.dxf [--height H]
"""
import argparse, json, sys
import ezdxf


def load_prims(g):
    out = []
    for v in g.get("views", []):
        out += v.get("primitives", [])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graph"); ap.add_argument("out")
    ap.add_argument("--height", type=int, default=0, help="image height for y-flip (else inferred)")
    a = ap.parse_args()

    g = json.load(open(a.graph))
    prims = load_prims(g)

    # y-flip reference height
    H = a.height or (g.get("image_size_px") or [0, 0])[1]
    if not H:
        ys = []
        for p in prims:
            if p["type"] == "line":
                ys += [p["p1"][1], p["p2"][1]]
            elif "center" in p:
                ys += [p["center"][1] + p.get("r_px", 0)]
        H = max(ys) if ys else 0
    fy = lambda y: H - y

    doc = ezdxf.new(setup=True)  # standard linetypes incl. DASHED
    for name, lt in (("VISIBLE", "CONTINUOUS"), ("HIDDEN", "DASHED"),
                     ("ARC", "CONTINUOUS"), ("DIM", "CONTINUOUS")):
        doc.layers.add(name, linetype=lt)
    msp = doc.modelspace()

    nline = ncirc = narc = 0
    for p in prims:
        lay = "HIDDEN" if p.get("line_role", p.get("visibility")) == "hidden" else "VISIBLE"
        if p["type"] == "line":
            (x1, y1), (x2, y2) = p["p1"], p["p2"]
            msp.add_line((x1, fy(y1)), (x2, fy(y2)), dxfattribs={"layer": lay}); nline += 1
        elif p["type"] == "circle":
            cx, cy = p["center"]
            msp.add_circle((cx, fy(cy)), p["r_px"], dxfattribs={"layer": lay}); ncirc += 1
        elif p["type"] == "arc":
            cx, cy = p["center"]
            msp.add_circle((cx, fy(cy)), p["r_px"], dxfattribs={"layer": "ARC"}); narc += 1

    # dimension callouts as text (geometry-light: schema carries value/type, sometimes text_px)
    pid = {p.get("id"): p for p in prims}
    pre = {"diameter": "Ø", "radius": "R", "angular": ""}
    ndim = 0
    for i, dm in enumerate(g.get("annotations", g.get("dimensions", []))):
        label = pre.get(dm.get("kind", dm.get("type")), "") + ("%g" % dm["value"])
        if "text_px" in dm:
            x, y = dm["text_px"]
        else:  # place near first bound primitive, else stack in a corner
            refp = next((pid[r] for r in (dm.get("refs") or []) if r in pid), None)
            if refp and "center" in refp:
                x, y = refp["center"][0] + refp.get("r_px", 0) + 10, refp["center"][1]
            elif refp and "p1" in refp:
                x, y = refp["p1"]
            else:
                x, y = 20, 20 + i * 30
        msp.add_text(label, height=14, dxfattribs={"layer": "DIM"}).set_placement((x, fy(y)))
        ndim += 1

    doc.saveas(a.out)
    print("wrote %s | lines=%d circles=%d arcs=%d dim-texts=%d (H=%d)"
          % (a.out, nline, ncirc, narc, ndim, H))


if __name__ == "__main__":
    main()
