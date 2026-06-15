#!/usr/bin/env python
"""Phase-1 renderer, stage 2: drawing-graph JSON -> Tier-0 raster PNG + Tier-1 DXF.

Reads the graph emitted by project_views.py and produces:
  - <id>.png : a raster engineering-drawing-like sheet (what the model sees at
               inference on real drawings) -- multi-view, visible/hidden lines, dims.
  - <id>.dxf : the structured 2D CAD (lines/circles/arcs on visible/hidden/dim
               layers) -- the "2D CAD" intermediate from Morpho's pipeline.

Run (cadrille env has matplotlib + ezdxf):
  /home/ryotaro/miniforge3/envs/cadrille/bin/python \
    scripts/renderer/render_drawing.py experiments/renderer_demo/demo_box_hole.graph.json
"""
import os, sys, json, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc
import ezdxf

# view placement offsets (sheet coords), third-angle-ish: top above front, right to its right
PLACE = {"front": (0, 0), "top": (0, 1), "right": (1, 0)}
GAP = 40.0


def view_bbox(view):
    xs, ys = [], []
    for p in view["primitives"]:
        t = p["type"]
        if t == "line":
            xs += [p["p1"][0], p["p2"][0]]; ys += [p["p1"][1], p["p2"][1]]
        elif t in ("circle", "arc"):
            xs += [p["center"][0] - p["r"], p["center"][0] + p["r"]]
            ys += [p["center"][1] - p["r"], p["center"][1] + p["r"]]
        elif t == "polyline":
            xs += [q[0] for q in p["pts"]]; ys += [q[1] for q in p["pts"]]
    if not xs:
        return (0, 0, 0, 0)
    return (min(xs), min(ys), max(xs), max(ys))


def draw_primitive(ax, p, ox, oy, style):
    kw = dict(color="black", lw=0.8) if style == "visible" else dict(color="0.4", lw=0.6, ls=(0, (4, 3)))
    t = p["type"]
    if t == "line":
        ax.plot([p["p1"][0] + ox, p["p2"][0] + ox], [p["p1"][1] + oy, p["p2"][1] + oy], **kw)
    elif t == "circle":
        ax.add_patch(Circle((p["center"][0] + ox, p["center"][1] + oy), p["r"], fill=False, **kw))
    elif t == "arc":
        ax.add_patch(Arc((p["center"][0] + ox, p["center"][1] + oy), 2 * p["r"], 2 * p["r"],
                         theta1=min(p["a0"], p["a1"]), theta2=max(p["a0"], p["a1"]), **kw))
    elif t == "polyline":
        ax.plot([q[0] + ox for q in p["pts"]], [q[1] + oy for q in p["pts"]], **kw)


def render_png(graph, out_png):
    fig, ax = plt.subplots(figsize=(11.7, 8.3))  # A4-ish landscape
    ax.set_aspect("equal"); ax.axis("off")
    bbs = {v["name"]: view_bbox(v) for v in graph["views"]}
    # normalize each view to its own origin, then place on a clean grid using
    # actual per-view sizes (front bottom-left, top above front, right to its right)
    def wh(n):
        b = bbs[n]; return (b[2] - b[0], b[3] - b[1])
    fw, fh = wh("front") if "front" in bbs else (0, 0)
    base = {"front": (0.0, 0.0),
            "top": (0.0, fh + GAP),
            "right": (fw + GAP, 0.0)}
    placed = {}
    for v in graph["views"]:
        name = v["name"]
        bb = bbs[name]
        bx, by = base.get(name, (0.0, 0.0))
        ox = bx - bb[0]   # shift so this view's min corner sits at (bx, by)
        oy = by - bb[1]
        placed[name] = (ox, oy, bb)   # bb stays in view-local coords (dim code adds ox/oy)
        for p in v["primitives"]:
            draw_primitive(ax, p, ox, oy, p["visibility"])
    # crude dimension rendering (structured values live in the graph/DXF; this is the visual)
    for a in graph["annotations"]:
        ref_view = a["refs"][0].split(":")[0] if a["refs"] else None
        if ref_view not in placed:
            continue
        ox, oy, bb = placed[ref_view]
        if a["type"] == "linear" and a.get("subtype") == "horizontal":
            y = oy + bb[1] - 12
            ax.annotate("", (ox + bb[0], y), (ox + bb[2], y), arrowprops=dict(arrowstyle="<->", lw=0.6))
            ax.text(ox + (bb[0] + bb[2]) / 2, y - 4, str(a["value"]), ha="center", fontsize=7)
        elif a["type"] == "linear":
            x = ox + bb[0] - 12
            ax.annotate("", (x, oy + bb[1]), (x, oy + bb[3]), arrowprops=dict(arrowstyle="<->", lw=0.6))
            ax.text(x - 3, oy + (bb[1] + bb[3]) / 2, str(a["value"]), va="center", rotation=90, fontsize=7)
        elif a["type"] == "diameter":
            ax.text(ox + bb[2] + 4, oy + bb[3], a.get("text", f'D{a["value"]}'), fontsize=7, color="navy")
    ax.set_title(f'{graph["part_id"]}   (synthetic dimensioned drawing — Tier-0 raster)', fontsize=8)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def export_dxf(graph, out_dxf):
    doc = ezdxf.new(); msp = doc.modelspace()
    for lyr, col in (("visible", 7), ("hidden", 8), ("dim", 1)):
        if lyr not in doc.layers:
            doc.layers.add(lyr, color=col)
    for v in graph["views"]:
        col, row = PLACE.get(v["name"], (0, 0))
        ox = col * (graph["bbox_3d"]["dx"] + GAP)
        oy = row * (graph["bbox_3d"]["dz"] + GAP)
        for p in v["primitives"]:
            lyr = "visible" if p["visibility"] == "visible" else "hidden"
            t = p["type"]
            if t == "line":
                msp.add_line((p["p1"][0] + ox, p["p1"][1] + oy), (p["p2"][0] + ox, p["p2"][1] + oy),
                             dxfattribs={"layer": lyr})
            elif t == "circle":
                msp.add_circle((p["center"][0] + ox, p["center"][1] + oy), p["r"], dxfattribs={"layer": lyr})
            elif t == "arc":
                msp.add_arc((p["center"][0] + ox, p["center"][1] + oy), p["r"],
                            min(p["a0"], p["a1"]), max(p["a0"], p["a1"]), dxfattribs={"layer": lyr})
            elif t == "polyline":
                msp.add_lwpolyline([(q[0] + ox, q[1] + oy) for q in p["pts"]], dxfattribs={"layer": lyr})
    doc.saveas(out_dxf)


def render_json(jp):
    graph = json.load(open(jp))
    base = jp[:-len(".graph.json")] if jp.endswith(".graph.json") else os.path.splitext(jp)[0]
    render_png(graph, base + ".png")
    export_dxf(graph, base + ".dxf")
    return base


def main():
    import glob, argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="?", help="single <id>.graph.json")
    ap.add_argument("--dir", help="batch: render every *.graph.json in dir")
    args = ap.parse_args()
    if args.dir:
        files = sorted(glob.glob(os.path.join(args.dir, "*.graph.json")))
        n = 0
        for jp in files:
            try:
                render_json(jp); n += 1
            except Exception as e:
                print(f"[render] FAIL {os.path.basename(jp)}: {type(e).__name__}: {e}")
        print(f"[render] {n}/{len(files)} drawings (png+dxf) in {args.dir}")
    else:
        base = render_json(args.json)
        print(f"[render] wrote {base}.png and {base}.dxf")


if __name__ == "__main__":
    main()
