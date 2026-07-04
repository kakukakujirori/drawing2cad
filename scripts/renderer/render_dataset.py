#!/usr/bin/env python
# render_dataset.py -- STEP -> (vector drawing SVG) + (ground-truth intra-/inter-view graph JSON)
#
# Reuses the proven headless recipe: TechDraw.project() under freecadcmd composes the SVG
# ourselves, then cairosvg rasterizes. The DrawViewPart/GUI path is NOT used (empty/hangs headless).
#
# The renderer now RECORDS every primitive it draws (with a stable id, view, visibility,
# coarse feature tag, and geometry) and every dimension (with the primitive ids it measures
# in `refs`). All coordinates in the emitted graph JSON are in FINAL PNG PIXEL SPACE, so a
# downstream model's predictions can be scored directly on the rendered image.
#
# NEW in Phase 4:
# Uses CADProjector and GraphBuilder as an "oracle" to mathematically match 3D topological
# entities (topo_origins) against the segmented Hidden Line Removal (HLR) primitives generated
# by TechDraw. This gives us both visual occlusion roles AND exact topological relationships.
#

import argparse
import glob
import json
import math
import os
import sys
from collections import Counter
from tqdm import tqdm

import freecad          # conda-forge shim: puts FreeCAD's libs on sys.path
import FreeCAD as App
import Part
import TechDraw

from scripts.renderer.cad_projector import CADProjector
from scripts.renderer.graph_builder import GraphBuilder
from scripts.renderer.dimensioner import LegacyDimensioner

# projection_dir recorded per ortho view: unit vector from the part TOWARD the
# viewer (= TechDraw Direction), third-angle. These are exactly the vectors
# passed to TechDraw.projectEx / CADProjector below — keep all three in sync.
_PROJ_DIR = {"front": [0, -1, 0], "top": [0, 0, 1], "right": [1, 0, 0]}

# canonical (u,v) model axes per view, as produced by View.REMAP; recorded in
# frame.axis_remap so a consumer can invert px -> model coords without guessing.
# matrix maps canonical (u,v) -> pixel-axis signs (px y grows downward).
_AXIS_REMAP = {
    "front": {"matrix": [[1, 0], [0, -1]], "px_x": "+X", "px_y": "-Z"},
    "top":   {"matrix": [[1, 0], [0, -1]], "px_x": "+X", "px_y": "-Y"},
    "right": {"matrix": [[1, 0], [0, -1]], "px_x": "+Y", "px_y": "-Z"},
}

# ============================================================================
#  geometry -> SVG path + primitive records
# ============================================================================

def edge_to_path(e, tol=0.05):
    c = getattr(e, "Curve", None)
    vs = e.Vertexes
    if c is not None and c.TypeId == "Part::GeomCircle":
        r = c.Radius
        cx, cy = c.Center.x, c.Center.y
        if e.Closed or len(vs) < 2:
            return ("M %.4f %.4f A %.4f %.4f 0 1 0 %.4f %.4f A %.4f %.4f 0 1 0 %.4f %.4f Z"
                    % (cx - r, cy, r, r, cx + r, cy, r, r, cx - r, cy))
        p0, p1 = vs[0].Point, vs[-1].Point
        # decide large/sweep from an actual point on the arc: circle params
        # from HLR can wrap around 2*pi, which made the old a1-a0 heuristic
        # pick the complementary arc for some edges.
        try:
            mid = e.discretize(Number=3)[1]
            t0 = math.atan2(p0.y - cy, p0.x - cx)
            tm = math.atan2(mid.y - cy, mid.x - cx)
            t1 = math.atan2(p1.y - cy, p1.x - cx)
            ccw_full = (t1 - t0) % (2 * math.pi)      # CCW span p0 -> p1
            ccw_mid = (tm - t0) % (2 * math.pi)       # CCW position of mid
            is_ccw = ccw_mid <= ccw_full + 1e-9       # mid reached going CCW?
            span = ccw_full if is_ccw else (2 * math.pi - ccw_full)
            large = 1 if span > math.pi else 0
            sweep = 1 if is_ccw else 0                # SVG sweep=1: +x toward +y
        except Exception:
            large, sweep = 0, 1
        return "M %.4f %.4f A %.4f %.4f 0 %d %d %.4f %.4f" % (
            p0.x, p0.y, r, r, large, sweep, p1.x, p1.y)
    try:
        pts = e.discretize(Deflection=tol)
    except Exception:
        pts = [v.Point for v in vs]
    if len(pts) < 2:
        return ""
    d = "M %.4f %.4f" % (pts[0].x, pts[0].y)
    for p in pts[1:]:
        d += " L %.4f %.4f" % (p.x, p.y)
    return d

def compound_edges(grp):
    try:
        return list(grp.Edges)
    except Exception:
        return []

def classify_edge(e):
    """Coarse primitive type + geometry in the edge's own 2D coords."""
    c = getattr(e, "Curve", None)
    vs = e.Vertexes
    if c is not None and c.TypeId == "Part::GeomCircle":
        r = c.Radius
        cen = (c.Center.x, c.Center.y)
        if e.Closed or len(vs) < 2:
            return ("circle", {"center": cen, "r": r})
        p0, p1 = vs[0].Point, vs[-1].Point
        try:
            m = e.discretize(Number=3)[1]
            mid = (m.x, m.y)
        except Exception:
            mid = None
        return ("arc", {"center": cen, "r": r, "mid": mid,
                        "p1": (p0.x, p0.y), "p2": (p1.x, p1.y)})
    if c is not None and c.TypeId == "Part::GeomEllipse":
        # HLR emits an exact GeomEllipse for an obliquely-projected circle/ellipse.
        # Keep it parametric (center + major/minor semi-axes + rotation) instead of
        # discretizing to a polyline. maj/min are unit directions in the edge's 2D
        # frame; _record maps them + the lengths through the view remap to px.
        cen = (c.Center.x, c.Center.y)
        xa = c.XAxis
        ya = getattr(c, "YAxis", None)
        maj = (xa.x, xa.y)
        mino = (ya.x, ya.y) if ya is not None else (-xa.y, xa.x)
        g = {"center": cen, "rmaj": c.MajorRadius, "rmin": c.MinorRadius,
             "maj": maj, "min": mino}
        if not (e.Closed or len(vs) < 2):
            p0, p1 = vs[0].Point, vs[-1].Point
            try:
                m = e.discretize(Number=3)[1]
                g["mid"] = (m.x, m.y)
            except Exception:
                g["mid"] = None
            g["p1"] = (p0.x, p0.y)
            g["p2"] = (p1.x, p1.y)
        return ("ellipse", g)
    # treat as line if 2 endpoints and (near-)straight, else polyline -> store endpoints + samples
    if len(vs) >= 2:
        p0, p1 = vs[0].Point, vs[-1].Point
        # check straightness via discretized midpoint deviation
        try:
            pts = e.discretize(Number=5)
            straight = True
            ax, ay = p1.x - p0.x, p1.y - p0.y
            L = math.hypot(ax, ay) or 1.0
            for q in pts[1:-1]:
                # distance from line p0->p1
                dist = abs(ax * (p0.y - q.y) - (p0.x - q.x) * ay) / L
                if dist > 0.05 * L + 0.05:
                    straight = False; break
        except Exception:
            straight = True
        if straight:
            return ("line", {"p1": (p0.x, p0.y), "p2": (p1.x, p1.y)})
        else:
            pts = [(p.x, p.y) for p in e.discretize(Deflection=0.2)]
            return ("polyline", {"pts": pts, "p1": (p0.x, p0.y), "p2": (p1.x, p1.y)})
    return ("line", {"p1": (0, 0), "p2": (0, 0)})

def point_on_segment(px, py, q1x, q1y, q2x, q2y, tol=1e-3):
    dx, dy = q2x - q1x, q2y - q1y
    L2 = dx*dx + dy*dy
    if L2 < 1e-6:
        return math.hypot(px-q1x, py-q1y) < tol
    t = ((px - q1x)*dx + (py - q1y)*dy) / L2
    if t < -tol or t > 1 + tol:
        return False
    projx = q1x + t*dx
    projy = q1y + t*dy
    return math.hypot(px-projx, py-projy) < tol

def points_same(p1, p2, tol=1e-3):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1]) < tol


# ============================================================================
#  View
# ============================================================================

class View:
    REMAP = {
        "front": (0, 1, -1, 0),
        "top":   (1, 0, 0, 1),
        "right": (0, -1, 1, 0),
    }

    def __init__(self, name, shape, direction, oracle_prims):
        self.name = name
        self.direction = direction
        self.oracle_prims = oracle_prims

        # projectEx groups: [0]V sharp [1]V1 smooth [2]VN seam [3]VO outline
        # [4]VI iso [5]H sharp [6]H1 smooth [7]HN seam [8]HO outline [9]HI iso.
        res = TechDraw.projectEx(shape, App.Vector(*direction))
        self.edges_vis = compound_edges(res[0]) + compound_edges(res[1]) + compound_edges(res[3])
        self.edges_hid = compound_edges(res[5]) + compound_edges(res[8])
        self.a, self.b, self.c, self.d = self.REMAP[name]
        self._bbox()

    def cmap(self, lx, ly):
        return (self.a * lx + self.b * ly, self.c * lx + self.d * ly)

    def _bbox(self):
        us, vs = [], []
        for e in self.edges_vis + self.edges_hid:
            try:
                for vtx in e.Vertexes:
                    cu, cv = self.cmap(vtx.X, vtx.Y)
                    us.append(cu); vs.append(cv)
                b = e.BoundBox
                for (lx, ly) in ((b.XMin, b.YMin), (b.XMax, b.YMax),
                                 (b.XMin, b.YMax), (b.XMax, b.YMin)):
                    cu, cv = self.cmap(lx, ly)
                    us.append(cu); vs.append(cv)
            except Exception:
                pass
        if not us:
            self.umin = self.vmin = 0.0; self.umax = self.vmax = 1.0
        else:
            self.umin, self.umax = min(us), max(us)
            self.vmin, self.vmax = min(vs), max(vs)
        self.w = self.umax - self.umin
        self.h = self.vmax - self.vmin

    def set_layout(self, ox, oy, scale):
        self.ox, self.oy, self.scale = ox, oy, scale

    def M(self, cu, cv):
        """canonical model-axis coords -> SVG sheet-mm."""
        return (self.ox + (cu - self.umin) * self.scale,
                self.oy + (self.vmax - cv) * self.scale)

    def sheet_w(self): return self.w * self.scale
    def sheet_h(self): return self.h * self.scale

    def _matrix(self):
        s = self.scale
        A = s * self.a; B = -s * self.c; C = s * self.b; D = -s * self.d
        E = self.ox - self.umin * s; F = self.oy + self.vmax * s
        return (A, B, C, D, E, F)

    def svg_group(self):
        m = self._matrix()
        tr = "matrix(%.6f %.6f %.6f %.6f %.4f %.4f)" % m
        parts = ['<g transform="%s">' % tr]
        for e in self.edges_hid:
            d = edge_to_path(e)
            if d:
                parts.append('<path class="hidden" d="%s"/>' % d)
        for e in self.edges_vis:
            d = edge_to_path(e)
            if d:
                parts.append('<path class="visible" d="%s"/>' % d)
        parts.append("</g>")
        return "\n".join(parts)

    def primitive_records(self, idprefix, px):
        """Yield GT primitive dicts in PNG-pixel space, MATCHED with Oracle topo_origins.
        HLR emits the same edge in several projectEx groups (sharp + outline), so
        exact duplicates and zero-length degenerates are dropped here, before ids
        are handed out to the dimensioner."""
        out = []
        seen = set()
        for vis_tag, edges in (("visible", self.edges_vis), ("hidden", self.edges_hid)):
            for e in edges:
                typ, g = classify_edge(e)
                rec = self._record(idprefix, len(out), typ, g, vis_tag, px)
                if not rec:
                    continue
                if typ == "line" and rec["p1"] == rec["p2"]:
                    continue
                key = (vis_tag, typ) + tuple(round(c, 1) for c in _prim_coord_sig(rec))
                if key in seen:
                    continue
                seen.add(key)
                out.append(rec)
        return out

    def _to_px(self, lx, ly, px):
        cu, cv = self.cmap(lx, ly)
        sx, sy = self.M(cu, cv)
        return [round(sx * px, 2), round(sy * px, 2)]

    def _record(self, idprefix, n, typ, g, vis_tag, px):
        rid = "%s%d" % (idprefix, n)
        feat = "outline"
        if typ in ("circle", "arc"):
            feat = "hole_or_round"

        base = {"id": rid, "type": typ, "line_role": vis_tag, "feature_tag": feat,
                "state": "known", "coords_source": "gt"}

        if typ == "line":
            base["p1"] = self._to_px(*g["p1"], px=px)
            base["p2"] = self._to_px(*g["p2"], px=px)
        elif typ == "polyline":
            base["pts"] = [self._to_px(x, y, px) for (x, y) in g["pts"]]
            base["p1"] = self._to_px(*g["p1"], px=px)
            base["p2"] = self._to_px(*g["p2"], px=px)
        elif typ == "circle":
            base["center"] = self._to_px(*g["center"], px=px)
            base["r_px"] = round(g["r"] * self.scale * px, 2)
            base["r_mm"] = round(g["r"], 3)
        elif typ == "arc":
            base["center"] = self._to_px(*g["center"], px=px)
            base["r_px"] = round(g["r"] * self.scale * px, 2)
            base["r_mm"] = round(g["r"], 3)
            base["p1"] = self._to_px(*g["p1"], px=px)
            base["p2"] = self._to_px(*g["p2"], px=px)
            # p1/p2 + center leave the major/minor side ambiguous; record the
            # swept angles. Convention: theta = atan2(y-cy, x-cx) in PX frame
            # (y down), degrees in [0,360); the arc runs from start_angle to
            # end_angle in INCREASING theta (mod 360).
            if g.get("mid"):
                mpx = self._to_px(*g["mid"], px=px)
                sa, ea = _arc_angles_px(base["center"], base["p1"], base["p2"], mpx)
                base["start_angle"] = sa
                base["end_angle"] = ea
        elif typ == "ellipse":
            cl = g["center"]
            base["center"] = self._to_px(*cl, px=px)
            cx, cy = base["center"]
            majpt = self._to_px(cl[0] + g["rmaj"] * g["maj"][0],
                                cl[1] + g["rmaj"] * g["maj"][1], px=px)
            minpt = self._to_px(cl[0] + g["rmin"] * g["min"][0],
                                cl[1] + g["rmin"] * g["min"][1], px=px)
            base["rmaj_px"] = round(math.hypot(majpt[0] - cx, majpt[1] - cy), 2)
            base["rmin_px"] = round(math.hypot(minpt[0] - cx, minpt[1] - cy), 2)
            base["rmaj_mm"] = round(g["rmaj"], 3)
            base["rmin_mm"] = round(g["rmin"], 3)
            base["rot_deg"] = round(math.degrees(math.atan2(majpt[1] - cy,
                                                            majpt[0] - cx)) % 180.0, 2)
            # partial ellipse (occlusion-split arc): swept eccentric angles, same
            # start->end-with-increasing-theta convention as arc.
            if g.get("mid") and g.get("p1") and g.get("p2"):
                base["p1"] = self._to_px(*g["p1"], px=px)
                base["p2"] = self._to_px(*g["p2"], px=px)
                mpx = self._to_px(*g["mid"], px=px)
                sa, ea = _ellipse_angles_px(base["center"], majpt, minpt,
                                            base["p1"], base["p2"], mpx)
                base["start_angle"] = sa
                base["end_angle"] = ea

        base["bbox_px"] = _prim_bbox_px(base)

        # --- MATCH WITH ORACLE ---
        origins = []
        for op in self.oracle_prims:
            # We match using local (lx, ly) space against Oracle (u, v) space
            if typ == "line" and op["type"] == "line":
                if point_on_segment(g["p1"][0], g["p1"][1], op["p1"][0], op["p1"][1], op["p2"][0], op["p2"][1]) and \
                   point_on_segment(g["p2"][0], g["p2"][1], op["p1"][0], op["p1"][1], op["p2"][0], op["p2"][1]):
                    origins.extend(op["topo_origins"])
            elif typ in ("circle", "arc") and op["type"] in ("circle", "arc"):
                if points_same(g["center"], op["center"]) and abs(g["r"] - op.get("radius", op.get("r", 0))) < 1e-3:
                    origins.extend(op["topo_origins"])
            elif typ == "ellipse" and op["type"] == "ellipse":
                # an occlusion-split arc shares the full ellipse's center + axes, so
                # match on those (angles ignored) — like circle/arc radius matching.
                if points_same(g["center"], op["center"]) and \
                   abs(g["rmaj"] - op["rmaj"]) < 1e-2 and abs(g["rmin"] - op["rmin"]) < 1e-2:
                    origins.extend(op["topo_origins"])
            elif typ == "polyline" and op["type"] == "line":
                # For polyline (rare but possible), check if all discretised points lie on the line
                all_on_line = True
                for pt in g["pts"]:
                    if not point_on_segment(pt[0], pt[1], op["p1"][0], op["p1"][1], op["p2"][0], op["p2"][1]):
                        all_on_line = False
                        break
                if all_on_line:
                    origins.extend(op["topo_origins"])

        unique_origins = []
        seen = set()
        for o in origins:
            if o["id"] not in seen:
                seen.add(o["id"])
                unique_origins.append(o)

        base["prov"] = {"topo_origins": unique_origins}
        # -------------------------

        return base


def _prim_coord_sig(rec):
    """Flat coord signature for duplicate detection (endpoint order normalized)."""
    t = rec["type"]
    if t == "line":
        a, b = sorted([tuple(rec["p1"]), tuple(rec["p2"])])
        return a + b
    if t == "polyline":
        pts = [tuple(q) for q in rec["pts"]]
        if pts and pts[0] > pts[-1]:
            pts = pts[::-1]
        return tuple(c for q in pts for c in q)
    if t == "circle":
        return tuple(rec["center"]) + (rec["r_px"],)
    if t == "arc":
        a, b = sorted([tuple(rec["p1"]), tuple(rec["p2"])])
        return tuple(rec["center"]) + (rec["r_px"],) + a + b
    if t == "ellipse":
        return tuple(rec["center"]) + (rec["rmaj_px"], rec["rmin_px"], rec["rot_deg"])
    return ()


def _arc_angles_px(c, p1, p2, mid):
    """(start_angle, end_angle) deg in the px frame; arc = start -> end with
    increasing atan2(y-cy, x-cx) mod 360, chosen so it passes through mid."""
    def ang(p):
        return math.degrees(math.atan2(p[1] - c[1], p[0] - c[0])) % 360.0
    t1, tm, t2 = ang(p1), ang(mid), ang(p2)
    if (tm - t1) % 360.0 <= (t2 - t1) % 360.0 + 1e-9:
        sa, ea = t1, t2
    else:
        sa, ea = t2, t1
    return round(sa, 2), round(ea, 2)


def _ellipse_angles_px(c, majpt, minpt, p1, p2, mid):
    """(start, end) eccentric angles (deg, px frame) of p1,p2 on the ellipse whose
    px-frame semi-axes are center->majpt (major) and center->minpt (minor). The arc
    runs start->end with increasing theta (mod 360), chosen so it passes through mid."""
    ux, uy = majpt[0] - c[0], majpt[1] - c[1]
    vx, vy = minpt[0] - c[0], minpt[1] - c[1]
    ru2, rv2 = (ux * ux + uy * uy) or 1.0, (vx * vx + vy * vy) or 1.0
    def ecc(p):
        dx, dy = p[0] - c[0], p[1] - c[1]
        a = (dx * ux + dy * uy) / ru2   # cos t
        b = (dx * vx + dy * vy) / rv2   # sin t
        return math.degrees(math.atan2(b, a)) % 360.0
    t1, tm, t2 = ecc(p1), ecc(mid), ecc(p2)
    if (tm - t1) % 360.0 <= (t2 - t1) % 360.0 + 1e-9:
        sa, ea = t1, t2
    else:
        sa, ea = t2, t1
    return round(sa, 2), round(ea, 2)


def _prim_bbox_px(p):
    xs, ys = [], []
    def add(pt): xs.append(pt[0]); ys.append(pt[1])
    if p["type"] in ("line",):
        add(p["p1"]); add(p["p2"])
    elif p["type"] == "polyline":
        for q in p["pts"]: add(q)
    elif p["type"] in ("circle", "arc"):
        cx, cy = p["center"]; r = p["r_px"]
        xs += [cx - r, cx + r]; ys += [cy - r, cy + r]
    elif p["type"] == "ellipse":
        cx, cy = p["center"]; r = max(p["rmaj_px"], p["rmin_px"])
        xs += [cx - r, cx + r]; ys += [cy - r, cy + r]
    if not xs:
        return [0, 0, 0, 0]
    return [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)]

# ============================================================================
#  isometric
# ============================================================================

def iso_group(shape, ox, oy, scale):
    res = TechDraw.projectEx(shape, App.Vector(1, -1, 1))
    edges = compound_edges(res[0]) + compound_edges(res[1]) + compound_edges(res[3])
    xs, ys = [], []
    for e in edges:
        b = e.BoundBox; xs += [b.XMin, b.XMax]; ys += [b.YMin, b.YMax]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    tr = ("translate(%.4f %.4f) scale(%.5f %.5f) translate(%.4f %.4f)"
          % (ox, oy, scale, -scale, -xmin, -ymax))
    parts = ['<g transform="%s">' % tr]
    for e in edges:
        d = edge_to_path(e)
        if d:
            parts.append('<path class="visible" d="%s"/>' % d)
    parts.append("</g>")
    return "\n".join(parts), (xmax - xmin) * scale, (ymax - ymin) * scale

# ============================================================================
#  main build  -> writes SVG, returns graph dict (px coords) + meta
# ============================================================================

PX_DEFAULT_W = 1800  # raster width

def build(step_path, out_svg, partname=None, out_width=PX_DEFAULT_W):
    shape = Part.Shape(); shape.read(step_path)
    bb = shape.BoundBox
    L, W, H = bb.XLength, bb.YLength, bb.ZLength
    if partname is None:
        partname = os.path.splitext(os.path.basename(step_path))[0].upper()

    # --- Phase 4 Oracle Extractor ---
    cad_projector = CADProjector(shape)
    graph_builder = GraphBuilder(shape)

    front_prims = graph_builder.enrich_topo_origins(cad_projector.project((0, -1, 0)))
    top_prims = graph_builder.enrich_topo_origins(cad_projector.project((0, 0, 1)))
    right_prims = graph_builder.enrich_topo_origins(cad_projector.project((1, 0, 0)))
    # --------------------------------

    front = View("front", shape, (0, -1, 0), front_prims)
    top   = View("top",   shape, (0, 0, 1),  top_prims)
    right = View("right", shape, (1, 0, 0),  right_prims)

    SW, SH = 420.0, 297.0
    MARGIN = 10.0; TB_H = 38.0; TB_W = 180.0
    PXMM = out_width / SW   # mm -> pixel (cairosvg maps viewBox mm linearly to width px)

    DIMPAD = 30.0; VGAP = 42.0
    draw_x0, draw_y0 = MARGIN + 10, MARGIN + 12
    draw_x1 = MARGIN + 235
    draw_y1 = SH - MARGIN - 16
    avail_w = (draw_x1 - draw_x0) - 2 * DIMPAD
    avail_h = (draw_y1 - draw_y0) - 2 * DIMPAD

    cluster_mw = front.w + right.w
    cluster_mh = top.h + front.h
    nice = [5, 2, 1, 0.75, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
    scale = nice[-1]
    for s in nice:
        if cluster_mw * s + VGAP <= avail_w and cluster_mh * s + VGAP <= avail_h:
            scale = s; break

    for v in (front, top, right):
        v.set_layout(0, 0, scale)
    clu_w = front.sheet_w() + VGAP + right.sheet_w()
    clu_h = top.sheet_h() + VGAP + front.sheet_h()
    cx0 = draw_x0 + DIMPAD + (avail_w - clu_w) / 2.0
    cy0 = draw_y0 + DIMPAD + (avail_h - clu_h) / 2.0
    top_y = cy0
    front_y = top_y + top.sheet_h() + VGAP
    front_x = cx0
    right_x = front_x + front.sheet_w() + VGAP
    front.set_layout(front_x, front_y, scale)
    top.set_layout(front_x, top_y, scale)
    right.set_layout(right_x, front_y, scale)

    # ---- SVG header ----
    css = """
    .visible { stroke:#111; stroke-width:0.5; fill:none; stroke-linecap:round; stroke-linejoin:round; }
    .hidden  { stroke:#111; stroke-width:0.3; fill:none; stroke-dasharray:2.2,1.6; }
    .center  { stroke:#111; stroke-width:0.25; stroke-dasharray:6,1.5,1,1.5; }
    .dim     { stroke:#111; stroke-width:0.3; fill:none; }
    .ext     { stroke:#111; stroke-width:0.25; fill:none; }
    .dimtxt  { font-family:'DejaVu Sans','Arial',sans-serif; font-size:4.0px; fill:#111; }
    .frame   { stroke:#111; fill:none; }
    .tbtxt   { font-family:'DejaVu Sans','Arial',sans-serif; fill:#111; }
    """
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="%dmm" height="%dmm" viewBox="0 0 %g %g">'
             % (int(SW), int(SH), SW, SH), '<style>%s</style>' % css,
             '<rect x="0" y="0" width="%g" height="%g" fill="white"/>' % (SW, SH)]

    # frame + zones
    parts.append('<rect class="frame" x="%g" y="%g" width="%g" height="%g" stroke-width="0.7"/>'
                 % (MARGIN, MARGIN, SW - 2 * MARGIN, SH - 2 * MARGIN))
    parts.append('<rect class="frame" x="%g" y="%g" width="%g" height="%g" stroke-width="0.35"/>'
                 % (MARGIN + 4, MARGIN + 4, SW - 2 * MARGIN - 8, SH - 2 * MARGIN - 8))
    fxs = SW - 2 * (MARGIN + 4)
    for i in range(8):
        zx = MARGIN + 4 + fxs * (i + 0.5) / 8
        parts.append('<text class="tbtxt" x="%.1f" y="%.1f" font-size="3" text-anchor="middle">%d</text>' % (zx, MARGIN + 3, i + 1))
        parts.append('<text class="tbtxt" x="%.1f" y="%.1f" font-size="3" text-anchor="middle">%d</text>' % (zx, SH - MARGIN - 1.2, i + 1))
    fys = SH - 2 * (MARGIN + 4)
    for j in range(4):
        zy = MARGIN + 4 + fys * (j + 0.5) / 4
        lab = "ABCD"[j]
        parts.append('<text class="tbtxt" x="%.1f" y="%.1f" font-size="3" text-anchor="middle">%s</text>' % (MARGIN + 2, zy + 1, lab))
        parts.append('<text class="tbtxt" x="%.1f" y="%.1f" font-size="3" text-anchor="middle">%s</text>' % (SW - MARGIN - 2, zy + 1, lab))

    # geometry
    parts.append(top.svg_group())
    parts.append(front.svg_group())
    parts.append(right.svg_group())

    # isometric (no GT for iso primitives; pictorial only)
    iso_band_x = MARGIN + 248
    iso_band_w = (SW - MARGIN - 8) - iso_band_x
    iso_scale = scale * 0.8
    iso_oy = MARGIN + 30
    iso_svg, iw, ih = iso_group(shape, iso_band_x + iso_band_w/2, iso_oy, iso_scale) # Approximated placement
    parts.append(iso_svg)

    # =====================================================================
    #  GT primitives (PNG pixel space) matched with Oracle
    # =====================================================================
    # model_origin (v0.3.1): the MODEL-frame coordinate (mm, the source STEP's frame)
    # at frame.origin_px. origin_px sits at the view's content min in px, and content
    # spans the full part under orthographic projection, so per signed axis it is the
    # model bbox min ('+': px grows with the axis) or max ('-'). With axis_remap +
    # px_per_mm this makes px -> model exact without the per-view shift discovery
    # train3d/serialize.py documents:  m = model_origin + sign * (px - origin_px) / px_per_mm.
    _bb_rng = {"X": (bb.XMin, bb.XMax), "Y": (bb.YMin, bb.YMax), "Z": (bb.ZMin, bb.ZMax)}

    def _model_origin(vname):
        rm = _AXIS_REMAP[vname]
        return [round(_bb_rng[spec[1]][0 if spec[0] == "+" else 1], 3)
                for spec in (rm["px_x"], rm["px_y"])]

    views_gt = []
    for v, pfx in ((front, "F"), (top, "T"), (right, "R")):
        prims = v.primitive_records(pfx, PXMM)
        views_gt.append({
            "name": v.name,
            "view_type": v.name,
            "projection_dir": _PROJ_DIR.get(v.name, [0, 0, -1]),
            "align_to": None,
            "frame": {
                "origin_px": [round(v.ox * PXMM, 2), round(v.oy * PXMM, 2)],
                "scale": v.scale,
                "px_per_mm": round(v.scale * PXMM, 4),
                "axis_remap": _AXIS_REMAP[v.name],
                "model_origin": _model_origin(v.name),
            },
            "primitives": prims,
        })

    # =====================================================================
    #  Dimensions (Delegated to LegacyDimensioner)
    # =====================================================================
    views_obj = {"front": front, "top": top, "right": right}
    dim_engine = LegacyDimensioner(shape, views_gt, views_obj, PXMM)
    dims_svg, dims_gt, features = dim_engine.annotate()

    parts.append('<g>%s</g>' % "\n".join(dims_svg))

    # labels + symbol + title block
    parts.append('<text class="tbtxt" x="%.1f" y="%.1f" font-size="4.5" text-anchor="middle">ISOMETRIC</text>'
                 % (iso_band_x + iso_band_w / 2, iso_oy + ih + 6))
    sym_x, sym_y = SW - MARGIN - 4 - TB_W - 26, SH - MARGIN - 4 - 14
    parts.append(_third_angle_symbol(sym_x, sym_y))
    parts.append(_title_block(SW - MARGIN - 4 - TB_W, SH - MARGIN - 4 - TB_H, TB_W, TB_H,
                              partname, scale, L, W, H))
    parts.append('</svg>')
    svg = "\n".join(parts)
    with open(out_svg, "w") as f:
        f.write(svg)

    rnum, rden = _ratio(scale)

    graph = {
        "amvdg_version": "0.3.1",
        "part_id": partname,
        "profile": "vectorized",
        "source": {"kind": "synthetic_gt", "image": None, "extractor": None, "scan_affine": None},
        "units": "mm",
        "angle_units": "deg",
        "world": {"handedness": "right", "up_axis": "Z", "bbox_3d": [round(L, 3), round(W, 3), round(H, 3)]},
        "coord_system": {"px_origin": "top_left", "y_axis_down": True},
        "sheet": {"size": "A3", "projection": "third_angle",
                  "scale": [int(rnum) if rnum.isdigit() else float(rnum), int(rden) if rden.isdigit() else float(rden)],
                  "width_px": out_width, "height_px": int(round(SH * PXMM)), "px_per_mm": round(PXMM, 4)},
        "views": views_gt,
        "annotations": dims_gt,
        "features": features,
        "dof": {"required": 0, "determined": 0, "determined_by_geometry": 0, "supplied_by_prior": [], "undetermined": [], "missing": 0, "fully_constrained": False, "coverage": 0.0, "self_declared": {"required": 0, "determined": 0, "missing": 0}},
    }

    meta = {"L": L, "W": W, "H": H, "scale": scale, "width_px": out_width,
            "height_px": int(round(SH * PXMM)), "pxmm": PXMM}
    return svg, graph, meta


def _third_angle_symbol(x, y):
    s = ['<g>']
    s.append('<text class="tbtxt" x="%.1f" y="%.1f" font-size="3" text-anchor="middle">THIRD ANGLE PROJECTION</text>' % (x + 9, y - 4))
    s.append('<polygon points="%.1f,%.1f %.1f,%.1f %.1f,%.1f %.1f,%.1f" class="frame" stroke-width="0.4"/>'
             % (x, y + 4, x + 12, y + 1, x + 12, y + 11, x, y + 8))
    s.append('<line class="frame" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke-width="0.4"/>' % (x + 4, y, x + 4, y + 12))
    s.append('<line class="frame" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke-width="0.4"/>' % (x + 8, y, x + 8, y + 12))
    s.append('<circle cx="%.1f" cy="%.1f" r="5.5" class="frame" stroke-width="0.4"/>' % (x + 22, y + 6))
    s.append('<circle cx="%.1f" cy="%.1f" r="2.2" class="frame" stroke-width="0.4"/>' % (x + 22, y + 6))
    s.append('<line class="frame" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke-width="0.3"/>' % (x + 14, y + 6, x + 30, y + 6))
    s.append('<line class="frame" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke-width="0.3"/>' % (x + 22, y - 1, x + 22, y + 13))
    s.append('</g>')
    return "\n".join(s)


def _title_block(x, y, w, h, name, scale, L, W, H):
    s = ['<g>']
    s.append('<rect class="frame" x="%.1f" y="%.1f" width="%.1f" height="%.1f" stroke-width="0.7"/>' % (x, y, w, h))
    for ry in [y + h * 0.45, y + h * 0.72]:
        s.append('<line class="frame" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke-width="0.3"/>' % (x, ry, x + w, ry))
    for cx in [x + w * 0.55, x + w * 0.78]:
        s.append('<line class="frame" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke-width="0.3"/>' % (cx, y, cx, y + h * 0.45))
    def t(tx, ty, txt, size=3.0, bold=False, anchor="start"):
        return ('<text class="tbtxt" x="%.1f" y="%.1f" font-size="%g" text-anchor="%s"%s>%s</text>'
                % (tx, ty, size, anchor, ' font-weight="bold"' if bold else '', txt))
    s.append(t(x + 3, y + h * 0.30, name, 7.5, True))
    s.append(t(x + 3, y + h * 0.40, "drawing2cad  synthetic drawing (FreeCAD TechDraw HLR)", 2.4))
    s.append(t(x + w * 0.56, y + h * 0.20, "DRAWN", 2.4))
    s.append(t(x + w * 0.79, y + h * 0.20, "drawing2cad", 2.6, True))
    s.append(t(x + w * 0.56, y + h * 0.36, "CHK", 2.4))
    s.append(t(x + w * 0.79, y + h * 0.36, "PoC", 2.6))
    s.append(t(x + 3, y + h * 0.62, "MATERIAL:  STEEL", 2.8))
    s.append(t(x + w * 0.40, y + h * 0.62, "SCALE  %s : %s" % (_ratio(scale)), 2.8, True))
    s.append(t(x + w * 0.72, y + h * 0.62, "SIZE A3", 2.8))
    s.append(t(x + 3, y + h * 0.90, "ENVELOPE  %g x %g x %g mm" % (round(L), round(W), round(H)), 2.8))
    s.append(t(x + w * 0.55, y + h * 0.90, "UNITS: mm   THIRD ANGLE", 2.6))
    s.append(t(x + w * 0.86, y + h * 0.90, "SHEET 1/1", 2.6, True))
    s.append('</g>')
    return "\n".join(s)


def _ratio(scale):
    std = {5.0: ("5", "1"), 2.0: ("2", "1"), 1.0: ("1", "1"),
           0.75: ("3", "4"), 0.5: ("1", "2"), 0.4: ("2", "5"),
           0.3: ("3", "10"), 0.25: ("1", "4"), 0.2: ("1", "5"),
           0.15: ("3", "20"), 0.1: ("1", "10"), 0.05: ("1", "20")}
    for k, v in std.items():
        if abs(scale - k) < 1e-6:
            return v
    if scale >= 1:
        return ("%g" % round(scale, 2), "1")
    return ("1", "%g" % round(1.0 / scale, 2))


def render_one(step_path, out_dir, name=None, width=PX_DEFAULT_W):
    os.makedirs(out_dir, exist_ok=True)
    if name is None:
        name = os.path.splitext(os.path.basename(step_path))[0]
    svg_path = os.path.join(out_dir, name + ".svg")
    svg, graph, meta = build(step_path, svg_path, name.upper(), width)
    graph_path = os.path.join(out_dir, name + ".graph.json")
    with open(graph_path, "w") as f:
        json.dump(graph, f, indent=1)
    return {"name": name, "svg": svg_path, "graph": graph_path, "meta": meta,
            "counts": _counts(graph)}


def _counts(graph):
    nv = sum(1 for gv in graph["views"] for p in gv["primitives"] if p["line_role"] == "visible")
    nh = sum(1 for gv in graph["views"] for p in gv["primitives"] if p["line_role"] == "hidden")
    by_type = Counter(d["kind"] for d in graph["annotations"])
    nrefless = sum(1 for d in graph["annotations"] if not d["refs"])
    return {"nv": nv, "nh": nh, "types": dict(by_type), "nrefless": nrefless}

def _run_batch(step_dir, out_dir, width, limit, logf):
    """Render every *.step under step_dir into out_dir (per-part isolation, OK/SKIP log)."""
    jobs = [(sp, os.path.splitext(os.path.basename(sp))[0])
            for sp in sorted(glob.glob(os.path.join(step_dir, "*.step")))]
    if limit > 0:
        jobs = jobs[:limit]
    results = []
    for sp, nm in tqdm(jobs):
        # Per-part isolation: a single bad solid (HLR throw, null shape) must
        # not abort the whole batch over hundreds of varied seeds.
        try:
            r = render_one(sp, out_dir, nm, width)
        except Exception:
            import traceback
            logf.write("SKIP %s\n%s\n" % (nm, traceback.format_exc())); logf.flush()
            continue
        results.append(r)
        logf.write("OK %s counts=%s\n" % (nm, json.dumps(r["counts"]))); logf.flush()
    with open(os.path.join(out_dir, "_build_results.json"), "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "meta"} for r in results], f, indent=1)


def _main():
    """CLI entry point. Two mutually-exclusive modes:
      --step-dir <dir>  batch: render every *.step under it into --out-dir
      --step <file>     single: render one part into --out-dir
    """
    parser = argparse.ArgumentParser(description="STEP -> multi-view drawing SVG + AMVDG graph JSON")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--step-dir", help="render every *.step under this dir (batch)")
    mode.add_argument("--step", help="render this single *.step file")
    parser.add_argument("--out-dir", default="dataset", help="output dir for <name>.svg/.graph.json")
    parser.add_argument("--name", default=None, help="part name (single mode; default = file stem)")
    parser.add_argument("--width", type=int, default=PX_DEFAULT_W, help="raster width in px")
    parser.add_argument("--limit", type=int, default=0, help="cap number of parts (batch mode)")
    parser.add_argument("--log", default="render_dataset.log", help="OK/SKIP/FAIL log path")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logf = open(args.log, "w")
    try:
        if args.step_dir:
            _run_batch(args.step_dir, args.out_dir, args.width, args.limit, logf)
        else:
            r = render_one(args.step, args.out_dir, args.name, args.width)
            logf.write("OK %s counts=%s\n" % (r["name"], json.dumps(r["counts"])))
    except Exception:
        import traceback
        logf.write("FAIL\n" + traceback.format_exc())
    logf.flush(); logf.close()


if __name__ == "__main__":
    _main()
