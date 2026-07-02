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

import sys, os, math, json
import freecad          # conda-forge shim: puts FreeCAD's libs on sys.path
import FreeCAD as App
import Part
import TechDraw
from collections import Counter

from scripts.renderer.cad_projector import CADProjector
from scripts.renderer.graph_builder import GraphBuilder
from scripts.renderer.amvdg_exporter import AMVDGExporter
from scripts.renderer.dimensioner import LegacyDimensioner

# viewing direction recorded per ortho view (third-angle); only used as recorded metadata.
_PROJ_DIR = {"front": [0, -1, 0], "top": [0, 0, -1], "right": [1, 0, 0]}

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
        try:
            a0 = c.parameter(App.Vector(p0.x, p0.y, 0))
            a1 = c.parameter(App.Vector(p1.x, p1.y, 0))
            da = a1 - a0
            large = 1 if abs(da) > math.pi else 0
            sweep = 1 if da > 0 else 0
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
            a0 = math.degrees(c.parameter(App.Vector(p0.x, p0.y, 0)))
            a1 = math.degrees(c.parameter(App.Vector(p1.x, p1.y, 0)))
        except Exception:
            a0 = a1 = 0.0
        return ("arc", {"center": cen, "r": r, "a0": a0, "a1": a1,
                        "p1": (p0.x, p0.y), "p2": (p1.x, p1.y)})
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
        """Yield GT primitive dicts in PNG-pixel space, MATCHED with Oracle topo_origins."""
        out = []
        n = 0
        for vis_tag, edges in (("visible", self.edges_vis), ("hidden", self.edges_hid)):
            for e in edges:
                typ, g = classify_edge(e)
                rec = self._record(idprefix, n, typ, g, vis_tag, px)
                if rec:
                    out.append(rec); n += 1
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
            if o not in seen:
                seen.add(o)
                unique_origins.append(o)
                
        base["prov"] = {"topo_origins": unique_origins}
        # -------------------------
        
        return base


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
    .center  { stroke:#c00; stroke-width:0.25; stroke-dasharray:6,1.5,1,1.5; }
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
    
    # Use AMVDGExporter to build final JSON
    exporter = AMVDGExporter(partname)
    
    graph = {
        "amvdg_version": "0.3",
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

if __name__ == "__main__":
    p = os.environ.get("RF_STEP")
    if p:
        render_one(p, os.path.dirname(os.environ.get("RF_OUT", "out.svg")),
                   os.environ.get("RF_NAME", "PART"))
    elif os.environ.get("RF_BATCH"):
        for f in ["Bracket.step", "Flange.step"]:
            render_one(os.path.join("FreeCAD_scrape/gt_prim5_fcstd", f), "poc/dataset")
