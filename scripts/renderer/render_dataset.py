#!/usr/bin/env python
# render_dataset.py -- STEP -> (realistic drawing PNG) + (ground-truth Tier-1/Tier-2 graph JSON)
#                      + scan-noise augmentation, self-verify overlay, batch + manifest.
#
# Reuses the proven headless recipe: TechDraw.project() under freecadcmd composes the SVG
# ourselves, then cairosvg rasterizes. The DrawViewPart/GUI path is NOT used (empty/hangs headless).
#
# The renderer now RECORDS every primitive it draws (with a stable id, view, visibility,
# coarse feature tag, and geometry) and every dimension (with the primitive ids it measures
# in `refs`). All coordinates in the emitted graph JSON are in FINAL PNG PIXEL SPACE, so a
# downstream model's predictions can be scored directly on the rendered image.
#
# Two-stage coordinate pipeline:
#   model-axis (cu,cv) --View.M()--> SVG sheet-mm (x,y) --raster--> PNG px = sheet_mm * PXMM
# where PXMM = output_width_px / sheet_width_mm.  cairosvg maps the SVG user units (=mm, since
# viewBox==mm) linearly to pixels, so pixel = mm * PXMM with no offset.
#
# Run (drawing2cad env; paths via env vars, one per setting):
#   RF_STEP=.../Bracket.step RF_OUT=.../out.svg RF_NAME=BRACKET \
#     python render_dataset.py
# or batch mode:
#   RF_BATCH=1  python render_dataset.py     (renders Bracket+Flange into poc/dataset/)
# (usually driven by batch_dataset.py, which sets the RF_* env and runs this under `python`.)
#
import sys, os, math, json
import freecad          # conda-forge shim: puts FreeCAD's libs on sys.path
import FreeCAD as App
import Part
import TechDraw
from collections import Counter

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


# ============================================================================
#  View
# ============================================================================

class View:
    REMAP = {
        "front": (0, 1, -1, 0),
        "top":   (1, 0, 0, 1),
        "right": (0, -1, 1, 0),
    }

    def __init__(self, name, shape, direction):
        self.name = name
        # projectEx groups: [0]V sharp [1]V1 smooth [2]VN seam [3]VO outline
        # [4]VI iso [5]H sharp [6]H1 smooth [7]HN seam [8]HO outline [9]HI iso.
        # project() (4-tuple) DROPS the outline groups -> curved-surface silhouettes
        # (e.g. a cylindrical boss/bore wall) go missing. Include VO/HO so the drawing
        # — and the Tier-1 GT primitives derived from it — are geometrically complete.
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
        """Yield GT primitive dicts in PNG-pixel space. px = mm->pixel factor.
        Each edge's local (lx,ly) -> canonical (cu,cv) -> sheet-mm -> pixel."""
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
        base = {"id": rid, "type": typ, "visibility": vis_tag, "feature": feat}
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
        # bbox in px for scoring
        base["bbox_px"] = _prim_bbox_px(base)
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
#  dimension drawing (SVG) -- returns (svg_str, text_anchor_mm)
# ============================================================================

ARROW = 3.2
GAP = 1.5
EXT = 9.0


def _arrow(x, y, ang, fill="black"):
    a = ARROW; w = a * 0.32
    bx, by = x - a * math.cos(ang), y - a * math.sin(ang)
    px, py = -math.sin(ang) * w, math.cos(ang) * w
    return ('<polygon points="%.2f,%.2f %.2f,%.2f %.2f,%.2f" fill="%s"/>'
            % (x, y, bx + px, by + py, bx - px, by - py, fill))


def hdim(x1, x2, y_geom, y_dim, text):
    s = []
    dirn = 1.0 if y_dim > y_geom else -1.0
    y_start = y_geom + dirn * GAP
    y_end = y_dim + dirn * EXT
    for xx in (x1, x2):
        s.append('<line class="ext" x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f"/>'
                 % (xx, y_start, xx, y_end))
    lo, hi = sorted((x1, x2))
    s.append('<line class="dim" x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f"/>' % (lo, y_dim, hi, y_dim))
    s.append(_arrow(lo, y_dim, math.pi))
    s.append(_arrow(hi, y_dim, 0))
    tx, ty = (lo + hi) / 2, y_dim - 1.4
    s.append('<text class="dimtxt" x="%.2f" y="%.2f" text-anchor="middle">%s</text>' % (tx, ty, text))
    return "\n".join(s), (tx, ty)


def vdim(y1, y2, x_geom, x_dim, text):
    s = []
    dirn = 1.0 if x_dim > x_geom else -1.0
    x_start = x_geom + dirn * GAP
    x_end = x_dim + dirn * EXT
    for yy in (y1, y2):
        s.append('<line class="ext" x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f"/>'
                 % (x_start, yy, x_end, yy))
    lo, hi = sorted((y1, y2))
    s.append('<line class="dim" x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f"/>' % (x_dim, lo, x_dim, hi))
    s.append(_arrow(x_dim, lo, -math.pi / 2))
    s.append(_arrow(x_dim, hi, math.pi / 2))
    tx, ty = x_dim - 1.4, (lo + hi) / 2
    s.append('<text class="dimtxt" x="%.2f" y="%.2f" text-anchor="middle" '
             'transform="rotate(-90 %.2f %.2f)">%s</text>' % (tx, ty, tx, ty, text))
    return "\n".join(s), (tx, ty)


def diadim(cx, cy, r, ang_deg, text):
    a = math.radians(ang_deg)
    x1, y1 = cx - r * math.cos(a), cy - r * math.sin(a)
    x2, y2 = cx + r * math.cos(a), cy + r * math.sin(a)
    lx, ly = x2 + 10 * math.cos(a), y2 + 10 * math.sin(a)
    s = ['<line class="dim" x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f"/>' % (x1, y1, lx, ly)]
    s.append(_arrow(x1, y1, a + math.pi))
    s.append(_arrow(x2, y2, a))
    anchor = "start" if math.cos(a) >= 0 else "end"
    tx = lx + (1.5 if anchor == "start" else -1.5); ty = ly - 1.2
    s.append('<text class="dimtxt" x="%.2f" y="%.2f" text-anchor="%s">%s</text>' % (tx, ty, anchor, text))
    return "\n".join(s), (tx, ty)


def centerlines_for(cx, cy, r, ext=2.5):
    L = r + ext
    return ('<line class="center" x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f"/>'
            '<line class="center" x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f"/>'
            % (cx - L, cy, cx + L, cy, cx, cy - L, cx, cy + L))


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
#  3D feature extraction (for dims + correspondences)
# ============================================================================

def axis_dir(ax):
    for nm, comp in (("X", ax.x), ("Y", ax.y), ("Z", ax.z)):
        if abs(abs(comp) - 1.0) < 1e-3 and abs(ax.x) + abs(ax.y) + abs(ax.z) - 1 < 1e-2:
            return nm
    # not axis-aligned
    return None


def extract_cylinders(shape):
    """Return list of holes/bosses: {center:(x,y,z), r, axis:'X'|'Y'|'Z', id}."""
    feats = []
    for f in shape.Faces:
        srf = f.Surface
        if srf.TypeId == "Part::GeomCylinder":
            a = axis_dir(srf.Axis)
            if a is None:
                continue
            c = srf.Center
            feats.append({"center": (c.x, c.y, c.z), "r": srf.Radius, "axis": a})
    # de-dup by (axis, rounded perpendicular-center, radius)
    seen = {}
    uniq = []
    for fe in feats:
        cx, cy, cz = fe["center"]
        key = (fe["axis"], round(cx, 1), round(cy, 1), round(cz, 1), round(fe["r"], 1))
        if key in seen:
            continue
        seen[key] = True
        fe = dict(fe); fe["id"] = "cyl%d" % len(uniq)
        uniq.append(fe)
    return uniq


# ============================================================================
#  main build  -> writes SVG, returns graph dict (px coords) + meta
# ============================================================================

def fmt(v):
    return ("%g" % round(v, 1))

PX_DEFAULT_W = 1800  # raster width


def build(step_path, out_svg, partname=None, out_width=PX_DEFAULT_W):
    shape = Part.Shape(); shape.read(step_path)
    bb = shape.BoundBox
    L, W, H = bb.XLength, bb.YLength, bb.ZLength
    if partname is None:
        partname = os.path.splitext(os.path.basename(step_path))[0].upper()

    front = View("front", shape, (0, -1, 0))
    top   = View("top",   shape, (0, 0, 1))
    right = View("right", shape, (1, 0, 0))

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
    _res = TechDraw.projectEx(shape, App.Vector(1, -1, 1))
    _e = compound_edges(_res[0]) + compound_edges(_res[1]) + compound_edges(_res[3])
    _xs = []
    for _ed in _e:
        _b = _ed.BoundBox; _xs += [_b.XMin, _b.XMax]
    _iw = (max(_xs) - min(_xs)) * iso_scale
    iso_ox = iso_band_x + max(0, (iso_band_w - _iw) / 2.0)
    iso_svg, iw, ih = iso_group(shape, iso_ox, iso_oy, iso_scale)
    parts.append(iso_svg)

    # =====================================================================
    #  GT primitives (PNG pixel space)
    # =====================================================================
    views_gt = []
    prim_index = {}   # (view, rounded geometry key) -> primitive id, for dim ref resolution
    for v, pfx in ((front, "F"), (top, "T"), (right, "R")):
        prims = v.primitive_records(pfx, PXMM)
        # build lookup tables for ref resolution (by feature geometry)
        for p in prims:
            prim_index[p["id"]] = p
        views_gt.append({
            "name": v.name,
            "origin_px": [round(v.ox * PXMM, 2), round(v.oy * PXMM, 2)],
            "scale": v.scale,
            "px_per_mm": round(v.scale * PXMM, 4),   # per-view px/mm = view scale * sheet PXMM (so r_px == r_mm * px_per_mm)
            "primitives": prims,
        })
    view_by_name = {gv["name"]: gv for gv in views_gt}

    def find_circle_prim(view_name, center_cu, center_cv, r, tolpx=4.0):
        """Resolve a circle/arc primitive in a view by its model-axis center+radius."""
        v = {"front": front, "top": top, "right": right}[view_name]
        sx, sy = v.M(center_cu, center_cv)
        tx, ty = sx * PXMM, sy * PXMM
        rpx = r * v.scale * PXMM
        best = None; bestd = 1e9
        for p in view_by_name[view_name]["primitives"]:
            if p["type"] not in ("circle", "arc"):
                continue
            d = math.hypot(p["center"][0] - tx, p["center"][1] - ty) + abs(p["r_px"] - rpx)
            if d < bestd:
                bestd = d; best = p
        if best and bestd < tolpx + rpx * 0.15:
            return best["id"]
        return None

    def find_extreme_line_refs(view_name, axis):
        """Resolve the 2 primitives that DEFINE the view's extent along an axis — one
        at each extreme. For each side, pick the primitive sitting AT the boundary:
        nearest to the extreme coordinate, tie-broken toward the smallest span along
        the measured axis (a true silhouette edge), then toward straight lines. This
        stops a sweeping bore arc — whose bbox merely spans the full width/height —
        from being mistaken for the defining edge. Geometry-derived, like the diameter
        dims; no per-part wiring.
        axis 'h' -> left/right extremes; 'v' -> top/bottom extremes."""
        gv = view_by_name[view_name]
        prims = [p for p in gv["primitives"] if p["visibility"] == "visible"]
        if not prims:
            return []
        i0, i1 = (0, 2) if axis == "h" else (1, 3)  # bbox indices for this axis
        lo = min(p["bbox_px"][i0] for p in prims)
        hi = max(p["bbox_px"][i1] for p in prims)

        def pick(target, idx):
            def key(p):
                bb = p["bbox_px"]
                span = bb[i1] - bb[i0]
                return (abs(bb[idx] - target), span, 0 if p["type"] == "line" else 1)
            return min(prims, key=key)["id"]

        refs = []
        for r in (pick(lo, i0), pick(hi, i1)):
            if r not in refs:
                refs.append(r)
        return refs

    # =====================================================================
    #  dimensions (draw + record GT with refs)
    # =====================================================================
    dims = []          # svg fragments
    dims_gt = []       # GT records
    dctr = [0]
    def add_dim(svg_frag, anchor_mm, dtype, subtype, value, view, refs, prov):
        dctr[0] += 1
        dims.append(svg_frag)
        dims_gt.append({
            "id": "D%d" % dctr[0], "type": dtype, "subtype": subtype,
            "value": round(value, 3), "refs": refs, "view": view,
            "text_px": [round(anchor_mm[0] * PXMM, 2), round(anchor_mm[1] * PXMM, 2)],
            "provenance": prov,
        })

    # FRONT overall length (model X) below, height (model Z) left
    fLx = front.M(front.umin, front.vmin)[0]
    fRx = front.M(front.umax, front.vmin)[0]
    fB  = front.M(front.umin, front.vmin)[1]
    fT  = front.M(front.umin, front.vmax)[1]
    frag, anc = hdim(fLx, fRx, fB, fB + 22, fmt(front.w))
    add_dim(frag, anc, "linear", "horizontal", front.w, "front",
            find_extreme_line_refs("front", "h"),
            {"feature": "bbox", "param": "dx"})
    frag, anc = vdim(fT, fB, fLx, fLx - 18, fmt(front.h))
    add_dim(frag, anc, "linear", "vertical", front.h, "front",
            find_extreme_line_refs("front", "v"),
            {"feature": "bbox", "param": "dz"})

    # TOP depth (model Y) left
    tTy = top.M(top.umin, top.vmax)[1]
    tBy = top.M(top.umin, top.vmin)[1]
    tLx = top.M(top.umin, top.vmin)[0]
    frag, anc = vdim(tTy, tBy, tLx, tLx - 18, fmt(top.h))
    add_dim(frag, anc, "linear", "vertical", top.h, "top",
            find_extreme_line_refs("top", "v"),
            {"feature": "bbox", "param": "dy"})

    # diameter dims: dimension EVERY distinct bore in the view where it projects as a circle
    # (Y-axis->front, Z-axis->top, X-axis->right), one dim per distinct (view,center,radius). This
    # raises dimension coverage toward full so the detector∪dimension fallback has bores to recover
    # (the old code dimensioned only 1 big + 1 small front bore => ~18% coverage).
    DIA = "⌀"
    cyls = extract_cylinders(shape)
    correspondences = []
    _axis_view = {"Y": ("front", front, lambda c: (c[0], c[2])),
                  "Z": ("top",   top,   lambda c: (c[0], c[1])),
                  "X": ("right", right, lambda c: (c[1], c[2]))}
    _seen_dim = set(); _off = {}
    for c in sorted(cyls, key=lambda c: -c["r"]):
        av = _axis_view.get(c["axis"])
        if not av:
            continue
        vname, vobj, proj = av
        cu, cv = proj(c["center"])
        key = (vname, round(cu, 1), round(cv, 1), round(c["r"], 2))
        if key in _seen_dim:
            continue
        ref = find_circle_prim(vname, cu, cv, c["r"])
        if not ref:                       # only dimension bores that actually project as a circle
            continue
        _seen_dim.add(key)
        sx, sy = vobj.M(cu, cv)
        _off[vname] = _off.get(vname, 0) + 1
        parts.append(centerlines_for(sx, sy, max(c["r"] * vobj.scale, 3)))
        frag, anc = diadim(sx, sy, c["r"] * vobj.scale, 30 + 14 * (_off[vname] % 4),
                           "%s%s" % (DIA, fmt(2 * c["r"])))
        add_dim(frag, anc, "diameter", None, 2 * c["r"], vname,
                [ref], {"feature": c["id"], "param": "diameter"})

    # =====================================================================
    #  Tier-2 correspondences: same 3D cylinder seen in multiple views
    # =====================================================================
    for c in cyls:
        cx, cy, cz = c["center"]
        per_view = {}
        # front sees Y-axis cyls as circles at (X,Z)
        if c["axis"] == "Y":
            rid = find_circle_prim("front", cx, cz, c["r"])
            if rid: per_view["front"] = rid
        # top sees Z-axis cyls as circles at (X,Y)
        if c["axis"] == "Z":
            rid = find_circle_prim("top", cx, cy, c["r"])
            if rid: per_view["top"] = rid
        # right sees X-axis cyls as circles; right canonical cu=Y, cv=Z
        if c["axis"] == "X":
            rid = find_circle_prim("right", cy, cz, c["r"])
            if rid: per_view["right"] = rid
        if len(per_view) >= 1:
            correspondences.append({"feature": c["id"], "axis": c["axis"],
                                    "r_mm": round(c["r"], 3), "views": per_view})

    parts.append('<g>%s</g>' % "\n".join(dims))

    # labels + symbol + title block
    parts.append('<text class="tbtxt" x="%.1f" y="%.1f" font-size="4.5" text-anchor="middle">ISOMETRIC</text>'
                 % (iso_ox + iw / 2, iso_oy + ih + 6))
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
        "part_id": partname,
        "units": "mm",
        "bbox_3d": [round(L, 3), round(W, 3), round(H, 3)],
        "sheet": {"size": "A3", "projection": "third_angle",
                  "scale": [int(rnum) if rnum.isdigit() else rnum,
                            int(rden) if rden.isdigit() else rden],
                  "width_px": out_width, "height_px": int(round(SH * PXMM)),
                  "px_per_mm": round(PXMM, 4)},
        "views": views_gt,
        "dimensions": dims_gt,
        "correspondences": correspondences,
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


# ============================================================================
#  driver: render one part -> svg + graph.json (rasterization done OUTSIDE
#  freecadcmd by the orchestrator, since cairosvg/PIL live in the venv).
# ============================================================================

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
    nv = sum(1 for gv in graph["views"] for p in gv["primitives"] if p["visibility"] == "visible")
    nh = sum(1 for gv in graph["views"] for p in gv["primitives"] if p["visibility"] == "hidden")
    by_type = Counter(d["type"] for d in graph["dimensions"])
    nrefless = sum(1 for d in graph["dimensions"] if not d["refs"])
    return {"prim_visible": nv, "prim_hidden": nh,
            "dims_total": len(graph["dimensions"]), "dims_by_type": dict(by_type),
            "dims_without_refs": nrefless,
            "correspondences": len(graph["correspondences"])}


def _main():
    parts_env = os.environ.get("RF_BATCH")
    width = int(os.environ.get("RF_WIDTH", PX_DEFAULT_W))
    logp = os.environ.get("RF_LOG", "render_dataset.log")
    logf = open(logp, "w")
    try:
        if parts_env:
            # batch: render every *.step under RF_STEPDIR (default cwd) into RF_OUTDIR.
            out_dir = os.environ.get("RF_OUTDIR", "dataset")
            import glob
            step_dir = os.environ.get("RF_STEPDIR", ".")
            jobs = [(sp, os.path.splitext(os.path.basename(sp))[0])
                    for sp in sorted(glob.glob(os.path.join(step_dir, "*.step")))]
            _lim = int(os.environ.get("RF_LIMIT", "0"))
            if _lim > 0:
                jobs = jobs[:_lim]
            results = []
            for sp, nm in jobs:
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
        else:
            step = os.environ["RF_STEP"]
            out_dir = os.environ.get("RF_OUTDIR", os.path.dirname(os.environ.get("RF_OUT", "/tmp/out.svg")))
            nm = os.environ.get("RF_NAME") or None
            r = render_one(step, out_dir, nm and nm.lower(), width)
            logf.write("OK %s counts=%s\n" % (r["name"], json.dumps(r["counts"])))
    except Exception:
        import traceback
        logf.write("FAIL\n" + traceback.format_exc())
    logf.flush(); logf.close()


_main()
