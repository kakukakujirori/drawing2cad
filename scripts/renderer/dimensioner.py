import math
import FreeCAD as App

_PARAM_ROLE = {"dx": "width", "dy": "depth", "dz": "height", "diameter": "diameter"}
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
    return ('<line class="center" x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f"/>\n'
            '<line class="center" x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f"/>'
            % (cx - L, cy, cx + L, cy, cx, cy - L, cx, cy + L))

def axis_dir(ax):
    for nm, comp in (("X", ax.x), ("Y", ax.y), ("Z", ax.z)):
        if abs(abs(comp) - 1.0) < 1e-3 and abs(ax.x) + abs(ax.y) + abs(ax.z) - 1 < 1e-2:
            return nm
    return None

def extract_cylinders(shape):
    feats = []
    for f_idx, f in enumerate(shape.Faces):
        srf = f.Surface
        if srf.TypeId == "Part::GeomCylinder":
            a = axis_dir(srf.Axis)
            if a is None:
                continue
            c = srf.Center
            feats.append({"center": (c.x, c.y, c.z), "r": srf.Radius, "axis": a,
                          "face_id": "Face_%d" % f_idx})
    seen = {}
    uniq = []
    for fe in feats:
        cx, cy, cz = fe["center"]
        key = (fe["axis"], round(cx, 1), round(cy, 1), round(cz, 1), round(fe["r"], 1))
        if key in seen:
            # coaxial split faces (half-bores etc.) collapse into one feature,
            # keeping every contributing B-rep face id
            uniq[seen[key]]["face_ids"].append(fe["face_id"])
            continue
        seen[key] = len(uniq)
        fe = dict(fe); fe["id"] = "cyl%d" % len(uniq); fe["face_ids"] = [fe.pop("face_id")]
        uniq.append(fe)
    return uniq

def fmt(v):
    return ("%g" % round(v, 1))

class LegacyDimensioner:
    def __init__(self, shape, views_gt, views_obj, PXMM):
        """
        :param shape: 3D Part.Shape
        :param views_gt: list of view dicts containing 'primitives'
        :param views_obj: dictionary of {view_name: ViewObject} providing M() and scale
        :param PXMM: pixels per mm mapping
        """
        self.shape = shape
        self.views_gt = views_gt
        self.view_by_name = {gv["name"]: gv for gv in views_gt}
        self.views_obj = views_obj
        self.PXMM = PXMM
        
        self.dims_svg = []
        self.dims_gt = []
        self.features = []
        self.dctr = 0

    def _find_circle_prim(self, view_name, center_cu, center_cv, r, tolpx=4.0):
        vobj = self.views_obj[view_name]
        sx, sy = vobj.M(center_cu, center_cv)
        tx, ty = sx * self.PXMM, sy * self.PXMM
        rpx = r * vobj.scale * self.PXMM
        best = None; bestd = 1e9
        for p in self.view_by_name[view_name]["primitives"]:
            if p["type"] not in ("circle", "arc"):
                continue
            d = math.hypot(p["center"][0] - tx, p["center"][1] - ty) + abs(p["r_px"] - rpx)
            if d < bestd:
                bestd = d; best = p
        if best and bestd < tolpx + rpx * 0.15:
            return best["id"]
        return None

    def _find_extreme_line_refs(self, view_name, axis):
        gv = self.view_by_name[view_name]
        prims = [p for p in gv["primitives"] if p.get("line_role") == "visible"]
        if not prims:
            return []
        i0, i1 = (0, 2) if axis == "h" else (1, 3)
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

    _VIEW_PFX = {"front": "F", "top": "T", "right": "R"}

    def _record_centerlines(self, vname, sx, sy, L, cyl):
        """Record the drawn centerline cross as GT primitives (line_role=center),
        so every inked line on the sheet is explained by the graph."""
        prims = self.view_by_name[vname]["primitives"]
        pfx = self._VIEW_PFX.get(vname, vname[:1].upper())
        origins = [{"dim": 2, "id": fid, "role": "axis"} for fid in cyl.get("face_ids", [])]
        for (x1, y1, x2, y2) in ((sx - L, sy, sx + L, sy), (sx, sy - L, sx, sy + L)):
            p1 = [round(x1 * self.PXMM, 2), round(y1 * self.PXMM, 2)]
            p2 = [round(x2 * self.PXMM, 2), round(y2 * self.PXMM, 2)]
            prims.append({
                "id": "%s%d" % (pfx, len(prims)), "type": "line",
                "line_role": "center", "feature_tag": "hole_or_round",
                "feature_id": cyl["id"], "state": "known", "coords_source": "gt",
                "p1": p1, "p2": p2,
                "bbox_px": [round(min(p1[0], p2[0]), 1), round(min(p1[1], p2[1]), 1),
                            round(max(p1[0], p2[0]), 1), round(max(p1[1], p2[1]), 1)],
                "prov": {"topo_origins": origins},
            })

    def _add_dim(self, svg_frag, anchor_mm, dtype, subtype, value, view, refs, prov):
        self.dctr += 1
        self.dims_svg.append(svg_frag)
        param = prov.get("param")
        feat = prov.get("feature")
        self.dims_gt.append({
            "id": "D%d" % self.dctr, "kind": dtype, "subtype": subtype,
            "param_role": _PARAM_ROLE.get(param, "unknown"),
            "value": round(value, 3), "value_source": "gt",
            "value_state": "known", "status": "known",
            "refs": refs, "view": view,
            "feature_id": feat if param == "diameter" else None,
            "text_px": [round(anchor_mm[0] * self.PXMM, 2), round(anchor_mm[1] * self.PXMM, 2)],
            "prov": {"feature_id": feat, "param": param, "origin": "synthetic_gt"},
        })

    def annotate(self):
        """Generates annotations and features."""
        front = self.views_obj["front"]
        top = self.views_obj["top"]
        right = self.views_obj["right"]

        # FRONT overall
        fLx = front.M(front.umin, front.vmin)[0]
        fRx = front.M(front.umax, front.vmin)[0]
        fB  = front.M(front.umin, front.vmin)[1]
        fT  = front.M(front.umin, front.vmax)[1]
        frag, anc = hdim(fLx, fRx, fB, fB + 22, fmt(front.w))
        self._add_dim(frag, anc, "linear", "horizontal", front.w, "front",
                      self._find_extreme_line_refs("front", "h"),
                      {"feature": "bbox", "param": "dx"})
        
        frag, anc = vdim(fT, fB, fLx, fLx - 18, fmt(front.h))
        self._add_dim(frag, anc, "linear", "vertical", front.h, "front",
                      self._find_extreme_line_refs("front", "v"),
                      {"feature": "bbox", "param": "dz"})

        # TOP overall
        tTy = top.M(top.umin, top.vmax)[1]
        tBy = top.M(top.umin, top.vmin)[1]
        tLx = top.M(top.umin, top.vmin)[0]
        frag, anc = vdim(tTy, tBy, tLx, tLx - 18, fmt(top.h))
        self._add_dim(frag, anc, "linear", "vertical", top.h, "top",
                      self._find_extreme_line_refs("top", "v"),
                      {"feature": "bbox", "param": "dy"})

        # Cylinders
        DIA = "⌀"
        cyls = extract_cylinders(self.shape)
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
            ref = self._find_circle_prim(vname, cu, cv, c["r"])
            if not ref:
                continue
            _seen_dim.add(key)
            sx, sy = vobj.M(cu, cv)
            _off[vname] = _off.get(vname, 0) + 1
            self.dims_svg.append(centerlines_for(sx, sy, max(c["r"] * vobj.scale, 3)))
            self._record_centerlines(vname, sx, sy, max(c["r"] * vobj.scale, 3) + 2.5, c)
            frag, anc = diadim(sx, sy, c["r"] * vobj.scale, 30 + 14 * (_off[vname] % 4),
                               "%s%s" % (DIA, fmt(2 * c["r"])))
            self._add_dim(frag, anc, "diameter", None, 2 * c["r"], vname,
                          [ref], {"feature": c["id"], "param": "diameter"})

        # Correspondences (features): a feature's members are ALL primitives —
        # across every view — whose topo_origins touch one of the cylinder's
        # B-rep faces (circle in the face-on view, silhouette/edge lines in the
        # side views, centerline crosses). This is what makes the cross-view
        # layer real: provenance-exact, not radius-guessed.
        for c in cyls:
            face_ids = set(c.get("face_ids", []))
            members, claimed = [], []
            for gv in self.views_gt:
                for p in gv["primitives"]:
                    origs = (p.get("prov") or {}).get("topo_origins") or []
                    if not any(o.get("id") in face_ids for o in origs):
                        continue
                    role = p.get("line_role", "visible")
                    if role not in ("visible", "hidden", "center"):
                        role = "visible"
                    members.append({"view": gv["name"], "primitive_id": p["id"],
                                    "projection_role": role})
                    claimed.append(p)
            if not members:
                # fallback (oracle found nothing): bind the face-on circle by
                # projected center+radius as before
                cx, cy, cz = c["center"]
                probe = {"Y": ("front", (cx, cz)), "Z": ("top", (cx, cy)),
                         "X": ("right", (cy, cz))}.get(c["axis"])
                if probe:
                    rid = self._find_circle_prim(probe[0], probe[1][0], probe[1][1], c["r"])
                    if rid:
                        members = [{"view": probe[0], "primitive_id": rid,
                                    "projection_role": "visible"}]
            if not members:
                continue
            for p in claimed:
                if not p.get("feature_id"):
                    p["feature_id"] = c["id"]
            self.features.append({"feature_id": c["id"], "kind": "cylinder", "axis": c["axis"],
                             "extrude_dir": None, "r_mm": round(c["r"], 3),
                             "members": members, "status": "known",
                             "prov": {"feature_id": c["id"], "feature_3d": "cylinder",
                                      "occ_face_ids": list(c.get("face_ids", [])),
                                      "params": {"r_mm": round(c["r"], 3), "axis": c["axis"]}}})

        return self.dims_svg, self.dims_gt, self.features
