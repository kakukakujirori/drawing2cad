"""AMVDG graph -> model-input text for the AMVDG->3D leg (+ inverse, for round-trip).

This is the 3D leg's serializer (train2d/serialize.py is the 2D leg's). The text format was
historically called "g2"/"g2.1"; that token is retired, the format is unchanged (SER_VERSION).

g2 realizes the 2026-07-02 research-log §D-4 decisions: the graph is re-serialized so a
text LLM can read cross-view correspondence without chasing id pointers.

  * px -> part-frame mm coordinates. frame.axis_remap + per-view px_per_mm map px to
    model-axis directions; each view/axis is then shifted so the part's content minimum
    is 0 (the renderer's per-view origins are view-local, offset from the model frame by
    a per-view constant — verified against build_flange.py ground truth). After the
    shift every view spans [0, extent] on its two axes, so the same model coordinate
    prints as the same number in every view: a hole center y=101.14 in top reappears as
    silhouette lines y=101.14∓r in right. The GT CadQuery frame differs from the part
    frame by a translation only — evaluation must align translation (as CADGenBench and
    cadrille already do). If the renderer later emits a global anchor (e.g.
    frame.model_origin), drop the shift and use the true model frame.
  * features[].members is NOT emitted. Membership is denormalized to an inline tag list
    on each primitive row: owner `feature_id` first, then any feature whose
    prov.occ_face_ids appears in the primitive's topo_origins with a non-parent_face
    role (recovers e.g. a silhouette line shared by two coaxial holes, while dropping
    the parent_face over-connection that put half the flange in one feature).
    Features keep a one-line header (kind/axis/r); members == {rows tagged Ck}.
  * diameter annotations with a single circle ref are inlined on the circle row
    (`DIA11`); remaining annotations become DIM records.
  * canonical order (line_role, type, geometry); no prov is emitted. Features are
    renamed C1..Cn by (kind, -r, first-member anchor).
  * numbers: mm, <=2 decimals, trailing zeros stripped. px storage noise stays under
    ~0.01mm at emitted scales, so cross-view equality survives the rounding.
  * arc angles are remapped px-frame -> per-view model frame; start/end swap when the
    axis remap flips orientation, preserving the start->end-with-increasing-theta rule.

g2.1 slimming (2026-07-03, ~30% fewer tokens; decided after token-length audit showed
median 4.1k tok / 41% coverage at max_len 4096):
  * duplicate rows are merged: identical (type, role, geometry) rows collapse (tag
    union), and a hidden row geometrically identical to a visible one (an HLR artifact,
    19% of all rows) merges into it with role `vh` (= visible+hidden coincide).
  * the per-row id column is dropped (ids carried no geometry; DIM refs are replaced by
    the measured span, e.g. `X0..120`, computed from the referenced primitives).
  * FEAT `views=` is dropped (recoverable from the tags).

Round-trip: `text_to_struct(graph_to_text(g))` re-emits byte-identically and matches the
graph's primitive/feature/annotation census (merges accounted). `--check GLOB...` also
measures cross-view consistency (circle center vs silhouette lines) dataset-wide.
"""
import glob
import json
import math
import re
import sys

SER_VERSION = "3d.1"  # AMVDG-graph -> 3D-leg model-input text (was "g2.1")

ROLE = {"visible": "vis", "hidden": "hid", "center": "cen", "phantom": "pha",
        "section_cut": "sec", "break": "brk"}
TYP = {"line": "L", "polyline": "PL", "circle": "CI", "arc": "A", "point": "PT",
       "ellipse": "EL"}
TYP_INV = {v: k for k, v in TYP.items()}
_ROLE_RANK = {"vis": 0, "vh": 0, "hid": 1, "cen": 2, "pha": 3, "sec": 4, "brk": 5}
_TYP_RANK = {"L": 0, "PL": 1, "CI": 2, "A": 3, "PT": 4, "EL": 5}
_VIEW_ORDER = ("front", "top", "right", "left", "back", "bottom")


def _fmt(x):
    s = f"{float(x):.2f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


class _ViewFrame:
    def __init__(self, view):
        fr = view["frame"]
        self.ox, self.oy = fr["origin_px"]
        self.ppm = fr["px_per_mm"]
        ax = fr["axis_remap"]
        self.sx = 1.0 if ax["px_x"][0] == "+" else -1.0
        self.sy = 1.0 if ax["px_y"][0] == "+" else -1.0
        self.nx, self.ny = ax["px_x"][1], ax["px_y"][1]

    def pt(self, p):
        return (self.sx * (p[0] - self.ox) / self.ppm,
                self.sy * (p[1] - self.oy) / self.ppm)

    def r(self, r_px):
        return r_px / self.ppm

    def ang(self, deg):
        t = math.radians(deg)
        return math.degrees(math.atan2(self.sy * math.sin(t),
                                       self.sx * math.cos(t))) % 360.0

    @property
    def flips(self):  # orientation reversal (exactly one axis mirrored)
        return self.sx * self.sy < 0


def _prim_geom(p, cv):
    """Primitive -> unshifted mm geometry dict, or None if unsupported/degenerate."""
    t = p["type"]
    try:
        if t == "line":
            return {"pts": [cv.pt(p["p1"]), cv.pt(p["p2"])]}
        if t == "polyline":
            return {"pts": [cv.pt(q) for q in p["pts"]]}
        if t == "point":
            return {"pts": [cv.pt(p.get("p1") or p["center"])]}
        if t in ("circle", "arc"):
            g = {"c": cv.pt(p["center"]), "r": cv.r(p.get("r_px", p.get("r", 0)))}
            if t == "arc":
                a1, a2 = p.get("start_angle"), p.get("end_angle")
                if a1 is not None and a2 is not None:
                    a1, a2 = cv.ang(a1), cv.ang(a2)
                    if cv.flips:
                        a1, a2 = a2, a1
                    g["a"] = (a1, a2)
            return g
        if t == "ellipse":
            g = {"c": cv.pt(p["center"]), "rmaj": cv.r(p.get("rmaj_px", 0)),
                 "rmin": cv.r(p.get("rmin_px", 0)),
                 "rot": cv.ang(p.get("rot_deg", 0.0)) % 180.0}
            a1, a2 = p.get("start_angle"), p.get("end_angle")
            if a1 is not None and a2 is not None:
                # eccentric angles are intrinsic to the ellipse axes; the px->model
                # remap is orthogonal, so a rotation leaves them unchanged and a
                # reflection negates them + reverses the start->end orientation.
                if cv.flips:
                    a1, a2 = (-a2) % 360.0, (-a1) % 360.0
                g["ea"] = (a1, a2)
            return g
    except (KeyError, TypeError):
        return None
    return None


def _geom_min(g):
    """(minx, miny) of the drawn geometry (arc-aware: only points ON the arc count)."""
    if "pts" in g:
        return (min(x for x, _ in g["pts"]), min(y for _, y in g["pts"]))
    if "rmaj" in g:  # ellipse: loose axis-aligned bound (major-radius box)
        (cx, cy), r = g["c"], max(g["rmaj"], g["rmin"])
        return (cx - r, cy - r)
    (cx, cy), r = g["c"], g["r"]
    if "a" not in g:
        return (cx - r, cy - r)
    a1, a2 = g["a"]
    span = (a2 - a1) % 360.0 or 360.0
    cand = [a1, a1 + span]
    k = math.ceil(a1 / 90.0) * 90.0
    while k <= a1 + span:
        cand.append(k)
        k += 90.0
    xs = [cx + r * math.cos(math.radians(a)) for a in cand]
    ys = [cy + r * math.sin(math.radians(a)) for a in cand]
    return (min(xs), min(ys))


def _shift_round(g, dx, dy):
    if "pts" in g:
        g["pts"] = [(round(x - dx, 2), round(y - dy, 2)) for x, y in g["pts"]]
        if len(g["pts"]) == 2 and g["pts"][1] < g["pts"][0]:
            g["pts"] = [g["pts"][1], g["pts"][0]]          # canonical endpoint order
    elif "rmaj" in g:
        g["c"] = (round(g["c"][0] - dx, 2), round(g["c"][1] - dy, 2))
        g["rmaj"] = round(g["rmaj"], 2)
        g["rmin"] = round(g["rmin"], 2)
        g["rot"] = round(g["rot"], 2)
        if "ea" in g:
            g["ea"] = (round(g["ea"][0], 2), round(g["ea"][1], 2))
    else:
        g["c"] = (round(g["c"][0] - dx, 2), round(g["c"][1] - dy, 2))
        g["r"] = round(g["r"], 2)
        if "a" in g:
            g["a"] = (round(g["a"][0], 2), round(g["a"][1], 2))
    return g


def _gkey(r):
    return (r["typ"], json.dumps(r["geom"], sort_keys=True))


def _union_tags(dst, src):
    dst += [t for t in src if t not in dst]


def struct_from_graph(graph, multi_tag=True):
    """Canonicalize an AMVDG graph into the intermediate struct (the single source of
    truth for emission). Returns dict {bbox, feats, views, dims, skipped, merged}."""
    feats_src = graph.get("features", [])
    face2feat = {}
    for f in feats_src:
        for fid in (f.get("prov") or {}).get("occ_face_ids", []):
            face2feat.setdefault(fid, set()).add(f["feature_id"])

    views, old2row, skipped, merged = [], {}, 0, 0
    anchor_key = {}  # (view name, source primitive id) -> deterministic sort key
    for vi, v in enumerate(graph.get("views", [])):
        cv = _ViewFrame(v)
        rows = []
        for p in v["primitives"]:
            typ = TYP.get(p["type"])
            geom = _prim_geom(p, cv) if typ else None
            if geom is None:
                skipped += 1
                continue
            tags_src = [p["feature_id"]] if p.get("feature_id") else []
            if multi_tag:
                extra = set()
                for t in (p.get("prov") or {}).get("topo_origins", []):
                    if t.get("dim") == 2 and t.get("role") != "parent_face":
                        extra |= face2feat.get(t["id"], set())
                tags_src += sorted(x for x in extra if x not in tags_src)
            rows.append({"typ": typ,
                         "role": ROLE.get(p.get("line_role", "visible"), "vis"),
                         "geom": geom, "tags_src": tags_src, "olds": [p["id"]],
                         "dias": []})
        # part-frame shift: content minimum (visible+hidden only) -> 0 per axis
        mins = [_geom_min(r["geom"]) for r in rows if r["role"] in ("vis", "hid")]
        dx = min((m[0] for m in mins), default=0.0)
        dy = min((m[1] for m in mins), default=0.0)
        for r in rows:
            _shift_round(r["geom"], dx, dy)
        # merge exact duplicates (same type+role+geometry), then hid-into-vis as `vh`
        by_key, uniq = {}, []
        for r in rows:
            k = _gkey(r) + (r["role"],)
            if k in by_key:
                t = by_key[k]
                _union_tags(t["tags_src"], r["tags_src"])
                t["olds"] += r["olds"]
                merged += 1
            else:
                by_key[k] = r
                uniq.append(r)
        vis_by = {_gkey(r): r for r in uniq if r["role"] == "vis"}
        rows = []
        for r in uniq:
            t = vis_by.get(_gkey(r)) if r["role"] == "hid" else None
            if t is not None:
                t["role"] = "vh"
                _union_tags(t["tags_src"], r["tags_src"])
                t["olds"] += r["olds"]
                merged += 1
            else:
                rows.append(r)
        rows.sort(key=lambda r: (_ROLE_RANK.get(r["role"], 9), _TYP_RANK.get(r["typ"], 9),
                                 tuple(r["geom"].get("c", r["geom"].get("pts", [(0, 0)])[0])),
                                 r["geom"].get("r", 0)))
        for r in rows:
            key = (vi, _TYP_RANK.get(r["typ"], 9),
                   tuple(r["geom"].get("c", r["geom"].get("pts", [(0, 0)])[0])))
            for old in r["olds"]:
                old2row[(v["name"], old)] = r
                anchor_key[(v["name"], old)] = key
        views.append({"name": v["name"], "axes": (cv.nx, cv.ny), "prims": rows})
    views.sort(key=lambda v: _VIEW_ORDER.index(v["name"]) if v["name"] in _VIEW_ORDER else 9)

    def _anchor(f):
        keys = [anchor_key.get((m["view"], m["primitive_id"]))
                for m in f.get("members", [])]
        keys = [k for k in keys if k is not None]
        return min(keys) if keys else (99,)
    feats_sorted = sorted(feats_src, key=lambda f: (f.get("kind") or "",
                                                    -(f.get("r_mm") or 0.0), _anchor(f)))
    fmap = {f["feature_id"]: f"C{i + 1}" for i, f in enumerate(feats_sorted)}
    for v in views:
        for r in v["prims"]:
            src = r.pop("tags_src")
            r["tags"] = ([fmap[src[0]]] if src and src[0] in fmap else []) + sorted(
                (fmap[x] for x in src[1:] if x in fmap), key=lambda s: int(s[1:]))

    # list-valued: two coaxial cylinders with close radii can both dimension the SAME
    # circle primitive (the dimensioner's radius-fallback binding) — a dict would
    # silently drop one value (caught by the census gate on 5/2825 Z2C-train graphs)
    dia_by_ref, dims_src = {}, []
    for a in graph.get("annotations", []):
        if a.get("kind") == "diameter" and len(a.get("refs", [])) == 1:
            dia_by_ref.setdefault((a.get("view"), a["refs"][0]), []).append(a["value"])
        else:
            dims_src.append(a)
    for v in views:
        for r in v["prims"]:
            r["dias"] = [d for old in r.pop("olds")
                         for d in dia_by_ref.get((v["name"], old), [])]

    views_by_name = {v["name"]: v for v in views}

    def _span(a):
        v = views_by_name.get(a.get("view"))
        ax_i = {"horizontal": 0, "vertical": 1}.get(a.get("subtype"))
        if v is None or ax_i is None:
            return None
        vals = []
        for ref in a.get("refs", []):
            r = old2row.get((a.get("view"), ref))
            if r is None:
                continue
            g = r["geom"]
            if "pts" in g:
                vals += [p[ax_i] for p in g["pts"]]
            else:
                # circle/arc: r; ellipse: loose axis-aligned bound (major-radius box),
                # matching _geom_min's ellipse handling
                rr = g.get("r", max(g.get("rmaj", 0.0), g.get("rmin", 0.0)))
                vals += [g["c"][ax_i] - rr, g["c"][ax_i] + rr]
        return (v["axes"][ax_i], min(vals), max(vals)) if vals else None
    dims = []
    for i, a in enumerate(dims_src):
        dims.append({"id": f"d{i + 1}", "kind": a.get("kind") or "-",
                     "role": a.get("param_role") or "-", "val": a["value"],
                     "view": a.get("view") or "-", "span": _span(a)})
    bbox = (graph.get("world") or {}).get("bbox_3d") or []
    # edge-blend features (fillet/chamfer): correspondence-layer records carrying
    # kind + value(mm) + a directional edge-set token (all|par:<axis>|face:<±axis>|
    # complex). Sorted for determinism; count preserved (no merge) so round-trip census
    # matches the injected `blends` list. Absent key -> [] (no BLEND lines emitted).
    blends = sorted(({"kind": b.get("kind") or "-", "sel": b.get("sel") or "all",
                      "val": b.get("value_mm")} for b in graph.get("blends", [])),
                    key=lambda b: (b["kind"], b["sel"],
                                   b["val"] if b["val"] is not None else -1.0))
    return {"bbox": list(bbox), "feats": [
                {"id": fmap[f["feature_id"]], "kind": f.get("kind") or "-",
                 "axis": f.get("axis") or "-", "r": f.get("r_mm"),
                 "dir": f.get("extrude_dir")} for f in feats_sorted],
            "views": views, "dims": dims, "blends": blends,
            "skipped": skipped, "merged": merged}


def _geom_str(row):
    g = row["geom"]
    if "pts" in g:
        return " ".join(f"{_fmt(x)},{_fmt(y)}" for x, y in g["pts"])
    if "rmaj" in g:
        s = (f"c{_fmt(g['c'][0])},{_fmt(g['c'][1])} rj{_fmt(g['rmaj'])} "
             f"rn{_fmt(g['rmin'])} rot{_fmt(g['rot'])}")
        if "ea" in g:
            s += f" e{_fmt(g['ea'][0])}..{_fmt(g['ea'][1])}"
        return s
    s = f"c{_fmt(g['c'][0])},{_fmt(g['c'][1])} r{_fmt(g['r'])}"
    if "a" in g:
        s += f" a{_fmt(g['a'][0])}..{_fmt(g['a'][1])}"
    return s


def struct_to_text(st):
    L = [f"PART unit=mm bbox={','.join(_fmt(x) for x in st['bbox'])}"]
    for f in st["feats"]:
        row = f"FEAT {f['id']} {f['kind']} axis={f['axis']}"
        row += f" r={_fmt(f['r'])}" if f.get("r") is not None else " r=-"
        if f.get("dir"):
            row += f" dir={f['dir']}"
        L.append(row)
    for b in st.get("blends", []):
        L.append(f"BLEND {b['kind']} {b['sel']} v"
                 + (_fmt(b["val"]) if b["val"] is not None else "-"))
    for v in st["views"]:
        L.append(f"VIEW {v['name']} axes={v['axes'][0]},{v['axes'][1]}")
        for r in v["prims"]:
            row = f"{r['typ']} {r['role']} {_geom_str(r)}"
            if r["tags"]:
                row += " " + ",".join(r["tags"])
            for d in r.get("dias", []):
                row += f" DIA{_fmt(d)}"
            L.append(row)
    for d in st["dims"]:
        sp = d.get("span")
        L.append(f"DIM {d['id']} {d['kind']} {d['role']} {_fmt(d['val'])} {d['view']} "
                 + (f"{sp[0]}{_fmt(sp[1])}..{_fmt(sp[2])}" if sp else "-"))
    return "\n".join(L)


def graph_to_text(graph, multi_tag=True):
    return struct_to_text(struct_from_graph(graph, multi_tag=multi_tag))


_PAIR = re.compile(r"^(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)$")
_TAGS = re.compile(r"^C\d+(?:,C\d+)*$")
_SPAN = re.compile(r"^([A-Za-z])(-?\d+(?:\.\d+)?)\.\.(-?\d+(?:\.\d+)?)$")


def text_to_struct(text):
    st = {"bbox": [], "feats": [], "views": [], "dims": [], "blends": [],
          "skipped": 0, "merged": 0}
    view = None
    for line in text.splitlines():
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "PART":
            kv = dict(t.split("=", 1) for t in tok[1:] if "=" in t)
            st["bbox"] = [float(x) for x in kv.get("bbox", "").split(",") if x]
        elif tok[0] == "FEAT":
            kv = dict(t.split("=", 1) for t in tok[3:] if "=" in t)
            st["feats"].append({"id": tok[1], "kind": tok[2], "axis": kv.get("axis", "-"),
                                "r": None if kv.get("r", "-") == "-" else float(kv["r"]),
                                "dir": kv.get("dir")})
        elif tok[0] == "BLEND":
            v = tok[3][1:] if len(tok) > 3 and tok[3].startswith("v") else "-"
            st["blends"].append({"kind": tok[1], "sel": tok[2],
                                 "val": None if v == "-" else float(v)})
        elif tok[0] == "VIEW":
            kv = dict(t.split("=", 1) for t in tok[2:] if "=" in t)
            view = {"name": tok[1], "axes": tuple(kv.get("axes", ",").split(",")),
                    "prims": []}
            st["views"].append(view)
        elif tok[0] == "DIM":
            m = _SPAN.match(tok[6]) if len(tok) > 6 else None
            st["dims"].append({"id": tok[1], "kind": tok[2], "role": tok[3],
                               "val": float(tok[4]), "view": tok[5],
                               "span": (m.group(1), float(m.group(2)),
                                        float(m.group(3))) if m else None})
        elif view is not None and tok[0] in TYP_INV and len(tok) >= 3:
            row = {"typ": tok[0], "role": tok[1], "geom": {}, "tags": [], "dias": []}
            pts = []
            for t in tok[2:]:
                m = _PAIR.match(t)
                if m:
                    pts.append((float(m.group(1)), float(m.group(2))))
                elif t.startswith("c") and _PAIR.match(t[1:]):
                    x, y = t[1:].split(",")
                    row["geom"]["c"] = (float(x), float(y))
                elif t.startswith("rj"):
                    row["geom"]["rmaj"] = float(t[2:])
                elif t.startswith("rn"):
                    row["geom"]["rmin"] = float(t[2:])
                elif t.startswith("rot"):
                    row["geom"]["rot"] = float(t[3:])
                elif t.startswith("e") and ".." in t:
                    a1, a2 = t[1:].split("..")
                    row["geom"]["ea"] = (float(a1), float(a2))
                elif t.startswith("r") and t != "r":
                    row["geom"]["r"] = float(t[1:])
                elif t.startswith("a") and ".." in t:
                    a1, a2 = t[1:].split("..")
                    row["geom"]["a"] = (float(a1), float(a2))
                elif t.startswith("DIA"):
                    row["dias"].append(float(t[3:]))
                elif _TAGS.match(t):
                    row["tags"] = t.split(",")
            if pts:
                row["geom"]["pts"] = pts
            view["prims"].append(row)
    return st


def crossview_dev(st, tol=0.1):
    """(max_dev_mm, matched, unmatched) over cylinder features: a circle at c in one
    view predicts tagged silhouette lines at c±r on the shared axis of the other views.
    Deviation = distance to the nearest tagged axis-parallel line when one lies within
    `tol` (the cross-view number-equality this format promises); expected positions with
    no tagged line within `tol` count as unmatched (tag-coverage gaps, not geometry)."""
    dev, matched, unmatched = 0.0, 0, 0
    views = {v["name"]: v for v in st["views"]}
    for f in st["feats"]:
        if f["kind"] != "cylinder" or f.get("r") is None:
            continue
        circles, sils = [], []
        for vn, v in views.items():
            rows = [r for r in v["prims"] if f["id"] in r["tags"]
                    and r["role"] in ("vis", "hid", "vh")]
            for r in rows:
                if "c" in r["geom"] and "a" not in r["geom"] and "rmaj" not in r["geom"]:
                    circles.append((v["axes"], r["geom"]["c"]))
            vert = {r["geom"]["pts"][0][0] for r in rows if r["typ"] == "L"
                    and abs(r["geom"]["pts"][0][0] - r["geom"]["pts"][1][0]) < 1e-9}
            if vert:
                sils.append((v["axes"][0], sorted(vert)))
        for caxes, c in circles:
            for axis, xs in sils:
                if axis not in caxes:
                    continue
                center = c[caxes.index(axis)]
                for exp in (center - f["r"], center + f["r"]):
                    d = min(abs(x - exp) for x in xs)
                    if d <= tol:
                        dev = max(dev, d)
                        matched += 1
                    else:
                        unmatched += 1
    return dev, matched, unmatched


def roundtrip_check(graph, multi_tag=True):
    """(ok, msg): emission is parse-stable and matches the source graph's census."""
    st = struct_from_graph(graph, multi_tag=multi_tag)
    text = struct_to_text(st)
    if struct_to_text(text_to_struct(text)) != text:
        return False, "re-emission not byte-identical"
    if st["skipped"]:
        return False, f"{st['skipped']} primitives skipped"
    n_src = sum(len(v["primitives"]) for v in graph.get("views", []))
    n_out = sum(len(v["prims"]) for v in st["views"])
    if n_out + st["merged"] != n_src:
        return False, f"primitive count {n_out}+{st['merged']}merged != {n_src}"
    if len(st["feats"]) != len(graph.get("features", [])):
        return False, "feature count mismatch"
    if len(st["blends"]) != len(graph.get("blends", [])):
        return False, "blend count mismatch"
    n_ann = len(graph.get("annotations", []))
    n_dia = sum(len(r["dias"]) for v in st["views"] for r in v["prims"])
    if len(st["dims"]) + n_dia != n_ann:
        return False, f"annotation count {len(st['dims'])}+{n_dia} != {n_ann}"
    return True, (f"prims={n_out}(+{st['merged']} merged) feats={len(st['feats'])} "
                  f"dims={len(st['dims'])}+{n_dia}dia")


if __name__ == "__main__":
    if sys.argv[1] == "--check":
        paths = sorted(p for a in sys.argv[2:] for p in glob.glob(a))
        bad, sizes, devs = [], [], []
        n_match = n_miss = 0
        for p in paths:
            g = json.load(open(p))
            ok, msg = roundtrip_check(g)
            if not ok:
                bad.append((p, msg))
            st = struct_from_graph(g)
            sizes.append(len(struct_to_text(st)))
            d, m, u = crossview_dev(st)
            devs.append((d, p))
            n_match += m
            n_miss += u
        sizes.sort()
        devs.sort(reverse=True)
        print(f"round-trip OK {len(paths) - len(bad)}/{len(paths)}; text chars "
              f"median={sizes[len(sizes)//2]} p90={sizes[int(.9*len(sizes))]} "
              f"max={sizes[-1]}")
        print(f"cross-view silhouette@(c±r): matched {n_match} "
              f"(worst dev {devs[0][0]:.3f}mm at {devs[0][1].split('/')[-1][:8]}), "
              f"unmatched(tag-coverage) {n_miss}")
        for p, msg in bad[:10]:
            print("  FAIL", p, "--", msg)
        sys.exit(1 if bad else 0)
    g = json.load(open(sys.argv[1]))
    text = graph_to_text(g)
    print(text)
    ok, msg = roundtrip_check(g)
    print(f"\n[roundtrip] {'OK' if ok else 'FAIL'} {msg} chars={len(text)}",
          file=sys.stderr)
