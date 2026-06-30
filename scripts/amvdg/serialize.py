"""Tier-1 graph <-> compact text serialization for cadrille 2D->2D training.

cadrille is image->text. We keep the image path untouched and only change the
generation TARGET from CadQuery code to a compact serialization of our Tier-1
drawing graph (per-view primitives + dimensions-with-refs).

Design choices:
  * Pixel coords are rounded to ints (sub-pixel precision is noise for a 2B VLM
    and every digit is a token). Dimension *values* (mm) keep one decimal.
  * Compact keys (t/v/f/c/r/p1/p2) cut token count vs the full schema; expansion
    back to the canonical Tier-1 schema is deterministic (see graph_from_text).
  * Output is minified JSON: directly parseable, matches the agreed Tier-1 output
    format, and is VLM-as-judge friendly. Swap SER_VERSION if we move to a DSL.
"""
import json

SER_VERSION = "g1"

_VIS = {"visible": "vis", "hidden": "hid"}
_VIS_INV = {v: k for k, v in _VIS.items()}
_SUB = {"horizontal": "h", "vertical": "v", "aligned": "a", "diameter": "d", "radius": "r"}
_SUB_INV = {v: k for k, v in _SUB.items()}


def _i(x):
    return int(round(float(x)))


def _prim_to_compact(p):
    o = {"id": p["id"], "t": p["type"],
         "v": _VIS.get(p.get("line_role", p.get("visibility", "visible")), "vis")}
    f = p.get("feature_tag", p.get("feature"))
    if f and f != "outline":
        o["f"] = f
    if p["type"] == "line":
        o["p1"] = [_i(p["p1"][0]), _i(p["p1"][1])]
        o["p2"] = [_i(p["p2"][0]), _i(p["p2"][1])]
    elif p["type"] == "polyline":
        o["pts"] = [[_i(x), _i(y)] for x, y in p["pts"]]
    elif p["type"] in ("circle", "arc"):
        o["c"] = [_i(p["center"][0]), _i(p["center"][1])]
        o["r"] = _i(p.get("r_px", p.get("r", 0)))
        if p["type"] == "arc":
            if "a1" in p:
                o["a1"] = _i(p["a1"])
            if "a2" in p:
                o["a2"] = _i(p["a2"])
    return o


def _dim_to_compact(d):
    o = {"id": d["id"], "t": d.get("kind", d.get("type"))}
    sub = d.get("subtype")
    if sub:
        o["s"] = _SUB.get(sub, sub)
    o["val"] = round(float(d["value"]), 1)
    o["view"] = d.get("view", "")
    o["refs"] = list(d.get("refs", []))
    if d.get("tol"):
        o["tol"] = d["tol"]
    return o


def graph_to_text(graph):
    """Canonical Tier-1 graph dict -> compact minified-JSON target string."""
    out = {"views": [], "dims": []}
    for v in graph.get("views", []):
        out["views"].append({
            "name": v["name"],
            "prims": [_prim_to_compact(p) for p in v.get("primitives", [])],
        })
    for d in graph.get("annotations", graph.get("dimensions", [])):
        out["dims"].append(_dim_to_compact(d))
    return json.dumps(out, ensure_ascii=False, separators=(",", ":"))


def graph_from_text(text):
    """Compact target string -> canonical-ish Tier-1 graph dict (for scoring).

    Inverse of graph_to_text up to the lossy rounding. Used to score model
    output with the existing poc/score.py harness.
    """
    o = json.loads(text)
    views = []
    for v in o.get("views", []):
        prims = []
        for p in v.get("prims", []):
            q = {"id": p["id"], "type": p["t"],
                 "line_role": _VIS_INV.get(p.get("v", "vis"), "visible"),
                 "feature_tag": p.get("f", "outline")}
            if p["t"] == "line" and "p1" in p:
                q["p1"] = [float(p["p1"][0]), float(p["p1"][1])]
                q["p2"] = [float(p["p2"][0]), float(p["p2"][1])]
            elif p["t"] == "polyline" and "pts" in p:
                q["pts"] = [[float(x), float(y)] for x, y in p["pts"]]
                q["p1"] = q["pts"][0]
                q["p2"] = q["pts"][-1]
            elif "c" in p:
                q["center"] = [float(p["c"][0]), float(p["c"][1])]
                q["r_px"] = float(p["r"])
                if "a1" in p:
                    q["a1"] = float(p["a1"])
                if "a2" in p:
                    q["a2"] = float(p["a2"])
            prims.append(q)
        views.append({"name": v["name"], "primitives": prims})
    dims = []
    for d in o.get("dims", []):
        dims.append({"id": d["id"], "kind": d["t"],
                     "subtype": _SUB_INV.get(d.get("s", ""), d.get("s", "")),
                     "value": float(d["val"]), "view": d.get("view", ""),
                     "refs": list(d.get("refs", [])), "tol": d.get("tol")})
    return {"views": views, "annotations": dims}


def _autoclose(frag):
    """Balance a possibly-truncated JSON fragment: close an open string, drop a
    trailing comma, and append the missing closing brackets in order."""
    instr = False; esc = False; stack = []
    for ch in frag:
        if instr:
            if esc: esc = False
            elif ch == '\\': esc = True
            elif ch == '"': instr = False
            continue
        if ch == '"': instr = True
        elif ch == '{': stack.append('}')
        elif ch == '[': stack.append(']')
        elif ch in '}]':
            if stack: stack.pop()
    out = frag
    if instr:
        out += '"'
    out = out.rstrip().rstrip(',')
    return out + ''.join(reversed(stack))


def graph_from_text_safe(text):
    """(graph, status) with status in {ok, repaired, fail}. Lifts parse rate without
    grammar-constrained decoding: parse as-is; else take the longest prefix ending at
    a complete element, auto-close it, and parse that (drops a half-written tail)."""
    try:
        return graph_from_text(text), "ok"
    except Exception:
        pass
    start = text.find('{')
    if start >= 0:
        body = text[start:]
        cuts = [i + 1 for i, ch in enumerate(body) if ch in '}]']
        for end in reversed(cuts):                 # longest valid sub-structure first
            try:
                return graph_from_text(_autoclose(body[:end])), "repaired"
            except Exception:
                continue
    return {"views": [], "annotations": []}, "fail"


PROMPT = ("Extract the engineering drawing as a Tier-1 graph. For each view emit "
          "primitives (lines/circles/arcs with visibility and feature) and the "
          "dimensions with their value and the ids of the primitives they measure. "
          "Output compact JSON only.")


if __name__ == "__main__":
    import sys
    g = json.load(open(sys.argv[1]))
    t = graph_to_text(g)
    print(t)
    # round-trip sanity
    back = graph_from_text(t)
    print("\n[roundtrip] views:", len(back["views"]),
          "prims:", sum(len(v["primitives"]) for v in back["views"]),
          "dims:", len(back["annotations"]),
          "target_chars:", len(t))
