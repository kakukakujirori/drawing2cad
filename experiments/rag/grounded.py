"""Grounded checks of final readings: each claim drawn on the views as a mark, a
reader's answers to that mark's decision tree, and a doubt where the tree's
conclusion contradicts the claim.

    python -m experiments.rag.grounded questions PAIRS.jsonl KEYS OUT_DIR [KIND,...]
    python -m experiments.rag.grounded observe OUT_DIR READER [WORKERS]
    python -m experiments.rag.grounded report OUT_DIR PAIRS.jsonl JUDGED.jsonl READER...

`questions` takes marks from each reading (KEYS: comma-separated pair keys, or
"all") and writes OUT_DIR/questions.jsonl with the images each question shows;
KIND limits it to some kinds of mark.
`observe` asks READER (sol, without the real part, or luna) every question and
writes OUT_DIR/answers_<READER>.jsonl. `report` scores the doubts against the
judge's errors and lists each doubt with its images.
"""

import base64
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from string import Template
from threading import Lock

import numpy as np
import yaml
from langchain_core.messages import HumanMessage

from experiments.rag.judge import sol
from experiments.rag.marks import (
    Edge,
    Marks,
    ViewMap,
    _coaxial,
    build_edges,
    extract,
    hole_band,
    hole_mouth,
    material_along,
    part_box,
    runs,
    stepped,
    view_maps,
)
from experiments.rag.reread import luna
from experiments.rag.schema import Judged, Pair, read_rows
from experiments.rag.trees import HIDDEN, MARKS, Tree, fold

PROMPT = Path(__file__).with_name("grounded_observe.md").read_text()
TREES = "trees_grounded_tested.json"
MAX_SCORE = 20.0  # views registered worse than this are left out
SCALE_SPREAD = 1.2  # views of one sheet registered at scales further apart than this raise a doubt
HOLES_PER_FEATURE = 2
HOLE_PLAIN = False  # show a hole band's enlarged area also without the band; it did not help the readers
EDGE_VIEWS = 2  # views shown for an edge mark
MIN_EDGE_PX = 20  # a feature's edges must be this long in a view to be shown there
EDGE_PX = 12  # width of an edge band and of its dots in an enlarged image
MIN_RUN_PX = 30  # shorter walls are not asked about
AMBIGUOUS_PX = 5.0  # another line of the same kind this close cannot be told apart from a wall
MAX_WALLS = 6  # walls asked about per cavity
MIN_CROP_PX = 120  # least width and height of an enlarged area, in the drawing's pixels
EXTENT_PX = 3.0  # a run this close to a body's extreme along an axis is that side's outermost edge
KINDS = ("hole edge", "cavity wall")
SIDE = 1100  # longest side of a shown image, in pixels
TRIES = 3
# What each kind of mark claims, and which judged mistakes contradict it.
CLAIMS = {"face bars": "as read", "wall line": "wall drawn", "body region": "solid",
          "body edges": "as read", "cavity edges": "walls drawn", "body interior": "solid", "cavity wall": "wall drawn",
          "body extent": "as read", "hole edge": "edge drawn"}
CONTRADICTED_BY = {
    "hole band": {("through or blind", None)},
    "face bars": {("thickness", None), ("how far", None)},
    "wall line": {("solid or hollow", "hollow"), ("open or closed", None)},
    "body region": {("solid or hollow", "solid")},
    "body edges": {("thickness", None), ("how far", None)},
    "cavity edges": {("solid or hollow", "hollow"), ("open or closed", None)},
    "body interior": {("solid or hollow", "solid")},
    "cavity wall": {("solid or hollow", "hollow"), ("open or closed", None)},
    "body extent": {("thickness", None), ("how far", None)},
    "hole edge": {("through or blind", None)},
}


def _image(view: ViewMap, shapes: list[tuple]):
    from PIL import Image

    image = Image.open(view.image_path)
    return _draw(image, shapes, lambda p: p, max(3, round(max(image.size) / 300)))


def _draw(image, shapes: list[tuple], place, width: int):
    """`shapes` drawn over `image`, their points taken to its pixels by `place`."""
    from PIL import Image, ImageDraw

    image = image.convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for shape in shapes:
        points = [tuple(place(np.asarray(p, float))) for p in shape[1]] if shape[0] != "line" else None
        if shape[0] == "band":
            draw.polygon(points, fill=(255, 0, 0, 70))
        elif shape[0] == "outline":
            draw.polygon(points, outline=(255, 0, 0, 255), width=2)
        elif shape[0] == "line":
            draw.line((tuple(place(np.asarray(shape[1], float))), tuple(place(np.asarray(shape[2], float)))),
                      fill=(255, 0, 0, 130), width=width)
        elif shape[0] == "tint":
            draw.polygon(points, fill=(255, 0, 0, 45))
        elif shape[0] == "edge":
            draw.line(points, fill=(230, 0, 0, 110), width=width, joint="curve")
        elif shape[0] == "dots":
            for x, y in _spaced(np.array(points), 2.2 * width):
                draw.ellipse((x - width / 2, y - width / 2, x + width / 2, y + width / 2), fill=(255, 140, 0, 150))
    return Image.alpha_composite(image, layer).convert("RGB")


def _spaced(points: np.ndarray, step: float) -> np.ndarray:
    """Points along a polyline, `step` apart."""
    t = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    s = np.arange(0, t[-1] + 1e-9, step)
    return np.column_stack([np.interp(s, t, points[:, i]) for i in range(2)])


def _save_pair(view: ViewMap, shapes: list[tuple], box: tuple[float, ...] | None, stem: Path,
               plain: bool = False) -> list[dict]:
    """The whole view with the mark and a blue frame, and the framed area enlarged; with
    `plain`, the framed area enlarged again without the mark."""
    from PIL import Image, ImageDraw

    marked = _image(view, shapes)
    shown = []
    if box is not None and (min(marked.width, box[2]) - max(0, box[0]) < 2 or min(marked.height, box[3]) - max(0, box[1]) < 2):
        box = None  # the area lies off the drawing
    if box is not None:
        x0, y0, x1, y1 = (max(0, box[0]), max(0, box[1]), min(marked.width, box[2]), min(marked.height, box[3]))
        frame = (round(x0), round(y0), round(x1), round(y1))
        crop = marked.crop(frame)
        scale = max(1.0, min(6.0, SIDE / max(crop.size)))
        size = (round(crop.width * scale), round(crop.height * scale))
        crop = crop.resize(size)
        bare = Image.open(view.image_path).convert("RGB").crop(frame).resize(size) if plain else None
        if bare is not None:
            # Marks drawn after enlarging, so they keep one size however far the area is enlarged.
            factor = np.array(size) / (np.array(frame[2:]) - frame[:2])
            crop = _draw(bare, shapes, lambda p: (p - frame[:2]) * factor, EDGE_PX)
        ImageDraw.Draw(marked).rectangle((x0, y0, x1, y1), outline=(0, 90, 255), width=max(3, round(max(marked.size) / 250)))
    whole = marked.resize((round(marked.width * min(1.0, SIDE / max(marked.size))), round(marked.height * min(1.0, SIDE / max(marked.size)))))
    whole.save(f"{stem}_whole.png")
    shown.append({"label": f"{view.name} ({view.role}), the whole view", "path": f"{stem}_whole.png"})
    if box is not None:
        crop.save(f"{stem}_enlarged.png")
        shown.append({"label": f"{view.name} ({view.role}), enlarged", "path": f"{stem}_enlarged.png"})
        if bare is not None:
            bare.save(f"{stem}_plain.png")
            shown.append({"label": f"{view.name} ({view.role}), enlarged, without the mark", "path": f"{stem}_plain.png"})
    return shown


def _edge_shapes(edges: list[Edge]) -> list[tuple]:
    return [("dots" if e.kind == "hidden" else "edge", e.px) for e in edges]


def _px_length(edges: list[Edge]) -> float:
    return sum(float(np.linalg.norm(np.diff(np.array(e.px), axis=0), axis=1).sum()) for e in edges)


def _edge_question(views: list[ViewMap], picked: dict[str, list[Edge]], extra: dict[str, list[tuple]], stem: Path,
                   name: str) -> list[dict]:
    """Images for an edge mark: in up to EDGE_VIEWS of the best views where its edges are long
    enough, the whole view, the marked area enlarged, and that area without the mark."""
    images = []
    for view in [v for v in views if _px_length(picked[v.name]) >= MIN_EDGE_PX][:EDGE_VIEWS]:
        shapes = extra.get(view.name, []) + _edge_shapes(picked[view.name])
        points = [p for e in picked[view.name] for p in e.px] + [p for s in extra.get(view.name, []) for p in s[1]]
        b = _bounds(points)
        size = max(b[2] - b[0], b[3] - b[1])
        images += _save_pair(view, shapes, _grow(b, max(40, 0.1 * size), 0.1), stem / f"{name}_{view.name}", plain=True)
    return images


def _bounds(points) -> np.ndarray:
    points = np.array(points)
    return np.r_[points.min(axis=0), points.max(axis=0)]


def _grow(bounds: np.ndarray, across: float, along: float) -> tuple[float, ...]:
    x0, y0, x1, y1 = bounds
    w, h = x1 - x0, y1 - y0
    gx, gy = (along * w, across) if w >= h else (across, along * h)
    return (x0 - gx, y0 - gy, x1 + gx, y1 + gy)


def questions_for(pair: Pair, marks: Marks, out: Path, kinds: tuple[str, ...] = KINDS) -> list[dict]:
    run = Path(pair.drawing).parents[2]
    drawing = yaml.safe_load((run / ".hydra/config.yaml").read_text())["sample"]["drawing"]
    common = {"key": pair.key, "part": pair.part, "tree_fold": 1 - fold(pair.part),
              "hidden_drawn": not drawing.get("hidden_lines_omitted"),
              "projection": drawing["projection_method"].replace("_", " ")}
    views = [v for v in view_maps(run).values() if v.match_score is not None and v.match_score <= MAX_SCORE]
    views.sort(key=lambda v: v.match_score)
    low, high = part_box(pair.build)
    stem = out / "images" / pair.part
    stem.mkdir(parents=True, exist_ok=True)
    found = []

    # Holes, steps on one axis taken as one hole: a view across each, the one other holes
    # overlap least, with the band over the bodies it runs through, and its mouth in a view along it.
    holes = stepped(marks.holes)
    kept, per_feature = [], {}
    for hole in holes:
        if per_feature.setdefault(hole.feature, 0) < HOLES_PER_FEATURE:
            per_feature[hole.feature] += 1
            kept.append(hole)
    for n, hole in enumerate(kept if "hole band" in kinds else []):
        options = []
        lo, hi = material_along(hole, marks.bodies, low, high)
        for view in views:
            if (band := hole_band(view, hole, lo, hi)) is None:
                continue
            b = _bounds(band)
            others = [_bounds(o) for o in (hole_band(view, h, lo, hi) for h in holes if h is not hole) if o is not None]
            overlap = sum(max(0, min(b[2], o[2]) - max(b[0], o[0])) * max(0, min(b[3], o[3]) - max(b[1], o[1])) for o in others)
            options.append((overlap / max(1e-9, (b[2] - b[0]) * (b[3] - b[1])), -min(b[2] - b[0], b[3] - b[1]), view, band))
        if not options:
            continue
        _, _, view, band = min(options, key=lambda o: o[:2])
        b = _bounds(band)
        width = min(b[2] - b[0], b[3] - b[1])
        images = _save_pair(view, [("band", band)], _grow(b, max(80, 3 * width), 0.5), stem / f"hole{n}_{view.name}",
                            plain=HOLE_PLAIN)
        along = next(((v, m) for v in views if (m := hole_mouth(v, hole)) is not None), None)
        if along is not None:
            mb = _bounds(along[1])
            size = max(mb[2] - mb[0], mb[3] - mb[1])
            images += _save_pair(along[0], [("outline", along[1])], _grow(mb, max(40, 1.5 * size), 1.5), stem / f"hole{n}_{along[0].name}")[1:]
        found.append({**common, "id": f"hole{n}", "kind": "hole band", "feature": hole.feature, "claim": hole.claim, "images": images})

    if {"body edges", "cavity edges", "body interior", "cavity wall", "body extent", "hole edge"} & set(kinds):
        found += _edge_questions(pair, marks, views, common, stem, kinds, kept)

    # Bodies: for each of its axes, the two faces as bars in the best view that sees them edge-on.
    for n, body in enumerate(marks.bodies if "face bars" in kinds or "body region" in kinds else []):
        lo, hi = np.array(body.low, float), np.array(body.high, float)
        for axis in range(3) if "face bars" in kinds else []:
            view = next((v for v in views if axis in (v.iu, v.iv)), None)
            if view is None:
                continue
            span = view.iv if axis == view.iu else view.iu
            pad = (hi[span] - lo[span]) * 0.1
            bars = []
            for value in (lo[axis], hi[axis]):
                a, b = lo.copy(), lo.copy()
                a[axis] = b[axis] = value
                a[span], b[span] = lo[span] - pad, hi[span] + pad
                bars.append((view.px(a), view.px(b)))
            bb = _bounds([p for bar in bars for p in bar])
            if max(bb[2] - bb[0], bb[3] - bb[1]) < 10:
                continue
            images = _save_pair(view, [("line", a, b) for a, b in bars], _grow(bb, 0.15 * max(bb[2] - bb[0], bb[3] - bb[1]), 0.05),
                                stem / f"body{n}_{'xyz'[axis]}_{view.name}")
            found.append({**common, "id": f"body{n}_{'xyz'[axis]}", "kind": "face bars", "feature": body.feature,
                          "claim": CLAIMS["face bars"], "images": images})
        if body.solid and "body region" in kinds:
            tinted = []
            for view in views[:2]:
                corners = [lo.copy() for _ in range(4)]
                for c, (fu, fv) in zip(corners, ((0, 0), (1, 0), (1, 1), (0, 1)), strict=True):
                    c[view.iu], c[view.iv] = (lo, hi)[fu][view.iu], (lo, hi)[fv][view.iv]
                tinted += _save_pair(view, [("tint", [view.px(c) for c in corners])], None, stem / f"body{n}_tint_{view.name}")
            if tinted:
                found.append({**common, "id": f"body{n}_region", "kind": "body region", "feature": body.feature,
                              "claim": CLAIMS["body region"], "images": tinted})

    # Cavities: each wall seen edge-on in the best view, one red line at a time. A face of the
    # cavity's box that lies on a face of the part or of a body is an opening, not a wall.
    faces = [(axis, value) for axis in range(3) for value in (low[axis], high[axis])]
    faces += [(axis, value) for body in marks.bodies for axis in range(3) for value in (body.low[axis], body.high[axis])]
    tolerance = 0.001 * float(np.max(high - low))
    for n, cavity in enumerate(marks.cavities if "wall line" in kinds else []):
        if not views:
            break
        view, lo, hi = views[0], np.array(cavity.low, float), np.array(cavity.high, float)
        for face_axis, span_axis in ((view.iu, view.iv), (view.iv, view.iu)):
            for side, value in (("low", lo[face_axis]), ("high", hi[face_axis])):
                if any(axis == face_axis and abs(value - v) < tolerance for axis, v in faces):
                    continue
                a, b = lo.copy(), lo.copy()
                a[face_axis] = b[face_axis] = value
                a[span_axis], b[span_axis] = lo[span_axis], hi[span_axis]
                a, b = view.px(a), view.px(b)
                bb = _bounds([a, b])
                length = max(bb[2] - bb[0], bb[3] - bb[1])
                if length < 10:
                    continue
                wall = f"cavity{n}_{'xyz'[face_axis]}{side}"
                images = _save_pair(view, [("line", a, b)], _grow(bb, max(40, 0.08 * length), 0.05), stem / f"{wall}_{view.name}")
                found.append({**common, "id": wall, "kind": "wall line", "feature": cavity.feature,
                              "claim": CLAIMS["wall line"], "images": images})
    return found


def _edge_questions(pair: Pair, marks: Marks, views: list[ViewMap], common: dict, stem: Path,
                    kinds: tuple[str, ...], holes: list) -> list[dict]:
    """Edge marks: the build's own edges of each body and cavity, and every build edge inside
    each solid body's box over a tint of that box."""
    from zeroshot.pipeline.verification.render.orthographic import load_shape
    from zeroshot.pipeline.verification.solid_mesh import shape_bbox

    build = load_shape(Path(pair.build))
    diag = float(np.linalg.norm(shape_bbox(build)[1]))
    edges = {v.name: build_edges(build, v, marks, diag, common["hidden_drawn"]) for v in views}
    found = []
    for kind, prefix, features in (("body edges", "body", marks.bodies), ("cavity edges", "cavity", marks.cavities)):
        for n, feature in enumerate(features if kind in kinds else []):
            picked = {v: [e for e in found_edges if e.owner == f"{prefix}{n}"] for v, found_edges in edges.items()}
            if images := _edge_question(views, picked, {}, stem, f"{prefix}{n}_edges"):
                found.append({**common, "id": f"{prefix}{n}_edges", "kind": kind, "feature": feature.feature,
                              "claim": CLAIMS[kind], "images": images})
    for n, body in enumerate(marks.bodies if "body interior" in kinds else []):
        if not body.solid:
            continue
        lo, hi = np.array(body.low, float), np.array(body.high, float)
        tints, picked = {}, {}
        for view in views:
            corners = [lo.copy() for _ in range(4)]
            for c, (fu, fv) in zip(corners, ((0, 0), (1, 0), (1, 1), (0, 1)), strict=True):
                c[view.iu], c[view.iv] = (lo, hi)[fu][view.iu], (lo, hi)[fv][view.iv]
            tints[view.name] = [("tint", [view.px(c) for c in corners])]
            picked[view.name] = [e for e in edges[view.name] if f"body{n}" in e.holders]
        if images := _edge_question(views, picked, tints, stem, f"body{n}_interior"):
            found.append({**common, "id": f"body{n}_interior", "kind": "body interior", "feature": body.feature,
                          "claim": CLAIMS["body interior"], "images": images})
    for n, body in enumerate(marks.bodies if "body extent" in kinds else []):
        for axis, view, sides in _extents(views, edges, f"body{n}"):
            b = _bounds([p for e in sides for p in e.px])
            size = max(b[2] - b[0], b[3] - b[1])
            name = f"body{n}_{'xyz'[axis]}_extent"
            images = _save_pair(view, _edge_shapes(sides), _at_least(_grow(b, max(40, 0.1 * size), 0.1), MIN_CROP_PX),
                                stem / f"{name}_{view.name}", plain=True)
            found.append({**common, "id": name, "kind": "body extent", "feature": body.feature,
                          "claim": CLAIMS["body extent"], "images": images})
    for n, hole in enumerate(holes if "hole edge" in kinds else []):
        owners = {f"hole{i}" for i, h in enumerate(marks.holes) if _coaxial(hole, h)}
        for k, (view, run) in enumerate(_hole_edges(views, edges, owners, hole)):
            b = _bounds([p for e in run for p in e.px])
            size = max(b[2] - b[0], b[3] - b[1])
            images = _save_pair(view, _edge_shapes(run), _at_least(_grow(b, max(40, 0.1 * size), 0.1), MIN_CROP_PX),
                                stem / f"hole{n}_edge{k}_{view.name}", plain=True)
            found.append({**common, "id": f"hole{n}_edge{k}", "kind": "hole edge", "feature": hole.feature,
                          "claim": CLAIMS["hole edge"], "images": images})
    for n, cavity in enumerate(marks.cavities if "cavity wall" in kinds else []):
        for k, (view, wall) in enumerate(_walls(views, edges, f"cavity{n}")):
            b = _bounds([p for e in wall for p in e.px])
            size = max(b[2] - b[0], b[3] - b[1])
            images = _save_pair(view, _edge_shapes(wall), _at_least(_grow(b, max(40, 0.1 * size), 0.1), MIN_CROP_PX),
                                stem / f"cavity{n}_wall{k}_{view.name}", plain=True)
            found.append({**common, "id": f"cavity{n}_wall{k}", "kind": "cavity wall", "feature": cavity.feature,
                          "claim": CLAIMS["cavity wall"], "images": images})
    return found


def _walls(views: list[ViewMap], edges: dict[str, list[Edge]], owner: str) -> list[tuple[ViewMap, list[Edge]]]:
    """A cavity's walls, each a run of its edges, in the best view that shows them: long enough,
    and not lying on another line of the same kind, which a reader could not tell from the wall."""
    from scipy.spatial import cKDTree

    view = next((v for v in views if _px_length([e for e in edges[v.name] if e.owner == owner]) >= MIN_EDGE_PX), None)
    if view is None:
        return []
    found = []
    for wall in runs([e for e in edges[view.name] if e.owner == owner]):
        if _px_length(wall) < MIN_RUN_PX:
            continue
        others = [p for e in edges[view.name] if e.kind == wall[0].kind and all(e is not w for w in wall) for p in e.px]
        points = np.array([p for e in wall for p in e.px])
        if others and (cKDTree(np.array(others)).query(points)[0] <= AMBIGUOUS_PX).mean() > 0.5:
            continue
        found.append(wall)
    found.sort(key=lambda w: (w[0].kind != "hidden", -_px_length(w)))
    return [(view, w) for w in found[:MAX_WALLS]]


def _extents(views: list[ViewMap], edges: dict[str, list[Edge]], owner: str) -> list[tuple[int, ViewMap, list[Edge]]]:
    """For each model axis, in the best view that has it, a body's outermost edges on both
    sides across that axis: the runs lying along its extreme there, or failing that, the runs
    reaching it, as a round body's outline does."""
    found = []
    for axis in range(3):
        for view in views:
            if axis not in (view.iu, view.iv):
                continue
            owned = [e for e in edges[view.name] if e.owner == owner]
            if _px_length(owned) < MIN_EDGE_PX:
                continue
            unit = np.eye(3)[axis]
            d = view.px(unit) - view.px(np.zeros(3))
            d /= np.linalg.norm(d)
            groups = runs(owned)
            reach = [np.array([p for e in g for p in e.px]) @ d for g in groups]
            low, high = min(r.min() for r in reach), max(r.max() for r in reach)
            sides = []
            for extreme in (low, high):
                along = [g for g, r in zip(groups, reach, strict=True) if (np.abs(r - extreme) <= EXTENT_PX).mean() >= 0.6]
                sides.append(along or [g for g, r in zip(groups, reach, strict=True) if np.abs(r - extreme).min() <= EXTENT_PX])
            if high - low >= MIN_RUN_PX and all(sides):
                found.append((axis, view, [e for side in sides for g in side for e in g]))
                break
    return found


def _hole_edges(views: list[ViewMap], edges: dict[str, list[Edge]], owners: set[str], hole) -> list[tuple[ViewMap, list[Edge]]]:
    """A hole's edges in the best view across it, each run a mark; a run along the hole much
    longer than its width is split into its near and far halves, so that a hole drawn shorter
    than built leaves one half bare."""
    from scipy.spatial import cKDTree

    view = next((v for v in views if v.in_plane(hole.axis)
                 and _px_length([e for e in edges[v.name] if e.owner in owners]) >= MIN_EDGE_PX), None)
    if view is None:
        return []
    width = min(w for _, w in hole.across) * np.linalg.norm(view.px(np.eye(3)[view.iu]) - view.px(np.zeros(3)))
    found = []
    for run in runs([e for e in edges[view.name] if e.owner in owners]):
        if _px_length(run) < MIN_RUN_PX:
            continue
        others = [p for e in edges[view.name] if e.kind == run[0].kind and all(e is not r for r in run) for p in e.px]
        points = np.array([p for e in run for p in e.px])
        if others and (cKDTree(np.array(others)).query(points)[0] <= AMBIGUOUS_PX).mean() > 0.5:
            continue
        if len(run) == 1 and _px_length(run) >= 4 * width:
            half = len(run[0].px) // 2
            found += [[run[0].model_copy(update={"px": run[0].px[:half + 1]})],
                      [run[0].model_copy(update={"px": run[0].px[half:]})]]
        else:
            found.append(run)
    found.sort(key=lambda r: -_px_length(r))
    return [(view, r) for r in found[:MAX_WALLS]]


def _at_least(box: tuple[float, ...], side: float) -> tuple[float, ...]:
    """A box grown about its centre to at least `side` wide and high."""
    x0, y0, x1, y1 = box
    gx, gy = max(0.0, side - (x1 - x0)) / 2, max(0.0, side - (y1 - y0)) / 2
    return (x0 - gx, y0 - gy, x1 + gx, y1 + gy)


def _reader(name: str):
    return sol("medium") if name == "sol" else luna("high", images=20)


def observe(model, question: dict, tree: Tree) -> tuple[dict | None, list, float]:
    asks = [node for node in tree.nodes if node.id != HIDDEN]
    text = Template(PROMPT).substitute(
        projection=question["projection"], hidden="" if question["hidden_drawn"] else " Hidden lines are not drawn.",
        mark=MARKS[question["kind"]].description,
        questions="\n".join(f"{node.id}: {node.ask} [{' / '.join(node.answers)}]" for node in asks))
    content = [{"type": "text", "text": text}]
    for shown in question["images"]:
        url = "data:image/png;base64," + base64.b64encode(Path(shown["path"]).read_bytes()).decode()
        content += [{"type": "text", "text": shown["label"]}, {"type": "image_url", "image_url": {"url": url}}]
    messages, usage, start = [HumanMessage(content=content)], [], time.monotonic()
    for _ in range(TRIES):
        answer = model.invoke(messages)
        usage.append(answer.usage_metadata)
        raw = answer.text
        try:
            got = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])["answers"]
            wrong = [f"{n.id} must be one of {list(n.answers)}" for n in asks if got.get(n.id) not in n.answers]
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            wrong = [f"unreadable answer: {error}"]
        if not wrong:
            return got, usage, time.monotonic() - start
        messages += [answer, HumanMessage(content="Fix your answer and give the whole JSON again: " + "; ".join(wrong))]
    return None, usage, time.monotonic() - start


def _trees(out: Path) -> dict:
    return json.loads((out / TREES).read_text())


def scale_spread(run: Path) -> float | None:
    """How far apart the registered scales of a reading's views are: on one sheet they should agree."""
    record = json.loads((run / "workspace/reconstruction.json").read_text())
    snapshots = [s for s in record["snapshots"] if (s.get("verification") or {}).get("drawing_diff_report")]
    if not snapshots:
        return None
    scales = []
    for result in snapshots[-1]["verification"]["drawing_diff_report"].values():
        matrix = ((result or {}).get("stats") or {}).get("drawing_to_model_uv")
        if matrix:
            scales.append(float(np.hypot(matrix[0][0], matrix[1][0])))
    return max(scales) / min(scales) if len(scales) > 1 else None


def doubt(question: dict, conclusion: str) -> bool:
    if question["kind"] == "hole band":
        return conclusion in ("through", "blind") and conclusion != question["claim"]
    return conclusion not in ("undecided", question["claim"])


def hole_doubt(conclusions: list[str]) -> bool:
    """A hole is in doubt when two of its edges, or all of them, have no line: one bare edge alone
    is as often a small difference elsewhere, such as an edge finish, as a hole of another depth."""
    bare = conclusions.count("no edge")
    return bare >= 2 or (bare >= 1 and bare == len(conclusions))


def report(out: Path, pairs: dict[str, Pair], judged: dict[str, Judged], readers: list[str]) -> None:
    questions = [json.loads(line) for line in open(out / "questions.jsonl")]
    trees = _trees(out)
    keys = sorted({q["key"] for q in questions})
    in_scope = {q for kind in CONTRADICTED_BY.values() for q, _ in kind}
    errors = [(key, e) for key in keys for e in judged[key].judgement.errors if e.decision.question in in_scope]
    print(f"{len(keys)} readings, {len(questions)} questions; {len(errors)} judged errors in scope: "
          + ", ".join(f"{k} {sum(q['kind'] == k for q in questions)}" for k in MARKS))
    spread = {key: scale_spread(Path(pairs[key].drawing).parents[2]) for key in keys}
    print("views disagreeing on scale (over 1.2x):", [k.split("/")[2] for k, v in spread.items() if v and v > SCALE_SPREAD])
    for reader in readers:
        answers = {(a["key"], a["id"]): a for a in map(json.loads, open(out / f"answers_{reader}.jsonl"))}
        decided = flags = hits = 0
        caught = set()
        listed = []
        holes: dict[tuple[str, str], list[tuple[dict, str]]] = {}
        for q in questions:
            a = answers.get((q["key"], q["id"]))
            if not a or not a.get("answers"):
                continue
            tree = Tree.model_validate(trees[str(q["tree_fold"])][q["kind"]])
            conclusion = tree.conclude(a["answers"], q["hidden_drawn"])
            decided += conclusion != "undecided"
            if q["kind"] == "hole edge":
                holes.setdefault((q["key"], q["id"].split("_")[0]), []).append((q, conclusion))
                continue
            if not doubt(q, conclusion):
                continue
            flags += 1
            match = [e for k, e in errors if k == q["key"] and e.feature in q["feature"].split(" + ")
                     and any(e.decision.question == eq and (r is None or e.decision.read == r) for eq, r in CONTRADICTED_BY[q["kind"]])
                     and (q["kind"] != "hole band" or e.decision.truth == conclusion)]
            hits += bool(match)
            caught.update(id(e) for e in match)
            listed.append(f"  {'HIT ' if match else 'miss'} {q['part']} {q['id']} {q['feature']} claim {q['claim']} -> {conclusion}: "
                          + " ".join(i["path"] for i in q["images"]))
        # A hole is one doubt, raised by its edges together.
        for (key, hole), edges in holes.items():
            if not hole_doubt([c for _, c in edges]):
                continue
            q = edges[0][0]
            flags += 1
            match = [e for k, e in errors if k == key and e.feature in q["feature"].split(" + ")
                     and any(e.decision.question == eq for eq, _ in CONTRADICTED_BY["hole edge"])]
            hits += bool(match)
            caught.update(id(e) for e in match)
            listed.append(f"  {'HIT ' if match else 'miss'} {q['part']} {hole} {q['feature']} edges without a line: "
                          + " ".join(e["images"][1]["path"] for e, c in edges if c == "no edge"))
        calls = [a for a in answers.values() if a.get("usage")]
        seconds = sum(a["seconds"] for a in calls)
        tokens = [sum(u["input_tokens"] for a in calls for u in a["usage"]), sum(u["output_tokens"] for a in calls for u in a["usage"])]
        print(f"\n{reader}: {len(calls)} answered, {decided} decided, {flags} doubts, {hits} match a judged error "
              f"(precision {hits / max(flags, 1):.0%}); judged errors caught {len(caught)}/{len(errors)}; "
              f"per reading {len(calls) / len(keys):.1f} marks, {sum(len(a['usage']) for a in calls) / len(keys):.1f} calls, "
              f"{seconds / len(keys):.0f} s, "
              f"{tokens[0] / len(keys):,.0f} input and {tokens[1] / len(keys):,.0f} output tokens")
        print("\n".join(listed))


if __name__ == "__main__" and sys.argv[1] == "questions":
    pairs = {p.key: p for p in read_rows(sys.argv[2], Pair) if p.which == "final"}
    out = Path(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)
    keys = list(pairs) if sys.argv[3] == "all" else sys.argv[3].split(",")
    marks_path = out / "marks.jsonl"
    done = {row["key"]: Marks.model_validate(row["marks"]) for row in map(json.loads, open(marks_path))} if marks_path.exists() else {}
    model, lock = luna("low"), Lock()

    kinds = tuple(sys.argv[5].split(",")) if len(sys.argv) > 5 else KINDS

    def work(key: str) -> list[dict]:
        pair = pairs[key]
        if key not in done:
            marks = extract(model, json.loads(Path(pair.interpretation).read_text()))
            with lock, open(marks_path, "a") as rows:
                rows.write(json.dumps({"key": key, "marks": marks.model_dump()}) + "\n")
            done[key] = marks
        try:
            return questions_for(pair, done[key], out, kinds)
        except Exception as error:  # one reading's failure must not stop the rest
            print(key, "FAILED", type(error).__name__, str(error)[:200], flush=True)
            return []

    with ThreadPoolExecutor(8) as pool:
        found = [q for qs in pool.map(work, keys) for q in qs]
    (out / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in found))
    print(len(found), "questions for", len(keys), "readings")
elif __name__ == "__main__" and sys.argv[1] == "observe":
    out, reader = Path(sys.argv[2]), sys.argv[3]
    questions = [json.loads(line) for line in open(out / "questions.jsonl")]
    trees, path, lock = _trees(out), out / f"answers_{reader}.jsonl", Lock()
    done = {(a["key"], a["id"]) for a in map(json.loads, open(path))} if path.exists() else set()
    model = _reader(reader)

    def ask(q: dict) -> None:
        tree = Tree.model_validate(trees[str(q["tree_fold"])][q["kind"]])
        try:
            got, usage, seconds = observe(model, q, tree)
        except Exception as error:  # one failing question must not stop the rest
            print(q["key"], q["id"], "FAILED", type(error).__name__, str(error)[:200], flush=True)
            return
        with lock, open(path, "a") as rows:
            rows.write(json.dumps({"key": q["key"], "id": q["id"], "answers": got, "usage": usage, "seconds": seconds}) + "\n")

    with ThreadPoolExecutor(int(sys.argv[4]) if len(sys.argv) > 4 else 8) as pool:
        list(pool.map(ask, [q for q in questions if (q["key"], q["id"]) not in done]))
elif __name__ == "__main__" and sys.argv[1] == "report":
    pairs = {p.key: p for p in read_rows(sys.argv[3], Pair)}
    judged = {row.key: row for row in read_rows(sys.argv[4], Judged) if row.judgement is not None}
    report(Path(sys.argv[2]), pairs, judged, sys.argv[5:])
