"""A reading's claims as marks on its views: holes, bodies and cavities in the model
frame, taken from the reading by the local model, and drawn onto each view through
the drawing-to-model map that the pipeline's comparison stores for the build.

    python -m experiments.rag.marks PAIRS.jsonl KEYS OUT_DIR

KEYS is a comma-separated list of pair keys. Writes OUT_DIR/<part>.json with the
marks, and OUT_DIR/<part>_<view>.png with all of them drawn, for checking by eye:
holes as red bands across the part with their mouths outlined, bodies as blue bars
at their faces, cavities as green lines at their walls.
"""

import json
import sys
from pathlib import Path
from string import Template
from typing import Literal

import numpy as np
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from experiments.rag.reread import luna
from experiments.rag.schema import Pair, read_rows

PROMPT = Path(__file__).with_name("marks.md").read_text()
TRIES = 3
AXES = {"+x": (0, 1), "-x": (0, -1), "+y": (1, 1), "-y": (1, -1), "+z": (2, 1), "-z": (2, -1)}

Vector = tuple[float, float, float]


class Hole(BaseModel):
    feature: str
    center: Vector
    axis: Vector
    across: list[tuple[Vector, float]]
    entry: float
    end: float
    claim: Literal["through", "blind"]


class Box(BaseModel):
    feature: str
    low: Vector
    high: Vector


class Body(Box):
    solid: bool


class Marks(BaseModel):
    holes: list[Hole] = []
    bodies: list[Body] = []
    cavities: list[Box] = []


def extract(model, interpretation: dict) -> Marks:
    reading = {"datum": interpretation.get("datum"), "features": [
        {k: f.get(k) for k in ("name", "description", "parameters")} for f in interpretation["features"]]}
    messages = [HumanMessage(content=Template(PROMPT).substitute(reading=json.dumps(reading, indent=1)))]
    for _ in range(TRIES):
        answer = model.invoke(messages)
        raw = answer.text
        try:
            return Marks.model_validate_json(raw[raw.index("{"):raw.rindex("}") + 1])
        except ValueError as error:
            messages += [answer, HumanMessage(content=f"Fix your answer and give the whole JSON again: {str(error)[:800]}")]
    raise RuntimeError("no valid marks")


class ViewMap:
    """One registered orthographic view: model points to its pixels."""

    def __init__(self, view: dict, drawing_to_model_uv: list[list[float]], image_path: Path,
                 match_score: float | None = None) -> None:
        self.name, self.role, self.image_path, self.match_score = view["name"], view["role"], image_path, match_score
        self.u_axis, self.v_axis = view["u_axis"], view["v_axis"]
        (self.iu, self.su), (self.iv, self.sv) = AXES[view["u_axis"]], AXES[view["v_axis"]]
        self.normal = 3 - self.iu - self.iv
        self.to_pixel = np.linalg.inv(np.array(drawing_to_model_uv))

    def px(self, point) -> np.ndarray:
        return (self.to_pixel @ [self.su * point[self.iu], self.sv * point[self.iv], 1.0])[:2]

    def in_plane(self, direction) -> bool:
        return abs(np.asarray(direction, float)[self.normal]) < 0.1


def view_maps(run: Path) -> dict[str, ViewMap]:
    """The views of the run's last verified build, with their stored drawing-to-model maps."""
    record = json.loads((run / "workspace/reconstruction.json").read_text())
    snapshot = [s for s in record["snapshots"] if (s.get("verification") or {}).get("drawing_diff_report")][-1]
    report = snapshot["verification"]["drawing_diff_report"]
    maps = {}
    for view in snapshot["interpretation"]["views"]:
        stats = (report.get(view["name"]) or {}).get("stats") or {}
        if view.get("u_axis") and "drawing_to_model_uv" in stats:
            maps[view["name"]] = ViewMap(view, stats["drawing_to_model_uv"], run / "workspace" / view["file"].removeprefix("/work/"),
                                         stats.get("match_score"))
    return maps


def _coaxial(a: Hole, b: Hole) -> bool:
    axis = np.array(a.axis, float) / np.linalg.norm(a.axis)
    if abs(axis @ (np.array(b.axis, float) / np.linalg.norm(b.axis))) < 0.99:
        return False
    offset = np.array(b.center, float) - np.array(a.center, float)
    tolerance = 0.25 * min(w for _, w in [*a.across, *b.across])
    return float(np.linalg.norm(offset - axis * (offset @ axis))) < tolerance


def stepped(holes: list[Hole]) -> list[Hole]:
    """Holes on one axis, such as a counterbore, countersink or step and the bore it
    opens, as one hole: as wide as its narrowest part, through if any part goes through."""
    groups: list[list[Hole]] = []
    for hole in holes:
        group = next((g for g in groups if _coaxial(g[0], hole)), None)
        if group is None:
            groups.append([hole])
        else:
            group.append(hole)
    return [min(group, key=lambda h: min(w for _, w in h.across)).model_copy(update={
        "feature": " + ".join(sorted({h.feature for h in group})),
        "claim": "through" if any(h.claim == "through" for h in group) else "blind"}) for group in groups]


def material_along(hole: Hole, bodies: list[Body], low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The part's box with its extent along the hole's axis cut to the bodies the hole runs through."""
    along = int(np.argmax(np.abs(hole.axis)))
    center = np.array(hole.center, float)
    around = [b for b in bodies if all(b.low[i] - 1e-6 <= center[i] <= b.high[i] + 1e-6 for i in range(3) if i != along)]
    lo, hi = low.copy(), high.copy()
    if around:
        lo[along] = max(low[along], min(b.low[along] for b in around))
        hi[along] = min(high[along], max(b.high[along] for b in around))
    return lo, hi


EDGE_TOL = 0.003  # share of the part's diagonal an edge may stray outside a feature's box
EDGE_INSIDE = 0.9  # share of an edge's length that must lie in a box for the box to hold it


def feature_boxes(marks: Marks) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Every body, cavity and hole of a reading as a named box in the model frame."""
    boxes = [(f"body{n}", np.array(b.low, float), np.array(b.high, float)) for n, b in enumerate(marks.bodies)]
    boxes += [(f"cavity{n}", np.array(c.low, float), np.array(c.high, float)) for n, c in enumerate(marks.cavities)]
    for n, hole in enumerate(marks.holes):
        centre = np.array(hole.center, float)
        low, high = centre.copy(), centre.copy()
        along = int(np.argmax(np.abs(hole.axis)))
        low[along], high[along] = min(hole.entry, hole.end), max(hole.entry, hole.end)
        for direction, width in hole.across:
            k = int(np.argmax(np.abs(direction)))
            low[k], high[k] = centre[k] - width / 2, centre[k] + width / 2
        boxes.append((f"hole{n}", low, high))
    return boxes


class Edge(BaseModel):
    kind: Literal["visible", "hidden"]
    px: list[tuple[float, float]]  # along the edge, in the view's pixels
    holders: list[str]  # the feature boxes it lies in
    owner: str | None  # of those, the one that looks smallest in this view


def build_edges(build, view: ViewMap, marks: Marks, diag: float, hidden_drawn: bool) -> list[Edge]:
    """The build's edges in a view, each given to the smallest-looking feature box that holds it:
    in projection a thin plate seen face-on can cover features in front of it."""
    from zeroshot.pipeline.verification.render.orthographic import project

    visible, hidden = project(build, view.u_axis, view.v_axis)
    rects = {}
    for name, low, high in feature_boxes(marks):
        a = np.array([view.su * low[view.iu], view.sv * low[view.iv]])
        b = np.array([view.su * high[view.iu], view.sv * high[view.iv]])
        rects[name] = (np.minimum(a, b) - EDGE_TOL * diag, np.maximum(a, b) + EDGE_TOL * diag)
    edges = []
    for kind, group in (("visible", visible), ("hidden", hidden if hidden_drawn else [])):
        for edge in group:
            count = max(2, int(edge.Length() / (0.002 * diag)) + 1)
            uv = np.array([(p.x, p.y) for p in edge.positions(np.linspace(0, 1, count).tolist())])
            holders = [n for n, (lo, hi) in rects.items() if ((uv >= lo) & (uv <= hi)).all(axis=1).mean() >= EDGE_INSIDE]
            owner = min(holders, key=lambda n: float(np.prod(rects[n][1] - rects[n][0]))) if holders else None
            px = [tuple((view.to_pixel @ [u, v, 1.0])[:2]) for u, v in uv]
            edges.append(Edge(kind=kind, px=px, holders=holders, owner=owner))
    return _distinct(edges)


def _distinct(edges: list[Edge], tol: float = 1.0) -> list[Edge]:
    """Edges less any that lies on a longer one of the same kind: hidden-line removal can return
    an edge once for each face that meets there."""
    from scipy.spatial import cKDTree

    kept: list[Edge] = []
    for edge in sorted(edges, key=lambda e: -len(e.px)):
        same = [p for k in kept if k.kind == edge.kind for p in k.px]
        if same and (cKDTree(np.array(same)).query(np.array(edge.px))[0] <= tol).mean() >= 0.95:
            continue
        kept.append(edge)
    return kept


JOIN_PX = 2.0  # edge ends this close in a view meet
SMOOTH_DEG = 30.0  # edges meeting at a smaller turn continue one line


def _ends(px: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Each end of a polyline with the direction leaving the line there."""
    out = []
    for end, inner in ((px[0], px[min(len(px) - 1, 3)]), (px[-1], px[max(0, len(px) - 4)])):
        d = end - inner
        out.append((end, d / (np.linalg.norm(d) + 1e-9)))
    return out


def runs(edges: list[Edge]) -> list[list[Edge]]:
    """Edges of one kind joined end to end where the line goes on smoothly: each run is one
    wall or outline piece as the view shows it, split at corners."""
    parent = list(range(len(edges)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    ends = [_ends(np.array(e.px)) for e in edges]
    for i in range(len(edges)):
        for j in range(i):
            if edges[i].kind != edges[j].kind:
                continue
            for a, da in ends[i]:
                for b, db in ends[j]:
                    # Smooth: leaving one line is entering the other, so the directions oppose.
                    if np.linalg.norm(a - b) <= JOIN_PX and -da @ db >= np.cos(np.radians(SMOOTH_DEG)):
                        parent[root(i)] = root(j)
    groups: dict[int, list[Edge]] = {}
    for i, e in enumerate(edges):
        groups.setdefault(root(i), []).append(e)
    return list(groups.values())


def part_box(step: str) -> tuple[np.ndarray, np.ndarray]:
    import cadquery as cq

    box = cq.importers.importStep(step).val().BoundingBox()
    return np.array([box.xmin, box.ymin, box.zmin]), np.array([box.xmax, box.ymax, box.zmax])


def hole_band(view: ViewMap, hole: Hole, low: np.ndarray, high: np.ndarray) -> list[np.ndarray] | None:
    """The band over a hole in a view across it: its width there, along its axis from outline to outline."""
    axis, center = np.array(hole.axis, float), np.array(hole.center, float)
    if not view.in_plane(axis):
        return None
    across = [(np.array(d, float), w) for d, w in hole.across if view.in_plane(d)]
    if not across:
        return None
    side, width = across[0]
    side /= np.linalg.norm(side)
    along = int(np.argmax(np.abs(axis)))
    ends = [center + axis * (t - center[along]) / axis[along] for t in (low[along], high[along])]
    return [view.px(p) for p in (ends[0] + side * width / 2, ends[1] + side * width / 2,
                                 ends[1] - side * width / 2, ends[0] - side * width / 2)]


def hole_mouth(view: ViewMap, hole: Hole) -> list[np.ndarray] | None:
    """The hole's outline in a view along its axis, as the box of its widths."""
    if view.in_plane(hole.axis):
        return None
    center = np.array(hole.center, float)
    corners = [center]
    for direction, width in hole.across:
        d = np.array(direction, float) / np.linalg.norm(direction)
        corners = [c + s * d * width / 2 for c in corners for s in (-1, 1)]
    points = np.array([view.px(c) for c in corners])
    (x0, y0), (x1, y1) = points.min(axis=0), points.max(axis=0)
    return [np.array(p) for p in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]


def box_lines(view: ViewMap, box: Box, margin: float = 0.1) -> list[tuple[np.ndarray, np.ndarray]]:
    """Segments at a box's faces that the view sees edge-on, each spanning the box's other in-plane extent."""
    low, high = np.array(box.low, float), np.array(box.high, float)
    segments = []
    for face_axis, span_axis in ((view.iu, view.iv), (view.iv, view.iu)):
        pad = (high[span_axis] - low[span_axis]) * margin
        for value in (low[face_axis], high[face_axis]):
            a, b = low.copy(), low.copy()
            a[face_axis] = b[face_axis] = value
            a[span_axis], b[span_axis] = low[span_axis] - pad, high[span_axis] + pad
            segments.append((view.px(a), view.px(b)))
    return segments


def draw_all(view: ViewMap, marks: Marks, low: np.ndarray, high: np.ndarray, path: Path) -> None:
    from PIL import Image, ImageDraw

    image = Image.open(view.image_path).convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for hole in marks.holes:
        if band := hole_band(view, hole, low, high):
            draw.polygon([tuple(p) for p in band], fill=(255, 0, 0, 60))
        if mouth := hole_mouth(view, hole):
            draw.polygon([tuple(p) for p in mouth], outline=(255, 0, 0, 255), width=2)
    for body in marks.bodies:
        for a, b in box_lines(view, body):
            draw.line((tuple(a), tuple(b)), fill=(0, 80, 255, 140), width=4)
    for cavity in marks.cavities:
        for a, b in box_lines(view, cavity):
            draw.line((tuple(a), tuple(b)), fill=(0, 170, 0, 160), width=3)
    Image.alpha_composite(image, layer).convert("RGB").save(path)


if __name__ == "__main__":
    pairs = {pair.key: pair for pair in read_rows(sys.argv[1], Pair)}
    out = Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    model = luna("low")
    for key in sys.argv[2].split(","):
        pair = pairs[key]
        interpretation = json.loads(Path(pair.interpretation).read_text())
        marks = extract(model, interpretation)
        (out / f"{pair.part}.json").write_text(marks.model_dump_json(indent=1))
        low, high = part_box(pair.build)
        for name, view in view_maps(Path(pair.drawing).parents[2]).items():
            draw_all(view, marks, low, high, out / f"{pair.part}_{name}.png")
        print(key, f"{len(marks.holes)} holes, {len(marks.bodies)} bodies, {len(marks.cavities)} cavities", flush=True)
