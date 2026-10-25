"""Evaluation only: for each hole question, whether the real part (moved into the build's frame)
and the build have material along the hole's axis over the band's extent.

    python hole_oracle.py FOLDER OUT.jsonl [WORKERS]

FOLDER holds hole band questions and marks.jsonl; hole edge questions of the same readings
number their holes the same way, so one truth file serves both.

Truth per hole: "through" when the axis is empty over the band, "blind" when material closes it
somewhere but the entry end is open, "absent" when both ends are closed.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

A = Path("/home/ryotaro/my_works/drawing2cad/outputs/20261024_rag_pilot/analysis_100")
SAMPLES = 120


def classify(shape, points):
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.gp import gp_Pnt
    from OCP.TopAbs import TopAbs_IN, TopAbs_ON

    out = []
    for solid in [shape] if not hasattr(shape, "Solids") else shape.Solids():
        classifier = BRepClass3d_SolidClassifier(solid.wrapped)
        inside = []
        for p in points:
            classifier.Perform(gp_Pnt(*map(float, p)), 1e-6)
            inside.append(classifier.State() in (TopAbs_IN, TopAbs_ON))
        out.append(inside)
    return np.any(np.array(out), axis=0)


def verdict(inside: np.ndarray) -> str:
    if inside.mean() < 0.02:
        return "through"
    n = len(inside)
    head, tail = inside[: max(1, n // 10)].mean(), inside[-max(1, n // 10):].mean()
    if head > 0.5 and tail > 0.5:
        return "absent"
    return "blind"


def work(job):
    from experiments.rag.grounded import HOLES_PER_FEATURE
    from experiments.rag.marks import Marks, material_along, part_box, stepped
    from experiments.rag.schema import Comparison, Pair, read_rows
    from zeroshot.evaluation.preprocess import ORIENTATIONS
    from zeroshot.pipeline.verification.render.orthographic import load_shape
    from zeroshot.pipeline.verification.solid_mesh import shape_bbox, transform_shape

    key, marks_json, ids = job
    pair = {p.key: p for p in read_rows(A / "pairs.jsonl", Pair)}[key]
    rotation = {c.key: c.rotation_index for c in read_rows(A / "geometry/comparisons.jsonl", Comparison)}[key]
    marks = Marks.model_validate(marks_json)
    build, part = load_shape(Path(pair.build)), load_shape(Path(pair.gt))
    centre = shape_bbox(build)[0]
    rot = ORIENTATIONS[rotation]
    part = transform_shape(part, rot.T, centre - rot.T @ shape_bbox(part)[0])
    low, high = part_box(pair.build)
    kept, per = [], {}
    for hole in stepped(marks.holes):
        if per.setdefault(hole.feature, 0) < HOLES_PER_FEATURE:
            per[hole.feature] += 1
            kept.append(hole)
    rows = []
    for n, hole in enumerate(kept):
        if f"hole{n}" not in ids:
            continue
        lo, hi = material_along(hole, marks.bodies, low, high)
        axis, centre_h = np.array(hole.axis, float), np.array(hole.center, float)
        along = int(np.argmax(np.abs(axis)))
        t = np.linspace(lo[along], hi[along], SAMPLES)
        points = np.array([centre_h + axis * (s - centre_h[along]) / axis[along] for s in t])
        # Entry end first: the reading's entry, if it lies at the high end of the band.
        if abs(hole.entry - hi[along]) < abs(hole.entry - lo[along]):
            points = points[::-1]
        rows.append({"key": key, "id": f"hole{n}", "feature": hole.feature, "claim": hole.claim,
                     "truth": verdict(classify(part, points)), "build": verdict(classify(build, points))})
    return rows


if __name__ == "__main__":
    folder, out = Path(sys.argv[1]), Path(sys.argv[2])
    qs = [json.loads(l) for l in open(folder / "questions.jsonl") if json.loads(l)["kind"] == "hole band"]
    marks = {json.loads(l)["key"]: json.loads(l)["marks"] for l in open(folder / "marks.jsonl")}
    jobs = {}
    for q in qs:
        jobs.setdefault(q["key"], set()).add(q["id"])
    with ProcessPoolExecutor(int(sys.argv[3]) if len(sys.argv) > 3 else 4) as pool, open(out, "w") as f:
        for rows in pool.map(work, [(k, marks[k], ids) for k, ids in jobs.items()]):
            for r in rows:
                f.write(json.dumps(r) + "\n")
