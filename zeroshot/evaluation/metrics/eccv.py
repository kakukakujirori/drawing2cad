"""ECCV 2026 CAD Challenge F1 of a pair already in the shared scale and IoU pose.

Sampling and matching are the challenge's own; the challenge itself rescales the
prediction independently and picks the pose by Chamfer distance.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from zeroshot.evaluation.metrics.eccv_components import (
    StepBRep,
    load_step_brep,
    match_entities,
    match_incidence,
    match_or_empty,
)
from zeroshot.evaluation.preprocess import PreparedPair

# Keeps the two sides' per-face seeds disjoint; must exceed `MAX_FACES` (5000).
_PRED_SEED_OFFSET = 10_007


def score_eccv(
    pair: PreparedPair, *, f1_threshold: float = 0.1, seed: int = 0
) -> dict[str, Any]:
    """Return the best mean F1 over the IoU-tied poses, then the columns behind it.

    Tied poses are the same geometry, but a symmetric part's seams make the F1
    depend on the pose. Independent sampling can also give F1 < 1 on small faces.
    """
    if not np.isfinite(f1_threshold) or f1_threshold <= 0:
        raise ValueError("f1_threshold must be finite and positive")
    with TemporaryDirectory(prefix="eccv-") as scratch:
        gt_path = Path(scratch) / "gt.step"
        pair.gt_shape.exportStep(str(gt_path))
        target = load_step_brep(gt_path, seed=seed)
        candidates = []
        for index, shape in pair.tied_shapes.items():
            pred_path = Path(scratch) / f"pred_{index}.step"
            shape.exportStep(str(pred_path))
            predicted = load_step_brep(pred_path, seed=seed + _PRED_SEED_OFFSET)
            candidates.append(
                _match(predicted, target, f1_threshold) | {"eccv_rotation_index": index}
            )
    return max(candidates, key=lambda columns: columns["eccv_mean_f1"])


def _match(predicted: StepBRep, target: StepBRep, threshold: float) -> dict[str, Any]:
    surface = match_entities(
        predicted.face_pc,
        target.face_pc,
        predicted.face_labels,
        target.face_labels,
        threshold=threshold,
    )
    edge = match_or_empty(
        predicted.edge_pc,
        target.edge_pc,
        predicted.edge_labels,
        target.edge_labels,
        threshold=threshold,
        gt_count=target.n_edges,
    )
    vertex = match_or_empty(
        predicted.vertex_pc,
        target.vertex_pc,
        predicted.vertex_labels,
        target.vertex_labels,
        threshold=threshold,
        gt_count=target.n_verts,
    )
    face_edge_f1 = match_incidence(
        predicted.fe_matrix, target.fe_matrix, surface.matches, edge.matches
    )
    edge_vertex_f1 = match_incidence(
        predicted.ev_matrix, target.ev_matrix, edge.matches, vertex.matches
    )
    topology_f1 = (face_edge_f1 + edge_vertex_f1) / 2
    return {
        "eccv_mean_f1": (surface.f1 + edge.f1 + vertex.f1 + topology_f1) / 4,
        "eccv_surface_f1": surface.f1,
        "eccv_surface_precision": surface.precision,
        "eccv_surface_recall": surface.recall,
        "eccv_edge_f1": edge.f1,
        "eccv_vertex_f1": vertex.f1,
        "eccv_face_edge_f1": face_edge_f1,
        "eccv_edge_vertex_f1": edge_vertex_f1,
        "eccv_topology_f1": topology_f1,
        "eccv_num_pred_faces": surface.num_pred,
        "eccv_num_gt_faces": surface.num_gt,
        "eccv_num_pred_edges": edge.num_pred,
        "eccv_num_gt_edges": edge.num_gt,
        "eccv_num_pred_verts": vertex.num_pred,
        "eccv_num_gt_verts": vertex.num_gt,
    }
