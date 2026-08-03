"""Score one predicted STEP against one ground-truth STEP, ECCV challenge rules.

The wrapper around the forked blackboxes in :mod:`.eccv_components`: it picks
the reference frames, seeds the two sampling streams apart, and names the
columns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from zeroshot.evaluation.metrics.eccv_components import (
    chamfer,
    load_step_brep,
    match_entities,
    match_incidence,
    match_or_empty,
    normalize_to_reference_bbox,
    reference_frame,
)

# Keeps the two sides' per-face seeds disjoint; must exceed `MAX_FACES` (5000).
_PRED_SEED_OFFSET = 10_007


def score_eccv(
    pred_step: Path,
    gt_step: Path,
    normalize_to_gt_bbox: bool = True,
    reference_extent: float | None = 1.8,
    f1_threshold: float = 0.1,
    seed: int = 0,
) -> dict[str, Any]:
    """Return the surface/edge/vertex/topology F1 columns for one prediction.

    ``reference_extent`` is the length each side's longest side is scaled to
    before anything is measured, so that the absolute ``f1_threshold`` and the
    per-unit-area sample density mean the same thing whatever units a file uses.
    ``None`` keeps the file's own units and only works when both sides are
    already normalized.
    """

    seed = int(seed)
    target_frame = (
        None if reference_extent is None else reference_frame(gt_step, reference_extent)
    )
    predicted_frame = (
        reference_frame(pred_step, reference_extent)
        if normalize_to_gt_bbox and reference_extent is not None
        else target_frame
    )
    # Separate seeds: a shared stream correlates the samples of similar faces
    # and biases every distance downwards.
    target = load_step_brep(gt_step, seed=seed, frame=target_frame)
    predicted = load_step_brep(
        pred_step, seed=seed + _PRED_SEED_OFFSET, frame=predicted_frame
    )
    if normalize_to_gt_bbox:
        predicted = normalize_to_reference_bbox(predicted, target)

    threshold = float(f1_threshold)
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
    return {
        "eccv_surface_f1": surface.f1,
        "eccv_surface_precision": surface.precision,
        "eccv_surface_recall": surface.recall,
        "eccv_edge_f1": edge.f1,
        "eccv_vertex_f1": vertex.f1,
        "eccv_face_edge_f1": face_edge_f1,
        "eccv_edge_vertex_f1": edge_vertex_f1,
        "eccv_topology_f1": (face_edge_f1 + edge_vertex_f1) / 2,
        "eccv_chamfer_surface": chamfer(predicted.face_pc, target.face_pc),
        "eccv_chamfer_edge": chamfer(predicted.edge_pc, target.edge_pc),
        "eccv_chamfer_vertex": chamfer(predicted.vertex_pc, target.vertex_pc),
        "eccv_num_pred_faces": surface.num_pred,
        "eccv_num_gt_faces": surface.num_gt,
        "eccv_num_pred_edges": edge.num_pred,
        "eccv_num_gt_edges": edge.num_gt,
        "eccv_num_pred_verts": vertex.num_pred,
        "eccv_num_gt_verts": vertex.num_gt,
    }


__all__ = ["score_eccv"]
