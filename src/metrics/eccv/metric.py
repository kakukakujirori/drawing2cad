"""ECCV 2026 CAD Challenge scoring: entity F1 and topology F1.

Ported from the challenge's own evaluator at
``data/eccv2026-cad-challenge-data/examples/min_eval/eval.py`` (terms in
``data/eccv2026-cad-challenge-data/ACKNOWLEDGEMENTS_AND_LICENSES.md``). The
matching rule, the distance threshold and the summary formula are the
leaderboard's, so they are reproduced rather than reinterpreted:

- Each predicted entity is matched to at most one ground-truth entity of the
  same kind by a minimum-cost assignment over their symmetric mean
  nearest-neighbour distance; a pair counts as correct when that distance is
  below ``f1_threshold``.
- Topology F1 compares the face-edge and edge-vertex incidence pairs, where a
  predicted pair can only be a true positive if both of its entities matched.
- ``summary = valid_ratio * (surface_f1 + edge_f1 + vertex_f1 + topology_f1) / 4``
  with every per-axis F1 averaged over the samples that scored at all.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.metrics.base import GT_STEP, PRED_STEP, CADMetric, MetricSample, stem
from src.metrics.geometry import finite_float, mean_or_zero, median_or_zero
from src.metrics.registry import register_metric


#: Offset between the candidate's and the target's sampling seeds. Any value
#: past the official face cap keeps their per-face streams disjoint.
_PRED_SEED_OFFSET = 10_007


@dataclass(frozen=True)
class EntityMatch:
    """F1 of one entity kind plus the predicted-to-ground-truth label map."""

    f1: float
    precision: float
    recall: float
    num_pred: int
    num_gt: int
    matches: dict[int, int]


def match_entities(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    *,
    threshold: float,
) -> EntityMatch:
    """Assign predicted entities to ground-truth entities and score the F1."""

    from scipy.optimize import linear_sum_assignment
    from scipy.spatial import cKDTree

    unique_pred = np.unique(pred_labels)
    unique_gt = np.unique(gt_labels)
    pred_groups = {label: pred_points[pred_labels == label] for label in unique_pred}
    gt_groups = {label: gt_points[gt_labels == label] for label in unique_gt}
    pred_trees = {label: cKDTree(points) for label, points in pred_groups.items()}
    gt_trees = {label: cKDTree(points) for label, points in gt_groups.items()}

    cost = np.zeros((len(unique_pred), len(unique_gt)))
    for i, pred_label in enumerate(unique_pred):
        pred_subset = pred_groups[pred_label]
        for j, gt_label in enumerate(unique_gt):
            gt_subset = gt_groups[gt_label]
            if len(pred_subset) == 0 or len(gt_subset) == 0:
                continue
            pred_to_gt, _ = gt_trees[gt_label].query(pred_subset, k=1)
            gt_to_pred, _ = pred_trees[pred_label].query(gt_subset, k=1)
            cost[i, j] = (np.mean(pred_to_gt) + np.mean(gt_to_pred)) / 2

    rows, columns = linear_sum_assignment(cost)
    matches: dict[int, int] = {}
    for row, column in zip(rows, columns, strict=True):
        if cost[row, column] < threshold:
            matches[int(unique_pred[row])] = int(unique_gt[column])
    correct = len(matches)
    precision = correct / len(unique_pred)
    recall = correct / len(unique_gt)
    return EntityMatch(
        f1=2 * precision * recall / (precision + recall + 1e-6),
        precision=precision,
        recall=recall,
        num_pred=len(unique_pred),
        num_gt=len(unique_gt),
        matches=matches,
    )


def match_incidence(
    pred_matrix: np.ndarray,
    gt_matrix: np.ndarray,
    row_matches: Mapping[int, int],
    column_matches: Mapping[int, int],
) -> float:
    """F1 over incidence pairs, given the entity matches of both axes.

    An incidence pair whose entities did not both match is a false positive; a
    ground-truth pair nothing mapped onto is a false negative.
    """

    gt_pairs = {(int(i), int(j)) for i, j in np.argwhere(gt_matrix == 1)}
    true_positive = 0
    false_positive = 0
    matched_gt_pairs: set[tuple[int, int]] = set()
    for i, j in np.argwhere(pred_matrix == 1):
        gt_row = row_matches.get(int(i))
        gt_column = column_matches.get(int(j))
        if gt_row is None or gt_column is None:
            false_positive += 1
            continue
        if (gt_row, gt_column) in gt_pairs:
            true_positive += 1
            matched_gt_pairs.add((gt_row, gt_column))
        else:
            false_positive += 1
    false_negative = len(gt_pairs) - len(matched_gt_pairs)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return 2 * precision * recall / (precision + recall + 1e-6)


def _chamfer(pred_points: np.ndarray, gt_points: np.ndarray) -> float:
    from scipy.spatial import cKDTree

    if not len(pred_points) or not len(gt_points):
        return 0.0
    pred_to_gt, _ = cKDTree(pred_points).query(gt_points, k=1)
    gt_to_pred, _ = cKDTree(gt_points).query(pred_points, k=1)
    return float((pred_to_gt.mean() + gt_to_pred.mean()) / 2)


@register_metric
@dataclass(frozen=True)
class ECCVChallengeMetric(CADMetric):
    """Valid ratio, surface/edge/vertex F1, topology F1 and the challenge summary.

    ``normalize_to_gt_bbox`` mirrors the official evaluator's
    ``NORMALIZE_PRED_TO_GT_BBOX`` switch and defaults to on: the match
    threshold is an absolute distance, and our inputs carry no dimensions, so
    scoring a shape-correct prediction at the wrong size would otherwise report
    zero. Turn it off to keep a size error penalized.

    ``reference_extent`` is the length the target's longest side is scaled to
    before anything is measured. It defaults to the challenge's own convention
    -- every published target is exactly 1.8 units long -- which makes this a
    no-op on challenge data and keeps the metric meaningful on a dataset stored
    in millimetres, where an absolute 0.1 threshold and a per-unit-area sample
    density would otherwise mean something entirely different. The target's
    frame is also the prediction's whenever normalization is off, so a size
    error survives.
    """

    normalize_to_gt_bbox: bool = True
    reference_extent: float | None = 1.8
    f1_threshold: float = 0.1
    seed: int = 0

    requires = frozenset({PRED_STEP, GT_STEP})
    row_keys = (
        "eccv_valid",
        "eccv_surface_f1",
        "eccv_surface_precision",
        "eccv_surface_recall",
        "eccv_edge_f1",
        "eccv_vertex_f1",
        "eccv_face_edge_f1",
        "eccv_edge_vertex_f1",
        "eccv_topology_f1",
        "eccv_chamfer_surface",
        "eccv_chamfer_edge",
        "eccv_chamfer_vertex",
        "eccv_num_pred_faces",
        "eccv_num_gt_faces",
        "eccv_num_pred_edges",
        "eccv_num_gt_edges",
        "eccv_num_pred_verts",
        "eccv_num_gt_verts",
    )

    def score(self, sample: MetricSample) -> Mapping[str, Any]:
        from src.metrics.eccv._step_brep import (
            load_step_brep,
            normalize_to_reference_bbox,
            reference_frame,
        )

        seed = int(self.seed)
        gt_step, pred_step = sample.gt_step(), sample.pred_step()
        target_frame = (
            None
            if self.reference_extent is None
            else reference_frame(gt_step, self.reference_extent)
        )
        # Scaling the prediction onto the reference length is the same mapping
        # as the official point-cloud normalization, but applied before the
        # sample counts are derived from face areas, so both sides are sampled
        # at the density the metric is calibrated for.
        predicted_frame = (
            reference_frame(pred_step, self.reference_extent)
            if self.normalize_to_gt_bbox and self.reference_extent is not None
            else target_frame
        )
        target = load_step_brep(gt_step, seed=seed, frame=target_frame)
        # The two sides must be sampled from independent random streams. Sharing
        # one would correlate the samples of geometrically similar faces and bias
        # every distance downwards, which is not what the official evaluator
        # (independent, unseeded sampling) measures.
        predicted = load_step_brep(
            pred_step, seed=seed + _PRED_SEED_OFFSET, frame=predicted_frame
        )
        if self.normalize_to_gt_bbox:
            # A no-op up to sampling noise when the frames above already did it;
            # it still matters when `reference_extent` is off, where it is the
            # official evaluator's own normalization.
            predicted = normalize_to_reference_bbox(predicted, target)

        threshold = float(self.f1_threshold)
        surface = match_entities(
            predicted.face_pc,
            target.face_pc,
            predicted.face_labels,
            target.face_labels,
            threshold=threshold,
        )
        edge = _match_or_empty(
            predicted.edge_pc,
            target.edge_pc,
            predicted.edge_labels,
            target.edge_labels,
            threshold=threshold,
            gt_count=target.n_edges,
        )
        vertex = _match_or_empty(
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
            "eccv_valid": True,
            "eccv_surface_f1": surface.f1,
            "eccv_surface_precision": surface.precision,
            "eccv_surface_recall": surface.recall,
            "eccv_edge_f1": edge.f1,
            "eccv_vertex_f1": vertex.f1,
            "eccv_face_edge_f1": face_edge_f1,
            "eccv_edge_vertex_f1": edge_vertex_f1,
            "eccv_topology_f1": (face_edge_f1 + edge_vertex_f1) / 2,
            "eccv_chamfer_surface": _chamfer(predicted.face_pc, target.face_pc),
            "eccv_chamfer_edge": _chamfer(predicted.edge_pc, target.edge_pc),
            "eccv_chamfer_vertex": _chamfer(predicted.vertex_pc, target.vertex_pc),
            "eccv_num_pred_faces": surface.num_pred,
            "eccv_num_gt_faces": surface.num_gt,
            "eccv_num_pred_edges": edge.num_pred,
            "eccv_num_gt_edges": edge.num_gt,
            "eccv_num_pred_verts": vertex.num_pred,
            "eccv_num_gt_verts": vertex.num_gt,
        }

    def reduce(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        prefix: str,
    ) -> Mapping[str, float | int]:
        # A sample counts as valid only when this family scored it end to end.
        # Everything else -- no executable prediction, unreadable STEP, an
        # entity count past the official caps -- is invalid and lowers the valid
        # ratio, exactly as a missing submission does on the leaderboard.
        scored = [row for row in rows if bool(row.get("eccv_valid", False))]
        total = len(rows)
        valid_ratio = len(scored) / total if total else 0.0

        def mean_of(key: str) -> float:
            values = [
                value
                for value in (finite_float(row.get(key)) for row in scored)
                if value is not None
            ]
            return mean_or_zero(values)

        def median_of(key: str) -> float:
            values = [
                value
                for value in (finite_float(row.get(key)) for row in scored)
                if value is not None
            ]
            return median_or_zero(values)

        surface_f1 = mean_of("eccv_surface_f1")
        edge_f1 = mean_of("eccv_edge_f1")
        vertex_f1 = mean_of("eccv_vertex_f1")
        # Averaged from the two incidence axes rather than from the per-sample
        # mean, matching the official aggregation.
        topology_f1 = (
            mean_of("eccv_face_edge_f1") + mean_of("eccv_edge_vertex_f1")
        ) / 2

        head = stem(prefix)
        return {
            f"{head}eccv_valid_n": len(scored),
            f"{head}eccv_valid_ratio": valid_ratio,
            f"{head}eccv_surface_f1": surface_f1,
            f"{head}eccv_surface_precision": mean_of("eccv_surface_precision"),
            f"{head}eccv_surface_recall": mean_of("eccv_surface_recall"),
            f"{head}eccv_edge_f1": edge_f1,
            f"{head}eccv_vertex_f1": vertex_f1,
            f"{head}eccv_face_edge_f1": mean_of("eccv_face_edge_f1"),
            f"{head}eccv_edge_vertex_f1": mean_of("eccv_edge_vertex_f1"),
            f"{head}eccv_topology_f1": topology_f1,
            f"{head}eccv_summary": valid_ratio
            * (surface_f1 + edge_f1 + vertex_f1 + topology_f1)
            / 4,
            f"{head}eccv_median_chamfer_surface": median_of("eccv_chamfer_surface"),
            f"{head}eccv_median_chamfer_edge": median_of("eccv_chamfer_edge"),
            f"{head}eccv_median_chamfer_vertex": median_of("eccv_chamfer_vertex"),
        }


def _match_or_empty(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    *,
    threshold: float,
    gt_count: int,
) -> EntityMatch:
    """Score an entity kind, or return a zero match when either side is empty.

    A part with no edges or no vertices is degenerate rather than erroneous, so
    the official evaluator scores that axis zero and carries on.
    """

    if len(pred_points) and len(gt_points):
        return match_entities(
            pred_points, gt_points, pred_labels, gt_labels, threshold=threshold
        )
    return EntityMatch(
        f1=0.0,
        precision=0.0,
        recall=0.0,
        num_pred=len(np.unique(pred_labels)) if len(pred_labels) else 0,
        num_gt=gt_count,
        matches={},
    )


__all__ = ["ECCVChallengeMetric", "EntityMatch", "match_entities", "match_incidence"]
