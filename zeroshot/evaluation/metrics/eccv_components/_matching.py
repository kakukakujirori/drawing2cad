"""Entity assignment and incidence F1 for the ECCV 2026 CAD Challenge.

Forked verbatim from the challenge's own evaluator at
``data/eccv2026-cad-challenge-data/examples/min_eval/eval.py`` (terms in
``data/eccv2026-cad-challenge-data/ACKNOWLEDGEMENTS_AND_LICENSES.md``). The
matching rule and the distance threshold are the leaderboard's, so they are
reproduced rather than reinterpreted:

- Each predicted entity is matched to at most one ground-truth entity of the
  same kind by a minimum-cost assignment over their symmetric mean
  nearest-neighbour distance; a pair counts as correct when that distance is
  below ``threshold``.
- Topology F1 compares the face-edge and edge-vertex incidence pairs, where a
  predicted pair can only be a true positive if both of its entities matched.

This module is a blackbox: it defines the metric, so it is not refactored,
tuned or "cleaned up". Its equivalence with the pre-existing port in
``src/metrics/eccv`` is pinned by a golden test.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


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


def match_or_empty(
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


def chamfer(pred_points: np.ndarray, gt_points: np.ndarray) -> float:
    from scipy.spatial import cKDTree

    if not len(pred_points) or not len(gt_points):
        return 0.0
    pred_to_gt, _ = cKDTree(pred_points).query(gt_points, k=1)
    gt_to_pred, _ = cKDTree(gt_points).query(pred_points, k=1)
    return float((pred_to_gt.mean() + gt_to_pred.mean()) / 2)


__all__ = [
    "EntityMatch",
    "chamfer",
    "match_entities",
    "match_incidence",
    "match_or_empty",
]
