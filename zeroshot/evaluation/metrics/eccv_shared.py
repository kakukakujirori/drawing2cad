"""ECCV matching with the shared GT scale and maximum-IoU pose.

Reuses the existing seeded challenge sampler and entity/incidence matching.
This is a reference F1 with dimension-preserving preprocessing, not a claim of
identical challenge preprocessing. No face splitting or independent rescaling.
Independent sampling at the official density can give F1 < 1 even on identical
geometry, particularly for small faces or closely spaced components.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from zeroshot.evaluation.metrics.eccv import score_eccv
from zeroshot.evaluation.preprocess import PreparedPair


def score_eccv_shared(
    pair: PreparedPair, *, f1_threshold: float = 0.1, seed: int = 0
) -> dict[str, Any]:
    if not np.isfinite(f1_threshold) or f1_threshold <= 0:
        raise ValueError("f1_threshold must be finite and positive")
    with TemporaryDirectory(prefix="eccv-shared-") as scratch:
        pred_path, gt_path = Path(scratch) / "pred.step", Path(scratch) / "gt.step"
        pair.pred_shape.exportStep(str(pred_path))
        pair.gt_shape.exportStep(str(gt_path))
        return score_eccv(
            pred_path,
            gt_path,
            normalize_to_gt_bbox=False,
            reference_extent=None,
            f1_threshold=f1_threshold,
            seed=seed,
        )
