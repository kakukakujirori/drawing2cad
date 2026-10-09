"""Ortho2CAD's official inertia-normalized B-Rep IoU, separate from main metrics.

The cq_align_shapes body is vendored unchanged from
https://github.com/AdityaJoglekar/Ortho2CAD/blob/546efb1facc4066e7904aacd259c9dd140c1e2ba/src/scripts/compute_iou.py
including its four candidates, transformGeometry, and Boolean exception->0
behavior. It independently normalizes each shape and can hide size errors.
The official environment pins CadQuery 2.5.2; record the installed version when
comparing scores. Never pass shared-preprocessed shapes to this comparator.
The official transformGeometry/Boolean path can also produce IoU < 1 for an
identical STEP pair; that behavior is retained for the literature comparison.
"""

# Preserve the official function's annotations and exception semantics.
# ruff: noqa: E722, UP006, UP035

import contextlib
import io
from pathlib import Path
from typing import Tuple

import cadquery as cq
import numpy as np

ORTHO2CAD_SOURCE_REVISION = "546efb1facc4066e7904aacd259c9dd140c1e2ba"


def cq_align_shapes(
    source: cq.Workplane, target: cq.Workplane
) -> Tuple[cq.Workplane, float]:
    """Align source to target using the center of mass and the principal axes of inertia. also return normalized IOU"""
    c_source = cq.Shape.centerOfMass(source.val())
    c_target = cq.Shape.centerOfMass(target.val())

    I_source = np.array(cq.Shape.matrixOfInertia(source.val()))
    I_target = np.array(cq.Shape.matrixOfInertia(target.val()))

    v_source = cq.Shape.computeMass(source.val())
    v_target = cq.Shape.computeMass(target.val())

    I_p_source, I_v_source = np.linalg.eigh(I_source)
    I_p_target, I_v_target = np.linalg.eigh(I_target)

    if v_source <= 0:
        return None, 0.0, c_source, c_target
    else:
        s_source = np.sqrt(np.abs(I_p_source).sum() / v_source)
        s_target = np.sqrt(np.abs(I_p_target).sum() / v_target)

        normalized_source = source.translate(-c_source).val().scale(1 / s_source)
        normalized_target = target.translate(-c_target).val().scale(1 / s_target)

        Rs = np.zeros((4, 3, 3))
        Rs[0] = I_v_target @ I_v_source.T

        for i in range(3):
            # all possible 2 out of 3 permutations
            alignment = 1 - 2 * np.array([i > 0, (i + 1) % 2, i % 3 <= 1])
            Rs[i + 1] = I_v_target @ (alignment[None, :] * I_v_source).T

        best_IOU = 0.0
        best_T = None
        for i in range(4):
            T = np.zeros([4, 4])
            T[:3, :3] = Rs[i]
            T[-1, -1] = 1

            aligned_source = normalized_source.transformGeometry(cq.Matrix(T.tolist()))

            try:
                intersect = aligned_source.intersect(normalized_target)
                union = aligned_source.fuse(normalized_target)

                IOU = intersect.Volume() / union.Volume()
            except:  # handle cases where IOU is undefined
                IOU = 0.0

            if IOU > best_IOU:
                best_IOU = IOU
                best_T = T
        if best_IOU > 1.0:
            print("Error: IOU greater than 1.0", best_IOU)
            best_IOU = 1.0
        if best_IOU < 0.2:
            print("Warning: Low IOU found:", best_IOU)
        if best_T is not None:
            aligned_source = (
                normalized_source.transformGeometry(cq.Matrix(best_T.tolist()))
                .scale(s_target)
                .translate(c_target)
            )
            print("IoU:", best_IOU)
            return cq.Workplane(aligned_source), best_IOU, c_source, c_target
        else:
            aligned_source = None
            print("IoU:", best_IOU)
            return aligned_source, best_IOU, c_source, c_target


def score_ortho2cad(pred_step: str | Path, gt_step: str | Path) -> dict[str, float]:
    pred = cq.importers.importStep(str(pred_step))
    gt = cq.importers.importStep(str(gt_step))
    # Keep the official diagnostics out of the standalone JSON report.
    with contextlib.redirect_stdout(io.StringIO()):
        _, iou, _, _ = cq_align_shapes(pred, gt)
    if not np.isfinite(iou):
        raise ValueError("Ortho2CAD returned nonfinite IoU")
    return {"ortho2cad_iou": float(iou)}
