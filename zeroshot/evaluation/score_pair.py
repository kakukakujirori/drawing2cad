"""Score one STEP pair; the pipeline runs the same function in a supervised subprocess.

python -m zeroshot.evaluation.score_pair --pred pred.step --gt gt.step
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

from zeroshot.evaluation.preprocess import (
    IOU_TIE_TOLERANCE,
    AlignmentError,
    InvalidPredictionError,
    PreprocessConfig,
    prepare_pair,
)


def _error(error: Exception, kind: str = "evaluation_error") -> dict[str, str]:
    return {"kind": kind, "message": f"{type(error).__name__}: {error}"[:1000]}


def score_pair(
    pred_step: str | Path,
    gt_step: str | Path,
    *,
    config: PreprocessConfig | None = None,
    sample_points: int = 8192,
    seed: int = 0,
    f1_threshold: float = 0.1,
    include_eccv: bool = True,
    include_ortho2cad: bool = True,
) -> dict[str, Any]:
    """Return a JSON-ready report that blames each failure on the model or the evaluator."""
    from zeroshot.evaluation.metrics.surface_distance import score_surface_distance

    config = config or PreprocessConfig()
    report: dict[str, Any] = {
        "status": "FAILED",
        "build_valid": None,
        "metrics": {},
        "errors": {},
        "protocol": {
            "normalization": "shared_gt_longest_side",
            **asdict(config),
            "alignment": "bbox_centres_24_rotations_maximum_mesh_iou",
            "alignment_tie_tolerance": IOU_TIE_TOLERANCE,
            "eccv_pose": "best_of_iou_ties",
            "sample_points": sample_points,
            "seed": seed,
            "f1_threshold": f1_threshold,
            "cd_definition": "sum_of_directional_mean_squared_distances",
            "hd_definition": "maximum_of_directional_maximum_squared_distances",
            "auc_tr_cd": "chamfer_diag",
            "cadquery_version": version("cadquery"),
            "manifold3d_version": version("manifold3d"),
            "trimesh_version": version("trimesh"),
        },
    }
    try:
        pair = prepare_pair(pred_step, gt_step, config)
    except InvalidPredictionError as error:
        report.update(status="GENERATION_FAILED", build_valid=False)
        report["errors"]["prediction"] = _error(error, "generation_failure")
        return report
    except Exception as error:  # noqa: BLE001 - record evaluator failure, never use unaligned CD
        stage = "alignment" if isinstance(error, AlignmentError) else "preprocess"
        report["errors"][stage] = _error(error)
    else:
        report["build_valid"] = True
        report["preprocess"] = pair.metadata
        report["metrics"]["mesh_iou"] = pair.iou
        try:
            report["metrics"].update(
                score_surface_distance(pair, sample_points=sample_points, seed=seed)
            )
        except Exception as error:  # noqa: BLE001 - independent metric failures retain other results
            report["errors"]["surface_distance"] = _error(error)
        if include_eccv:
            from zeroshot.evaluation.metrics.eccv import score_eccv

            try:
                report["metrics"].update(
                    score_eccv(pair, f1_threshold=f1_threshold, seed=seed)
                )
            except Exception as error:  # noqa: BLE001
                report["errors"]["eccv"] = _error(error)
    # Ortho2CAD reads the original STEPs, so a shared-preprocessing error does not block it.
    if include_ortho2cad:
        from zeroshot.evaluation.metrics.ortho2cad import (
            ORTHO2CAD_SOURCE_REVISION,
            score_ortho2cad,
        )

        report["ortho2cad_protocol"] = {
            "normalization": "official_independent_inertia",
            "source_revision": ORTHO2CAD_SOURCE_REVISION,
            "official_cadquery_version": "2.5.2",
            "cadquery_version": version("cadquery"),
        }
        try:
            report["metrics"].update(score_ortho2cad(pred_step, gt_step))
        except Exception as error:  # noqa: BLE001
            report["errors"]["ortho2cad"] = _error(error)
    report["status"] = (
        "OK" if not report["errors"] else "PARTIAL" if report["metrics"] else "FAILED"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-extent", type=float, default=1.8)
    parser.add_argument("--sample-points", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--f1-threshold", type=float, default=0.1)
    parser.add_argument("--skip-eccv", action="store_true")
    parser.add_argument("--skip-ortho2cad", action="store_true")
    args = parser.parse_args()
    report = score_pair(
        args.pred,
        args.gt,
        config=PreprocessConfig(reference_extent=args.reference_extent),
        sample_points=args.sample_points,
        seed=args.seed,
        f1_threshold=args.f1_threshold,
        include_eccv=not args.skip_eccv,
        include_ortho2cad=not args.skip_ortho2cad,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
