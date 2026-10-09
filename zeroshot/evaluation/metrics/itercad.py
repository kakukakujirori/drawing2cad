"""AUC-TR and valid-only mean/median CD of IterCAD@55998c4 eval/evalution.py.

Thresholds follow its DEFAULT_CONFIG (1e-5..1e-1), not compute_metrics' 1e-6.
CD uses IterCAD's GT bbox-diagonal unit, but the shared scale and IoU pose.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


def _valid_cd(cd: float | None) -> bool:
    return cd is not None and bool(np.isfinite(cd)) and cd >= 0


def summarize_cds(
    cds: Iterable[float | None],
    *,
    min_cd: float = 1e-5,
    max_cd: float = 1e-1,
    num_points: int = 401,
) -> dict[str, Any]:
    """A None, nonfinite or negative CD is a failed generation, kept in the AUC denominator."""
    if not np.isfinite([min_cd, max_cd]).all() or not 0 < min_cd < max_cd:
        raise ValueError("require finite 0 < min_cd < max_cd")
    if (
        isinstance(num_points, bool)
        or not isinstance(num_points, int)
        or num_points < 2
    ):
        raise ValueError("num_points must be an integer >= 2")
    values = list(cds)
    valid = np.sort([float(cd) for cd in values if _valid_cd(cd)])
    xs = np.linspace(-np.log10(max_cd), -np.log10(min_cd), num_points)
    # Preserve exact end thresholds, including equality at the strictest one.
    thresholds = 10.0 ** (-xs)
    thresholds[0], thresholds[-1] = max_cd, min_cd
    recalls = np.searchsorted(valid, thresholds, side="right") / max(len(values), 1)
    return {
        "total_samples": len(values),
        "valid_cd_samples": len(valid),
        "invalid_predictions": len(values) - len(valid),
        "auc_tr": float(np.trapezoid(recalls, xs) / (xs[-1] - xs[0]))
        if values
        else 0.0,
        "mean_cd": float(valid.mean()) if len(valid) else None,
        "median_cd": float(np.median(valid)) if len(valid) else None,
        "auc_tr_min_cd": min_cd,
        "auc_tr_max_cd": max_cd,
        "auc_tr_num_points": num_points,
    }


def aggregate_itercad(
    reports: Iterable[Mapping[str, Any]], **auc_options: Any
) -> dict[str, Any]:
    """Reduce score_pair reports; a CD lost to an evaluator error nulls the CD aggregates."""
    reports = list(reports)
    protocols = [report.get("protocol") for report in reports if report.get("protocol")]
    if protocols and any(protocol != protocols[0] for protocol in protocols):
        raise ValueError("cannot aggregate different evaluation protocols")
    cds = []
    cd_errors = 0
    for report in reports:
        cd = report.get("metrics", {}).get("chamfer_diag")
        if report.get("build_valid") is False:
            cds.append(None)
        elif report.get("build_valid") is True and _valid_cd(cd):
            cds.append(cd)
        else:
            cd_errors += 1
    summary = summarize_cds(cds, **auc_options)
    summary["total_samples"] = len(reports)
    summary["evaluation_errors"] = cd_errors + sum(
        bool(report.get("errors"))
        for report in reports
        if report.get("build_valid") is True
        and _valid_cd(report.get("metrics", {}).get("chamfer_diag"))
    )
    summary["cd_evaluation_errors"] = cd_errors
    summary["protocol"] = protocols[0] if protocols else None
    if cd_errors:
        summary.update(auc_tr=None, mean_cd=None, median_cd=None)
    return summary
