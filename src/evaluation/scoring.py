"""Crash- and timeout-contained metric scoring for one sample.

Metrics read STEP files through CAD kernels (OCP for CADGenBench, pythonocc for
the ECCV challenge). A malformed B-Rep can abort or hang inside native code,
which no Python ``try`` can catch, so scoring gets the same containment as
CadQuery execution: a fresh ``spawn`` child, a wall-clock timeout, and a result
delivered over a pipe. The training process therefore never imports a CAD
kernel.

All selected metrics share one child per sample. That keeps the process count
bounded and lets a metric family reuse another's cached artifacts through
:class:`~src.metrics.base.MetricSample`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import multiprocessing as mp
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

# Imported for its side effect: it caps the worker thread counts before NumPy is
# ever loaded, with the same rationale as for CadQuery execution.
import src.evaluation.executor  # noqa: F401
from src.metrics.base import (
    GT_MESH,
    GT_STEP,
    PRED_MESH,
    PRED_STEP,
    CADMetric,
    MetricSample,
)


@dataclass(frozen=True)
class ScoringResult:
    """Columns produced for one sample, plus whatever refused to be computed."""

    columns: dict[str, Any] = field(default_factory=dict)
    #: Metric name to short error text, for metrics that raised on this sample.
    metric_errors: dict[str, str] = field(default_factory=dict)
    #: Set when the whole scoring child failed (timeout, crash, bad payload).
    error: str | None = None


def _short_error(error: BaseException) -> str:
    message = " ".join(str(error).split())
    label = type(error).__name__
    return (f"{label}: {message}" if message else label)[:500]


def _detach_main_module() -> None:
    """Stop this worker's own children from re-importing the training entrypoint.

    CADGenBench meshes STEP files in a nested ``spawn`` pool of its own. A
    ``spawn`` child re-imports whatever ``__main__`` names, which here would be
    ``src/train_sft.py`` -- pulling torch, transformers and Hydra into a process
    that only needs to tessellate a solid (and failing outright when the entry
    point is not an importable file). Dropping ``__file__``/``__spec__`` removes
    that step from the grandchild's preparation data; nothing else in the
    scoring path reads them.
    """

    import sys

    main = sys.modules.get("__main__")
    if main is None:
        return
    main.__spec__ = None
    if getattr(main, "__file__", None) is not None:
        main.__file__ = None


def _score_worker(
    child: Connection,
    sample_id: str,
    workdir: str,
    paths: Mapping[str, str | None],
    metrics: Sequence[CADMetric],
) -> None:
    payload: dict[str, Any] = {"columns": {}, "metric_errors": {}}
    try:
        _detach_main_module()
        sample = MetricSample(
            sample_id,
            workdir=workdir,
            pred_mesh_path=paths.get(PRED_MESH),
            gt_mesh_path=paths.get(GT_MESH),
            pred_step_path=paths.get(PRED_STEP),
            gt_step_path=paths.get(GT_STEP),
        )
        for metric in metrics:
            name = type(metric).__name__
            try:
                payload["columns"].update(metric.score(sample))
            except BaseException as error:
                # One metric failing must not cost the others their columns:
                # a sample that trips CADGenBench's mesh gate can still carry a
                # perfectly meaningful voxel IoU.
                payload["metric_errors"][name] = _short_error(error)
        child.send(payload)
    except BaseException as error:
        payload["error"] = _short_error(error)
        try:
            child.send(payload)
        except BaseException:
            pass
    finally:
        child.close()


def score_sample(
    sample_id: str,
    *,
    workdir: str | Path,
    metrics: Sequence[CADMetric],
    pred_mesh_path: str | Path | None = None,
    gt_mesh_path: str | Path | None = None,
    pred_step_path: str | Path | None = None,
    gt_step_path: str | Path | None = None,
    timeout_s: float = 600.0,
) -> ScoringResult:
    """Score one sample with every metric, in an isolated child process."""

    if timeout_s <= 0.0:
        raise ValueError(f"timeout_s must be positive, got {timeout_s!r}")
    if not metrics:
        return ScoringResult()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    paths = {
        PRED_MESH: None if pred_mesh_path is None else str(pred_mesh_path),
        GT_MESH: None if gt_mesh_path is None else str(gt_mesh_path),
        PRED_STEP: None if pred_step_path is None else str(pred_step_path),
        GT_STEP: None if gt_step_path is None else str(gt_step_path),
    }

    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_score_worker,
        args=(
            child,
            sample_id,
            str(workdir),
            paths,
            tuple(metrics),
        ),
        daemon=False,
    )
    process.start()
    child.close()
    process.join(timeout_s)
    if process.is_alive():
        process.terminate()
        process.join(1.0)
        if process.is_alive():
            process.kill()
            process.join(1.0)
        parent.close()
        process.close()
        return ScoringResult(error="timeout")

    received: Any = None
    if parent.poll(0.25):
        try:
            received = parent.recv()
        except (EOFError, OSError):
            received = None
    parent.close()
    exit_code = process.exitcode
    process.close()
    if not isinstance(received, Mapping):
        return ScoringResult(error=f"process_exit:{exit_code}")
    return ScoringResult(
        columns=dict(received.get("columns") or {}),
        metric_errors=dict(received.get("metric_errors") or {}),
        error=received.get("error"),
    )


__all__ = ["ScoringResult", "score_sample"]
