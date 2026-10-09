"""Score one run's final solid against its ground truth, in a child process.

The metrics read STEP files through a CAD kernel. A malformed B-Rep can abort or hang
inside native code, which no Python ``try`` can catch, so scoring gets the same
containment as CadQuery execution and rendering: a fresh ``spawn`` child, a
wall-clock timeout, and a result delivered over a pipe.

Rescore finished samples with::

    python -m zeroshot.evaluation.run_scoring \\
        --run-dir outputs/<run>/*/ --target-dir data/test_vlm/target_step_ori
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import partial
from math import isfinite
from multiprocessing.connection import Connection
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from zeroshot.evaluation.align_orientation import (
    SAMPLE_POINTS as _ALIGN_SAMPLE_POINTS,
)
from zeroshot.evaluation.metrics import score_eccv, score_voxel

# A CAD kernel error can run to thousands of characters; this one is read by
# a human out of a JSON file.
_MAX_ERROR_CHARS = 500


class ScoreStatus(Enum):
    OK = "OK"
    PARTIAL = "PARTIAL"  # some metric raised; the others kept their columns
    FAILED = "FAILED"  # nothing scored
    TIMEOUT = "TIMEOUT"  # the scoring child overran its budget
    NO_PREDICTION = "NO_PREDICTION"  # the run produced no verified STEP
    GENERATION_FAILED = "GENERATION_FAILED"  # predicted STEP is not a valid solid


@dataclass(frozen=True)
class ScoreReport:
    status: ScoreStatus
    metrics: Mapping[str, float | int] = field(default_factory=dict)
    errors: Mapping[str, Any] = field(default_factory=dict)
    details: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "metrics": dict(self.metrics),
            "errors": dict(self.errors),
            **self.details,
        }


def _error_text(error: BaseException) -> str:
    message = " ".join(str(error).split())
    label = type(error).__name__
    return (f"{label}: {message}" if message else label)[:_MAX_ERROR_CHARS]


def _worker(
    scorer: StepScorer | SharedStepScorer,
    pred_step: Path,
    gt_step: Path,
    connection: Connection,
) -> None:
    try:
        connection.send(scorer._run_families(pred_step, gt_step))
    finally:
        connection.close()


def _isolated_score(
    scorer: StepScorer | SharedStepScorer, pred_step: Path, gt_step: Path
) -> ScoreReport | tuple[dict[str, float | int], dict[str, str]]:
    """Keep native crashes and hangs inside the same containment for both scorers."""
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker, args=(scorer, pred_step, gt_step, sender), daemon=True
    )
    process.start()
    sender.close()
    # Receive before joining: a full pipe otherwise blocks the scoring child.
    ready = receiver.poll(scorer.timeout_s)
    try:
        payload = receiver.recv() if ready else None
    except EOFError:
        payload = None
    finally:
        receiver.close()
    if not ready:
        process.terminate()
    process.join(5.0)
    if process.is_alive():
        process.kill()
        process.join()
    if payload is not None:
        return payload
    message = (
        f"scoring timed out after {scorer.timeout_s:g}s"
        if not ready
        else f"scoring process exited without a result (exitcode={process.exitcode})"
    )
    return ScoreReport(
        status=ScoreStatus.TIMEOUT if not ready else ScoreStatus.FAILED,
        errors={"scorer": {"kind": "evaluation_error", "message": message}},
        details={"build_valid": None},
    )


@dataclass(frozen=True)
class StepScorer:
    """Legacy ECCV/voxel scorer, retained for reproducing earlier runs."""

    timeout_s: float = 600.0
    f1_threshold: float = 0.1
    normalize_to_gt_bbox: bool = True
    reference_extent: float | None = 1.8
    voxel_resolution: int = 64
    seed: int = 0
    align_sample_points: int = _ALIGN_SAMPLE_POINTS
    split_closed_faces: bool = False

    def __post_init__(self) -> None:
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if self.align_sample_points < 1:
            raise ValueError("align_sample_points must be positive")

    def families(
        self,
    ) -> Mapping[str, Callable[[Path, Path], Mapping[str, float | int]]]:
        """Which metric families this scorer measures, bound to its parameters.

        The only place that knows which metrics exist. A family's key prefixes
        every column it returns, and is the key its failure appears under in
        ``ScoreReport.errors``.
        """

        return {
            "eccv": partial(
                score_eccv,
                normalize_to_gt_bbox=self.normalize_to_gt_bbox,
                reference_extent=self.reference_extent,
                f1_threshold=self.f1_threshold,
                seed=self.seed,
            ),
            "voxel": partial(score_voxel, resolution=self.voxel_resolution),
        }

    def _run_families(
        self,
        pred_step: Path,
        gt_step: Path,
    ) -> tuple[dict[str, float | int], dict[str, str]]:
        """Run every family, keeping the columns of the ones that succeeded.

        Private because it is the unsupervised path: it runs in the scoring
        child, and only :meth:`score` supervises the timeout and the crash.
        """

        columns: dict[str, float | int] = {}
        errors: dict[str, str] = {}
        with TemporaryDirectory() as scratch:
            # Every family measures the aligned solid, because every one of
            # them is orientation-sensitive: the voxel IoU of a correct part in
            # the wrong pose is as wrong as its F1.  There is no setting for
            # skipping this, because a number measured against a solid in the
            # wrong pose describes neither the part nor the leaderboard.
            try:
                pred_step = self._aligned(pred_step, gt_step, Path(scratch), columns)
            except Exception as error:  # noqa: BLE001 - an unaligned score beats none
                errors["align"] = _error_text(error)
            if self.split_closed_faces:
                try:
                    pred_step = self._split_closed(pred_step, Path(scratch))
                except Exception as error:  # noqa: BLE001 - as above
                    errors["normalize"] = _error_text(error)
            for name, call in self.families().items():
                try:
                    columns.update(call(pred_step, gt_step))
                except Exception as error:  # noqa: BLE001
                    errors[name] = _error_text(error)
        return columns, errors

    def _aligned(
        self,
        pred_step: Path,
        gt_step: Path,
        scratch: Path,
        columns: dict[str, float | int],
    ) -> Path:
        """Rotate the prediction into the target's pose, recording which pose.

        Imported here rather than at module scope for the same reason as
        `latest_verified_step`: every `spawn` scoring child re-imports this
        module, and only the one that scores needs CadQuery.
        """

        from zeroshot.evaluation.align_orientation import align_step

        output = scratch / "aligned.step"
        alignment = align_step(
            pred_step,
            gt_step,
            output,
            sample_points=self.align_sample_points,
            seed=self.seed,
        )
        columns["align_rotation_index"] = alignment.rotation_index
        columns["align_chamfer"] = alignment.chamfer
        # Above one, this sample's pose was a draw between orientations the
        # surfaces cannot tell apart, and the columns that depend on pose are
        # worth no more than the draw.
        columns["align_tied"] = alignment.tied
        return output

    def _split_closed(self, pred_step: Path, scratch: Path) -> Path:
        """Re-partition the prediction's faces the way the target's writer does.

        After the pose search, so both families measure the same file; the
        voxel IoU cannot tell the two partitions apart either way.
        """

        from zeroshot.evaluation.normalize_brep import split_closed_faces

        return split_closed_faces(pred_step, scratch / "normalized.step")

    def score(self, pred_step: Path, gt_step: Path) -> ScoreReport:
        """Score one prediction while supervising native-code hangs.

        A missing prediction is a run outcome and is reported. A missing target
        is a configuration error and is raised, since reporting it as an
        unscorable prediction would understate the model.
        """

        if not gt_step.is_file():
            raise FileNotFoundError(f"target STEP not found: {gt_step}")
        if not pred_step.is_file():
            return ScoreReport(
                status=ScoreStatus.NO_PREDICTION,
                errors={"prediction": f"no STEP at {pred_step}"},
            )

        payload = _isolated_score(self, pred_step, gt_step)
        if isinstance(payload, ScoreReport):
            return payload
        columns, errors = payload
        if not errors:
            status = ScoreStatus.OK
        elif columns:
            status = ScoreStatus.PARTIAL
        else:
            status = ScoreStatus.FAILED
        return ScoreReport(status=status, metrics=columns, errors=errors)


@dataclass(frozen=True)
class SharedStepScorer:
    """Dimension-preserving metrics, supervised independently of the agent pipeline."""

    timeout_s: float = 600.0
    reference_extent: float = 1.8
    mesh_tolerance: float = 0.001
    angular_tolerance: float = 0.1
    sample_points: int = 8192
    seed: int = 0
    f1_threshold: float = 0.1
    include_eccv: bool = True
    include_ortho2cad: bool = True
    auc_tr_min_cd: float = 1e-5
    auc_tr_max_cd: float = 1e-1
    auc_tr_num_points: int = 401

    def __post_init__(self) -> None:
        for name in (
            "timeout_s",
            "reference_extent",
            "mesh_tolerance",
            "angular_tolerance",
            "f1_threshold",
            "auc_tr_min_cd",
            "auc_tr_max_cd",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, minimum in (
            ("sample_points", 1),
            ("seed", 0),
            ("auc_tr_num_points", 2),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.auc_tr_min_cd >= self.auc_tr_max_cd:
            raise ValueError("auc_tr_min_cd must be smaller than auc_tr_max_cd")

    @staticmethod
    def families() -> tuple[str, ...]:
        return ("mesh", "chamfer", "hausdorff", "eccv", "ortho2cad")

    def _run_families(self, pred_step: Path, gt_step: Path) -> ScoreReport:
        from zeroshot.evaluation.preprocess import PreprocessConfig
        from zeroshot.evaluation.score_pair import score_pair

        report = score_pair(
            pred_step,
            gt_step,
            config=PreprocessConfig(
                self.reference_extent, self.mesh_tolerance, self.angular_tolerance
            ),
            sample_points=self.sample_points,
            seed=self.seed,
            f1_threshold=self.f1_threshold,
            include_eccv=self.include_eccv,
            include_ortho2cad=self.include_ortho2cad,
        )
        return ScoreReport(
            status=ScoreStatus(report["status"]),
            metrics=report["metrics"],
            errors=report["errors"],
            details={
                key: value
                for key, value in report.items()
                if key not in {"status", "metrics", "errors"}
            },
        )

    def score(self, pred_step: Path, gt_step: Path) -> ScoreReport:
        if not gt_step.is_file():
            raise FileNotFoundError(f"target STEP not found: {gt_step}")
        if not pred_step.is_file():
            return ScoreReport(
                status=ScoreStatus.NO_PREDICTION,
                errors={
                    "prediction": {
                        "kind": "generation_failure",
                        "message": f"no STEP at {pred_step}",
                    }
                },
                details={"build_valid": False},
            )
        report = _isolated_score(self, pred_step, gt_step)
        assert isinstance(report, ScoreReport)
        return report


def latest_verified_step(
    run_dir: Path,
    last_only: bool = True,
    verification_dirname: str = "attempts",
) -> Path | None:
    """Return the STEP to score, or ``None`` when the run offers none.

    Rounds and coding attempts are numbered upwards. The largest coding attempt
    in the latest round is therefore the run's submission.
    """

    # Imported here rather than at module scope: `PipelineRunner` pulls in
    # langchain, langgraph and torch, and this module is re-imported by every
    # `spawn` scoring child, which needs none of them.
    from zeroshot.pipeline.runner import PipelineRunner

    attempts_dir = run_dir / PipelineRunner.WORKSPACE_DIRNAME / verification_dirname
    if not attempts_dir.is_dir():
        return None
    round_dirs = sorted(
        (
            path
            for path in attempts_dir.iterdir()
            if path.is_dir()
            and path.name.startswith("round_")
            and path.name.removeprefix("round_").isdigit()
        ),
        key=lambda path: int(path.name.removeprefix("round_")),
        reverse=True,
    )
    for round_dir in round_dirs:
        coding_dir = round_dir / "coding"
        if not coding_dir.is_dir():
            continue
        attempt_dirs = sorted(
            (
                path
                for path in coding_dir.iterdir()
                if path.is_dir() and path.name.isdigit()
            ),
            key=lambda path: int(path.name),
            reverse=True,
        )
        for attempt_dir in attempt_dirs:
            step_path = attempt_dir / "output.step"
            if step_path.is_file():
                return step_path
            if last_only:
                return None

    return None


def score_run(
    run_dir: Path,
    target_step: Path,
    scorer: StepScorer | SharedStepScorer,
    last_only: bool = True,
) -> dict[str, object]:
    """Score what a finished run submitted, as a JSON-ready document.

    The settings are recorded next to the numbers because a metric value only
    means something together with the threshold, seed and attempt choice that
    produced it.
    """

    pred_step = latest_verified_step(run_dir, last_only)
    report = (
        ScoreReport(
            status=ScoreStatus.NO_PREDICTION,
            errors={
                "prediction": {
                    "kind": "generation_failure",
                    "message": f"no verified STEP under {run_dir}",
                }
            },
            details={"build_valid": False},
        )
        if pred_step is None
        else scorer.score(pred_step, target_step)
    )
    return {
        "run_dir": str(run_dir),
        "pred_step": None if pred_step is None else str(pred_step),
        "target_step": str(target_step),
        "last_only": last_only,
        "scorer": asdict(scorer),
        "evaluator": type(scorer).__name__,
        **report.as_dict(),
    }


def main() -> None:
    """Score sample directories against ``<target-dir>/<sample_id>.step``.

    Exit 1 only when a scorer crashed or timed out, not when a model failed.
    """

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        nargs="+",
        required=True,
        help="sample directories, e.g. outputs/<run>/*/ (skips ones without events.jsonl)",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        required=True,
        help="ground-truth STEPs named <sample_id>.step",
    )
    parser.add_argument("--timeout-s", type=float, default=SharedStepScorer.timeout_s)
    parser.add_argument(
        "--last-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "score only the run's final attempt, which is what it submitted. "
            "--no-last-only falls back to the last attempt that produced a "
            "solid, which can only raise the score"
        ),
    )
    parser.add_argument(
        "--sample-points",
        type=int,
        default=SharedStepScorer.sample_points,
        help="surface samples per side for squared CD and Hausdorff",
    )
    parser.add_argument(
        "--reference-extent",
        type=float,
        default=SharedStepScorer.reference_extent,
    )
    parser.add_argument("--seed", type=int, default=SharedStepScorer.seed)
    parser.add_argument(
        "--f1-threshold", type=float, default=SharedStepScorer.f1_threshold
    )
    parser.add_argument("--skip-eccv", action="store_true")
    parser.add_argument("--skip-ortho2cad", action="store_true")
    parser.add_argument(
        "--auc-tr-min-cd", type=float, default=SharedStepScorer.auc_tr_min_cd
    )
    parser.add_argument(
        "--auc-tr-max-cd", type=float, default=SharedStepScorer.auc_tr_max_cd
    )
    parser.add_argument(
        "--auc-tr-num-points", type=int, default=SharedStepScorer.auc_tr_num_points
    )
    args = parser.parse_args()
    if not args.target_dir.is_dir():
        parser.error(f"target directory not found: {args.target_dir}")

    scorer = SharedStepScorer(
        timeout_s=args.timeout_s,
        sample_points=args.sample_points,
        reference_extent=args.reference_extent,
        seed=args.seed,
        f1_threshold=args.f1_threshold,
        include_eccv=not args.skip_eccv,
        include_ortho2cad=not args.skip_ortho2cad,
        auc_tr_min_cd=args.auc_tr_min_cd,
        auc_tr_max_cd=args.auc_tr_max_cd,
        auc_tr_num_points=args.auc_tr_num_points,
    )
    scorer_failed = False
    for run_dir in args.run_dir:
        target_step = args.target_dir / f"{run_dir.name}.step"
        if not (run_dir / "events.jsonl").is_file():
            print(f"skip       {run_dir}: no events.jsonl")
            continue
        if not target_step.is_file():
            print(f"skip       {run_dir}: no target {target_step}")
            continue
        print(f"sample     {run_dir}")
        document = score_run(run_dir, target_step, scorer, args.last_only)
        output_path = run_dir / "score.json"
        output_path.write_text(
            json.dumps(document, indent=2, allow_nan=False), encoding="utf-8"
        )
        print(f"status     {document['status']}")
        print(f"prediction {document['pred_step']}")
        for key, value in document["metrics"].items():  # type: ignore[attr-defined]
            print(f"  {key:<26} {value}")
        for name, message in document["errors"].items():  # type: ignore[attr-defined]
            print(f"  ! {name}: {message}")
        print(f"written    {output_path}")
        scorer_failed |= document["status"] in {
            ScoreStatus.FAILED.value,
            ScoreStatus.TIMEOUT.value,
        }

    if scorer_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
