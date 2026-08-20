"""Score one run's final solid against its ground truth, in a child process.

The metrics read STEP files through a CAD kernel. A malformed B-Rep can abort or hang
inside native code, which no Python ``try`` can catch, so scoring gets the same
containment as CadQuery execution and rendering: a fresh ``spawn`` child, a
wall-clock timeout, and a result delivered over a pipe.

Run it over a finished sample directory with::

    python -m zeroshot.evaluation.run_scoring \\
        --run-dir outputs/<run>/<sample_id> \\
        --target-step data/test_vlm/target_step/<sample_id>.step
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import partial
from multiprocessing.connection import Connection
from pathlib import Path
from tempfile import TemporaryDirectory

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


@dataclass(frozen=True)
class ScoreReport:
    status: ScoreStatus
    metrics: Mapping[str, float | int] = field(default_factory=dict)
    errors: Mapping[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "metrics": dict(self.metrics),
            "errors": dict(self.errors),
        }


def _error_text(error: BaseException) -> str:
    message = " ".join(str(error).split())
    label = type(error).__name__
    return (f"{label}: {message}" if message else label)[:_MAX_ERROR_CHARS]


def _worker(
    scorer: StepScorer,
    pred_step: Path,
    gt_step: Path,
    connection: Connection,
) -> None:
    try:
        connection.send(scorer._run_families(pred_step, gt_step))
    finally:
        connection.close()


@dataclass(frozen=True)
class StepScorer:
    """Score a predicted STEP against a target STEP under a wall-clock budget."""

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

        context = mp.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=_worker,
            args=(self, pred_step, gt_step, sender),
            daemon=True,
        )
        process.start()
        sender.close()

        process.join(self.timeout_s)
        timed_out = process.is_alive()
        if timed_out:
            process.terminate()
            process.join(5.0)
            if process.is_alive():
                process.kill()
                process.join()

        try:
            payload = receiver.recv() if not timed_out and receiver.poll() else None
        except EOFError:
            payload = None
        finally:
            receiver.close()

        if timed_out:
            return ScoreReport(
                status=ScoreStatus.TIMEOUT,
                errors={"scorer": f"scoring timed out after {self.timeout_s:g}s"},
            )
        if payload is None:
            # A native abort leaves no per-family attribution to report.
            return ScoreReport(
                status=ScoreStatus.FAILED,
                errors={
                    "scorer": "scoring process exited without a result "
                    f"(exitcode={process.exitcode})"
                },
            )

        columns, errors = payload
        if not errors:
            status = ScoreStatus.OK
        elif columns:
            status = ScoreStatus.PARTIAL
        else:
            status = ScoreStatus.FAILED
        return ScoreReport(status=status, metrics=columns, errors=errors)


def latest_verified_step(
    run_dir: Path,
    last_only: bool = True,
    verification_dirname: str = "attempts",
) -> Path | None:
    """Return the STEP to score, or ``None`` when the run offers none.

    Attempts are numbered upwards and the workflow-driven final verification
    runs last, so the largest numbered directory is the run's submission.
    """

    # Imported here rather than at module scope: `PipelineRunner` pulls in
    # langchain, langgraph and torch, and this module is re-imported by every
    # `spawn` scoring child, which needs none of them.
    from zeroshot.pipeline.runner import PipelineRunner

    attempts_dir = run_dir / PipelineRunner.WORKSPACE_DIRNAME / verification_dirname
    if not attempts_dir.is_dir():
        return None
    # Mirrors how `tools/verify_output.py` issues these names.
    attempt_ids = sorted(
        (path.name for path in attempts_dir.iterdir() if path.name.isdigit()),
        key=int,
        reverse=True,
    )
    for attempt_id in attempt_ids:
        step_path = attempts_dir / attempt_id / "output.step"
        if step_path.is_file():
            return step_path
        if last_only:
            return None
    return None


def score_run(
    run_dir: Path,
    target_step: Path,
    scorer: StepScorer,
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
            errors={"prediction": f"no verified STEP under {run_dir}"},
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
        **report.as_dict(),
    }


def main() -> None:
    """Score a finished sample directory against its ground-truth STEP.

    Writes ``score.json`` next to the run's ``events.jsonl`` and prints the
    columns. The document records the scorer's settings alongside the numbers,
    because a metric value is only meaningful with the threshold and seed that
    produced it.

    The exit code reports whether *scoring* worked, not whether the model did:
    a prediction that scored zero, or a run that never produced a solid, is a
    successful measurement and exits 0. Only a scorer that crashed or timed out
    exits 1.
    """

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="a sample's artifact directory, e.g. outputs/<run>/<sample_id>",
    )
    parser.add_argument(
        "--target-step",
        type=Path,
        required=True,
        help="ground-truth STEP for that sample",
    )
    parser.add_argument("--timeout-s", type=float, default=StepScorer.timeout_s)
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
        "--align-sample-points",
        type=int,
        default=StepScorer.align_sample_points,
        help="surface samples per side for the 24-orientation pose search",
    )
    parser.add_argument(
        "--split-closed-faces",
        action=argparse.BooleanOptionalAction,
        default=StepScorer.split_closed_faces,
        help=(
            "split the prediction's closed periodic faces at their seams, so "
            "its face partition matches the SolidWorks writer the targets came "
            "from. A dataset-specific normalization, not a metric change"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="where to write the report (default: <run-dir>/score.json)",
    )
    args = parser.parse_args()

    # `score` raises on a missing target; an operator typo deserves better.
    if not args.target_step.is_file():
        parser.error(f"target STEP not found: {args.target_step}")
    if not args.run_dir.is_dir():
        parser.error(f"run directory not found: {args.run_dir}")

    document = score_run(
        args.run_dir,
        args.target_step,
        StepScorer(
            timeout_s=args.timeout_s,
            align_sample_points=args.align_sample_points,
            split_closed_faces=args.split_closed_faces,
        ),
        args.last_only,
    )

    output_path = args.output or args.run_dir / "score.json"
    output_path.write_text(json.dumps(document, indent=2), encoding="utf-8")

    print(f"status     {document['status']}")
    print(f"last only  {document['last_only']}")
    print(f"prediction {document['pred_step']}")
    for key, value in document["metrics"].items():  # type: ignore[attr-defined]
        print(f"  {key:<26} {value}")
    for name, message in document["errors"].items():  # type: ignore[attr-defined]
        print(f"  ! {name}: {message}")
    print(f"written    {output_path}")

    if document["status"] in {ScoreStatus.FAILED.value, ScoreStatus.TIMEOUT.value}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
