"""Score one run's final solid against its ground truth, in a child process.

The metrics read STEP files through a CAD kernel. A malformed B-Rep can abort or hang
inside native code, which no Python ``try`` can catch, so scoring gets the same
containment as CadQuery execution and rendering: a fresh ``spawn`` child, a
wall-clock timeout, and a result delivered over a pipe.

Rescore finished samples with::

    python -m zeroshot.evaluation.run_scoring \\
        --run-dir outputs/<run>/*/ --target-dir data/test_vlm/target_step_ori

A staged run is scored by its newest verified attempt. A native run
(`zeroshot.pipeline_native`) leaves only `workspace/model.py`, so that program
is built first, in the same sandbox and through the same STEP checks.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from math import isfinite
from multiprocessing.connection import Connection
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from zeroshot.pipeline.sandbox import SandboxRunner

# A native run's program is built beside its workspace, not in it, so the
# workspace stays as the agent left it.
NATIVE_BUILD_DIRNAME = "scoring_build"
# What both pipelines give a program to build: `sandbox_runner.default_timeout_s`.
DEFAULT_BUILD_TIMEOUT_S = 30.0


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


def _isolated_score(scorer: StepScorer, pred_step: Path, gt_step: Path) -> ScoreReport:
    """Score in a spawned child, so native crashes and hangs become a report."""
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
    """Score a predicted STEP against a target STEP under a wall-clock budget."""

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
        return _isolated_score(self, pred_step, gt_step)


def latest_verified_step(
    run_dir: Path,
    last_only: bool = True,
    verification_dirname: str = "attempts",
) -> tuple[Path | None, dict[str, str]]:
    """Return the STEP to score, or ``None``, and why each newer attempt was skipped.

    Rounds and coding attempts are numbered upwards, so the newest attempt whose
    STEP passes the pipeline's own STEP checks is the run's submission.
    """

    # Imported here rather than at module scope: `PipelineRunner` pulls in
    # langchain, langgraph and torch, and this module is re-imported by every
    # `spawn` scoring child, which needs none of them.
    from zeroshot.pipeline.runner import PipelineRunner
    from zeroshot.pipeline.verification.run_cadquery import (
        CadQueryExecutor,
        StepVerificationError,
    )

    skipped: dict[str, str] = {}
    attempts_dir = run_dir / PipelineRunner.WORKSPACE_DIRNAME / verification_dirname
    if not attempts_dir.is_dir():
        return None, skipped
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
            try:
                CadQueryExecutor.verify_step(step_path)
                return step_path, skipped
            except StepVerificationError as error:
                skipped[str(attempt_dir.relative_to(attempts_dir))] = str(error)
            if last_only:
                return None, skipped

    return None, skipped


def is_native_run(run_dir: Path) -> bool:
    """Whether `zeroshot.pipeline_native` made this sample, as its run.json says.

    Asked of run.json rather than read off a missing `attempts/`: a staged run
    whose interpretation never validated has a `model.py` and no attempt either,
    and building that program would score what the staged pipeline never did.
    """
    try:
        config = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        return config["config"]["workflow"]["name"] == "native"
    except (OSError, ValueError, KeyError, TypeError):
        return False


def sandbox_for_builds(timeout_s: float = DEFAULT_BUILD_TIMEOUT_S) -> SandboxRunner:
    """The sandbox a native program is built in, under this interpreter."""
    from zeroshot.pipeline.sandbox import SandboxRunner

    return SandboxRunner(
        python_executable=Path(sys.executable), default_timeout_s=timeout_s
    )


def native_step(
    run_dir: Path, sandbox_runner: SandboxRunner
) -> tuple[Path | None, dict[str, str], str | None]:
    """Build a native run's `workspace/model.py`, or say why it gave no solid.

    Also returns the build's status, `None` when there is no program to build.

    The staged pipeline's executor builds it, so its STEP export and checks are
    the ones a staged attempt passes. Catching exceptions is allowed: the native
    agent was never told otherwise, as the staged coder is.
    """
    from zeroshot.pipeline.runner import PipelineRunner
    from zeroshot.pipeline.verification.run_cadquery import (
        CadQueryExecutor,
        ExecutionStatus,
    )

    relative = Path(PipelineRunner.WORKSPACE_DIRNAME) / "model.py"
    program = run_dir / relative
    # The native runner records a symlinked or missing program as a failure.
    if program.is_symlink() or not program.is_file():
        return None, {}, None
    build_dir = run_dir / NATIVE_BUILD_DIRNAME
    build_dir.mkdir(exist_ok=True)
    step_path = build_dir / "output.step"
    step_path.unlink(missing_ok=True)
    report = CadQueryExecutor(sandbox_runner, forbid_try_except=False).execute(
        program, step_path
    )
    if report.status is ExecutionStatus.VERIFIED and report.step_path is not None:
        return report.step_path, {}, report.status.value
    reason = (
        report.executor_error
        or report.stderr.strip()[-2000:]
        or f"exit code {report.returncode}"
    )
    return None, {str(relative): f"{report.status}: {reason}"}, report.status.value


def score_run(
    run_dir: Path,
    target_step: Path,
    scorer: StepScorer,
    last_only: bool = True,
    sandbox_runner: SandboxRunner | None = None,
) -> dict[str, object]:
    """Score what a finished run submitted, as a JSON-ready document.

    The settings are recorded next to the numbers because a metric value only
    means something together with the threshold, seed and attempt choice that
    produced it. A native run's program is built in `sandbox_runner`, by
    default `sandbox_for_builds()`; `last_only` has no choice to make there.
    """

    native = is_native_run(run_dir)
    build: dict[str, object] = {}
    if native:
        pred_step, skipped, status = native_step(
            run_dir, sandbox_runner or sandbox_for_builds()
        )
        # What the staged pipeline's final verification says of its program.
        build = {"native_build_status": status}
    else:
        pred_step, skipped = latest_verified_step(run_dir, last_only)
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
        # Newer attempts, with the STEP check each failed.
        "skipped_attempts": skipped,
        "target_step": str(target_step),
        "last_only": last_only,
        **build,
        "scorer": asdict(scorer),
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
    parser.add_argument("--timeout-s", type=float, default=StepScorer.timeout_s)
    parser.add_argument(
        "--build-timeout-s",
        type=float,
        default=DEFAULT_BUILD_TIMEOUT_S,
        help="sandbox time to build a native run's model.py, as both pipelines allow",
    )
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
        default=StepScorer.sample_points,
        help="surface samples per side for squared CD and Hausdorff",
    )
    parser.add_argument(
        "--reference-extent",
        type=float,
        default=StepScorer.reference_extent,
    )
    parser.add_argument("--seed", type=int, default=StepScorer.seed)
    parser.add_argument("--f1-threshold", type=float, default=StepScorer.f1_threshold)
    parser.add_argument("--skip-eccv", action="store_true")
    parser.add_argument("--skip-ortho2cad", action="store_true")
    parser.add_argument("--auc-tr-min-cd", type=float, default=StepScorer.auc_tr_min_cd)
    parser.add_argument("--auc-tr-max-cd", type=float, default=StepScorer.auc_tr_max_cd)
    parser.add_argument(
        "--auc-tr-num-points", type=int, default=StepScorer.auc_tr_num_points
    )
    args = parser.parse_args()
    if not args.target_dir.is_dir():
        parser.error(f"target directory not found: {args.target_dir}")

    scorer = StepScorer(
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
    # Made on the first native run, so staged runs alone never need the sandbox.
    sandbox_runner: SandboxRunner | None = None
    for run_dir in args.run_dir:
        target_step = args.target_dir / f"{run_dir.name}.step"
        if not (run_dir / "events.jsonl").is_file():
            print(f"skip       {run_dir}: no events.jsonl")
            continue
        if not target_step.is_file():
            print(f"skip       {run_dir}: no target {target_step}")
            continue
        print(f"sample     {run_dir}")
        if sandbox_runner is None and is_native_run(run_dir):
            sandbox_runner = sandbox_for_builds(args.build_timeout_s)
        document = score_run(
            run_dir, target_step, scorer, args.last_only, sandbox_runner
        )
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
