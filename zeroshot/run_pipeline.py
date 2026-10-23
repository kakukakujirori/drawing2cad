from pathlib import Path

import hydra
import rootutils
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from zeroshot.evaluation.run_scoring import score_run, write_score
from zeroshot.pipeline.messages.manifest import InputManifest, register_view
from zeroshot.pipeline.runner import PipelineRunner
from zeroshot.pipeline.sandbox import SandboxRunner
from zeroshot.pipeline.workflow import ReconstructionState
from zeroshot.provenance import record_run


def run(config: DictConfig) -> ReconstructionState | None:
    manifest = InputManifest(
        sample_id=config.sample.sample_id,
        drawing=[
            register_view(
                name=sheet.name,
                role=sheet.role,
                file=to_absolute_path(sheet.file),
            )
            for sheet in config.sample.drawing.sheets
        ],
        projection_method=config.sample.drawing.projection_method,
        hidden_lines_omitted=config.sample.drawing.hidden_lines_omitted,
        max_input_image_side=config.sample.max_input_image_side,
    )

    sandbox_runner = SandboxRunner(
        python_executable=Path(
            to_absolute_path(config.sandbox_runner.python_executable)
        ),
        default_timeout_s=config.sandbox_runner.default_timeout_s,
        max_stdout_bytes=config.sandbox_runner.max_stdout_bytes,
        max_stderr_bytes=config.sandbox_runner.max_stderr_bytes,
    )

    resume_from = config.get("resume_from")
    runner = PipelineRunner(
        sandbox_runner=sandbox_runner,
        graph_factory=instantiate(config.workflow),
        artifact_presenter=instantiate(config.artifact_presenter),
        artifact_root=Path(to_absolute_path(config.artifact_root)),
        on_existing=config.on_existing,
        console_reporter=instantiate(config.console),
        resume_from=(
            Path(to_absolute_path(str(resume_from)))
            if resume_from is not None
            else None
        ),
    )

    return runner.run_sample(manifest)


def score(config: DictConfig) -> None:
    """Score the finished run against the ground truth.

    Runs only after `run` has returned and the sample's artifacts are closed,
    and is the one place in the pipeline entrypoint that touches the target.
    """
    run_dir = Path(to_absolute_path(config.artifact_root)) / config.sample.sample_id
    document = score_run(
        run_dir=run_dir,
        target_step=Path(to_absolute_path(config.sample.target_step_path)),
        scorer=instantiate(config.evaluation.scorer),
        last_only=config.evaluation.last_only,
    )
    write_score(run_dir, document)


def _first_event(events_path: Path) -> str | None:
    """The log's first line, which names the run that wrote it, or None.

    The runner opens a new log for every run it starts and writes its
    `run_id` first, so this changes exactly when a run started.
    """
    try:
        with events_path.open(encoding="utf-8") as handle:
            return handle.readline()
    except FileNotFoundError:
        return None


@hydra.main(version_base="1.3", config_path="configs", config_name="default")
def main(config: DictConfig) -> None:
    """Run one sample and, when it has a target, score it.

    The exit code reports whether the pipeline ran, not whether the model
    succeeded. A run that reaches no valid solid is a result, recorded in
    `events.jsonl` and `score.json`, not an error. A run that raised is
    re-raised once scored; Hydra's basic launcher still runs the rest of a
    sweep and exits non-zero after the last job.
    """
    artifact_root = Path(to_absolute_path(config.artifact_root))
    events_path = artifact_root / config.sample.sample_id / "events.jsonl"
    earlier_run = _first_event(events_path)
    failure: Exception | None = None
    try:
        if run(config) is None:
            return
    except Exception as error:  # noqa: BLE001 - re-raised once the run is described
        # A run that raised still produced everything up to the point it did,
        # and that is what the artifacts are for.  Scoring it is how a failure
        # becomes a comparable result rather than a hole.
        failure = error

    # Hydra makes the sample directory before the job runs, so only a new log
    # says this run started. One the runner refused leaves an earlier run's
    # log, record and score as they were.
    if failure is None or _first_event(events_path) != earlier_run:
        record_run(artifact_root, config.sample.sample_id)
        if config.sample.target_step_path is not None:
            score(config)

    if failure is not None:
        raise failure


if __name__ == "__main__":
    main()
