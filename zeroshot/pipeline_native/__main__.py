"""Run the native baseline with the shared Hydra config directory."""

from pathlib import Path

import hydra
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig

from zeroshot.evaluation.run_scoring import format_score, score_run, write_score
from zeroshot.pipeline_native.runner import run


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


@hydra.main(version_base="1.3", config_path="../configs", config_name="native")
def main(config: DictConfig) -> None:
    """Run one sample and, when it has a target, score it.

    The target and scorer are checked before generation, so a bad one costs no
    model calls. The runner never touches either. As in `zeroshot.run_pipeline`,
    a run that raised is scored too, so a failure becomes a comparable result
    rather than a hole, and is then re-raised.
    """
    if config.sample.get("target_step_path") is None:
        run(config)
        return
    target = Path(to_absolute_path(str(config.sample.target_step_path)))
    if not target.is_file():
        raise FileNotFoundError(f"Target STEP not found: {target}")
    scorer = instantiate(config.evaluation.scorer)

    run_dir = Path(to_absolute_path(str(config.artifact_root))) / str(
        config.sample.sample_id
    )
    events_path = run_dir / "events.jsonl"
    earlier_run = _first_event(events_path)
    failure: Exception | None = None
    try:
        if run(config) is None:
            return
    except Exception as error:  # noqa: BLE001 - re-raised once the run is scored
        failure = error
    # Hydra makes the sample directory before the job runs, so only a new log
    # says this run started. One the runner refused leaves an earlier run's
    # log and score as they were.
    if failure is not None and _first_event(events_path) == earlier_run:
        raise failure
    try:
        document = score_run(run_dir, target, scorer, config.evaluation.last_only)
        path = write_score(run_dir, document)
        if config.console:
            print(format_score(document))
            print(f"written    {path}")
    finally:
        # A scoring error must not hide why the run itself failed.
        if failure is not None:
            raise failure


if __name__ == "__main__":
    main()
