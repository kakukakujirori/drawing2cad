import json
from pathlib import Path

import hydra
import rootutils
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from zeroshot.evaluation.run_scoring import score_run
from zeroshot.pipeline.messages import InputManifest
from zeroshot.pipeline.runner import PipelineRunner
from zeroshot.pipeline.sandbox import SandboxRunner
from zeroshot.pipeline.verification import StepRenderer
from zeroshot.pipeline.workflow import ReconstructionState
from zeroshot.provenance import record_run


def run(config: DictConfig) -> ReconstructionState | None:
    """Run one sample with already constructed model dependency."""
    sandbox_runner = SandboxRunner(
        python_executable=Path(
            to_absolute_path(config.sandbox_runner.python_executable)
        ),
        default_timeout_s=config.sandbox_runner.default_timeout_s,
        max_stdout_bytes=config.sandbox_runner.max_stdout_bytes,
        max_stderr_bytes=config.sandbox_runner.max_stderr_bytes,
    )

    runner = PipelineRunner(
        model=instantiate(config.model),
        message_builder=instantiate(config.message_builder),
        sandbox_runner=sandbox_runner,
        artifact_root=Path(to_absolute_path(config.artifact_root)),
        renderer=instantiate(config.renderer) or StepRenderer(),
        graph_factory=instantiate(config.workflow),
        on_existing=config.on_existing,
        console_reporter=instantiate(config.console),
    )

    manifest = InputManifest(
        sample_id=config.sample.sample_id,
        dxf_path=Path(to_absolute_path(config.sample.dxf_path)),
        render3d_paths={
            style: Path(to_absolute_path(path))
            for style, path in config.sample.render3d_paths.items()
        },
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
    (run_dir / "score.json").write_text(json.dumps(document, indent=2), "utf-8")


@hydra.main(version_base="1.3", config_path="configs", config_name="default")
def main(config: DictConfig) -> None:
    """Run one sample and, when it has a target, score it.

    The exit code reports whether the pipeline ran, not whether the model
    succeeded. A run that reaches no valid solid is a result, recorded in
    `events.jsonl` and `score.json`; exiting non-zero for it would abort a
    Hydra sweep at the first such sample and lose every one after it.
    """
    if run(config) is None:
        return

    artifact_root = Path(to_absolute_path(config.artifact_root))
    record_run(artifact_root, config.sample.sample_id)

    if config.sample.target_step_path is not None:
        score(config)


if __name__ == "__main__":
    main()
