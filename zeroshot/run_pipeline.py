import rootutils
from pathlib import Path

import hydra
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from zeroshot.pipeline.manifest import InputManifest
from zeroshot.pipeline.runner import PipelineRunner
from zeroshot.pipeline.sandbox import SandboxRunner
from zeroshot.pipeline.workflow import ReconstructionState


def run(config: DictConfig) -> ReconstructionState:
    """Run one sample with already constructed model dependency."""
    message_builder = instantiate(config.message_builder)

    model = instantiate(config.model)

    sandbox_runner = SandboxRunner(
        python_executable=Path(
            to_absolute_path(config.sandbox_runner.python_executable)
        ),
        default_timeout_s=config.sandbox_runner.default_timeout_s,
        max_stdout_bytes=config.sandbox_runner.max_stdout_bytes,
        max_stderr_bytes=config.sandbox_runner.max_stderr_bytes,
    )

    runner = PipelineRunner(
        model=model,
        message_builder=message_builder,
        sandbox_runner=sandbox_runner,
        artifact_root=Path(to_absolute_path(config.artifact_root)),
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


@hydra.main(version_base="1.3", config_path="configs", config_name="default")
def main(config: DictConfig) -> None:
    result = run(config)

    report = result["last_verification"]
    if report.status != "VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
