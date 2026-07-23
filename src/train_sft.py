"""Hydra composition root for the explicit Accelerate SFT loop."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import hydra
import rootutils
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from omegaconf import DictConfig, OmegaConf

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ruff: noqa: E402
from src.data.factory import build_sft_dataloaders
from src.data.layout import resolve_dataset_roots
from src.evaluation import CADGenerationEvaluator
from src.models.factory import build_sft_model, set_sft_train_mode
from src.training.checkpoint import AdapterCheckpointIO
from src.training.sft import (
    SFTLoopConfig,
    build_optimizer,
    build_scheduler,
    resolve_training_schedule,
    run_sft,
)
from src.training.state import TrainingProgress
from src.utils import (
    CheckpointManager,
    ExperimentLogger,
    MetricRouter,
    RichEpochProgressBar,
    setup_run,
)

CONFIG_DIR = str(ROOT / "configs")


def _plain_config(config: Any) -> dict[str, Any]:
    if OmegaConf.is_config(config):
        value = OmegaConf.to_container(config, resolve=True, enum_to_str=True)
    else:
        value = config
    if not isinstance(value, Mapping):
        raise TypeError("training config must resolve to a mapping")
    return dict(value)


def _mixed_precision(config: Mapping[str, Any]) -> str:
    value = str(config.get("mixed_precision", "no")).lower()
    return "no" if value in {"none", "null", "false"} else value


def execute(config: Any) -> dict[str, float | int]:
    """Construct every concrete dependency, then hand them to ``run_sft``."""

    cfg = _plain_config(config)
    training: Mapping[str, Any] = cfg["training"]
    # Gradient accumulation is owned by run_sft's explicit group loop so each
    # optimizer step can be token-weighted; the Accelerator must not divide
    # losses or gate optimizer/scheduler stepping itself.
    #
    # The primitive encoder's group-context branch is skipped for samples that
    # contain no grouped primitives, so those parameters receive no gradient on
    # steps whose batch happens to be ungrouped. Under DDP this is a
    # data-dependent unused-parameter pattern (it varies per step, so a static
    # graph would be unsafe), which requires find_unused_parameters. The flag
    # only affects the multi-process DDP wrapper and is inert on a single GPU.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=_mixed_precision(training),
        kwargs_handlers=[ddp_kwargs],
    )
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    resume = bool(training.get("resume_from_latest", False))
    paths: Mapping[str, Any] = cfg.get("paths", {})
    # Hydra resolves ${now:...} independently per process, so under a multi-GPU
    # `accelerate launch` each rank would otherwise pick its own timestamped run
    # directory. Broadcast rank zero's directory so every rank agrees without the
    # caller having to pin hydra.run.dir on the command line.
    output_dir = paths.get("output_dir")
    if accelerator.num_processes > 1:
        from accelerate.utils import broadcast_object_list

        payload = [output_dir]
        broadcast_object_list(payload, from_process=0)
        output_dir = payload[0]
    run_context = setup_run(
        cfg,
        output_dir=output_dir,
        project_root=paths.get("project_root"),
        is_main_process=accelerator.is_main_process,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        resume=resume,
    )
    accelerator.wait_for_everyone()

    model_bundle = build_sft_model(cfg["model"])
    dataloaders = build_sft_dataloaders(
        cfg["data"],
        processor=model_bundle.processor,
        primitive_config=model_bundle.primitive_config,
        seed=seed,
        show_progress=accelerator.is_main_process,
    )
    optimizer = build_optimizer(
        model_bundle.model,
        cfg["optimizer"],
    )
    model, optimizer, train_loader = accelerator.prepare(
        model_bundle.model,
        optimizer,
        dataloaders.train,
    )
    # One loss loader per validation dataset, prepared in the factory's (config)
    # order so every rank enters the eval collectives in the same sequence.
    validation_loaders = {
        name: accelerator.prepare(loader)
        for name, loader in dataloaders.validation.items()
    }
    schedule = resolve_training_schedule(training, len(train_loader))
    # The scheduler stays unprepared: run_sft steps it exactly once per
    # optimizer step, so total_steps is in true optimizer-step units and the
    # schedule is identical for any number of processes.
    scheduler = build_scheduler(
        optimizer,
        cfg["scheduler"],
        total_steps=schedule.max_steps,
    )

    checkpoint_io = AdapterCheckpointIO(
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader_generator=dataloaders.generator,
    )
    checkpoint_manager = CheckpointManager.from_config(
        run_context.run_dir,
        cfg["checkpoint"],
        is_main_process=accelerator.is_main_process,
    )
    progress = TrainingProgress()
    if resume:
        progress = checkpoint_io.load(checkpoint_manager.latest_dir)
        accelerator.wait_for_everyone()

    evaluation: Mapping[str, Any] = cfg.get("evaluation", {})
    generation_evaluators = []
    if bool(evaluation.get("generation_enabled", False)):
        # Every validation root is benchmarked, including STEP-only roots that
        # supply no loss above; predictions are kept under a per-root directory.
        generation_evaluators = [
            CADGenerationEvaluator(
                accelerator=accelerator,
                processor=model_bundle.processor,
                data_config=cfg["data"],
                dataset_root=root,
                evaluation_config=evaluation,
                primitive_config=model_bundle.primitive_config,
                predictions_dir=run_context.predictions_dir / root.name,
            )
            for root in resolve_dataset_roots(cfg["data"]["val_root"], split="val")
        ]

    logger = ExperimentLogger.from_config(
        run_context.run_dir,
        cfg["logger"],
        resolved_config=cfg,
        is_main_process=accelerator.is_main_process,
        resume=resume,
    )
    utils_config = cfg.get("utils", {})
    progress_bar = RichEpochProgressBar.from_config(
        utils_config.get("progress_bar", {}),
        is_main_process=accelerator.is_main_process,
    )
    metrics = MetricRouter(logger, progress_bar)
    try:
        return run_sft(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dataloader=train_loader,
            validation_dataloaders=validation_loaders,
            schedule=schedule,
            loop_config=SFTLoopConfig.from_mapping(training),
            metrics=metrics,
            progress_bar=progress_bar,
            checkpoint_io=checkpoint_io,
            checkpoint_manager=checkpoint_manager,
            set_train_mode=set_sft_train_mode,
            progress=progress,
            generation_evaluators=generation_evaluators,
            dataloader_generator=dataloaders.generator,
            dataloader_seed=seed,
        )
    finally:
        progress_bar.stop()
        metrics.finish()


@hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name="train_sft")
def main(config: DictConfig) -> None:
    execute(config)


if __name__ == "__main__":
    main()
