"""Hydra composition root for the explicit Accelerate SFT loop."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from accelerate import Accelerator
import hydra
from omegaconf import DictConfig, OmegaConf
import rootutils


ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.factory import build_sft_dataloaders
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
    seed_everything,
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


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value)}")
    return dict(value)


def _mixed_precision(config: Mapping[str, Any]) -> str:
    value = str(config.get("mixed_precision", "no")).lower()
    return "no" if value in {"none", "null", "false"} else value


def execute(config: Any) -> dict[str, float | int]:
    """Construct every concrete dependency, then hand them to ``run_sft``."""

    cfg = _plain_config(config)
    training = _mapping(cfg["training"], name="training")
    accelerator = Accelerator(
        gradient_accumulation_steps=int(
            training.get("gradient_accumulation_steps", 1)
        ),
        mixed_precision=_mixed_precision(training),
    )
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    resume = bool(training.get("resume_from_latest", False))
    paths = _mapping(cfg.get("paths", {}), name="paths")
    run_context = setup_run(
        cfg,
        output_dir=paths.get("output_dir"),
        project_root=paths.get("project_root"),
        is_main_process=accelerator.is_main_process,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        resume=resume,
    )
    accelerator.wait_for_everyone()

    model_bundle = build_sft_model(_mapping(cfg["model"], name="model"))
    data = build_sft_dataloaders(
        _mapping(cfg["data"], name="data"),
        processor=model_bundle.processor,
        primitive_config=model_bundle.primitive_config,
        seed=seed,
    )
    optimizer = build_optimizer(
        model_bundle.model,
        _mapping(cfg["optimizer"], name="optimizer"),
    )
    model, optimizer, train_loader, validation_loader = accelerator.prepare(
        model_bundle.model,
        optimizer,
        data.train,
        data.validation,
    )
    schedule = resolve_training_schedule(training, len(train_loader))
    scheduler = build_scheduler(
        optimizer,
        _mapping(cfg["scheduler"], name="scheduler"),
        total_steps=schedule.max_steps,
    )
    scheduler = accelerator.prepare(scheduler)

    checkpoint_io = AdapterCheckpointIO(
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader_generator=data.generator,
    )
    checkpoint_manager = CheckpointManager.from_config(
        run_context.run_dir,
        _mapping(cfg["checkpoint"], name="checkpoint"),
        is_main_process=accelerator.is_main_process,
    )
    progress = TrainingProgress()
    if resume:
        progress = checkpoint_io.load(checkpoint_manager.latest_dir)
        accelerator.wait_for_everyone()

    evaluation = _mapping(cfg.get("evaluation", {}), name="evaluation")
    generation_evaluator = None
    if bool(evaluation.get("generation_enabled", False)):
        generation_evaluator = CADGenerationEvaluator(
            accelerator=accelerator,
            processor=model_bundle.processor,
            data_config=_mapping(cfg["data"], name="data"),
            evaluation_config=evaluation,
            primitive_config=model_bundle.primitive_config,
            predictions_dir=run_context.predictions_dir,
        )

    logger = ExperimentLogger.from_config(
        run_context.run_dir,
        _mapping(cfg["logger"], name="logger"),
        resolved_config=cfg,
        is_main_process=accelerator.is_main_process,
    )
    utils_config = _mapping(cfg.get("utils", {}), name="utils")
    progress_bar = RichEpochProgressBar.from_config(
        _mapping(utils_config.get("progress_bar", {}), name="utils.progress_bar"),
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
            validation_dataloader=validation_loader,
            schedule=schedule,
            loop_config=SFTLoopConfig.from_mapping(training),
            metrics=metrics,
            progress_bar=progress_bar,
            checkpoint_io=checkpoint_io,
            checkpoint_manager=checkpoint_manager,
            set_train_mode=set_sft_train_mode,
            progress=progress,
            generation_evaluator=generation_evaluator,
        )
    finally:
        progress_bar.stop()
        metrics.finish()


@hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name="train_sft")
def main(config: DictConfig) -> None:
    execute(config)


if __name__ == "__main__":
    main()
