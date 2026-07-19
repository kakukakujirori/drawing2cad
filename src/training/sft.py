"""Transparent Accelerate SFT loop for multimodal Drawing2CAD models.

The checkpoint path is intentionally PEFT-aware: it stores the LoRA adapter,
the trainable primitive encoder (through ``modules_to_save``), and resumable
training state.  It never writes a duplicate copy of the frozen Qwen base
checkpoint.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
from transformers import AutoProcessor, get_scheduler
from torch.utils.data import DataLoader

from src.data import (
    DXFPrimitiveConfig,
    DXFPrimitiveParser,
    Drawing2CADBatch,
    Drawing2CADCollator,
    Drawing2CADDataset,
    Drawing2CADPreprocessor,
    RasterImageSource,
)
from src.evaluation import SFTGenerationEvaluator
from src.models import (
    Drawing2CADQwen3VLForConditionalGeneration,
    PrimitiveEncoderConfig,
)
from src.utils import (
    CheckpointManager,
    ExperimentLogger,
    seed_everything,
    seed_worker,
    setup_run,
)


LOGGER = logging.getLogger(__name__)
GenerationEvaluator = Callable[[torch.nn.Module, int], Mapping[str, float | int]]


@dataclass
class TrainingProgress:
    """Optimizer-boundary progress persisted in a lightweight checkpoint."""

    global_step: int = 0
    epoch: int = 0
    batches_seen_in_epoch: int = 0

    def __post_init__(self) -> None:
        for name in ("global_step", "epoch", "batches_seen_in_epoch"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value!r}")


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


def _torch_dtype(name: str | torch.dtype) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    aliases = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return aliases[str(name).lower()]
    except KeyError as error:
        raise ValueError(f"unsupported torch dtype: {name!r}") from error


def _vision_module(model: torch.nn.Module) -> torch.nn.Module:
    """Locate Qwen3-VL's native visual tower before or after PEFT wrapping."""

    candidates = (
        ("model", "visual"),
        ("base_model", "model", "model", "visual"),
        ("base_model", "model", "model", "model", "visual"),
    )
    for path in candidates:
        current: Any = model
        for component in path:
            current = getattr(current, component, None)
            if current is None:
                break
        if isinstance(current, torch.nn.Module):
            return current
    raise AttributeError("could not locate Qwen3-VL visual tower")


def freeze_vision_encoder(model: torch.nn.Module) -> torch.nn.Module:
    """Freeze the native vision tower and keep it in evaluation mode."""

    visual = _vision_module(model)
    visual.requires_grad_(False)
    visual.eval()
    return visual


def apply_language_lora(model: torch.nn.Module, config: Mapping[str, Any]) -> torch.nn.Module:
    """Attach language LoRA and preserve primitive weights in adapter saves."""

    lora = _mapping(config, name="model.lora")
    if not bool(lora.get("enabled", True)):
        raise ValueError(
            "baseline SFT requires model.lora.enabled=true; full fine-tuning is "
            "intentionally unsupported"
        )

    target_modules = tuple(str(value) for value in lora.get("target_modules", ()))
    if not target_modules:
        raise ValueError("model.lora.target_modules must not be empty")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora.get("rank", lora.get("r", 16))),
        lora_alpha=int(lora.get("alpha", 32)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        bias=str(lora.get("bias", "none")),
        target_modules=list(target_modules),
        modules_to_save=["primitive_encoder"],
    )
    output = get_peft_model(model, peft_config)
    # PEFT freezes the base model and re-enables the modules_to_save copy.  Make
    # this contract explicit so a future PEFT behavior change fails early.
    primitive_parameters = [
        parameter
        for name, parameter in output.named_parameters()
        if "primitive_encoder.modules_to_save" in name
    ]
    if not primitive_parameters or not all(
        parameter.requires_grad for parameter in primitive_parameters
    ):
        raise RuntimeError("PEFT did not retain a trainable primitive_encoder copy")
    return output


def _rng_state(generator: torch.Generator | None) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    state["numpy"] = np.random.get_state()
    if generator is not None:
        state["dataloader_generator"] = generator.get_state()
    return state


def _restore_rng_state(
    state: Mapping[str, Any], generator: torch.Generator | None
) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if generator is not None and "dataloader_generator" in state:
        generator.set_state(state["dataloader_generator"])


class AdapterCheckpointIO:
    """Save/load adapter-only model state plus optimizer, scheduler, and RNG."""

    def __init__(
        self,
        *,
        accelerator: Any,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        dataloader_generator: torch.Generator | None = None,
    ) -> None:
        self.accelerator = accelerator
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader_generator = dataloader_generator
        self.progress = TrainingProgress()

    def set_progress(self, progress: TrainingProgress) -> None:
        self.progress = progress

    def save(self, directory: Path) -> None:
        """CheckpointManager callback; called on the main rank only."""

        directory.mkdir(parents=True, exist_ok=True)
        model = self.accelerator.unwrap_model(self.model)
        if not hasattr(model, "peft_config") or not hasattr(model, "save_pretrained"):
            raise TypeError("lightweight SFT checkpoints require a PEFT model")
        adapter_dir = directory / "adapter"
        # PeftModel.save_pretrained filters to adapter tensors and modules_to_save;
        # unlike Accelerator.save_state, it does not serialize the frozen base.
        model.save_pretrained(adapter_dir, safe_serialization=True)
        torch.save(self.optimizer.state_dict(), directory / "optimizer.pt")
        torch.save(self.scheduler.state_dict(), directory / "scheduler.pt")
        torch.save(
            _rng_state(self.dataloader_generator), directory / "rng_state.pt"
        )
        (directory / "training_progress.json").write_text(
            json.dumps(asdict(self.progress), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def load(self, directory: Path) -> TrainingProgress:
        """Restore a checkpoint into an already prepared PEFT training stack."""

        required = (
            directory / "adapter",
            directory / "optimizer.pt",
            directory / "scheduler.pt",
            directory / "rng_state.pt",
            directory / "training_progress.json",
        )
        missing = [path for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"incomplete SFT checkpoint; missing: {missing}")

        model = self.accelerator.unwrap_model(self.model)
        adapter_state = load_peft_weights(
            str(directory / "adapter"), device=str(self.accelerator.device)
        )
        load_result = set_peft_model_state_dict(model, adapter_state)
        unexpected = tuple(getattr(load_result, "unexpected_keys", ()))
        if unexpected:
            raise RuntimeError(f"unexpected adapter checkpoint keys: {unexpected}")
        self.optimizer.load_state_dict(
            torch.load(
                directory / "optimizer.pt",
                map_location=self.accelerator.device,
                weights_only=False,
            )
        )
        self.scheduler.load_state_dict(
            torch.load(
                directory / "scheduler.pt", map_location="cpu", weights_only=False
            )
        )
        rng = torch.load(
            directory / "rng_state.pt", map_location="cpu", weights_only=False
        )
        _restore_rng_state(rng, self.dataloader_generator)
        raw_progress = json.loads(
            (directory / "training_progress.json").read_text(encoding="utf-8")
        )
        self.progress = TrainingProgress(**raw_progress)
        return self.progress


def _model_inputs(batch: Any, device: torch.device) -> dict[str, Any]:
    if isinstance(batch, Drawing2CADBatch):
        return batch.to(device).model_inputs
    if isinstance(batch, Mapping):
        output: dict[str, Any] = {}
        for key, value in batch.items():
            output[key] = value.to(device) if hasattr(value, "to") else value
        return output
    raise TypeError(f"training loader must emit Drawing2CADBatch or mapping, got {type(batch)}")


@torch.no_grad()
def evaluate_loss(accelerator: Any, model: torch.nn.Module, dataloader: DataLoader) -> dict[str, float]:
    """Compute distributed token-weighted validation cross entropy."""

    was_training = model.training
    model.eval()
    total_loss = torch.zeros((), device=accelerator.device, dtype=torch.float64)
    total_tokens = torch.zeros((), device=accelerator.device, dtype=torch.float64)
    for batch in dataloader:
        inputs = _model_inputs(batch, accelerator.device)
        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("validation batches must contain completion-only labels")
        token_count = (labels != -100).sum().to(torch.float64)
        outputs = model(**inputs, use_cache=False)
        if outputs.loss is None or not torch.isfinite(outputs.loss):
            raise FloatingPointError("validation produced a non-finite loss")
        total_loss += outputs.loss.detach().to(torch.float64) * token_count
        total_tokens += token_count
    totals = accelerator.reduce(
        torch.stack((total_loss, total_tokens)), reduction="sum"
    )
    if totals[1].item() <= 0:
        raise ValueError("validation loader contains no supervised tokens")
    if was_training:
        model.train()
        _vision_module(model).eval()
    return {"val/loss": float((totals[0] / totals[1]).item())}


def _optimizer(model: torch.nn.Module, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    cfg = _mapping(config, name="optimizer")
    if str(cfg.get("name", "adamw")).lower() != "adamw":
        raise ValueError("only AdamW is supported by the baseline SFT loop")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    fused_value = cfg.get("fused", "auto")
    fused = torch.cuda.is_available() if fused_value == "auto" else bool(fused_value)
    return torch.optim.AdamW(
        parameters,
        lr=float(cfg.get("learning_rate", 2.0e-4)),
        betas=tuple(float(value) for value in cfg.get("betas", (0.9, 0.95))),
        eps=float(cfg.get("eps", 1.0e-8)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        fused=fused,
    )


def _scheduler(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    *,
    total_steps: int,
) -> Any:
    cfg = _mapping(config, name="scheduler")
    warmup_steps = cfg.get("warmup_steps")
    if warmup_steps is None:
        warmup_steps = round(total_steps * float(cfg.get("warmup_ratio", 0.0)))
    kwargs: dict[str, Any] = {}
    if str(cfg.get("name", "cosine")) == "cosine":
        kwargs["num_cycles"] = float(cfg.get("num_cycles", 0.5))
    return get_scheduler(
        name=str(cfg.get("name", "cosine")),
        optimizer=optimizer,
        num_warmup_steps=int(warmup_steps),
        num_training_steps=total_steps,
        scheduler_specific_kwargs=kwargs,
    )


def _datasets_and_loaders(
    config: Mapping[str, Any],
    processor: Any,
    primitive_config: PrimitiveEncoderConfig,
    *,
    seed: int,
) -> tuple[DataLoader, DataLoader, torch.Generator]:
    data_cfg = _mapping(config, name="data")
    if bool(data_cfg.get("scale_augmentation", False)):
        raise ValueError("scale augmentation is not implemented for the SFT baseline")
    dxf_config = DXFPrimitiveConfig(**_mapping(data_cfg["dxf"], name="data.dxf"))
    if dxf_config.sample_feature_dim != primitive_config.sample_feature_dim:
        raise ValueError("DXF and primitive encoder sample feature dimensions differ")
    image_sources = tuple(
        RasterImageSource(str(item["style"]), str(item["directory"]))
        for item in data_cfg["image_sources"]
    )
    preprocessor = Drawing2CADPreprocessor(
        processor,
        primitive_config.num_primitive_latents,
        include_labels=True,
        max_length=data_cfg.get("max_sequence_length"),
    )

    def dataset(root_key: str, max_key: str) -> Drawing2CADDataset:
        return Drawing2CADDataset(
            data_cfg[root_key],
            dxf_parser=DXFPrimitiveParser(dxf_config),
            image_sources=image_sources,
            include_target=True,
            max_samples=data_cfg.get(max_key),
            strict_files=bool(data_cfg.get("strict_files", True)),
            image_max_edge=data_cfg.get("image_max_edge"),
            transform=preprocessor,
        )

    train_dataset = dataset("train_root", "train_max_samples")
    val_dataset = dataset("val_root", "val_max_samples")
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("processor tokenizer has no pad token ID")
    collator = Drawing2CADCollator(
        int(pad_token_id), padding_side=processor.tokenizer.padding_side
    )
    generator = torch.Generator().manual_seed(seed)
    common = {
        "num_workers": int(data_cfg.get("num_workers", 0)),
        "pin_memory": bool(data_cfg.get("pin_memory", True)),
        "persistent_workers": bool(data_cfg.get("persistent_workers", False)),
        "worker_init_fn": seed_worker,
        "collate_fn": collator,
    }
    if common["num_workers"] == 0:
        common["persistent_workers"] = False
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(data_cfg.get("train_batch_size", 1)),
        shuffle=bool(data_cfg.get("shuffle_train", True)),
        drop_last=bool(data_cfg.get("drop_last", False)),
        generator=generator,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(data_cfg.get("val_batch_size", 1)),
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader, generator


def _max_steps(training: Mapping[str, Any], loader_length: int) -> tuple[int, int]:
    accumulation = int(training.get("gradient_accumulation_steps", 1))
    if accumulation <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    steps_per_epoch = math.ceil(loader_length / accumulation)
    configured_steps = training.get("max_steps")
    configured_epochs = training.get("num_train_epochs")
    if configured_steps is None and configured_epochs is None:
        raise ValueError("set training.max_steps or training.num_train_epochs")
    if configured_steps is not None:
        max_steps = int(configured_steps)
        if max_steps <= 0:
            raise ValueError("training.max_steps must be positive")
        epochs = math.ceil(max_steps / steps_per_epoch)
    else:
        epochs = int(configured_epochs)
        if epochs <= 0:
            raise ValueError("training.num_train_epochs must be positive")
        max_steps = steps_per_epoch * epochs
    return max_steps, epochs


def _log_train_step(
    logger: ExperimentLogger,
    *,
    accelerator: Any,
    step: int,
    losses: Sequence[float],
    optimizer: torch.optim.Optimizer,
    grad_norm: float | torch.Tensor | None,
    tokens: int,
    elapsed: float,
    tokens_per_second: bool,
) -> None:
    local = torch.tensor(
        [sum(losses), len(losses)], device=accelerator.device, dtype=torch.float64
    )
    reduced = accelerator.reduce(local, reduction="sum")
    metrics: dict[str, float] = {
        "train/loss": float((reduced[0] / reduced[1]).item()),
        "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
    }
    if grad_norm is not None:
        metrics["train/grad_norm"] = float(
            grad_norm.detach().float().item()
            if isinstance(grad_norm, torch.Tensor)
            else grad_norm
        )
    if tokens_per_second and elapsed > 0:
        token_tensor = torch.tensor(float(tokens), device=accelerator.device)
        total_tokens = accelerator.reduce(token_tensor, reduction="sum")
        metrics["train/tokens_per_second"] = float(total_tokens.item() / elapsed)
    logger.log(metrics, step=step)


def run_sft(
    config: Any,
    *,
    generation_evaluator: GenerationEvaluator | None = None,
) -> dict[str, float | int]:
    """Construct and run the baseline SFT experiment."""
    cfg = _plain_config(config)
    training = _mapping(cfg["training"], name="training")
    mixed_precision = str(training.get("mixed_precision", "no")).lower()
    if mixed_precision in {"none", "null", "false"}:
        mixed_precision = "no"
    accelerator = Accelerator(
        gradient_accumulation_steps=int(
            training.get("gradient_accumulation_steps", 1)
        ),
        mixed_precision=mixed_precision,
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

    model_cfg = _mapping(cfg["model"], name="model")
    primitive_config = PrimitiveEncoderConfig(
        **_mapping(model_cfg["primitive"], name="model.primitive")
    )
    processor = AutoProcessor.from_pretrained(
        str(model_cfg["model_name_or_path"]), trust_remote_code=True
    )
    evaluation_cfg = _mapping(cfg.get("evaluation", {}), name="evaluation")
    train_loader, val_loader, loader_generator = _datasets_and_loaders(
        _mapping(cfg["data"], name="data"),
        processor,
        primitive_config,
        seed=seed,
    )
    dtype = _torch_dtype(model_cfg.get("torch_dtype", "bfloat16"))
    model = Drawing2CADQwen3VLForConditionalGeneration.from_qwen_pretrained(
        str(model_cfg["model_name_or_path"]),
        primitive_config=primitive_config,
        attn_implementation=str(model_cfg.get("attn_implementation", "auto")),
        dtype=dtype,
    )
    if not bool(model_cfg.get("freeze_vision_encoder", True)):
        raise ValueError("the baseline requires freeze_vision_encoder=true")
    freeze_vision_encoder(model)
    model = apply_language_lora(model, _mapping(model_cfg["lora"], name="model.lora"))
    freeze_vision_encoder(model)
    if bool(model_cfg.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    optimizer = _optimizer(model, _mapping(cfg["optimizer"], name="optimizer"))
    max_steps, num_epochs = _max_steps(training, len(train_loader))
    scheduler = _scheduler(
        optimizer, _mapping(cfg["scheduler"], name="scheduler"), total_steps=max_steps
    )
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    if generation_evaluator is None and bool(
        evaluation_cfg.get("generation_enabled", False)
    ):
        generation_evaluator = SFTGenerationEvaluator(
            accelerator=accelerator,
            processor=processor,
            data_config=_mapping(cfg["data"], name="data"),
            evaluation_config=evaluation_cfg,
            primitive_config=primitive_config,
            predictions_dir=run_context.predictions_dir,
        )
    checkpoint_io = AdapterCheckpointIO(
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader_generator=loader_generator,
    )
    manager = CheckpointManager.from_config(
        run_context.run_dir,
        _mapping(cfg["checkpoint"], name="checkpoint"),
        is_main_process=accelerator.is_main_process,
    )
    progress = TrainingProgress()
    if resume:
        progress = checkpoint_io.load(manager.latest_dir)
        accelerator.wait_for_everyone()

    logger = ExperimentLogger.from_config(
        run_context.run_dir,
        _mapping(cfg["logger"], name="logger"),
        resolved_config=cfg,
        is_main_process=accelerator.is_main_process,
    )
    eval_every = int(training.get("eval_every_steps", 100))
    save_every = int(training.get("save_every_steps", eval_every))
    log_every = int(training.get("log_every_steps", 1))
    if min(eval_every, save_every, log_every) <= 0:
        raise ValueError("eval/save/log cadences must be positive")
    max_grad_norm = float(training.get("max_grad_norm", 1.0))
    accumulated_losses: list[float] = []
    accumulated_tokens = 0
    interval_start = time.monotonic()
    last_metrics: dict[str, float | int] = {}
    last_evaluated_step: int | None = None
    last_saved_step: int | None = None

    try:
        model.train()
        # ``Module.train()`` recurses into the frozen tower.  Keep its dropout
        # and normalization behavior deterministic as well as its weights frozen.
        _vision_module(model).eval()
        optimizer.zero_grad(set_to_none=True)
        stop = progress.global_step >= max_steps
        for epoch in range(progress.epoch, num_epochs):
            if stop:
                break
            active_loader = train_loader
            skip = progress.batches_seen_in_epoch if epoch == progress.epoch else 0
            if skip:
                active_loader = accelerator.skip_first_batches(train_loader, skip)
            for relative_batch, batch in enumerate(active_loader):
                batch_in_epoch = skip + relative_batch
                inputs = _model_inputs(batch, accelerator.device)
                labels = inputs.get("labels")
                token_count = 0 if labels is None else int((labels != -100).sum().item())
                with accelerator.accumulate(model):
                    outputs = model(**inputs, use_cache=False)
                    loss = outputs.loss
                    if loss is None or not torch.isfinite(loss):
                        raise FloatingPointError("training produced a non-finite loss")
                    accelerator.backward(loss)
                    grad_norm = None
                    if accelerator.sync_gradients and max_grad_norm > 0:
                        grad_norm = accelerator.clip_grad_norm_(
                            model.parameters(), max_grad_norm
                        )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                accumulated_losses.append(float(loss.detach().float().item()))
                accumulated_tokens += token_count
                progress.batches_seen_in_epoch = batch_in_epoch + 1

                if not accelerator.sync_gradients:
                    continue
                progress.global_step += 1
                if progress.global_step % log_every == 0:
                    _log_train_step(
                        logger,
                        accelerator=accelerator,
                        step=progress.global_step,
                        losses=accumulated_losses,
                        optimizer=optimizer,
                        grad_norm=grad_norm,
                        tokens=accumulated_tokens,
                        elapsed=time.monotonic() - interval_start,
                        tokens_per_second=bool(training.get("tokens_per_second", True)),
                    )
                    accumulated_losses.clear()
                    accumulated_tokens = 0
                    interval_start = time.monotonic()

                should_eval = progress.global_step % eval_every == 0
                should_save = progress.global_step % save_every == 0
                if should_eval or should_save:
                    last_metrics = evaluate_loss(accelerator, model, val_loader)
                    if generation_evaluator is not None:
                        last_metrics.update(
                            generation_evaluator(model, progress.global_step)
                        )
                    logger.log(last_metrics, step=progress.global_step)
                    last_evaluated_step = progress.global_step
                    if should_save:
                        checkpoint_io.set_progress(progress)
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            manager.save(
                                step=progress.global_step,
                                metrics=last_metrics,
                                save_state=checkpoint_io.save,
                                extra_metadata={
                                    "epoch": epoch,
                                    "batches_seen_in_epoch": progress.batches_seen_in_epoch,
                                },
                            )
                        accelerator.wait_for_everyone()
                        last_saved_step = progress.global_step
                if progress.global_step >= max_steps:
                    stop = True
                    break
            if not stop:
                progress.epoch = epoch + 1
                progress.batches_seen_in_epoch = 0

        needs_final_save = last_saved_step != progress.global_step
        needs_final_evaluation = (
            last_evaluated_step != progress.global_step
            and (
                bool(training.get("final_evaluation", True))
                or needs_final_save
            )
        )
        if needs_final_evaluation:
            last_metrics = evaluate_loss(accelerator, model, val_loader)
            if generation_evaluator is not None:
                last_metrics.update(generation_evaluator(model, progress.global_step))
            logger.log(last_metrics, step=progress.global_step)
            last_evaluated_step = progress.global_step
        if needs_final_save:
            checkpoint_io.set_progress(progress)
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                manager.save(
                    step=progress.global_step,
                    metrics=last_metrics,
                    save_state=checkpoint_io.save,
                    extra_metadata={
                        "epoch": progress.epoch,
                        "batches_seen_in_epoch": progress.batches_seen_in_epoch,
                    },
                )
            accelerator.wait_for_everyone()
        return last_metrics
    finally:
        logger.finish()
