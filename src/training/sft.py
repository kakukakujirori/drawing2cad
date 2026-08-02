"""Transparent Accelerate SFT loop for pre-built training components.

This module owns supervised forward/backward behavior and its validation,
logging, evaluation, and checkpoint cadence.  The Hydra entrypoint constructs
all concrete models, data loaders, and services and injects them into
``run_sft``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Protocol

import torch
from transformers import get_scheduler

from src.utils.metric_router import LoggedMetric, MetricRouter
from src.utils.progress import RichEpochProgressBar

from .state import TrainingProgress


class GenerationEvaluator(Protocol):
    def __call__(
        self,
        model: torch.nn.Module,
        step: int,
        *,
        progress_bar: "RichEpochProgressBar | None" = None,
    ) -> Mapping[str, float | int]: ...


TrainModeSetter = Callable[[torch.nn.Module], None]


class CheckpointStateIO(Protocol):
    def set_progress(self, progress: TrainingProgress) -> None: ...

    def save(self, directory: Path) -> None: ...


class CheckpointController(Protocol):
    monitor: str

    def save(
        self,
        *,
        step: int,
        metrics: Mapping[str, Any],
        save_state: Callable[[Path], None],
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> Any: ...


@dataclass(frozen=True)
class TrainingSchedule:
    max_steps: int
    num_epochs: int
    steps_per_epoch: int


@dataclass(frozen=True)
class SFTLoopConfig:
    """Validated settings that affect the SFT loop itself."""

    gradient_accumulation_steps: int = 1
    eval_every_steps: int = 100
    save_every_steps: int = 100
    log_every_steps: int = 1
    max_grad_norm: float = 1.0
    tokens_per_second: bool = True
    final_evaluation: bool = True

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "SFTLoopConfig":
        eval_every = int(config.get("eval_every_steps", 100))
        value = cls(
            gradient_accumulation_steps=int(
                config.get("gradient_accumulation_steps", 1)
            ),
            eval_every_steps=eval_every,
            save_every_steps=int(config.get("save_every_steps", eval_every)),
            log_every_steps=int(config.get("log_every_steps", 1)),
            max_grad_norm=float(config.get("max_grad_norm", 1.0)),
            tokens_per_second=bool(config.get("tokens_per_second", True)),
            final_evaluation=bool(config.get("final_evaluation", True)),
        )
        if value.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if (
            min(
                value.eval_every_steps,
                value.save_every_steps,
                value.log_every_steps,
            )
            <= 0
        ):
            raise ValueError("eval/save/log cadences must be positive")
        if value.max_grad_norm < 0:
            raise ValueError("max_grad_norm must be non-negative")
        return value


def build_optimizer(
    model: torch.nn.Module, config: Mapping[str, Any]
) -> torch.optim.Optimizer:
    """Build the baseline optimizer from the Hydra optimizer group.

    ``optimizer.param_groups`` maps a parameter-name substring to per-group
    overrides (``learning_rate``, ``weight_decay``). It exists because the
    trainable parameters are not one population: the LoRA adapters perturb a
    converged 2B backbone and want a small rate, while the primitive encoder is
    randomly initialized and needs roughly an order of magnitude more to move at
    all. Each parameter joins the first group whose substring it matches;
    everything left over stays in the default group, which is kept at index 0 so
    ``train/learning_rate`` keeps meaning the base rate.
    """

    if str(config.get("name", "adamw")).lower() != "adamw":
        raise ValueError("only AdamW is supported by the baseline SFT loop")
    default_lr = float(config.get("learning_rate", 2.0e-4))
    default_weight_decay = float(config.get("weight_decay", 0.01))
    overrides = config.get("param_groups") or {}
    if not isinstance(overrides, Mapping):
        raise TypeError(
            "optimizer.param_groups must map a parameter-name substring to its "
            f"overrides, got {type(overrides)}"
        )

    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise ValueError("model has no trainable parameters")

    claimed: set[int] = set()
    groups: list[dict[str, Any]] = []
    for group_name, settings in overrides.items():
        if not isinstance(settings, Mapping):
            raise TypeError(
                f"optimizer.param_groups.{group_name} must be a mapping, got "
                f"{type(settings)}"
            )
        members = [
            parameter
            for name, parameter in trainable
            if str(group_name) in name and id(parameter) not in claimed
        ]
        if not members:
            raise ValueError(
                f"optimizer.param_groups.{group_name} matches no trainable "
                "parameter name"
            )
        claimed.update(id(parameter) for parameter in members)
        group: dict[str, Any] = {
            "name": str(group_name),
            "params": members,
            "lr": float(settings.get("learning_rate", default_lr)),
            "weight_decay": float(settings.get("weight_decay", default_weight_decay)),
        }
        # Optional per-group clip, read back by clip_gradients; absent means the
        # loop's training.max_grad_norm applies to this group too.
        if settings.get("max_grad_norm") is not None:
            group["max_grad_norm"] = float(settings["max_grad_norm"])
        groups.append(group)

    remaining = [
        parameter for _, parameter in trainable if id(parameter) not in claimed
    ]
    if not remaining:
        raise ValueError(
            "optimizer.param_groups claims every trainable parameter; the default "
            "group must stay non-empty"
        )
    groups.insert(
        0,
        {"params": remaining, "lr": default_lr, "weight_decay": default_weight_decay},
    )

    fused_value = config.get("fused", "auto")
    fused = torch.cuda.is_available() if fused_value == "auto" else bool(fused_value)
    return torch.optim.AdamW(
        groups,
        lr=default_lr,
        betas=tuple(float(value) for value in config.get("betas", (0.9, 0.95))),
        eps=float(config.get("eps", 1.0e-8)),
        weight_decay=default_weight_decay,
        fused=fused,
    )


def resolve_training_schedule(
    training_config: Mapping[str, Any], loader_length: int
) -> TrainingSchedule:
    """Resolve optimizer steps after Accelerate has sharded the loader."""

    if loader_length <= 0:
        raise ValueError("training dataloader must contain at least one batch")
    accumulation = int(training_config.get("gradient_accumulation_steps", 1))
    if accumulation <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    steps_per_epoch = math.ceil(loader_length / accumulation)
    configured_steps = training_config.get("max_steps")
    configured_epochs = training_config.get("num_train_epochs")
    if configured_steps is None and configured_epochs is None:
        raise ValueError("set training.max_steps or training.num_train_epochs")
    if configured_steps is not None:
        max_steps = int(configured_steps)
        if max_steps <= 0:
            raise ValueError("training.max_steps must be positive")
        num_epochs = math.ceil(max_steps / steps_per_epoch)
    else:
        num_epochs = int(configured_epochs)
        if num_epochs <= 0:
            raise ValueError("training.num_train_epochs must be positive")
        max_steps = steps_per_epoch * num_epochs
    return TrainingSchedule(
        max_steps=max_steps,
        num_epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    *,
    total_steps: int,
) -> Any:
    """Build a Transformers scheduler for a resolved optimizer-step budget."""

    warmup_steps = config.get("warmup_steps")
    if warmup_steps is None:
        warmup_steps = round(total_steps * float(config.get("warmup_ratio", 0.0)))
    kwargs: dict[str, Any] = {}
    if str(config.get("name", "cosine")) == "cosine":
        kwargs["num_cycles"] = float(config.get("num_cycles", 0.5))
    return get_scheduler(
        name=str(config.get("name", "cosine")),
        optimizer=optimizer,
        num_warmup_steps=int(warmup_steps),
        num_training_steps=total_steps,
        scheduler_specific_kwargs=kwargs,
    )


def _model_inputs(batch: Any, device: torch.device) -> dict[str, Any]:
    if isinstance(batch, Mapping):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }
    if hasattr(batch, "to"):
        moved = batch.to(device)
        inputs = getattr(moved, "model_inputs", None)
        if isinstance(inputs, Mapping):
            return dict(inputs)
    raise TypeError(
        "SFT dataloaders must emit a mapping or a batch exposing to().model_inputs"
    )


def _accumulation_groups(loader: Any, group_size: int) -> Iterator[list[Any]]:
    """Yield micro-batches in optimizer-step groups.

    Token-weighted loss normalization needs every supervised-token count of a
    group before the first backward, so the group is materialized up front.
    """

    iterator = iter(loader)
    while True:
        group = list(islice(iterator, group_size))
        if not group:
            return
        yield group


def clip_gradients(
    accelerator: Any,
    optimizer: torch.optim.Optimizer,
    max_grad_norm: float,
) -> dict[str, Any]:
    """Clip every optimizer param group on its own gradient norm.

    One global clip couples populations whose gradient scales have no reason to
    match: a spike in the randomly initialized primitive encoder would scale the
    LoRA gradients down in the same step, so the adapters would silently take a
    shorter step because another module misbehaved. Now that the groups carry
    their own learning rates they get their own clip budget too. A group may
    override the shared limit through ``optimizer.param_groups.<name>``.

    Returns the pre-clip norms keyed for logging, mirroring the learning-rate
    keys: the unnamed default group is ``train/grad_norm`` and every override is
    ``train/grad_norm/<name>``.
    """

    norms: dict[str, Any] = {}
    if max_grad_norm <= 0:
        return norms
    for group in optimizer.param_groups:
        limit = float(group.get("max_grad_norm", max_grad_norm))
        if limit <= 0:
            continue
        norm = accelerator.clip_grad_norm_(group["params"], limit)
        group_name = group.get("name")
        key = (
            "train/grad_norm" if group_name is None else f"train/grad_norm/{group_name}"
        )
        norms[key] = norm
    return norms


def _supervised_loss(model: torch.nn.Module, inputs: Mapping[str, Any]) -> torch.Tensor:
    outputs = model(**inputs, use_cache=False)
    if outputs.loss is None or not torch.isfinite(outputs.loss):
        raise FloatingPointError("validation produced a non-finite loss")
    return outputs.loss.detach().to(torch.float64)


def _mismatched_primitives(current: Any, donor: Any) -> Any | None:
    """Return primitives belonging to a different drawing, or ``None``.

    Any batch of the same size works as a donor because the encoder always emits
    exactly ``num_primitive_latents`` latents per sample, so the placeholder
    positions the collator reserved still line up. The previous batch is
    preferred over an in-batch rotation because the configured validation batch
    size is 1, where rotating is a no-op.
    """

    if current is None:
        return None
    if donor is not None and donor.batch_size == current.batch_size:
        return donor
    if current.batch_size > 1:
        return current.roll(1)
    return None


@torch.no_grad()
def evaluate_loss(
    accelerator: Any,
    model: torch.nn.Module,
    dataloader: Any,
    *,
    set_train_mode: TrainModeSetter,
    on_batch: Callable[[int], None] | None = None,
    prefix: str = "val",
    primitive_ablation: bool = False,
) -> dict[str, float]:
    """Compute distributed token-weighted SFT validation cross entropy.

    ``prefix`` namespaces the returned key (``<prefix>/loss``) so several
    validation datasets can be evaluated in one pass without colliding.

    ``primitive_ablation`` additionally scores every batch against another
    drawing's primitives and reports ``<prefix>/loss_shuffled_primitives`` plus
    ``<prefix>/primitive_gain``, the amount by which the real primitives beat the
    mismatched ones. A gain near zero means the model is answering from the
    raster views alone and the primitive path is contributing nothing, which is
    otherwise invisible in a loss curve that keeps improving. It costs a second
    forward pass over the validation set.
    """

    was_training = model.training
    model.eval()

    def zero() -> torch.Tensor:
        return torch.zeros((), device=accelerator.device, dtype=torch.float64)

    total_loss, total_tokens = zero(), zero()
    ablated_loss, ablated_tokens = zero(), zero()
    # The first batch has no donor yet, so it is held back and scored against the
    # last batch's primitives; both passes then cover exactly the same tokens.
    deferred: tuple[dict[str, Any], torch.Tensor] | None = None
    previous_primitives: Any = None

    for batch in dataloader:
        inputs = _model_inputs(batch, accelerator.device)
        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("validation batches must contain completion-only labels")
        token_count = (labels != -100).sum().to(torch.float64)
        total_loss += _supervised_loss(model, inputs) * token_count
        total_tokens += token_count
        if primitive_ablation:
            primitives = inputs.get("primitive_batch")
            donor = _mismatched_primitives(primitives, previous_primitives)
            if donor is None:
                if deferred is None and primitives is not None:
                    deferred = (inputs, token_count)
            else:
                ablated_loss += (
                    _supervised_loss(model, {**inputs, "primitive_batch": donor})
                    * token_count
                )
                ablated_tokens += token_count
            if primitives is not None:
                previous_primitives = primitives
        if on_batch is not None:
            on_batch(1)

    if deferred is not None:
        inputs, token_count = deferred
        primitives = inputs.get("primitive_batch")
        donor = _mismatched_primitives(primitives, previous_primitives)
        # A pass holding a single batch of one has no other drawing to borrow
        # from, and scoring it against its own primitives would report a gain of
        # exactly zero as though the primitive path were useless.
        if donor is not None and donor is not primitives:
            ablated_loss += (
                _supervised_loss(model, {**inputs, "primitive_batch": donor})
                * token_count
            )
            ablated_tokens += token_count

    totals = accelerator.reduce(
        torch.stack((total_loss, total_tokens, ablated_loss, ablated_tokens)),
        reduction="sum",
    )
    if totals[1].item() <= 0:
        raise ValueError("validation loader contains no supervised tokens")
    if was_training:
        set_train_mode(model)

    loss = float((totals[0] / totals[1]).item())
    values = {f"{prefix}/loss": loss}
    # Only report the comparison when the ablated pass covered exactly the same
    # supervised tokens, so the difference is a paired measurement.
    if primitive_ablation and totals[3].item() == totals[1].item():
        ablated = float((totals[2] / totals[3]).item())
        values[f"{prefix}/loss_shuffled_primitives"] = ablated
        values[f"{prefix}/primitive_gain"] = ablated - loss
    return values


def _log_train_step(
    metrics: MetricRouter,
    *,
    accelerator: Any,
    step: int,
    loss_sum: float,
    optimizer: torch.optim.Optimizer,
    grad_norms: Mapping[str, float | torch.Tensor] | None,
    tokens: int,
    elapsed: float,
    include_tokens_per_second: bool,
) -> None:
    local = torch.tensor(
        [loss_sum, float(tokens)], device=accelerator.device, dtype=torch.float64
    )
    reduced = accelerator.reduce(local, reduction="sum")
    if reduced[1].item() <= 0:
        raise ValueError("training log interval contains no supervised tokens")
    values: dict[str, Any] = {
        "train/loss": LoggedMetric(
            float((reduced[0] / reduced[1]).item()),
            prog_bar=True,
            display_name="loss",
            format_spec=".3f",
        ),
    }
    # The unnamed default group is index 0 (see build_optimizer), so the base
    # rate keeps its historical key and every override gets a suffixed one.
    for group in optimizer.param_groups:
        group_name = group.get("name")
        if group_name is None:
            values["train/learning_rate"] = LoggedMetric(
                float(group["lr"]),
                prog_bar=True,
                display_name="lr",
                format_spec=".2e",
            )
        else:
            values[f"train/learning_rate/{group_name}"] = float(group["lr"])
    for key, grad_norm in (grad_norms or {}).items():
        values[key] = float(
            grad_norm.detach().float().item()
            if isinstance(grad_norm, torch.Tensor)
            else grad_norm
        )
    if include_tokens_per_second and elapsed > 0:
        values["train/tokens_per_second"] = LoggedMetric(
            float(reduced[1].item() / elapsed),
            prog_bar=True,
            display_name="tok/s",
            format_spec=".1f",
        )
    metrics.log(values, step=step)


PREFERRED_EVAL_METRICS = (
    "loss",
    "valid_rate",
    "mean_iou_including_failures",
    "median_iou",
    "mean_iou_valid_only",
)


def _split_metric_key(key: str) -> tuple[str, str]:
    """Split ``val/<dataset>/<metric>`` into its dataset name and metric name.

    Keys are namespaced per validation dataset, so the dataset is the middle
    segment. A key without one (``val/loss``) reports an empty dataset name.
    """
    head, _, metric = key.rpartition("/")
    return head.removeprefix("val").strip("/"), metric


def _print_evaluation(
    progress_bar: RichEpochProgressBar,
    values: Mapping[str, float | int],
    *,
    step: int,
) -> None:
    """Emit one compact, persistent validation line per dataset."""

    grouped: dict[str, dict[str, float | int]] = {}
    for key, value in values.items():
        dataset, metric = _split_metric_key(key)
        grouped.setdefault(dataset, {})[metric] = value
    for dataset, metrics in grouped.items():
        parts = [
            f"{metric}={float(metrics[metric]):.4f}"
            for metric in PREFERRED_EVAL_METRICS
            if metric in metrics
        ]
        # Fall back to every scalar when none of the preferred metrics is
        # present, so an unexpected metric set is still visible.
        if not parts:
            parts = [f"{key}={value}" for key, value in sorted(metrics.items())]
        label = f" {dataset}" if dataset else ""
        progress_bar.log_line(f"[eval @ step {step}]{label} " + " ".join(parts))


def _log_evaluation(
    router: MetricRouter,
    values: Mapping[str, float | int],
    *,
    step: int,
) -> None:
    events: dict[str, float | int | LoggedMetric] = dict(values)
    for key, value in values.items():
        dataset, metric = _split_metric_key(key)
        if metric != "loss":
            continue
        events[key] = LoggedMetric(
            value,
            prog_bar=True,
            display_name=f"{dataset}_loss" if dataset else "val_loss",
            format_spec=".3f",
        )
    router.log(events, step=step)


def _evaluate(
    *,
    accelerator: Any,
    model: torch.nn.Module,
    validation_dataloaders: Mapping[str, Any],
    set_train_mode: TrainModeSetter,
    generation_evaluators: Sequence[GenerationEvaluator],
    progress_bar: RichEpochProgressBar,
    step: int,
    primitive_ablation: bool = False,
) -> dict[str, float | int]:
    """Score every validation dataset, namespacing metrics by dataset name.

    Both collections are iterated in their construction order, which is
    identical on every rank; the collectives inside ``evaluate_loss`` and the
    generation evaluators require all ranks to reach them in the same sequence.
    """
    values: dict[str, float | int] = {}
    for name, dataloader in validation_dataloaders.items():
        try:
            loss_total = len(dataloader)
        except TypeError:
            loss_total = 0
        with progress_bar.sub_task(f"eval loss [{name}]", loss_total) as advance:
            values.update(
                evaluate_loss(
                    accelerator,
                    model,
                    dataloader,
                    set_train_mode=set_train_mode,
                    on_batch=advance,
                    prefix=f"val/{name}",
                    primitive_ablation=primitive_ablation,
                )
            )
    for generation_evaluator in generation_evaluators:
        values.update(generation_evaluator(model, step, progress_bar=progress_bar))
        set_train_mode(model)
    return values


def run_sft(
    *,
    accelerator: Any,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    train_dataloader: Any,
    validation_dataloaders: Mapping[str, Any],
    schedule: TrainingSchedule,
    loop_config: SFTLoopConfig,
    metrics: MetricRouter,
    progress_bar: RichEpochProgressBar,
    checkpoint_io: CheckpointStateIO,
    checkpoint_manager: CheckpointController,
    set_train_mode: TrainModeSetter,
    progress: TrainingProgress | None = None,
    generation_evaluators: Sequence[GenerationEvaluator] = (),
    dataloader_generator: torch.Generator | None = None,
    dataloader_seed: int | None = None,
    primitive_ablation: bool = False,
) -> dict[str, float | int]:
    """Run SFT using components constructed by the application entrypoint.

    Gradient accumulation is managed explicitly here rather than through
    ``accelerator.accumulate`` so that each optimizer step minimizes the mean
    cross entropy over every supervised token in its accumulation group across
    all processes.  ``scheduler`` must be the raw (unprepared) scheduler; it is
    advanced exactly once per optimizer step.
    """

    state = progress if progress is not None else TrainingProgress()
    interval_loss_sum = 0.0
    interval_tokens = 0
    interval_start = time.monotonic()
    last_metrics: dict[str, float | int] = {}
    last_evaluated_step: int | None = None
    last_saved_step: int | None = None

    set_train_mode(model)
    optimizer.zero_grad(set_to_none=True)
    stop = state.global_step >= schedule.max_steps
    for epoch in range(state.epoch, schedule.num_epochs):
        if stop:
            break
        if dataloader_generator is not None and dataloader_seed is not None:
            # Draw each epoch's shuffle permutation from (seed, epoch) alone so
            # a mid-epoch resume skips into the same permutation instead of one
            # freshly drawn from the restored generator state.
            dataloader_generator.manual_seed(dataloader_seed + epoch)
        active_loader = train_dataloader
        skip = state.batches_seen_in_epoch if epoch == state.epoch else 0
        if skip:
            active_loader = accelerator.skip_first_batches(train_dataloader, skip)
        completed_epoch_steps = min(
            math.ceil(skip / loop_config.gradient_accumulation_steps),
            schedule.steps_per_epoch,
        )
        progress_bar.start_epoch(
            epoch + 1,
            total_steps=schedule.steps_per_epoch,
            completed_steps=completed_epoch_steps,
        )
        batches_consumed = skip
        for group in _accumulation_groups(
            active_loader, loop_config.gradient_accumulation_steps
        ):
            group_inputs = [_model_inputs(batch, accelerator.device) for batch in group]
            token_counts: list[int] = []
            for inputs in group_inputs:
                labels = inputs.get("labels")
                if labels is None:
                    raise ValueError(
                        "training batches must contain completion-only labels"
                    )
                token_counts.append(int((labels != -100).sum().item()))
            group_tokens = float(
                accelerator.reduce(
                    torch.tensor(
                        float(sum(token_counts)),
                        device=accelerator.device,
                        dtype=torch.float64,
                    ),
                    reduction="sum",
                ).item()
            )
            if group_tokens <= 0:
                raise ValueError("accumulation group contains no supervised tokens")
            for index, (inputs, token_count) in enumerate(
                zip(group_inputs, token_counts, strict=True)
            ):
                # DDP averages gradients across processes on the synchronized
                # backward, hence the num_processes factor in the weight.
                weight = token_count * accelerator.num_processes / group_tokens
                sync_context = (
                    nullcontext()
                    if index == len(group_inputs) - 1
                    else accelerator.no_sync(model)
                )
                with sync_context:
                    outputs = model(**inputs, use_cache=False)
                    loss = outputs.loss
                    if loss is None or not torch.isfinite(loss):
                        raise FloatingPointError("training produced a non-finite loss")
                    accelerator.backward(loss * weight)
                interval_loss_sum += float(loss.detach().float().item()) * token_count
                interval_tokens += token_count
                batches_consumed += 1
                state.batches_seen_in_epoch = batches_consumed
            grad_norms = clip_gradients(
                accelerator, optimizer, loop_config.max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            state.global_step += 1
            progress_bar.advance()
            if state.global_step % loop_config.log_every_steps == 0:
                _log_train_step(
                    metrics,
                    accelerator=accelerator,
                    step=state.global_step,
                    loss_sum=interval_loss_sum,
                    optimizer=optimizer,
                    grad_norms=grad_norms,
                    tokens=interval_tokens,
                    elapsed=time.monotonic() - interval_start,
                    include_tokens_per_second=loop_config.tokens_per_second,
                )
                interval_loss_sum = 0.0
                interval_tokens = 0
                interval_start = time.monotonic()

            should_eval = state.global_step % loop_config.eval_every_steps == 0
            should_save = state.global_step % loop_config.save_every_steps == 0
            if should_eval or should_save:
                last_metrics = _evaluate(
                    accelerator=accelerator,
                    model=model,
                    validation_dataloaders=validation_dataloaders,
                    set_train_mode=set_train_mode,
                    generation_evaluators=generation_evaluators,
                    progress_bar=progress_bar,
                    step=state.global_step,
                    primitive_ablation=primitive_ablation,
                )
                _log_evaluation(metrics, last_metrics, step=state.global_step)
                if accelerator.is_main_process:
                    _print_evaluation(
                        progress_bar, last_metrics, step=state.global_step
                    )
                last_evaluated_step = state.global_step
                if should_save:
                    checkpoint_io.set_progress(state)
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        checkpoint_manager.save(
                            step=state.global_step,
                            metrics=last_metrics,
                            save_state=checkpoint_io.save,
                            extra_metadata={
                                "epoch": epoch,
                                "batches_seen_in_epoch": state.batches_seen_in_epoch,
                            },
                        )
                    accelerator.wait_for_everyone()
                    last_saved_step = state.global_step
            if state.global_step >= schedule.max_steps:
                stop = True
                break
        if not stop:
            progress_bar.finish_epoch()
            state.epoch = epoch + 1
            state.batches_seen_in_epoch = 0

    needs_final_save = last_saved_step != state.global_step
    needs_final_evaluation = last_evaluated_step != state.global_step and (
        loop_config.final_evaluation or needs_final_save
    )
    if needs_final_evaluation:
        last_metrics = _evaluate(
            accelerator=accelerator,
            model=model,
            validation_dataloaders=validation_dataloaders,
            set_train_mode=set_train_mode,
            generation_evaluators=generation_evaluators,
            progress_bar=progress_bar,
            step=state.global_step,
            primitive_ablation=primitive_ablation,
        )
        _log_evaluation(metrics, last_metrics, step=state.global_step)
        if accelerator.is_main_process:
            _print_evaluation(progress_bar, last_metrics, step=state.global_step)
    if needs_final_save:
        checkpoint_io.set_progress(state)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            checkpoint_manager.save(
                step=state.global_step,
                metrics=last_metrics,
                save_state=checkpoint_io.save,
                extra_metadata={
                    "epoch": state.epoch,
                    "batches_seen_in_epoch": state.batches_seen_in_epoch,
                },
            )
        accelerator.wait_for_everyone()
    return last_metrics


__all__ = [
    "SFTLoopConfig",
    "TrainingSchedule",
    "build_optimizer",
    "clip_gradients",
    "build_scheduler",
    "evaluate_loss",
    "resolve_training_schedule",
    "run_sft",
]
