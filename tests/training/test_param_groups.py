"""Per-group optimizer rates and their logging."""

from __future__ import annotations

import unittest

import torch

from src.models.factory import apply_language_lora, freeze_vision_encoder
from src.training.sft import (
    _log_train_step,
    build_optimizer,
    build_scheduler,
    clip_gradients,
)
from src.utils.metric_router import MetricRouter
from tests.model_helpers import tiny_drawing_model


LORA_CONFIG = {
    "enabled": True,
    "rank": 2,
    "alpha": 4,
    "dropout": 0.0,
    "bias": "none",
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}

OPTIMIZER_CONFIG = {
    "name": "adamw",
    "learning_rate": 2.0e-4,
    "weight_decay": 0.01,
    "betas": [0.9, 0.95],
    "eps": 1.0e-8,
    "fused": False,
    "param_groups": {
        "primitive_encoder": {"learning_rate": 1.0e-3, "weight_decay": 0.0}
    },
}


def peft_model() -> torch.nn.Module:
    model = tiny_drawing_model()
    freeze_vision_encoder(model)
    wrapped = apply_language_lora(model, LORA_CONFIG)
    freeze_vision_encoder(wrapped)
    return wrapped


class RecordingLogger:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def log(self, metrics, *, step: int) -> None:
        self.events.append(dict(metrics))

    def log_artifact(self, path, *, name=None, kind="artifact") -> None:
        raise AssertionError("not expected")

    def finish(self) -> None:
        pass


class RecordingProgress:
    def __init__(self) -> None:
        self.displayed: dict[str, str] = {}

    def update_metrics(self, metrics) -> None:
        self.displayed.update(metrics)


class StubAccelerator:
    device = torch.device("cpu")
    num_processes = 1

    @staticmethod
    def reduce(tensor, reduction="sum"):
        return tensor


class ParamGroupTest(unittest.TestCase):
    def test_primitive_encoder_gets_its_own_rate_and_decay(self) -> None:
        model = peft_model()
        optimizer = build_optimizer(model, OPTIMIZER_CONFIG)
        self.assertEqual(len(optimizer.param_groups), 2)

        default, primitive = optimizer.param_groups
        self.assertIsNone(default.get("name"))
        self.assertEqual(default["lr"], 2.0e-4)
        self.assertEqual(default["weight_decay"], 0.01)
        self.assertEqual(primitive["name"], "primitive_encoder")
        self.assertEqual(primitive["lr"], 1.0e-3)
        self.assertEqual(primitive["weight_decay"], 0.0)

    def test_every_trainable_parameter_lands_in_exactly_one_group(self) -> None:
        model = peft_model()
        optimizer = build_optimizer(model, OPTIMIZER_CONFIG)
        grouped = [
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        trainable = {
            id(parameter) for parameter in model.parameters() if parameter.requires_grad
        }
        self.assertEqual(len(grouped), len(set(grouped)))
        self.assertEqual(set(grouped), trainable)

    def test_group_membership_follows_the_parameter_names(self) -> None:
        model = peft_model()
        optimizer = build_optimizer(model, OPTIMIZER_CONFIG)
        by_id = {
            id(parameter): name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        default, primitive = optimizer.param_groups
        self.assertTrue(
            all(
                "primitive_encoder" in by_id[id(parameter)]
                for parameter in primitive["params"]
            )
        )
        self.assertTrue(
            all(
                "primitive_encoder" not in by_id[id(parameter)]
                for parameter in default["params"]
            )
        )
        self.assertTrue(
            any("lora_" in by_id[id(parameter)] for parameter in default["params"])
        )

    def test_absent_param_groups_keeps_a_single_default_group(self) -> None:
        model = peft_model()
        config = {key: value for key, value in OPTIMIZER_CONFIG.items()}
        config.pop("param_groups")
        optimizer = build_optimizer(model, config)
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.param_groups[0]["lr"], 2.0e-4)

    def test_a_group_matching_nothing_is_a_configuration_error(self) -> None:
        model = peft_model()
        config = {**OPTIMIZER_CONFIG, "param_groups": {"typo_encoder": {}}}
        with self.assertRaises(ValueError):
            build_optimizer(model, config)

    def test_scheduler_scales_both_groups_from_their_own_base_rate(self) -> None:
        model = peft_model()
        optimizer = build_optimizer(model, OPTIMIZER_CONFIG)
        scheduler = build_scheduler(
            optimizer,
            {"name": "cosine", "warmup_steps": 10, "num_cycles": 0.5},
            total_steps=100,
        )
        # Warmup is linear from zero, so after 5 of 10 warmup steps both groups
        # sit at half of their own configured rate.
        for _ in range(5):
            scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1.0e-4, places=9)
        self.assertAlmostEqual(optimizer.param_groups[1]["lr"], 5.0e-4, places=9)

    def test_named_groups_are_logged_under_their_own_key(self) -> None:
        model = peft_model()
        optimizer = build_optimizer(model, OPTIMIZER_CONFIG)
        logger = RecordingLogger()
        progress = RecordingProgress()
        _log_train_step(
            MetricRouter(logger, progress),
            accelerator=StubAccelerator(),
            step=1,
            loss_sum=4.0,
            optimizer=optimizer,
            grad_norms={"train/grad_norm": torch.tensor(0.5)},
            tokens=2,
            elapsed=1.0,
            include_tokens_per_second=True,
        )
        (event,) = logger.events
        self.assertEqual(event["train/learning_rate"], 2.0e-4)
        self.assertEqual(event["train/learning_rate/primitive_encoder"], 1.0e-3)
        # The base rate keeps the progress-bar slot; the override is logged only.
        self.assertEqual(progress.displayed["lr"], "2.00e-04")
        self.assertNotIn("train/learning_rate/primitive_encoder", progress.displayed)


class StubClipAccelerator(StubAccelerator):
    """Records what each clip call saw and reports the pre-clip norm."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, float]] = []

    def clip_grad_norm_(self, parameters, max_norm):
        parameters = list(parameters)
        self.calls.append((len(parameters), float(max_norm)))
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)


class GradientClipTest(unittest.TestCase):
    def _model_with_gradients(self, default_scale: float, primitive_scale: float):
        model = peft_model()
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            scale = primitive_scale if "primitive_encoder" in name else default_scale
            parameter.grad = torch.full_like(parameter, scale)
        return model

    def test_each_group_is_clipped_on_its_own_norm(self) -> None:
        model = self._model_with_gradients(1e-4, 10.0)
        optimizer = build_optimizer(model, OPTIMIZER_CONFIG)
        accelerator = StubClipAccelerator()
        norms = clip_gradients(accelerator, optimizer, 1.0)

        self.assertEqual(len(accelerator.calls), 2)
        self.assertEqual(
            sorted(norms), ["train/grad_norm", "train/grad_norm/primitive_encoder"]
        )
        # The primitive group blew past the limit; the tiny LoRA gradients must
        # not have been rescaled along with it.
        self.assertGreater(float(norms["train/grad_norm/primitive_encoder"]), 1.0)
        self.assertLess(float(norms["train/grad_norm"]), 1.0)
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and "primitive_encoder" not in name:
                torch.testing.assert_close(
                    parameter.grad, torch.full_like(parameter, 1e-4)
                )

    def test_a_single_global_clip_would_have_shrunk_the_untouched_group(self) -> None:
        # Same gradients, one clip over everything: this is the behaviour the
        # per-group clip replaces.
        model = self._model_with_gradients(1e-4, 10.0)
        trainable = [p for p in model.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        lora = [
            parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and "primitive_encoder" not in name
        ]
        self.assertTrue(all(float(p.grad.abs().max()) < 1e-4 for p in lora))

    def test_a_group_can_override_the_shared_limit(self) -> None:
        config = {
            **OPTIMIZER_CONFIG,
            "param_groups": {
                "primitive_encoder": {"learning_rate": 1.0e-3, "max_grad_norm": 0.1}
            },
        }
        model = self._model_with_gradients(1e-4, 10.0)
        optimizer = build_optimizer(model, config)
        accelerator = StubClipAccelerator()
        clip_gradients(accelerator, optimizer, 1.0)
        self.assertEqual(sorted(limit for _, limit in accelerator.calls), [0.1, 1.0])

    def test_clipping_is_disabled_by_a_non_positive_limit(self) -> None:
        model = self._model_with_gradients(1e-4, 10.0)
        optimizer = build_optimizer(model, OPTIMIZER_CONFIG)
        accelerator = StubClipAccelerator()
        self.assertEqual(clip_gradients(accelerator, optimizer, 0.0), {})
        self.assertEqual(accelerator.calls, [])

    def test_every_group_norm_reaches_the_log(self) -> None:
        model = self._model_with_gradients(1e-4, 10.0)
        optimizer = build_optimizer(model, OPTIMIZER_CONFIG)
        norms = clip_gradients(StubClipAccelerator(), optimizer, 1.0)
        logger = RecordingLogger()
        _log_train_step(
            MetricRouter(logger),
            accelerator=StubAccelerator(),
            step=1,
            loss_sum=4.0,
            optimizer=optimizer,
            grad_norms=norms,
            tokens=2,
            elapsed=1.0,
            include_tokens_per_second=False,
        )
        (event,) = logger.events
        self.assertIn("train/grad_norm", event)
        self.assertIn("train/grad_norm/primitive_encoder", event)
        self.assertIsInstance(event["train/grad_norm/primitive_encoder"], float)


if __name__ == "__main__":
    unittest.main()
