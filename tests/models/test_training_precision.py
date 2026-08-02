"""Regressions for the numerics that silently stall primitive-path training."""

from __future__ import annotations

import unittest

import torch

from src.models.factory import (
    apply_language_lora,
    cast_trainable_parameters,
    freeze_vision_encoder,
    primitive_encoder_modules,
)
from tests.model_helpers import (
    make_primitive_batch,
    primitive_config,
    tiny_drawing_model,
)


LORA_CONFIG = {
    "enabled": True,
    "rank": 2,
    "alpha": 4,
    "dropout": 0.0,
    "bias": "none",
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}


def peft_model(model: torch.nn.Module) -> torch.nn.Module:
    freeze_vision_encoder(model)
    wrapped = apply_language_lora(model, LORA_CONFIG)
    freeze_vision_encoder(wrapped)
    return wrapped


class Bfloat16MasterWeightTest(unittest.TestCase):
    """bf16 parameters cannot represent an AdamW step at the configured rate."""

    def test_bf16_weight_swallows_a_typical_adamw_step(self) -> None:
        # A LayerNorm gain sits at exactly 1.0 and AdamW's normalized step is
        # about `lr`, which is far below bf16's ~1/256 resolution at that
        # magnitude, so the weight never changes.
        weight = torch.ones((), dtype=torch.bfloat16)
        self.assertEqual((weight - 2.0e-4).item(), 1.0)
        self.assertEqual((weight - 1.0e-3).item(), 1.0)
        self.assertNotEqual((weight.float() - 2.0e-4).item(), 1.0)

    def test_bf16_primitive_encoder_does_not_move_under_adamw(self) -> None:
        model = tiny_drawing_model().to(torch.bfloat16)
        model = peft_model(model)
        encoder = model.base_model.model.primitive_encoder
        gain = dict(encoder.named_parameters())[
            "modules_to_save.default.local_feature_norm.weight"
        ]
        self.assertEqual(gain.dtype, torch.bfloat16)
        before = gain.detach().clone()

        optimizer = torch.optim.AdamW([gain], lr=1.0e-3)
        for _ in range(5):
            optimizer.zero_grad()
            gain.grad = torch.full_like(gain, 0.1)
            optimizer.step()
        self.assertTrue(torch.equal(gain.detach(), before))

    def test_fp32_cast_lets_the_same_steps_land(self) -> None:
        model = tiny_drawing_model().to(torch.bfloat16)
        model = peft_model(model)
        cast_trainable_parameters(model)
        encoder = model.base_model.model.primitive_encoder
        gain = dict(encoder.named_parameters())[
            "modules_to_save.default.local_feature_norm.weight"
        ]
        self.assertEqual(gain.dtype, torch.float32)
        before = gain.detach().clone()

        optimizer = torch.optim.AdamW([gain], lr=1.0e-3)
        for _ in range(5):
            optimizer.zero_grad()
            gain.grad = torch.full_like(gain, 0.1)
            optimizer.step()
        self.assertFalse(torch.equal(gain.detach(), before))

    def test_cast_leaves_the_frozen_backbone_in_bf16(self) -> None:
        model = tiny_drawing_model().to(torch.bfloat16)
        model = peft_model(model)
        cast_trainable_parameters(model)
        visual = freeze_vision_encoder(model)
        for parameter in visual.parameters():
            self.assertEqual(parameter.dtype, torch.bfloat16)
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.assertEqual(parameter.dtype, torch.float32, name)

    def test_cast_covers_both_peft_copies_so_encoding_keeps_one_dtype(self) -> None:
        # `encode_primitives` casts the incoming batch to the dtype of the first
        # primitive parameter, which is PEFT's frozen original copy. If that copy
        # stayed bf16 the fp32 trainable copy would receive bf16 inputs.
        model = tiny_drawing_model().to(torch.bfloat16)
        model = peft_model(model)
        cast_trainable_parameters(model)
        modules = primitive_encoder_modules(model)
        self.assertEqual(len(modules), 1)
        for name, parameter in modules[0].named_parameters():
            self.assertEqual(parameter.dtype, torch.float32, name)

        inner = model.base_model.model
        latents, counts = inner.encode_primitives(make_primitive_batch((3, 2)))
        self.assertEqual(latents.dtype, torch.float32)
        self.assertEqual(tuple(counts.tolist()), (2, 2))


class OutputScaleTest(unittest.TestCase):
    """Injected latents must start on the host model's embedding scale."""

    def _latent_rms(self, model: torch.nn.Module) -> float:
        latents, _ = model.encode_primitives(make_primitive_batch((4, 6)))
        return float(latents.detach().float().pow(2).mean().sqrt().item())

    def test_unnormalized_projection_overshoots_the_embedding_scale(self) -> None:
        torch.manual_seed(0)
        model = tiny_drawing_model(primitive_config(use_group_context=False))
        self.assertIsNone(model.primitive_encoder.output_norm)
        embedding = model.get_input_embeddings().weight
        embedding_rms = float(embedding.detach().pow(2).mean().sqrt().item())
        self.assertGreater(self._latent_rms(model), 4.0 * embedding_rms)

    def test_calibration_matches_the_token_embedding_rms(self) -> None:
        torch.manual_seed(0)
        config = primitive_config(use_group_context=False)
        model = tiny_drawing_model(
            type(config)(**{**config.to_dict(), "normalize_output": True})
        )
        embedding = model.get_input_embeddings().weight
        embedding_rms = float(embedding.detach().pow(2).mean().sqrt().item())

        applied = model.calibrate_primitive_output_scale()
        self.assertAlmostEqual(applied, embedding_rms, places=5)
        self.assertAlmostEqual(
            model.primitive_encoder.output_rms, embedding_rms, places=5
        )
        self.assertAlmostEqual(self._latent_rms(model), embedding_rms, places=5)

    def test_calibration_requires_the_normalizing_head(self) -> None:
        model = tiny_drawing_model(primitive_config(use_group_context=False))
        with self.assertRaises(ValueError):
            model.calibrate_primitive_output_scale()

    def test_normalize_output_defaults_off_for_existing_checkpoints(self) -> None:
        self.assertFalse(primitive_config().normalize_output)
        restored = type(primitive_config()).from_dict(
            {
                "sample_feature_dim": 2,
                "num_primitive_types": 7,
                "primitive_dim": 16,
                "num_primitive_latents": 4,
            }
        )
        self.assertFalse(restored.normalize_output)

    def test_output_norm_survives_reset_parameters(self) -> None:
        config = primitive_config(use_group_context=False)
        model = tiny_drawing_model(
            type(config)(**{**config.to_dict(), "normalize_output": True})
        )
        encoder = model.primitive_encoder
        encoder.set_output_rms(0.02)
        encoder.reset_parameters()
        # reset_parameters restores the module default rather than leaving the
        # tensor uninitialized, which is what checkpoint loading relies on.
        self.assertAlmostEqual(encoder.output_rms, 1.0, places=6)


GELU_FLOOR = -0.17


def _superseded_branch(block, inputs, mask):
    """The previous ordering: activate the convolution's output.

    Kept here so the property that motivated the change is asserted against the
    alternative rather than against hand-picked absolute thresholds.
    """
    mask_values = mask.unsqueeze(-1)
    normalized = block.norm(inputs).masked_fill(~mask_values, 0.0)
    return block.activation(block.conv(normalized.transpose(1, 2)).transpose(1, 2))


class ResidualBlockTest(unittest.TestCase):
    """The conv block must not bias the residual stream upward with depth."""

    SEEDS = (0, 5, 11)

    def _block(self, seed: int = 5):
        from src.models.primitive_encoder import _MaskedResidualConvBlock

        torch.manual_seed(seed)
        block = _MaskedResidualConvBlock(32)
        # Isolate the branch's own contribution from the convolution's bias.
        torch.nn.init.zeros_(block.conv.bias)
        return block

    def _blocks(self, seed: int, count: int):
        """A stack whose layers are independently initialized, as in the model."""
        from src.models.primitive_encoder import _MaskedResidualConvBlock

        torch.manual_seed(seed)
        blocks = [_MaskedResidualConvBlock(32) for _ in range(count)]
        for block in blocks:
            torch.nn.init.zeros_(block.conv.bias)
        return blocks

    @torch.no_grad()
    def test_the_branch_escapes_the_activation_floor(self) -> None:
        mask = torch.ones(64, 12, dtype=torch.bool)
        for seed in self.SEEDS:
            with self.subTest(seed=seed):
                block = self._block(seed)
                torch.manual_seed(99)
                inputs = torch.randn(64, 12, 32)
                current = block(inputs, mask) - inputs
                superseded = _superseded_branch(block, inputs, mask)
                # Activating the convolution's output clamps every element of
                # the residual branch at GELU's minimum, so the block can only
                # ever subtract 0.17 no matter what the data asks for.
                self.assertAlmostEqual(float(superseded.min()), GELU_FLOOR, places=2)
                self.assertLess(float(current.min()), -0.5)

    @torch.no_grad()
    def test_the_branch_contribution_is_centred(self) -> None:
        mask = torch.ones(64, 12, dtype=torch.bool)
        for seed in self.SEEDS:
            with self.subTest(seed=seed):
                block = self._block(seed)
                torch.manual_seed(99)
                inputs = torch.randn(64, 12, 32)
                current = block(inputs, mask) - inputs
                superseded = _superseded_branch(block, inputs, mask)
                self.assertGreater(float(superseded.mean()), 0.05)
                self.assertLess(abs(float(current.mean())), 0.1)

    @torch.no_grad()
    def test_depth_does_not_accumulate_a_positive_shift(self) -> None:
        mask = torch.ones(64, 12, dtype=torch.bool)
        for seed in self.SEEDS:
            with self.subTest(seed=seed):
                blocks = self._blocks(seed, 6)
                torch.manual_seed(99)
                inputs = torch.randn(64, 12, 32)

                hidden = inputs
                for block in blocks:
                    hidden = block(hidden, mask)
                current_drift = float((hidden - inputs).mean())

                hidden = inputs
                for block in blocks:
                    hidden = hidden + _superseded_branch(block, hidden, mask)
                superseded_drift = float((hidden - inputs).mean())

                self.assertGreater(superseded_drift, 0.5)
                self.assertLess(abs(current_drift), 0.3)

    def test_padding_stays_zero_through_the_block(self) -> None:
        block = self._block()
        torch.nn.init.normal_(block.conv.bias, std=1.0)
        inputs = torch.randn(4, 10, 32)
        mask = torch.zeros(4, 10, dtype=torch.bool)
        mask[:, :6] = True
        inputs = inputs.masked_fill(~mask.unsqueeze(-1), 0.0)
        output = block(inputs, mask)
        self.assertTrue(torch.all(output[~mask] == 0.0))
        self.assertTrue(torch.isfinite(output).all())

    def test_masked_positions_do_not_change_valid_outputs(self) -> None:
        # The convolution has padding=1, so a non-zero pad would bleed into the
        # last valid position.
        block = self._block()
        mask = torch.zeros(2, 10, dtype=torch.bool)
        mask[:, :6] = True
        base = torch.randn(2, 10, 32).masked_fill(~mask.unsqueeze(-1), 0.0)
        other = base.clone()
        other[:, 6:] = torch.randn(2, 4, 32)  # garbage behind the mask
        torch.testing.assert_close(block(base, mask)[mask], block(other, mask)[mask])


if __name__ == "__main__":
    unittest.main()
