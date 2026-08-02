"""Validation ablation that measures what the primitive path contributes."""

from __future__ import annotations

import unittest

import torch

from src.models import PrimitiveBatch
from src.training.sft import evaluate_loss
from tests.model_helpers import make_primitive_batch, tiny_drawing_model


class StubAccelerator:
    device = torch.device("cpu")
    num_processes = 1

    @staticmethod
    def reduce(tensor, reduction="sum"):
        return tensor


def batch(primitive_count: int, *, batch_size: int = 1, token: int = 5) -> dict:
    # `token` varies the prompt across batches. A cyclic donor assignment over
    # batches that all share one prompt would permute an identical multiset of
    # losses, so the gain would be zero no matter what the primitives do.
    input_ids = torch.tensor([[token, 6, 7, 8, 9, 10]]).repeat(batch_size, 1)
    labels = torch.tensor([[-100, -100, -100, -100, 9, 10]]).repeat(batch_size, 1)
    mask = torch.tensor([[False, True, True, False, False, False]]).repeat(
        batch_size, 1
    )
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "mm_token_type_ids": torch.zeros_like(input_ids, dtype=torch.int32),
        "labels": labels,
        "primitive_batch": make_primitive_batch((primitive_count,) * batch_size),
        "primitive_token_mask": mask,
    }


def noop_train_mode(model) -> None:
    model.train()


class PrimitiveBatchRollTest(unittest.TestCase):
    def test_roll_rotates_every_field_together(self) -> None:
        original = make_primitive_batch((3, 5, 2))
        rolled = original.roll(1)
        self.assertIsInstance(rolled, PrimitiveBatch)
        for index in range(3):
            source = (index - 1) % 3
            torch.testing.assert_close(
                rolled.sample_features[index], original.sample_features[source]
            )
            self.assertTrue(
                torch.equal(
                    rolled.primitive_mask[index], original.primitive_mask[source]
                )
            )
            self.assertTrue(
                torch.equal(
                    rolled.primitive_type_ids[index],
                    original.primitive_type_ids[source],
                )
            )

    def test_roll_keeps_optional_group_ids_absent(self) -> None:
        original = make_primitive_batch((3, 5), with_groups=False)
        self.assertIsNone(original.roll(1).primitive_group_ids)


class PrimitiveAblationTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.model = tiny_drawing_model()
        self.model.eval()
        self.accelerator = StubAccelerator()

    def _evaluate(self, batches, *, ablation: bool) -> dict[str, float]:
        return evaluate_loss(
            self.accelerator,
            self.model,
            batches,
            set_train_mode=noop_train_mode,
            prefix="val/z2c_val",
            primitive_ablation=ablation,
        )

    def test_disabled_by_default_reports_only_the_loss(self) -> None:
        values = self._evaluate([batch(3), batch(5, token=11)], ablation=False)
        self.assertEqual(list(values), ["val/z2c_val/loss"])

    def test_enabled_reports_the_shuffled_loss_and_the_gain(self) -> None:
        batches = [batch(3), batch(5, token=11), batch(2, token=12)]
        values = self._evaluate(batches, ablation=True)
        self.assertEqual(
            sorted(values),
            [
                "val/z2c_val/loss",
                "val/z2c_val/loss_shuffled_primitives",
                "val/z2c_val/primitive_gain",
            ],
        )
        self.assertAlmostEqual(
            values["val/z2c_val/primitive_gain"],
            values["val/z2c_val/loss_shuffled_primitives"] - values["val/z2c_val/loss"],
            places=10,
        )

    def test_the_clean_loss_is_unchanged_by_enabling_the_ablation(self) -> None:
        batches = [batch(3), batch(5, token=11), batch(2, token=12)]
        without = self._evaluate(batches, ablation=False)
        with_ablation = self._evaluate(batches, ablation=True)
        self.assertAlmostEqual(
            without["val/z2c_val/loss"], with_ablation["val/z2c_val/loss"], places=10
        )

    def test_batch_size_one_still_finds_a_donor_for_every_batch(self) -> None:
        # The configured validation batch size is 1, where an in-batch rotation
        # is a no-op; the donor must come from another batch or the gain would
        # be identically zero.
        batches = [batch(3), batch(5, token=11)]
        values = self._evaluate(batches, ablation=True)
        self.assertNotAlmostEqual(values["val/z2c_val/primitive_gain"], 0.0, places=9)

    def test_identical_primitives_everywhere_give_no_gain(self) -> None:
        # Sanity check on the paired accounting: when every batch carries the
        # same primitives, swapping them changes nothing.
        batches = [batch(4), batch(4, token=11), batch(4, token=12)]
        values = self._evaluate(batches, ablation=True)
        self.assertAlmostEqual(values["val/z2c_val/primitive_gain"], 0.0, places=9)

    def test_a_single_batch_falls_back_to_an_in_batch_rotation(self) -> None:
        values = self._evaluate([batch(3, batch_size=2)], ablation=True)
        self.assertIn("val/z2c_val/primitive_gain", values)

    def test_a_lone_batch_of_one_cannot_be_ablated(self) -> None:
        # No donor exists anywhere in the pass, so the comparison is dropped
        # rather than reported against a different token set.
        values = self._evaluate([batch(3)], ablation=True)
        self.assertEqual(list(values), ["val/z2c_val/loss"])


if __name__ == "__main__":
    unittest.main()
