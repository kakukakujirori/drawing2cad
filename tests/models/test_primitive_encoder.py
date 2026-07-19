import unittest
from unittest import mock

import torch

from src.models import PrimitiveEncoder, PrimitiveEncoderConfig
from tests.model_helpers import (
    make_primitive_batch,
    primitive_config,
    replace_primitive_batch,
)


class PrimitiveEncoderTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = PrimitiveEncoder(primitive_config(), output_dim=32).eval()

    def test_variable_primitive_counts_keep_one_fixed_latent_block(self) -> None:
        for primitive_count in (1, 3, 9):
            with self.subTest(primitive_count=primitive_count):
                batch = make_primitive_batch((primitive_count,))
                flat, counts = self.model(batch)
                self.assertEqual(counts.tolist(), [4])
                self.assertEqual(flat.shape, (4, 32))

    def test_global_resampler_runs_once_for_a_mixed_batch(self) -> None:
        batch = make_primitive_batch((1, 5, 9))
        with mock.patch.object(
            self.model.resampler,
            "forward",
            wraps=self.model.resampler.forward,
        ) as resample:
            flat, counts = self.model(batch)
        resample.assert_called_once()
        args = resample.call_args.args
        self.assertEqual(args[0].shape[:2], (3, 9))
        self.assertIs(args[1], batch.primitive_mask)
        self.assertEqual(flat.shape, (12, 32))
        self.assertEqual(counts.tolist(), [4, 4, 4])

    def test_primitive_permutation_invariance(self) -> None:
        batch = make_primitive_batch((4,))
        output, _ = self.model(batch)
        permutation = torch.tensor([2, 0, 3, 1])
        permuted = replace_primitive_batch(
            batch,
            sample_features=batch.sample_features[:, permutation],
            sample_mask=batch.sample_mask[:, permutation],
            primitive_mask=batch.primitive_mask[:, permutation],
            primitive_type_ids=batch.primitive_type_ids[:, permutation],
            view_direction_ids=batch.view_direction_ids[:, permutation],
            primitive_group_ids=batch.primitive_group_ids[:, permutation],
        )
        permuted_output, _ = self.model(permuted)
        torch.testing.assert_close(output, permuted_output, atol=1e-5, rtol=1e-5)

    def test_direction_embedding_is_attached_per_primitive(self) -> None:
        batch = make_primitive_batch((3,), with_groups=False)
        original = self.model.encode_local(batch)
        changed_ids = batch.view_direction_ids.clone()
        changed_ids[0, 0] = (changed_ids[0, 0] + 1) % 3
        changed = self.model.encode_local(
            replace_primitive_batch(batch, view_direction_ids=changed_ids)
        )
        self.assertFalse(torch.equal(original[0, 0], changed[0, 0]))
        torch.testing.assert_close(original[0, 1:], changed[0, 1:])

    def test_curve_reversal_invariance(self) -> None:
        batch = make_primitive_batch((5,))
        reversed_features = batch.sample_features.clone()
        for index in batch.primitive_mask.nonzero(as_tuple=False):
            batch_index, primitive_index = index.tolist()
            sample_count = int(batch.sample_mask[batch_index, primitive_index].sum())
            reversed_features[batch_index, primitive_index, :sample_count] = (
                batch.sample_features[batch_index, primitive_index, :sample_count].flip(
                    0
                )
            )
        reversed_batch = replace_primitive_batch(
            batch, sample_features=reversed_features
        )
        original, _ = self.model(batch)
        reversed_output, _ = self.model(reversed_batch)
        torch.testing.assert_close(original, reversed_output)

    def test_padded_values_do_not_affect_output(self) -> None:
        batch = make_primitive_batch((2, 5))
        changed = batch.sample_features.clone()
        changed[~batch.sample_mask] = (
            torch.randn_like(changed[~batch.sample_mask]) * 1e7
        )
        changed_batch = replace_primitive_batch(batch, sample_features=changed)
        original, _ = self.model(batch)
        changed_output, _ = self.model(changed_batch)
        torch.testing.assert_close(original, changed_output)

    def test_local_encoding_accepts_mixed_precision_input_under_autocast(self) -> None:
        batch = make_primitive_batch((3,))
        batch = replace_primitive_batch(
            batch, sample_features=batch.sample_features.to(torch.bfloat16)
        )
        with torch.autocast("cpu", dtype=torch.bfloat16):
            local = self.model.encode_local(batch)
        self.assertEqual(
            local.shape,
            (1, batch.sample_features.shape[1], self.model.config.primitive_dim),
        )
        self.assertTrue(torch.isfinite(local).all())

    def test_group_context_spans_view_directions_and_is_id_invariant(self) -> None:
        batch = make_primitive_batch((4,))
        self.assertNotEqual(
            batch.view_direction_ids[0, 0].item(),
            batch.view_direction_ids[0, 1].item(),
        )
        local = self.model.encode_local(batch)
        grouped = self.model.apply_group_context(local, batch)

        group_mask = batch.primitive_mask[0] & (batch.primitive_group_ids[0] == 10)
        members = local[0][group_mask]
        context_module = self.model.group_context
        shared_context = context_module.context_projection(
            context_module.context_mlp(members.mean(dim=0, keepdim=True))
        )
        expected = context_module.output_norm(members + shared_context)
        torch.testing.assert_close(grouped[0][group_mask], expected)

        ungrouped = batch.primitive_mask & (batch.primitive_group_ids == -1)
        torch.testing.assert_close(grouped[ungrouped], local[ungrouped])

        renamed_ids = batch.primitive_group_ids.clone()
        renamed_ids[renamed_ids == 10] = 901
        renamed_ids[renamed_ids == 20] = 42
        renamed = replace_primitive_batch(batch, primitive_group_ids=renamed_ids)
        renamed_output, _ = self.model(renamed)
        original_output, _ = self.model(batch)
        torch.testing.assert_close(original_output, renamed_output)

    def test_all_ungrouped_is_exact_no_op(self) -> None:
        batch = make_primitive_batch((4,))
        batch = replace_primitive_batch(
            batch,
            primitive_group_ids=torch.full_like(batch.primitive_group_ids, -1),
        )
        local = self.model.encode_local(batch)
        self.assertTrue(
            torch.equal(local, self.model.apply_group_context(local, batch))
        )

        no_ids_batch = make_primitive_batch((4,), with_groups=False)
        no_ids_local = self.model.encode_local(no_ids_batch)
        self.assertTrue(
            torch.equal(
                no_ids_local,
                self.model.apply_group_context(no_ids_local, no_ids_batch),
            )
        )

    def test_config_round_trip_uses_direction_vocabulary(self) -> None:
        config = primitive_config()
        values = config.to_dict()
        self.assertEqual(values["view_directions"], ["front", "top", "right"])
        self.assertNotIn("view_types", values)
        self.assertEqual(PrimitiveEncoderConfig.from_dict(values), config)

    def test_validation_errors_are_descriptive(self) -> None:
        batch = make_primitive_batch((3,))

        with self.subTest("mask dtype"):
            invalid = replace_primitive_batch(
                batch, sample_mask=batch.sample_mask.long()
            )
            with self.assertRaisesRegex(TypeError, "sample_mask must be a BoolTensor"):
                self.model(invalid)

        with self.subTest("empty sample"):
            invalid = replace_primitive_batch(
                batch,
                primitive_mask=torch.zeros_like(batch.primitive_mask),
                sample_mask=torch.zeros_like(batch.sample_mask),
            )
            with self.assertRaisesRegex(ValueError, "at least one active primitive"):
                self.model(invalid)

        with self.subTest("type range"):
            invalid_types = batch.primitive_type_ids.clone()
            invalid_types[batch.primitive_mask] = 99
            with self.assertRaisesRegex(ValueError, "primitive_type_ids"):
                self.model(
                    replace_primitive_batch(batch, primitive_type_ids=invalid_types)
                )

        with self.subTest("direction range"):
            invalid_directions = batch.view_direction_ids.clone()
            invalid_directions[batch.primitive_mask] = 99
            with self.assertRaisesRegex(ValueError, "view_direction_ids"):
                self.model(
                    replace_primitive_batch(
                        batch, view_direction_ids=invalid_directions
                    )
                )

        with self.subTest("non-prefix sample mask"):
            invalid_mask = batch.sample_mask.clone()
            invalid_mask[0, 0, :3] = torch.tensor([True, False, True])
            with self.assertRaisesRegex(ValueError, "contiguous prefix"):
                self.model(replace_primitive_batch(batch, sample_mask=invalid_mask))

    def test_forward_backward_is_finite_and_reaches_all_major_modules(self) -> None:
        self.model.train()
        output, _ = self.model(make_primitive_batch((2, 7)))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        parameters = (
            self.model.curve_encoder.input_projection.weight,
            self.model.view_direction_embedding.weight,
            self.model.resampler.queries,
            self.model.output_projection.weight,
        )
        for parameter in parameters:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
