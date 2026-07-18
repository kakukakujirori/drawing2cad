import unittest

import torch

from src.models import PrimitiveEncoder
from tests.model_helpers import (
    make_primitive_batch,
    primitive_config,
    replace_primitive_batch,
)


class PrimitiveEncoderTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = PrimitiveEncoder(primitive_config(), output_dim=32).eval()

    def test_one_through_five_views_and_variable_primitive_count(self) -> None:
        for view_count in range(1, 6):
            with self.subTest(view_count=view_count):
                batch = make_primitive_batch((view_count,))
                flat, counts = self.model(batch)
                self.assertEqual(counts.tolist(), [4 * view_count])
                self.assertEqual(flat.shape, (4 * view_count, 32))

    def test_primitive_permutation_invariance(self) -> None:
        batch = make_primitive_batch((3,))
        output, _ = self.model(batch)
        permutation = torch.tensor([2, 0, 3, 1])
        permuted = replace_primitive_batch(
            batch,
            sample_features=batch.sample_features[:, :, permutation],
            sample_mask=batch.sample_mask[:, :, permutation],
            primitive_mask=batch.primitive_mask[:, :, permutation],
            primitive_type_ids=batch.primitive_type_ids[:, :, permutation],
            primitive_group_ids=batch.primitive_group_ids[:, :, permutation],
        )
        permuted_output, _ = self.model(permuted)
        torch.testing.assert_close(output, permuted_output, atol=1e-5, rtol=1e-5)

    def test_view_slot_permutation_is_canonicalized_by_semantic_id(self) -> None:
        batch = make_primitive_batch((3,))
        output, _ = self.model(batch)
        permutation = torch.tensor([2, 0, 1, 4, 3])
        permuted = replace_primitive_batch(
            batch,
            sample_features=batch.sample_features[:, permutation],
            sample_mask=batch.sample_mask[:, permutation],
            primitive_mask=batch.primitive_mask[:, permutation],
            primitive_type_ids=batch.primitive_type_ids[:, permutation],
            primitive_group_ids=batch.primitive_group_ids[:, permutation],
            view_type_ids=batch.view_type_ids[:, permutation],
            view_mask=batch.view_mask[:, permutation],
        )
        permuted_output, _ = self.model(permuted)
        torch.testing.assert_close(output, permuted_output)

    def test_curve_reversal_invariance(self) -> None:
        batch = make_primitive_batch((3,))
        reversed_features = batch.sample_features.clone()
        for index in batch.primitive_mask.nonzero(as_tuple=False):
            batch_index, view_index, primitive_index = index.tolist()
            sample_count = int(
                batch.sample_mask[batch_index, view_index, primitive_index].sum()
            )
            reversed_features[
                batch_index, view_index, primitive_index, :sample_count
            ] = batch.sample_features[
                batch_index, view_index, primitive_index, :sample_count
            ].flip(0)
        reversed_batch = replace_primitive_batch(
            batch, sample_features=reversed_features
        )
        original, _ = self.model(batch)
        reversed_output, _ = self.model(reversed_batch)
        torch.testing.assert_close(original, reversed_output)

    def test_padded_values_do_not_affect_output(self) -> None:
        batch = make_primitive_batch((2,))
        changed = batch.sample_features.clone()
        changed[~batch.sample_mask] = torch.randn_like(changed[~batch.sample_mask]) * 1e7
        changed_batch = replace_primitive_batch(batch, sample_features=changed)
        original, _ = self.model(batch)
        changed_output, _ = self.model(changed_batch)
        torch.testing.assert_close(original, changed_output)

    def test_group_context_invariances_and_cross_view_broadcast(self) -> None:
        batch = make_primitive_batch((3,))
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
        batch = make_primitive_batch((2,))
        batch = replace_primitive_batch(
            batch,
            primitive_group_ids=torch.full_like(batch.primitive_group_ids, -1),
        )
        local = self.model.encode_local(batch)
        self.assertTrue(torch.equal(local, self.model.apply_group_context(local, batch)))

        no_ids_batch = make_primitive_batch((2,), with_groups=False)
        no_ids_local = self.model.encode_local(no_ids_batch)
        self.assertTrue(
            torch.equal(
                no_ids_local,
                self.model.apply_group_context(no_ids_local, no_ids_batch),
            )
        )

    def test_validation_errors_are_descriptive(self) -> None:
        batch = make_primitive_batch((1,))

        with self.subTest("mask dtype"):
            invalid = replace_primitive_batch(
                batch, sample_mask=batch.sample_mask.long()
            )
            with self.assertRaisesRegex(TypeError, "sample_mask must be a BoolTensor"):
                self.model(invalid)

        with self.subTest("empty active view"):
            invalid = replace_primitive_batch(
                batch,
                primitive_mask=torch.zeros_like(batch.primitive_mask),
                sample_mask=torch.zeros_like(batch.sample_mask),
            )
            with self.assertRaisesRegex(ValueError, "active primitive view"):
                self.model(invalid)

        with self.subTest("type range"):
            invalid_types = batch.primitive_type_ids.clone()
            invalid_types[batch.primitive_mask] = 99
            with self.assertRaisesRegex(ValueError, "primitive_type_ids"):
                self.model(
                    replace_primitive_batch(batch, primitive_type_ids=invalid_types)
                )

        with self.subTest("view range"):
            invalid_views = batch.view_type_ids.clone()
            invalid_views[batch.view_mask] = 99
            with self.assertRaisesRegex(ValueError, "view_type_ids"):
                self.model(replace_primitive_batch(batch, view_type_ids=invalid_views))

        with self.subTest("non-prefix sample mask"):
            invalid_mask = batch.sample_mask.clone()
            location = batch.primitive_mask.nonzero(as_tuple=False)[0].tolist()
            invalid_mask[location[0], location[1], location[2], :3] = torch.tensor(
                [True, False, True]
            )
            with self.assertRaisesRegex(ValueError, "contiguous prefix"):
                self.model(replace_primitive_batch(batch, sample_mask=invalid_mask))

    def test_forward_backward_is_finite_and_reaches_all_major_modules(self) -> None:
        self.model.train()
        output, _ = self.model(make_primitive_batch((2, 5)))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        parameters = (
            self.model.curve_encoder.input_projection.weight,
            self.model.resampler.queries,
            self.model.output_projection.weight,
        )
        for parameter in parameters:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
