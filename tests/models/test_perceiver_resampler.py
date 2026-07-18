import unittest

import torch

from src.models import PerceiverResampler


class PerceiverResamplerTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = PerceiverResampler(
            dim=16,
            num_latents=4,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
        ).eval()

    def test_fixed_output_count_for_variable_input_count(self) -> None:
        for primitive_count in (1, 3, 9):
            with self.subTest(primitive_count=primitive_count):
                tokens = torch.randn(2, primitive_count, 16)
                mask = torch.ones(2, primitive_count, dtype=torch.bool)
                output = self.model(tokens, mask)
                self.assertEqual(output.shape, (2, 4, 16))

    def test_padding_mask_matches_truncated_input(self) -> None:
        tokens = torch.randn(2, 6, 16)
        mask = torch.zeros(2, 6, dtype=torch.bool)
        mask[:, :2] = True
        padded_output = self.model(tokens, mask)
        truncated_output = self.model(
            tokens[:, :2], torch.ones(2, 2, dtype=torch.bool)
        )
        torch.testing.assert_close(padded_output, truncated_output)

        changed_padding = tokens.clone()
        changed_padding[:, 2:] = torch.randn_like(changed_padding[:, 2:]) * 1e6
        torch.testing.assert_close(
            padded_output,
            self.model(changed_padding, mask),
        )

    def test_gradients_reach_queries_and_context(self) -> None:
        tokens = torch.randn(2, 3, 16, requires_grad=True)
        mask = torch.ones(2, 3, dtype=torch.bool)
        self.model(tokens, mask).square().mean().backward()
        self.assertIsNotNone(self.model.queries.grad)
        self.assertGreater(self.model.queries.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(tokens.grad)
        self.assertGreater(tokens.grad.abs().sum().item(), 0.0)

    def test_rejects_an_empty_active_view(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one primitive"):
            self.model(
                torch.randn(1, 3, 16),
                torch.zeros(1, 3, dtype=torch.bool),
            )


if __name__ == "__main__":
    unittest.main()
