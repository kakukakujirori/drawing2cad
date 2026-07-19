import unittest

import torch

from src.metrics.language import (
    causal_token_accuracy,
    exact_match_rate,
    mean_edit_similarity,
)


class LanguageMetricsTest(unittest.TestCase):
    def test_causal_token_accuracy_respects_shift_and_ignore_mask(self) -> None:
        logits = torch.zeros(1, 4, 4)
        labels = torch.tensor([[-100, 1, 2, -100]])
        logits[0, 0, 1] = 10.0
        logits[0, 1, 2] = 10.0
        self.assertEqual(causal_token_accuracy(logits, labels), 1.0)

        logits[0, 1, 2] = -10.0
        logits[0, 1, 3] = 10.0
        self.assertEqual(causal_token_accuracy(logits, labels), 0.5)

    def test_causal_token_accuracy_rejects_empty_supervision(self) -> None:
        with self.assertRaisesRegex(ValueError, "no supervised"):
            causal_token_accuracy(torch.zeros(1, 3, 2), torch.full((1, 3), -100))

    def test_text_metrics(self) -> None:
        predictions = [" result = 1\n", "result = 20"]
        references = ["result = 1", "result = 2"]
        self.assertEqual(exact_match_rate(predictions, references), 0.5)
        similarity = mean_edit_similarity(predictions, references)
        self.assertGreater(similarity, 0.9)
        self.assertLess(similarity, 1.0)

    def test_text_metric_lengths_must_match(self) -> None:
        with self.assertRaisesRegex(ValueError, "same length"):
            exact_match_rate(["one"], [])


if __name__ == "__main__":
    unittest.main()
