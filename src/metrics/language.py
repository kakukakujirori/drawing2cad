"""Language-level metrics for completion-only causal language modelling."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Sequence

from torch import Tensor


def causal_token_accuracy(
    logits: Tensor,
    labels: Tensor,
    *,
    ignore_index: int = -100,
) -> float:
    """Return next-token accuracy over supervised causal-LM positions.

    ``logits[:, t]`` predicts ``labels[:, t + 1]``. Positions carrying
    ``ignore_index`` are excluded, which makes the function compatible with the
    completion-only labels produced by the SFT preprocessor.
    """

    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [B, L, V], got {tuple(logits.shape)}")
    if labels.ndim != 2:
        raise ValueError(f"labels must have shape [B, L], got {tuple(labels.shape)}")
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            "logits and labels must agree on batch and sequence dimensions: "
            f"{tuple(logits.shape[:2])} != {tuple(labels.shape)}"
        )
    if labels.shape[1] < 2:
        raise ValueError("causal token accuracy requires a sequence length of at least two")

    shifted_labels = labels[:, 1:]
    supervised = shifted_labels.ne(ignore_index)
    count = int(supervised.sum().item())
    if count == 0:
        raise ValueError("labels contain no supervised causal-LM positions")
    predictions = logits[:, :-1].argmax(dim=-1)
    correct = predictions.eq(shifted_labels) & supervised
    return float(correct.sum().item() / count)


def exact_match_rate(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    strip: bool = True,
) -> float:
    """Return the fraction of generated strings exactly matching references."""

    _validate_text_pairs(predictions, references)
    if not predictions:
        return 0.0
    if strip:
        pairs = zip(
            (prediction.strip() for prediction in predictions),
            (reference.strip() for reference in references),
        )
    else:
        pairs = zip(predictions, references)
    return sum(prediction == reference for prediction, reference in pairs) / len(predictions)


def mean_edit_similarity(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    strip: bool = True,
) -> float:
    """Return mean character-level edit similarity in ``[0, 1]``.

    This is a diagnostic text metric only; executable CAD and geometry metrics
    remain the meaningful validation signals.
    """

    _validate_text_pairs(predictions, references)
    if not predictions:
        return 0.0
    values: list[float] = []
    for prediction, reference in zip(predictions, references):
        if strip:
            prediction, reference = prediction.strip(), reference.strip()
        values.append(SequenceMatcher(None, prediction, reference).ratio())
    return float(sum(values) / len(values))


def _validate_text_pairs(predictions: Sequence[str], references: Sequence[str]) -> None:
    if len(predictions) != len(references):
        raise ValueError(
            "predictions and references must have the same length: "
            f"{len(predictions)} != {len(references)}"
        )


__all__ = ["causal_token_accuracy", "exact_match_rate", "mean_edit_similarity"]
