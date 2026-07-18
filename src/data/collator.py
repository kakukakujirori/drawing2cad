"""Qwen3-VL prompt and tensor collation for drawing-to-CAD samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from src.models import PrimitiveBatch, PrimitiveEncoderConfig

from .dataset import Drawing2CADSample


DEFAULT_INSTRUCTION = (
    "Generate CadQuery Python code for the depicted part and assign the final "
    "CadQuery object to `result`."
)


@dataclass(frozen=True)
class Drawing2CADBatch:
    """DataLoader output with model kwargs separated from sample metadata."""

    model_inputs: dict[str, Any]
    sample_ids: tuple[str, ...]

    def to(self, device: torch.device | str) -> "Drawing2CADBatch":
        moved: dict[str, Any] = {}
        for key, value in self.model_inputs.items():
            if isinstance(value, PrimitiveBatch):
                moved[key] = value.to(device)
            elif isinstance(value, torch.Tensor):
                moved[key] = value.to(device)
            else:
                moved[key] = value
        return Drawing2CADBatch(model_inputs=moved, sample_ids=self.sample_ids)


def _find_last_subsequence(sequence: torch.Tensor, pattern: Sequence[int]) -> int:
    pattern_length = len(pattern)
    if pattern_length == 0:
        raise ValueError("assistant prefix token pattern must not be empty")
    pattern_tensor = torch.tensor(pattern, dtype=sequence.dtype, device=sequence.device)
    for start in range(sequence.numel() - pattern_length, -1, -1):
        if torch.equal(sequence[start : start + pattern_length], pattern_tensor):
            return start
    return -1


class Drawing2CADCollator:
    """Create native Qwen image inputs plus padded :class:`PrimitiveBatch`.

    Primitive placeholders use the processor tokenizer's existing pad token in
    attended prompt positions. Their identity is immaterial because the model
    overwrites those embeddings; using an existing token avoids resizing Qwen's
    vocabulary. Right-padding occurrences are excluded by ``attention_mask``.
    """

    def __init__(
        self,
        processor,
        primitive_config: PrimitiveEncoderConfig,
        *,
        instruction: str = DEFAULT_INSTRUCTION,
        include_labels: bool = True,
        max_length: int | None = None,
    ) -> None:
        if not hasattr(processor, "tokenizer"):
            raise TypeError("processor must expose a tokenizer")
        tokenizer = processor.tokenizer
        if tokenizer.pad_token_id is None or tokenizer.pad_token is None:
            raise ValueError("processor tokenizer must define a pad token")
        if "drawing" not in primitive_config.view_types:
            raise ValueError("primitive view vocabulary must contain 'drawing'")
        if not instruction.strip():
            raise ValueError("instruction must not be empty")
        if max_length is not None and max_length <= 0:
            raise ValueError("max_length must be positive when provided")

        self.processor = processor
        self.primitive_config = primitive_config
        self.instruction = instruction
        self.include_labels = include_labels
        self.max_length = max_length
        self.drawing_view_id = primitive_config.view_types.index("drawing")
        self.placeholder_token_id = tokenizer.pad_token_id
        self.placeholder_text = (
            tokenizer.pad_token * primitive_config.num_primitive_latents
        )
        self.assistant_prefix_ids = tokenizer.encode(
            "<|im_start|>assistant\n",
            add_special_tokens=False,
        )

    def _conversation(self, sample: Drawing2CADSample) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "drawing view vector primitives:\n"
                    f"{self.placeholder_text}\n"
                ),
            }
        ]
        for style, image in zip(sample.image_styles, sample.images, strict=True):
            content.extend(
                [
                    {
                        "type": "text",
                        "text": f"isometric view image ({style}):\n",
                    },
                    {"type": "image", "image": image},
                ]
            )
        content.append({"type": "text", "text": self.instruction})
        conversation: list[dict[str, Any]] = [
            {"role": "user", "content": content}
        ]
        if self.include_labels:
            if sample.target_code is None:
                raise ValueError(
                    f"sample {sample.sample_id} has no target code but labels are enabled"
                )
            conversation.append(
                {"role": "assistant", "content": sample.target_code}
            )
        return conversation

    def _collate_primitives(
        self,
        samples: Sequence[Drawing2CADSample],
    ) -> PrimitiveBatch:
        batch_size = len(samples)
        max_primitives = max(sample.primitives.num_primitives for sample in samples)
        sample_count = samples[0].primitives.sample_features.shape[1]
        feature_dim = samples[0].primitives.sample_features.shape[2]
        if feature_dim != self.primitive_config.sample_feature_dim:
            raise ValueError(
                f"dataset sample feature dim {feature_dim} does not match model config "
                f"{self.primitive_config.sample_feature_dim}"
            )

        features = torch.zeros(
            batch_size,
            1,
            max_primitives,
            sample_count,
            feature_dim,
            dtype=torch.float32,
        )
        sample_mask = torch.zeros(
            batch_size,
            1,
            max_primitives,
            sample_count,
            dtype=torch.bool,
        )
        primitive_mask = torch.zeros(
            batch_size, 1, max_primitives, dtype=torch.bool
        )
        primitive_type_ids = torch.full(
            (batch_size, 1, max_primitives), -1, dtype=torch.long
        )
        primitive_group_ids = torch.full_like(primitive_type_ids, -1)
        view_type_ids = torch.full((batch_size, 1), self.drawing_view_id, dtype=torch.long)
        view_mask = torch.ones(batch_size, 1, dtype=torch.bool)

        for batch_index, sample in enumerate(samples):
            primitive_data = sample.primitives
            if primitive_data.sample_features.ndim != 3:
                raise ValueError(
                    f"sample {sample.sample_id} primitive features must have shape [N, S, C]"
                )
            if primitive_data.sample_features.shape[1:] != (sample_count, feature_dim):
                raise ValueError(
                    "all samples in a batch must share samples_per_primitive and "
                    "sample_feature_dim"
                )
            primitive_count = primitive_data.num_primitives
            if primitive_count == 0:
                raise ValueError(f"sample {sample.sample_id} contains no primitives")
            if torch.any(primitive_data.primitive_type_ids < 0) or torch.any(
                primitive_data.primitive_type_ids
                >= self.primitive_config.num_primitive_types
            ):
                raise ValueError(
                    f"sample {sample.sample_id} has primitive type IDs outside model config"
                )
            features[batch_index, 0, :primitive_count] = (
                primitive_data.sample_features
            )
            sample_mask[batch_index, 0, :primitive_count] = True
            primitive_mask[batch_index, 0, :primitive_count] = True
            primitive_type_ids[batch_index, 0, :primitive_count] = (
                primitive_data.primitive_type_ids
            )

        return PrimitiveBatch(
            sample_features=features,
            sample_mask=sample_mask,
            primitive_mask=primitive_mask,
            primitive_type_ids=primitive_type_ids,
            primitive_group_ids=primitive_group_ids,
            view_type_ids=view_type_ids,
            view_mask=view_mask,
        )

    def _build_labels(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        full_labels = input_ids.clone()
        answer_starts: list[int] = []
        for batch_index in range(input_ids.shape[0]):
            valid_length = int(attention_mask[batch_index].sum().item())
            prefix_start = _find_last_subsequence(
                input_ids[batch_index, :valid_length],
                self.assistant_prefix_ids,
            )
            if prefix_start < 0:
                raise ValueError(
                    "could not locate assistant answer boundary in processor output"
                )
            answer_start = prefix_start + len(self.assistant_prefix_ids)
            answer_starts.append(answer_start)
            full_labels[batch_index, :answer_start] = -100
            full_labels[batch_index, valid_length:] = -100

        sequence_length = input_ids.shape[1]
        logits_to_keep = sequence_length - min(answer_starts) + 1
        labels = torch.cat(
            (
                torch.full(
                    (input_ids.shape[0], 1),
                    -100,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                ),
                full_labels[:, sequence_length - logits_to_keep + 1 :],
            ),
            dim=1,
        )
        return labels, logits_to_keep

    def __call__(self, samples: Sequence[Drawing2CADSample]) -> Drawing2CADBatch:
        if not samples:
            raise ValueError("cannot collate an empty batch")
        conversations = [self._conversation(sample) for sample in samples]
        encoded = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=not self.include_labels,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )
        model_inputs = dict(encoded)
        required = {"input_ids", "attention_mask", "mm_token_type_ids"}
        missing = required.difference(model_inputs)
        if missing:
            raise ValueError(f"processor output is missing required Qwen inputs: {missing}")
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        if self.max_length is not None and input_ids.shape[1] > self.max_length:
            raise ValueError(
                f"batch sequence length {input_ids.shape[1]} exceeds max_length="
                f"{self.max_length}; examples are not truncated because that would "
                "corrupt image/primitive placeholders or CadQuery targets"
            )

        primitive_token_mask = (
            (input_ids == self.placeholder_token_id) & attention_mask.bool()
        )
        actual_counts = primitive_token_mask.sum(dim=-1)
        expected_count = self.primitive_config.num_primitive_latents
        if torch.any(actual_counts != expected_count):
            raise ValueError(
                "primitive placeholder construction failed: expected "
                f"{expected_count} per sample, got {actual_counts.tolist()}"
            )
        if torch.any(model_inputs["mm_token_type_ids"][primitive_token_mask] != 0):
            raise ValueError("processor assigned non-text modality to primitive placeholders")

        model_inputs["primitive_batch"] = self._collate_primitives(samples)
        model_inputs["primitive_token_mask"] = primitive_token_mask
        if self.include_labels:
            labels, logits_to_keep = self._build_labels(input_ids, attention_mask)
            model_inputs["labels"] = labels
            model_inputs["logits_to_keep"] = logits_to_keep
        else:
            model_inputs["logits_to_keep"] = 1
        return Drawing2CADBatch(
            model_inputs=model_inputs,
            sample_ids=tuple(sample.sample_id for sample in samples),
        )


__all__ = [
    "DEFAULT_INSTRUCTION",
    "Drawing2CADBatch",
    "Drawing2CADCollator",
]
