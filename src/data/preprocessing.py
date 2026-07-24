"""Single-sample Qwen preprocessing for Drawing2CAD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .dataset import Drawing2CADSample
from .dxf import DXFPrimitiveData


DEFAULT_INSTRUCTION = (
    "Generate CadQuery Python code for the depicted part and assign the final "
    "CadQuery object to `result`."
)


def _find_last_subsequence(sequence: torch.Tensor, pattern: Sequence[int]) -> int:
    pattern_length = len(pattern)
    if pattern_length == 0:
        raise ValueError("assistant prefix token pattern must not be empty")
    pattern_tensor = torch.tensor(pattern, dtype=sequence.dtype, device=sequence.device)
    for start in range(sequence.numel() - pattern_length, -1, -1):
        if torch.equal(sequence[start : start + pattern_length], pattern_tensor):
            return start
    return -1


@dataclass(frozen=True)
class PreparedDrawing2CADSample:
    """Unbatched Qwen tensors plus the semantic primitive data."""

    sample_id: str
    primitives: DXFPrimitiveData
    model_inputs: Mapping[str, torch.Tensor]


class Drawing2CADPreprocessor:
    """Convert one semantic sample into one non-truncated Qwen example.

    Conversation construction, the single primitive placeholder block, native
    image processing, and completion-only labels all happen here. The collator
    receives only prepared tensors and performs batching operations.
    """

    _SEQUENCE_KEYS = ("input_ids", "attention_mask", "mm_token_type_ids")

    def __init__(
        self,
        processor,
        num_primitive_latents: int,
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
        if not isinstance(num_primitive_latents, int) or num_primitive_latents <= 0:
            raise ValueError("num_primitive_latents must be a positive integer")
        if not instruction.strip():
            raise ValueError("instruction must not be empty")
        if max_length is not None and max_length <= 0:
            raise ValueError("max_length must be positive when provided")

        self.processor = processor
        self.num_primitive_latents = num_primitive_latents
        self.instruction = instruction
        self.include_labels = include_labels
        self.max_length = max_length
        self.placeholder_token_id = tokenizer.pad_token_id
        self.placeholder_text = tokenizer.pad_token * num_primitive_latents
        self.assistant_prefix_ids = tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )

    @staticmethod
    def _image_heading(style: str) -> str:
        if style == "isometric":
            return "isometric view image:\n"
        if style == "drawing":
            return "complete technical drawing image:\n"
        return f"{style} image:\n"

    def build_conversation(self, sample: Drawing2CADSample) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"technical drawing vector primitives:\n{self.placeholder_text}\n"
                ),
            }
        ]
        for style, image in zip(sample.image_styles, sample.images, strict=True):
            content.extend(
                [
                    {"type": "text", "text": self._image_heading(style)},
                    {"type": "image", "image": image},
                ]
            )
        content.append({"type": "text", "text": self.instruction})
        conversation: list[dict[str, Any]] = [{"role": "user", "content": content}]
        if self.include_labels:
            if sample.target_code is None:
                raise ValueError(
                    f"sample {sample.sample_id} has no target code but labels are enabled"
                )
            if not sample.target_code.strip():
                raise ValueError(f"sample {sample.sample_id} has an empty target")
            conversation.append(
                {"role": "assistant", "content": sample.target_code.strip()}
            )
        return conversation

    def sequence_length(self, sample: Drawing2CADSample) -> int:
        """Exact token length of the fully rendered sample without raising.

        Mirrors the tokenization performed in ``__call__`` (same conversation,
        images, and target). Used to measure the fixed per-sample overhead once;
        the bulk length filtering uses :meth:`target_token_count` instead.
        """
        conversation = self.build_conversation(sample)
        encoded = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=not self.include_labels,
            return_dict=True,
            return_tensors="pt",
        )
        return int(encoded["input_ids"].shape[-1])

    def target_token_count(self, target_code: str) -> int:
        """Token count of the assistant target text alone.

        Everything else in the sequence (chat template, instruction, image
        headings, the fixed image tokens, and the primitive placeholders) is a
        constant that does not depend on the sample, so a full sequence length
        is well approximated by this count plus that constant overhead.
        """
        return len(
            self.processor.tokenizer(target_code, add_special_tokens=False)["input_ids"]
        )

    def _unbatch_processor_output(
        self,
        encoded: Mapping[str, Any],
        *,
        sample_id: str,
    ) -> dict[str, torch.Tensor]:
        output: dict[str, torch.Tensor] = {}
        for key, value in encoded.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"processor output {key!r} for sample {sample_id} must be a tensor, "
                    f"got {type(value)}"
                )
            if key in self._SEQUENCE_KEYS:
                if value.ndim != 2 or value.shape[0] != 1:
                    raise ValueError(
                        f"processor output {key!r} must have single-sample shape [1,T], "
                        f"got {tuple(value.shape)}"
                    )
                output[key] = value[0]
            else:
                # Qwen image tensors are already flattened across the images of
                # this sample: pixel_values[P,D], image_grid_thw[I,3].
                output[key] = value
        return output

    def __call__(self, sample: Drawing2CADSample) -> PreparedDrawing2CADSample:
        conversation = self.build_conversation(sample)
        encoded = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=not self.include_labels,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = self._unbatch_processor_output(
            encoded, sample_id=sample.sample_id
        )
        required = set(self._SEQUENCE_KEYS)
        missing = required.difference(model_inputs)
        if missing:
            raise ValueError(
                f"processor output for sample {sample.sample_id} is missing Qwen "
                f"inputs: {sorted(missing)}"
            )
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        mm_token_type_ids = model_inputs["mm_token_type_ids"]
        if input_ids.ndim != 1 or attention_mask.shape != input_ids.shape:
            raise ValueError(
                "Qwen token tensors must have matching unbatched shape [T]"
            )
        if mm_token_type_ids.shape != input_ids.shape:
            raise ValueError("mm_token_type_ids must have unbatched shape [T]")
        if self.max_length is not None and input_ids.numel() > self.max_length:
            raise ValueError(
                f"sample {sample.sample_id} sequence length {input_ids.numel()} exceeds "
                f"max_length={self.max_length}; examples are not truncated because "
                "that could corrupt image markers, primitive placeholders, or the "
                "CadQuery target"
            )

        primitive_token_mask = (
            input_ids == self.placeholder_token_id
        ) & attention_mask.bool()
        actual_count = int(primitive_token_mask.sum().item())
        if actual_count != self.num_primitive_latents:
            raise ValueError(
                f"sample {sample.sample_id} primitive placeholder construction failed: "
                f"expected {self.num_primitive_latents}, got {actual_count}"
            )
        if torch.any(mm_token_type_ids[primitive_token_mask] != 0):
            raise ValueError(
                "processor assigned non-text modality to primitive placeholders"
            )
        model_inputs["primitive_token_mask"] = primitive_token_mask

        if self.include_labels:
            prefix_start = _find_last_subsequence(input_ids, self.assistant_prefix_ids)
            if prefix_start < 0:
                raise ValueError(
                    f"could not locate assistant answer boundary for sample "
                    f"{sample.sample_id}"
                )
            answer_start = prefix_start + len(self.assistant_prefix_ids)
            labels = input_ids.clone()
            labels[:answer_start] = -100
            labels[~attention_mask.bool()] = -100
            if torch.any(labels[primitive_token_mask] != -100):
                raise ValueError("primitive placeholders must be excluded from labels")
            if not torch.any(labels != -100):
                raise ValueError(f"sample {sample.sample_id} has no supervised tokens")
            model_inputs["labels"] = labels

        return PreparedDrawing2CADSample(
            sample_id=sample.sample_id,
            primitives=sample.primitives,
            model_inputs=model_inputs,
        )


__all__ = [
    "DEFAULT_INSTRUCTION",
    "Drawing2CADPreprocessor",
    "PreparedDrawing2CADSample",
]
