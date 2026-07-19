"""Pure batching for preprocessed Drawing2CAD samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from src.models import PrimitiveBatch

from .dxf import DXF_PRIMITIVE_TYPES
from .metadata import SampleMetadata, VIEW_DIRECTIONS
from .preprocessing import PreparedDrawing2CADSample


@dataclass(frozen=True)
class Drawing2CADBatch:
    """DataLoader output with model kwargs and non-model sample metadata."""

    model_inputs: dict[str, Any]
    sample_ids: tuple[str, ...]
    sample_metadata: tuple[SampleMetadata, ...]

    def to(self, device: torch.device | str) -> "Drawing2CADBatch":
        moved: dict[str, Any] = {}
        for key, value in self.model_inputs.items():
            if isinstance(value, PrimitiveBatch):
                moved[key] = value.to(device)
            elif isinstance(value, torch.Tensor):
                moved[key] = value.to(device)
            else:
                moved[key] = value
        return Drawing2CADBatch(
            model_inputs=moved,
            sample_ids=self.sample_ids,
            sample_metadata=self.sample_metadata,
        )


class Drawing2CADCollator:
    """Pad/stack prepared tensors without constructing sample semantics."""

    _PAD_VALUES = {
        "attention_mask": 0,
        "mm_token_type_ids": 0,
        "primitive_token_mask": False,
        "labels": -100,
    }
    _VISION_CONCAT_KEYS = (
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
    )

    def __init__(self, pad_token_id: int, *, padding_side: str = "right") -> None:
        if not isinstance(pad_token_id, int) or pad_token_id < 0:
            raise ValueError("pad_token_id must be a non-negative integer")
        if padding_side not in {"left", "right"}:
            raise ValueError("padding_side must be 'left' or 'right'")
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side

    def _pad_sequences(
        self,
        samples: Sequence[PreparedDrawing2CADSample],
    ) -> dict[str, torch.Tensor]:
        keys = ("input_ids", *self._PAD_VALUES)
        label_presence = ["labels" in sample.model_inputs for sample in samples]
        if any(label_presence) and not all(label_presence):
            raise ValueError("a batch cannot mix labelled and unlabelled samples")
        active_keys = keys if all(label_presence) else keys[:-1]
        max_length = max(
            int(sample.model_inputs["input_ids"].numel()) for sample in samples
        )
        output: dict[str, torch.Tensor] = {}
        for key in active_keys:
            tensors = [sample.model_inputs[key] for sample in samples]
            for sample, tensor in zip(samples, tensors, strict=True):
                if tensor.ndim != 1:
                    raise ValueError(
                        f"sample {sample.sample_id} input {key!r} must have shape [T], "
                        f"got {tuple(tensor.shape)}"
                    )
            pad_value = (
                self.pad_token_id if key == "input_ids" else self._PAD_VALUES[key]
            )
            padded = tensors[0].new_full(
                (len(samples), max_length), pad_value
            )
            for batch_index, tensor in enumerate(tensors):
                length = tensor.numel()
                if self.padding_side == "right":
                    padded[batch_index, :length] = tensor
                else:
                    padded[batch_index, max_length - length :] = tensor
            output[key] = padded
        return output

    @staticmethod
    def _collate_primitives(
        samples: Sequence[PreparedDrawing2CADSample],
    ) -> PrimitiveBatch:
        batch_size = len(samples)
        max_primitives = max(sample.primitives.num_primitives for sample in samples)
        max_sample_count = max(
            sample.primitives.sample_features.shape[1] for sample in samples
        )
        feature_dim = samples[0].primitives.sample_features.shape[2]

        features = torch.zeros(
            batch_size,
            max_primitives,
            max_sample_count,
            feature_dim,
            dtype=torch.float32,
        )
        sample_mask = torch.zeros(
            batch_size,
            max_primitives,
            max_sample_count,
            dtype=torch.bool,
        )
        primitive_mask = torch.zeros(
            batch_size, max_primitives, dtype=torch.bool
        )
        primitive_type_ids = torch.full(
            (batch_size, max_primitives), -1, dtype=torch.long
        )
        view_direction_ids = torch.full_like(primitive_type_ids, -1)
        any_groups = any(
            sample.primitives.primitive_group_ids is not None for sample in samples
        )
        primitive_group_ids = (
            torch.full_like(primitive_type_ids, -1) if any_groups else None
        )

        for batch_index, sample in enumerate(samples):
            primitive_data = sample.primitives
            if primitive_data.sample_features.ndim != 3:
                raise ValueError(
                    f"sample {sample.sample_id} primitive features must have shape "
                    "[N,S,C]"
                )
            primitive_count, sample_count, current_feature_dim = (
                primitive_data.sample_features.shape
            )
            if primitive_count == 0:
                raise ValueError(f"sample {sample.sample_id} contains no primitives")
            if current_feature_dim != feature_dim:
                raise ValueError(
                    "all samples in a batch must share sample_feature_dim; "
                    f"expected {feature_dim}, got {current_feature_dim} for "
                    f"{sample.sample_id}"
                )
            expected_shape = (primitive_count,)
            if primitive_data.primitive_type_ids.shape != expected_shape:
                raise ValueError(
                    f"sample {sample.sample_id} primitive_type_ids must have shape "
                    f"{expected_shape}"
                )
            if primitive_data.view_direction_ids.shape != expected_shape:
                raise ValueError(
                    f"sample {sample.sample_id} view_direction_ids must have shape "
                    f"{expected_shape}"
                )
            if torch.any(primitive_data.primitive_type_ids < 0) or torch.any(
                primitive_data.primitive_type_ids >= len(DXF_PRIMITIVE_TYPES)
            ):
                raise ValueError(
                    f"sample {sample.sample_id} has invalid primitive type IDs"
                )
            if torch.any(primitive_data.view_direction_ids < 0) or torch.any(
                primitive_data.view_direction_ids >= len(VIEW_DIRECTIONS)
            ):
                raise ValueError(
                    f"sample {sample.sample_id} has invalid view direction IDs"
                )

            features[
                batch_index, :primitive_count, :sample_count
            ] = primitive_data.sample_features
            sample_mask[batch_index, :primitive_count, :sample_count] = True
            primitive_mask[batch_index, :primitive_count] = True
            primitive_type_ids[batch_index, :primitive_count] = (
                primitive_data.primitive_type_ids
            )
            view_direction_ids[batch_index, :primitive_count] = (
                primitive_data.view_direction_ids
            )
            if primitive_group_ids is not None:
                groups = primitive_data.primitive_group_ids
                if groups is not None:
                    if groups.shape != expected_shape:
                        raise ValueError(
                            f"sample {sample.sample_id} primitive_group_ids must have "
                            f"shape {expected_shape}"
                        )
                    primitive_group_ids[batch_index, :primitive_count] = groups

        return PrimitiveBatch(
            sample_features=features,
            sample_mask=sample_mask,
            primitive_mask=primitive_mask,
            primitive_type_ids=primitive_type_ids,
            view_direction_ids=view_direction_ids,
            primitive_group_ids=primitive_group_ids,
        )

    def _concatenate_vision_inputs(
        self,
        samples: Sequence[PreparedDrawing2CADSample],
    ) -> dict[str, torch.Tensor]:
        output: dict[str, torch.Tensor] = {}
        for key in self._VISION_CONCAT_KEYS:
            presence = [key in sample.model_inputs for sample in samples]
            if any(presence) and not all(presence):
                raise ValueError(
                    f"a batch cannot mix samples with and without vision input {key!r}"
                )
            if not any(presence):
                continue
            values = [sample.model_inputs[key] for sample in samples]
            if any(value.ndim == 0 for value in values):
                raise ValueError(f"vision input {key!r} must have a concat dimension")
            trailing_shape = values[0].shape[1:]
            if any(value.shape[1:] != trailing_shape for value in values[1:]):
                raise ValueError(
                    f"vision input {key!r} has incompatible trailing dimensions"
                )
            output[key] = torch.cat(values, dim=0)
        return output

    def __call__(
        self,
        samples: Sequence[PreparedDrawing2CADSample],
    ) -> Drawing2CADBatch:
        if not samples:
            raise ValueError("cannot collate an empty batch")
        model_inputs: dict[str, Any] = self._pad_sequences(samples)
        model_inputs.update(self._concatenate_vision_inputs(samples))

        handled_keys = {
            *model_inputs.keys(),
            *self._VISION_CONCAT_KEYS,
        }
        unexpected = set().union(
            *(set(sample.model_inputs) for sample in samples)
        ).difference(handled_keys)
        if unexpected:
            raise ValueError(
                "collator has no declared padding/stacking rule for processor "
                f"inputs: {sorted(unexpected)}"
            )

        # Qwen can project only the completion window instead of materializing
        # vocabulary logits for the full multimodal prompt.  Keep one ignored
        # position immediately before the earliest supervised token so causal
        # label shifting remains correct.  This is a batch-shape optimization:
        # answer semantics were already encoded as -100 by the preprocessor.
        if "labels" in model_inputs:
            labels = model_inputs["labels"]
            supervised_starts: list[int] = []
            for batch_index in range(labels.shape[0]):
                supervised = torch.nonzero(
                    labels[batch_index] != -100, as_tuple=False
                ).squeeze(-1)
                if supervised.numel() == 0:
                    raise ValueError(
                        f"sample {samples[batch_index].sample_id} has no supervised "
                        "tokens after batching"
                    )
                supervised_starts.append(int(supervised[0].item()))
            window_start = min(supervised_starts) - 1
            if window_start < 0:
                raise ValueError("completion labels require a preceding causal position")
            model_inputs["labels"] = labels[:, window_start:]
            model_inputs["logits_to_keep"] = labels.shape[1] - window_start
        else:
            model_inputs["logits_to_keep"] = 1

        model_inputs["primitive_batch"] = self._collate_primitives(samples)
        return Drawing2CADBatch(
            model_inputs=model_inputs,
            sample_ids=tuple(sample.sample_id for sample in samples),
            sample_metadata=tuple(sample.metadata for sample in samples),
        )


__all__ = ["Drawing2CADBatch", "Drawing2CADCollator"]
