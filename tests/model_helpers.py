from __future__ import annotations

from dataclasses import replace

import torch
from transformers import Qwen3VLConfig

from src.models import (
    Drawing2CADQwen3VLForConditionalGeneration,
    PrimitiveBatch,
    PrimitiveEncoderConfig,
)


def primitive_config(*, use_group_context: bool = True) -> PrimitiveEncoderConfig:
    return PrimitiveEncoderConfig(
        sample_feature_dim=2,
        num_primitive_types=7,
        primitive_dim=16,
        num_primitive_latents=4,
        primitive_encoder_layers=2,
        resampler_layers=2,
        resampler_heads=4,
        dropout=0.0,
        use_group_context=use_group_context,
    )


def make_primitive_batch(
    active_primitive_counts: tuple[int, ...] = (4, 7),
    *,
    with_groups: bool = True,
) -> PrimitiveBatch:
    if not active_primitive_counts or any(
        count <= 0 for count in active_primitive_counts
    ):
        raise ValueError("every active primitive count must be positive")

    generator = torch.Generator().manual_seed(123)
    batch_size = len(active_primitive_counts)
    primitive_count = max(active_primitive_counts)
    sample_count, feature_dim = 6, 2
    features = torch.randn(
        batch_size,
        primitive_count,
        sample_count,
        feature_dim,
        generator=generator,
    )
    sample_mask = torch.zeros(
        batch_size, primitive_count, sample_count, dtype=torch.bool
    )
    primitive_mask = torch.zeros(batch_size, primitive_count, dtype=torch.bool)
    primitive_type_ids = torch.full((batch_size, primitive_count), -1, dtype=torch.long)
    view_direction_ids = torch.full_like(primitive_type_ids, -1)
    group_ids = torch.full_like(primitive_type_ids, -1)

    for batch_index, active_count in enumerate(active_primitive_counts):
        primitive_mask[batch_index, :active_count] = True
        for primitive_index in range(active_count):
            primitive_type_ids[batch_index, primitive_index] = (
                batch_index + primitive_index
            ) % 7
            view_direction_ids[batch_index, primitive_index] = primitive_index % 3
            valid_samples = 2 + (primitive_index % (sample_count - 1))
            sample_mask[batch_index, primitive_index, :valid_samples] = True

        if with_groups:
            # Group 10 spans directions; group 20 is a separate local group.
            group_ids[batch_index, 0] = 10
            if active_count > 1:
                group_ids[batch_index, 1] = 10
            if active_count > 2:
                group_ids[batch_index, 2] = 20

    return PrimitiveBatch(
        sample_features=features,
        sample_mask=sample_mask,
        primitive_mask=primitive_mask,
        primitive_type_ids=primitive_type_ids,
        view_direction_ids=view_direction_ids,
        primitive_group_ids=group_ids if with_groups else None,
    )


def replace_primitive_batch(batch: PrimitiveBatch, **changes) -> PrimitiveBatch:
    return replace(batch, **changes)


def tiny_qwen_config(
    primitive: PrimitiveEncoderConfig | None = None,
) -> Qwen3VLConfig:
    text_config = {
        "vocab_size": 128,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 128,
        "rope_parameters": {
            "rope_type": "default",
            "mrope_section": [2, 1, 1],
            "mrope_interleaved": True,
        },
        "pad_token_id": 0,
    }
    vision_config = {
        "depth": 2,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_heads": 4,
        "in_channels": 3,
        "patch_size": 2,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "out_hidden_size": 32,
        "num_position_embeddings": 16,
        "deepstack_visual_indexes": [0],
    }
    config = Qwen3VLConfig(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=120,
        video_token_id=121,
        vision_start_token_id=122,
        vision_end_token_id=123,
    )
    if primitive is not None:
        config.primitive_config = primitive.to_dict()
    return config


def tiny_drawing_model(
    primitive: PrimitiveEncoderConfig | None = None,
) -> Drawing2CADQwen3VLForConditionalGeneration:
    primitive = primitive or PrimitiveEncoderConfig(
        sample_feature_dim=2,
        num_primitive_types=7,
        primitive_dim=16,
        num_primitive_latents=2,
        primitive_encoder_layers=2,
        resampler_layers=1,
        resampler_heads=4,
        dropout=0.0,
    )
    return Drawing2CADQwen3VLForConditionalGeneration(tiny_qwen_config(primitive))
