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
    active_view_counts: tuple[int, ...] = (2, 3),
    *,
    with_groups: bool = True,
) -> PrimitiveBatch:
    generator = torch.Generator().manual_seed(123)
    batch_size = len(active_view_counts)
    view_count, primitive_count, sample_count, feature_dim = 5, 4, 6, 2
    features = torch.randn(
        batch_size,
        view_count,
        primitive_count,
        sample_count,
        feature_dim,
        generator=generator,
    )
    sample_mask = torch.zeros(
        batch_size, view_count, primitive_count, sample_count, dtype=torch.bool
    )
    primitive_mask = torch.zeros(
        batch_size, view_count, primitive_count, dtype=torch.bool
    )
    primitive_type_ids = torch.full(
        (batch_size, view_count, primitive_count), -1, dtype=torch.long
    )
    group_ids = torch.full_like(primitive_type_ids, -1)
    view_type_ids = torch.full((batch_size, view_count), -1, dtype=torch.long)
    view_mask = torch.zeros(batch_size, view_count, dtype=torch.bool)

    for batch_index, active_view_count in enumerate(active_view_counts):
        if not 0 <= active_view_count <= view_count:
            raise ValueError("active_view_count must lie in [0, 5]")
        # Reverse semantic IDs in physical slots to exercise canonical sorting.
        for slot in range(active_view_count):
            semantic_id = active_view_count - 1 - slot
            view_mask[batch_index, slot] = True
            view_type_ids[batch_index, slot] = semantic_id
            active_primitives = 1 + (slot % primitive_count)
            primitive_mask[batch_index, slot, :active_primitives] = True
            for primitive_index in range(active_primitives):
                primitive_type_ids[batch_index, slot, primitive_index] = (
                    batch_index + slot + primitive_index
                ) % 7
                valid_samples = 2 + ((slot + primitive_index) % (sample_count - 1))
                sample_mask[
                    batch_index, slot, primitive_index, :valid_samples
                ] = True

        if with_groups and active_view_count:
            # Group 10 spans views when possible; group 20 remains view-local.
            group_ids[batch_index, 0, 0] = 10
            if active_view_count > 1:
                group_ids[batch_index, 1, 0] = 10
            if primitive_mask[batch_index, 0, 1]:
                group_ids[batch_index, 0, 1] = 20

    return PrimitiveBatch(
        sample_features=features,
        sample_mask=sample_mask,
        primitive_mask=primitive_mask,
        primitive_type_ids=primitive_type_ids,
        primitive_group_ids=group_ids if with_groups else None,
        view_type_ids=view_type_ids,
        view_mask=view_mask,
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
    return Drawing2CADQwen3VLForConditionalGeneration(
        tiny_qwen_config(primitive)
    )
