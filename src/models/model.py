"""Qwen3-VL integration for raster views and sampled DXF primitives."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoConfig, Qwen3VLConfig, Qwen3VLForConditionalGeneration
from transformers.cache_utils import Cache

from .primitive_encoder import PrimitiveBatch, PrimitiveEncoder, PrimitiveEncoderConfig


DEFAULT_MODEL_NAME_OR_PATH = "ADSKAILab/Zero-To-CAD-Qwen3-VL-2B"


def _resolve_attn(attn_implementation: str) -> str:
    """Use FlashAttention when installed, otherwise the native SDPA backend."""
    if attn_implementation != "auto":
        return attn_implementation
    try:
        import flash_attn  # noqa: F401
    except Exception:
        return "sdpa"
    return "flash_attention_2"


def _cache_has_tokens(past_key_values: Cache | None) -> bool:
    if past_key_values is None:
        return False
    if hasattr(past_key_values, "get_seq_length"):
        return past_key_values.get_seq_length() > 0
    # Compatibility with legacy tuple caches accepted by older Transformers.
    try:
        return bool(past_key_values) and past_key_values[0][0].shape[-2] > 0
    except (IndexError, TypeError, AttributeError):
        return False


class Drawing2CADQwen3VLForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """Qwen3-VL with fixed-budget primitive latent embedding replacement.

    The native Qwen vision path remains owned by Transformers. Primitive latents
    replace only positions selected by ``primitive_token_mask`` before the call to
    the parent implementation.
    """

    def __init__(
        self,
        config: Qwen3VLConfig,
        primitive_config: PrimitiveEncoderConfig | dict[str, Any] | None = None,
    ) -> None:
        if primitive_config is None:
            primitive_config = getattr(config, "primitive_config", None)
        if primitive_config is None:
            raise ValueError(
                "primitive_config is required. Initialize a base Qwen checkpoint with "
                "from_qwen_pretrained(..., primitive_config=...) or reload a checkpoint "
                "saved by this class."
            )
        if isinstance(primitive_config, dict):
            primitive_config = PrimitiveEncoderConfig.from_dict(primitive_config)
        elif not isinstance(primitive_config, PrimitiveEncoderConfig):
            raise TypeError(
                "primitive_config must be PrimitiveEncoderConfig or a dictionary, got "
                f"{type(primitive_config)}"
            )

        config.primitive_config = primitive_config.to_dict()
        config.architectures = [self.__class__.__name__]
        super().__init__(config)
        hidden_size = config.text_config.hidden_size
        self.primitive_encoder = PrimitiveEncoder(primitive_config, output_dim=hidden_size)
        self.primitive_config = primitive_config

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Strictly reload a checkpoint saved by this implementation.

        Missing primitive keys are accepted only through ``from_qwen_pretrained``;
        all other missing or unexpected parameters are treated as checkpoint errors.
        """
        allow_missing_primitive = kwargs.pop("_allow_missing_primitive", False)
        return_loading_info = kwargs.pop("output_loading_info", False)
        model, loading_info = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            output_loading_info=True,
            **kwargs,
        )

        missing_keys = loading_info.get("missing_keys", [])
        unexpected_keys = loading_info.get("unexpected_keys", [])
        mismatched_keys = loading_info.get("mismatched_keys", [])
        error_messages = loading_info.get("error_msgs", [])
        disallowed_missing = [
            key
            for key in missing_keys
            if not (allow_missing_primitive and key.startswith("primitive_encoder."))
        ]
        if disallowed_missing or unexpected_keys or mismatched_keys or error_messages:
            details = []
            if disallowed_missing:
                details.append(f"missing keys: {disallowed_missing}")
            if unexpected_keys:
                details.append(f"unexpected keys: {unexpected_keys}")
            if mismatched_keys:
                details.append(f"mismatched keys: {mismatched_keys}")
            if error_messages:
                details.append(f"load errors: {error_messages}")
            raise RuntimeError(
                "Checkpoint is not an exact Drawing2CAD/Qwen3-VL match ("
                + "; ".join(details)
                + "). Use from_qwen_pretrained() only when initializing the newly "
                "added primitive modules from a base Qwen3-VL checkpoint."
            )
        if allow_missing_primitive and any(
            not key.startswith("primitive_encoder.") for key in missing_keys
        ):
            raise RuntimeError("base Qwen checkpoint is missing non-primitive weights")
        if allow_missing_primitive:
            # ``from_pretrained`` suppresses normal constructor initialization.
            # Explicitly initialize all newly added compound/bare parameters only
            # after strict loading proves that no Qwen parameter was missing.
            model.primitive_encoder.reset_parameters()
        return (model, loading_info) if return_loading_info else model

    @classmethod
    def from_qwen_pretrained(
        cls,
        model_name_or_path: str = DEFAULT_MODEL_NAME_OR_PATH,
        *,
        primitive_config: PrimitiveEncoderConfig | dict[str, Any],
        attn_implementation: str = "auto",
        **kwargs,
    ) -> "Drawing2CADQwen3VLForConditionalGeneration":
        """Initialize primitive modules on exactly the requested Qwen3-VL checkpoint."""
        if isinstance(primitive_config, dict):
            primitive_config = PrimitiveEncoderConfig.from_dict(primitive_config)
        elif not isinstance(primitive_config, PrimitiveEncoderConfig):
            raise TypeError(
                "primitive_config must be PrimitiveEncoderConfig or a dictionary, got "
                f"{type(primitive_config)}"
            )

        trust_remote_code = kwargs.pop("trust_remote_code", True)
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        if not isinstance(config, Qwen3VLConfig) or config.model_type != "qwen3_vl":
            raise ValueError(
                f"{model_name_or_path!r} is not a Qwen3-VL checkpoint "
                f"(model_type={getattr(config, 'model_type', None)!r})"
            )
        architectures = config.architectures or []
        if architectures and "Qwen3VLForConditionalGeneration" not in architectures:
            raise ValueError(
                f"{model_name_or_path!r} declares incompatible architectures: "
                f"{architectures}"
            )

        config.primitive_config = primitive_config.to_dict()
        config.architectures = [cls.__name__]
        resolved_attention = _resolve_attn(attn_implementation)
        return cls.from_pretrained(
            model_name_or_path,
            config=config,
            trust_remote_code=trust_remote_code,
            attn_implementation=resolved_attention,
            _allow_missing_primitive=True,
            **kwargs,
        )

    def encode_primitives(
        self,
        primitive_batch: PrimitiveBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode primitives into flat Qwen-width latents and per-sample counts."""
        if not isinstance(primitive_batch, PrimitiveBatch):
            raise TypeError(
                f"primitive_batch must be PrimitiveBatch, got {type(primitive_batch)}"
            )
        parameter = next(self.primitive_encoder.parameters())
        primitive_batch = primitive_batch.to(parameter.device, dtype=parameter.dtype)
        return self.primitive_encoder(primitive_batch)

    @staticmethod
    def _validate_primitive_placeholders(
        input_ids: torch.Tensor,
        primitive_token_mask: torch.Tensor,
        primitive_latent_counts: torch.Tensor,
        attention_mask: torch.Tensor | None,
        labels: torch.Tensor | None,
        mm_token_type_ids: torch.Tensor | None,
    ) -> None:
        if primitive_token_mask.dtype != torch.bool:
            raise TypeError(
                "primitive_token_mask must be a BoolTensor, got "
                f"{primitive_token_mask.dtype}"
            )
        if primitive_token_mask.shape != input_ids.shape:
            raise ValueError(
                "primitive_token_mask must have the same [B, T] shape as input_ids; "
                f"got {tuple(primitive_token_mask.shape)} and {tuple(input_ids.shape)}"
            )
        if primitive_token_mask.device != input_ids.device:
            raise ValueError("primitive_token_mask and input_ids must share a device")
        if primitive_latent_counts.shape != (input_ids.shape[0],):
            raise ValueError(
                "primitive latent counts must have shape [B], got "
                f"{tuple(primitive_latent_counts.shape)}"
            )
        actual_counts = primitive_token_mask.sum(dim=-1).to(
            primitive_latent_counts.device
        )
        if not torch.equal(actual_counts, primitive_latent_counts):
            raise ValueError(
                "primitive placeholder count mismatch: primitive_token_mask contains "
                f"{actual_counts.tolist()} positions per sample, but active primitive "
                f"views require {primitive_latent_counts.tolist()}"
            )

        if attention_mask is None:
            raise ValueError("attention_mask is required when primitive placeholders are used")
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask must have the same [B, T] shape as input_ids when "
                "primitive placeholders are used"
            )
        if not torch.all(attention_mask[primitive_token_mask].bool()):
            raise ValueError("primitive placeholder positions must have attention_mask=1")

        if mm_token_type_ids is None:
            raise ValueError(
                "mm_token_type_ids is required when primitive placeholders are used; "
                "primitive positions must use text modality ID 0"
            )
        if mm_token_type_ids.shape != input_ids.shape:
            raise ValueError(
                "mm_token_type_ids must have the same [B, T] shape as input_ids"
            )
        if torch.any(mm_token_type_ids[primitive_token_mask] != 0):
            raise ValueError("primitive placeholder positions must have mm_token_type_ids=0")

        if labels is not None:
            if labels.ndim != 2 or labels.shape[0] != input_ids.shape[0]:
                raise ValueError("labels must have shape [B, L]")
            if labels.shape[1] > input_ids.shape[1]:
                raise ValueError("labels cannot be longer than input_ids")
            label_offset = input_ids.shape[1] - labels.shape[1]
            overlapping_mask = primitive_token_mask[:, label_offset:]
            if torch.any(labels[overlapping_mask] != -100):
                raise ValueError("primitive placeholder positions must have labels=-100")

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        primitive_batch: PrimitiveBatch | None = None,
        primitive_token_mask: torch.BoolTensor | None = None,
        **kwargs,
    ):
        # Prefix conditioning is already represented in a non-empty KV cache.
        if _cache_has_tokens(past_key_values):
            primitive_batch = None
            primitive_token_mask = None

        has_primitive_batch = primitive_batch is not None
        has_primitive_mask = primitive_token_mask is not None
        if not has_primitive_batch and not has_primitive_mask:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )
        if has_primitive_batch != has_primitive_mask:
            raise ValueError(
                "primitive_batch and primitive_token_mask must be supplied together"
            )
        if input_ids is None or inputs_embeds is not None:
            raise ValueError(
                "primitive injection requires input_ids and does not accept caller-provided "
                "inputs_embeds on the full-prefix step"
            )
        if primitive_batch.batch_size != input_ids.shape[0]:
            raise ValueError(
                f"primitive batch size {primitive_batch.batch_size} does not match "
                f"input batch size {input_ids.shape[0]}"
            )

        primitive_latents, primitive_latent_counts = self.encode_primitives(
            primitive_batch
        )
        self._validate_primitive_placeholders(
            input_ids,
            primitive_token_mask,
            primitive_latent_counts,
            attention_mask,
            labels,
            mm_token_type_ids,
        )

        text_embeds = self.get_input_embeddings()(input_ids).clone()
        primitive_latents = primitive_latents.to(
            device=text_embeds.device,
            dtype=text_embeds.dtype,
        )
        offset = 0
        for batch_index, count_tensor in enumerate(primitive_latent_counts):
            count = int(count_tensor.item())
            if count:
                text_embeds[batch_index, primitive_token_mask[batch_index]] = (
                    primitive_latents[offset : offset + count]
                )
            offset += count

        if position_ids is None:
            has_native_visual_input = (
                image_grid_thw is not None or video_grid_thw is not None
            )
            if has_native_visual_input:
                if mm_token_type_ids is None:
                    raise ValueError(
                        "mm_token_type_ids is required to compute native multimodal RoPE"
                    )
                position_ids, rope_deltas = self.model.get_rope_index(
                    input_ids,
                    mm_token_type_ids=mm_token_type_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    attention_mask=attention_mask,
                )
                self.model.rope_deltas = rope_deltas
            else:
                # Parent Qwen ignores stale multimodal deltas on a fresh input_ids
                # forward. We pass inputs_embeds, so reset them explicitly for parity.
                self.model.rope_deltas = None

        return super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=text_embeds,
            labels=labels,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        is_first_iteration=False,
        primitive_batch: PrimitiveBatch | None = None,
        primitive_token_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            is_first_iteration=is_first_iteration,
            primitive_batch=primitive_batch,
            primitive_token_mask=primitive_token_mask,
            **kwargs,
        )
        if not _cache_has_tokens(past_key_values):
            model_inputs["primitive_batch"] = primitive_batch
            model_inputs["primitive_token_mask"] = primitive_token_mask
        else:
            model_inputs["primitive_batch"] = None
            model_inputs["primitive_token_mask"] = None
        return model_inputs

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: torch.LongTensor | None = None,
        **model_kwargs,
    ):
        primitive_batch = model_kwargs.get("primitive_batch")
        expanded_ids, expanded_kwargs = super()._expand_inputs_for_generation(
            expand_size=expand_size,
            is_encoder_decoder=is_encoder_decoder,
            input_ids=input_ids,
            **model_kwargs,
        )
        if primitive_batch is not None and expand_size != 1:
            expanded_kwargs["primitive_batch"] = primitive_batch.repeat_interleave(
                expand_size
            )
        return expanded_ids, expanded_kwargs


__all__ = ["Drawing2CADQwen3VLForConditionalGeneration"]
