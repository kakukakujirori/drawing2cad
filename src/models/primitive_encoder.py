"""Masked 2D drawing-primitive encoding and per-view resampling."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import torch
from torch import nn

from .perceiver_resampler import PerceiverResampler


DEFAULT_VIEW_TYPES = ("front", "top", "side", "drawing", "isometric")


@dataclass(frozen=True)
class PrimitiveEncoderConfig:
    """Serializable configuration for the drawing primitive path."""

    sample_feature_dim: int
    num_primitive_types: int
    primitive_dim: int = 256
    num_primitive_latents: int = 32
    primitive_encoder_layers: int = 2
    resampler_layers: int = 2
    resampler_heads: int = 8
    dropout: float = 0.1
    use_group_context: bool = True
    view_types: tuple[str, ...] = DEFAULT_VIEW_TYPES

    def __post_init__(self) -> None:
        object.__setattr__(self, "view_types", tuple(self.view_types))
        positive_fields = (
            "sample_feature_dim",
            "num_primitive_types",
            "primitive_dim",
            "num_primitive_latents",
            "primitive_encoder_layers",
            "resampler_layers",
            "resampler_heads",
        )
        for field_name in positive_fields:
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer, got {value!r}")
        if self.primitive_dim % self.resampler_heads != 0:
            raise ValueError(
                "primitive_dim must be divisible by resampler_heads; "
                f"got {self.primitive_dim} and {self.resampler_heads}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not self.view_types:
            raise ValueError("view_types must not be empty")
        if len(set(self.view_types)) != len(self.view_types):
            raise ValueError(f"view_types must be unique, got {self.view_types!r}")
        if any(not isinstance(name, str) or not name for name in self.view_types):
            raise ValueError("every view type must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["view_types"] = list(self.view_types)
        return output

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "PrimitiveEncoderConfig":
        if not isinstance(values, dict):
            raise TypeError(f"primitive configuration must be a dictionary, got {type(values)}")
        return cls(**values)


@dataclass(frozen=True)
class PrimitiveBatch:
    """Padded tensor contract consumed by :class:`PrimitiveEncoder`.

    Shapes:

    * ``sample_features``: ``[B, V, N, S, C]`` floating point features.
    * ``sample_mask``: ``[B, V, N, S]`` valid-sample mask.
    * ``primitive_mask``: ``[B, V, N]`` valid-primitive mask.
    * ``primitive_type_ids``: ``[B, V, N]`` upstream DXF type IDs.
    * ``primitive_group_ids``: optional ``[B, V, N]`` sample-local group IDs;
      ``-1`` means ungrouped.
    * ``view_type_ids``: ``[B, V]`` semantic IDs indexing the configured view
      vocabulary, independent of physical view-slot order.
    * ``view_mask``: ``[B, V]`` valid-view mask.

    Masks use ``True`` for real values. Valid samples must be a contiguous
    prefix so reversal keeps padding at the end. Group IDs may span views of one
    sample but are never interpreted across different batch items.
    """

    sample_features: torch.Tensor
    sample_mask: torch.Tensor
    primitive_mask: torch.Tensor
    primitive_type_ids: torch.Tensor
    view_type_ids: torch.Tensor
    view_mask: torch.Tensor
    primitive_group_ids: torch.Tensor | None = None

    @property
    def batch_size(self) -> int:
        if self.sample_features.ndim == 0:
            raise ValueError("sample_features must have a batch dimension")
        return self.sample_features.shape[0]

    def to(
        self,
        device: torch.device | str,
        *,
        dtype: torch.dtype | None = None,
    ) -> "PrimitiveBatch":
        return replace(
            self,
            sample_features=self.sample_features.to(device=device, dtype=dtype),
            sample_mask=self.sample_mask.to(device=device),
            primitive_mask=self.primitive_mask.to(device=device),
            primitive_type_ids=self.primitive_type_ids.to(device=device),
            primitive_group_ids=(
                None
                if self.primitive_group_ids is None
                else self.primitive_group_ids.to(device=device)
            ),
            view_type_ids=self.view_type_ids.to(device=device),
            view_mask=self.view_mask.to(device=device),
        )

    def repeat_interleave(self, repeats: int) -> "PrimitiveBatch":
        """Repeat batch items for beam/sample expansion."""
        if repeats <= 0:
            raise ValueError(f"repeats must be positive, got {repeats}")

        def repeat(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value.repeat_interleave(repeats, dim=0)

        return PrimitiveBatch(
            sample_features=repeat(self.sample_features),
            sample_mask=repeat(self.sample_mask),
            primitive_mask=repeat(self.primitive_mask),
            primitive_type_ids=repeat(self.primitive_type_ids),
            primitive_group_ids=repeat(self.primitive_group_ids),
            view_type_ids=repeat(self.view_type_ids),
            view_mask=repeat(self.view_mask),
        )


class _MaskedResidualConvBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_values = mask.unsqueeze(-1)
        normalized = self.norm(inputs).masked_fill(~mask_values, 0.0)
        update = self.conv(normalized.transpose(1, 2)).transpose(1, 2)
        output = inputs + self.activation(update)
        return output.masked_fill(~mask_values, 0.0)


class _CurveEncoder(nn.Module):
    """UV-Net-inspired shared 1D encoder with masked reversal symmetry."""

    def __init__(self, input_dim: int, dim: int, num_layers: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, dim)
        self.blocks = nn.ModuleList(
            [_MaskedResidualConvBlock(dim) for _ in range(num_layers)]
        )
        self.pool_projection = nn.Linear(2 * dim, dim)

    @staticmethod
    def reverse_valid_samples(
        sample_features: torch.Tensor,
        sample_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid_counts = sample_mask.sum(dim=-1, keepdim=True)
        positions = torch.arange(
            sample_features.shape[1], device=sample_features.device
        ).unsqueeze(0)
        reverse_positions = valid_counts - 1 - positions
        gather_positions = torch.where(
            sample_mask,
            reverse_positions,
            positions,
        ).clamp_min(0)
        gather_positions = gather_positions.unsqueeze(-1).expand_as(sample_features)
        return sample_features.gather(1, gather_positions), sample_mask

    def _encode_sequence(
        self,
        sample_features: torch.Tensor,
        sample_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask_values = sample_mask.unsqueeze(-1)
        hidden = self.input_projection(sample_features).masked_fill(~mask_values, 0.0)
        for block in self.blocks:
            hidden = block(hidden, sample_mask)

        valid_counts = sample_mask.sum(dim=-1, keepdim=True).to(hidden.dtype)
        mean_pooled = hidden.sum(dim=1) / valid_counts
        finite_min = torch.finfo(hidden.dtype).min
        max_pooled = hidden.masked_fill(~mask_values, finite_min).max(dim=1).values
        return self.pool_projection(torch.cat((mean_pooled, max_pooled), dim=-1))

    def forward(
        self,
        sample_features: torch.Tensor,
        sample_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.all(sample_mask.any(dim=-1)):
            raise ValueError("every encoded primitive must contain at least one valid sample")
        reversed_features, reversed_mask = self.reverse_valid_samples(
            sample_features, sample_mask
        )
        forward = self._encode_sequence(sample_features, sample_mask)
        backward = self._encode_sequence(reversed_features, reversed_mask)
        return 0.5 * (forward + backward)


class _GroupContext(nn.Module):
    """Mean-pool arbitrary sample-local groups without embedding group IDs."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.context_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.context_projection = nn.Linear(dim, dim)
        self.output_norm = nn.LayerNorm(dim)

    def forward(
        self,
        primitive_tokens: torch.Tensor,
        primitive_mask: torch.Tensor,
        group_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if group_ids is None or not torch.any((group_ids >= 0) & primitive_mask):
            return primitive_tokens

        output = primitive_tokens.clone()
        batch_size = primitive_tokens.shape[0]
        for batch_index in range(batch_size):
            grouped = primitive_mask[batch_index] & (group_ids[batch_index] >= 0)
            if not torch.any(grouped):
                continue
            tokens = primitive_tokens[batch_index][grouped]
            ids = group_ids[batch_index][grouped]
            _, inverse = torch.unique(ids, sorted=True, return_inverse=True)
            group_count = int(inverse.max().item()) + 1
            contexts = tokens.new_zeros((group_count, tokens.shape[-1]))
            contexts.index_add_(0, inverse, tokens)
            counts = torch.bincount(inverse, minlength=group_count).to(tokens.dtype)
            contexts = contexts / counts.unsqueeze(-1)
            contexts = self.context_projection(self.context_mlp(contexts))
            output[batch_index][grouped] = self.output_norm(
                tokens + contexts[inverse]
            )
        return output


class PrimitiveEncoder(nn.Module):
    """Encode padded primitives and emit fixed-K Qwen-width tokens per active view."""

    def __init__(
        self,
        config: PrimitiveEncoderConfig,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.output_dim = output_dim or config.primitive_dim
        dim = config.primitive_dim

        self.curve_encoder = _CurveEncoder(
            config.sample_feature_dim,
            dim,
            config.primitive_encoder_layers,
        )
        self.primitive_type_embedding = nn.Embedding(config.num_primitive_types, dim)
        self.view_embedding = nn.Embedding(len(config.view_types), dim)
        self.local_feature_norm = nn.LayerNorm(dim)
        self.group_context = _GroupContext(dim) if config.use_group_context else None
        self.resampler = PerceiverResampler(
            dim=dim,
            num_latents=config.num_primitive_latents,
            num_layers=config.resampler_layers,
            num_heads=config.resampler_heads,
            dropout=config.dropout,
        )
        self.primitive_modality_embedding = nn.Parameter(torch.empty(dim))
        self.output_projection = nn.Linear(dim, self.output_dim)
        nn.init.normal_(self.primitive_modality_embedding, std=dim**-0.5)

    def reset_parameters(self) -> None:
        """Initialize every added parameter after loading a base Qwen checkpoint.

        Hugging Face constructs models under a no-initialization context during
        ``from_pretrained``. Standard missing ``Linear`` weights are recovered by
        its loader, but compound PyTorch modules and bare learned parameters are
        not guaranteed to be. The base-checkpoint factory calls this method only
        after it has verified that every missing key belongs to this encoder.
        """
        for module in self.modules():
            if isinstance(module, nn.MultiheadAttention):
                module._reset_parameters()
            elif isinstance(
                module,
                (nn.Conv1d, nn.Embedding, nn.LayerNorm, nn.Linear),
            ):
                module.reset_parameters()
        nn.init.normal_(
            self.resampler.queries,
            std=self.config.primitive_dim**-0.5,
        )
        nn.init.normal_(
            self.primitive_modality_embedding,
            std=self.config.primitive_dim**-0.5,
        )

    def _validate_batch(self, batch: PrimitiveBatch) -> tuple[int, int, int, int]:
        fields = {
            "sample_features": batch.sample_features,
            "sample_mask": batch.sample_mask,
            "primitive_mask": batch.primitive_mask,
            "primitive_type_ids": batch.primitive_type_ids,
            "view_type_ids": batch.view_type_ids,
            "view_mask": batch.view_mask,
        }
        if batch.primitive_group_ids is not None:
            fields["primitive_group_ids"] = batch.primitive_group_ids
        for name, value in fields.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor, got {type(value)}")

        if batch.sample_features.ndim != 5:
            raise ValueError(
                "sample_features must have shape [B, V, N, S, C], got "
                f"{tuple(batch.sample_features.shape)}"
            )
        batch_size, view_count, primitive_count, sample_count, feature_dim = (
            batch.sample_features.shape
        )
        if min(batch_size, view_count, primitive_count, sample_count) <= 0:
            raise ValueError("B, V, N, and S dimensions must all be positive")
        if feature_dim != self.config.sample_feature_dim:
            raise ValueError(
                f"expected sample feature dim {self.config.sample_feature_dim}, "
                f"got {feature_dim}"
            )
        if not batch.sample_features.is_floating_point():
            raise TypeError(
                f"sample_features must be floating point, got {batch.sample_features.dtype}"
            )

        expected_shapes = {
            "sample_mask": (batch_size, view_count, primitive_count, sample_count),
            "primitive_mask": (batch_size, view_count, primitive_count),
            "primitive_type_ids": (batch_size, view_count, primitive_count),
            "view_type_ids": (batch_size, view_count),
            "view_mask": (batch_size, view_count),
        }
        if batch.primitive_group_ids is not None:
            expected_shapes["primitive_group_ids"] = (
                batch_size,
                view_count,
                primitive_count,
            )
        for name, shape in expected_shapes.items():
            actual = getattr(batch, name)
            if actual.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(actual.shape)}")

        for name in ("sample_mask", "primitive_mask", "view_mask"):
            value = getattr(batch, name)
            if value.dtype != torch.bool:
                raise TypeError(f"{name} must be a BoolTensor, got {value.dtype}")
        for name in ("primitive_type_ids", "view_type_ids"):
            value = getattr(batch, name)
            if value.dtype != torch.long:
                raise TypeError(f"{name} must be a LongTensor, got {value.dtype}")
        if (
            batch.primitive_group_ids is not None
            and batch.primitive_group_ids.dtype != torch.long
        ):
            raise TypeError(
                "primitive_group_ids must be a LongTensor, got "
                f"{batch.primitive_group_ids.dtype}"
            )

        devices = {value.device for value in fields.values()}
        if len(devices) != 1:
            raise ValueError(f"all primitive tensors must share one device, got {devices}")

        if torch.any(batch.primitive_mask & ~batch.view_mask.unsqueeze(-1)):
            raise ValueError("absent view slots must not contain active primitives")
        if torch.any(batch.sample_mask & ~batch.primitive_mask.unsqueeze(-1)):
            raise ValueError("inactive primitive slots must not contain valid samples")
        active_sample_counts = batch.sample_mask.sum(dim=-1)
        if torch.any(batch.primitive_mask & (active_sample_counts == 0)):
            raise ValueError("every active primitive must contain at least one valid sample")
        active_primitive_counts = batch.primitive_mask.sum(dim=-1)
        if torch.any(batch.view_mask & (active_primitive_counts == 0)):
            raise ValueError("every active primitive view must contain at least one primitive")

        positions = torch.arange(sample_count, device=batch.sample_mask.device)
        prefix_mask = positions < active_sample_counts.unsqueeze(-1)
        if not torch.equal(batch.sample_mask, prefix_mask & batch.primitive_mask.unsqueeze(-1)):
            raise ValueError("valid samples must form a contiguous prefix with padding at the end")

        active_types = batch.primitive_type_ids[batch.primitive_mask]
        if torch.any(active_types < 0) or torch.any(
            active_types >= self.config.num_primitive_types
        ):
            raise ValueError(
                "active primitive_type_ids must lie in [0, "
                f"{self.config.num_primitive_types})"
            )
        active_views = batch.view_type_ids[batch.view_mask]
        if torch.any(active_views < 0) or torch.any(
            active_views >= len(self.config.view_types)
        ):
            raise ValueError(
                f"active view_type_ids must lie in [0, {len(self.config.view_types)})"
            )
        for sample_index in range(batch_size):
            ids = batch.view_type_ids[sample_index][batch.view_mask[sample_index]]
            if ids.numel() != torch.unique(ids).numel():
                raise ValueError(
                    f"active view_type_ids must be unique within sample {sample_index}"
                )
        if batch.primitive_group_ids is not None and torch.any(
            batch.primitive_group_ids < -1
        ):
            raise ValueError("primitive_group_ids must be -1 or non-negative")
        return batch_size, view_count, primitive_count, sample_count

    def encode_local(self, batch: PrimitiveBatch) -> torch.Tensor:
        """Return masked local primitive tokens with shape ``[B, V, N, D]``."""
        batch_size, view_count, primitive_count, _ = self._validate_batch(batch)
        dim = self.config.primitive_dim
        output = batch.sample_features.new_zeros(
            (batch_size, view_count, primitive_count, dim)
        )
        if not torch.any(batch.primitive_mask):
            return output

        active_features = batch.sample_features[batch.primitive_mask]
        active_sample_mask = batch.sample_mask[batch.primitive_mask]
        geometry = self.curve_encoder(active_features, active_sample_mask)
        primitive_types = self.primitive_type_embedding(
            batch.primitive_type_ids[batch.primitive_mask]
        )
        expanded_views = batch.view_type_ids.unsqueeze(-1).expand_as(
            batch.primitive_type_ids
        )
        views = self.view_embedding(expanded_views[batch.primitive_mask])
        output[batch.primitive_mask] = self.local_feature_norm(
            geometry + primitive_types + views
        )
        return output

    def apply_group_context(
        self,
        primitive_tokens: torch.Tensor,
        batch: PrimitiveBatch,
    ) -> torch.Tensor:
        """Broadcast sample-local group context, leaving ungrouped tokens unchanged."""
        if self.group_context is None:
            return primitive_tokens
        return self.group_context(
            primitive_tokens,
            batch.primitive_mask,
            batch.primitive_group_ids,
        )

    def resample_views(
        self,
        primitive_tokens: torch.Tensor,
        batch: PrimitiveBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``[active_views, Kp, D]`` latents in canonical sample/view order."""
        tokens: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        view_ids: list[torch.Tensor] = []
        active_view_counts: list[int] = []

        for batch_index in range(batch.sample_features.shape[0]):
            slots = torch.nonzero(batch.view_mask[batch_index], as_tuple=False).squeeze(-1)
            if slots.numel():
                semantic_ids = batch.view_type_ids[batch_index, slots]
                slots = slots[torch.argsort(semantic_ids)]
                for slot in slots:
                    tokens.append(primitive_tokens[batch_index, slot])
                    masks.append(batch.primitive_mask[batch_index, slot])
                    view_ids.append(batch.view_type_ids[batch_index, slot])
            active_view_counts.append(int(slots.numel()))

        counts = torch.tensor(
            active_view_counts,
            dtype=torch.long,
            device=batch.sample_features.device,
        ) * self.config.num_primitive_latents
        if not tokens:
            return primitive_tokens.new_empty(
                (0, self.config.num_primitive_latents, self.config.primitive_dim)
            ), counts

        token_batch = torch.stack(tokens)
        mask_batch = torch.stack(masks)
        view_id_batch = torch.stack(view_ids)
        view_embeddings = self.view_embedding(view_id_batch)
        latents = self.resampler(token_batch, mask_batch, view_embeddings)
        return latents, counts

    def forward(self, batch: PrimitiveBatch) -> tuple[torch.Tensor, torch.Tensor]:
        local_tokens = self.encode_local(batch)
        grouped_tokens = self.apply_group_context(local_tokens, batch)
        view_latents, counts = self.resample_views(grouped_tokens, batch)
        if view_latents.numel() == 0:
            return view_latents.new_empty((0, self.output_dim)), counts
        view_latents = view_latents + self.primitive_modality_embedding
        projected = self.output_projection(view_latents)
        return projected.reshape(-1, self.output_dim), counts


__all__ = [
    "PrimitiveBatch",
    "PrimitiveEncoder",
    "PrimitiveEncoderConfig",
]
