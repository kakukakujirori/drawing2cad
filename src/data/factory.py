"""Construct SFT datasets and dataloaders from the data config boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.models import PrimitiveEncoderConfig
from src.utils import seed_worker

from .collator import Drawing2CADCollator
from .dataset import Drawing2CADDataset, RasterImageSource
from .dxf import DXFPrimitiveConfig, DXFPrimitiveParser
from .preprocessing import Drawing2CADPreprocessor


@dataclass(frozen=True)
class SFTDataLoaders:
    train: DataLoader
    validation: DataLoader
    generator: torch.Generator


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value)}")
    return dict(value)


def build_sft_dataloaders(
    config: Mapping[str, Any],
    *,
    processor: Any,
    primitive_config: PrimitiveEncoderConfig,
    seed: int,
) -> SFTDataLoaders:
    data = _mapping(config, name="data")
    if bool(data.get("scale_augmentation", False)):
        raise ValueError("scale augmentation is not implemented for the SFT baseline")
    dxf_config = DXFPrimitiveConfig(**_mapping(data["dxf"], name="data.dxf"))
    if dxf_config.sample_feature_dim != primitive_config.sample_feature_dim:
        raise ValueError("DXF and primitive encoder sample feature dimensions differ")
    image_sources = tuple(
        RasterImageSource(str(item["style"]), str(item["directory"]))
        for item in data["image_sources"]
    )
    preprocessor = Drawing2CADPreprocessor(
        processor,
        primitive_config.num_primitive_latents,
        include_labels=True,
        max_length=data.get("max_sequence_length"),
    )

    def dataset(root_key: str, max_key: str) -> Drawing2CADDataset:
        return Drawing2CADDataset(
            data[root_key],
            dxf_parser=DXFPrimitiveParser(dxf_config),
            image_sources=image_sources,
            include_target=True,
            max_samples=data.get(max_key),
            strict_files=bool(data.get("strict_files", True)),
            image_max_edge=data.get("image_max_edge"),
            transform=preprocessor,
        )

    train_dataset = dataset("train_root", "train_max_samples")
    validation_dataset = dataset("val_root", "val_max_samples")
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("processor tokenizer has no pad token ID")
    collator = Drawing2CADCollator(
        int(pad_token_id), padding_side=processor.tokenizer.padding_side
    )
    generator = torch.Generator().manual_seed(seed)
    workers = int(data.get("num_workers", 0))
    common = {
        "num_workers": workers,
        "pin_memory": bool(data.get("pin_memory", True)),
        "persistent_workers": (
            bool(data.get("persistent_workers", False)) if workers > 0 else False
        ),
        "worker_init_fn": seed_worker,
        "collate_fn": collator,
    }
    train = DataLoader(
        train_dataset,
        batch_size=int(data.get("train_batch_size", 1)),
        shuffle=bool(data.get("shuffle_train", True)),
        drop_last=bool(data.get("drop_last", False)),
        generator=generator,
        **common,
    )
    validation = DataLoader(
        validation_dataset,
        batch_size=int(data.get("val_batch_size", 1)),
        shuffle=False,
        drop_last=False,
        **common,
    )
    return SFTDataLoaders(train=train, validation=validation, generator=generator)


__all__ = ["SFTDataLoaders", "build_sft_dataloaders"]
