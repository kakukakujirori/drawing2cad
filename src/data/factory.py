"""Construct SFT datasets and dataloaders from the data config boundary."""

from __future__ import annotations

import os
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.models import PrimitiveEncoderConfig
from src.utils import seed_worker

from .audit.gate import gate_present_ids_from_config
from .collator import Drawing2CADCollator
from .dataset import Drawing2CADDataset, RasterImageSource
from .preprocessing import Drawing2CADPreprocessor
from .dxf import (
    DXF_ORIENTED_SAMPLE_FEATURE_INDICES,
    DXFPrimitiveConfig,
    DXFPrimitiveParser,
)
from .preprocessing import Drawing2CADPreprocessor


@dataclass(frozen=True)
class SFTDataLoaders:
    train: DataLoader
    validation: DataLoader
    generator: torch.Generator


def _filter_by_length(
    dataset: Drawing2CADDataset,
    preprocessor: Drawing2CADPreprocessor,
    max_length: int,
    *,
    split: str,
    show_progress: bool,
) -> Subset:
    """Drop samples whose fully rendered sequence exceeds ``max_length``.

    Only the target text varies in length: the chat template, instruction,
    image headings, the fixed-size render's image tokens, and the primitive
    placeholders form a constant overhead. So the length is estimated as that
    overhead (measured once from a reference sample) plus the target token
    count, which avoids per-sample image decoding, DXF parsing, and image
    processing -- roughly two orders of magnitude faster than an exact scan.
    The estimate matches the exact length to within about one token and errs
    high, so it never admits an overlength sample.

    Runs identically on every rank (deterministic), so all ranks build the same
    filtered dataset without any rank waiting on a collective while another
    scans the corpus. Overlength samples are removed, never truncated.
    """
    total = len(dataset)
    reference = dataset.load_sample(0)
    if reference.target_code is None:
        raise ValueError("length filtering requires targets (include_target=True)")
    overhead = preprocessor.sequence_length(reference) - preprocessor.target_token_count(
        reference.target_code
    )
    kept: list[int] = []
    # The scan reads one small target file per sample; with a cold OS page cache
    # each read is a random disk seek (~10 ms) and the corpus has 100k+ files, so
    # a serial scan is dominated by seek latency. Issue the reads concurrently to
    # let the disk pipeline them (they are latency-, not CPU-bound); the cheap
    # tokenization stays on this thread because fast tokenizers are not guaranteed
    # thread-safe. ``map`` preserves input order, so enumeration recovers the index.
    io_workers = min(32, (os.cpu_count() or 4) * 4)
    with ThreadPoolExecutor(max_workers=io_workers) as pool:
        target_codes = pool.map(dataset.target_code, range(total))
        for index, target_code in enumerate(target_codes):
            length = overhead + preprocessor.target_token_count(target_code)
            if length <= max_length:
                kept.append(index)
            done = index + 1
            if show_progress and (done % 10000 == 0 or done == total):
                print(
                    f"[data:{split}] length filter {done}/{total} "
                    f"(kept {len(kept)}, dropped {done - len(kept)}, "
                    f"overhead={overhead}, max_length={max_length})",
                    flush=True,
                )
    if not kept:
        raise ValueError(
            f"[data:{split}] every sample exceeds max_sequence_length={max_length}"
        )
    return Subset(dataset, kept)


def _record_sample_ids(dataset: Drawing2CADDataset | Subset) -> list[str]:
    """Sample id per positional index, for a raw dataset or a ``Subset`` of one.

    ``_filter_by_length`` may already have wrapped the dataset in a ``Subset``
    (which has no ``.records``), so resolve ids through the underlying dataset
    when needed. This keeps the two filters composable in either order.
    """
    if isinstance(dataset, Subset):
        base: Drawing2CADDataset = dataset.dataset  # type: ignore[assignment]
        return [base.records[i].sample_id for i in dataset.indices]
    return [record.sample_id for record in dataset.records]


def _filter_by_audit(
    dataset: Drawing2CADDataset | Subset,
    allowed_ids: set[str],
    *,
    split: str,
    show_progress: bool,
) -> Subset:
    """Keep only samples whose uuid passes the GT-audit allow policy.

    ``allowed_ids`` is the verdict-derived allow-set (see
    ``src.data.audit.gate``); it is intersected with the samples actually
    present, so a clean verdict for a sample that never rendered is simply
    absent here. Fail-closed: an empty intersection is an error, not a silent
    empty corpus (it almost always means a wrong/missing audit directory).
    """
    kept = [
        index
        for index, sample_id in enumerate(_record_sample_ids(dataset))
        if sample_id in allowed_ids
    ]
    if not kept:
        raise ValueError(
            f"[data:{split}] no samples pass the audit allow-list "
            f"(is the audit dir correct and audited?)"
        )
    if show_progress:
        total = len(dataset)
        print(
            f"[data:{split}] audit gate kept {len(kept)}/{total} "
            f"(dropped {total - len(kept)})",
            flush=True,
        )
    return Subset(dataset, kept)


def build_sft_dataloaders(
    data_config: Mapping[str, Any],
    *,
    processor: Any,
    primitive_config: PrimitiveEncoderConfig,
    seed: int,
    show_progress: bool = True,
) -> SFTDataLoaders:
    if bool(data_config.get("scale_augmentation", False)):
        raise ValueError("scale augmentation is not implemented for the SFT baseline")
    dxf_config = DXFPrimitiveConfig(**data_config["dxf"])
    if dxf_config.sample_feature_dim != primitive_config.sample_feature_dim:
        raise ValueError("DXF and primitive encoder sample feature dimensions differ")
    if primitive_config.oriented_feature_indices != DXF_ORIENTED_SAMPLE_FEATURE_INDICES:
        raise ValueError(
            "model oriented_feature_indices "
            f"{primitive_config.oriented_feature_indices} must match the DXF "
            f"oriented sample channels {DXF_ORIENTED_SAMPLE_FEATURE_INDICES}"
        )
    image_sources = tuple(
        RasterImageSource(str(item["style"]), str(item["directory"]))
        for item in data_config["image_sources"]
    )
    preprocessor = Drawing2CADPreprocessor(
        processor,
        primitive_config.num_primitive_latents,
        include_labels=True,
        max_length=data_config.get("max_sequence_length"),
    )

    def dataset(root_key: str, max_key: str) -> Drawing2CADDataset:
        return Drawing2CADDataset(
            data_config[root_key],
            dxf_parser=DXFPrimitiveParser(dxf_config),
            image_sources=image_sources,
            include_target=True,
            max_samples=data_config.get(max_key),
            strict_files=bool(data_config.get("strict_files", True)),
            image_max_edge=data_config.get("image_max_edge"),
            transform=preprocessor,
        )

    audit_config = data_config.get("audit")

    def build_split(
        root_key: str, max_key: str, split: str, audit_dir_key: str
    ) -> Dataset:
        full = dataset(root_key, max_key)
        result: Drawing2CADDataset | Subset = full
        max_sequence_length = data_config.get("max_sequence_length")
        if max_sequence_length is not None:
            result = _filter_by_length(
                full,
                preprocessor,
                int(max_sequence_length),
                split=split,
                show_progress=show_progress,
            )
        if audit_config:
            kept_ids = gate_present_ids_from_config(
                audit_config,
                audit_dir_key,
                _record_sample_ids(result),
                context=f"data:{split}",
            )
            result = _filter_by_audit(
                result, kept_ids, split=split, show_progress=show_progress
            )
        return result

    train_dataset = build_split("train_root", "train_max_samples", "train", "train_dir")
    validation_dataset = build_split("val_root", "val_max_samples", "val", "val_dir")
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("processor tokenizer has no pad token ID")
    collator = Drawing2CADCollator(
        int(pad_token_id), padding_side=processor.tokenizer.padding_side
    )
    generator = torch.Generator().manual_seed(seed)
    workers = int(data_config.get("num_workers", 0))
    common = {
        "num_workers": workers,
        "pin_memory": bool(data_config.get("pin_memory", True)),
        "persistent_workers": (
            bool(data_config.get("persistent_workers", False)) if workers > 0 else False
        ),
        "worker_init_fn": seed_worker,
        "collate_fn": collator,
    }
    train = DataLoader(
        train_dataset,
        batch_size=int(data_config.get("train_batch_size", 1)),
        shuffle=bool(data_config.get("shuffle_train", True)),
        drop_last=bool(data_config.get("drop_last", False)),
        generator=generator,
        **common,
    )
    validation = DataLoader(
        validation_dataset,
        batch_size=int(data_config.get("val_batch_size", 1)),
        shuffle=False,
        drop_last=False,
        **common,
    )
    return SFTDataLoaders(train=train, validation=validation, generator=generator)


__all__ = ["SFTDataLoaders", "build_sft_dataloaders"]
