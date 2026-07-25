"""Construct SFT datasets and dataloaders from the data config boundary.

Selection happens before construction: :func:`select_sample_ids` narrows a root
to the ids that are complete on disk and pass the GT-audit policy, the optional
length filter narrows that list further, and the dataset is then built once from
the result. Nothing downstream has to re-derive which samples are in play.

``select_sample_ids``, :func:`build_dataset` and :func:`build_collator` are also
what ``src/evaluation/generate.py`` and ``src/test.py`` assemble their loaders
from, so training, in-run validation and offline testing cannot drift apart on
which samples exist or which audit policy applies to them.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from src.models import PrimitiveEncoderConfig
from src.utils import seed_worker

from .audit.gate import gate_present_ids_from_config
from .collator import Drawing2CADCollator
from .dataset import Drawing2CADDataset, Drawing2CADSample, RasterImageSource
from .layout import DatasetRoot, code_target_path, resolve_dataset_roots, take_inventory
from .preprocessing import Drawing2CADPreprocessor
from .dxf import (
    DXF_ORIENTED_SAMPLE_FEATURE_INDICES,
    DXFPrimitiveConfig,
    DXFPrimitiveParser,
)


@dataclass(frozen=True)
class SFTDataLoaders:
    """Loaders for one SFT run.

    ``train`` is a single loader over every configured training root (they are
    concatenated and shuffled together) because the loop's epoch/step budget,
    resume offset, and token-weighted accumulation are all defined against one
    stream. ``validation`` stays split per dataset root, keyed by root name, so
    each contributes its own ``val/<name>/loss``. Roots without CadQuery targets
    have no labels and so are absent here; they are scored by generation only.
    """

    train: DataLoader
    validation: Mapping[str, DataLoader]
    generator: torch.Generator


def image_sources_from_config(
    data_config: Mapping[str, Any],
) -> tuple[RasterImageSource, ...]:
    """The rasters every sample must carry, as configured under ``image_sources``."""
    return tuple(
        RasterImageSource(str(item["style"]), str(item["directory"]))
        for item in data_config["image_sources"]
    )


def select_sample_ids(
    root: DatasetRoot,
    data_config: Mapping[str, Any],
    *,
    require_target: bool,
    context: str,
    show_progress: bool = True,
) -> tuple[str, ...]:
    """Ids of ``root`` that are complete on disk and pass the GT-audit policy.

    Both halves of the question are answered here so every consumer agrees on
    them: which samples the renderer actually produced (an incomplete sample is
    reported and dropped -- see ``src.data.layout``), and which of those the
    audit policy admits (fail-closed -- see ``src.data.audit.gate``).
    """
    ids, incomplete = take_inventory(
        root.path,
        image_dirs=[
            source.directory for source in image_sources_from_config(data_config)
        ],
        require_target=require_target,
    )
    if incomplete and show_progress:
        example_id, missing = next(iter(incomplete.items()))
        print(
            f"[{context}] skipped {len(incomplete)}/{len(ids) + len(incomplete)} "
            f"incomplete samples (e.g. {example_id} is missing {missing})",
            flush=True,
        )
    if not ids:
        raise ValueError(
            f"[{context}] {root.path} yields no complete samples "
            f"({len(incomplete)} drawing(s) found); has it been rendered?"
        )
    audit_config = data_config.get("audit")
    if not audit_config:
        return ids
    allowed = gate_present_ids_from_config(
        audit_config, root.audit_dir, ids, context=context
    )
    gated = tuple(sample_id for sample_id in ids if sample_id in allowed)
    if not gated:
        raise ValueError(
            f"[{context}] no samples pass the audit allow-list "
            f"(is {root.audit_dir} audited?)"
        )
    if show_progress:
        print(
            f"[{context}] audit gate kept {len(gated)}/{len(ids)} "
            f"(dropped {len(ids) - len(gated)})",
            flush=True,
        )
    return gated


def build_dataset(
    root: str | Path,
    data_config: Mapping[str, Any],
    *,
    sample_ids: Sequence[str],
    include_target: bool,
    transform: Callable[[Drawing2CADSample], Any] | None,
) -> Drawing2CADDataset:
    """One root's dataset over an already-selected id list."""
    return Drawing2CADDataset(
        root,
        sample_ids,
        dxf_parser=DXFPrimitiveParser(DXFPrimitiveConfig(**data_config["dxf"])),
        image_sources=image_sources_from_config(data_config),
        include_target=include_target,
        image_max_edge=data_config.get("image_max_edge"),
        transform=transform,
    )


def build_collator(processor: Any, *, padding_side: str) -> Drawing2CADCollator:
    """Batching collator for ``processor``'s pad token."""
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("processor tokenizer has no pad token ID")
    return Drawing2CADCollator(int(pad_token_id), padding_side=padding_side)


def _length_filtered_ids(
    root: str | Path,
    data_config: Mapping[str, Any],
    *,
    sample_ids: Sequence[str],
    preprocessor: Drawing2CADPreprocessor,
    max_length: int,
    context: str,
    show_progress: bool,
) -> tuple[str, ...]:
    """Drop ids whose fully rendered sequence would exceed ``max_length``.

    Only the target text varies in length: the chat template, instruction,
    image headings, the fixed-size render's image tokens, and the primitive
    placeholders form a constant overhead. So the length is estimated as that
    overhead (measured once from a reference sample) plus the target token
    count, which avoids per-sample image decoding, DXF parsing, and image
    processing -- roughly two orders of magnitude faster than an exact scan.
    The estimate matches the exact length to within about one token and errs
    high, so it never admits an overlength sample.

    Runs identically on every rank (deterministic), so all ranks derive the same
    id list without any rank waiting on a collective while another scans the
    corpus. Overlength samples are removed, never truncated.
    """
    probe = build_dataset(
        root,
        data_config,
        sample_ids=sample_ids[:1],
        include_target=True,
        transform=None,
    )
    reference = probe.load_sample(0)
    if reference.target_code is None:
        raise ValueError("length filtering requires targets (include_target=True)")
    overhead = preprocessor.sequence_length(
        reference
    ) - preprocessor.target_token_count(reference.target_code)
    total = len(sample_ids)
    kept: list[str] = []
    # The scan reads one small target file per sample; with a cold OS page cache
    # each read is a random disk seek (~10 ms) and the corpus has 100k+ files, so
    # a serial scan is dominated by seek latency. Issue the reads concurrently to
    # let the disk pipeline them (they are latency-, not CPU-bound); the cheap
    # tokenization stays on this thread because fast tokenizers are not guaranteed
    # thread-safe. ``map`` preserves input order, so enumeration recovers the id.
    io_workers = min(32, (os.cpu_count() or 4) * 4)
    paths = [code_target_path(root, sample_id) for sample_id in sample_ids]
    with ThreadPoolExecutor(max_workers=io_workers) as pool:
        target_codes = pool.map(lambda path: path.read_text(encoding="utf-8"), paths)
        for index, target_code in enumerate(target_codes):
            length = overhead + preprocessor.target_token_count(target_code)
            if length <= max_length:
                kept.append(sample_ids[index])
            done = index + 1
            if show_progress and (done % 10000 == 0 or done == total):
                print(
                    f"[{context}] length filter {done}/{total} "
                    f"(kept {len(kept)}, dropped {done - len(kept)}, "
                    f"overhead={overhead}, max_length={max_length})",
                    flush=True,
                )
    if not kept:
        raise ValueError(
            f"[{context}] every sample exceeds max_sequence_length={max_length}"
        )
    return tuple(kept)


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
    max_sequence_length = data_config.get("max_sequence_length")
    preprocessor = Drawing2CADPreprocessor(
        processor,
        primitive_config.num_primitive_latents,
        include_labels=True,
        max_length=max_sequence_length,
    )

    def build_root(root: DatasetRoot, max_key: str, split: str) -> Dataset:
        """Select this root's ids, then build its dataset once from them."""
        context = f"data:{split}:{root.name}"
        ids = select_sample_ids(
            root,
            data_config,
            require_target=True,
            context=context,
            show_progress=show_progress,
        )
        limit = data_config.get(max_key)
        if limit is not None:
            if int(limit) <= 0:
                raise ValueError(f"[{context}] {max_key} must be positive when set")
            ids = ids[: int(limit)]
        # After the audit gate, so no target file is read for a sample the
        # policy has already rejected.
        if max_sequence_length is not None:
            ids = _length_filtered_ids(
                root.path,
                data_config,
                sample_ids=ids,
                preprocessor=preprocessor,
                max_length=int(max_sequence_length),
                context=context,
                show_progress=show_progress,
            )
        return build_dataset(
            root.path,
            data_config,
            sample_ids=ids,
            include_target=True,
            transform=preprocessor,
        )

    train_roots = resolve_dataset_roots(data_config["train_root"], split="train")
    unlabeled = [root.name for root in train_roots if not root.has_code_targets]
    if unlabeled:
        raise ValueError(
            f"training roots {unlabeled} have no *.cadquery.py targets, so they "
            f"carry no supervision; remove them from data.train_root"
        )
    train_parts = [
        build_root(root, "train_max_samples", "train") for root in train_roots
    ]
    # Concatenating before the loader (rather than one loader per root) keeps a
    # single shuffled stream, which is what run_sft's step budget, resume
    # offset, and token-weighted accumulation are defined against.
    train_dataset: Dataset = (
        train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
    )

    val_roots = resolve_dataset_roots(data_config["val_root"], split="val")
    # Only roots shipping CadQuery sources can supply labels; a STEP-only root
    # is still scored by the generation benchmark, which needs no labels.
    validation_datasets = {
        root.name: build_root(root, "val_max_samples", "val")
        for root in val_roots
        if root.has_code_targets
    }
    collator = build_collator(processor, padding_side=processor.tokenizer.padding_side)
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
    validation = {
        name: DataLoader(
            dataset,
            batch_size=int(data_config.get("val_batch_size", 1)),
            shuffle=False,
            drop_last=False,
            **common,
        )
        for name, dataset in validation_datasets.items()
    }
    return SFTDataLoaders(train=train, validation=validation, generator=generator)


__all__ = [
    "SFTDataLoaders",
    "build_collator",
    "build_dataset",
    "build_sft_dataloaders",
    "image_sources_from_config",
    "select_sample_ids",
]
