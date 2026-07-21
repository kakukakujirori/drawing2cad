"""Deterministic Qwen generation and CAD scoring for model validation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import nullcontext
import json
from pathlib import Path
import random
from typing import Any, Mapping

from accelerate.utils import gather_object
import torch
from torch.utils.data import DataLoader

from src.data import (
    DXFPrimitiveConfig,
    DXFPrimitiveParser,
    Drawing2CADBatch,
    Drawing2CADCollator,
    Drawing2CADDataset,
    Drawing2CADPreprocessor,
    ManifestSampleMetadataProvider,
    RasterImageSource,
)
from src.data.audit.gate import gate_present_ids_from_config
from src.evaluation.evaluator import (
    EvaluationConfig,
    EvaluationItem,
    aggregate_evaluation_metrics,
    evaluate_predictions,
    evaluation_error_histogram,
)
from src.models import PrimitiveEncoderConfig


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class CADGenerationEvaluator:
    """Generate a fixed validation subset on rank zero and broadcast metrics."""

    def __init__(
        self,
        *,
        accelerator: Any,
        processor: Any,
        data_config: Mapping[str, Any],
        evaluation_config: Mapping[str, Any],
        primitive_config: PrimitiveEncoderConfig,
        predictions_dir: str | Path,
    ) -> None:
        self.accelerator = accelerator
        self.processor = processor
        self.data_config = data_config
        self.evaluation_config = evaluation_config
        self.primitive_config = primitive_config
        self.predictions_dir = Path(predictions_dir)
        self.sample_ids = self._select_sample_ids()
        if accelerator.is_main_process:
            self.predictions_dir.mkdir(parents=True, exist_ok=True)
            _atomic_json(
                self.predictions_dir / "subset.json",
                {
                    "seed": int(self.evaluation_config.get("generation_seed", 42)),
                    "sample_ids": list(self.sample_ids),
                },
            )
        accelerator.wait_for_everyone()

    def _select_sample_ids(self) -> tuple[str, ...]:
        provider = ManifestSampleMetadataProvider(
            Path(self.data_config["val_root"]) / "manifest.jsonl"
        )
        available = list(provider.sample_ids)
        # Gate the benchmark on the same GT-audit allow policy as training, so a
        # broken/mislabeled GT solid never contributes a meaningless IoU target.
        audit_config = self.data_config.get("audit")
        if audit_config:
            allowed = gate_present_ids_from_config(
                audit_config, "val_dir", available, context="val benchmark"
            )
            available = [sample_id for sample_id in available if sample_id in allowed]
            if not available:
                raise ValueError(
                    "no validation samples pass the audit allow-list "
                    "(is data.audit.val_dir correct and audited?)"
                )
        requested = int(
            self.evaluation_config.get("generation_subset_size", len(available))
        )
        if requested <= 0 or requested >= len(available):
            return tuple(available)
        generator = random.Random(
            int(self.evaluation_config.get("generation_seed", 42))
        )
        # Sort after sampling so execution and artifacts have a stable order.
        return tuple(sorted(generator.sample(available, requested)))

    def _shard_sample_ids(self) -> tuple[str, ...]:
        """Return this rank's disjoint, deterministic slice of the subset.

        A strided split keeps every rank's shard a contiguous stride of the
        globally sorted ``sample_ids``, so the union is exactly the subset for
        any world size and the per-rank sizes differ by at most one.
        """

        index = int(self.accelerator.process_index)
        stride = int(self.accelerator.num_processes)
        return tuple(self.sample_ids[index::stride])

    def _loader(self, sample_ids: Sequence[str]) -> DataLoader:
        dxf_config = DXFPrimitiveConfig(**self.data_config["dxf"])
        image_sources = tuple(
            RasterImageSource(str(item["style"]), str(item["directory"]))
            for item in self.data_config["image_sources"]
        )
        # Generation is forward-only, so a long prompt does not carry the
        # training memory cost that motivates max_sequence_length filtering.
        # Leave it unset here: the validation benchmark must be scored over
        # every sample, not silently pruned to the training length budget.
        preprocessor = Drawing2CADPreprocessor(
            self.processor,
            self.primitive_config.num_primitive_latents,
            include_labels=False,
            max_length=None,
        )
        dataset = Drawing2CADDataset(
            self.data_config["val_root"],
            dxf_parser=DXFPrimitiveParser(dxf_config),
            image_sources=image_sources,
            include_target=False,
            sample_ids=tuple(sample_ids),
            strict_files=bool(self.data_config.get("strict_files", True)),
            image_max_edge=self.data_config.get("image_max_edge"),
            transform=preprocessor,
        )
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("processor tokenizer has no pad token ID")
        # Decoder-only batched generation requires left padding.
        collator = Drawing2CADCollator(int(pad_token_id), padding_side="left")
        workers = int(self.data_config.get("num_workers", 0))
        return DataLoader(
            dataset,
            batch_size=int(self.data_config.get("val_batch_size", 1)),
            shuffle=False,
            num_workers=workers,
            pin_memory=bool(self.data_config.get("pin_memory", True)),
            persistent_workers=(
                bool(self.data_config.get("persistent_workers", False))
                if workers > 0
                else False
            ),
            collate_fn=collator,
        )

    def _evaluation_config(self) -> EvaluationConfig:
        return EvaluationConfig(
            timeout_s=float(
                self.evaluation_config.get("cadquery_timeout_seconds", 30.0)
            ),
            tessellation_tolerance=float(
                self.evaluation_config.get("tessellation_tolerance", 0.1)
            ),
            volume_tolerance=float(
                self.evaluation_config.get("volume_tolerance", 1e-6)
            ),
            voxel_resolution=int(self.evaluation_config.get("voxel_resolution", 64)),
            compute_normalized_iou=bool(
                self.evaluation_config.get("compute_normalized_iou", False)
            ),
            compute_chamfer=bool(self.evaluation_config.get("compute_chamfer", False)),
            chamfer_points=int(self.evaluation_config.get("chamfer_points", 8192)),
            chamfer_seed=int(self.evaluation_config.get("generation_seed", 42)),
        )

    def _generate_codes(
        self,
        model: torch.nn.Module,
        sample_ids: Sequence[str],
        *,
        on_generated: Callable[[int], None] | None = None,
    ) -> dict[str, str]:
        """Greedily decode CadQuery source for one shard of sample IDs."""

        if not sample_ids:
            return {}
        unwrapped = self.accelerator.unwrap_model(model)
        was_training = unwrapped.training
        unwrapped.eval()
        codes: dict[str, str] = {}
        try:
            for batch in self._loader(sample_ids):
                if not isinstance(batch, Drawing2CADBatch):
                    raise TypeError("generation loader must emit Drawing2CADBatch")
                inputs = batch.to(self.accelerator.device).model_inputs
                prompt_length = int(inputs["input_ids"].shape[1])
                with torch.inference_mode():
                    generated = unwrapped.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=int(
                            self.evaluation_config.get("max_new_tokens", 1024)
                        ),
                        use_cache=True,
                    )
                completions = generated[:, prompt_length:]
                decoded = self.processor.batch_decode(
                    completions,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                codes.update(zip(batch.sample_ids, decoded, strict=True))
                if on_generated is not None:
                    on_generated(len(decoded))
        finally:
            unwrapped.train(was_training)
        return codes

    def _write_and_evaluate(
        self,
        codes: Mapping[str, str],
        sample_ids: Sequence[str],
        step_dir: Path,
        *,
        on_evaluated: Callable[[int], None] | None = None,
    ) -> list[dict[str, object]]:
        """Persist this shard's completions and score them against targets."""

        step_dir.mkdir(parents=True, exist_ok=True)
        for sample_id in sample_ids:
            (step_dir / f"{sample_id}.cadquery.py").write_text(
                codes[sample_id], encoding="utf-8"
            )
        target_root = Path(self.data_config["val_root"]) / "target"
        items = [
            EvaluationItem(
                sample_id=sample_id,
                prediction=codes[sample_id],
                target_step_path=target_root / f"{sample_id}.step",
            )
            for sample_id in sample_ids
        ]
        return evaluate_predictions(
            items,
            config=self._evaluation_config(),
            on_item=on_evaluated,
            max_workers=int(self.evaluation_config.get("execution_workers", 1)),
        )

    def __call__(
        self,
        model: torch.nn.Module,
        step: int,
        *,
        progress_bar: Any = None,
    ) -> dict[str, float | int]:
        # Every rank generates and scores a disjoint shard in parallel, then all
        # ranks all-gather the per-sample rows. Because the gathered rows are
        # re-sorted into a single deterministic order, every rank aggregates
        # identical metrics without a separate broadcast, and the result is
        # independent of world size.
        step_dir = self.predictions_dir / f"step_{step:08d}"
        self.accelerator.wait_for_everyone()
        shard_ids = self._shard_sample_ids()

        # A no-op factory keeps the call sites branch-free when no bar is wired
        # (non-main ranks pass a disabled bar whose sub_task is already a no-op).
        def _null_sub_task(_description: str, _total: int) -> Any:
            return nullcontext(lambda _steps=1: None)

        sub_task = progress_bar.sub_task if progress_bar is not None else _null_sub_task
        with sub_task("eval generate", len(shard_ids)) as advance:
            codes = self._generate_codes(model, shard_ids, on_generated=advance)
        with sub_task("eval execute", len(shard_ids)) as advance:
            shard_rows = self._write_and_evaluate(
                codes, shard_ids, step_dir, on_evaluated=advance
            )

        gathered = gather_object(shard_rows)
        rows: list[dict[str, object]] = []
        for entry in gathered:
            # gather_object concatenates list inputs, but flatten defensively in
            # case an implementation nests each rank's list.
            if isinstance(entry, list):
                rows.extend(entry)
            else:
                rows.append(entry)
        rows.sort(key=lambda row: str(row.get("id", "")))

        metrics = aggregate_evaluation_metrics(rows, prefix="val")
        if self.accelerator.is_main_process:
            _atomic_json(step_dir / "rows.json", rows)
            _atomic_json(step_dir / "metrics.json", metrics)
            _atomic_json(
                step_dir / "error_histogram.json", evaluation_error_histogram(rows)
            )
        self.accelerator.wait_for_everyone()
        return metrics


__all__ = ["CADGenerationEvaluator"]
