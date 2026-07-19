"""Deterministic Qwen generation and CAD scoring for model validation."""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any, Mapping

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
from src.evaluation.evaluator import (
    EvaluationConfig,
    EvaluationItem,
    aggregate_evaluation_metrics,
    evaluate_predictions,
    evaluation_error_histogram,
)
from src.models import PrimitiveEncoderConfig


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value)}")
    return dict(value)


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
        self.data_config = _mapping(data_config, name="data")
        self.evaluation_config = _mapping(evaluation_config, name="evaluation")
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

    def _loader(self) -> DataLoader:
        dxf_config = DXFPrimitiveConfig(
            **_mapping(self.data_config["dxf"], name="data.dxf")
        )
        image_sources = tuple(
            RasterImageSource(str(item["style"]), str(item["directory"]))
            for item in self.data_config["image_sources"]
        )
        preprocessor = Drawing2CADPreprocessor(
            self.processor,
            self.primitive_config.num_primitive_latents,
            include_labels=False,
            max_length=self.data_config.get("max_sequence_length"),
        )
        dataset = Drawing2CADDataset(
            self.data_config["val_root"],
            dxf_parser=DXFPrimitiveParser(dxf_config),
            image_sources=image_sources,
            include_target=False,
            sample_ids=self.sample_ids,
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

    def _generate_on_main(
        self, model: torch.nn.Module, step: int
    ) -> dict[str, float | int]:
        unwrapped = self.accelerator.unwrap_model(model)
        was_training = unwrapped.training
        unwrapped.eval()
        codes: dict[str, str] = {}
        try:
            for batch in self._loader():
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
        finally:
            unwrapped.train(was_training)

        step_dir = self.predictions_dir / f"step_{step:08d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        for sample_id in self.sample_ids:
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
            for sample_id in self.sample_ids
        ]
        config = EvaluationConfig(
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
        rows = evaluate_predictions(items, config=config)
        metrics = aggregate_evaluation_metrics(rows, prefix="val")
        _atomic_json(step_dir / "rows.json", rows)
        _atomic_json(step_dir / "metrics.json", metrics)
        _atomic_json(
            step_dir / "error_histogram.json", evaluation_error_histogram(rows)
        )
        return metrics

    def __call__(self, model: torch.nn.Module, step: int) -> dict[str, float | int]:
        metrics: dict[str, float | int] | None = None
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            metrics = self._generate_on_main(model, step)
        if self.accelerator.num_processes > 1:
            from accelerate.utils import broadcast_object_list

            payload: list[Any] = [metrics]
            broadcast_object_list(payload, from_process=0)
            metrics = payload[0]
        self.accelerator.wait_for_everyone()
        if not isinstance(metrics, dict):
            raise RuntimeError("rank zero did not broadcast generation metrics")
        return metrics


__all__ = ["CADGenerationEvaluator"]
