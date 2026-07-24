"""Standalone inference/scoring for a trained SFT adapter checkpoint.

Loads the LoRA + primitive-encoder adapter saved under a training run, generates
CadQuery code for every sample in ``--test_dir``, and writes one ``.cadquery.py``
and one executed ``.step`` per sample into ``--out_dir`` (mirroring the layout of
``<dataset>/target``). When ``<test_dir>/target`` exists, geometry metrics are
computed against the reference STEP files and written alongside the predictions.

Example:
    python src/test.py \
        --ckpt logs/train_sft/[yyyy-mm-dd_hh-mm-ss]/checkpoints/latest \
        --test_dir data/z2c_val \
        --out_dir outputs/z2c_val/[yyyy-mm-dd_hh-mm-ss]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping

import rootutils
import torch
from tqdm import tqdm

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ruff: noqa: E402
from omegaconf import OmegaConf
from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
from torch.utils.data import DataLoader

from src.data import (
    DXFPrimitiveConfig,
    DXFPrimitiveParser,
    Drawing2CADBatch,
    Drawing2CADCollator,
    Drawing2CADDataset,
    Drawing2CADPreprocessor,
    RasterImageSource,
    discover_sample_ids,
)
from src.evaluation.evaluator import (
    EvaluationConfig,
    EvaluationItem,
    aggregate_evaluation_metrics,
    evaluate_predictions,
    evaluation_error_histogram,
)
from src.evaluation.executor import execute_cadquery
from src.metrics.registry import build_metrics
from src.models.factory import build_sft_model


# Offline scoring has no step budget to protect, so it defaults to every metric
# family, including the two expensive B-Rep benchmarks that training-time
# validation leaves off.
TEST_METRICS: tuple[str, ...] = (
    "CadExecutionMetric",
    "VoxelIoUMetric",
    "BoundingBoxMetric",
    "ChamferMetric",
    "CADGenBenchScoreMetric",
    "ECCVChallengeMetric",
)


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_run_config(ckpt: Path, override: Path | None) -> Mapping[str, Any]:
    """Recover the resolved training config that produced ``ckpt``.

    ``setup_run`` writes the fully resolved config to ``<run_dir>/config.yaml``.
    The checkpoint may be either ``<run_dir>/checkpoints/latest`` or a top-K
    snapshot ``<run_dir>/checkpoints/topk/<name>``, so walk up to the nearest
    ancestor that holds a ``config.yaml`` instead of assuming a fixed depth.
    """

    if override is not None:
        config_path = override
    else:
        config_path = next(
            (
                candidate
                for parent in ckpt.resolve().parents
                if (candidate := parent / "config.yaml").is_file()
            ),
            None,
        )
    if config_path is None or not config_path.is_file():
        raise FileNotFoundError(
            f"could not locate a run config.yaml above {ckpt}; pass --config"
        )
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(config, Mapping):
        raise TypeError("run config must resolve to a mapping")
    return {str(key): value for key, value in config.items()}


def _select_sample_ids(test_dir: Path, subset_size: int, seed: int) -> tuple[str, ...]:
    available = list(discover_sample_ids(test_dir))
    if subset_size <= 0 or subset_size >= len(available):
        return tuple(available)
    import random

    generator = random.Random(seed)
    return tuple(sorted(generator.sample(available, subset_size)))


def _build_loader(
    *,
    data_config: Mapping[str, Any],
    processor: Any,
    num_primitive_latents: int,
    test_dir: Path,
    sample_ids: tuple[str, ...],
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dxf_config = DXFPrimitiveConfig(**data_config["dxf"])
    image_sources = tuple(
        RasterImageSource(str(item["style"]), str(item["directory"]))
        for item in data_config["image_sources"]
    )
    preprocessor = Drawing2CADPreprocessor(
        processor,
        num_primitive_latents,
        include_labels=False,
        max_length=data_config.get("max_sequence_length"),
    )
    dataset = Drawing2CADDataset(
        test_dir,
        dxf_parser=DXFPrimitiveParser(dxf_config),
        image_sources=image_sources,
        include_target=False,
        sample_ids=sample_ids,
        strict_files=bool(data_config.get("strict_files", True)),
        image_max_edge=data_config.get("image_max_edge"),
        transform=preprocessor,
    )
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("processor tokenizer has no pad token ID")
    # Decoder-only batched generation requires left padding.
    collator = Drawing2CADCollator(int(pad_token_id), padding_side="left")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collator,
    )


@torch.inference_mode()
def _generate(
    *,
    model: Any,
    loader: DataLoader,
    processor: Any,
    device: torch.device,
    max_new_tokens: int,
) -> dict[str, str]:
    model.eval()
    codes: dict[str, str] = {}
    for batch in tqdm(loader, desc="Generating"):
        if not isinstance(batch, Drawing2CADBatch):
            raise TypeError("generation loader must emit Drawing2CADBatch")
        inputs = batch.to(device).model_inputs
        prompt_length = int(inputs["input_ids"].shape[1])
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
        decoded = processor.batch_decode(
            generated[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        codes.update(zip(batch.sample_ids, decoded, strict=True))
    return codes


def _load_model(cfg: Mapping[str, Any], ckpt: Path, device: torch.device):
    bundle = build_sft_model(cfg["model"])
    adapter_dir = ckpt / "adapter"
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"checkpoint has no adapter directory: {adapter_dir}")
    bundle.model.to(device)
    adapter_state = load_peft_weights(str(adapter_dir), device=str(device))
    result = set_peft_model_state_dict(bundle.model, adapter_state)
    unexpected = tuple(getattr(result, "unexpected_keys", ()))
    if unexpected:
        raise RuntimeError(f"unexpected adapter checkpoint keys: {unexpected}")
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True, help="checkpoint dir")
    parser.add_argument("--test_dir", type=Path, required=True, help="dataset root")
    parser.add_argument("--out_dir", type=Path, required=True, help="prediction dir")
    parser.add_argument("--config", type=Path, default=None, help="override run config")
    parser.add_argument("--subset_size", type=int, default=0, help="0 = all samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--no_metrics",
        action="store_true",
        help="skip scoring even if target/ is present",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        metavar="CLASS",
        help=(
            "metric class names to score with; defaults to every family "
            f"({', '.join(TEST_METRICS)})"
        ),
    )
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    cfg = _load_run_config(args.ckpt, args.config)
    data_config = cfg["data"]
    evaluation_config = cfg.get("evaluation", {})

    bundle = _load_model(cfg, args.ckpt, device)
    sample_ids = _select_sample_ids(args.test_dir, args.subset_size, args.seed)
    loader = _build_loader(
        data_config=data_config,
        processor=bundle.processor,
        num_primitive_latents=bundle.primitive_config.num_primitive_latents,
        test_dir=args.test_dir,
        sample_ids=sample_ids,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"Generating {len(sample_ids)} samples on {device} ...", flush=True)
    codes = _generate(
        model=bundle.model,
        loader=loader,
        processor=bundle.processor,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    timeout_s = float(evaluation_config.get("cadquery_timeout_seconds", 30.0))
    tess = float(evaluation_config.get("tessellation_tolerance", 0.1))
    volume_tol = float(evaluation_config.get("volume_tolerance", 1e-6))

    def _verify_one(sample_id: str) -> dict[str, Any]:
        code = codes[sample_id]
        (args.out_dir / f"{sample_id}.cadquery.py").write_text(code, encoding="utf-8")
        step_path = args.out_dir / f"{sample_id}.step"
        if step_path.exists():
            step_path.unlink()
        execution = execute_cadquery(
            code,
            timeout_s=timeout_s,
            step_output_path=step_path,
            tessellation_tolerance=tess,
            volume_tolerance=volume_tol,
        )
        return {"id": sample_id, **execution.__dict__}

    print(f"Verifying {len(sample_ids)} CADQuery scripts ...", flush=True)
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        exec_rows = list(
            tqdm(
                pool.map(_verify_one, sample_ids),
                total=len(sample_ids),
                desc="Verifying",
            )
        )
    _atomic_json(args.out_dir / "execution.json", exec_rows)
    exec_ok = sum(1 for row in exec_rows if row["valid"])
    print(f"Wrote {len(sample_ids)} prediction(s); {exec_ok} produced a valid solid.")

    target_dir = args.test_dir / "target"
    if args.no_metrics or not target_dir.is_dir():
        reason = "requested" if args.no_metrics else f"no target dir at {target_dir}"
        print(f"Skipping metrics ({reason}).")
        return

    items = [
        EvaluationItem(
            sample_id=sample_id,
            prediction=codes[sample_id],
            target_step_path=target_dir / f"{sample_id}.step",
        )
        for sample_id in sample_ids
    ]
    metrics_spec = evaluation_config.get("metrics", TEST_METRICS)
    if args.metrics:
        metrics_spec = args.metrics
    selected_metrics = build_metrics(metrics_spec)
    eval_config = EvaluationConfig(
        timeout_s=timeout_s,
        tessellation_tolerance=tess,
        volume_tolerance=volume_tol,
        metric_timeout_s=float(
            evaluation_config.get("metric_timeout_seconds", 300.0)
        ),
        metrics=selected_metrics,
    )
    print("Scoring against target/ ...", flush=True)
    rows = evaluate_predictions(items, config=eval_config)
    metrics = aggregate_evaluation_metrics(rows, metrics=selected_metrics, prefix="test")
    _atomic_json(args.out_dir / "rows.json", rows)
    _atomic_json(args.out_dir / "metrics.json", metrics)
    _atomic_json(
        args.out_dir / "error_histogram.json", evaluation_error_histogram(rows)
    )
    print("Metrics:")
    for key in sorted(metrics):
        print(f"  {key}: {metrics[key]}")


if __name__ == "__main__":
    main()
