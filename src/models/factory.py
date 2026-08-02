"""Build the Qwen/primitive SFT model and enforce its train-mode policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from peft import LoraConfig, TaskType, get_peft_model
import torch
from transformers import AutoProcessor

from .model import Drawing2CADQwen3VLForConditionalGeneration
from .primitive_encoder import PrimitiveEncoderConfig


@dataclass(frozen=True)
class SFTModelBundle:
    model: torch.nn.Module
    processor: Any
    primitive_config: PrimitiveEncoderConfig


def _torch_dtype(name: str | torch.dtype) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    aliases = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return aliases[str(name).lower()]
    except KeyError as error:
        raise ValueError(f"unsupported torch dtype: {name!r}") from error


def vision_module(model: torch.nn.Module) -> torch.nn.Module:
    """Locate Qwen3-VL's visual tower before/after PEFT and DDP wrapping."""

    current_model = model
    while isinstance(getattr(current_model, "module", None), torch.nn.Module):
        current_model = current_model.module
    candidates = (
        ("model", "visual"),
        ("base_model", "model", "model", "visual"),
        ("base_model", "model", "model", "model", "visual"),
    )
    for path in candidates:
        current: Any = current_model
        for component in path:
            current = getattr(current, component, None)
            if current is None:
                break
        if isinstance(current, torch.nn.Module):
            return current
    raise AttributeError("could not locate Qwen3-VL visual tower")


def freeze_vision_encoder(model: torch.nn.Module) -> torch.nn.Module:
    visual = vision_module(model)
    visual.requires_grad_(False)
    visual.eval()
    return visual


def apply_language_lora(
    model: torch.nn.Module, lora_config: Mapping[str, Any]
) -> torch.nn.Module:
    if not bool(lora_config.get("enabled", True)):
        raise ValueError(
            "baseline SFT requires model.lora.enabled=true; full fine-tuning is "
            "intentionally unsupported"
        )
    target_modules = tuple(
        str(value) for value in lora_config.get("target_modules", ())
    )
    if not target_modules:
        raise ValueError("model.lora.target_modules must not be empty")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_config.get("rank", lora_config.get("r", 16))),
        lora_alpha=int(lora_config.get("alpha", 32)),
        lora_dropout=float(lora_config.get("dropout", 0.05)),
        bias=str(lora_config.get("bias", "none")),
        target_modules=list(target_modules),
        modules_to_save=["primitive_encoder"],
    )
    output = get_peft_model(model, peft_config)
    primitive_parameters = [
        parameter
        for name, parameter in output.named_parameters()
        if "primitive_encoder.modules_to_save" in name
    ]
    if not primitive_parameters or not all(
        parameter.requires_grad for parameter in primitive_parameters
    ):
        raise RuntimeError("PEFT did not retain a trainable primitive_encoder copy")
    return output


def primitive_encoder_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Every primitive-encoder copy, including PEFT's frozen original."""

    return [
        module
        for name, module in model.named_modules()
        if name.rsplit(".", 1)[-1] == "primitive_encoder"
    ]


def cast_trainable_parameters(
    model: torch.nn.Module, dtype: torch.dtype = torch.float32
) -> int:
    """Hold every updated weight in fp32 while the forward pass stays bf16.

    Accelerate's ``mixed_precision="bf16"`` is autocast-only: unlike fp16 AMP it
    keeps no fp32 master copy, so whatever dtype the parameters were loaded in is
    also the dtype AdamW updates and stores its moments in. bf16 carries 8
    mantissa bits, i.e. a relative resolution of about 1/256, so an update of
    ``lr`` applied to a weight of order one rounds straight back to the old value
    and the module silently stops learning. That is fatal for the primitive
    encoder, whose LayerNorm gains sit at exactly 1.0 and whose embeddings sit at
    ``primitive_dim ** -0.5``, and it quietly degrades the LoRA adapters too.

    Casting only the trainable parameters keeps the frozen ~2B Qwen backbone in
    bf16, so the memory cost is limited to the adapters and the primitive path.
    Autocast still runs the matmuls in bf16, so throughput is unchanged.

    Returns the number of parameters that were cast.
    """

    count = sum(
        1
        for parameter in model.parameters()
        if parameter.is_floating_point() and parameter.dtype != dtype
    )
    # The frozen ``original_module`` PEFT keeps beside the trainable copy is not
    # updated, but ``encode_primitives`` reads its dtype to cast the incoming
    # primitive batch; leaving it bf16 would feed bf16 inputs to fp32 weights
    # outside autocast (evaluation and tests) and raise a dtype mismatch.
    for module in primitive_encoder_modules(model):
        module.to(dtype)
    for parameter in model.parameters():
        if (
            parameter.requires_grad
            and parameter.is_floating_point()
            and parameter.dtype != dtype
        ):
            parameter.data = parameter.data.to(dtype)
    remaining = sum(
        1
        for parameter in model.parameters()
        if parameter.is_floating_point() and parameter.dtype != dtype
    )
    return count - remaining


def set_sft_train_mode(model: torch.nn.Module) -> None:
    """Enable train mode while keeping the frozen vision tower deterministic."""

    model.train()
    vision_module(model).eval()


def build_sft_model(model_config: Mapping[str, Any]) -> SFTModelBundle:
    primitive_config = PrimitiveEncoderConfig(**model_config["primitive"])
    model_name = str(model_config["model_name_or_path"])
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = Drawing2CADQwen3VLForConditionalGeneration.from_qwen_pretrained(
        model_name,
        primitive_config=primitive_config,
        attn_implementation=str(model_config.get("attn_implementation", "auto")),
        dtype=_torch_dtype(model_config.get("torch_dtype", "bfloat16")),
    )
    if not bool(model_config.get("freeze_vision_encoder", True)):
        raise NotImplementedError("the baseline requires freeze_vision_encoder=true")
    # Before PEFT, so the deep copy under ``modules_to_save`` starts calibrated.
    if primitive_config.normalize_output:
        model.calibrate_primitive_output_scale()
    freeze_vision_encoder(model)
    model = apply_language_lora(model, model_config["lora"])
    freeze_vision_encoder(model)
    if bool(model_config.get("fp32_trainable_params", True)):
        cast_trainable_parameters(model)
    if bool(model_config.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    return SFTModelBundle(
        model=model,
        processor=processor,
        primitive_config=primitive_config,
    )


__all__ = [
    "SFTModelBundle",
    "apply_language_lora",
    "build_sft_model",
    "cast_trainable_parameters",
    "freeze_vision_encoder",
    "primitive_encoder_modules",
    "set_sft_train_mode",
    "vision_module",
]
