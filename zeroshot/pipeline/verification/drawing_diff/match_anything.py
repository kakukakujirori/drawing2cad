"""CPU runner for the ORIGINAL MatchAnything Space ELoFTR (not Transformers).

Python >= 3.10. Use an environment where torch and torchvision already work.
Additional packages:
  python -m pip install numpy pillow opencv-python-headless einops loguru \
      'kornia>=0.7.2,<0.9' 'yacs==0.1.8' 'timm>=0.9,<2' huggingface-hub gdown

Run:
  python infer_pair_native.py a.png b.png --diagnose -o native/matches.npz
  python infer_pair_native.py a.png b.png --offline -o native2/matches.npz
  python infer_pair_native.py --test-utils
  python infer_pair_native.py --check-kornia

This is one ENTRY file, not a bundled model. On first use it downloads pinned
original Python source from the Space and the author's original checkpoint ZIP
from Google Drive. Downloaded Python executes locally; use trusted sources only.
Images are never sent to an inference API. --offline disables resource downloads.

Library use: load_runtime(...) prepares the matcher (offline by default), then
infer_pairs(runtime, rgb0, rgb1) reuses it without writing files or downloading.
Array inputs are uint8 RGB; outputs retain original integer-center coordinates.
The shared align API calls validate_options/estimate_transform to fit those raw
correspondences with USAC_MAGSAC and the requested geometric model.

The ORIGINAL neural model, its coarse/fine matching and the Space's preprocessing
and coordinate restoration are used. No Transformers classes or rewritten fine
matcher are used. To avoid Gradio/Lightning/training-only imports, selected
pre/postprocessing definitions are compiled unchanged from their source AST.
The native src.loftr.LoFTR class is imported normally.

CPU adaptations, recorded in each JSON:
 * eval(), CPU, float32, FP16/AMP off;
 * if kornia.utils.grid is missing, alias it to kornia.geometry.grid in memory;
   check a CPU (x,y) grid before using it; no source/cache/site-packages edits;
 * default --attention full selects the ORIGINAL FullAttention class by setting
   COARSE.XFORMER=False; --attention upstream leaves that upstream flag unchanged;
 * Lightning is bypassed; all trainable parameters AND required buffers must load;
 * transparent pixels are composited on white and EXIF orientation is applied.
These choices do not establish bit-for-bit equivalence to the hosted UI.

Defaults are read from the pinned Space's match_dense.confs: threshold .001,
force resize 640x480, then original bottom/right ZERO padding + valid masks.
This is deliberately NOT the earlier centered-white-letterbox preprocessing.
Raw infer_pairs/CLI coordinates remain paired float32; no RANSAC, no sorting/reindexing, no hidden
confidence/bounds filter. The NPZ contains an additional in_bounds mask.

Validation status at delivery: utility/integration tests with substitute models
only. Original pretrained CPU inference was NOT executed in the authoring sandbox
because outbound downloads were unavailable. --diagnose performs real-weight
self0/self1/pair tests in YOUR environment; it is not a simulated inference mode.

Original sources and original checkpoint locator:
 https://huggingface.co/spaces/LittleFrog/MatchAnything
 https://drive.google.com/file/d/12L3g9-w8rR9K2L4rYaGaDJ7NqX1D713d/view
Original source keeps its own upstream licenses; this launcher does not relicense it.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

SPACE = "LittleFrog/MatchAnything"
REVISION = "48f422d1f82d2d8e3cbd6606514573da3bdc8036"
ROOT_REL = Path("imcui/third_party/MatchAnything")
WRAPPER_REL = Path("imcui/hloc/matchers/matchanything.py")
DENSE_REL = Path("imcui/hloc/match_dense.py")
EXTRACT_REL = Path("imcui/hloc/extract_features.py")
DRIVE_ID = "12L3g9-w8rR9K2L4rYaGaDJ7NqX1D713d"
CHECKPOINT_NAME = "matchanything_eloftr.ckpt"
VERSION = "1.0.1-kornia-grid-compat"


@dataclass
class NativeRuntime:
    """Initialized CPU matcher, reusable across image pairs."""

    model: Any
    adapter: Any
    match_images: Any
    preset: dict
    provenance: dict
    threads: int


@dataclass
class PairMatches:
    """Paired float32 points in original images' integer-center coordinates."""

    points0: Any
    points1: Any
    confidence: Any
    in_bounds: Any
    diagnostics: dict
    debug: NativeCapture | None = None


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def version_info() -> dict[str, Any]:
    result = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in (
        "torch",
        "torchvision",
        "numpy",
        "Pillow",
        "kornia",
        "yacs",
        "einops",
        "timm",
        "loguru",
        "huggingface-hub",
        "gdown",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def memory_peak_mib() -> float | None:
    try:
        import resource

        n = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return float(n / (1024 * 1024 if sys.platform == "darwin" else 1024))
    except (ImportError, AttributeError):
        return None


def atomic_write(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    atomic_write(path, lambda tmp: tmp.write_text(text, encoding="utf-8"))


def write_npz(path: Path, **arrays) -> None:
    import numpy as np

    def writer(tmp):
        with tmp.open("wb") as f:
            np.savez_compressed(f, **arrays)

    atomic_write(path, writer)


def output_paths(base: Path) -> dict[str, Path]:
    stem = base.with_suffix("")
    return {
        k: Path(str(stem) + s)
        for k, s in {
            "npz": ".npz",
            "json": ".json",
            "png": ".png",
            "raw": ".raw.npz",
            "input0": ".input0.png",
            "input1": ".input1.png",
            "mask0": ".mask0.png",
            "mask1": ".mask1.png",
        }.items()
    }


def guard_outputs(paths: list[Path], inputs: list[Path], overwrite: bool) -> None:
    protected = {p.resolve() for p in inputs}
    for p in paths:
        if p.resolve() in protected:
            raise ValueError(f"Output would overwrite an input/resource: {p}")
        if p.exists() and (not overwrite or not p.is_file()):
            raise FileExistsError(
                f"Output exists: {p}. Choose a new output or use --overwrite."
            )


def selected_definitions(path: Path, requests: list[str], namespace: dict) -> dict:
    """Compile ONLY requested original definitions, without module-level imports.

    This is dependency isolation, NOT a security sandbox: the selected functions
    and imported original model are trusted executable Python. Function bodies,
    signatures and decorators are left unchanged. A request 'C.f' selects a method.
    """
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    chosen = []
    for qualified in requests:
        parts = qualified.split(".")
        nodes = tree.body
        if len(parts) == 2:
            classes = [
                n for n in nodes if isinstance(n, ast.ClassDef) and n.name == parts[0]
            ]
            if len(classes) != 1:
                raise RuntimeError(f"Expected class {parts[0]} in {path}")
            nodes = classes[0].body
        functions = [
            n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == parts[-1]
        ]
        if len(functions) != 1:
            raise RuntimeError(f"Expected original definition {qualified} in {path}")
        chosen.append(copy.deepcopy(functions[0]))
    # Postponing annotation evaluation does not alter numerical function bodies.
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, *chosen], type_ignores=[])
    )
    namespace.setdefault("__name__", "_matchanything_selected_original_definitions")
    namespace.setdefault("__file__", str(path))
    # Execute selected upstream definitions without their module-level imports.
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102
    return {q: namespace[q.split(".")[-1]] for q in requests}


def read_space_preset(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for n in tree.body:
        if (
            isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "confs" for t in n.targets)
            and isinstance(n.value, ast.Dict)
        ):
            for k, v in zip(n.value.keys, n.value.values):
                if isinstance(k, ast.Constant) and k.value == "matchanything_eloftr":
                    result = ast.literal_eval(v)
                    if result["model"]["model_name"] != "matchanything_eloftr":
                        raise RuntimeError("Unexpected Space model preset")
                    return result
    raise RuntimeError(f"Cannot find literal matchanything_eloftr preset in {path}")


def source_snapshot(args) -> Path:
    if args.space_dir:
        directory = args.space_dir.expanduser().resolve()
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise RuntimeError("Install huggingface-hub, or pass --space-dir.") from e
        log(f"[source] {SPACE} @ {REVISION} (offline={args.offline})")
        directory = Path(
            snapshot_download(
                repo_id=SPACE,
                repo_type="space",
                revision=REVISION,
                cache_dir=str(args.cache_dir / "hf"),
                local_files_only=args.offline,
                allow_patterns=[
                    ROOT_REL.as_posix() + "/**",
                    WRAPPER_REL.as_posix(),
                    DENSE_REL.as_posix(),
                    EXTRACT_REL.as_posix(),
                ],
                max_workers=4,
            )
        )
    required = [
        WRAPPER_REL,
        DENSE_REL,
        EXTRACT_REL,
        ROOT_REL / "src/loftr/loftr.py",
        ROOT_REL / "src/config/default.py",
        ROOT_REL / "configs/models/eloftr_model.py",
    ]
    for rel in required:
        if not (directory / rel).is_file():
            raise FileNotFoundError(f"Missing original Space file: {directory / rel}")
    return directory


def extract_eloftr_checkpoint(archive: Path, destination: Path) -> None:
    """Copy just the unambiguous checkpoint member; never extract archive paths."""
    with zipfile.ZipFile(archive) as z:
        candidates = [
            i
            for i in z.infolist()
            if not i.is_dir()
            and PurePosixPath(i.filename.replace("\\", "/")).name == CHECKPOINT_NAME
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Expected exactly one {CHECKPOINT_NAME}; found {len(candidates)} in {archive}"
            )
        info = candidates[0]
        name = PurePosixPath(info.filename.replace("\\", "/"))
        if name.is_absolute() or ".." in name.parts or ":" in info.filename:
            raise ValueError("Unsafe checkpoint member name in ZIP")
        if not 0 < info.file_size <= 4 * 1024**3:
            raise ValueError(
                "Checkpoint member is empty or exceeds the 4 GiB extraction limit"
            )

        def writer(tmp):
            with z.open(info) as src, tmp.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            if tmp.stat().st_size != info.file_size:
                raise RuntimeError("Incomplete checkpoint extraction")

        atomic_write(destination, writer)


def get_checkpoint(args, space_dir: Path) -> Path:
    if args.checkpoint:
        path = args.checkpoint.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    destination = args.cache_dir / "weights" / CHECKPOINT_NAME
    if args.weights_zip:
        extract_eloftr_checkpoint(args.weights_zip.expanduser(), destination)
        return destination
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    local = space_dir / ROOT_REL / "weights" / CHECKPOINT_NAME
    if local.is_file():
        return local
    archive = args.cache_dir / "weights" / "official_weights.zip"
    if not archive.is_file():
        if args.offline:
            raise FileNotFoundError(
                "Original checkpoint is not cached. The previous Transformers cache is NOT this checkpoint. "
                "Run once online, or pass --checkpoint /path/matchanything_eloftr.ckpt or --weights-zip /path/weights.zip."
            )
        try:
            import gdown
        except ImportError as e:
            raise RuntimeError(
                "Install gdown, or pass --checkpoint / --weights-zip."
            ) from e
        log(
            "[weights] Downloading the author's Google Drive ZIP (includes models besides ELoFTR)."
        )

        def writer(tmp):
            result = gdown.download(id=DRIVE_ID, output=str(tmp), quiet=False)
            if result is None or not zipfile.is_zipfile(tmp):
                raise RuntimeError(
                    "Google Drive did not return a valid ZIP. Check quota/network; use --weights-zip or --checkpoint."
                )

        atomic_write(archive, writer)
    extract_eloftr_checkpoint(archive, destination)
    return destination


def lower_config(value):
    if hasattr(value, "items"):
        return {str(k).lower(): lower_config(v) for k, v in value.items()}
    if isinstance(value, list):
        return [lower_config(v) for v in value]
    if isinstance(value, tuple):
        return tuple(lower_config(v) for v in value)
    return value


def load_weights_checked(model, path: Path, allow_unsafe: bool) -> dict:
    import torch

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=not allow_unsafe)
    except Exception as e:
        if not allow_unsafe:
            raise RuntimeError(
                "Safe checkpoint loading failed; no unsafe retry was performed. "
                "If this is the trusted author's original .ckpt and the traceback is a weights_only metadata error, "
                "--allow-unsafe-checkpoint opts into torch pickle loading (which can execute code). "
                f"Original error: {e}"
            ) from e
        raise
    state = (
        checkpoint.get("state_dict", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    # Invalid checkpoint contents retain the existing ValueError contract.
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a state_dict dictionary")  # noqa: TRY004
    cleaned = {}
    for k, v in state.items():
        if not isinstance(k, str):
            raise ValueError("Non-string checkpoint key")  # noqa: TRY004
        key = k.removeprefix("module.").removeprefix("matcher.")
        if key in cleaned:
            raise ValueError(f"Duplicate normalized checkpoint key: {key}")
        cleaned[key] = v
    expected = model.state_dict()
    missing = sorted(set(expected) - set(cleaned))
    # Missing num_batches_tracked is the only tolerated legacy BatchNorm buffer.
    missing_required = [k for k in missing if not k.endswith(".num_batches_tracked")]
    mismatched = [
        k
        for k in set(expected) & set(cleaned)
        if not isinstance(cleaned[k], torch.Tensor)
        or tuple(cleaned[k].shape) != tuple(expected[k].shape)
    ]
    if missing_required or mismatched:
        raise RuntimeError(
            "Checkpoint/model mismatch; refusing partly random model. "
            f"missing={missing_required[:20]}, wrong_shape={mismatched[:20]}"
        )
    applicable = {k: v for k, v in cleaned.items() if k in expected}
    result = model.load_state_dict(applicable, strict=False)
    named = dict(model.named_parameters())
    if set(result.missing_keys) & set(named):
        raise RuntimeError("A trainable model parameter was not loaded")
    return {
        "missing_keys": list(result.missing_keys),
        "ignored_extra_keys": sorted(set(cleaned) - set(expected)),
        "loaded_state_entries": len(applicable),
        "trainable_parameter_tensors": len(named),
        "number_of_parameters": sum(p.numel() for p in named.values()),
        "unsafe_pickle_opt_in": allow_unsafe,
    }


class NullProfiler:
    def profile(self, *_args, **_kwargs):
        return contextlib.nullcontext()

    def summary(self):
        return "No-op profiler; wall-clock times recorded by native runner."


def install_kornia_grid_compat() -> dict[str, Any]:
    """Resolve the legacy grid import, without rewriting upstream Python files.

    Kornia 0.8.3 provides create_meshgrid under kornia.geometry.grid.
    The original MatchAnything FineMatching imports kornia.utils.grid.
    Only that missing module is aliased; unrelated dependency errors propagate.
    The installed Kornia function is used directly (not a reimplementation).
    """
    import torch

    kornia = importlib.import_module("kornia")
    utils = importlib.import_module("kornia.utils")
    legacy = "kornia.utils.grid"
    target = "kornia.geometry.grid"
    aliased = False
    try:
        grid = importlib.import_module(legacy)
    except ModuleNotFoundError as exc:
        if exc.name != legacy:
            raise
        # Do not catch/replace errors from a missing Kornia dependency.
        grid = importlib.import_module(target)
        aliased = True

    create_meshgrid = getattr(grid, "create_meshgrid", None)
    # A missing dependency API is a runtime compatibility failure.
    if not callable(create_meshgrid):
        raise RuntimeError(f"{grid.__name__} has no callable create_meshgrid")  # noqa: TRY004
    # Validate shape, x/y order, positional device argument and normalization.
    # These tiny probes use the REAL installed function, not a mock.
    for normalized in (False, True):
        probe = create_meshgrid(
            3, 3, normalized, torch.device("cpu"), dtype=torch.float32
        )
        axis = torch.tensor([-1.0, 0.0, 1.0] if normalized else [0.0, 1.0, 2.0])
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        expected = torch.stack((xx, yy), dim=-1).unsqueeze(0)
        if (
            not isinstance(probe, torch.Tensor)
            or tuple(probe.shape) != (1, 3, 3, 2)
            or probe.device.type != "cpu"
            or probe.dtype != torch.float32
            or not torch.equal(probe, expected)
        ):
            raise RuntimeError(
                "Kornia create_meshgrid CPU shape/(x,y)/normalization check failed"
            )

    if aliased:
        # Process-local import alias. Do NOT overwrite anything on disk.
        sys.modules[legacy] = grid
        utils.grid = grid
        log(f"[compat] {legacy} -> {target} (in memory; CPU grid check passed)")
    return {
        "kornia_version": getattr(kornia, "__version__", "unknown"),
        "requested_module": legacy,
        "resolved_module": grid.__name__,
        "alias_active": grid.__name__ != legacy,
        "cpu_grid_check": "passed",
        "files_modified": False,
    }


def build_native_model(space_dir: Path, checkpoint: Path, preset: dict, args):
    import torch

    kornia_compat = install_kornia_grid_compat()
    root = (space_dir / ROOT_REL).resolve()
    # 'src' is the upstream package name. Never silently import a user's other src.
    loaded = sys.modules.get("src")
    if loaded is not None:
        origins = list(getattr(loaded, "__path__", []))
        if not origins or any(
            not Path(p).resolve().is_relative_to(root) for p in origins
        ):
            raise RuntimeError(
                "Another 'src' package is loaded. Run this file in a fresh Python process."
            )
    sys.path.insert(0, str(root))
    try:
        from src.config.default import get_cfg_defaults
        from src.loftr import LoFTR
    except ImportError as e:
        raise RuntimeError(
            "Could not import original model. Install the additional packages in the file header. "
            "No Transformers fallback is used. Import error: " + str(e)
        ) from e
    config = get_cfg_defaults()
    config.merge_from_file(str(root / "configs/models/eloftr_model.py"))
    model_conf = preset["model"]
    config.METHOD = "matchanything_eloftr"
    config.LOFTR.FP16 = False
    config.ROMA.MODEL.AMP = False
    config.LOFTR.MATCH_COARSE.THR = model_conf["match_threshold"]
    if config.DATASET.NPE_NAME == "megadepth":
        config.LOFTR.COARSE.NPE = [
            832,
            832,
            model_conf["img_resize"],
            model_conf["img_resize"],
        ]
    if args.attention == "full":
        config.LOFTR.COARSE.XFORMER = False
        config.LOFTR.COARSE.ATTENTION = "full"
    plain = lower_config(config)
    log(
        f"[model] ORIGINAL LoFTR; CPU float32; attention={args.attention}; threshold={model_conf['match_threshold']}"
    )
    matcher = LoFTR(config=plain["loftr"], profiler=NullProfiler())
    weight_report = load_weights_checked(
        matcher, checkpoint, args.allow_unsafe_checkpoint
    )
    matcher = matcher.eval().to(device=torch.device("cpu"), dtype=torch.float32)
    matcher._native_kornia_grid_compatibility = kornia_compat
    return matcher, plain["loftr"], weight_report


def make_original_adapter(space_dir: Path, matcher, model_conf: dict):
    import cv2
    import numpy as np
    import PIL
    import torch
    import torch.nn.functional as nnf
    import torchvision.transforms.functional as visionf
    from PIL import Image

    common = {
        "np": np,
        "PIL": PIL,
        "Image": Image,
        "cv2": cv2,
        "torch": torch,
        "Path": Path,
        "os": os,
        "SimpleNamespace": SimpleNamespace,
    }
    wrapper_ns = {**common, "F": nnf, "DEVICE": torch.device("cpu")}
    functions = selected_definitions(
        space_dir / WRAPPER_REL,
        [
            "resize",
            "process_resize",
            "resize_image",
            "pad_bottom_right",
            "dict_to_cuda",
            "list_to_cuda",
            "MatchAnything._forward",
        ],
        wrapper_ns,
    )
    dense_ns = {**common, "F": visionf}
    selected_definitions(space_dir / EXTRACT_REL, ["resize_image"], dense_ns)
    functions_dense = selected_definitions(
        space_dir / DENSE_REL, ["scale_keypoints", "match_images"], dense_ns
    )

    class Adapter:
        def __init__(self):
            self.net = matcher
            self.conf = model_conf

        def __call__(self, data):
            return functions["MatchAnything._forward"](self, data)

    return Adapter(), functions_dense["match_images"]


def load_runtime(
    *,
    cache_dir: str | Path | None = None,
    space_dir: str | Path | None = None,
    checkpoint: str | Path | None = None,
    weights_zip: str | Path | None = None,
    offline: bool = True,
    allow_unsafe_checkpoint: bool = False,
    threshold: float | None = None,
    width: int | None = None,
    height: int | None = None,
    attention: str = "full",
    threads: int = 4,
) -> NativeRuntime:
    """Initialize original CPU weights once; resource downloads require offline=False.

    Local source and checkpoint preparation happens here, never in infer_pairs.
    Model/source imports and torch's CPU thread/seed setup are process-wide.
    """
    validate_runtime_options(threads, width, height, threshold, attention)
    args = SimpleNamespace(
        cache_dir=(
            Path(cache_dir)
            if cache_dir is not None
            else Path.home() / ".cache/matchanything-native"
        )
        .expanduser()
        .resolve(),
        space_dir=Path(space_dir) if space_dir is not None else None,
        checkpoint=Path(checkpoint) if checkpoint is not None else None,
        weights_zip=Path(weights_zip) if weights_zip is not None else None,
        offline=offline,
        allow_unsafe_checkpoint=allow_unsafe_checkpoint,
        threshold=threshold,
        width=width,
        height=height,
        attention=attention,
        threads=threads,
    )
    tload = time.perf_counter()
    source = source_snapshot(args)
    checkpoint = get_checkpoint(args, source)
    import torch

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(0)
    preset = read_space_preset(source / DENSE_REL)
    if args.threshold is not None:
        preset["model"]["match_threshold"] = args.threshold
    for key in ("width", "height"):
        if getattr(args, key) is not None:
            preset["preprocessing"][key] = getattr(args, key)
    if not preset["preprocessing"].get("force_resize"):
        raise RuntimeError("Expected original force_resize preset; source changed")
    model, config, weight_report = build_native_model(source, checkpoint, preset, args)
    adapter, match_images = make_original_adapter(source, model, preset["model"])
    source_files = [
        WRAPPER_REL,
        DENSE_REL,
        EXTRACT_REL,
        ROOT_REL / "src/loftr/loftr.py",
        ROOT_REL / "src/loftr/utils/coarse_matching.py",
        ROOT_REL / "src/loftr/utils/fine_matching.py",
        ROOT_REL / "src/loftr/loftr_module/linear_attention.py",
        ROOT_REL / "configs/models/eloftr_model.py",
    ]
    provenance = {
        "space": SPACE,
        "pinned_revision": None if args.space_dir else REVISION,
        "local_source_override": bool(args.space_dir),
        "source_directory": str(source),
        "source_sha256": {str(p): sha256(source / p) for p in source_files},
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checksum_note": "Local recorded hash, not a comparison against an independently published reference hash.",
        "weight_load_report": weight_report,
        "effective_loftr_config": config,
        "versions": version_info(),
        "load_seconds_including_download_and_hashing": time.perf_counter() - tload,
        "attention_choice": args.attention,
        "kornia_grid_compatibility": model._native_kornia_grid_compatibility,
        "adaptations": [
            "Bypass Lightning; instantiate original src.loftr.LoFTR",
            "CPU float32; FP16/AMP off",
            "Original pre/post definitions selected without importing Gradio/Lightning",
            "COARSE.XFORMER=False, original FullAttention"
            if args.attention == "full"
            else "Original COARSE.XFORMER unchanged",
            "CLI read_rgb: EXIF orientation + transparent pixels composited on white",
        ],
        "not_claimed": "Bit-identical hosted-UI results or verified line-drawing matching accuracy.",
    }
    return NativeRuntime(model, adapter, match_images, preset, provenance, threads)


class NativeCapture:
    """Observe the real model without modifying its tensors/selection mathematics."""

    def __init__(self, model, capture_debug=True):
        self.model = model
        self.capture_debug = capture_debug
        self.handles = []
        self.inputs = {}
        self.raw = {}
        self.diagnostics = {"coarse_forward_calls": 0, "fine_forward_calls": 0}
        self.started = None

    def __enter__(self):
        if not hasattr(self.model, "coarse_matching") or not hasattr(
            self.model, "fine_matching"
        ):
            raise RuntimeError(
                "Unexpected original model structure: coarse/fine modules not found"
            )
        self.handles = [
            self.model.register_forward_pre_hook(self.before),
            self.model.register_forward_hook(self.after),
            self.model.coarse_matching.register_forward_hook(self.coarse),
            self.model.fine_matching.register_forward_hook(self.fine),
        ]
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()

    @staticmethod
    def data_from_args(args):
        return next((x for x in args if isinstance(x, dict) and "image0" in x), None)

    def before(self, module, args):
        data = self.data_from_args(args)
        if data is None:
            raise RuntimeError("Native model input dictionary missing")
        for key in ("image0", "image1", "mask0", "mask1"):
            if key in data:
                if data[key].device.type != "cpu":
                    raise RuntimeError("CPU runner received non-CPU input")
                if self.capture_debug:
                    self.inputs[key] = data[key].detach().cpu().numpy().copy()
        self.started = time.perf_counter()

    def coarse(self, module, args, result):
        self.diagnostics["coarse_forward_calls"] += 1
        data = self.data_from_args(args)
        if data is not None:
            if "mconf" in data:
                self.diagnostics["num_after_coarse"] = int(data["mconf"].numel())
            cm = data.get("conf_matrix")
            if cm is not None and cm.numel():
                n = float(cm.max().item())
                self.diagnostics["coarse_confidence_matrix_max"] = (
                    n if math.isfinite(n) else None
                )

    def fine(self, module, args, result):
        self.diagnostics["fine_forward_calls"] += 1

    def after(self, module, args, result):
        if self.started is not None:
            self.diagnostics["native_forward_seconds_including_hooks"] = (
                time.perf_counter() - self.started
            )
        data = self.data_from_args(args)
        if data is None:
            raise RuntimeError("Native model did not receive expected data dictionary")
        # COPY now: original wrapper scales these tensors IN PLACE on CPU later.
        for key in ("mkpts0_f", "mkpts1_f", "mconf"):
            if key not in data:
                raise RuntimeError(f"Native model did not return {key}")
            if self.capture_debug:
                self.raw[key] = data[key].detach().cpu().float().numpy().copy()
        self.diagnostics["num_after_native_fine"] = len(data["mconf"])


def read_rgb(path: Path):
    import numpy as np
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        image = ImageOps.exif_transpose(im).convert("RGBA")
        bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
        return np.array(Image.alpha_composite(bg, image).convert("RGB"), dtype=np.uint8)


def checked_pairs(result: dict, shapes):
    import numpy as np

    p0 = np.asarray(result["mkeypoints0_orig"], dtype=np.float32).copy()
    p1 = np.asarray(result["mkeypoints1_orig"], dtype=np.float32).copy()
    scores = np.asarray(result["mconf"], dtype=np.float32).reshape(-1).copy()
    if p0.shape != (len(scores), 2) or p1.shape != p0.shape:
        raise RuntimeError(
            f"Unpaired native output shapes: {p0.shape}, {p1.shape}, {scores.shape}"
        )
    if not (
        np.isfinite(p0).all() and np.isfinite(p1).all() and np.isfinite(scores).all()
    ):
        raise RuntimeError("Non-finite original model output; not silently filtered")
    valid = np.ones(len(scores), dtype=bool)
    for pts, shape in zip((p0, p1), shapes):
        h, w = shape[:2]
        valid &= (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    return p0, p1, scores, valid


def identity_metrics(raw: dict, p0, p1) -> dict:
    import numpy as np

    n = len(p0)

    def errors(a, b):
        err = np.linalg.norm(a - b, axis=1)
        return {
            "median": float(np.median(err)),
            "p90": float(np.quantile(err, 0.9)),
            "max": float(err.max()),
            "fraction_within_2px": float(np.mean(err <= 2)),
        }

    if not n:
        return {
            "num_matches": 0,
            "smoke_pass": False,
            "reason": "No self correspondences",
        }
    native = errors(raw["mkpts0_f"], raw["mkpts1_f"])
    return {
        "num_matches": n,
        "native_input_pixel_error": native,
        "original_image_pixel_error": errors(p0, p1),
        "smoke_pass": bool(n >= 8 and native["median"] <= 1 and native["p90"] <= 3),
        "criteria": "At least 8 self matches; native-input median <=1px and p90<=3px. Runner smoke criteria, not a benchmark.",
    }


def save_gray(path: Path, array) -> None:
    import numpy as np
    from PIL import Image

    image = np.squeeze(array)
    if image.ndim != 2:
        raise RuntimeError(f"Expected grayscale tensor for preview, got {image.shape}")
    image = Image.fromarray(np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8))
    atomic_write(path, lambda tmp: image.save(tmp, format="PNG"))


def draw_matches(
    path: Path, rgb0, rgb1, p0, p1, scores, in_bounds, limit: int, label: str
) -> None:
    import numpy as np
    from PIL import Image, ImageDraw

    panels, scales = [], []
    for arr in (rgb0, rgb1):
        im = Image.fromarray(arr)
        scale = min(520 / im.width, 650 / im.height)
        size = (max(1, round(im.width * scale)), max(1, round(im.height * scale)))
        panels.append(im.resize(size, Image.Resampling.LANCZOS))
        scales.append(np.array(size, dtype=float) / np.array(im.size, dtype=float))
    top, gap = 38, 20
    canvas = Image.new(
        "RGB",
        (panels[0].width + gap + panels[1].width, max(p.height for p in panels) + top),
        "white",
    )
    offsets = [np.array([0, top]), np.array([panels[0].width + gap, top])]
    for im, xy in zip(panels, offsets):
        canvas.paste(im, tuple(xy))
    draw = ImageDraw.Draw(canvas)
    ids = np.flatnonzero(in_bounds)
    ids = ids[np.argsort(-scores[ids], kind="stable")][:limit]
    draw.text(
        (6, 5),
        f"{label} | refined: {len(scores)} | drawn: {len(ids)} | no RANSAC",
        fill="black",
    )
    for i in ids:
        # Same pixel-center convention as the original restoration.
        a = (p0[i] + 0.5) * scales[0] - 0.5 + offsets[0]
        b = (p1[i] + 0.5) * scales[1] - 0.5 + offsets[1]
        draw.line([tuple(a), tuple(b)], fill="black", width=1)
        for x, y in (a, b):
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline="black", width=1)
    atomic_write(path, lambda tmp: canvas.save(tmp, format="PNG"))


def infer_pairs(
    runtime: NativeRuntime, rgb0, rgb1, *, capture_debug=False
) -> PairMatches:
    """Infer correspondences without loading resources or writing artifacts.

    Inputs are nonempty uint8 HxWx3 RGB arrays, with orientation already resolved
    by the caller. Points retain the original Space integer-center convention;
    neither bounds nor confidence silently remove or reorder correspondences.
    Use a runtime sequentially: the original model and its hooks are mutable.
    """
    import numpy as np
    import torch

    for name, rgb in (("rgb0", rgb0), ("rgb1", rgb1)):
        if (
            not isinstance(rgb, np.ndarray)
            or rgb.dtype != np.uint8
            or rgb.ndim != 3
            or rgb.shape[2] != 3
            or min(rgb.shape[:2]) < 1
        ):
            raise ValueError(f"{name} must be a nonempty uint8 HxWx3 RGB array")
    t0 = time.perf_counter()
    # no_grad instead of inference_mode: preserve original mutating forward behavior.
    with torch.no_grad(), NativeCapture(runtime.model, capture_debug) as capture:
        result = runtime.match_images(
            runtime.adapter,
            rgb0,
            rgb1,
            copy.deepcopy(runtime.preset["preprocessing"]),
            device="cpu",
        )
    elapsed = time.perf_counter() - t0
    if capture.diagnostics["fine_forward_calls"] < 1:
        raise RuntimeError(
            "Native fine matching was not invoked; refusing coarse-only output"
        )
    p0, p1, scores, bounds = checked_pairs(result, (rgb0.shape, rgb1.shape))
    if len(scores) != capture.diagnostics["num_after_native_fine"]:
        raise RuntimeError("Native/preprocessed match counts disagree")
    return PairMatches(
        p0,
        p1,
        scores,
        bounds,
        {**capture.diagnostics, "preprocess_and_model_seconds": elapsed},
        capture if capture_debug else None,
    )


def validate_options(model: str, options: dict[str, Any]) -> dict[str, Any]:
    """Resolve RANSAC settings; the shared align entry point validates the model."""
    defaults = {
        "ransac_reproj_threshold": 3.0,
        "ransac_confidence": 0.999,
        "ransac_max_iter": 10000,
    }
    # Reject unknown fields before filling the backend's defaults.
    if unknown := options.keys() - defaults.keys():
        raise ValueError(f"unknown MatchAnything options: {sorted(unknown)}")
    settings = defaults | options
    if (
        not math.isfinite(settings["ransac_reproj_threshold"])
        or settings["ransac_reproj_threshold"] <= 0
        or not 0 < settings["ransac_confidence"] < 1
        or type(settings["ransac_max_iter"]) is not int
        or settings["ransac_max_iter"] < 1
    ):
        raise ValueError("invalid MatchAnything RANSAC settings")
    return settings


def _similarity(points0: np.ndarray, points1: np.ndarray) -> np.ndarray:
    """Least squares for x'=a*x-b*y+tx, y'=b*x+a*y+ty."""
    import numpy as np

    x, y = points0.T
    design = np.zeros((len(x), 2, 4))
    design[:, 0, :] = np.column_stack((x, -y, np.ones(len(x)), np.zeros(len(x))))
    design[:, 1, :] = np.column_stack((y, x, np.zeros(len(x)), np.ones(len(x))))
    values, _, rank, _ = np.linalg.lstsq(
        design.reshape(-1, 4), points1.ravel(), rcond=None
    )
    if rank < 4:
        raise ValueError("inliers do not determine a similarity transform")
    a, b, tx, ty = values
    return np.array([[a, -b, tx], [b, a, ty], [0, 0, 1.0]])


def _fit_matches(
    pairs: Any,
    drawing_shape: tuple[int, ...],
    projection_shape: tuple[int, ...],
    model: str,
    options: dict[str, Any],
    diagnostics: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    import cv2
    import numpy as np

    # Filter malformed, non-finite and out-of-image correspondences before fitting.
    points0, points1 = (
        np.asarray(pairs.points0, dtype=float),
        np.asarray(pairs.points1, dtype=float),
    )
    scores = np.asarray(pairs.confidence)
    bounds = np.asarray(pairs.in_bounds, dtype=bool)
    count = len(points0)
    if (
        points0.shape != (count, 2)
        or points1.shape != points0.shape
        or scores.shape != (count,)
        or bounds.shape != (count,)
    ):
        raise ValueError("matcher returned unpaired points, confidence or bounds")
    valid = (
        bounds
        & np.isfinite(points0).all(axis=1)
        & np.isfinite(points1).all(axis=1)
        & np.isfinite(scores)
    )
    for points, shape in ((points0, drawing_shape), (points1, projection_shape)):
        height, width = shape[:2]
        valid &= ((points >= 0) & (points < [width, height])).all(axis=1)
    source, target = points0[valid], points1[valid]
    diagnostics.update(
        match_count=count,
        valid_match_count=len(source),
        prefilter_model="homography" if model == "homography" else "affine",
    )
    minimum = 4 if model == "homography" else 3
    if len(source) < minimum:
        raise ValueError(
            f"need at least {minimum} finite, in-bounds matches for {model}"
        )
    if any(
        np.linalg.matrix_rank(points - points.mean(axis=0)) < 2
        for points in (source, target)
    ):
        raise ValueError("matched points are collinear or coincident")

    # MAGSAC rejects outliers using the chosen projective or affine model.
    threshold = options["ransac_reproj_threshold"]
    arguments = {
        "method": cv2.USAC_MAGSAC,
        "ransacReprojThreshold": threshold,
        "confidence": options["ransac_confidence"],
        "maxIters": options["ransac_max_iter"],
    }
    if model == "homography":
        matrix, mask = cv2.findHomography(source, target, **arguments)
    else:
        matrix, mask = cv2.estimateAffine2D(source, target, **arguments)
        if matrix is not None:
            matrix = np.vstack((matrix, [0, 0, 1.0]))
    if matrix is None or mask is None:
        raise ValueError("USAC_MAGSAC could not estimate a transform")
    prefilter = mask.ravel().astype(bool)
    diagnostics["prefilter_inlier_count"] = int(prefilter.sum())
    if prefilter.sum() < minimum:
        raise ValueError("USAC_MAGSAC found too few inliers")
    if model == "similarity":
        # OpenCV does not implement USAC for estimateAffinePartial2D.
        matrix = _similarity(source[prefilter], target[prefilter])

    # Recompute support for the final model, especially the constrained similarity.
    homogeneous = np.column_stack((source, np.ones(len(source)))) @ matrix.T
    errors = np.full(len(source), np.inf)
    finite = np.abs(homogeneous[:, 2]) > 1e-10
    errors[finite] = np.linalg.norm(
        homogeneous[finite, :2] / homogeneous[finite, 2:] - target[finite], axis=1
    )
    inliers = np.isfinite(errors) & (errors <= threshold)
    diagnostics.update(
        inlier_count=int(inliers.sum()), inlier_fraction=float(inliers.mean())
    )
    if inliers.sum() < minimum:
        raise ValueError(f"too few matches support the final {model} transform")
    if any(
        np.linalg.matrix_rank(points[inliers] - points[inliers].mean(axis=0)) < 2
        for points in (source, target)
    ):
        raise ValueError("final inliers are collinear or coincident")

    # Report both spatial support and reprojection error in projection pixels.
    def coverage(points: np.ndarray, shape: tuple[int, ...]) -> float:
        hull = cv2.convexHull(points[inliers].astype(np.float32))
        return float(cv2.contourArea(hull) / (shape[0] * shape[1]))

    diagnostics.update(
        {
            "drawing_support_fraction": coverage(source, drawing_shape),
            "projection_support_fraction": coverage(target, projection_shape),
            "inlier_median_reprojection_distance_px": float(np.median(errors[inliers])),
            "inlier_p95_reprojection_distance_px": float(
                np.quantile(errors[inliers], 0.95)
            ),
            "distance_frame": "projection pixels (matched points, not drawing-line residuals)",
        }
    )
    warnings = []
    # ponytail: structural/low-support checks only; calibrate on real drawings
    # before using these diagnostics to judge geometric correctness.
    if inliers.sum() < 8 or inliers.mean() < 0.1:
        warnings.append("Few matches support this transform; inspect the alignment.")
    diagnostics["support_warning_thresholds"] = {
        "min_inliers": 8,
        "min_inlier_fraction": 0.1,
    }
    return matrix, warnings


def estimate_transform(
    drawing_rgb: np.ndarray,
    projection_rgb: np.ndarray,
    *,
    model: str,
    options: dict[str, Any],
    runtime: Any,
    diagnostics: dict[str, Any],
) -> tuple[np.ndarray, str, list[str]]:
    """Return drawing→projection in integer-centre coordinates, without file I/O.

    The shared align entry point validates images/options and handles failures.
    Updating its diagnostics directly preserves match counts if fitting fails.
    """
    # Reuse the caller's explicit runtime; alignment never loads a model implicitly.
    if runtime is None:
        raise ValueError(
            "MatchAnything requires a runtime from load_runtime(); no model was loaded"
        )
    pairs = infer_pairs(runtime, drawing_rgb, projection_rgb)
    diagnostics["native"] = pairs.diagnostics
    diagnostics["provenance"] = runtime.provenance

    # Fit and assess the requested transform using native full-resolution matches.
    matrix, warnings = _fit_matches(
        pairs, drawing_rgb.shape, projection_rgb.shape, model, options, diagnostics
    )
    return matrix, "uncertain" if warnings else "ok", warnings


def run_case(
    adapter,
    match_images,
    model,
    preset,
    rgb0,
    rgb1,
    output,
    provenance,
    args,
    case_name,
    self_pair=False,
):
    paths = output_paths(output)
    runtime = NativeRuntime(
        model, adapter, match_images, preset, provenance, args.threads
    )
    pairs = infer_pairs(runtime, rgb0, rgb1, capture_debug=True)
    p0, p1, scores, bounds = (
        pairs.points0,
        pairs.points1,
        pairs.confidence,
        pairs.in_bounds,
    )
    capture = pairs.debug
    elapsed = pairs.diagnostics["preprocess_and_model_seconds"]
    record = {
        "runner": "infer_pair_native.py",
        "runner_version": VERSION,
        "case": case_name,
        "implementation": "ORIGINAL_SPACE_ELoFTR_COARSE_AND_FINE",
        "num_matches": len(scores),
        "num_in_bounds": int(bounds.sum()),
        "postprocessing_filter": "None; all finite original refined pairs kept in upstream order",
        "coordinate_system": "float32 (x,y); EXIF-oriented original image pixels; Space's +0.5/-0.5 restoration",
        "geometric_filter": None,
        "original_wh": [list(rgb0.shape[:2][::-1]), list(rgb1.shape[:2][::-1])],
        "device": "cpu",
        "dtype": "float32",
        "threads": args.threads,
        "preset": preset,
        "native_diagnostics": capture.diagnostics,
        "input_tensor_shapes": {k: list(v.shape) for k, v in capture.inputs.items()},
        "input_tensor_minmax": {
            k: [float(v.min()), float(v.max())] for k, v in capture.inputs.items()
        },
        "preprocess_and_model_seconds": elapsed,
        "lifetime_peak_process_rss_mib": memory_peak_mib(),
        "rss_note": "Entire process high-water mark, including imports/loading/previous test cases; not incremental model memory.",
        "raw_note": "raw.npz is native AFTER fine matching, BEFORE original wrapper rescales. Suppressed coarse candidates are NOT saved.",
        "provenance": provenance,
    }
    if self_pair:
        record["identity_test"] = identity_metrics(capture.raw, p0, p1)
    write_npz(paths["npz"], points0=p0, points1=p1, confidence=scores, in_bounds=bounds)
    write_npz(paths["raw"], **capture.raw)
    for i in range(2):
        save_gray(paths[f"input{i}"], capture.inputs[f"image{i}"])
        if f"mask{i}" in capture.inputs:
            save_gray(paths[f"mask{i}"], capture.inputs[f"mask{i}"])
    draw_matches(
        paths["png"], rgb0, rgb1, p0, p1, scores, bounds, args.max_draw, case_name
    )
    write_json(paths["json"], record)
    log(
        f"[{case_name}] native refined={len(scores)}, in_bounds={int(bounds.sum())}, preprocess+forward={elapsed:.3f}s -> {paths['npz']}"
    )
    return {
        "case": case_name,
        "json": str(paths["json"]),
        "num_matches": len(scores),
        "num_in_bounds": int(bounds.sum()),
        "native_diagnostics": capture.diagnostics,
        "identity_test": record.get("identity_test"),
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Original MatchAnything Space ELoFTR on CPU, full coarse+fine (not Transformers).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("image0", type=Path, nargs="?")
    p.add_argument("image1", type=Path, nargs="?")
    p.add_argument("-o", "--output", type=Path, default=Path("native/matches.npz"))
    p.add_argument(
        "--diagnose",
        action="store_true",
        help="Run original-weight image0 self, image1 self, then pair; save diagnosis JSON",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override original coarse threshold; None reads Space preset (.001)",
    )
    p.add_argument(
        "--width",
        type=int,
        default=None,
        help="Override force-resize width; original is 640",
    )
    p.add_argument(
        "--height",
        type=int,
        default=None,
        help="Override force-resize height; original is 480",
    )
    p.add_argument(
        "--attention",
        choices=("full", "upstream"),
        default="full",
        help="full: original CPU FullAttention; upstream: keep original XFORMER selection",
    )
    p.add_argument("--threads", type=int, default=4)
    p.add_argument(
        "--max-draw",
        type=int,
        default=100,
        help="Visualization only; NPZ keeps all correspondences",
    )
    p.add_argument(
        "--cache-dir", type=Path, default=Path.home() / ".cache/matchanything-native"
    )
    p.add_argument(
        "--space-dir",
        type=Path,
        help="Use local original SPACE root containing imcui/ (trusted Python)",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        help="Use original .ckpt, not Transformers .safetensors",
    )
    p.add_argument(
        "--weights-zip",
        type=Path,
        help="Use author's downloaded weights.zip instead of downloading",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="Disable resource downloads; requires native source + ckpt already cached",
    )
    p.add_argument(
        "--allow-unsafe-checkpoint",
        action="store_true",
        help="Explicitly allow trusted checkpoint pickle execution; never automatic",
    )
    p.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare original source/checkpoint, without importing or running model",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--test-utils",
        action="store_true",
        help="No-download launcher tests with substitute data/models, NOT pretrained tests",
    )
    p.add_argument(
        "--check-kornia",
        action="store_true",
        help="Check installed Kornia grid imports/CPU coordinates only; no source or weight downloads",
    )
    p.add_argument("--traceback", action="store_true")
    return p


def validate_runtime_options(threads, width, height, threshold, attention) -> None:
    if not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be an integer >=1")
    if attention not in ("full", "upstream"):
        raise ValueError("attention must be 'full' or 'upstream'")
    for name, n in (("width", width), ("height", height)):
        if n is not None and not isinstance(n, int):
            raise ValueError(f"{name} must be an integer")
        if n is not None and (n < 128 or n % 32):
            raise ValueError(f"{name} must be >=128 and a multiple of 32")
    if threshold is not None and (
        not math.isfinite(threshold) or not 0 <= threshold < 1
    ):
        raise ValueError("threshold must be finite and in [0,1)")


def validate_args(args) -> None:
    validate_runtime_options(
        args.threads, args.width, args.height, args.threshold, args.attention
    )
    if args.max_draw < 0:
        raise ValueError("--max-draw must be >=0")
    if args.output.suffix.lower() != ".npz":
        raise ValueError("--output must end in .npz")
    if not args.prepare_only:
        if args.image0 is None or args.image1 is None:
            raise ValueError(
                "Provide two original input images, or use --prepare-only / --test-utils"
            )
        for image in (args.image0, args.image1):
            if not image.is_file():
                raise FileNotFoundError(image)


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.test_utils:
        return test_utils()
    if args.check_kornia:
        print(json.dumps(install_kornia_grid_compat(), indent=2))
        return 0
    validate_args(args)
    args.cache_dir = args.cache_dir.expanduser().resolve()
    stem = args.output.with_suffix("")
    diagnosis_path = Path(str(stem) + ".diagnosis.json")
    case_bases = [args.output]
    if args.diagnose:
        case_bases = [
            Path(str(stem) + "_self0.npz"),
            Path(str(stem) + "_self1.npz"),
            args.output,
        ]
    protected = [
        p for p in (args.image0, args.image1, args.checkpoint, args.weights_zip) if p
    ]
    if not args.prepare_only:
        targets = [p for base in case_bases for p in output_paths(base).values()] + [
            diagnosis_path
        ]
        guard_outputs(targets, protected, args.overwrite)
    if args.prepare_only:
        source = source_snapshot(args)
        checkpoint = get_checkpoint(args, source)
        log(
            f"Ready: source={source}\ncheckpoint={checkpoint}\nNo model inference performed."
        )
        return 0
    runtime = load_runtime(
        cache_dir=args.cache_dir,
        space_dir=args.space_dir,
        checkpoint=args.checkpoint,
        weights_zip=args.weights_zip,
        offline=args.offline,
        allow_unsafe_checkpoint=args.allow_unsafe_checkpoint,
        threshold=args.threshold,
        width=args.width,
        height=args.height,
        attention=args.attention,
        threads=args.threads,
    )
    model, adapter, match_images = runtime.model, runtime.adapter, runtime.match_images
    preset, provenance = runtime.preset, runtime.provenance
    rgb0, rgb1 = read_rgb(args.image0), read_rgb(args.image1)
    cases = []
    planned = [("pair", rgb0, rgb1, args.output, False)]
    if args.diagnose:
        planned = [
            ("self0", rgb0, rgb0, case_bases[0], True),
            ("self1", rgb1, rgb1, case_bases[1], True),
            *planned,
        ]
    for name, a, b, out, self_pair in planned:
        try:
            item = run_case(
                adapter,
                match_images,
                model,
                preset,
                a,
                b,
                out,
                provenance,
                args,
                name,
                self_pair,
            )
            cases.append(item)
        except Exception as exc:
            cases.append(
                {"case": name, "error": repr(exc), "traceback": traceback.format_exc()}
            )
            write_json(
                diagnosis_path,
                {"status": "execution_error", "cases": cases, "provenance": provenance},
            )
            raise
        write_json(
            diagnosis_path,
            {
                "status": "running" if len(cases) < len(planned) else "completed",
                "cases": cases,
                "provenance": provenance,
            },
        )
    failed = [
        c["case"]
        for c in cases
        if c.get("identity_test") and not c["identity_test"]["smoke_pass"]
    ]
    if failed:
        log(
            "WARNING: identity smoke criteria not met: "
            + ", ".join(failed)
            + ". Do not treat nonzero matches as correctness."
        )
    log(f"[summary] {diagnosis_path}")
    return 0


def test_utils() -> int:
    """No-download tests. No real model is loaded by this mode."""
    import unittest

    import numpy as np
    import torch
    from PIL import Image

    class Tests(unittest.TestCase):
        def setUp(self):
            self.tmp = tempfile.TemporaryDirectory()
            self.root = Path(self.tmp.name)

        def tearDown(self):
            self.tmp.cleanup()

        def test_ast_isolates_unrelated_imports_and_preserves_body(self):
            f = self.root / "source.py"
            f.write_text(
                "import nonexistent_training_dependency\nraise RuntimeError('top level')\n"
                "def helper(x):\n    return x+0.25\n"
                "class C:\n    def forward(self,x):\n        return helper(x)*self.scale\n",
                encoding="utf-8",
            )
            funcs = selected_definitions(f, ["helper", "C.forward"], {})
            self.assertEqual(funcs["C.forward"](SimpleNamespace(scale=2), 3), 6.5)

        def test_missing_original_definition_fails(self):
            f = self.root / "source.py"
            f.write_text("def f(): return 1\n")
            with self.assertRaises(RuntimeError):
                selected_definitions(f, ["C.forward"], {})

        def test_preset_literal_only(self):
            f = self.root / "dense.py"
            f.write_text(
                "confs={'bad': make_bad(), 'matchanything_eloftr': {'model': {'model_name': 'matchanything_eloftr', 'match_threshold': .001}}}\n"
            )
            self.assertEqual(read_space_preset(f)["model"]["match_threshold"], 0.001)

        def test_zip_only_selected_weight(self):
            z = self.root / "a.zip"
            dest = self.root / "model.ckpt"
            with zipfile.ZipFile(z, "w") as f:
                f.writestr("weights/" + CHECKPOINT_NAME, b"correct")
                f.writestr("../../unrelated.txt", b"never extracted")
            extract_eloftr_checkpoint(z, dest)
            self.assertEqual(dest.read_bytes(), b"correct")
            self.assertEqual(
                sorted(p.name for p in self.root.iterdir()), ["a.zip", "model.ckpt"]
            )

        def test_zip_traversal_and_duplicate_rejected(self):
            for members in (
                ("../" + CHECKPOINT_NAME,),
                ("a/" + CHECKPOINT_NAME, "b/" + CHECKPOINT_NAME),
            ):
                z = self.root / "a.zip"
                with zipfile.ZipFile(z, "w") as f:
                    for name in members:
                        f.writestr(name, b"x")
                with self.assertRaises(ValueError):
                    extract_eloftr_checkpoint(z, self.root / "out.ckpt")

        def test_safe_checkpoint_all_parameters_load(self):
            model = torch.nn.Linear(3, 2)
            state = {
                "matcher." + k: torch.ones_like(v)
                for k, v in model.state_dict().items()
            }
            path = self.root / "model.ckpt"
            torch.save({"state_dict": state}, path)
            result = load_weights_checked(model, path, False)
            self.assertEqual(result["loaded_state_entries"], 2)
            self.assertTrue(torch.equal(model.weight, torch.ones_like(model.weight)))

        def test_missing_parameter_fails(self):
            model = torch.nn.Linear(3, 2)
            path = self.root / "model.ckpt"
            torch.save({"state_dict": {"matcher.bias": torch.zeros(2)}}, path)
            with self.assertRaises(RuntimeError):
                load_weights_checked(model, path, False)

        def test_transparent_image_white(self):
            f = self.root / "input.png"
            Image.new("RGBA", (5, 7), (0, 0, 0, 0)).save(f)
            self.assertTrue(np.all(read_rgb(f) == 255))

        def test_output_protection(self):
            f = self.root / "input.png"
            f.touch()
            with self.assertRaises(ValueError):
                guard_outputs([f], [f], True)
            with self.assertRaises(FileExistsError):
                guard_outputs([f], [], False)

        def test_pairs_preserve_fractional_coords_and_mark_bounds(self):
            p = np.array([[1.25, 2.5], [-1.0, 0.0]], dtype=np.float32)
            r = {
                "mkeypoints0_orig": p,
                "mkeypoints1_orig": p.copy(),
                "mconf": [0.3, 0.2],
            }
            a, _b, _s, valid = checked_pairs(r, [(10, 10), (10, 10)])
            np.testing.assert_array_equal(a, p)
            self.assertEqual(valid.tolist(), [True, False])
            self.assertEqual(a.dtype, np.float32)

        def test_empty_output_and_preview(self):
            a = np.empty((0, 2), np.float32)
            s = np.empty(0, np.float32)
            v = np.empty(0, bool)
            write_npz(self.root / "out.npz", points0=a, points1=a, confidence=s)
            with np.load(self.root / "out.npz", allow_pickle=False) as d:
                self.assertEqual(d["points0"].shape, (0, 2))
            rgb = np.full((40, 60, 3), 255, np.uint8)
            draw_matches(self.root / "out.png", rgb, rgb, a, a, s, v, 10, "MOCK_EMPTY")
            with Image.open(self.root / "out.png") as im:
                self.assertGreater(im.width, 0)

        def test_hooks_snapshot_before_cpu_inplace_mutation(self):
            class Stage(torch.nn.Module):
                def forward(self, data):
                    pass

            class Fake(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.coarse_matching = Stage()
                    self.fine_matching = Stage()

                def forward(self, d):
                    d["mconf"] = torch.tensor([0.5])
                    self.coarse_matching(d)
                    d["mkpts0_f"] = torch.tensor([[3.25, 4.5]])
                    d["mkpts1_f"] = d["mkpts0_f"].clone()
                    self.fine_matching(d)

            net = Fake()
            data = {
                "image0": torch.zeros(1, 1, 8, 8),
                "image1": torch.zeros(1, 1, 8, 8),
            }
            with NativeCapture(net) as c:
                net(data)
            data["mkpts0_f"] *= 5
            np.testing.assert_allclose(c.raw["mkpts0_f"], [[3.25, 4.5]])
            self.assertEqual(c.diagnostics["fine_forward_calls"], 1)
            self.assertEqual(len(net._forward_hooks), 0)

        def test_reusable_inference_and_cli_artifacts(self):
            class Stage(torch.nn.Module):
                def forward(self, data):
                    pass

            class Fake(torch.nn.Module):
                calls = 0
                skip_fine = False

                def __init__(self):
                    super().__init__()
                    self.coarse_matching = Stage()
                    self.fine_matching = Stage()

                def forward(self, d):
                    self.calls += 1
                    d["mconf"] = torch.tensor([0.5, 0.2])
                    self.coarse_matching(d)
                    d["mkpts0_f"] = torch.tensor([[3.25, 4.5], [-1.0, 0.0]])
                    d["mkpts1_f"] = d["mkpts0_f"].clone()
                    if not self.skip_fine:
                        self.fine_matching(d)

            net = Fake()

            def match_images(adapter, a, b, preprocessing, device):
                self.assertEqual(device, "cpu")
                self.assertFalse(torch.is_grad_enabled())
                preprocessing["nested"]["value"] = 2
                data = {
                    "image0": torch.zeros(1, 1, 8, 8),
                    "image1": torch.zeros(1, 1, 8, 8),
                }
                adapter(data)
                data["mkpts0_f"] *= 2
                data["mkpts1_f"] *= 2
                return {
                    "mkeypoints0_orig": data["mkpts0_f"].numpy(),
                    "mkeypoints1_orig": data["mkpts1_f"].numpy(),
                    "mconf": data["mconf"].numpy(),
                }

            preset = {"preprocessing": {"nested": {"value": 1}}}
            runtime = NativeRuntime(net, net, match_images, preset, {}, 4)
            rgb = np.full((20, 20, 3), 255, np.uint8)
            for _ in range(2):
                pairs = infer_pairs(runtime, rgb, rgb)
                np.testing.assert_allclose(pairs.points0, [[6.5, 9.0], [-2.0, 0.0]])
                self.assertEqual(pairs.in_bounds.tolist(), [True, False])
                self.assertIsNone(pairs.debug)
                self.assertEqual(pairs.diagnostics["fine_forward_calls"], 1)
            self.assertEqual(net.calls, 2)
            self.assertEqual(preset["preprocessing"]["nested"]["value"], 1)
            self.assertEqual(list(self.root.iterdir()), [])
            for invalid in (rgb.astype(float), rgb[:, :, 0], rgb[:0]):
                with self.assertRaises(ValueError):
                    infer_pairs(runtime, invalid, rgb)
            output = self.root / "matches.npz"
            run_case(
                net,
                match_images,
                net,
                preset,
                rgb,
                rgb,
                output,
                {},
                SimpleNamespace(threads=4, max_draw=5),
                "self0",
                self_pair=True,
            )
            paths = output_paths(output)
            with np.load(paths["npz"]) as saved:
                np.testing.assert_array_equal(saved["points0"], pairs.points0)
            with np.load(paths["raw"]) as raw:
                np.testing.assert_allclose(raw["mkpts0_f"], [[3.25, 4.5], [-1.0, 0.0]])
            record = json.loads(paths["json"].read_text())
            self.assertEqual(record["num_matches"], 2)
            self.assertEqual(record["input_tensor_shapes"]["image0"], [1, 1, 8, 8])
            for name in ("png", "input0", "input1"):
                self.assertTrue(paths[name].is_file())
            net.skip_fine = True
            with self.assertRaisesRegex(RuntimeError, "fine matching was not invoked"):
                infer_pairs(runtime, rgb, rgb)
            self.assertEqual(len(net._forward_hooks), 0)
            self.assertEqual(len(net.coarse_matching._forward_hooks), 0)

        def test_identity_metrics_not_count_only(self):
            a = np.zeros((10, 2), np.float32)
            raw = {"mkpts0_f": a, "mkpts1_f": a.copy()}
            self.assertTrue(identity_metrics(raw, a, a)["smoke_pass"])
            raw["mkpts1_f"] = a + 5
            self.assertFalse(identity_metrics(raw, a, a + 5)["smoke_pass"])

        def test_bad_arguments(self):
            p = parser()
            for tokens in (
                ["--width", "127"],
                ["--threshold", "nan"],
                ["--threads", "0"],
            ):
                with self.assertRaises(ValueError):
                    validate_args(p.parse_args(tokens))

    log("Launcher utility tests only; NO original pretrained inference is executed.")
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("Interrupted.")
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001 - CLI logs failures and exits with code 1.
        log(f"ERROR: {type(exc).__name__}: {exc}")
        if "--traceback" in sys.argv:
            traceback.print_exc()
        raise SystemExit(1)
