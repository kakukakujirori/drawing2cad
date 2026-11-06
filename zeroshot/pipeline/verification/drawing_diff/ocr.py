"""Read the text lines of an image with OCR, whichever way they run.

A comparison runs in a new process for each build while its input drawing stays
the same, so the lines are kept on disk under the image's content.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class TextLine:
    corners: list[list[float]]  # (x, y) around the line, in the image's pixels
    string: str
    confidence: float


def read_text_lines(rgb: np.ndarray) -> list[TextLine]:
    """Every text line of an RGB image, read once per image content."""
    path = _cache_path(rgb)
    if path.is_file():
        return [TextLine(**line) for line in json.loads(path.read_text("utf-8"))]
    lines = _read_every_way(rgb)
    _store(path, lines)
    return lines


def _read_every_way(rgb: np.ndarray) -> list[TextLine]:
    """OCR reads a line upright or upside down, so a quarter turn reaches the rest."""
    height = rgb.shape[0]
    turned = _read(cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE))
    # A pixel (x, y) of the turned image is (y, height - 1 - x) of the image.
    return _read(rgb) + [
        replace(line, corners=[[y, height - 1 - x] for x, y in line.corners])
        for line in turned
    ]


def _read(rgb: np.ndarray) -> list[TextLine]:
    result = _engine()(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if result.boxes is None:
        return []
    return [
        TextLine(box.tolist(), string, float(confidence))
        for box, string, confidence in zip(
            result.boxes, result.txts, result.scores, strict=True
        )
    ]


@lru_cache(maxsize=1)
def _engine() -> Any:
    # Imported here: a comparison whose text is cached never needs it.
    from rapidocr import RapidOCR

    # One thread: comparisons of many samples run side by side.
    return RapidOCR(
        params={
            "Global.log_level": "error",
            "EngineConfig.onnxruntime.intra_op_num_threads": 1,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        }
    )


def _cache_path(rgb: np.ndarray) -> Path:
    digest = hashlib.sha256(str(rgb.shape).encode() + rgb.tobytes()).hexdigest()
    root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return root / "zeroshot" / f"rapidocr-{version('rapidocr')}" / f"{digest}.json"


def _store(path: Path, lines: list[TextLine]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Renamed into place: comparisons of one drawing can run side by side.
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps([asdict(line) for line in lines]), "utf-8")
    temporary.replace(path)
