"""The drawing files a run holds: the ones it was handed, and the ones a
verification drew. Registering one means measuring the file it names.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from types import MappingProxyType
from typing import Any

import ezdxf
from ezdxf import bbox
from PIL import Image

from zeroshot.pipeline.stages.interpretation.contracts import (
    DrawingView,
    Region,
    View,
    require_unique,
)

DRAWING_SUFFIXES = frozenset({".dxf", ".png", ".jpg", ".jpeg"})


def read_dxf_frame(path: Path, mm_per_unit: float) -> dict[str, Any]:
    """Input metadata to expose before asking a model for native-DXF regions.

    A DXF unit need not be a millimetre. The input adapter must supply its
    physical conversion, accounting for the dataset's drawing-scale convention.
    For a native point (x, y), u = (x - origin_x) * mm_per_unit, and likewise v.
    The source file itself is preserved; its entity order is irrelevant.
    """
    if not math.isfinite(mm_per_unit) or mm_per_unit <= 0:
        raise ValueError("DXF mm_per_unit must be finite and positive")
    bounds = bbox.extents(ezdxf.readfile(path).modelspace())
    if not bounds.has_data:
        raise ValueError(f"{path}: DXF has no measurable sheet bounds")
    if not all(
        math.isfinite(value)
        for point in (bounds.extmin, bounds.extmax)
        for value in point
    ):
        raise ValueError(f"{path}: DXF bounds must be finite")
    if abs(bounds.extmin.z) > 1e-6 or abs(bounds.extmax.z) > 1e-6:
        raise ValueError(f"{path}: expected a planar XY drawing")
    size = (bounds.size.x * mm_per_unit, bounds.size.y * mm_per_unit)
    if any(not math.isfinite(value) or value <= 0 for value in size):
        raise ValueError(
            f"{path}: DXF must have positive finite sheet width and height"
        )
    return {
        "origin_native": (bounds.extmin.x, bounds.extmin.y),
        "mm_per_unit": mm_per_unit,
        "size_mm": size,
    }


def register_view(
    name: str,
    role: View,
    file: str | PurePath,
    mm_per_unit: float | None = None,
) -> DrawingView:
    """Address one drawing file as a view, its region covering the whole of it."""
    path = Path(file)
    if path.suffix.lower() not in DRAWING_SUFFIXES:
        raise ValueError(f"unsupported drawing file: {path}")
    if path.suffix.lower() == ".dxf":
        if mm_per_unit is None:
            raise ValueError(f"{name}: a native DXF input needs its mm_per_unit")
        width, height = read_dxf_frame(path, mm_per_unit)["size_mm"]
        region = Region(view=name, box_uv=(0.0, 0.0, width, height))
    else:
        with Image.open(path) as image:
            width, height = image.size
        region = Region(view=name, box_px=(0, 0, width, height))
    return DrawingView(
        name=name, role=role, file=str(file), region=region, dimensions=[]
    )


def _safe_identifier(name: str, field_name: str) -> str:
    stripped = name.strip()
    if not stripped:
        raise ValueError(f"{field_name} must not be empty")
    if stripped in {".", ".."} or "/" in stripped or "\\" in stripped:
        raise ValueError(f"unsafe {field_name}: {name!r}")
    return stripped


def _present(files: Iterable[str]) -> None:
    for path in map(Path, files):
        if path.suffix.lower() not in DRAWING_SUFFIXES:
            raise ValueError(f"unsupported drawing file: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Not Found: {path}")


@dataclass(frozen=True)
class InputManifest:
    """The drawings a sample is made of.

    Views rather than a path because a sample may arrive as one sheet, as one
    file per view, as DXF, as PNG, or as a mixture, and every stage after this
    one should be unable to tell which. A perspective render offered alongside
    the drawing is a view like any other.
    """

    sample_id: str
    drawing: Sequence[DrawingView]

    def __post_init__(self) -> None:
        if not self.drawing:
            raise ValueError("a sample needs at least one view")
        require_unique((view.name for view in self.drawing), "input views")
        _present(view.file for view in self.drawing)
        object.__setattr__(
            self, "sample_id", _safe_identifier(self.sample_id, "sample_id")
        )


@dataclass(frozen=True)
class FeedbackManifest:
    """What a verification drew of the solid it built, and what it could not.

    Each view covers its rendered file. `errors` is keyed by the name the
    view would have been announced under, because a file that was never
    produced cannot carry its own explanation.
    """

    verification_id: str
    drawing: Sequence[DrawingView] = ()
    errors: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        errors = MappingProxyType(dict(self.errors))
        _present(sheet.file for sheet in self.drawing)
        # An artifact is either present or explained, never both.
        drawn = {sheet.name for sheet in self.drawing}
        if both := sorted(drawn & set(errors)):
            raise ValueError(f"sheets are both drawn and failed: {both}")

        object.__setattr__(
            self,
            "verification_id",
            _safe_identifier(self.verification_id, "verification_id"),
        )
        object.__setattr__(self, "errors", errors)
