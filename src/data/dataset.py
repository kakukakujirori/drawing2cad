"""Dataset records joining DXF primitives, raster views, and CadQuery targets."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

from PIL import Image
from torch.utils.data import Dataset

from .dxf import DXFPrimitiveData, DXFPrimitiveParser


DEFAULT_IMAGE_STYLES = ("hlg_perspective",)


@dataclass(frozen=True)
class Drawing2CADRecord:
    sample_id: str
    dxf_path: Path
    image_paths: tuple[Path, ...]
    target_path: Path | None


@dataclass(frozen=True)
class Drawing2CADSample:
    sample_id: str
    primitives: DXFPrimitiveData
    images: tuple[Image.Image, ...]
    image_styles: tuple[str, ...]
    target_code: str | None


class Drawing2CADDataset(Dataset[Drawing2CADSample]):
    """Read the generated ECCV-style multimodal sample tree.

    The combined orthographic DXF sheet is one primitive-bearing semantic
    ``drawing`` view. Perspective PNGs are native Qwen image inputs labelled as
    ``isometric`` in the collator prompt.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        dxf_parser: DXFPrimitiveParser | None = None,
        image_styles: Sequence[str] = DEFAULT_IMAGE_STYLES,
        include_target: bool = True,
        sample_ids: Sequence[str] | None = None,
        max_samples: int | None = None,
        strict_files: bool = True,
        image_max_edge: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.dxf_parser = dxf_parser or DXFPrimitiveParser()
        self.image_styles = tuple(image_styles)
        self.include_target = include_target
        self.image_max_edge = image_max_edge
        if not self.root.is_dir():
            raise FileNotFoundError(f"dataset root does not exist: {self.root}")
        if not self.image_styles:
            raise ValueError("at least one image style is required")
        if len(set(self.image_styles)) != len(self.image_styles):
            raise ValueError(f"image styles must be unique, got {self.image_styles}")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive when provided")
        if image_max_edge is not None and image_max_edge <= 0:
            raise ValueError("image_max_edge must be positive when provided")

        manifest_path = self.root / "manifest.jsonl"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"dataset manifest does not exist: {manifest_path}")
        successful: dict[str, dict] = {}
        ordered_ids: list[str] = []
        for line_number, line in enumerate(
            manifest_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                if strict_files:
                    raise ValueError(
                        f"invalid JSON in {manifest_path}:{line_number}: {error}"
                    ) from error
                continue
            name = record.get("name")
            if not isinstance(name, str) or not name or not record.get("ok", False):
                continue
            if name not in successful:
                ordered_ids.append(name)
            successful[name] = record

        if sample_ids is not None:
            selected_ids = list(sample_ids)
            missing_ids = [name for name in selected_ids if name not in successful]
            if missing_ids:
                raise KeyError(
                    f"requested sample IDs are absent from successful manifest rows: {missing_ids}"
                )
        else:
            selected_ids = ordered_ids
        if max_samples is not None:
            selected_ids = selected_ids[:max_samples]

        records: list[Drawing2CADRecord] = []
        skipped: list[str] = []
        for sample_id in selected_ids:
            dxf_path = self.root / "techdraw" / "dxf" / f"{sample_id}.dxf"
            image_paths = tuple(
                self.root / "render_3d" / style / f"{sample_id}.png"
                for style in self.image_styles
            )
            target_path = (
                self.root / "target" / f"{sample_id}.cadquery.py"
                if include_target
                else None
            )
            required_paths = [dxf_path, *image_paths]
            if target_path is not None:
                required_paths.append(target_path)
            missing_paths = [path for path in required_paths if not path.is_file()]
            if missing_paths:
                if strict_files:
                    raise FileNotFoundError(
                        f"sample {sample_id} is missing required files: {missing_paths}"
                    )
                skipped.append(sample_id)
                continue
            records.append(
                Drawing2CADRecord(
                    sample_id=sample_id,
                    dxf_path=dxf_path,
                    image_paths=image_paths,
                    target_path=target_path,
                )
            )

        if not records:
            suffix = f"; skipped missing samples: {skipped}" if skipped else ""
            raise ValueError(f"dataset contains no usable records{suffix}")
        self.records = tuple(records)
        self.skipped_sample_ids = tuple(skipped)

    def __len__(self) -> int:
        return len(self.records)

    def _load_rgb(self, path: Path) -> Image.Image:
        with Image.open(path) as image:
            output = image.convert("RGB")
            if self.image_max_edge is not None:
                output.thumbnail(
                    (self.image_max_edge, self.image_max_edge),
                    Image.Resampling.LANCZOS,
                )
            return output.copy()

    def __getitem__(self, index: int) -> Drawing2CADSample:
        record = self.records[index]
        primitives = self.dxf_parser.parse(record.dxf_path)
        images = tuple(self._load_rgb(path) for path in record.image_paths)
        target_code = (
            None
            if record.target_path is None
            else record.target_path.read_text(encoding="utf-8")
        )
        return Drawing2CADSample(
            sample_id=record.sample_id,
            primitives=primitives,
            images=images,
            image_styles=self.image_styles,
            target_code=target_code,
        )


__all__ = [
    "DEFAULT_IMAGE_STYLES",
    "Drawing2CADDataset",
    "Drawing2CADRecord",
    "Drawing2CADSample",
]
