"""Dataset records joining DXF primitives, raster images, and CadQuery targets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from PIL import Image
from torch.utils.data import Dataset

from .dxf import DXFPrimitiveData, DXFPrimitiveParser
from .layout import TECHDRAW_DXF_SUBDIR, code_target_path


@dataclass(frozen=True)
class RasterImageSource:
    """Semantic image style and its directory relative to the dataset root."""

    style: str
    directory: Path

    def __init__(self, style: str, directory: str | Path) -> None:
        if not isinstance(style, str) or not style.strip():
            raise ValueError("raster image style must be a non-empty string")
        path = Path(directory)
        if path.is_absolute():
            raise ValueError("raster image source directory must be relative to root")
        object.__setattr__(self, "style", style)
        object.__setattr__(self, "directory", path)


DEFAULT_IMAGE_SOURCES = (RasterImageSource("isometric", "render_3d/hlg_perspective"),)
# Kept as a convenient public semantic vocabulary value. Paths are configured
# independently through RasterImageSource.
DEFAULT_IMAGE_STYLES = tuple(source.style for source in DEFAULT_IMAGE_SOURCES)


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


class Drawing2CADDataset(Dataset):
    """Load semantic, unpadded samples and optionally apply a sample transform.

    The dataset is handed the ids it should serve; deciding which samples exist
    belongs to :func:`src.data.layout.take_inventory`, which owns the
    completeness rule for the whole codebase. Each primitive's view is read from
    its DXF layer, so no sidecar metadata is needed either.
    """

    def __init__(
        self,
        root: str | Path,
        sample_ids: Sequence[str],
        *,
        dxf_parser: DXFPrimitiveParser | None = None,
        image_sources: Sequence[RasterImageSource] = DEFAULT_IMAGE_SOURCES,
        include_target: bool = True,
        image_max_edge: int | None = None,
        transform: Callable[[Drawing2CADSample], Any] | None = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"dataset root does not exist: {self.root}")
        self.dxf_parser = dxf_parser or DXFPrimitiveParser()
        self.image_sources = tuple(image_sources)
        self.image_styles = tuple(source.style for source in self.image_sources)
        self.include_target = include_target
        self.image_max_edge = image_max_edge
        self.transform = transform

        if not self.image_sources:
            raise ValueError("at least one raster image source is required")
        if len(set(self.image_styles)) != len(self.image_styles):
            raise ValueError(
                f"raster image styles must be unique, got {self.image_styles}"
            )
        if image_max_edge is not None and image_max_edge <= 0:
            raise ValueError("image_max_edge must be positive when provided")

        selected = tuple(sample_ids)
        if not selected:
            raise ValueError(f"no sample ids were selected for {self.root}")
        if len(set(selected)) != len(selected):
            raise ValueError("sample_ids must not contain duplicates")

        self.records = tuple(
            Drawing2CADRecord(
                sample_id=sample_id,
                dxf_path=self.root / TECHDRAW_DXF_SUBDIR / f"{sample_id}.dxf",
                image_paths=tuple(
                    self.root / source.directory / f"{sample_id}.png"
                    for source in self.image_sources
                ),
                target_path=(
                    code_target_path(self.root, sample_id) if include_target else None
                ),
            )
            for sample_id in selected
        )

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

    def load_sample(self, index: int) -> Drawing2CADSample:
        """Load one semantic sample without applying ``transform``."""
        record = self.records[index]
        primitives = self.dxf_parser.parse(
            record.dxf_path,
            sample_id=record.sample_id,
        )
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

    def __getitem__(self, index: int):
        sample = self.load_sample(index)
        if self.transform is None:
            return sample
        return self.transform(sample)


__all__ = [
    "DEFAULT_IMAGE_SOURCES",
    "DEFAULT_IMAGE_STYLES",
    "Drawing2CADDataset",
    "Drawing2CADRecord",
    "Drawing2CADSample",
    "RasterImageSource",
]
