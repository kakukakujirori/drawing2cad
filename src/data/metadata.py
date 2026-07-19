"""Typed metadata boundary for drawing samples.

Only this module knows the current ``manifest.jsonl`` layout.  Dataset and DXF
code consume :class:`SampleMetadata`, allowing future datasets to obtain the
same information from a sidecar, database, or another manifest schema.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Protocol, runtime_checkable


VIEW_DIRECTIONS = ("front", "top", "right")
VIEW_DIRECTION_TO_ID = {
    direction: index for index, direction in enumerate(VIEW_DIRECTIONS)
}


class MetadataError(ValueError):
    """Raised when sample metadata violates the model-independent contract."""


@dataclass(frozen=True)
class ViewBBox:
    """One orthographic view bounding box in raw DXF sheet coordinates."""

    direction: str
    xyxy_mm: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if self.direction not in VIEW_DIRECTION_TO_ID:
            raise MetadataError(
                f"unsupported view direction {self.direction!r}; "
                f"expected one of {VIEW_DIRECTIONS}"
            )
        if len(self.xyxy_mm) != 4:
            raise MetadataError("view bbox must contain exactly four xyxy values")
        values = tuple(float(value) for value in self.xyxy_mm)
        if not all(math.isfinite(value) for value in values):
            raise MetadataError(f"view bbox must be finite, got {values}")
        x_min, y_min, x_max, y_max = values
        if x_min >= x_max or y_min >= y_max:
            raise MetadataError(
                f"view bbox must have positive area in xyxy order, got {values}"
            )
        object.__setattr__(self, "xyxy_mm", values)

    @property
    def direction_id(self) -> int:
        return VIEW_DIRECTION_TO_ID[self.direction]

    def contains(self, x: float, y: float, *, tolerance_mm: float = 0.0) -> bool:
        if tolerance_mm < 0:
            raise ValueError("bbox tolerance must be non-negative")
        x_min, y_min, x_max, y_max = self.xyxy_mm
        return (
            x_min - tolerance_mm <= x <= x_max + tolerance_mm
            and y_min - tolerance_mm <= y <= y_max + tolerance_mm
        )


@dataclass(frozen=True)
class SampleMetadata:
    """Format-independent metadata required to construct one dataset sample."""

    sample_id: str
    view_bboxes: tuple[ViewBBox, ...]
    drawing_scale: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise MetadataError("sample_id must be a non-empty string")
        bboxes = tuple(self.view_bboxes)
        directions = tuple(bbox.direction for bbox in bboxes)
        if len(bboxes) != len(VIEW_DIRECTIONS) or set(directions) != set(
            VIEW_DIRECTIONS
        ):
            raise MetadataError(
                "sample metadata must contain exactly one bbox for each direction "
                f"{VIEW_DIRECTIONS}, got {directions}"
            )
        # Canonical ordering makes IDs and audit output deterministic even if an
        # upstream format stores view records in another order.
        ordered = tuple(
            next(bbox for bbox in bboxes if bbox.direction == direction)
            for direction in VIEW_DIRECTIONS
        )
        object.__setattr__(self, "view_bboxes", ordered)
        if self.drawing_scale is not None:
            scale = float(self.drawing_scale)
            if not math.isfinite(scale) or scale <= 0:
                raise MetadataError(
                    f"drawing_scale must be finite and positive, got {scale}"
                )
            object.__setattr__(self, "drawing_scale", scale)


@runtime_checkable
class SampleMetadataProvider(Protocol):
    """Source of typed per-sample metadata.

    Providers may expose a ``sample_ids`` property for dataset discovery, but
    it is deliberately not required: callers can supply ``sample_ids`` to the
    dataset when metadata lives in a random-access source without enumeration.
    """

    def get(self, sample_id: str) -> SampleMetadata: ...


class ManifestSampleMetadataProvider:
    """Adapter for the current generated-dataset ``manifest.jsonl`` schema."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"dataset manifest does not exist: {self.manifest_path}"
            )

        records: dict[str, SampleMetadata] = {}
        ordered_ids: list[str] = []
        for line_number, line in enumerate(
            self.manifest_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise MetadataError(
                    f"invalid JSON in {self.manifest_path}:{line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise MetadataError(
                    f"manifest row {self.manifest_path}:{line_number} must be an object"
                )
            sample_id = row.get("name")
            if not isinstance(sample_id, str) or not sample_id:
                raise MetadataError(
                    f"manifest row {self.manifest_path}:{line_number} has no valid name"
                )
            if not row.get("ok", False):
                continue
            try:
                metadata = self._parse_row(sample_id, row)
            except (KeyError, TypeError, MetadataError, ValueError) as error:
                if isinstance(error, MetadataError):
                    detail = str(error)
                else:
                    detail = f"missing or invalid field: {error}"
                raise MetadataError(
                    f"invalid metadata for sample {sample_id!r} at "
                    f"{self.manifest_path}:{line_number}: {detail}"
                ) from error
            if sample_id not in records:
                ordered_ids.append(sample_id)
            records[sample_id] = metadata

        if not records:
            raise MetadataError(
                f"manifest contains no successful rows with view metadata: "
                f"{self.manifest_path}"
            )
        self._records = records
        self._sample_ids = tuple(ordered_ids)

    @staticmethod
    def _parse_row(sample_id: str, row: dict) -> SampleMetadata:
        techdraw = row["extra"]["techdraw"]
        if not isinstance(techdraw, dict):
            raise MetadataError("extra.techdraw must be an object")
        if techdraw.get("bbox_format") != "xyxy":
            raise MetadataError(
                "extra.techdraw.bbox_format must be exactly 'xyxy'"
            )
        coordinate_system = techdraw.get("bbox_coordinate_system")
        expected_coordinate_system = {
            "unit": "mm",
            "origin": "sheet_bottom_left",
            "x_axis": "right",
            "y_axis": "up",
        }
        if coordinate_system != expected_coordinate_system:
            raise MetadataError(
                "extra.techdraw.bbox_coordinate_system must equal "
                f"{expected_coordinate_system}, got {coordinate_system!r}"
            )
        views = techdraw.get("views")
        if not isinstance(views, dict) or set(views) != set(VIEW_DIRECTIONS):
            raise MetadataError(
                "extra.techdraw.views must contain exactly "
                f"{VIEW_DIRECTIONS}, got "
                f"{tuple(views) if isinstance(views, dict) else views!r}"
            )
        bboxes: list[ViewBBox] = []
        for direction in VIEW_DIRECTIONS:
            view = views[direction]
            if not isinstance(view, dict):
                raise MetadataError(f"view {direction!r} must be an object")
            bbox = view.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                raise MetadataError(
                    f"view {direction!r} bbox must be a four-value sequence"
                )
            bboxes.append(ViewBBox(direction, tuple(float(value) for value in bbox)))
        return SampleMetadata(
            sample_id=sample_id,
            view_bboxes=tuple(bboxes),
            drawing_scale=techdraw.get("scale"),
        )

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return self._sample_ids

    def get(self, sample_id: str) -> SampleMetadata:
        try:
            return self._records[sample_id]
        except KeyError as error:
            raise KeyError(
                f"sample {sample_id!r} is absent from successful manifest rows in "
                f"{self.manifest_path}"
            ) from error


__all__ = [
    "ManifestSampleMetadataProvider",
    "MetadataError",
    "SampleMetadata",
    "SampleMetadataProvider",
    "VIEW_DIRECTIONS",
    "VIEW_DIRECTION_TO_ID",
    "ViewBBox",
]
