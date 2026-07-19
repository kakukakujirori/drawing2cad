"""DXF entity parsing and uniform 2D primitive sampling."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Sequence

import ezdxf
from ezdxf.entities import DXFEntity
import torch

from .metadata import VIEW_DIRECTIONS, ViewBBox


# Stable IDs shared by the dataset and PrimitiveEncoderConfig. New entity types
# must be appended so existing checkpoints keep their meaning.
DXF_PRIMITIVE_TYPES = (
    "LINE",
    "ARC",
    "CIRCLE",
    "ELLIPSE",
    "SPLINE",
    "LWPOLYLINE",
    "POLYLINE",
)
DXF_PRIMITIVE_TYPE_TO_ID = {
    entity_type: index for index, entity_type in enumerate(DXF_PRIMITIVE_TYPES)
}
DXF_SAMPLE_FEATURE_NAMES = (
    "x",
    "y",
    "visibility",
    "tangent_x",
    "tangent_y",
    "curvature",
    "arc_length_position",
)
# Channels that follow the traversal direction and must be negated whenever a
# sample sequence is reversed (model-side reversal symmetrization relies on
# this contract through PrimitiveEncoderConfig.oriented_feature_indices).
DXF_ORIENTED_SAMPLE_FEATURE_INDICES = (3, 4, 5, 6)


class DXFParseError(ValueError):
    """Raised when a drawing cannot provide a valid primitive tensor."""


@dataclass(frozen=True)
class DXFPrimitiveConfig:
    """Configuration for parser-side primitive feature construction.

    Coordinates are transformed by one dataset-wide sheet transform, never by
    a sample-local bounding box. With the defaults, the 297 mm sheet width maps
    to ``[-0.9, 0.9]`` and its aspect ratio is retained.
    """

    samples_per_primitive: int = 16
    spline_flattening_distance_mm: float = 0.05
    included_layers: tuple[str, ...] = ("0",)
    include_visible: bool = True
    include_hidden: bool = True
    sheet_width_mm: float = 297.0
    sheet_height_mm: float = 210.0
    normalized_longest_side: float = 1.8
    discard_outside_sheet: bool = True
    sheet_tolerance_mm: float = 1.0
    view_bbox_tolerance_mm: float = 0.1
    strict_entity_types: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "included_layers", tuple(self.included_layers))
        if self.samples_per_primitive < 2:
            raise ValueError("samples_per_primitive must be at least 2")
        if self.spline_flattening_distance_mm <= 0:
            raise ValueError("spline_flattening_distance_mm must be positive")
        if not self.included_layers:
            raise ValueError("included_layers must not be empty")
        if self.sheet_width_mm <= 0 or self.sheet_height_mm <= 0:
            raise ValueError("sheet dimensions must be positive")
        if self.normalized_longest_side <= 0:
            raise ValueError("normalized_longest_side must be positive")
        if self.sheet_tolerance_mm < 0:
            raise ValueError("sheet_tolerance_mm must be non-negative")
        if self.view_bbox_tolerance_mm < 0:
            raise ValueError("view_bbox_tolerance_mm must be non-negative")
        if not self.include_visible and not self.include_hidden:
            raise ValueError(
                "at least one of visible or hidden entities must be enabled"
            )

    @property
    def sample_feature_dim(self) -> int:
        return len(DXF_SAMPLE_FEATURE_NAMES)

    @property
    def num_primitive_types(self) -> int:
        return len(DXF_PRIMITIVE_TYPES)


@dataclass(frozen=True)
class DXFPrimitiveData:
    """Unpadded primitive tensors for one complete technical drawing sheet."""

    sample_features: torch.FloatTensor  # [N, S, len(DXF_SAMPLE_FEATURE_NAMES)]
    primitive_type_ids: torch.LongTensor  # [N]
    view_direction_ids: torch.LongTensor  # [N]
    entity_handles: tuple[str, ...]
    entity_type_names: tuple[str, ...]
    primitive_group_ids: torch.LongTensor | None = None
    num_skipped_degenerate_entities: int = 0

    @property
    def num_primitives(self) -> int:
        return self.sample_features.shape[0]


def _xy(point) -> tuple[float, float]:
    return float(point[0]), float(point[1])


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _signed_menger_curvature(
    previous: tuple[float, float],
    current: tuple[float, float],
    following: tuple[float, float],
) -> float:
    """Curvature of the circumcircle through three points, positive for a left
    turn in traversal order. Exact for points sampled from a circular arc."""
    a = _distance(previous, current)
    b = _distance(current, following)
    c = _distance(previous, following)
    denominator = a * b * c
    if denominator <= 1e-30:
        return 0.0
    cross = (current[0] - previous[0]) * (following[1] - current[1]) - (
        current[1] - previous[1]
    ) * (following[0] - current[0])
    return 2.0 * cross / denominator


def _sample_geometry(
    points: Sequence[tuple[float, float]],
) -> tuple[list[tuple[float, float]], list[float], list[float]]:
    """Per-sample unit tangent, signed curvature, and centered arc position.

    All values follow the traversal direction of ``points`` and are exactly
    negated by reversing the sequence. Sequences whose first and last point
    coincide are treated as closed and wrap across the seam, so the duplicated
    endpoint carries the same tangent and curvature as the start.
    """
    count = len(points)
    if count < 2:
        raise DXFParseError("geometry features require at least two samples")
    segment_lengths = [
        _distance(points[index], points[index + 1]) for index in range(count - 1)
    ]
    total_length = sum(segment_lengths)
    if total_length <= 1e-12:
        raise DXFParseError("primitive has zero geometric length")
    arc_positions = [-0.5]
    cumulative = 0.0
    for length in segment_lengths:
        cumulative += length
        arc_positions.append(cumulative / total_length - 0.5)

    closed = count > 3 and _distance(points[0], points[-1]) < 1e-9
    cycle = points[:-1] if closed else points

    def neighbors(
        index: int,
    ) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
        if closed:
            wrapped = index % len(cycle)
            return (
                cycle[(wrapped - 1) % len(cycle)],
                cycle[(wrapped + 1) % len(cycle)],
            )
        return (
            points[index - 1] if index > 0 else None,
            points[index + 1] if index < count - 1 else None,
        )

    tangents: list[tuple[float, float]] = []
    curvatures: list[float | None] = []
    for index in range(count):
        current = points[index]
        previous, following = neighbors(index)
        candidate_chords = []
        if previous is not None and following is not None:
            candidate_chords.append(
                (following[0] - previous[0], following[1] - previous[1])
            )
        if following is not None:
            candidate_chords.append(
                (following[0] - current[0], following[1] - current[1])
            )
        if previous is not None:
            candidate_chords.append(
                (current[0] - previous[0], current[1] - previous[1])
            )
        # Hairpins can zero the central chord; fall back to one-sided chords so
        # the tangent stays finite.
        tangent = (0.0, 0.0)
        for chord in candidate_chords:
            norm = math.hypot(chord[0], chord[1])
            if norm > 1e-12:
                tangent = (chord[0] / norm, chord[1] / norm)
                break
        tangents.append(tangent)
        curvatures.append(
            _signed_menger_curvature(previous, current, following)
            if previous is not None and following is not None
            else None
        )
    # Open endpoints copy the nearest interior curvature estimate.
    interior = [value for value in curvatures if value is not None]
    filled: list[float] = []
    for index, value in enumerate(curvatures):
        if value is not None:
            filled.append(value)
        elif not interior:
            filled.append(0.0)
        elif index == 0:
            filled.append(interior[0])
        else:
            filled.append(interior[-1])
    return tangents, filled, arc_positions


def _deduplicate_neighbors(
    points: Iterable[tuple[float, float]],
    tolerance: float = 1e-10,
) -> list[tuple[float, float]]:
    output: list[tuple[float, float]] = []
    for point in points:
        if not output or _distance(output[-1], point) > tolerance:
            output.append(point)
    return output


def _rotate_closed_to_canonical_start(
    points: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Choose a deterministic phase before closed-curve resampling."""
    unique_points = list(points)
    if (
        len(unique_points) > 1
        and _distance(unique_points[0], unique_points[-1]) < 1e-10
    ):
        unique_points.pop()
    if not unique_points:
        return []
    start = min(
        range(len(unique_points)),
        key=lambda index: (unique_points[index][0], unique_points[index][1]),
    )
    return unique_points[start:] + unique_points[:start]


def _resample_polyline(
    points: Sequence[tuple[float, float]],
    count: int,
    *,
    closed: bool,
) -> list[tuple[float, float]]:
    points = _deduplicate_neighbors(points)
    if closed:
        points = _rotate_closed_to_canonical_start(points)
        if points:
            points = [*points, points[0]]
    if len(points) < 2:
        raise DXFParseError("primitive degenerates to fewer than two distinct points")

    segment_lengths = [
        _distance(points[index], points[index + 1]) for index in range(len(points) - 1)
    ]
    total_length = sum(segment_lengths)
    if total_length <= 1e-12:
        raise DXFParseError("primitive has zero geometric length")

    # Closed curves deliberately duplicate the canonical start at both ends.
    # This makes sequence reversal retain the same phase in the model encoder.
    target_distances = [total_length * index / (count - 1) for index in range(count)]
    output: list[tuple[float, float]] = []
    segment_index = 0
    segment_start_distance = 0.0
    for target in target_distances:
        while (
            segment_index < len(segment_lengths) - 1
            and target > segment_start_distance + segment_lengths[segment_index]
        ):
            segment_start_distance += segment_lengths[segment_index]
            segment_index += 1
        length = segment_lengths[segment_index]
        fraction = min(max((target - segment_start_distance) / length, 0.0), 1.0)
        start_point = points[segment_index]
        end_point = points[segment_index + 1]
        output.append(
            (
                start_point[0] + fraction * (end_point[0] - start_point[0]),
                start_point[1] + fraction * (end_point[1] - start_point[1]),
            )
        )
    return output


def _sample_line(entity: DXFEntity, count: int) -> list[tuple[float, float]]:
    start, end = _xy(entity.dxf.start), _xy(entity.dxf.end)
    return [
        (
            start[0] + index / (count - 1) * (end[0] - start[0]),
            start[1] + index / (count - 1) * (end[1] - start[1]),
        )
        for index in range(count)
    ]


def _sample_arc(entity: DXFEntity, count: int) -> list[tuple[float, float]]:
    start = float(entity.dxf.start_angle)
    span = (float(entity.dxf.end_angle) - start) % 360.0
    if span <= 1e-10:
        raise DXFParseError("ARC has a zero angular span")
    angles = [start + span * index / (count - 1) for index in range(count)]
    return [_xy(point) for point in entity.vertices(angles)]


def _sample_circle(entity: DXFEntity, count: int) -> list[tuple[float, float]]:
    angles = [360.0 * index / (count - 1) for index in range(count)]
    return [_xy(point) for point in entity.vertices(angles)]


def _sample_ellipse(entity: DXFEntity, count: int) -> list[tuple[float, float]]:
    start = float(entity.dxf.start_param)
    raw_span = float(entity.dxf.end_param) - start
    span = raw_span % math.tau
    is_closed = bool(getattr(entity, "closed", False)) or math.isclose(
        abs(raw_span), math.tau, rel_tol=0.0, abs_tol=1e-8
    )
    if is_closed:
        start, span = 0.0, math.tau
    elif span <= 1e-12:
        raise DXFParseError("ELLIPSE has a zero parameter span")
    dense_count = max(64, count * 8)
    denominator = dense_count if is_closed else dense_count - 1
    parameters = [start + span * index / denominator for index in range(dense_count)]
    dense = [_xy(point) for point in entity.vertices(parameters)]
    return _resample_polyline(dense, count, closed=is_closed)


def _sample_spline(
    entity: DXFEntity,
    count: int,
    flattening_distance: float,
) -> list[tuple[float, float]]:
    try:
        dense = [
            _xy(point)
            for point in entity.flattening(
                distance=flattening_distance,
                segments=max(4, count),
            )
        ]
    except (ValueError, ZeroDivisionError):
        dense = [_xy(point) for point in entity.fit_points]
        if len(dense) < 2:
            dense = [_xy(point) for point in entity.control_points]
    return _resample_polyline(dense, count, closed=bool(entity.closed))


def _dense_virtual_entity_points(
    entity: DXFEntity,
    count: int,
    flattening_distance: float,
) -> list[tuple[float, float]]:
    entity_type = entity.dxftype()
    if entity_type == "LINE":
        return [_xy(entity.dxf.start), _xy(entity.dxf.end)]
    if entity_type == "ARC":
        start = float(entity.dxf.start_angle)
        span = (float(entity.dxf.end_angle) - start) % 360.0
        dense_count = max(8, math.ceil(count * 8 * span / 360.0))
        angles = [
            start + span * index / (dense_count - 1) for index in range(dense_count)
        ]
        return [_xy(point) for point in entity.vertices(angles)]
    if entity_type == "SPLINE":
        return [
            _xy(point)
            for point in entity.flattening(
                distance=flattening_distance,
                segments=max(4, count),
            )
        ]
    raise DXFParseError(f"unsupported virtual polyline entity {entity_type}")


def _sample_polyline(
    entity: DXFEntity,
    count: int,
    flattening_distance: float,
) -> list[tuple[float, float]]:
    dense: list[tuple[float, float]] = []
    for virtual_entity in entity.virtual_entities():
        segment = _dense_virtual_entity_points(
            virtual_entity,
            count,
            flattening_distance,
        )
        if dense and segment and _distance(dense[-1], segment[0]) < 1e-10:
            segment = segment[1:]
        dense.extend(segment)
    return _resample_polyline(dense, count, closed=bool(entity.closed))


def sample_dxf_entity(
    entity: DXFEntity,
    config: DXFPrimitiveConfig,
) -> list[tuple[float, float]]:
    """Sample a supported entity approximately uniformly by arc length."""
    count = config.samples_per_primitive
    entity_type = entity.dxftype()
    if entity_type == "LINE":
        points = _sample_line(entity, count)
    elif entity_type == "ARC":
        points = _sample_arc(entity, count)
    elif entity_type == "CIRCLE":
        points = _sample_circle(entity, count)
    elif entity_type == "ELLIPSE":
        points = _sample_ellipse(entity, count)
    elif entity_type == "SPLINE":
        points = _sample_spline(
            entity,
            count,
            config.spline_flattening_distance_mm,
        )
    elif entity_type in {"LWPOLYLINE", "POLYLINE"}:
        points = _sample_polyline(
            entity,
            count,
            config.spline_flattening_distance_mm,
        )
    else:
        raise DXFParseError(f"unsupported DXF primitive type {entity_type}")
    if len(_deduplicate_neighbors(points)) < 2:
        raise DXFParseError("primitive has zero geometric length")
    return points


class DXFPrimitiveParser:
    """Parse supported modelspace entities into model-ready sample features."""

    def __init__(self, config: DXFPrimitiveConfig | None = None) -> None:
        self.config = config or DXFPrimitiveConfig()

    def _is_hidden(self, document, entity: DXFEntity) -> bool:
        linetype = str(entity.dxf.get("linetype", "BYLAYER"))
        if linetype.upper() == "BYLAYER":
            try:
                linetype = str(document.layers.get(entity.dxf.layer).dxf.linetype)
            except ezdxf.DXFTableEntryError:
                linetype = "CONTINUOUS"
        return linetype.upper().startswith("HIDDEN")

    def _normalize_xy(self, point: tuple[float, float]) -> tuple[float, float]:
        config = self.config
        scale = config.normalized_longest_side / max(
            config.sheet_width_mm,
            config.sheet_height_mm,
        )
        return (
            (point[0] - config.sheet_width_mm / 2.0) * scale,
            (point[1] - config.sheet_height_mm / 2.0) * scale,
        )

    def _sample_feature_rows(
        self,
        points: Sequence[tuple[float, float]],
        visibility: float,
    ) -> list[tuple[float, ...]]:
        """Assemble the per-sample feature rows named by
        ``DXF_SAMPLE_FEATURE_NAMES`` in one normalized coordinate frame.

        The geometry is derived from the normalized points, so curvature is
        already expressed in normalized units; ``log1p`` compresses its wide
        dynamic range (a 1 mm-radius fillet reaches |curvature| ≈ 165) while
        ``copysign`` keeps it an odd function of traversal direction.
        """
        normalized = [self._normalize_xy(point) for point in points]
        tangents, curvatures, arc_positions = _sample_geometry(normalized)
        return [
            (
                point[0],
                point[1],
                visibility,
                tangent[0],
                tangent[1],
                math.copysign(math.log1p(abs(curvature)), curvature),
                arc_position,
            )
            for point, tangent, curvature, arc_position in zip(
                normalized, tangents, curvatures, arc_positions, strict=True
            )
        ]

    def _is_inside_sheet(
        self,
        points: Sequence[tuple[float, float]],
    ) -> bool:
        config = self.config
        if not config.discard_outside_sheet:
            return True
        tolerance = config.sheet_tolerance_mm
        return all(
            -tolerance <= x <= config.sheet_width_mm + tolerance
            and -tolerance <= y <= config.sheet_height_mm + tolerance
            for x, y in points
        )

    def _assign_view_direction(
        self,
        points: Sequence[tuple[float, float]],
        view_bboxes: Sequence[ViewBBox],
        *,
        sample_id: str,
        entity_handle: str,
        entity_type: str,
    ) -> int:
        representative_x = sum(point[0] for point in points) / len(points)
        representative_y = sum(point[1] for point in points) / len(points)
        matches = tuple(
            bbox
            for bbox in view_bboxes
            if bbox.contains(
                representative_x,
                representative_y,
                tolerance_mm=self.config.view_bbox_tolerance_mm,
            )
        )
        if len(matches) != 1:
            candidates = {
                bbox.direction: tuple(round(value, 6) for value in bbox.xyxy_mm)
                for bbox in view_bboxes
            }
            raise DXFParseError(
                "primitive view assignment requires exactly one bbox match: "
                f"sample_id={sample_id!r}, entity_handle={entity_handle!r}, "
                f"entity_type={entity_type!r}, "
                f"representative_point=({representative_x:.6f}, "
                f"{representative_y:.6f}), "
                f"matched_directions={[bbox.direction for bbox in matches]}, "
                f"candidate_bboxes={candidates}"
            )
        return matches[0].direction_id

    def parse(
        self,
        path: str | Path,
        *,
        view_bboxes: Sequence[ViewBBox],
        sample_id: str | None = None,
    ) -> DXFPrimitiveData:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"DXF file does not exist: {path}")
        try:
            document = ezdxf.readfile(path)
        except (OSError, ezdxf.DXFError) as error:
            raise DXFParseError(f"failed to read DXF {path}: {error}") from error

        view_bboxes = tuple(view_bboxes)
        directions = tuple(bbox.direction for bbox in view_bboxes)
        if len(view_bboxes) != len(VIEW_DIRECTIONS) or set(directions) != set(
            VIEW_DIRECTIONS
        ):
            raise DXFParseError(
                "DXF parsing requires exactly one bbox for each view direction "
                f"{VIEW_DIRECTIONS}, got {directions}"
            )
        resolved_sample_id = sample_id or path.stem

        features: list[list[tuple[float, ...]]] = []
        type_ids: list[int] = []
        view_direction_ids: list[int] = []
        handles: list[str] = []
        type_names: list[str] = []
        num_skipped_degenerate_entities = 0
        included_layers = set(self.config.included_layers)

        for entity in document.modelspace():
            if entity.dxf.layer not in included_layers:
                continue
            entity_type = entity.dxftype()
            if entity_type not in DXF_PRIMITIVE_TYPE_TO_ID:
                if self.config.strict_entity_types:
                    raise DXFParseError(
                        f"unsupported entity {entity_type} on included layer "
                        f"{entity.dxf.layer!r} in {path}"
                    )
                continue

            hidden = self._is_hidden(document, entity)
            if hidden and not self.config.include_hidden:
                continue
            if not hidden and not self.config.include_visible:
                continue
            try:
                points = sample_dxf_entity(entity, self.config)
            except DXFParseError as error:
                if self.config.strict_entity_types:
                    raise
                if "zero" in str(error) or "fewer than two distinct" in str(error):
                    num_skipped_degenerate_entities += 1
                continue
            if not self._is_inside_sheet(points):
                continue
            entity_handle = str(entity.dxf.get("handle", ""))
            view_direction_id = self._assign_view_direction(
                points,
                view_bboxes,
                sample_id=resolved_sample_id,
                entity_handle=entity_handle,
                entity_type=entity_type,
            )
            visibility = -1.0 if hidden else 1.0
            features.append(self._sample_feature_rows(points, visibility))
            type_ids.append(DXF_PRIMITIVE_TYPE_TO_ID[entity_type])
            view_direction_ids.append(view_direction_id)
            handles.append(entity_handle)
            type_names.append(entity_type)

        if not features:
            raise DXFParseError(f"DXF contains no usable primitives: {path}")
        sample_features = torch.tensor(features, dtype=torch.float32)
        primitive_type_ids = torch.tensor(type_ids, dtype=torch.long)
        direction_ids = torch.tensor(view_direction_ids, dtype=torch.long)
        if not torch.isfinite(sample_features).all():
            raise DXFParseError(f"DXF produced non-finite sample features: {path}")
        return DXFPrimitiveData(
            sample_features=sample_features,
            primitive_type_ids=primitive_type_ids,
            view_direction_ids=direction_ids,
            entity_handles=tuple(handles),
            entity_type_names=tuple(type_names),
            num_skipped_degenerate_entities=num_skipped_degenerate_entities,
        )


__all__ = [
    "DXFParseError",
    "DXFPrimitiveConfig",
    "DXFPrimitiveData",
    "DXFPrimitiveParser",
    "DXF_ORIENTED_SAMPLE_FEATURE_INDICES",
    "DXF_PRIMITIVE_TYPES",
    "DXF_PRIMITIVE_TYPE_TO_ID",
    "DXF_SAMPLE_FEATURE_NAMES",
    "sample_dxf_entity",
]
