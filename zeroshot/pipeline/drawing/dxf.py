"""Read a DXF page into the drawing contract, and write the contract back out.

Reading interprets nothing; the two directions are inverses.
"""

import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ezdxf
from ezdxf.document import Drawing as DxfDocument
from ezdxf.layouts import Modelspace

from zeroshot.pipeline.messages.contracts.drawings import (
    ClaimSource,
    DrawingEvidence,
    DrawingSheet,
    DrawingSource,
    DrawnEntity,
    EdgeStyle,
    View,
    edge_style_for_linetype,
)
from zeroshot.pipeline.messages.contracts.parameters import Parameter

# A sheet coordinate this far off the drawing plane, or an extrusion this far
# from +Z, means the entity's numbers are not the page's own.
_PLANAR_TOLERANCE = 1e-6

type _Reading = tuple[DrawnEntity, list[tuple[str, list[float]]]]


@dataclass(frozen=True)
class DrawingReading:
    """A drawing as read, beside what the reading could not carry."""

    drawing: DrawingSource
    skipped: Mapping[str, int]

    @property
    def sheet(self) -> DrawingSheet:
        return self.drawing.sheets[0]


# --- Reading ---


def _point(place: Any) -> list[float]:
    return [place[0], place[1]]


def _flatten(places: Iterable[Any]) -> list[float]:
    return [value for place in places for value in (place[0], place[1])]


def _on_circle(center: Any, radius: float, degrees: float) -> list[float]:
    turn = math.radians(degrees)
    return [center[0] + radius * math.cos(turn), center[1] + radius * math.sin(turn)]


def _on_ellipse(
    center: Any, major: Any, minor_radius: float, parameter: float
) -> list[float]:
    """Where an ellipse's own parameter lands on the page."""
    semi_major = math.hypot(major[0], major[1])
    across = (-major[1] / semi_major, major[0] / semi_major)
    return [
        center[0]
        + major[0] * math.cos(parameter)
        + across[0] * minor_radius * math.sin(parameter),
        center[1]
        + major[1] * math.cos(parameter)
        + across[1] * minor_radius * math.sin(parameter),
    ]


def _read_line(entity: Any) -> _Reading:
    return DrawnEntity.LINE, [
        ("start", _point(entity.dxf.start)),
        ("end", _point(entity.dxf.end)),
    ]


def _read_circle(entity: Any) -> _Reading:
    return DrawnEntity.CIRCLE, [
        ("center", _point(entity.dxf.center)),
        ("radius", [entity.dxf.radius]),
    ]


def _read_arc(entity: Any) -> _Reading:
    center, radius = entity.dxf.center, entity.dxf.radius
    return DrawnEntity.ARC, [
        ("center", _point(center)),
        ("radius", [radius]),
        ("start", _on_circle(center, radius, entity.dxf.start_angle)),
        ("end", _on_circle(center, radius, entity.dxf.end_angle)),
    ]


def _read_ellipse(entity: Any) -> _Reading:
    center, major = entity.dxf.center, entity.dxf.major_axis
    minor_radius = math.hypot(major[0], major[1]) * entity.dxf.ratio
    return DrawnEntity.ELLIPSE, [
        ("center", _point(center)),
        ("major_axis", _point(major)),
        ("minor_radius", [minor_radius]),
        ("start", _on_ellipse(center, major, minor_radius, entity.dxf.start_param)),
        ("end", _on_ellipse(center, major, minor_radius, entity.dxf.end_param)),
    ]


def _read_spline(entity: Any) -> _Reading:
    return DrawnEntity.SPLINE, [
        ("control_points", _flatten(entity.control_points)),
        ("degree", [entity.dxf.degree]),
        ("knots", list(entity.knots)),
    ]


def _read_polyline(entity: Any) -> _Reading:
    return DrawnEntity.POLYLINE, [("vertices", _flatten(entity.get_points()))]


# What this module can transcribe. A DXF type absent from here is reported as
# skipped rather than approximated by something the contract does hold.
_READERS: Mapping[str, Callable[[Any], _Reading]] = {
    "LINE": _read_line,
    "CIRCLE": _read_circle,
    "ARC": _read_arc,
    "ELLIPSE": _read_ellipse,
    "SPLINE": _read_spline,
    "LWPOLYLINE": _read_polyline,
}


def _resolved_linetype(document: DxfDocument, entity: Any) -> str:
    """The entity's own linetype, or the one its layer lends it."""
    linetype = str(entity.dxf.get("linetype", "BYLAYER"))
    if linetype.upper() not in {"BYLAYER", "BYBLOCK"}:
        return linetype
    try:
        return str(document.layers.get(entity.dxf.layer).dxf.linetype)
    except ezdxf.DXFTableEntryError:
        return "CONTINUOUS"


def _depths(entity: Any) -> list[float]:
    """Every z the entity states, whichever attributes it happens to carry."""
    places = [
        entity.dxf.get(name)
        for name in ("start", "end", "center")
        if entity.dxf.hasattr(name)
    ]
    if entity.dxftype() == "SPLINE":
        places.extend(entity.control_points)
    if entity.dxf.hasattr("elevation"):
        return [entity.dxf.elevation, *(place[2] for place in places if len(place) > 2)]
    return [place[2] for place in places if len(place) > 2]


def _off_the_page(entity: Any) -> bool:
    """Whether the entity's numbers mean something other than page x, y."""
    if entity.dxf.hasattr("extrusion"):
        x, y, z = entity.dxf.extrusion
        if abs(x) > _PLANAR_TOLERANCE or abs(y) > _PLANAR_TOLERANCE or z < 0:
            return True
    return any(abs(depth) > _PLANAR_TOLERANCE for depth in _depths(entity))


def _skip_label(entity: Any) -> str:
    kind = entity.dxftype()
    return f"{kind} {entity.dxf.name}" if kind == "INSERT" else kind


def _entry(document: DxfDocument, entity: Any) -> DrawingEvidence:
    kind, parameters = _READERS[entity.dxftype()](entity)
    return DrawingEvidence(
        name=f"ev_{entity.dxf.handle.lower()}",
        entity=kind,
        edge_style=edge_style_for_linetype(_resolved_linetype(document, entity)),
        source=ClaimSource.GIVEN,
        parameters=[Parameter(name=name, values=values) for name, values in parameters],
    )


def read_drawing(
    path: Path | str, name: str = "sheet_page", label: str | None = "drawing"
) -> DrawingReading:
    """The page a DXF draws, transcribed entity for entity and left unsplit."""
    document = ezdxf.readfile(str(path))
    entries: list[DrawingEvidence] = []
    skipped: Counter[str] = Counter()
    for entity in document.modelspace():
        if entity.dxftype() not in _READERS:
            skipped[_skip_label(entity)] += 1
        elif _off_the_page(entity):
            skipped[f"{entity.dxftype()} off the drawing plane"] += 1
        else:
            entries.append(_entry(document, entity))

    sheet = DrawingSheet(
        name=name,
        role=View.UNKNOWN,
        label=label,
        derived_from=None,
        file=str(path),
        origin=None,
        evidence=entries,
        dimensions=[],
    )
    return DrawingReading(DrawingSource(sheets=[sheet]), dict(skipped))


# --- Writing ---


_EXPORT_LINETYPES: Mapping[EdgeStyle, str] = {
    EdgeStyle.VISIBLE: "CONTINUOUS",
    EdgeStyle.HIDDEN: "HIDDEN",
    EdgeStyle.CENTERLINE: "CENTER",
    EdgeStyle.PHANTOM: "PHANTOM",
    EdgeStyle.OTHER: "CONTINUOUS",
}

# AutoCAD ships HIDDEN and ezdxf's standard setup does not, so it is defined
# here rather than looked up. Dash lengths are in drawing units.
_HIDDEN_PATTERN = [0.6, 0.5, -0.1]


def _angle_at(center: Sequence[float], place: Sequence[float]) -> float:
    return math.degrees(math.atan2(place[1] - center[1], place[0] - center[0]))


def _parameter_at(
    center: Sequence[float],
    major: Sequence[float],
    minor_radius: float,
    place: Sequence[float],
) -> float:
    """The ellipse parameter whose point is `place`, the inverse of `_on_ellipse`."""
    semi_major = math.hypot(major[0], major[1])
    x, y = place[0] - center[0], place[1] - center[1]
    along = (x * major[0] + y * major[1]) / semi_major
    across = (y * major[0] - x * major[1]) / semi_major
    return math.atan2(across / minor_radius, along / semi_major) % math.tau


def _write_line(space: Modelspace, values: Mapping[str, list[float]], attribs) -> None:
    space.add_line(values["start"], values["end"], dxfattribs=attribs)


def _write_circle(
    space: Modelspace, values: Mapping[str, list[float]], attribs
) -> None:
    space.add_circle(values["center"], values["radius"][0], dxfattribs=attribs)


def _write_arc(space: Modelspace, values: Mapping[str, list[float]], attribs) -> None:
    center = values["center"]
    start = _angle_at(center, values["start"])
    end = _angle_at(center, values["end"])
    space.add_arc(
        center,
        values["radius"][0],
        start,
        end if values["start"] != values["end"] else start + 360.0,
        dxfattribs=attribs,
    )


def _write_ellipse(
    space: Modelspace, values: Mapping[str, list[float]], attribs
) -> None:
    center, major = values["center"], values["major_axis"]
    minor_radius = values["minor_radius"][0]
    whole = values["start"] == values["end"]
    space.add_ellipse(
        center,
        major_axis=major,
        ratio=minor_radius / math.hypot(major[0], major[1]),
        start_param=(
            0.0
            if whole
            else _parameter_at(center, major, minor_radius, values["start"])
        ),
        end_param=(
            math.tau
            if whole
            else _parameter_at(center, major, minor_radius, values["end"])
        ),
        dxfattribs=attribs,
    )


def _write_spline(
    space: Modelspace, values: Mapping[str, list[float]], attribs
) -> None:
    flat = values["control_points"]
    spline = space.add_spline(degree=int(values["degree"][0]), dxfattribs=attribs)
    spline.control_points = list(zip(flat[0::2], flat[1::2]))
    spline.knots = values["knots"]


def _write_polyline(
    space: Modelspace, values: Mapping[str, list[float]], attribs
) -> None:
    flat = values["vertices"]
    space.add_lwpolyline(list(zip(flat[0::2], flat[1::2])), dxfattribs=attribs)


_WRITERS: Mapping[
    DrawnEntity, Callable[[Modelspace, Mapping[str, list[float]], dict[str, Any]], None]
] = {
    DrawnEntity.LINE: _write_line,
    DrawnEntity.CIRCLE: _write_circle,
    DrawnEntity.ARC: _write_arc,
    DrawnEntity.ELLIPSE: _write_ellipse,
    DrawnEntity.SPLINE: _write_spline,
    DrawnEntity.POLYLINE: _write_polyline,
}


def _new_document() -> DxfDocument:
    document = ezdxf.new("R2010", setup=True)
    if "HIDDEN" not in document.linetypes:
        document.linetypes.add(
            "HIDDEN", pattern=_HIDDEN_PATTERN, description="Hidden __ __ __"
        )
    return document


def _draw(space: Modelspace, sheet: DrawingSheet, layer: str) -> None:
    for entry in sheet.evidence:
        values = {
            parameter.name.value: parameter.values for parameter in entry.parameters
        }
        _WRITERS[entry.entity](
            space,
            values,
            {"layer": layer, "linetype": _EXPORT_LINETYPES[entry.edge_style]},
        )


def export_sheet(sheet: DrawingSheet, path: Path | str) -> Path:
    """Write one sheet's linework as a DXF, so ezdxf can be asked about it."""
    document = _new_document()
    document.layers.add(sheet.name)
    _draw(document.modelspace(), sheet, sheet.name)
    document.saveas(str(path))
    return Path(path)


def export_drawing(drawing: DrawingSource, path: Path | str) -> Path:
    """Write a whole drawing as one DXF, each sheet on a layer of its own."""
    document = _new_document()
    space = document.modelspace()
    for sheet in drawing.sheets:
        if not sheet.evidence:
            continue
        document.layers.add(sheet.name)
        _draw(space, sheet, sheet.name)
    document.saveas(str(path))
    return Path(path)
