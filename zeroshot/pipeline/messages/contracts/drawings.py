"""Drawing observations, their contracts, and supporting conversions."""

import math
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path, PurePath
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from zeroshot.pipeline.messages.contracts.parameters import (
    Parameter,
    describe_parameters,
    require_name,
    require_parameters,
    require_unique,
)

# --- Vocabulary and projection frames ---


class View(StrEnum):
    # orthographic views
    FRONT = "front"
    BACK = "back"
    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"
    # other views
    SECTION = "section"
    DETAIL = "detail"
    ISOMETRIC = "isometric"
    PERSPECTIVE = "perspective"
    # A whole page carrying every view, before anything separates them.
    FULL_PAGE = "full_page"
    UNKNOWN = "unknown"


# Sheet-right, sheet-up, and toward the viewer, in model axes.  Model XY is
# the horizontal plane and +Z is vertical.  The signs make every tuple a
# right-handed screen frame: right = up x out.
VIEW_FRAME: Mapping[View, tuple[str, str, str]] = {
    View.FRONT: ("+x", "+z", "-y"),
    View.BACK: ("-x", "+z", "+y"),
    View.TOP: ("+x", "+y", "+z"),
    View.BOTTOM: ("+x", "-y", "-z"),
    View.RIGHT: ("+y", "+z", "+x"),
    View.LEFT: ("-y", "+z", "-x"),
}
ORTHOGRAPHIC_VIEWS = tuple(VIEW_FRAME)

DRAWING_SUFFIXES = frozenset({".dxf", ".png", ".jpg", ".jpeg"})


class DrawnEntity(StrEnum):
    LINE = "line"
    ARC = "arc"
    CIRCLE = "circle"
    ELLIPSE = "ellipse"
    SPLINE = "spline"
    POLYLINE = "polyline"


class EdgeStyle(StrEnum):
    VISIBLE = "visible"
    HIDDEN = "hidden"
    CENTERLINE = "centerline"
    PHANTOM = "phantom"
    OTHER = "other"


class DimensionKind(StrEnum):
    LINEAR = "linear"
    DIAMETER = "diameter"
    RADIUS = "radius"
    ANGULAR = "angular"


# --- Printed dimensions and primitive evidence ---


# A printed figure is what turns a pixel measurement into millimetres.
# One figure may govern several entities (e.g., 4X R10) so each is
# written once here and named by every entity it measures.
class Dimension(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        description="Stable dim_ name in lower_snake_case, unique across the drawing.",
    )
    kind: DimensionKind = Field(..., description="The kind of printed dimension.")
    text: str = Field(
        ...,
        description=(
            "The callout exactly as printed, symbols and all: 4X \u230012 THRU, "
            "M12x1.75-6H, R10. Split a callout that states two measurements, "
            "such as \u230012 THRU 15 DEEP, into one figure each."
        ),
    )
    nominal: float = Field(
        ...,
        description="Degrees for angle, millimetres for length.",
    )
    quantity: int = Field(
        ...,
        description="Feature count: 4 for '4X 12 THRU', or 1 if no count is printed.",
    )
    note: str | None = Field(
        ...,
        description=(
            "A remark printed beside the callout in words rather than symbols, "
            "such as AFTER PLATING or SEE DETAIL B; null if absent."
        ),
    )

    @model_validator(mode="after")
    def validate_dimension(self) -> Self:
        require_name(self.name, "dim_")
        if self.quantity < 1:
            raise ValueError("dimension quantity must be at least 1")
        return self


_DRAWN_PARAMETERS = {
    DrawnEntity.LINE: ("start", "end"),
    DrawnEntity.ARC: ("center", "radius", "start", "end"),
    DrawnEntity.CIRCLE: ("center", "radius"),
    DrawnEntity.ELLIPSE: ("center", "major_axis", "minor_radius", "start", "end"),
    DrawnEntity.SPLINE: ("control_points", "degree", "knots"),
    DrawnEntity.POLYLINE: ("vertices",),
}


class DrawingEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        description=(
            "Stable ev_ name in lower_snake_case, unique across the drawing. "
            "Keep the name when revising the same entity; use a new name for "
            "a new entity. Cite parameters as ev_front_edge.start."
        ),
    )
    entity: DrawnEntity = Field(
        ..., description="The 2D entity as drawn, not its 3D shape."
    )
    edge_style: EdgeStyle = Field(
        ..., description="Visible, hidden, centerline, phantom, or other linework."
    )
    parameters: list[Parameter] = Field(
        ...,
        description=(
            "Supply exactly the parameters listed below for your entity, and "
            "no others. Points are in this sheet's own coordinates in "
            "millimetres: u rightward and v upward from its bottom-left "
            "corner. For a raster, that origin is the lower-left corner of "
            "its bottom-left pixel, not the pixel centre. Never report pixels.\n"
            f"{describe_parameters(_DRAWN_PARAMETERS)}\n"
            "ARC and ELLIPSE sweep counterclockwise from start to end, both "
            "points lying on the curve; equal endpoints mean a full turn. An "
            "ELLIPSE gives major_axis as the vector from center to the "
            "major-axis endpoint, whose length is the semi-major radius."
        ),
    )
    source: list[str] = Field(
        ...,
        description=(
            "dim_ names of the printed figures this reading rests on, from the "
            "same sheet: the diameter beside a circle, the length between two "
            "edges. Empty when nothing is printed and the numbers were "
            "measured or taken from a vector definition."
        ),
    )

    @model_validator(mode="after")
    def validate_entity(self) -> Self:
        require_name(self.name, "ev_")
        require_unique(self.source, f"{self.name} sources")
        values = require_parameters(
            f"{self.name} ({self.entity.value})",
            _DRAWN_PARAMETERS[self.entity],
            self.parameters,
        )
        if self.entity is DrawnEntity.ARC:
            radius = values["radius"][0]
            _require_endpoints_on_curve(self.entity, values, (radius, 0.0), radius)
        elif self.entity is DrawnEntity.ELLIPSE:
            major = values["major_axis"]
            _require_endpoints_on_curve(
                self.entity, values, (major[0], major[1]), values["minor_radius"][0]
            )
        return self


def _require_endpoints_on_curve(
    entity: DrawnEntity,
    values: Mapping[str, list[float]],
    major_axis: tuple[float, float],
    minor_radius: float,
    tolerance: float = 0.05,
) -> None:
    """Hold start and end to the curve they bound, an arc being the circular case."""
    center = values["center"]
    semi_major = math.hypot(*major_axis)
    if minor_radius <= 0 or semi_major <= 0:
        raise ValueError(f"{entity.value} needs positive radii")
    for name in ("start", "end"):
        x, y = values[name][0] - center[0], values[name][1] - center[1]
        along = (x * major_axis[0] + y * major_axis[1]) / semi_major
        across = (y * major_axis[0] - x * major_axis[1]) / semi_major
        reach = math.hypot(along / semi_major, across / minor_radius)
        if abs(reach - 1) > tolerance:
            raise ValueError(
                f"{name} is off the {entity.value} (relative radius {reach:.2f}); "
                "give the endpoint in sheet coordinates"
            )


class CropOf(BaseModel):
    """Where a sheet sits on the one it was taken from."""

    model_config = ConfigDict(extra="forbid")

    sheet: str = Field(..., description="The parent sheet_ name.")
    box: list[float] = Field(
        ...,
        description=(
            "u0, v0, u1, v1: the region this sheet covers, in the parent's own "
            "coordinates, u rightward and v upward from its bottom-left corner. "
            "For a raster parent, the origin is the lower-left corner of its "
            "bottom-left pixel, not the pixel centre."
        ),
    )

    @model_validator(mode="after")
    def validate_crop(self) -> Self:
        require_name(self.sheet, "sheet_")
        if len(self.box) != 4:
            raise ValueError(f"{self.sheet} crop box takes u0, v0, u1, v1")
        u0, v0, u1, v1 = self.box
        if u1 <= u0 or v1 <= v0:
            raise ValueError(f"{self.sheet} crop box must have u0 < u1 and v0 < v1")
        return self


class DrawingSheet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        description=(
            "Stable sheet_ name in lower_snake_case, unique across the drawing. "
            "Keep it for the same sheet or view; use a new name for a new one."
        ),
    )
    role: View = Field(..., description="Projection type; unknown if not established.")
    label: str | None = Field(
        ...,
        description="Printed caption such as SECTION A-A or DETAIL B; null if absent.",
    )
    crop_of: CropOf | None = Field(
        ...,
        description=(
            "Where this sheet was taken from, when it was taken from another; "
            "null for a page the run was handed."
        ),
    )
    scale: float = Field(
        ...,
        description=(
            "Millimetres per unit on this sheet: the factor you converted by, "
            "or 1.0 if you measured millimetres already. "
            "Coordinates are reported in millimetres either way."
        ),
    )
    file: str = Field(
        ...,
        description=(
            "Path to this sheet's own drawing. A view you cut out of another "
            "sheet must be saved to a file of its own and named here."
        ),
    )
    evidence: list[DrawingEvidence] = Field(
        ...,
        description=(
            "Entities read from this sheet, in this sheet's own coordinates. "
            "Empty before reading. Entries are independent of 3D features and "
            "may support several features."
        ),
    )
    dimensions: list[Dimension] = Field(
        ..., description="Printed figures on this sheet; empty if none."
    )

    @field_validator("file", mode="before")
    @classmethod
    def accept_path(cls, value: object) -> object:
        return str(value) if isinstance(value, PurePath) else value

    @model_validator(mode="after")
    def validate_sheet(self) -> Self:
        require_name(self.name, "sheet_")
        if self.scale <= 0:
            raise ValueError(f"{self.name} scale must be positive")
        if Path(self.file).suffix.lower() not in DRAWING_SUFFIXES:
            raise ValueError(f"unsupported drawing file: {self.file}")
        if self.crop_of is not None and self.crop_of.sheet == self.name:
            raise ValueError(f"{self.name} cannot be cut from itself")
        printed = [figure.name for figure in self.dimensions]
        require_unique([*(entry.name for entry in self.evidence), *printed], self.name)
        for entry in self.evidence:
            missing = sorted(set(entry.source) - set(printed))
            if missing:
                raise ValueError(
                    f"{entry.name} cites figures absent from {self.name}: {missing}"
                )
        return self

    @property
    def frame(self) -> tuple[str, str, str] | None:
        return VIEW_FRAME.get(self.role)


class DrawingSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sheets: list[DrawingSheet] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_drawing(self) -> Self:
        if not self.sheets:
            raise ValueError("a drawing source needs at least one sheet")
        require_unique(
            (sheet.role.value for sheet in self.sheets if sheet.role in VIEW_FRAME),
            "orthographic views",
        )
        require_unique(
            [sheet.name for sheet in self.sheets]
            + [entry.name for entry in self.evidence()]
            + [figure.name for figure in self.dimensions()],
            "drawing",
        )
        by_name = {sheet.name: sheet for sheet in self.sheets}
        for sheet in self.sheets:
            if (
                sheet.crop_of is not None
                and sheet.crop_of.sheet in by_name
                and sheet.file == by_name[sheet.crop_of.sheet].file
            ):
                raise ValueError(
                    f"{sheet.name}: a cropped view must use its own file, not "
                    f"{sheet.crop_of.sheet}'s file"
                )
            seen = {sheet.name}
            cut = sheet.crop_of
            while cut is not None:
                if cut.sheet not in by_name:
                    raise ValueError(
                        f"{sheet.name} has a missing ancestor: {cut.sheet}"
                    )
                if cut.sheet in seen:
                    raise ValueError(
                        f"{sheet.name} has an ancestry cycle through {cut.sheet}"
                    )
                seen.add(cut.sheet)
                cut = by_name[cut.sheet].crop_of
        return self

    def orthographic(self) -> list[DrawingSheet]:
        return [sheet for sheet in self.sheets if sheet.role in VIEW_FRAME]

    def evidence(self) -> list[DrawingEvidence]:
        return [entry for sheet in self.sheets for entry in sheet.evidence]

    def dimensions(self) -> list[Dimension]:
        return [figure for sheet in self.sheets for figure in sheet.dimensions]

    def cited_names(self) -> set[str]:
        """Every name a later stage may cite, entries and printed figures alike."""
        return {entry.name for entry in self.evidence()} | {
            figure.name for figure in self.dimensions()
        }

    def paths(self) -> list[Path]:
        """Every file this drawing is made of."""
        return [Path(sheet.file) for sheet in self.sheets]

    def frame_sentence(self) -> str:
        """Describe the known orthographic frames, or all six before view assignment."""
        wanted = {sheet.role for sheet in self.orthographic()} or set(
            ORTHOGRAPHIC_VIEWS
        )
        views = "; ".join(
            f"{view.value.capitalize()} is right={VIEW_FRAME[view][0]}, "
            f"up={VIEW_FRAME[view][1]}, toward viewer={VIEW_FRAME[view][2]}"
            for view in ORTHOGRAPHIC_VIEWS
            if view in wanted
        )
        return (
            "Model XY is the horizontal plane and +z is up. Every sheet keeps "
            f"its own UV coordinates, with u=right and v=up: {views}"
        )


# --- DXF line-type conversion ---


_LINETYPE_MEANINGS = {
    "CONTINUOUS": EdgeStyle.VISIBLE,
    "DASHED": EdgeStyle.HIDDEN,
    "DOT": EdgeStyle.HIDDEN,
    "HIDDEN": EdgeStyle.HIDDEN,
    "CENTER": EdgeStyle.CENTERLINE,
    "DASHDOT": EdgeStyle.CENTERLINE,
    "PHANTOM": EdgeStyle.PHANTOM,
    "DIVIDE": EdgeStyle.PHANTOM,
}


def edge_style_for_linetype(linetype: str) -> EdgeStyle:
    """Classify a resolved DXF linetype; resolve BYLAYER/BYBLOCK before calling."""
    base = linetype.strip().upper().removesuffix("X2").removesuffix("2")
    return _LINETYPE_MEANINGS.get(base, EdgeStyle.OTHER)


# --- Input names and files ---


def unread_sheet(name: str, role: View, file: str | PurePath) -> DrawingSheet:
    """A sheet the run holds and nothing has read: a page in, or a render out."""
    return DrawingSheet(
        name=name,
        role=role,
        label=None,
        crop_of=None,
        scale=1.0,
        file=str(file),
        evidence=[],
        dimensions=[],
    )
