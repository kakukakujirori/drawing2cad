import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


def require_name(name: str, prefix: str) -> None:
    if re.fullmatch(rf"{prefix}[a-z0-9_]+", name):
        return
    stray = dict.fromkeys(re.findall(r"[^a-z0-9_]", name.removeprefix(prefix)))
    raise ValueError(
        f"{name!r} is not a usable {prefix.removesuffix('_')} name. "
        f"Begin with {prefix} and carry on in lower_snake_case."
        + (f" Remove {', '.join(map(repr, stray))}." if stray else "")
    )


def require_unique(names: Iterable[str], subject: str) -> None:
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate names in {subject}: {', '.join(duplicates)}")


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


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


VIEW_FRAME: Mapping[View, tuple[str, str, str]] = {
    View.FRONT: ("+x", "+z", "-y"),
    View.BACK: ("-x", "+z", "+y"),
    View.TOP: ("+x", "+y", "+z"),
    View.BOTTOM: ("+x", "-y", "-z"),
    View.RIGHT: ("+y", "+z", "+x"),
    View.LEFT: ("-y", "+z", "-x"),
}
ORTHOGRAPHIC_VIEWS = tuple(VIEW_FRAME)

# A pictorial is offered for context: it fixes no axes, so nothing lifts a coordinate from one.
PICTORIAL_VIEWS = frozenset({View.PERSPECTIVE, View.ISOMETRIC})

# What a seeded artifact carries where the model has yet to decide.
# Validation refuses a submission that still holds one, so it cannot survive a round.
UNDECIDED = "???"


class Region(Contract):
    model_config = ConfigDict(frozen=True)

    view: str = Field(
        ...,
        description="DrawingView.name (view_...) identifying the file and scale used by these coordinates.",
    )
    box_px: tuple[int, int, int, int] | None = Field(
        default=None,
        description="x0, y0, x1, y1 pixel boundaries in the referenced DrawingView.file: origin at that file's top left, x right, y down. Required for raster images, forbidden for native DXF. The displayed image or FULL_PAGE role does not change this reference.",
    )
    box_uv: tuple[float, float, float, float] | None = Field(
        default=None,
        description="u0, v0, u1, v1 in mm, u right and v up, from the lower left of the referenced DrawingView.file (geometry bbox for DXF), using that sheet's scale. Supply normalized UV for DXF; leave null for raster calibration from box_px.",
    )

    @model_validator(mode="after")
    def require_ordered_bounds(self) -> Self:
        if self.box_px is None and self.box_uv is None:
            raise ValueError("at least one of box_px or box_uv is required")
        for name, box in (("box_px", self.box_px), ("box_uv", self.box_uv)):
            if box is not None:
                x0, y0, x1, y1 = box
                if x0 >= x1 or y0 >= y1:
                    raise ValueError(f"{name} must satisfy x0 < x1 and y0 < y1")
                if x0 < 0 or y0 < 0:
                    raise ValueError(f"{name} must use the referenced file's origin")
        require_name(self.view, "view_")
        return self

    def matches_bounds(self, other: Self, *, tol: float = 1e-7) -> bool:
        """Check if this region references the same view and covers the same bounds."""
        if self.view != other.view:
            return False
        if other.box_px is not None:
            return self.box_px == other.box_px
        if other.box_uv is not None:
            return self.box_uv is not None and all(
                math.isclose(a, b, rel_tol=0, abs_tol=tol)
                for a, b in zip(self.box_uv, other.box_uv)
            )
        return False


class Dimension(Contract):
    name: str = Field(
        ...,
        description="Stable dim_ name in lower_snake_case, unique across the drawing.",
    )
    kind: Literal["linear", "diameter", "radius", "angular"] = Field(
        ..., description="The kind of printed dimension."
    )
    text: str = Field(
        ...,
        description=(
            "The callout exactly as printed, symbols and all: 4X \u230012 THRU, "
            "M12x1.75-6H, R10. Split a callout that states two measurements, "
            "such as \u230012 THRU 15 DEEP, into one figure each."
        ),
    )
    nominal_value: float | None = Field(
        ...,
        ge=0,
        description="Printed nominal value: degrees for angle, millimetres for length; null if unreadable.",
    )
    measured_length: float | None = Field(
        default=None,
        gt=0,
        description="Independently measured length in the containing DrawingView.file: pixels at that file's resolution for raster images, native drawing units for DXF. Match nominal_value's extent (diameter with diameter, radius with radius). Null for angles or unmeasured lengths; never infer it from nominal_value or calibration scale.",
    )
    region: Region = Field(
        ...,
        description="Region containing the printed callout and its indicated target; may refer to an original input view even when the measurement is on another view's file.",
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
        if self.kind == "angular" and self.measured_length is not None:
            raise ValueError("angular dimensions cannot have measured_length")
        return self


class DrawingView(Contract):
    name: str = Field(..., description="Stable view_ name, kept across revisions.")
    role: View = Field(
        ...,
        description="Keep each registered input's role. FULL_PAGE means an unsplit page; other roles identify projections such as front, top or right, or pictorial context.",
    )
    file: str = Field(
        ...,
        min_length=1,
        description="Path to this view's image or DXF in the workspace. Keep registered input files; save a separate crop only when identifying a new view within another file.",
    )
    region: Region = Field(
        ...,
        description="Where this view is located in the referenced DrawingView. Preserve each registered input's self-reference and full-file bounds; a new crop refers to its region in the parent view.",
    )
    dimensions: list[Dimension] = Field(
        ..., description="Printed figures on this sheet; empty if none."
    )
    image_size: tuple[int, int] | None = Field(
        default=None,
        description="Width and height in pixels of this DrawingView.file. Submit null; read from that file at validation. Null for native DXF.",
    )
    scale: float | None = Field(
        default=None,
        gt=0,
        description="Millimetres per pixel for this view. Submit null; filled only when independent dimension measurements reach consensus. Null for native DXF or unavailable calibration.",
    )

    @model_validator(mode="after")
    def require_stable_name(self) -> Self:
        require_name(self.name, "view_")
        if self.image_size is not None and min(self.image_size) <= 0:
            raise ValueError("image_size must contain positive width and height")
        return self


class SemanticFeature(Contract):
    name: str = Field(..., description="Stable sem_ name, kept across revisions.")
    description: str = Field(
        ...,
        min_length=1,
        description=(
            "Finished shape and material/void meaning. Define parameter anchors, "
            "directions and termination; do not repeat numbers or specify CAD operations."
        ),
    )
    parameters: dict[str, float | list[float | None] | None] = Field(
        ...,
        description=(
            "Named sizes, model xyz positions and necessary directions, stored once. "
            "Lengths/positions in mm, angles in degrees, directions unit vectors. "
            "Null means unknown, never zero; omit irrelevant quantities."
        ),
    )
    evidence: list[Region] = Field(
        ...,
        min_length=1,
        description="Source regions supporting this feature; no primitive or calculation transcript.",
    )
    dimension_refs: list[str] = Field(
        ...,
        description="Supporting dim_ names, stored once in views[].dimensions. Empty if no printed dimension supports this feature; no calculation transcript.",
    )

    @model_validator(mode="after")
    def require_stable_name(self) -> Self:
        require_name(self.name, "sem_")
        require_unique(self.dimension_refs, "dimension_refs")
        for ref in self.dimension_refs:
            require_name(ref, "dim_")
        return self


class DrawingInterpretation(Contract):
    datum: str = Field(
        ...,
        min_length=1,
        description="Define the shared model origin in 3D. Replace the seeded '???'.",
    )
    views: list[DrawingView] = Field(
        ...,
        description="Registered input files with their existing names, roles and full-file Regions, plus newly identified crops. Pictorial inputs may be omitted. Add printed dimensions to the appropriate views.",
    )
    features: list[SemanticFeature] = Field(
        ...,
        description="One adopted, consistent model of the part; keep competing interpretations in questions.",
    )
    questions: list[str] = Field(
        default_factory=list,
        description=(
            "Only unresolved issues and important estimates or provisional choices; "
            "name affected features/parameters. No resolved history or routine calculations."
        ),
    )

    @model_validator(mode="after")
    def require_unique_names(self) -> Self:
        require_unique((view.name for view in self.views), "views")
        require_unique((feature.name for feature in self.features), "features")
        dimensions = [dim.name for view in self.views for dim in view.dimensions]
        require_unique(dimensions, "dimensions")
        return self

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        names = {view.name for view in self.views}
        dimensions = {dim.name for view in self.views for dim in view.dimensions}
        for view in self.views:
            if view.region.view not in names:
                raise ValueError(
                    f"{view.name}.region.view: unknown view {view.region.view}"
                )
            for dim in view.dimensions:
                if dim.region.view not in names:
                    raise ValueError(
                        f"{dim.name}.region.view: unknown view {dim.region.view}"
                    )
        for feature in self.features:
            missing = set(feature.dimension_refs) - dimensions
            if missing:
                raise ValueError(
                    f"{feature.name}: unknown dimensions {sorted(missing)}"
                )
            for index, region in enumerate(feature.evidence):
                if region.view not in names:
                    raise ValueError(
                        f"{feature.name}.evidence[{index}].view: unknown view {region.view}"
                    )
        return self
