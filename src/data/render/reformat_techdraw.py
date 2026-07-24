"""Stamp foreign technical-drawing DXFs with per-entity view layer/linetype.

Real/GT technical drawings (e.g. data/eccv2026-cad-challenge-data, and the
working test_vlm/techdraw_gt) are real SolidWorks DXFs: every primitive sits
on layer "0" and carries no per-view provenance. The dataset loader instead
reads each primitive's orthographic view straight off its DXF layer, using
exactly the lowercase names "front" / "top" / "right" (see src/data/dxf.py).
A drawing shipped this way has to be stamped before the loader can use it:
every geometry entity's layer is rewritten to the view it belongs to, and its
linetype is pinned explicitly to "Continuous" or "HIDDEN" so visible/hidden
classification cannot silently flip when the layer -- and its default
linetype -- changes underneath the entity. Geometry itself is never touched:
entity types, counts, coordinates, arc angles, and spline points come out
exactly as they went in.

View assignment relies on the third-angle L-arrangement that ``layout.py``
encodes and that was verified against the GT drawings by raster IoU::

    front  (main, bottom-left) : screen (X, Y)
    top    (above front)       : screen (X, Z)   -> shares front's x-extent
    right  (right of front)    : screen (-Z, Y)  -> shares front's y-extent

Because top reuses front's screen X and right reuses front's screen Y, one
vertical line always separates {front, top} from {right} and one horizontal
line always separates {front, right} from {top}.  So the sheet splits into an
empty top-right quadrant plus the three views, and the shared-extent
equalities above become a verification of the split rather than an
assumption about it.

Entity selection for the split mirrors what the dataset loader keeps (same
sampling, same off-sheet rejection), except it reads from the raw source
layer "0" instead of the view layers that only exist after stamping. Once
the three view bounding boxes are known, every qualifying entity is
revisited and moved onto whichever box contains its sample centroid (the
mean of the same points used for the split); an entity matching zero or more
than one box is left on "0", where the loader's layer filter simply drops
it, mirroring the ~0.01% corner-clip entities the old geometric recovery
also had to skip.

Usage:
    python src/data/render/reformat_techdraw.py --techdraw_dir <TECHDRAW_DIR>

TECHDRAW_DIR holds ``dxf/{stem}.dxf``; each file is rewritten in place,
so this is meant to run once per drawing while it is still in its raw,
all-on-layer-"0" form.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import ezdxf
import rootutils

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.dxf import (  # noqa: E402
    DXF_PRIMITIVE_TYPE_TO_ID,
    DXFParseError,
    DXFPrimitiveConfig,
    VIEW_DIRECTIONS,
    sample_dxf_entity,
)

# Views are aligned to a hair by construction; this only absorbs the flattening
# error of sampled splines/ellipses.  A mis-split shifts an extent by an entire
# view width, so any threshold in this range separates the two cases.
ALIGNMENT_TOLERANCE_MM = 1.0
# A gap narrower than this is drawing detail (a slot, a counterbore), not the
# inter-view spacing; the observed GT minimum is ~9 mm.
MIN_VIEW_GAP_MM = 2.0
CENTER_MARK_LAYER = "10"
# Layer every geometry entity starts on in a foreign (unstamped) drawing, and
# where an entity is left if its centroid cannot be assigned to one view.
UNSTAMPED_LAYER = "0"
# Generous relative to the >= MIN_VIEW_GAP_MM separation between view boxes;
# only absorbs float noise between the split pass and the mutation pass.
VIEW_MATCH_TOLERANCE_MM = 0.1


class ViewSplitError(ValueError):
    """The DXF does not decompose into a third-angle three-view arrangement."""


class _Entity:
    """One parsed primitive reduced to what the split needs."""

    __slots__ = ("x0", "y0", "x1", "y1", "cx", "cy", "hidden")

    def __init__(self, points, hidden: bool) -> None:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        self.x0, self.x1 = min(xs), max(xs)
        self.y0, self.y1 = min(ys), max(ys)
        # The loader assigns a view by the mean of the same samples, so keeping
        # the mean here is what makes "inside the bbox" agree between the two.
        self.cx = sum(xs) / len(xs)
        self.cy = sum(ys) / len(ys)
        self.hidden = hidden


def _is_hidden(document, entity) -> bool:
    linetype = str(entity.dxf.get("linetype", "BYLAYER"))
    if linetype.upper() == "BYLAYER":
        try:
            linetype = str(document.layers.get(entity.dxf.layer).dxf.linetype)
        except ezdxf.DXFTableEntryError:
            linetype = "CONTINUOUS"
    return linetype.upper().startswith("HIDDEN")


def _inside_sheet(entity: _Entity, config: DXFPrimitiveConfig) -> bool:
    if not config.discard_outside_sheet:
        return True
    tol = config.sheet_tolerance_mm
    return (
        -tol <= entity.x0
        and entity.x1 <= config.sheet_width_mm + tol
        and -tol <= entity.y0
        and entity.y1 <= config.sheet_height_mm + tol
    )


def read_entities(path: Path, config: DXFPrimitiveConfig) -> tuple[list[_Entity], int]:
    """Parse the primitives the dataset loader would keep, plus the mark count."""
    document = ezdxf.readfile(path)
    included = set(config.included_layers)
    entities: list[_Entity] = []
    n_marks = 0
    for entity in document.modelspace():
        layer = entity.dxf.layer
        if layer == CENTER_MARK_LAYER and entity.dxftype() == "INSERT":
            n_marks += 1
        if layer not in included or entity.dxftype() not in DXF_PRIMITIVE_TYPE_TO_ID:
            continue
        hidden = _is_hidden(document, entity)
        if (hidden and not config.include_hidden) or (
            not hidden and not config.include_visible
        ):
            continue
        try:
            points = sample_dxf_entity(entity, config)
        except DXFParseError:
            continue  # degenerate; the loader drops it too
        parsed = _Entity(points, hidden)
        if _inside_sheet(parsed, config):
            entities.append(parsed)
    return entities, n_marks


def _gaps(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Maximal uncovered ranges between the merged 1D projections."""
    ordered = sorted(intervals)
    merged = [list(ordered[0])]
    for low, high in ordered[1:]:
        if low <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low, high])
    return [
        (merged[index][1], merged[index + 1][0]) for index in range(len(merged) - 1)
    ]


def _bbox(group: list[_Entity]) -> tuple[float, float, float, float]:
    return (
        min(e.x0 for e in group),
        min(e.y0 for e in group),
        max(e.x1 for e in group),
        max(e.y1 for e in group),
    )


def _alignment_error(
    front: tuple[float, ...], top: tuple[float, ...], right: tuple[float, ...]
) -> float:
    """How far the split departs from the third-angle shared-extent equalities."""
    return max(
        abs(top[0] - front[0]),
        abs(top[2] - front[2]),
        abs(right[1] - front[1]),
        abs(right[3] - front[3]),
    )


def split_views(
    entities: list[_Entity],
) -> dict[str, tuple[float, float, float, float]]:
    """Partition sheet primitives into front / top / right bounding boxes.

    Candidate cuts are the gaps in the x and y projections, tried widest first.
    A candidate is accepted only when the top-right quadrant comes out empty, the
    other three are populated, and the shared-extent equalities hold — so an
    internal gap inside one view (a slot wide enough to look like view spacing)
    is rejected rather than silently splitting that view in half.
    """
    if len(entities) < 3:
        raise ViewSplitError(f"only {len(entities)} usable primitives on the sheet")
    x_gaps = _gaps([(e.x0, e.x1) for e in entities])
    y_gaps = _gaps([(e.y0, e.y1) for e in entities])
    x_gaps = [g for g in x_gaps if g[1] - g[0] >= MIN_VIEW_GAP_MM]
    y_gaps = [g for g in y_gaps if g[1] - g[0] >= MIN_VIEW_GAP_MM]
    if not x_gaps or not y_gaps:
        raise ViewSplitError(
            f"need a horizontal and a vertical separation of at least "
            f"{MIN_VIEW_GAP_MM} mm, found {len(x_gaps)} x-gap(s), "
            f"{len(y_gaps)} y-gap(s)"
        )

    best_error = float("inf")
    best_detail = ""
    for x_low, x_high in sorted(x_gaps, key=lambda g: g[0] - g[1]):
        x_cut = (x_low + x_high) / 2.0
        for y_low, y_high in sorted(y_gaps, key=lambda g: g[0] - g[1]):
            y_cut = (y_low + y_high) / 2.0
            quadrants: dict[str, list[_Entity]] = {
                "bl": [],
                "tl": [],
                "br": [],
                "tr": [],
            }
            for entity in entities:
                key = ("t" if entity.y0 > y_cut else "b") + (
                    "r" if entity.x0 > x_cut else "l"
                )
                quadrants[key].append(entity)
            if quadrants["tr"] or not all(quadrants[k] for k in ("bl", "tl", "br")):
                continue
            front = _bbox(quadrants["bl"])
            top = _bbox(quadrants["tl"])
            right = _bbox(quadrants["br"])
            error = _alignment_error(front, top, right)
            if error <= ALIGNMENT_TOLERANCE_MM:
                return {"front": front, "top": top, "right": right}
            if error < best_error:
                best_error = error
                best_detail = (
                    f"cut=({x_cut:.2f}, {y_cut:.2f}) misaligns the shared view "
                    f"extents by {error:.2f} mm"
                )
    raise ViewSplitError(
        best_detail
        or "no cut leaves the top-right quadrant empty with three populated views"
    )


def _containing_view(
    cx: float,
    cy: float,
    boxes: dict[str, tuple[float, float, float, float]],
    tolerance: float,
) -> str | None:
    """Name of the single view bbox containing (cx, cy), or None if 0 or 2+ do."""
    matches = [
        name
        for name, (x0, y0, x1, y1) in boxes.items()
        if x0 - tolerance <= cx <= x1 + tolerance
        and y0 - tolerance <= cy <= y1 + tolerance
    ]
    return matches[0] if len(matches) == 1 else None


def stamp_dxf(dxf_path: Path, config: DXFPrimitiveConfig) -> dict[str, int]:
    """Rewrite one DXF's geometry entities onto their view layer, in place.

    ``config.included_layers`` must name the layer the raw geometry lives on
    (``UNSTAMPED_LAYER`` for foreign drawings), which is also where the split
    is computed from. Returns the count assigned to each view plus the count
    left on ``UNSTAMPED_LAYER`` (ambiguous or unmatched centroid).

    Raises :class:`ViewSplitError` (via ``split_views``) if the drawing does
    not decompose into a third-angle three-view arrangement.
    """
    entities, _n_marks = read_entities(dxf_path, config)
    boxes = split_views(entities)

    document = ezdxf.readfile(dxf_path)
    for name in VIEW_DIRECTIONS:
        if name not in document.layers:
            document.layers.add(name)

    included = set(config.included_layers)
    counts = {name: 0 for name in VIEW_DIRECTIONS}
    counts[UNSTAMPED_LAYER] = 0
    for entity in document.modelspace():
        layer = entity.dxf.layer
        if layer not in included or entity.dxftype() not in DXF_PRIMITIVE_TYPE_TO_ID:
            continue
        hidden = _is_hidden(document, entity)
        if (hidden and not config.include_hidden) or (
            not hidden and not config.include_visible
        ):
            continue
        try:
            points = sample_dxf_entity(entity, config)
        except DXFParseError:
            continue  # degenerate; never entered `entities` either
        parsed = _Entity(points, hidden)
        if not _inside_sheet(parsed, config):
            continue
        # CRITICAL ordering: pin the linetype before any layer change. Moving
        # an entity onto a layer whose default linetype differs, while the
        # entity itself is BYLAYER, would otherwise silently flip its
        # visible/hidden classification.
        entity.dxf.linetype = "HIDDEN" if hidden else "Continuous"
        view = _containing_view(parsed.cx, parsed.cy, boxes, VIEW_MATCH_TOLERANCE_MM)
        if view is None:
            counts[UNSTAMPED_LAYER] += 1
            continue
        entity.dxf.layer = view
        counts[view] += 1
    document.saveas(dxf_path)
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--techdraw_dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="process at most N parts")
    args = parser.parse_args(argv)

    dxf_dir = args.techdraw_dir / "dxf"
    if not dxf_dir.is_dir():
        print(f"no dxf directory under {args.techdraw_dir}", file=sys.stderr)
        return 1
    dxf_files = sorted(dxf_dir.glob("*.dxf"))
    if args.limit:
        dxf_files = dxf_files[: args.limit]
    if not dxf_files:
        print(f"no .dxf files found in {dxf_dir}", file=sys.stderr)
        return 1

    # Foreign drawings carry every primitive on layer "0"; the split is
    # computed from that raw layer and each entity's recovered view is then
    # written onto it.
    config = DXFPrimitiveConfig(included_layers=(UNSTAMPED_LAYER,))
    n_ok = n_fail = 0
    for index, dxf_path in enumerate(dxf_files, start=1):
        try:
            counts = stamp_dxf(dxf_path, config)
        except (ViewSplitError, DXFParseError, OSError, ezdxf.DXFError) as error:
            n_fail += 1
            print(
                f"[{index}/{len(dxf_files)}] {dxf_path.name}: FAIL "
                f"{type(error).__name__}: {error}"
            )
            continue
        n_ok += 1
        print(
            f"[{index}/{len(dxf_files)}] {dxf_path.name}: "
            f"front={counts['front']} top={counts['top']} right={counts['right']} "
            f"unassigned={counts[UNSTAMPED_LAYER]}"
        )
    print(f"done: {n_ok} ok, {n_fail} failed")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
