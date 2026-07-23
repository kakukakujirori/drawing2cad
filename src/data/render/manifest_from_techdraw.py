"""Reconstruct ``manifest.jsonl`` from techdraw DXFs when no STEP source exists.

``render_dataset.py`` emits the manifest as a *by-product* of the STEP -> techdraw
projection: the per-view bounding boxes come straight out of the layout stage,
which knows which projection produced which cluster of edges.  Datasets that ship
the drawings already rendered (data/eccv2026-cad-challenge-data) have no such
provenance, so the same metadata has to be recovered from the DXF alone.

Recovery relies on the third-angle L-arrangement that ``layout.py`` encodes and
that was verified against the GT drawings by raster IoU::

    front  (main, bottom-left) : screen (X, Y)
    top    (above front)       : screen (X, Z)   -> shares front's x-extent
    right  (right of front)    : screen (-Z, Y)  -> shares front's y-extent

Because top reuses front's screen X and right reuses front's screen Y, one
vertical line always separates {front, top} from {right} and one horizontal line
always separates {front, right} from {top}.  So the sheet splits into an empty
top-right quadrant plus the three views, and the shared-extent equalities above
become a verification of the split rather than an assumption about it.

Entity selection mirrors :class:`~src.data.dxf.DXFPrimitiveParser` exactly (same
layers, same sampling, same off-sheet rejection), so every primitive the loader
will later assign to a view is guaranteed to fall inside the bbox written here.
The renderer instead records the analytic HLR extent; the two differ only by the
curve-flattening error (< 0.05 mm), and nothing downstream consumes the bbox for
anything but view assignment.

Usage:
    python src/data/render/manifest_from_techdraw.py --data_dir <DATA_DIR>

DATA_DIR holds ``techdraw/dxf/{stem}.dxf``; ``manifest.jsonl`` is written into
DATA_DIR itself, replacing any existing one (the result is a pure function of the
DXFs, so there is nothing to resume).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import ezdxf
import rootutils

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.dxf import (  # noqa: E402
    DXF_PRIMITIVE_TYPE_TO_ID,
    DXFParseError,
    DXFPrimitiveConfig,
    sample_dxf_entity,
)
from src.data.render.config import PartResult, render3d_paths  # noqa: E402
from src.data.render.render_dataset import MANIFEST_NAME  # noqa: E402

# Views are aligned to a hair by construction; this only absorbs the flattening
# error of sampled splines/ellipses.  A mis-split shifts an extent by an entire
# view width, so any threshold in this range separates the two cases.
ALIGNMENT_TOLERANCE_MM = 1.0
# A gap narrower than this is drawing detail (a slot, a counterbore), not the
# inter-view spacing; the observed GT minimum is ~9 mm.
MIN_VIEW_GAP_MM = 2.0
CENTER_MARK_LAYER = "10"


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


def build_record(dxf_path: Path, config: DXFPrimitiveConfig, data_dir: Path) -> dict:
    """One manifest row, schema-identical to what ``render_dataset.py`` writes."""
    started = time.time()
    result = PartResult(name=dxf_path.stem, ok=True)
    # Nothing here renders; the flag reports whether the shipped PNGs — which the
    # loader needs as model input — are actually present for this part.
    r3 = render3d_paths(data_dir, dxf_path.stem)
    result.render3d_ok = all(
        path.is_file() for path in (r3.hlg, r3.shaded, r3.hlg_translucent)
    )
    try:
        entities, n_marks = read_entities(dxf_path, config)
        boxes = split_views(entities)
        views = {}
        for name, box in boxes.items():
            inside = [
                e
                for e in entities
                if box[0] <= e.cx <= box[2] and box[1] <= e.cy <= box[3]
            ]
            views[name] = {
                "bbox": [round(value, 3) for value in box],
                "visible": sum(1 for e in inside if not e.hidden),
                "hidden": sum(1 for e in inside if e.hidden),
            }
        result.extra["techdraw"] = {
            # The drawing scale is provenance the DXF simply does not carry; the
            # metadata contract already makes it optional.
            "bbox_format": "xyxy",
            "bbox_coordinate_system": {
                "unit": "mm",
                "origin": "sheet_bottom_left",
                "x_axis": "right",
                "y_axis": "up",
            },
            "cluster_bbox": [
                round(value, 3)
                for value in (
                    min(box[0] for box in boxes.values()),
                    min(box[1] for box in boxes.values()),
                    max(box[2] for box in boxes.values()),
                    max(box[3] for box in boxes.values()),
                )
            ],
            "views": views,
            "n_center_marks": n_marks,
            "source": "manifest_from_techdraw",
        }
        result.techdraw_ok = True
    except (ViewSplitError, DXFParseError, OSError, ezdxf.DXFError) as error:
        result.ok = False
        result.error = f"{type(error).__name__}: {error}"
    result.seconds = round(time.time() - started, 3)
    return result.__dict__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="process at most N parts")
    args = parser.parse_args(argv)

    dxf_dir = args.data_dir / "techdraw" / "dxf"
    if not dxf_dir.is_dir():
        print(f"no techdraw/dxf directory under {args.data_dir}", file=sys.stderr)
        return 1
    dxf_files = sorted(dxf_dir.glob("*.dxf"))
    if args.limit:
        dxf_files = dxf_files[: args.limit]
    if not dxf_files:
        print(f"no .dxf files found in {dxf_dir}", file=sys.stderr)
        return 1

    config = DXFPrimitiveConfig()
    manifest_path = args.data_dir / MANIFEST_NAME
    tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    n_ok = n_fail = 0
    with tmp_path.open("w") as handle:
        for index, dxf_path in enumerate(dxf_files, start=1):
            record = build_record(dxf_path, config, args.data_dir)
            handle.write(json.dumps(record) + "\n")
            n_ok += bool(record["ok"])
            n_fail += not record["ok"]
            if not record["ok"]:
                name, error = record["name"], record["error"]
                print(f"[{index}/{len(dxf_files)}] {name}: FAIL {error}")
    os.replace(tmp_path, manifest_path)
    print(f"done: {n_ok} ok, {n_fail} failed -> {manifest_path}")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
