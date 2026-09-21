"""Pictures of what an audit finding points at, for the stages that must fix it.

The paths go on the ticket; the images themselves are never attached to a
prompt, because every round would then pay for every ticket's pixels.
"""

from math import ceil, floor
from pathlib import Path

import ezdxf
from PIL import Image

from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.stages.audit.contracts import AuditFinding, AuditRegion
from zeroshot.pipeline.verification.render.export_dxf import (
    DEFAULT_PNG_MARGIN_RATIO,
    export_to_png,
    png_bounds,
)


def crop_evidence(
    finding: AuditFinding,
    into: Path,
    workdir: SandboxWorkdir,
    *,
    margin_ratio: float = DEFAULT_PNG_MARGIN_RATIO,
) -> list[str]:
    """Write each evidence region as its own picture, in the finding's order."""
    into.mkdir(parents=True, exist_ok=True)
    written = []
    for index, region in enumerate(finding.evidence):
        crop = into / f"evidence_{index}.png"
        _write_crop(
            workdir.sandbox_to_host_path(region.file), region, crop,
            margin_ratio=margin_ratio,
        )
        written.append(str(workdir.host_to_sandbox_path(crop)))
    return written


def _write_crop(
    source: Path, region: AuditRegion, destination: Path, *, margin_ratio: float
) -> None:
    """Cut the region out of its own file, rasterising a DXF whole first."""
    sheet = (
        export_to_png(
            source, destination.with_stem(f"{destination.stem}_sheet"),
            margin_ratio=margin_ratio,
        )
        if source.suffix.lower() == ".dxf"
        else None
    )
    with Image.open(sheet or source) as image:
        image.crop(
            _pixels_of(region, source, image.size, margin_ratio=margin_ratio)
        ).save(destination)
    if sheet is not None:
        sheet.unlink()


def _pixels_of(
    region: AuditRegion, source: Path, size: tuple[int, int], *, margin_ratio: float
) -> tuple[int, int, int, int]:
    """The region as whole pixels of its picture, never thinner than one.

    A raster box is already in pixels. A DXF is measured in the millimetres it
    carries, mapped through the same padded bounds used when rendering.
    """
    x0, y0, x1, y1 = region.box
    if source.suffix.lower() == ".dxf":
        left, bottom, right, top = png_bounds(
            ezdxf.readfile(source).modelspace(), margin_ratio=margin_ratio
        )
        x_scale = size[0] / (right - left)
        y_scale = size[1] / (top - bottom)
        x0, x1 = (x0 - left) * x_scale, (x1 - left) * x_scale
        y0, y1 = (  # picture y runs down
            size[1] - (y1 - bottom) * y_scale,
            size[1] - (y0 - bottom) * y_scale,
        )
    return (
        max(floor(x0), 0),
        max(floor(y0), 0),
        min(ceil(x1), size[0]),
        min(ceil(y1), size[1]),
    )
