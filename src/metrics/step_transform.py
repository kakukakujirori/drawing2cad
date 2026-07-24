"""Similarity transforms on STEP files, used to make scale-sensitive metrics fair.

The drawings carry no dimension annotations, so a prediction's absolute size is
not recoverable from the input. Metrics whose thresholds are absolute lengths
therefore need the prediction placed in the ground truth's scale before they
mean anything. This module writes that rescaled copy.

Only a centre translation and one isotropic scale factor are applied. Rotation
is left to the consuming metric (CADGenBench runs its own rigid alignment), and
per-axis scaling is deliberately not applied: stretching a prediction onto the
target's box would hide real proportion errors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BoxNormalization:
    """The similarity transform that was applied, kept for diagnostics."""

    scale: float
    source_extent: float
    reference_extent: float


def normalize_step_to_reference(
    source: str | Path,
    reference: str | Path,
    output: str | Path,
) -> BoxNormalization:
    """Write ``source`` recentred and rescaled onto ``reference``'s bounding box.

    Both boxes are measured with the same routine, so the factor is exactly the
    ratio of the two longest extents. Uses CadQuery (and therefore the OCP
    binding) because the consumers of the result read STEP through the same
    binding.
    """

    import cadquery as cq

    output = Path(output)
    source_shape = cq.importers.importStep(str(source)).val()
    reference_shape = cq.importers.importStep(str(reference)).val()

    source_box = source_shape.BoundingBox()
    reference_box = reference_shape.BoundingBox()
    source_extent = max(source_box.xlen, source_box.ylen, source_box.zlen)
    reference_extent = max(reference_box.xlen, reference_box.ylen, reference_box.zlen)
    if not source_extent > 0.0:
        raise ValueError(f"cannot normalize a shape with zero extent: {source}")
    if not reference_extent > 0.0:
        raise ValueError(f"reference shape has zero extent: {reference}")

    factor = float(reference_extent / source_extent)
    source_centre = cq.Vector(
        (source_box.xmin + source_box.xmax) / 2.0,
        (source_box.ymin + source_box.ymax) / 2.0,
        (source_box.zmin + source_box.zmax) / 2.0,
    )
    reference_centre = cq.Vector(
        (reference_box.xmin + reference_box.xmax) / 2.0,
        (reference_box.ymin + reference_box.ymax) / 2.0,
        (reference_box.zmin + reference_box.zmax) / 2.0,
    )
    # ``Shape.scale`` scales about the origin, so the shape is centred first.
    transformed = (
        source_shape.translate(source_centre.multiply(-1.0))
        .scale(factor)
        .translate(reference_centre)
    )
    transformed.exportStep(str(output))
    if not output.is_file():
        raise RuntimeError(f"failed to write normalized STEP: {output}")
    return BoxNormalization(
        scale=factor,
        source_extent=float(source_extent),
        reference_extent=float(reference_extent),
    )


__all__ = ["BoxNormalization", "normalize_step_to_reference"]
