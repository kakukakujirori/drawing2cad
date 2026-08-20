"""Put a predicted STEP into the ground truth's face-partition convention.

The targets in this dataset were written by SolidWorks 2025 (`SwSTEP 2.0`),
whose STEP writer never emits a closed periodic face: a full cylinder leaves as
two half-cylinders split at the seam.  A prediction built with CadQuery leaves
through Open CASCADE, which keeps the 360-degree face whole.  The ECCV metric
assigns faces one-to-one, so the same solid scores far lower from the second
writer than the first -- on 001014, whose voxel IoU against its target is
0.9998, mean F1 goes from 0.464 to 0.886 once the seams are split.

This is a dataset-specific normalization, not a metric improvement: it exists
because two writers disagree about where a face ends, and it is opt-in through
``StepScorer.split_closed_faces`` for that reason.

Like :mod:`.metrics.eccv_components._step_brep`, this module reaches for
pythonocc, so it must only be imported inside the isolated scoring subprocess.
"""

from __future__ import annotations

from pathlib import Path

# One extra split point turns a closed face into two.  More is worse, not
# better: on 001014, splitting once scores 0.886 where twice scores 0.523 and
# three times 0.460, because the target splits at the seam and nowhere else.
_SPLIT_POINTS = 1


def split_closed_faces(source_step: Path, output_step: Path) -> Path:
    """Write `source_step` with every closed periodic face split at its seam."""

    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.Interface import Interface_Static
    from OCC.Core.ShapeUpgrade import ShapeUpgrade_ShapeDivideClosed
    from OCC.Core.STEPControl import (
        STEPControl_AsIs,
        STEPControl_Reader,
        STEPControl_Writer,
    )

    reader = STEPControl_Reader()
    if reader.ReadFile(str(source_step)) != IFSelect_RetDone:
        raise ValueError(f"failed to read STEP file {source_step}")
    reader.TransferRoots()

    divider = ShapeUpgrade_ShapeDivideClosed(reader.OneShape())
    divider.SetNbSplitPoints(_SPLIT_POINTS)
    divider.Perform()

    Interface_Static.SetCVal("write.step.schema", "AP203")
    writer = STEPControl_Writer()
    writer.Transfer(divider.Result(), STEPControl_AsIs)
    output_step.parent.mkdir(parents=True, exist_ok=True)
    if writer.Write(str(output_step)) != IFSelect_RetDone:
        raise ValueError(f"failed to write STEP file {output_step}")
    return output_step


__all__ = ["split_closed_faces"]
