"""Decide whether a sheet shows tangent boundaries, and draw its views without them.

Whether a drawing draws the boundary between two smooth faces is a drafting
convention of the whole sheet, so it is decided once for all of its views. The
sheet omits them if almost none of the output's tangent boundaries lie on an
input line that the output's required lines do not already explain. Once a sheet
is taken to draw them, every boundary stays, however far it has moved.

A build far from the drawing shows little either way: too little evidence leaves
the boundaries drawn, and a view without a sound alignment gives no evidence.

Display only: alignment, scores and difference clusters use every tangent
boundary in either case.

Enabled by diff_drawer_config.auto_hide_tangents. To remove this feature, delete
this module and tests/zeroshot/verification/test_tangent_display.py, resolve every
`hide_tangents` reference (it also finds the flag), and drop the "may be omitted"
legend line in stages/coding/verify.py.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

import cv2
import numpy as np
from ezdxf.layouts import Modelspace
from skimage.morphology import thin

from .align import opencv_transform
from .image_ops import beside_part, distance_map, drawing_masks, foreground_mask
from .unmatched import SHARED_PX, warp_output_to_drawing

if TYPE_CHECKING:
    from .worker import AlignedView

# Chosen on ground-truth and predicted builds of ECCV, IterCAD and Ortho2CAD:
# a sheet omits tangents when less than this share of their length is drawn...
MIN_DRAWN_FRACTION = 0.15
# ...and at least this much of their length, in input pixels, could have shown it.
MIN_INFORMATIVE_PX = 400


def measure_tangent_evidence(
    tangent: np.ndarray, part_ink: np.ndarray, required: np.ndarray
) -> dict[str, int]:
    """Count a view's tangent pixels on the input drawing's pixel grid.

    total_px: all of them. informative_px: those away from required output lines,
    where the input could show the boundary. drawn_px: those on input part lines
    that no required output line explains.
    """
    edges = thin(tangent)
    near_required = distance_map(thin(required)) <= SHARED_PX
    unexplained = thin(part_ink) & ~near_required
    drawn = distance_map(unexplained) <= SHARED_PX
    return {
        "total_px": int(edges.sum()),
        "informative_px": int((edges & ~near_required).sum()),
        "drawn_px": int((edges & drawn).sum()),
    }


def view_evidence(view: AlignedView) -> dict[str, int] | None:
    """Measure one registered view; None when it can say nothing about the sheet."""
    alignment = view.alignment
    if view.tangent_mask is None or alignment.status != "ok":
        return None
    try:
        # The part ink the comparison measures against, drafting marks included:
        # they can be real tangents.
        drawing_gray = cv2.cvtColor(view.drawing, cv2.COLOR_RGB2GRAY)
        drawing_ink = foreground_mask(drawing_gray)
        matrix = opencv_transform(alignment.H_drawing_to_projection)
        shape = drawing_ink.shape
        required = warp_output_to_drawing(
            foreground_mask(cv2.cvtColor(view.non_tangent_rgb, cv2.COLOR_RGB2GRAY)),
            matrix,
            shape,
        )
        masks = drawing_masks(drawing_gray, drawing_ink, required)
        part_ink = drawing_ink & ~masks.annotation
        non_part = masks.annotation | beside_part(part_ink, masks.outline)
        return measure_tangent_evidence(
            warp_output_to_drawing(view.tangent_mask, matrix, shape),
            drawing_ink & ~non_part,
            required,
        )
    except (ValueError, cv2.error):
        return None


def evidence_omits_tangents(evidence: Iterable[Mapping[str, int]]) -> bool:
    """Pool the views' evidence and decide the sheet's drafting convention."""
    views = list(evidence)
    total, informative, drawn = (
        sum(view[key] for view in views)
        for key in ("total_px", "informative_px", "drawn_px")
    )
    return informative >= MIN_INFORMATIVE_PX and drawn < MIN_DRAWN_FRACTION * total


def sheet_omits_tangents(views: Iterable[AlignedView]) -> bool:
    """Whether a sheet, given all of its registered views, omits tangent boundaries."""
    return evidence_omits_tangents(filter(None, map(view_evidence, views)))


def projection_without_tangents(modelspace: Modelspace) -> np.ndarray:
    """The view's ordinary PNG, on its usual pixel grid, with no tangent boundary."""
    from ..render.export_dxf import create_drawing_figure

    figure = create_drawing_figure(
        modelspace, filter_func=lambda entity: entity.dxf.layer != "TANGENT"
    )
    figure.canvas.draw()
    return np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
