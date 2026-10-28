"""Question a pictorial view drawn where a missing orthographic view belongs."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.stages.interpretation.contracts import (
    PICTORIAL_VIEWS,
    VIEW_PLACEMENT,
    DrawingInterpretation,
    ProjectionMethod,
    View,
)
from zeroshot.pipeline.stages.interpretation.validate import (
    _SIDE_OF,
    _Box,
    _page_centre,
)

# A projected view's outline ends where front's does along the axis they share,
# to within this fraction of the page's longer side.
_PROJECTION_TOLERANCE = 0.01


def _line_pixels(path: Path) -> np.ndarray:
    """The page's line pixels, at a level each page sets: sheets draw in black or gray."""
    with Image.open(path) as page:
        white = Image.new("RGBA", page.size, "white")
        gray = np.asarray(
            Image.alpha_composite(white, page.convert("RGBA")).convert("L")
        )
    level, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return gray <= level  # inclusive: a two-tone page puts the level at black


def _outline(lines: np.ndarray, box: _Box) -> _Box | None:
    """The page box of the largest connected drawing inside a crop of line pixels."""
    x0, y0, x1, y1 = (int(edge) for edge in box)
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        lines[y0:y1, x0:x1].astype(np.uint8), connectivity=8
    )
    if count < 2:
        return None
    x, y, width, height = stats[1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])), :4]
    return (box[0] + x, box[1] + y, box[0] + x + width, box[1] + y + height)


def pictorials_in_projected_places(
    interpretation: DrawingInterpretation,
    *,
    workdir: SandboxWorkdir,
    projection_method: ProjectionMethod,
) -> list[str]:
    """Question each pictorial drawn where a missing orthographic view would be.

    It lies off front on a side whose role no view holds, and its outline ends
    where front's does across that side, as a projected view's does.
    """
    views = interpretation.views
    front = next((view for view in views if view.role is View.FRONT), None)
    if front is None or (front_box := front.region.box_px) is None:
        return []
    roles = {view.role for view in views}
    empty = {
        role: step
        for role, step in VIEW_PLACEMENT[projection_method].items()
        if role not in roles
    }
    pictorials = [
        (view, box)
        for view in views
        if view.role in PICTORIAL_VIEWS
        and (box := view.region.box_px) is not None
        and view.region.view == front.region.view
    ]
    if not empty or not pictorials:
        return []

    page = next(view for view in views if view.name == front.region.view)
    lines = _line_pixels(workdir.sandbox_to_host_path(page.file))
    front_outline = _outline(lines, front_box)
    if front_outline is None:
        return []
    tolerance = _PROJECTION_TOLERANCE * max(lines.shape)
    fx, fy = _page_centre(front.region)
    method = projection_method.value.replace("_", "-")
    questions = []
    for view, box in pictorials:
        outline = _outline(lines, box)
        if outline is None:
            continue
        x, y = _page_centre(view.region)
        for role, step in empty.items():
            axis = 0 if step[0] == 0 else 1  # above or below: x ends; beside: y ends
            toward = (x - fx) * step[0] + (y - fy) * step[1]
            gap = min(abs(front_outline[i] - outline[i]) for i in (axis, axis + 2))
            if toward <= 0 or gap > tolerance:
                continue
            questions.append(
                f"Highly suspicious: {view.name} is registered as {view.role.value}, "
                f"yet it sits {_SIDE_OF[step]} {front.name} and lines up with it, "
                f"exactly where {method} projection draws the {role.value} view, "
                f"and no {role.value} view is registered. A drawing places a "
                "pictorial view apart from its orthographic views, never in the place "
                "of one, and a view that carries dimensions is orthographic. "
                f"{view.name} should be the {role.value} view. "
                f"Look at it again as the {role.value} view, compare feature positions "
                "and dimension spans along the axes shared with the other source views, "
                "and revise its role, axes and every feature read from it. "
                f"Note: once {view.name} is registered as the {role.value} view, "
                "it counts toward the match score, so the score may rise. "
                f"That is never a reason to keep it as {view.role.value}."
            )
            break
    return questions
