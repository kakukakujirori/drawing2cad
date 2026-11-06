"""Which ink of a drawing is text, where neither tone nor position can tell.

Text printed in the part's own tone inside its outline otherwise counts as part
lines: the output is measured against it, and it shows as a missing line.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import cv2
import numpy as np

from .image_ops import distance_map
from .ocr import TextLine
from .unmatched import SHARED_PX

# OCR also reads line work, such as parallel dashes as "II" or a mark as a symbol,
# mostly with less confidence than this.
MIN_CONFIDENCE = 0.7
# Fewer letters can be line work whatever the confidence.
MIN_LETTERS = 3
# A hole seen end-on, or two of them, read as a number.
ROUND_NUMBERS = frozenset({"0", "8", "00"})


def text_regions(lines: Iterable[TextLine], shape: tuple[int, int]) -> np.ndarray:
    """The areas of the lines taken to be text, on a grid of the given shape."""
    regions = np.zeros(shape, np.uint8)
    for line in lines:
        if _is_text(line):
            cv2.fillPoly(regions, [np.rint(line.corners).astype(np.int32)], 1)
    return regions.astype(bool)


def text_marks(
    drawing_ink: np.ndarray, regions: np.ndarray, output: np.ndarray
) -> np.ndarray:
    """Ink inside the text regions that no output line explains.

    A part line can cross a region, and a region can cover line work. Either is
    then near the output's lines on a build that draws it, and stays compared.
    """
    if not regions.any():
        return regions
    return drawing_ink & regions & (distance_map(output) > SHARED_PX)


def _is_text(line: TextLine) -> bool:
    if line.confidence < MIN_CONFIDENCE:
        return False
    string = line.string.strip()
    if len(re.findall(r"[A-Za-z]", string)) >= MIN_LETTERS:
        return True
    return bool(re.search(r"[0-9]", string)) and string not in ROUND_NUMBERS
