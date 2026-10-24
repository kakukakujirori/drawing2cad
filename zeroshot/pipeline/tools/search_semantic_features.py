"""Find semantic features whose drawings look like a region of the drawing."""

import json
import threading
from pathlib import Path
from string import ascii_uppercase
from typing import Any, Literal

from langchain_core.tools import BaseTool, tool
from PIL import Image, ImageDraw, ImageOps
from pydantic import BaseModel, Field, field_validator, model_validator

from zeroshot.pipeline.retrieval.index import (
    MIN_SCORE,
    K,
    Reference,
    RegionIndex,
    crop_region,
    standard_turns,
    turn,
    unranked,
)
from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.stages.interpretation.contracts import (
    TOWARD_VIEWER,
    Axis,
    View,
    cross_axis,
)
from zeroshot.pipeline.tools.errors import ToolFeedbackError
from zeroshot.pipeline.tools.load_image import resolve_image_path

CELL = 512  # most crops fit at full size
SAVED_DIRNAME = "semantic_search"
_SAVE_LOCK = threading.Lock()


class RegionItem(BaseModel):
    image_path: str = Field(
        description="A workspace image: the drawing or a crop of it."
    )
    box_px: list[float] = Field(
        min_length=4, max_length=4, description="x0, y0, x1, y1 in that image's pixels."
    )
    role: Literal["front", "back", "top", "bottom", "left", "right"] = Field(
        description="The view the box lies in."
    )
    u_axis: Axis = Field(
        description="Model axis pointing rightwards in that view, as in your interpretation."
    )
    v_axis: Axis = Field(
        description="Model axis pointing upwards in that view, as in your interpretation."
    )

    @model_validator(mode="after")
    def _axes_fit_the_view(self) -> "RegionItem":
        out = TOWARD_VIEWER[View(self.role)]
        if cross_axis(self.u_axis, self.v_axis) != out:
            raise ValueError(
                f"u_axis {self.u_axis} cross v_axis {self.v_axis} must be {out}, "
                f"the axis pointing at the viewer of a {self.role} view"
            )
        return self


class SearchSemanticFeaturesInput(BaseModel):
    items: list[RegionItem] = Field(
        min_length=1,
        max_length=6,
        description="For each view that shows the feature, one box tight around its lines.",
    )

    @field_validator("items")
    @classmethod
    def _one_box_per_view(cls, items: list[RegionItem]) -> list[RegionItem]:
        roles = [item.role for item in items]
        if len(set(roles)) < len(roles):
            raise ValueError(f"give one box per view; roles repeat in {roles}")
        return items


_DESCRIPTION = """\
Search a database for the semantic features that match one feature of your \
drawing. Use it when you are unsure which 3D shape a region shows, e.g. a hole \
or a boss, before you describe it.

For each view that shows the feature, give that view's role and axes as in your \
interpretation, and one box in the pixels of its image. Draw the box tight around \
the feature's own lines and leave out its neighbours: the database holds \
features boxed this way, and extra lines match other features. Include every \
view that shows the feature: a 3D shape can only be determined by comparing \
multiple views.

The result lists, in no order, the semantic descriptions of up to ten features \
of other drawings that look like yours. Looking alike does not make them the \
same shape: a hole and a boss can look the same in one view. Take each as a \
hypothesis. Where one differs from your reading, find the view whose lines \
decide between them, solid or hidden, and read it. Take sizes and positions \
from your own drawing. Each comes with an image you can load: your crops in the \
top row, its crops below, one column per view."""


def create_search_semantic_features_tool(
    workdir: SandboxWorkdir, db: str | Path
) -> BaseTool:
    path = Path(db).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"No region index at {path}")
    index = RegionIndex(path)

    @tool(
        "search_semantic_features",
        args_schema=SearchSemanticFeaturesInput,
        description=_DESCRIPTION,
    )
    def search_semantic_features(items: list[RegionItem]) -> list[dict[str, Any]]:
        crops = [(item.role, _crop(workdir, item)) for item in items]
        turns = {
            item.role: standard_turns(item.role, item.u_axis, item.v_axis)
            for item in items
        }
        found = index.search(
            [(role, turn(image, turns[role])) for role, image in crops], k=K
        )
        close = unranked([match for match in found if match.score >= MIN_SCORE])
        images = _save(workdir, items, found, crops, close, turns)
        if not close:
            best = f"{found[0].score:.2f}" if found else "none"
            return [
                _text(
                    "No semantic feature in the database matches this region "
                    f"closely enough (best similarity {best}, needed "
                    f"{MIN_SCORE}). Read it from your drawing alone."
                )
            ]
        return [
            _text(
                "\n\n".join(
                    f"{letter}. {match.description}\n"
                    f"Crops: {workdir.host_to_sandbox_path(image)}"
                    for letter, match, image in zip(
                        ascii_uppercase, close, images, strict=False
                    )
                )
            )
        ]

    return search_semantic_features


def _save(
    workdir: SandboxWorkdir,
    items: list[RegionItem],
    found: list[Reference],
    crops: list[tuple[str, Image.Image]],
    close: list[Reference],
    turns: dict[str, int],
) -> list[Path]:
    """Number each search; keep its record, and one image per shown match."""
    folder = workdir.host_bind_dir / SAVED_DIRNAME
    folder.mkdir(exist_ok=True)
    letters = {
        (match.part, match.name): letter
        for letter, match in zip(ascii_uppercase, close, strict=False)
    }
    record = {
        "items": [item.model_dump() for item in items],
        "matches": [
            {
                "part": match.part,
                "source": match.source,
                "name": match.name,
                "score": match.score,
                "shown_as": letters.get((match.part, match.name)),
                "description": match.description,
            }
            for match in found
        ],
    }
    with _SAVE_LOCK:
        stem = folder / f"{len(list(folder.glob('*.json'))) + 1:03d}"
        stem.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    images = []
    for letter, match in zip(ascii_uppercase, close, strict=False):
        stem.mkdir(exist_ok=True)
        images.append(stem / f"{letter}.png")
        _sheet(crops, letter, match, turns).save(images[-1])
    return images


def _crop(workdir: SandboxWorkdir, item: RegionItem) -> Image.Image:
    image = Image.open(resolve_image_path(workdir, item.image_path)).convert("L")
    x0, y0, x1, y1 = item.box_px
    if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
        raise ToolFeedbackError(
            f"box_px {item.box_px} must satisfy 0 <= x0 < x1 <= {image.width} and "
            f"0 <= y0 < y1 <= {image.height} in {item.image_path}"
        )
    return crop_region(image, item.box_px)


def _sheet(
    crops: list[tuple[str, Image.Image]],
    letter: str,
    match: Reference,
    turns: dict[str, int],
) -> Image.Image:
    """Your crops over the match's, turned back to the way the drawing shows each view."""
    rows = [
        ("yours", dict(crops)),
        (letter, {role: turn(image, -turns[role]) for role, image in match.crops}),
    ]
    sheet = Image.new("L", (CELL * len(crops), CELL * len(rows)), 255)
    draw = ImageDraw.Draw(sheet)
    for r, (label, by_role) in enumerate(rows):
        for c, (role, _) in enumerate(crops):
            x, y = c * CELL, r * CELL
            if role in by_role:
                cell = ImageOps.pad(by_role[role], (CELL - 8, CELL - 20), color=255)
                sheet.paste(cell, (x + 4, y + 16))
            draw.rectangle((x, y, x + CELL - 1, y + CELL - 1), outline=128)
            draw.text((x + 4, y + 2), f"{label} {role}", fill=0)
    return sheet


def _text(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}
