"""Crops of an interpretation's evidence, and what the region index finds for them."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

from zeroshot.pipeline.retrieval.index import (
    MIN_SCORE,
    K,
    RegionIndex,
    RoleCrop,
    crop_region,
    describe,
    standard_turns,
    turn,
)

ORTHOGRAPHIC = {"front", "back", "top", "bottom", "left", "right"}
SANDBOX = PurePosixPath("/work")

Box = list[float]


def feature_crops(
    interpretation: dict[str, Any], workspace: Path
) -> dict[str, tuple[str, list[RoleCrop]]]:
    """Each feature's description and evidence crops, labelled by orthographic
    view and turned to its standard frame."""
    views = {view["name"]: view for view in interpretation["views"]}
    pages: dict[str, Image.Image] = {}

    def on_page(name: str, box: Box) -> tuple[str, Box]:
        """The root view a box lies on, and the box in its pixels; crops are 1:1."""
        region = views[name]["region"]
        if region["view"] == name:
            return name, box
        x0, y0 = region["box_px"][:2]
        return on_page(
            region["view"], [box[0] + x0, box[1] + y0, box[2] + x0, box[3] + y0]
        )

    def page(root: str) -> Image.Image:
        if root not in pages:
            file = PurePosixPath(views[root]["file"]).relative_to(SANDBOX)
            pages[root] = Image.open(workspace / file).convert("L")
        return pages[root]

    drawn = [
        (view["name"], *on_page(view["region"]["view"], view["region"]["box_px"]))
        for view in views.values()
        if view["role"] in ORTHOGRAPHIC and view["region"]["view"] != view["name"]
    ]

    def drawn_in(view: str, root: str, box: Box) -> dict[str, Any] | None:
        """The orthographic view a box lies in: its own, or the smallest around its centre."""
        if views[view]["role"] in ORTHOGRAPHIC:
            return views[view]
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        holders = [
            ((b[2] - b[0]) * (b[3] - b[1]), name)
            for name, r, b in drawn
            if r == root and b[0] <= cx <= b[2] and b[1] <= cy <= b[3]
        ]
        return views[min(holders)[1]] if holders else None

    features = {}
    for feature in interpretation["features"]:
        crops = []
        for evidence in feature["evidence"]:
            root, box = on_page(evidence["view"], evidence["box_px"])
            view = drawn_in(evidence["view"], root, box)
            if view is not None and not views[root]["file"].endswith(".dxf"):
                turns = standard_turns(view["role"], view["u_axis"], view["v_axis"])
                crops.append((view["role"], turn(crop_region(page(root), box), turns)))
        if crops:
            features[feature["name"]] = (feature["description"], crops)
    return features


_HEADER = """\
[Semantic feature search]
For your semantic hypotheses, searching the database with each feature's \
evidence crops returned the matches below. They are hypotheses, not facts \
about this drawing. Where one disagrees with \
your description, compare the lines of every view, solid and hidden, or build \
both shapes as trials."""


class EvidenceLookup:
    """Search the index once for each feature evidence the agent has not seen yet."""

    def __init__(self, index: RegionIndex, workspace: Path) -> None:
        self.index = index
        self.workspace = workspace
        self._searched: set[str] = set()

    def report(self, interpretation: dict[str, Any]) -> str | None:
        found = []
        for name, (description, crops) in feature_crops(
            interpretation, self.workspace
        ).items():
            key = sha256(
                json.dumps([name, [role for role, _ in crops]]).encode()
                + b"".join(image.tobytes() for _, image in crops)
            ).hexdigest()
            if key in self._searched:
                continue
            self._searched.add(key)
            matches = [
                match
                for match in self.index.search(crops, K)
                if match.score >= MIN_SCORE
            ]
            if matches:
                found.append(f"{name} (yours: {description})\n{describe(matches)}")
        return "\n\n---\n".join([_HEADER, *found]) if found else None
