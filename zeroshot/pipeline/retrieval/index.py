"""Find drawn features that look like a query feature in every view it shows.

Crops are stored and searched turned to their view's standard frame
(STANDARD_VIEW_FRAMES), so a search can try every orientation of the part.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from io import BytesIO
from itertools import permutations, product
from pathlib import Path
from string import ascii_uppercase
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from zeroshot.pipeline.stages.interpretation.contracts import AXIS_VECTOR, Axis
from zeroshot.pipeline.verification.render.orthographic import STANDARD_VIEW_FRAMES

ENCODER = "facebook/dinov3-vits16-pretrain-lvd1689m"
# What a DB records: vectors of another model or recipe do not compare.
EMBEDDING = f"{ENCODER} cls+patch-mean thickened"
SIZE = 224
MARGIN = 0.25  # context kept around a box, as a fraction of its longer side
# Below this a match's kind (material or void) agrees with its query's no more
# than chance, on a 61-part Sol DB queried with each part left out.
MIN_SCORE = 0.70
K = 10
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS encoder (name TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS feature (
    id INTEGER PRIMARY KEY,
    part TEXT NOT NULL,
    source TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    meta TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS crop (
    feature INTEGER NOT NULL REFERENCES feature(id),
    role TEXT NOT NULL,
    png BLOB NOT NULL,
    vector BLOB NOT NULL
);
"""

RoleCrop = tuple[str, Image.Image]


def crop_region(page: Image.Image, box: Sequence[float]) -> Image.Image:
    """The box with some context around it, clipped to the page."""
    x0, y0, x1, y1 = box
    pad = MARGIN * max(x1 - x0, y1 - y0) + 4
    return page.crop(
        (
            round(max(0, x0 - pad)),
            round(max(0, y0 - pad)),
            round(min(page.width, x1 + pad)),
            round(min(page.height, y1 + pad)),
        )
    )


_TURNS = (
    None,
    Image.Transpose.ROTATE_90,
    Image.Transpose.ROTATE_180,
    Image.Transpose.ROTATE_270,
)


def turn(image: Image.Image, quarter_turns: int) -> Image.Image:
    """The image turned counterclockwise by that many quarter turns."""
    transpose = _TURNS[quarter_turns % 4]
    return image if transpose is None else image.transpose(transpose)


def _quarter_turns(right: np.ndarray, frame: tuple[np.ndarray, np.ndarray]) -> int:
    """Turns that bring a view whose rightward model axis is `right` into `frame`."""
    u, v = frame
    for turns, axis in enumerate((u, v, -u, -v)):
        if np.array_equal(right, axis):
            return turns
    raise ValueError(f"{right} does not lie in the view's plane")


_FRAMES = {
    str(view): (np.array(AXIS_VECTOR[u]), np.array(AXIS_VECTOR[v]))
    for view, (u, v) in STANDARD_VIEW_FRAMES.items()
}


def standard_turns(role: str, u_axis: Axis, v_axis: Axis) -> int:
    """Turns that bring a view whose rightward model axis is `u_axis` to its standard frame."""
    return _quarter_turns(np.array(AXIS_VECTOR[u_axis]), _FRAMES[role])


def _rotations() -> list[dict[str, tuple[str, int]]]:
    """For each of the 24 cube rotations: the view each view becomes, and its turns."""
    toward = {role: np.cross(u, v) for role, (u, v) in _FRAMES.items()}
    found = []
    for axes in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            rotation = np.diag(signs) @ np.eye(3, dtype=int)[list(axes)]
            if round(np.linalg.det(rotation)) != 1:
                continue
            moves = {}
            for role, (u, v) in _FRAMES.items():
                out = rotation @ np.cross(u, v)
                target = next(r for r, w in toward.items() if np.array_equal(w, out))
                moves[role] = (target, _quarter_turns(rotation @ u, _FRAMES[target]))
            found.append(moves)
    return found


ROTATIONS = _rotations()


_ENCODER_LOCK = threading.Lock()


def _encoder() -> Any:
    # Parallel tool calls embed at once; transformers' lazy import is not thread-safe.
    with _ENCODER_LOCK:
        return _load_encoder()


@lru_cache(maxsize=1)
def _load_encoder() -> Any:
    from transformers import AutoModel

    return AutoModel.from_pretrained(ENCODER).eval()


def _pixels(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    shrink = max(image.size) / SIZE
    if shrink >= 2:
        # Thicken the ink first, or downscaling leaves thin lines as grey haze.
        image = image.filter(ImageFilter.MinFilter(min(int(shrink) | 1, 9)))
    square = ImageOps.pad(image, (SIZE, SIZE), color="white")
    return ((np.asarray(square, np.float32) / 255 - _MEAN) / _STD).transpose(2, 0, 1)


def embed(images: Sequence[Image.Image]) -> np.ndarray:
    """One unit vector per image: CLS joined to the mean patch token."""
    import torch

    with torch.inference_mode():
        batch = torch.from_numpy(np.stack([_pixels(image) for image in images]))
        model = _encoder()
        out = model(pixel_values=batch)
        # Register tokens sit between CLS and the patches.
        first_patch = 1 + model.config.num_register_tokens
        patches = out.last_hidden_state[:, first_patch:].mean(dim=1)
        vectors = torch.cat([out.pooler_output, patches], dim=1).numpy()
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def _png(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@dataclass(frozen=True)
class Reference:
    part: str
    source: str
    name: str
    description: str
    meta: dict[str, Any]
    score: float
    # The best match for each query crop, turned to line up with it.
    crops: list[RoleCrop] = field(repr=False)


def unranked(matches: Sequence[Reference]) -> list[Reference]:
    """Matches in no order of score: among close ones, it says little about kind."""
    return sorted(matches, key=lambda match: (match.part, match.name))


def describe(matches: Sequence[Reference]) -> str:
    return "\n\n".join(
        f"{letter}. {match.description}"
        for letter, match in zip(ascii_uppercase, matches, strict=False)
    )


def _view_scores(sims: np.ndarray, table: _Crops, role: str, size: int) -> np.ndarray:
    """Per feature, the best cosine among its crops of that view."""
    same = table.role == role
    # A feature without this view scores what an average feature scores.
    best = np.full(size, sims[same].mean() if same.any() else sims.mean(), np.float32)
    best[table.feature[same]] = -np.inf
    np.maximum.at(best, table.feature[same], sims[same])
    return best


@dataclass(frozen=True)
class _Crops:
    rowid: np.ndarray
    feature: np.ndarray
    role: np.ndarray
    vector: np.ndarray


class RegionIndex:
    """Features of other drawings, one crop per view, in a SQLite file."""

    def __init__(self, path: Path | str) -> None:
        # Tools run on worker threads; the tool only reads.
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(_SCHEMA)
        self._loaded: _Crops | None = None
        self._check_encoder(path)

    def _check_encoder(self, path: Path | str) -> None:
        """Vectors of different encoders do not compare; refuse a mixed index."""
        recorded = self.db.execute("SELECT name FROM encoder").fetchone()
        if recorded is None and self.db.execute("SELECT 1 FROM crop").fetchone():
            raise ValueError(f"{path} records no encoder; rebuild it with {EMBEDDING}")
        if recorded is None:
            with self.db:
                self.db.execute("INSERT INTO encoder VALUES (?)", (EMBEDDING,))
        elif recorded[0] != EMBEDDING:
            raise ValueError(f"{path} was built with {recorded[0]}, not {EMBEDDING}")

    def add(
        self,
        part: str,
        source: str,
        name: str,
        description: str,
        crops: Sequence[RoleCrop],
        meta: dict[str, Any] | None = None,
    ) -> None:
        vectors = embed([image for _, image in crops])
        with self.db:
            feature = self.db.execute(
                "INSERT INTO feature (part, source, name, description, meta) "
                "VALUES (?, ?, ?, ?, ?)",
                (part, source, name, description, json.dumps(meta or {})),
            ).lastrowid
            self.db.executemany(
                "INSERT INTO crop VALUES (?, ?, ?, ?)",
                [
                    (feature, role, _png(image), vector.astype(np.float32).tobytes())
                    for (role, image), vector in zip(crops, vectors, strict=True)
                ],
            )
        self._loaded = None

    def search(self, crops: Sequence[RoleCrop], k: int = 3) -> list[Reference]:
        """Rank features by their best orientation of the query part; under each,
        the mean over query views of the best cosine in the view it turns into."""
        table = self._crops()
        if not crops or table.rowid.size == 0:
            return []
        turned = embed([turn(image, t) for t in range(4) for _, image in crops])
        # ponytail: brute force over every crop; past ~1e6 crops, take candidates
        # from an ANN index and estimate the neutral score from a fixed sample.
        sims = (turned @ table.vector.T).reshape(
            4, len(crops), -1
        )  # turns, query, crop
        size = table.feature.max() + 1
        score = np.full(size, -np.inf)
        pose = np.zeros(size, int)
        for number, moves in enumerate(ROTATIONS):
            posed = np.mean(
                [
                    _view_scores(sims[moves[role][1], row], table, moves[role][0], size)
                    for row, (role, _) in enumerate(crops)
                ],
                axis=0,
            )
            better = posed > score
            score[better], pose[better] = posed[better], number
        known = np.zeros(size, bool)
        known[table.feature] = True
        score[~known] = -np.inf
        top = [int(f) for f in np.argsort(-score)[:k] if np.isfinite(score[f])]
        return [
            self._reference(f, float(score[f]), crops, sims, table, ROTATIONS[pose[f]])
            for f in top
        ]

    def _reference(
        self,
        feature: int,
        score: float,
        crops: Sequence[RoleCrop],
        sims: np.ndarray,
        table: _Crops,
        moves: dict[str, tuple[str, int]],
    ) -> Reference:
        part, source, name, description, meta = self.db.execute(
            "SELECT part, source, name, description, meta FROM feature WHERE id = ?",
            (feature,),
        ).fetchone()
        matched: list[RoleCrop] = []
        for row, (role, _) in enumerate(crops):
            target, turns = moves[role]
            mine = np.flatnonzero((table.feature == feature) & (table.role == target))
            if mine.size == 0:
                continue
            rowid = int(table.rowid[mine[np.argmax(sims[turns, row, mine])]])
            (png,) = self.db.execute(
                "SELECT png FROM crop WHERE rowid = ?", (rowid,)
            ).fetchone()
            matched.append((role, turn(Image.open(BytesIO(png)), -turns)))
        return Reference(
            part, source, name, description, json.loads(meta), score, matched
        )

    def _crops(self) -> _Crops:
        if self._loaded is None:
            rows = self.db.execute(
                "SELECT rowid, feature, role, vector FROM crop"
            ).fetchall()
            self._loaded = _Crops(
                rowid=np.array([r[0] for r in rows], np.int64),
                feature=np.array([r[1] for r in rows], np.int64),
                role=np.array([r[2] for r in rows], object),
                vector=np.stack([np.frombuffer(r[3], np.float32) for r in rows])
                if rows
                else np.zeros((0, 0), np.float32),
            )
        return self._loaded
