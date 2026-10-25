"""Rows the stages pass on, one JSON object per line."""

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from zeroshot.pipeline.retrieval.mistakes import Decision


class Pair(BaseModel):
    """A reading of a drawing and the build made under it."""

    key: str
    model: str
    dataset: str
    part: str
    which: Literal["first", "final"]
    drawing: str
    gt: str
    interpretation: str
    build: str


class Component(BaseModel):
    """A connected piece of missing (M) or extra (E) material."""

    id: str
    share: float  # of the part's volume


class Comparison(BaseModel):
    key: str
    iou: float
    rotation_index: int
    components: list[Component]
    images: dict[str, str]


class Call(BaseModel):
    """What a component comes from; one piece may join several features' errors."""

    id: str
    features: list[str]
    cause: Literal["reading", "building", "negligible", "unclear"]


class Error(BaseModel):
    feature: str | None  # None: the reading lacks the feature altogether
    decision: Decision
    reading: str
    truth: str
    cue: str
    components: list[str] = Field(min_length=1)


class Judgement(BaseModel):
    components: list[Call]
    errors: list[Error]


class Judged(BaseModel):
    key: str
    judge: str  # model, effort and prompt hash
    judgement: Judgement | None
    raw: str | None = None
    usage: dict | None = None


def read_rows[T: BaseModel](path: Path | str, model: type[T]) -> Iterator[T]:
    if Path(path).exists():
        with open(path) as rows:
            yield from (model.model_validate_json(line) for line in rows if line.strip())


def append_row(path: Path | str, row: BaseModel) -> None:
    with open(path, "a") as rows:
        rows.write(row.model_dump_json() + "\n")
