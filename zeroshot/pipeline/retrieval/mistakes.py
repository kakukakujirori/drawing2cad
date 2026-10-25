"""Past reading mistakes on other parts, found by the description of a feature."""

from functools import lru_cache
from pathlib import Path
from string import Template

import numpy as np
from pydantic import BaseModel, model_validator

ENCODER = "Qwen/Qwen3-Embedding-0.6B"
_TASK = (
    "Given a description of a feature read from an engineering drawing, "
    "find descriptions of the same kind of 3D shape"
)
K = 3
MIN_SIMILARITY = 0.6

# The questions a reader answers about a feature, the more specific first: each
# with its answers, if it has a fixed set, and what it asks.
QUESTIONS: dict[str, tuple[tuple[str, ...], str]] = {
    "through or blind": (("through", "blind"), "whether a hole, cut or recess passes through the material or stops inside it"),
    "solid or hollow": (("solid", "hollow"), "whether a body is solid or a shell around a cavity"),
    "open or closed": (("open", "closed"), "whether a cavity or cut opens onto a face or edge, or is enclosed"),
    "edge": (("sharp", "chamfer", "round", "step"), "the edge or mouth of a feature: sharp, bevelled or countersunk, rounded, or stepped like a counterbore"),
    "taper": (("constant", "tapered"), "whether a section stays the same along a direction, or tapers or drafts"),
    "how far": (("shorter", "longer"), "how far a feature reaches along its depth or length, such as whether it reaches another feature"),
    "which face": ((), "which face a feature lies on or opens to"),
    "thickness": (("thin", "thick"), "the thickness or proportions of a wall, plate or projection"),
    "count": (("fewer", "more"), "how many of the feature there are"),
    "merge": (("one", "several"), "whether a feature is one piece or several"),
    "present or absent": (("present", "absent"), "whether a hole, recess, projection, rib or other feature exists at all"),
    "profile": ((), "the outline or section otherwise"),
    "other": ((), "anything else"),
}


class Decision(BaseModel):
    """The question a reading got wrong, with its answer and the part's."""

    question: str
    read: str | None = None
    truth: str | None = None

    @model_validator(mode="after")
    def _answers(self) -> "Decision":
        if self.question not in QUESTIONS:
            raise ValueError(f"question must be one of {list(QUESTIONS)}")
        answers = QUESTIONS[self.question][0]
        if not answers and (self.read, self.truth) != (None, None):
            raise ValueError(f"{self.question!r} takes null read and truth")
        if answers and not (self.read in answers and self.truth in answers and self.read != self.truth):
            raise ValueError(f"{self.question!r} takes two different answers of {list(answers)}")
        return self


class Mistake(BaseModel):
    """One reading error of a judged run, keyed by its feature's description."""

    part: str
    dataset: str
    model: str
    which: str  # the run's first reading or its final one
    persisted: bool  # still wrong in the run's final reading
    iou: float
    share: float | None = None  # of the part's volume, in the pieces the error explains
    feature: str
    description: str
    decision: Decision | None = None  # None in rows judged without one
    reading: str
    truth: str
    cue: str


@lru_cache(maxsize=1)
def _encoder():
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ENCODER, padding_side="left")
    # Its own bfloat16 weights: a run with the DB holds the encoder in memory throughout.
    return tokenizer, AutoModel.from_pretrained(ENCODER).eval()


def embed(texts: list[str], batch: int = 16) -> np.ndarray:
    """Unit vectors; the last token's state, as the encoder is meant to be read."""
    import torch

    tokenizer, model = _encoder()
    out = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch):
            chunk = [f"Instruct: {_TASK}\nQuery: {text}" for text in texts[start:start + batch]]
            tokens = tokenizer(chunk, padding=True, truncation=True, max_length=512, return_tensors="pt")
            last = model(**tokens).last_hidden_state[:, -1].float()
            out.append(torch.nn.functional.normalize(last, dim=1).numpy())
    return np.concatenate(out) if out else np.zeros((0, 0), np.float32)


class MistakeIndex:
    def __init__(self, path: Path | str, embedder=embed) -> None:
        with open(path) as rows:
            self.mistakes = [Mistake.model_validate_json(line) for line in rows if line.strip()]
        self.embed = embedder
        self.vectors = embedder([mistake.description for mistake in self.mistakes])

    def lookup(
        self, description: str, exclude_part: str | None = None, k: int = K
    ) -> list[Mistake]:
        """Up to k mistakes of other parts whose descriptions are close to this one."""
        if not self.mistakes:
            return []
        similarity = self.vectors @ self.embed([description])[0]
        order = [
            n for n in np.argsort(-similarity)
            if similarity[n] >= MIN_SIMILARITY and self.mistakes[n].part != exclude_part
        ]
        return [self.mistakes[n] for n in order[:k]]


class MistakeLookup:
    """Past mistakes for the feature descriptions the coder has not been shown yet.

    Each mistake goes to the one new description closest to it, and is shown
    once per run. `list` shows up to K per feature under one header after the
    verification report; `focused` shows the closest one per feature, quoting
    the coder's description, ahead of the report.
    """

    def __init__(
        self, index: MistakeIndex, exclude_part: str | None, template: str, style: str = "list"
    ) -> None:
        if style not in ("list", "focused"):
            raise ValueError(f"unknown mistake style {style!r}")
        self.index = index
        self.exclude_part = exclude_part
        self.template = template
        self.first = style == "focused"
        self.k = 1 if self.first else K
        self._seen: set[tuple[str, str]] = set()
        self._shown: set[int] = set()

    def report(self, features: list[tuple[str, str]]) -> str | None:
        new = [feature for feature in features if feature not in self._seen]
        self._seen.update(new)
        if not new or not self.index.mistakes:
            return None
        similarity = self.index.embed([description for _, description in new]) @ self.index.vectors.T
        allowed = np.array([
            m.part != self.exclude_part and n not in self._shown for n, m in enumerate(self.index.mistakes)
        ])
        similarity[:, ~allowed] = -np.inf
        similarity[similarity < MIN_SIMILARITY] = -np.inf
        closest = similarity.argmax(axis=0)
        blocks = []
        for f, (name, description) in enumerate(new):
            mine = [n for n in np.argsort(-similarity[f]) if closest[n] == f and np.isfinite(similarity[f, n])][: self.k]
            self._shown.update(mine)
            found = [self.index.mistakes[n] for n in mine]
            if found and self.first:
                blocks += [Template(self.template).substitute(
                    name=name, description=description, reading=m.reading, truth=m.truth, cue=m.cue)
                    for m in found]
            elif found:
                blocks.append(f"{name}\n" + "\n".join(
                    f"{n}. Read as: {m.reading} In fact: {m.truth} Decided by: {m.cue}"
                    for n, m in enumerate(found, 1)))
        if not blocks:
            return None
        return "\n\n".join(blocks if self.first else [self.template, *blocks])
