import json

import numpy as np

from zeroshot.pipeline.retrieval.mistakes import MistakeIndex, MistakeLookup

# Descriptions about holes point one way, about ribs another.
_DIRECTIONS = {"hole": [1.0, 0.0], "rib": [0.0, 1.0]}


def _embed(texts: list[str]) -> np.ndarray:
    return np.array([next(v for word, v in _DIRECTIONS.items() if word in text) for text in texts])


def _db(tmp_path, rows):
    path = tmp_path / "mistakes.jsonl"
    common = dict(dataset="d", model="m", which="final", persisted=True, iou=0.5, feature="f",
                  truth="t", cue="c")
    path.write_text("".join(json.dumps(common | row) + "\n" for row in rows))
    return MistakeIndex(path, embedder=_embed)


def test_a_lookup_returns_near_mistakes_of_other_parts_only(tmp_path) -> None:
    index = _db(tmp_path, [
        dict(part="own", description="a hole", reading="own hole"),
        dict(part="other", description="a blind hole", reading="other hole"),
        dict(part="other", description="a thin rib", reading="other rib"),
    ])
    assert [m.reading for m in index.lookup("a counterbored hole", exclude_part="own")] == ["other hole"]


def test_each_mistake_goes_once_to_the_closest_new_feature(tmp_path) -> None:
    index = _db(tmp_path, [dict(part="other", description="a hole", reading="r")])
    lookup = MistakeLookup(index, "own", "[Past]")
    first = lookup.report([("sem_rib", "a rib"), ("sem_bore", "a through hole")])
    assert first is not None and first.startswith("[Past]") and "sem_bore\n1. Read as: r" in first
    assert "sem_rib" not in first
    assert lookup.report([("sem_bore", "a through hole")]) is None
    assert lookup.report([("sem_bore", "a blind hole")]) is None  # already shown


def test_the_focused_style_quotes_the_description_and_shows_one(tmp_path) -> None:
    index = _db(tmp_path, [dict(part="other", description="a hole", reading=f"r{n}") for n in range(2)])
    lookup = MistakeLookup(index, "own", "$name: you wrote $description; it read $reading", "focused")
    assert lookup.report([("sem_bore", "a through hole")]) == "sem_bore: you wrote a through hole; it read r0"
    assert lookup.first


def test_a_decision_takes_two_answers_of_its_question_or_none() -> None:
    import pytest

    from zeroshot.pipeline.retrieval.mistakes import Decision

    assert Decision(question="through or blind", read="through", truth="blind").truth == "blind"
    assert Decision(question="profile").read is None
    for bad in [dict(question="termination"), dict(question="edge", read="sharp", truth="sharp"),
                dict(question="profile", read="a", truth="b")]:
        with pytest.raises(ValueError):
            Decision(**bad)
