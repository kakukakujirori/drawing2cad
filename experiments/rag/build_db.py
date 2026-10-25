"""The mistake DB: each reading error of a judged pair that names a feature,
keyed by that feature's description.

    python -m experiments.rag.build_db PAIRS.jsonl COMPARISONS.jsonl JUDGED.jsonl OUT.jsonl

A first reading's error persists when the same run's final reading still has an
error on a feature of that name. Refuses any test part.
"""

import json
import sys
from pathlib import Path

from experiments.rag.samples import test_ids
from experiments.rag.schema import Comparison, Judged, Pair, read_rows
from zeroshot.pipeline.retrieval.mistakes import Mistake


def mistakes(pairs: dict[str, Pair], comparisons: dict[str, Comparison], judged: list[Judged]) -> list[Mistake]:
    wrong = {
        row.key: {error.feature for error in row.judgement.errors if error.feature is not None}
        for row in judged if row.judgement is not None
    }
    found = []
    for row in judged:
        if row.judgement is None:
            continue
        pair = pairs[row.key]
        final = wrong.get(row.key.removesuffix(pair.which) + "final", set())
        descriptions = {f["name"]: f["description"] for f in json.loads(Path(pair.interpretation).read_text())["features"]}
        shares = {c.id: c.share for c in comparisons[row.key].components}
        for error in row.judgement.errors:
            if error.feature is None:
                continue  # no description to find it by
            found.append(Mistake(
                part=pair.part, dataset=pair.dataset, model=pair.model, which=pair.which,
                persisted=pair.which == "final" or error.feature in final, iou=comparisons[row.key].iou,
                share=sum(shares.get(piece, 0.0) for piece in error.components),
                feature=error.feature, description=descriptions[error.feature], decision=error.decision,
                reading=error.reading, truth=error.truth, cue=error.cue,
            ))
    return found


if __name__ == "__main__":
    pairs = {pair.key: pair for pair in read_rows(sys.argv[1], Pair)}
    comparisons = {row.key: row for row in read_rows(sys.argv[2], Comparison)}
    rows = mistakes(pairs, comparisons, list(read_rows(sys.argv[3], Judged)))
    if leaked := {row.part for row in rows} & test_ids():
        sys.exit(f"test parts in the DB: {sorted(leaked)}")
    Path(sys.argv[4]).write_text("".join(row.model_dump_json() + "\n" for row in rows))
    print(len(rows), "mistakes from", len({row.part for row in rows}), "parts")
