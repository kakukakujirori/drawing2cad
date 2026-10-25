"""Can the judge be trusted? The pilot's pass marks, fixed before the data came in:
- on pairs with IoU >= 0.99, at most one in ten has a reading error;
- judging the same pairs twice, at least 80% of features get the same verdict.

    python -m experiments.rag.qa PAIRS.jsonl COMPARISONS.jsonl JUDGED.jsonl [REPEAT.jsonl]

Also prints each reading error with its pictures, for a reading by eye.
"""

import json
import sys
from collections import Counter
from pathlib import Path

from experiments.rag.schema import Comparison, Judged, Pair, read_rows

NEAR_EXACT = 0.99
MAX_FALSE_SHARE = 0.1
MIN_AGREEMENT = 0.8


def wrong_features(row: Judged) -> set[str]:
    return {error.feature for error in row.judgement.errors if error.feature is not None}


def agreement(first: dict[str, Judged], second: dict[str, Judged], pairs: dict[str, Pair]) -> tuple[int, int]:
    """Features with the same verdict in both judgements, and features judged twice."""
    same = total = 0
    for key in first.keys() & second.keys():
        names = {f["name"] for f in json.loads(Path(pairs[key].interpretation).read_text())["features"]}
        a, b = wrong_features(first[key]), wrong_features(second[key])
        total += len(names)
        same += len(names) - len((a ^ b) & names)
    return same, total


def questions(row: Judged) -> dict[str | None, set[tuple]]:
    found: dict[str | None, set[tuple]] = {}
    for error in row.judgement.errors:
        found.setdefault(error.feature, set()).add((error.decision.question, error.decision.read, error.decision.truth))
    return found


def question_agreement(first: dict[str, Judged], second: dict[str, Judged]) -> tuple[int, int, int]:
    """Of the features both judgements find wrong: how many get the same questions,
    the same answers too, and how many there are."""
    questions_same = answers_same = total = 0
    for key in first.keys() & second.keys():
        a, b = questions(first[key]), questions(second[key])
        for feature in (a.keys() & b.keys()) - {None}:
            total += 1
            questions_same += {q for q, _, _ in a[feature]} == {q for q, _, _ in b[feature]}
            answers_same += a[feature] == b[feature]
    return questions_same, answers_same, total


if __name__ == "__main__":
    pairs = {row.key: row for row in read_rows(sys.argv[1], Pair)}
    comparisons = {row.key: row for row in read_rows(sys.argv[2], Comparison)}
    judged = {row.key: row for row in read_rows(sys.argv[3], Judged) if row.judgement is not None}
    print(f"judged {len(judged)}, unparsed {sum(1 for _ in read_rows(sys.argv[3], Judged)) - len(judged)}")
    # Pairs without pieces cannot have errors; only those with pieces test the judge.
    exact = [key for key in judged if comparisons[key].iou >= NEAR_EXACT and comparisons[key].components]
    false = [key for key in exact if judged[key].judgement.errors]
    share = len(false) / len(exact) if exact else float("nan")
    print(f"IoU >= {NEAR_EXACT} with pieces: {len(false)}/{len(exact)} with a reading error "
          f"({'pass' if share <= MAX_FALSE_SHARE else 'FAIL'}): {false}")
    if len(sys.argv) > 4:
        repeat = {row.key: row for row in read_rows(sys.argv[4], Judged) if row.judgement is not None}
        asked = {key: row for key, row in judged.items() if comparisons[key].components}
        same, total = agreement(asked, repeat, pairs)
        print(f"repeat: {same}/{total} features agree ({'pass' if same >= MIN_AGREEMENT * total else 'FAIL'})")
        questions_same, answers_same, wrong = question_agreement(asked, repeat)
        print(f"repeat, features wrong in both: {questions_same}/{wrong} same questions, {answers_same}/{wrong} same answers too")
    causes = Counter(call.cause for row in judged.values() for call in row.judgement.components)
    print("piece causes", dict(causes))
    print("questions", dict(Counter(e.decision.question for row in judged.values() for e in row.judgement.errors).most_common()))
    for key, row in sorted(judged.items()):
        if not row.judgement.errors:
            continue
        print(f"\n## {key}  IoU {comparisons[key].iou:.3f}\n{pairs[key].drawing}\n{comparisons[key].images['diff']}")
        shares = {c.id: c.share for c in comparisons[key].components}
        for error in row.judgement.errors:
            pieces = ", ".join(f"{p} {shares.get(p, 0):.1%}" for p in error.components)
            d = error.decision
            print(f"- {error.feature} [{d.question}: {d.read} -> {d.truth}] ({pieces})\n  read: {error.reading}\n  truth: {error.truth}\n  cue: {error.cue}")
