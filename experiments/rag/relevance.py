"""Do a method's warnings ask the question an error got wrong? Leave one part out.

    python -m experiments.rag.relevance DB.jsonl [K]

Each mistake that persisted to its run's final reading is a query. A method shows
K warnings from other parts' mistakes, and one hits when its decision asks the
query's question ("other" never hits). Methods that rank the DB's mistakes are
compared with checklists that show the K questions most often wrong, overall or
on features of the query's kind, and with warnings drawn at random.
"""

import re
import sys
from collections import Counter
from math import comb

import numpy as np

from zeroshot.pipeline.retrieval.mistakes import K, Mistake, embed

# Numbers, model axes and planes, and directions: the frame, not the shape.
FRAME = re.compile(r"[-+]?\d+(?:\.\d+)?|\b[+-]?[XYZ]\b|\b(?:XY|XZ|YZ)\b|[+-](?=[XYZ]\b)|\bR(?=\d)")
# Feature kinds, with the words a reader uses for each.
KINDS = {
    "hole": "hole holes bore bores opening openings passage passages through-hole",
    "counterbore": "counterbore counterbores countersink countersinks stepped step steps shoulder",
    "boss": "boss bosses projection projections collar sleeve pin pins stud",
    "slot": "slot slots notch notches groove grooves channel channels keyway",
    "pocket": "pocket pockets recess recesses cavity cavities relief reliefs",
    "rib": "rib ribs web webs gusset gussets",
    "plate": "plate plates flange flanges tab tabs lug lugs ear ears",
    "shell": "shell shells wall walls hollow enclosure",
    "round": "fillet fillets round rounded blend blends",
    "chamfer": "chamfer chamfers bevel bevels beveled",
    "taper": "taper tapered draft flare flared cone conical",
}
_WORD = {word: kind for kind, words in KINDS.items() for word in words.split()}


def kinds(text: str) -> set[str]:
    return {_WORD[word] for word in re.findall(r"[a-z-]+", text.lower()) if word in _WORD}


def stripped(text: str) -> str:
    return re.sub(r"\s+", " ", FRAME.sub(" ", text))


def question(mistake: Mistake) -> str:
    return mistake.decision.question


def answers(mistake: Mistake) -> tuple:
    return question(mistake), mistake.decision.read, mistake.decision.truth


def rankings(db: list[Mistake]) -> dict[str, np.ndarray]:
    """Per method, a query-by-mistake similarity over the whole DB."""
    raw = embed([m.description for m in db])
    clean = embed([stripped(m.description) for m in db])
    names = [kinds(m.feature.replace("_", " ")) for m in db]
    texts = [kinds(m.feature.replace("_", " ") + " " + m.description) for m in db]

    def jaccard(sets):
        return np.array([[len(a & b) / len(a | b) if a | b else 0.0 for b in sets] for a in sets])

    return {"embedding": raw @ raw.T, "embedding without numbers and axes": clean @ clean.T,
            "kind words in the name": jaccard(names), "kind words in name and description": jaccard(texts)}


def checklist(db: list[Mistake], query: int, others: list[int], k: int, by_kind: bool,
              skip: tuple[str, ...] = ("other",)) -> list[str]:
    """The k questions most often wrong on other parts, on features of the query's kind if
    asked, other than the skipped ones."""
    pool = others
    if by_kind and (mine := kinds(db[query].feature.replace("_", " "))):
        pool = [j for j in others if kinds(db[j].feature.replace("_", " ")) & mine] or others
    counts = Counter(question(db[j]) for j in pool if question(db[j]) not in skip)
    return [q for q, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:k]]


def evaluate(db: list[Mistake], k: int = K, skip: tuple[str, ...] = ("other",),
             min_share: float = 0.0) -> dict[str, dict[str, float]]:
    """Per method: the share of queries with a hit among its k warnings, the share of
    warnings that hit, and the share of queries with a warning of the same answers too.
    Mistakes of the skipped questions, or below min_share of the part, are no queries,
    and checklists do not show those questions."""
    queries = [n for n, m in enumerate(db)
               if m.persisted and question(m) not in skip and (m.share or 0.0) >= min_share]
    others = {n: [j for j, m in enumerate(db) if m.part != db[n].part] for n in queries}
    results = {}
    for name, similarity in rankings(db).items():
        hits = []
        for n in queries:
            top = sorted(others[n], key=lambda j: (-similarity[n, j], j))[:k]
            hits.append(([question(db[j]) == question(db[n]) for j in top], any(answers(db[j]) == answers(db[n]) for j in top)))
        results[name] = {"hit": np.mean([any(h) for h, _ in hits]), "precision": np.mean([np.mean(h) for h, _ in hits]),
                         "same answers": np.mean([a for _, a in hits])}
    for name, by_kind in (("checklist of the commonest questions", False), ("checklist for the feature's kind", True)):
        shown = [checklist(db, n, others[n], k, by_kind, skip) for n in queries]
        hit = [question(db[n]) in s for n, s in zip(queries, shown, strict=True)]
        results[name] = {"hit": np.mean(hit), "precision": np.mean(hit) / k, "same answers": float("nan")}
    # Drawn at random from other parts: the chance that k of them miss every mistake of the query's question.
    random_hit, random_precision = [], []
    for n in queries:
        size, same = len(others[n]), sum(question(db[j]) == question(db[n]) for j in others[n])
        random_hit.append(1 - comb(size - same, k) / comb(size, k))
        random_precision.append(same / size)
    results["random"] = {"hit": np.mean(random_hit), "precision": np.mean(random_precision), "same answers": float("nan")}
    return results


if __name__ == "__main__":
    db = [m for m in (Mistake.model_validate_json(line) for line in open(sys.argv[1]) if line.strip()) if m.decision]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else K
    persisted = [m for m in db if m.persisted]
    print(f"{len(db)} mistakes from {len({m.part for m in db})} parts; {len(persisted)} persisted")
    print("questions of persisted mistakes:", dict(Counter(question(m) for m in persisted).most_common()))
    for subset, skip, min_share in (("all questions", ("other",), 0.0), ("profile left out", ("other", "profile"), 0.0),
                                    ("profile left out, errors of 1% of the part or more", ("other", "profile"), 0.01)):
        queries = sum(m.persisted and question(m) not in skip and (m.share or 0.0) >= min_share for m in db)
        print(f"\n{subset} ({queries} queries), K={k}: queries with a hit / warnings that hit / queries with the same answers too")
        for name, result in evaluate(db, k, skip, min_share).items():
            print(f"  {name}: {result['hit']:.0%} / {result['precision']:.0%} / {result['same answers']:.0%}")
