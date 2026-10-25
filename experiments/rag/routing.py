"""Which decision trees to run on which feature, judged against the reference labels.

    python -m experiments.rag.routing PAIRS.jsonl JUDGED.jsonl ORACLE.jsonl LLM_ROUTES.jsonl

A method routes a (feature, question) pair to its tree. It is scored against Sol's
"applies" labels, and by the share of the judge's errors on these questions that it
routes. LLM_ROUTES caches Luna's routing calls; readings missing from it are asked.
"""

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from string import Template

import numpy as np
from langchain_core.messages import HumanMessage

from experiments.rag.reread import luna
from experiments.rag.schema import Judged, Pair, read_rows
from experiments.rag.trees import TREE_QUESTIONS
from zeroshot.pipeline.retrieval.mistakes import QUESTIONS, embed

PROMPT = Path(__file__).with_name("route.md").read_text()
# Words in a feature's name or description that raise each question.
WORDS = {
    "through or blind": "hole holes bore bores opening openings passage passages port ports slot slots notch notches groove "
                        "grooves channel channels pocket pockets recess recesses cavity cavities counterbore counterbores "
                        "window windows cutout cutouts cut cuts",
    "solid or hollow": "body housing block shell shells wall walls hollow enclosure sleeve tube boss bosses collar hub",
    "open or closed": "cavity cavities recess recesses pocket pockets slot slots notch notches groove grooves channel "
                      "channels opening openings cutout cutouts window windows",
    "edge": "hole holes bore bores opening openings counterbore counterbores countersink chamfer chamfers bevel bevelled "
            "beveled fillet fillets round rounded mouth mouths rim edge edges",
}
_WORDS = {q: set(words.split()) for q, words in WORDS.items()}
_ANY = set().union(*_WORDS.values())


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def features(pairs: dict[str, Pair], oracle: list[dict]) -> list[dict]:
    """Every feature of the labelled final readings, with Sol's labels."""
    found = []
    for row in oracle:
        if not row.get("answers"):
            continue
        described = {f["name"]: f for f in json.loads(Path(pairs[row["key"]].interpretation).read_text())["features"]}
        applies = {(a["item"], a["property"]): a["applies"] == "yes" for a in row["answers"]}
        for n, name in enumerate(row["features"], 1):
            found.append({"key": row["key"], "part": row["key"].split("/")[2], "item": n, "name": name,
                          "description": described[name]["description"],
                          "applies": {q: applies[n, q] for q in TREE_QUESTIONS}})
    return found


def keyword(feats: list[dict], source: str) -> dict:
    routes = {}
    for f in feats:
        name, text = words(f["name"].replace("_", " ")), words(f["description"])
        for q in TREE_QUESTIONS:
            hit_name, hit_text = bool(name & _WORDS[q]), bool(text & _WORDS[q])
            if source == "name":
                routes[id(f), q] = hit_name
            elif source == "name, else description":
                routes[id(f), q] = hit_name if name & _ANY else hit_text
            else:
                routes[id(f), q] = hit_name or hit_text
    return routes


def zero_shot(feats: list[dict], threshold: float) -> dict:
    vectors = embed([f["name"].replace("_", " ") + ": " + f["description"] for f in feats])
    targets = embed([QUESTIONS[q][1] for q in TREE_QUESTIONS])
    similarity = vectors @ targets.T
    return {(id(f), q): bool(similarity[n, j] >= threshold) for n, f in enumerate(feats) for j, q in enumerate(TREE_QUESTIONS)}


def supervised(feats: list[dict], kind: str) -> dict:
    """Trained on Sol's labels of the other parts, five folds by part."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.neighbors import KNeighborsClassifier

    x = embed([f["name"].replace("_", " ") + ": " + f["description"] for f in feats])
    groups = [f["part"] for f in feats]
    routes = {}
    for q in TREE_QUESTIONS:
        y = np.array([f["applies"][q] for f in feats])
        predicted = np.zeros(len(feats), bool)
        for train, test in GroupKFold(5).split(x, y, groups):
            if len(set(y[train])) < 2:
                predicted[test] = y[train][0]
                continue
            model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000) if kind == "logistic" \
                else KNeighborsClassifier(5, metric="cosine")
            predicted[test] = model.fit(x[train], y[train]).predict(x[test])
        routes.update({(id(f), q): bool(predicted[n]) for n, f in enumerate(feats)})
    return routes


def llm(pairs: dict[str, Pair], feats: list[dict], cache: Path) -> dict:
    done = {json.loads(line)["key"]: json.loads(line) for line in open(cache)} if cache.exists() else {}
    keys = sorted({f["key"] for f in feats} - done.keys())
    model = luna("low")
    questions = "\n".join(f"- {q}: {QUESTIONS[q][1]}" for q in TREE_QUESTIONS)

    def ask(key: str) -> dict:
        described = json.loads(Path(pairs[key].interpretation).read_text())["features"]
        text = Template(PROMPT).substitute(questions=questions, features="\n".join(
            f"{n}. {f['name']}: {f['description']}" for n, f in enumerate(described, 1)))
        raw = model.invoke([HumanMessage(content=text)]).text
        rows = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])["answers"]
        return {"key": key, "routes": {f"{r['item']}/{r['property']}": r.get("applies") == "yes" for r in rows}}

    with ThreadPoolExecutor(8) as pool:
        for row in pool.map(ask, keys):
            done[row["key"]] = row
            with open(cache, "a") as rows:
                rows.write(json.dumps(row) + "\n")
    return {(id(f), q): done[f["key"]]["routes"].get(f"{f['item']}/{q}", False) for f in feats for q in TREE_QUESTIONS}


def score(name: str, routes: dict, feats: list[dict], errors: set) -> None:
    by_question = []
    for q in TREE_QUESTIONS:
        pairs = [(routes[id(f), q], f["applies"][q]) for f in feats]
        tp = sum(r and a for r, a in pairs)
        by_question.append(f"{q} {tp / max(sum(r for r, _ in pairs), 1):.0%}/{tp / max(sum(a for _, a in pairs), 1):.0%}")
    allpairs = [(routes[id(f), q], f["applies"][q]) for f in feats for q in TREE_QUESTIONS]
    tp = sum(r and a for r, a in allpairs)
    routed = sum(r for r, _ in allpairs)
    caught = sum(routes[id(f), q] for f in feats for q in TREE_QUESTIONS if (f["key"], f["name"], q) in errors)
    print(f"  {name}: precision {tp / max(routed, 1):.0%}, recall {tp / sum(a for _, a in allpairs):.0%}, "
          f"{routed / len({f['key'] for f in feats}):.1f} trees per reading, judged errors routed {caught}/{len(errors)}"
          f"\n      per question (precision/recall): " + ", ".join(by_question))


if __name__ == "__main__":
    pairs = {pair.key: pair for pair in read_rows(sys.argv[1], Pair)}
    judged = {row.key: row for row in read_rows(sys.argv[2], Judged) if row.judgement is not None}
    oracle = [json.loads(line) for line in open(sys.argv[3])]
    feats = features(pairs, oracle)
    keys = {f["key"] for f in feats}
    errors = {(key, e.feature, e.decision.question) for key in keys for e in judged[key].judgement.errors
              if e.feature and e.decision.question in TREE_QUESTIONS}
    applicable = sum(f["applies"][q] for f in feats for q in TREE_QUESTIONS)
    print(f"{len(feats)} features in {len(keys)} readings; {applicable} of {len(feats) * len(TREE_QUESTIONS)} "
          f"(feature, question) pairs apply; {len(errors)} judged errors on these questions")
    for source in ("name", "name, else description", "name and description"):
        score(f"keywords in {source}", keyword(feats, source), feats, errors)
    for threshold in (0.3, 0.4, 0.5):
        score(f"embedding similarity to the question >= {threshold}", zero_shot(feats, threshold), feats, errors)
    for kind in ("logistic", "nearest"):
        score(f"embedding, {kind} on Sol's labels", supervised(feats, kind), feats, errors)
    score("Luna (low) asked to route", llm(pairs, feats, Path(sys.argv[4])), feats, errors)
    score("every tree on every feature", {(id(f), q): True for f in feats for q in TREE_QUESTIONS}, feats, errors)
