"""Reference labels for the decision trees: Sol, shown the real part, says for every
feature of a final reading which tree questions apply, the real answer, and its
own answers to the trees' observation questions.

    python -m experiments.rag.oracle PAIRS.jsonl COMPARISONS.jsonl JUDGED.jsonl TREES.json OUT.jsonl [WORKERS]

A reading's part takes the trees of the other fold. Readings already in OUT are
skipped. Then each tree is scored on Sol's observations: how often it decides, and
whether it reaches the real answer, above all where the judge found the reading wrong.
"""

import base64
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from string import Template
from threading import Lock

import yaml
from langchain_core.messages import HumanMessage

from experiments.rag.judge import sol
from experiments.rag.reread import boxed_views
from experiments.rag.schema import Comparison, Judged, Pair, read_rows
from experiments.rag.trees import HIDDEN, TREE_QUESTIONS, Tree, fold
from zeroshot.pipeline.retrieval.mistakes import QUESTIONS

PROMPT = Path(__file__).with_name("oracle.md").read_text()
EFFORT = "high"
TRIES = 3


def trees_for(part: str, trees: dict) -> dict[str, Tree]:
    return {q: Tree.model_validate(tree) for q, tree in trees[str(1 - fold(part))].items()}


def _procedure(tree: Tree) -> str:
    asks = [f"  {node.id}: {node.ask} [{' / '.join(node.answers)}]" for node in tree.nodes if node.id != HIDDEN]
    answers = ", ".join(QUESTIONS[tree.question][0])
    return f"- {tree.question} ({answers}): {QUESTIONS[tree.question][1]}\n" + "\n".join(asks)


def problems(rows: list[dict], items: int, trees: dict[str, Tree]) -> list[str]:
    got = {(row.get("item"), row.get("property")): row for row in rows}
    found = []
    for item in range(1, items + 1):
        for q, tree in trees.items():
            row = got.get((item, q))
            if row is None:
                found.append(f"item {item}: {q} is missing")
            elif row.get("applies") not in ("yes", "no"):
                found.append(f"item {item}: {q} applies must be yes or no")
            elif row["applies"] == "yes":
                if row.get("truth") not in QUESTIONS[q][0]:
                    found.append(f"item {item}: {q} truth must be one of {list(QUESTIONS[q][0])}")
                seen = row.get("observations") or {}
                for node in tree.nodes:
                    if node.id != HIDDEN and seen.get(node.id) not in node.answers:
                        found.append(f"item {item}: {q} {node.id} must be one of {list(node.answers)}")
    return found


def label(model, pair: Pair, comparison: Comparison, trees: dict) -> dict:
    interpretation = json.loads(Path(pair.interpretation).read_text())
    features = interpretation["features"]
    found = [{"item": n, "feature": f["name"]} for n, f in enumerate(features, 1)]
    mine = trees_for(pair.part, trees)
    drawing = yaml.safe_load((Path(pair.drawing).parents[2] / ".hydra/config.yaml").read_text())["sample"]["drawing"]
    text = Template(PROMPT).substitute(
        hidden=" Hidden lines are not drawn in this drawing." if drawing.get("hidden_lines_omitted") else "",
        features="\n".join(f"{n}. {f['name']}: {f['description']}" for n, f in enumerate(features, 1)),
        procedures="\n".join(_procedure(tree) for tree in mine.values()))
    content = [{"type": "text", "text": text}]
    images = [*boxed_views(pair, interpretation, found), ("gt.png (the real part)", Path(comparison.images["gt"]).read_bytes())]
    for name, data in images:
        url = "data:image/png;base64," + base64.b64encode(data).decode()
        content += [{"type": "text", "text": name}, {"type": "image_url", "image_url": {"url": url}}]
    messages, usage = [HumanMessage(content=content)], []
    for _ in range(TRIES):
        answer = model.invoke(messages)
        usage.append(answer.usage_metadata)
        raw = answer.text
        try:
            rows = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])["answers"]
        except (ValueError, KeyError, TypeError) as error:
            found_problems = [f"unreadable answer: {error}"]
        else:
            if not (found_problems := problems(rows, len(features), mine)):
                return {"key": pair.key, "fold": 1 - fold(pair.part), "hidden_drawn": not drawing.get("hidden_lines_omitted"),
                        "features": [f["name"] for f in features], "answers": rows, "usage": usage}
        messages += [answer, HumanMessage(content="Fix your answer and give the whole JSON again: " + "; ".join(found_problems[:40]))]
    return {"key": pair.key, "answers": None, "raw": raw, "usage": usage}


def report(rows: list[dict], judged: dict[str, Judged], trees: dict) -> None:
    for q in TREE_QUESTIONS:
        applied = decided = right = 0
        wrong_readings = caught = agree_judge = 0
        for row in rows:
            if not row.get("answers"):
                continue
            tree = trees_for(row["key"].split("/")[2], trees)[q]
            errors = {e.feature: e.decision for e in judged[row["key"]].judgement.errors if e.decision.question == q}
            for answer in row["answers"]:
                if answer["property"] != q or answer["applies"] != "yes":
                    continue
                applied += 1
                conclusion = tree.conclude(answer["observations"], row["hidden_drawn"])
                decided += conclusion != "undecided"
                right += conclusion == answer["truth"]
                if (decision := errors.get(row["features"][answer["item"] - 1])) is not None:
                    wrong_readings += 1
                    caught += conclusion == decision.truth
                    agree_judge += answer["truth"] == decision.truth
        print(f"{q}: applies {applied}, decided {decided}, right {right}, wrong {decided - right}; "
              f"readings the judge found wrong {wrong_readings}: tree reaches the judge's answer {caught}, "
              f"Sol's truth agrees with the judge {agree_judge}")


if __name__ == "__main__":
    pairs = [pair for pair in read_rows(sys.argv[1], Pair) if pair.which == "final"]
    comparisons = {row.key: row for row in read_rows(sys.argv[2], Comparison)}
    judged = {row.key: row for row in read_rows(sys.argv[3], Judged) if row.judgement is not None}
    trees = json.loads(Path(sys.argv[4]).read_text())
    out = Path(sys.argv[5])
    done = {json.loads(line)["key"] for line in open(out)} if out.exists() else set()
    model, lock = sol(EFFORT), Lock()

    def work(pair: Pair) -> None:
        try:
            row = label(model, pair, comparisons[pair.key], trees)
        except Exception as error:  # one failing reading must not stop the rest
            print(pair.key, "FAILED", type(error).__name__, str(error)[:300], flush=True)
            return
        with lock, open(out, "a") as rows:
            rows.write(json.dumps(row) + "\n")
        print(pair.key, "ok" if row["answers"] else "unanswered", flush=True)

    jobs = [pair for pair in pairs if pair.key in comparisons and pair.key not in done]
    with ThreadPoolExecutor(int(sys.argv[6]) if len(sys.argv) > 6 else 8) as pool:
        list(pool.map(work, jobs))
    report([json.loads(line) for line in open(out)], judged, trees)
