"""The decision trees run on a fresh reader's observations: Luna, shown only the boxed
views and no word of the reading, answers the trees' questions for every feature; the
trees conclude, and a flag goes up where a conclusion differs from the reading's answer.

    python -m experiments.rag.observe PAIRS.jsonl JUDGED.jsonl TREES.json ORACLE.jsonl OUT.jsonl [WORKERS] [zoom]

A reading's part takes the trees of the other fold. With zoom, each item's boxes
also come cut out and enlarged. Flags are scored against the judge with every tree
on every feature, and with the trees Sol says apply.
"""

import base64
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from string import Template
from threading import Lock

import numpy as np
import yaml
from langchain_core.messages import HumanMessage

from experiments.rag.oracle import trees_for
from experiments.rag.reread import CLAIMS, _ask, boxed_views, luna
from experiments.rag.schema import Judged, Pair, read_rows
from experiments.rag.trees import HIDDEN, Tree
from zeroshot.pipeline.retrieval.mistakes import QUESTIONS

PROMPT = Path(__file__).with_name("observe.md").read_text()
USED = ["through or blind", "solid or hollow", "edge"]
TRIES = 3
ENLARGED = 800  # an enlarged box's longer side in pixels, at most three times its size
ZOOM = " After the views, each item's boxes are cut out with a margin and enlarged."


def _procedures(trees: dict[str, Tree]) -> str:
    return "\n".join(f"  {q}/{node.id}: {node.ask} [{' / '.join(node.answers)}]"
                     for q in USED for node in trees[q].nodes if node.id != HIDDEN)


def enlarged(pair: Pair, interpretation: dict, found: list[dict]) -> list[tuple[str, bytes]]:
    """Per item, its evidence boxes cut out of their views with a margin, enlarged and set side by side."""
    import io

    from PIL import Image, ImageDraw, ImageFont

    workspace = Path(pair.drawing).parents[1]
    views = {view["name"]: view for view in interpretation["views"]}
    features = {feature["name"]: feature for feature in interpretation["features"]}
    font = ImageFont.load_default(size=28)
    sheets = []
    for item in found:
        tiles = []
        for region in features[item["feature"]]["evidence"]:
            if not region.get("box_px") or region["view"] not in views:
                continue
            image = Image.open(workspace / views[region["view"]]["file"].removeprefix("/work/")).convert("RGB")
            x0, y0, x1, y1 = region["box_px"]
            mx, my = max(20, (x1 - x0) // 5), max(20, (y1 - y0) // 5)
            crop = image.crop((max(0, x0 - mx), max(0, y0 - my), min(image.width, x1 + mx), min(image.height, y1 + my)))
            scale = min(3.0, max(1.0, ENLARGED / max(crop.size)))
            crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (crop.width, crop.height + 40), "white")
            tile.paste(crop, (0, 40))
            ImageDraw.Draw(tile).text((4, 4), f"{region['view']} ({views[region['view']]['role']})", fill="black", font=font)
            tiles.append(tile)
        if not tiles:
            continue
        sheet = Image.new("RGB", (sum(t.width for t in tiles) + 20 * (len(tiles) - 1), max(t.height for t in tiles)), "white")
        x = 0
        for tile in tiles:
            sheet.paste(tile, (x, 0))
            x += tile.width + 20
        buffer = io.BytesIO()
        sheet.save(buffer, "PNG")
        sheets.append((f"item {item['item']}, enlarged", buffer.getvalue()))
    return sheets


def observe(model, pair: Pair, trees: dict[str, Tree], drawing: dict, interpretation: dict, zoom: bool = False) -> tuple:
    features = interpretation["features"]
    found = [{"item": n, "feature": f["name"]} for n, f in enumerate(features, 1)]
    text = Template(PROMPT).substitute(
        projection=drawing["projection_method"].replace("_", " "),
        hidden=" Hidden lines are not drawn." if drawing.get("hidden_lines_omitted") else "",
        procedures=_procedures(trees), items=", ".join(str(item["item"]) for item in found), zoom=ZOOM if zoom else "")
    content = [{"type": "text", "text": text}]
    images = boxed_views(pair, interpretation, found) + (enlarged(pair, interpretation, found) if zoom else [])
    for name, data in images:
        url = "data:image/png;base64," + base64.b64encode(data).decode()
        content += [{"type": "text", "text": name}, {"type": "image_url", "image_url": {"url": url}}]
    wanted = {f"{q}/{node.id}": node for q in USED for node in trees[q].nodes if node.id != HIDDEN}
    messages, usage, start = [HumanMessage(content=content)], [], time.monotonic()
    for _ in range(TRIES):
        answer = model.invoke(messages)
        usage.append(answer.usage_metadata)
        raw = answer.text
        try:
            got = {int(row["item"]): row["observations"] for row in json.loads(raw[raw.index("{"):raw.rindex("}") + 1])["answers"]}
        except (ValueError, KeyError, TypeError) as error:
            problems = [f"unreadable answer: {error}"]
        else:
            problems = [f"item {item['item']}: {key} must be one of {list(node.answers)}" for item in found
                        for key, node in wanted.items() if got.get(item["item"], {}).get(key) not in node.answers]
            if not problems:
                return got, usage, time.monotonic() - start
        messages += [answer, HumanMessage(content="Fix your answer and give the whole JSON again: " + "; ".join(problems[:40]))]
    return None, usage, time.monotonic() - start


def run(models: dict, pair: Pair, trees: dict, zoom: bool = False) -> dict:
    interpretation = json.loads(Path(pair.interpretation).read_text())
    drawing = yaml.safe_load((Path(pair.drawing).parents[2] / ".hydra/config.yaml").read_text())["sample"]["drawing"]
    mine = trees_for(pair.part, trees)
    row = {"key": pair.key, "hidden_drawn": not drawing.get("hidden_lines_omitted"),
           "features": [f["name"] for f in interpretation["features"]]}
    row["observations"], row["observe_usage"], row["observe_seconds"] = observe(models["observe"], pair, mine, drawing, interpretation, zoom)
    found = [{"item": n, "feature": f["name"], "questions": USED} for n, f in enumerate(interpretation["features"], 1)]
    text = Template(CLAIMS).substitute(
        questions="\n".join(f"- {q} ({', '.join(QUESTIONS[q][0])}): {QUESTIONS[q][1]}" for q in USED),
        items="\n".join(f"{n}. {json.dumps({k: f.get(k) for k in ('description', 'parameters')})}: {', '.join(USED)}"
                        for n, f in enumerate(interpretation["features"], 1)))
    row["claims"], row["claims_usage"], row["claims_seconds"] = _ask(models["claims"], [{"type": "text", "text": text}], found)
    return row


def report(rows: list[dict], judged: dict[str, Judged], oracle: dict[str, dict], trees: dict) -> None:
    for routing in ("every tree on every feature", "trees Sol says apply"):
        print(f"\n{routing}:")
        for q in [*USED, "all"]:
            flags = hits = errors = checks = decided = 0
            for row in rows:
                if not row.get("observations") or not row.get("claims"):
                    continue
                mine = trees_for(row["key"].split("/")[2], trees)
                wrong = {(e.feature, e.decision.question): e.decision for e in judged[row["key"]].judgement.errors if e.feature}
                applies = {(a["item"], a["property"]): a["applies"] == "yes" for a in oracle[row["key"]]["answers"]}
                for n, name in enumerate(row["features"], 1):
                    for question in (USED if q == "all" else [q]):
                        errors += (name, question) in wrong
                        if routing != "every tree on every feature" and not applies.get((n, question)):
                            continue
                        checks += 1
                        seen = {key.split("/", 1)[1]: v for key, v in row["observations"].get(n, row["observations"].get(str(n), {})).items()
                                if key.startswith(question + "/")}
                        conclusion = mine[question].conclude(seen, row["hidden_drawn"])
                        claim = row["claims"][f"{n}/{question}"]["answer"]
                        decided += conclusion != "undecided"
                        if conclusion != "undecided" and claim not in ("unstated", "mixed") and conclusion != claim:
                            flags += 1
                            decision = wrong.get((name, question))
                            hits += decision is not None and decision.truth == conclusion
            print(f"  {q}: {checks} checks, {decided} decided, {flags} flags, {hits} hit "
                  f"(precision {hits / max(flags, 1):.0%}, recall {hits / max(errors, 1):.0%} of {errors} errors)")
    for call in ("observe", "claims"):
        done = [row for row in rows if row.get(f"{call}_usage")]
        tokens = [[sum(u[k] for u in row[f"{call}_usage"]) for row in done] for k in ("input_tokens", "output_tokens")]
        print(f"{call}: {len(done)} readings, per reading median {np.median([r[f'{call}_seconds'] for r in done]):.0f} s, "
              f"{np.median(tokens[0]):,.0f} input and {np.median(tokens[1]):,.0f} output tokens")


if __name__ == "__main__":
    pairs = [pair for pair in read_rows(sys.argv[1], Pair) if pair.which == "final"]
    judged = {row.key: row for row in read_rows(sys.argv[2], Judged) if row.judgement is not None}
    trees = json.loads(Path(sys.argv[3]).read_text())
    oracle = {row["key"]: row for row in map(json.loads, open(sys.argv[4])) if row.get("answers")}
    out = Path(sys.argv[5])
    done = {json.loads(line)["key"] for line in open(out)} if out.exists() else set()
    zoom = len(sys.argv) > 7 and sys.argv[7] == "zoom"
    models, lock = {"observe": luna("high", images=20), "claims": luna("low")}, Lock()

    def work(pair: Pair) -> None:
        try:
            row = run(models, pair, trees, zoom)
        except Exception as error:  # one failing reading must not stop the rest
            print(pair.key, "FAILED", type(error).__name__, str(error)[:300], flush=True)
            return
        with lock, open(out, "a") as rows:
            rows.write(json.dumps(row) + "\n")
        print(pair.key, "ok" if row["observations"] and row["claims"] else "unanswered", flush=True)

    with ThreadPoolExecutor(int(sys.argv[6]) if len(sys.argv) > 6 else 8) as pool:
        list(pool.map(work, [pair for pair in pairs if pair.key in oracle and pair.key not in done]))
    report([json.loads(line) for line in open(out)], judged, oracle, trees)
