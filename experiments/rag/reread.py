"""A fresh reader, shown only the drawing and where a feature is, answers the questions
the feature's kind raises; a flag goes up where its answer differs from the reading's.

    python -m experiments.rag.reread PAIRS.jsonl COMPARISONS.jsonl JUDGED.jsonl OUT.jsonl [WORKERS]

Per final reading, one call re-reads the views with each feature's evidence boxed and
no word of the reading, and one text call states the answers the reading's
descriptions imply. OUT keeps both, with each call's tokens and seconds; readings
already in OUT are skipped. The flags are then scored against the judge, beside
flagging every question the kinds raise.
"""

import base64
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from string import Template
from threading import Lock

import numpy as np
import yaml
from hydra.utils import instantiate
from langchain_core.messages import HumanMessage
from omegaconf import OmegaConf

from experiments.rag.relevance import kinds
from experiments.rag.schema import Comparison, Judged, Pair, read_rows
from zeroshot.pipeline.retrieval.mistakes import QUESTIONS

READ = Path(__file__).with_name("reread.md").read_text()
CLAIMS = Path(__file__).with_name("reread_claims.md").read_text()
MODEL_CONFIG = Path("zeroshot/configs/model/gpt6_luna_codex.yaml")
EFFORTS = {"reread": "high", "claims": "low"}
# The questions a kind of feature raises, among those whose answers a view settles
# and that hold for a feature as coarse as a reading's.
RAISES = {
    "hole": ("through or blind", "edge"),
    "counterbore": ("edge",),
    "slot": ("through or blind",),
    "pocket": ("through or blind",),
    "round": ("edge",),
    "chamfer": ("edge",),
}
ASKED = [q for q in QUESTIONS if any(q in raised for raised in RAISES.values())]
COLOURS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#008080", "#9a6324", "#800000"]
TRIES = 3


def luna(effort: str, images: int | None = None):
    config = OmegaConf.load(MODEL_CONFIG).model
    config.reasoning.effort = effort
    if images is not None:
        config.image_history_limit = images
    return instantiate(config)


def _questions() -> str:
    return "\n".join(f"- {q} ({', '.join(QUESTIONS[q][0])}): {QUESTIONS[q][1]}" for q in ASKED)


def items(interpretation: dict) -> list[dict]:
    """The features whose kind raises a question: the kind its name gives, or else its description."""
    found = []
    for feature in interpretation["features"]:
        mine = (kinds(feature["name"].replace("_", " ")) or kinds(feature["description"])) & RAISES.keys()
        if asked := [q for q in ASKED if any(q in RAISES[kind] for kind in mine)]:
            found.append({"item": len(found) + 1, "feature": feature["name"], "kinds": sorted(mine), "questions": asked})
    return found


def boxed_views(pair: Pair, interpretation: dict, found: list[dict]) -> list[tuple[str, bytes]]:
    """Each view an item's evidence lies in, with the item's boxes drawn and numbered."""
    from PIL import Image, ImageDraw, ImageFont

    workspace = Path(pair.drawing).parents[1]
    views = {view["name"]: view for view in interpretation["views"]}
    features = {feature["name"]: feature for feature in interpretation["features"]}
    marks: dict[str, list] = {}
    for item in found:
        for region in features[item["feature"]]["evidence"]:
            if region.get("box_px") and region["view"] in views:
                marks.setdefault(region["view"], []).append((item["item"], region["box_px"]))
    drawn = []
    for name, boxes in marks.items():
        image = Image.open(workspace / views[name]["file"].removeprefix("/work/")).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=max(24, image.width // 35))
        for number, (x0, y0, x1, y1) in boxes:
            colour = COLOURS[(number - 1) % len(COLOURS)]
            draw.rectangle((x0, y0, x1, y1), outline=colour, width=max(2, image.width // 400))
            draw.text((x0 + 4, y0 + 2), str(number), fill=colour, font=font)
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        drawn.append((f"{name} ({views[name]['role']})", buffer.getvalue()))
    return drawn


def _ask(model, content: list[dict], found: list[dict]) -> tuple[dict | None, list, float]:
    """Answers keyed "item/question", each call's usage, and the seconds taken."""
    wanted = {(item["item"], q) for item in found for q in item["questions"]}
    messages, usage, start = [HumanMessage(content=content)], [], time.monotonic()
    for _ in range(TRIES):
        answer = model.invoke(messages)
        usage.append(answer.usage_metadata)
        raw = answer.text
        try:
            got = {(int(row["item"]), row["question"]): row for row in json.loads(raw[raw.index("{"):raw.rindex("}") + 1])["answers"]}
        except (ValueError, KeyError, TypeError) as error:
            problems = [f"unreadable answer: {error}"]
        else:
            problems = [f"item {i}: {q} is missing" for i, q in sorted(wanted - got.keys())]
            problems += [f"item {i}: {q} cannot be {got[i, q].get('answer')!r}" for i, q in sorted(wanted & got.keys())
                         if got[i, q].get("answer") not in (*QUESTIONS[q][0], "mixed", "unstated")]
            if not problems:
                return {f"{i}/{q}": got[i, q] for i, q in sorted(wanted)}, usage, time.monotonic() - start
        messages += [answer, HumanMessage(content="Fix your answer and give the whole JSON again: " + "; ".join(problems))]
    return None, usage, time.monotonic() - start


def reread(pair: Pair, models: dict) -> dict:
    interpretation = json.loads(Path(pair.interpretation).read_text())
    found = items(interpretation)
    row = {"key": pair.key, "items": found}
    if not found:
        return row
    drawing = yaml.safe_load((Path(pair.drawing).parents[2] / ".hydra/config.yaml").read_text())["sample"]["drawing"]
    text = Template(READ).substitute(
        projection=drawing["projection_method"].replace("_", " "),
        hidden=" Hidden lines are not drawn." if drawing.get("hidden_lines_omitted") else "",
        questions=_questions(),
        items="\n".join(f"{item['item']}. {', '.join(item['kinds'])}: {', '.join(item['questions'])}" for item in found))
    content = [{"type": "text", "text": text}]
    for name, data in boxed_views(pair, interpretation, found):
        url = "data:image/png;base64," + base64.b64encode(data).decode()
        content += [{"type": "text", "text": name}, {"type": "image_url", "image_url": {"url": url}}]
    row["reread"], row["reread_usage"], row["reread_seconds"] = _ask(models["reread"], content, found)
    features = {feature["name"]: feature for feature in interpretation["features"]}
    text = Template(CLAIMS).substitute(questions=_questions(), items="\n".join(
        f"{item['item']}. {json.dumps({k: features[item['feature']].get(k) for k in ('description', 'parameters')})}: "
        f"{', '.join(item['questions'])}" for item in found))
    row["claims"], row["claims_usage"], row["claims_seconds"] = _ask(models["claims"], [{"type": "text", "text": text}], found)
    return row


def report(rows: list[dict], judged: dict[str, Judged], comparisons: dict[str, Comparison]) -> None:
    wrong = {}  # (key, feature, question) -> (decision, share of the part)
    for key in {row["key"] for row in rows}:
        shares = {c.id: c.share for c in comparisons[key].components}
        for error in judged[key].judgement.errors:
            if error.feature and error.decision.question in ASKED:
                wrong[key, error.feature, error.decision.question] = (
                    error.decision, sum(shares.get(piece, 0.0) for piece in error.components))
    total, at_stake = len(wrong), sum(share for _, share in wrong.values())
    asked = flagged = 0
    hits_triggered, hits_flagged, right_truth, right_claim, answered_wrong, flagged_share = 0, 0, 0, 0, 0, 0.0
    for row in rows:
        if not row.get("reread") or not row.get("claims"):
            continue
        for item in row["items"]:
            for q in item["questions"]:
                asked += 1
                said, read = row["claims"][f"{item['item']}/{q}"]["answer"], row["reread"][f"{item['item']}/{q}"]["answer"]
                truth = wrong.get((row["key"], item["feature"], q))
                hits_triggered += truth is not None
                if truth is not None:
                    answered_wrong += 1
                    right_truth += read == truth[0].truth
                    right_claim += said == truth[0].read
                if "unstated" not in (said, read) and said != read:
                    flagged += 1
                    if truth is not None:
                        hits_flagged += 1
                        flagged_share += truth[1]
    n = len(rows)
    print(f"{total} errors on these questions in {n} final readings, {at_stake / n:.2%} of a part's volume on average")
    print(f"every triggered question flagged: {asked} flags, {hits_triggered} hit "
          f"(precision {hits_triggered / max(asked, 1):.0%}, recall {hits_triggered / max(total, 1):.0%})")
    print(f"re-read disagrees: {flagged} flags, {hits_flagged} hit (precision {hits_flagged / max(flagged, 1):.0%}, "
          f"recall {hits_flagged / max(total, 1):.0%}); volume of the errors it flags {flagged_share / n:.2%} of a part on average")
    print(f"on the {answered_wrong} triggered questions the judge found wrong: the re-read gives the part's answer "
          f"{right_truth}, the claims give the reading's answer {right_claim}")
    for call in ("reread", "claims"):
        done = [row for row in rows if row.get(f"{call}_usage")]
        tokens = [[sum(u[k] for u in row[f"{call}_usage"]) for row in done] for k in ("input_tokens", "output_tokens")]
        seconds = [row[f"{call}_seconds"] for row in done]
        print(f"{call}: {len(done)} readings, per reading median {np.median(seconds):.0f} s, "
              f"{np.median(tokens[0]):,.0f} input and {np.median(tokens[1]):,.0f} output tokens")


if __name__ == "__main__":
    pairs = [pair for pair in read_rows(sys.argv[1], Pair) if pair.which == "final"]
    comparisons = {row.key: row for row in read_rows(sys.argv[2], Comparison)}
    judged = {row.key: row for row in read_rows(sys.argv[3], Judged) if row.judgement is not None}
    out = Path(sys.argv[4])
    done = {json.loads(line)["key"] for line in open(out)} if out.exists() else set()
    models, lock = {call: luna(effort) for call, effort in EFFORTS.items()}, Lock()

    def work(pair: Pair) -> None:
        try:
            row = reread(pair, models)
        except Exception as error:  # one failing reading must not stop the rest
            print(pair.key, "FAILED", type(error).__name__, str(error)[:300], flush=True)
            return
        with lock, open(out, "a") as rows:
            rows.write(json.dumps(row) + "\n")
        print(pair.key, len(row["items"]), "items", "ok" if row.get("reread") and row.get("claims") else "unanswered", flush=True)

    with ThreadPoolExecutor(int(sys.argv[5]) if len(sys.argv) > 5 else 8) as pool:
        list(pool.map(work, [pair for pair in pairs if pair.key in judged and pair.key not in done]))
    report([json.loads(line) for line in open(out)], judged, comparisons)
