"""Sol says which feature of a reading each difference from the real part comes
from, and words each reading error so that it carries over to other drawings.

    python -m experiments.rag.judge PAIRS.jsonl COMPARISONS.jsonl OUT.jsonl [WORKERS]

Pairs already in OUT are skipped; judging the same pairs into a second OUT
repeats the judgement.
"""

import base64
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from string import Template
from threading import Lock

from hydra.utils import instantiate
from langchain_core.messages import HumanMessage
from omegaconf import OmegaConf

from experiments.rag.schema import Comparison, Judged, Judgement, Pair, append_row, read_rows
from zeroshot.pipeline.retrieval.mistakes import QUESTIONS

PROMPT = Template(Path(__file__).with_name("judge.md").read_text()).safe_substitute(questions="\n".join(
    f"  - {question}{f' ({", ".join(answers)})' if answers else ''}: {meaning}"
    for question, (answers, meaning) in QUESTIONS.items()))
MODEL_CONFIG = Path("zeroshot/configs/model/gpt6.1_sol_codex.yaml")
EFFORT = "high"
JUDGE = f"gpt-6.1-sol/{EFFORT}/{hashlib.sha256(PROMPT.encode()).hexdigest()[:12]}"
TRIES = 3


def sol(effort: str = EFFORT):
    config = OmegaConf.load(MODEL_CONFIG).model
    config.reasoning.effort = effort
    return instantiate(config)


def _image(path: str) -> dict:
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}


def reading(interpretation: dict) -> str:
    views = [{key: view.get(key) for key in ("name", "role", "u_axis", "v_axis")} for view in interpretation["views"]]
    features = [{key: feature.get(key) for key in ("name", "description", "parameters")}
                for feature in interpretation["features"]]
    return json.dumps({"datum": interpretation.get("datum"), "views": views, "features": features}, indent=1)


def problems(judgement: Judgement, comparison: Comparison, features: set[str]) -> list[str]:
    """Where an answer breaks the rules: unknown names, pieces left out, unexplained reading pieces."""
    ids = {component.id for component in comparison.components}
    called = {call.id for call in judgement.components}
    explained = {piece for error in judgement.errors for piece in error.components}
    named = {name for call in judgement.components for name in call.features}
    named |= {error.feature for error in judgement.errors if error.feature is not None}
    found = []
    if called != ids:
        found.append(f"pieces called {sorted(called)}, but the pieces are {sorted(ids)}")
    if unknown := explained - ids:
        found.append(f"errors cite unknown pieces {sorted(unknown)}")
    if unknown := named - features:
        found.append(f"unknown features {sorted(unknown)}")
    if bare := {call.id for call in judgement.components if call.cause == "reading"} - explained:
        found.append(f"pieces {sorted(bare)} have cause reading but no error")
    return found


def judge(model, pair: Pair, comparison: Comparison) -> Judged:
    if not comparison.components:
        # Every error must explain a piece, so there is nothing to ask.
        return Judged(key=pair.key, judge=JUDGE, judgement=Judgement(components=[], errors=[]))
    interpretation = json.loads(Path(pair.interpretation).read_text())
    pieces = "\n".join(f"- {c.id}: {c.share:.1%}" for c in comparison.components) or "(none)"
    text = Template(PROMPT).substitute(components=pieces, reading=reading(interpretation))
    content = [{"type": "text", "text": text}, _image(pair.drawing)]
    content += [_image(comparison.images[name]) for name in ("gt", "build", "diff")]
    messages = [HumanMessage(content=content)]
    features = {feature["name"] for feature in interpretation["features"]}
    raw = None
    for _ in range(TRIES):
        answer = model.invoke(messages)
        raw = answer.text
        try:
            judgement = Judgement.model_validate_json(raw[raw.index("{"):raw.rindex("}") + 1])
        except ValueError as error:
            found = [str(error)[:500]]
        else:
            if not (found := problems(judgement, comparison, features)):
                return Judged(key=pair.key, judge=JUDGE, judgement=judgement, usage=answer.usage_metadata)
        messages += [answer, HumanMessage(content="Fix your answer and give the whole JSON again: " + "; ".join(found))]
    return Judged(key=pair.key, judge=JUDGE, judgement=None, raw=raw)


if __name__ == "__main__":
    pairs = {pair.key: pair for pair in read_rows(sys.argv[1], Pair)}
    comparisons = [row for row in read_rows(sys.argv[2], Comparison) if row.key in pairs]
    out = Path(sys.argv[3])
    done = {row.key for row in read_rows(out, Judged)}
    model, lock = sol(), Lock()

    def work(comparison: Comparison) -> None:
        try:
            judged = judge(model, pairs[comparison.key], comparison)
        except Exception as error:  # one failing pair must not stop the rest
            print(comparison.key, "FAILED", type(error).__name__, str(error)[:300], flush=True)
            return
        with lock:
            append_row(out, judged)
        print(comparison.key, "ok" if judged.judgement else "unparsed", flush=True)

    with ThreadPoolExecutor(int(sys.argv[4]) if len(sys.argv) > 4 else 8) as pool:
        list(pool.map(work, [row for row in comparisons if row.key not in done]))
