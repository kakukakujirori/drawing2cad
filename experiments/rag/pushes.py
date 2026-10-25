"""Did the coder act on the past mistakes it was shown?

    python -m experiments.rag.pushes SWEEP_DIR...

Per run: the pushes and their entries, the coder's one-line answers, its trials
after the first push, its reasoning headings right after it, and the IoU.
"""

import json
import re
import sys
from pathlib import Path

MARK = "[Past mistakes]"
ANSWER = re.compile(r"(\w+ \d+): (same mistake|not here)")


def blocks(event: dict) -> list[dict]:
    return [block for message in event["data"].get("messages", [])
            if isinstance(message.get("content"), list) for block in message["content"]]


def message_texts(event: dict) -> list[str]:
    return [message["content"] if isinstance(message.get("content"), str) else "" for message in
            event["data"].get("messages", [])] + [block.get("text", "") for block in blocks(event)]


def report(run: Path) -> dict:
    events = [json.loads(line) for line in open(run / "events.jsonl")]
    pushes = [n for n, e in enumerate(events) if e["event"] == "message" and MARK in json.dumps(e)]
    after = events[pushes[0]:] if pushes else []
    texts = [b.get("text", "") for e in after if e["event"] == "message" and e["data"].get("node") == "model"
             for b in blocks(e) if b.get("type") == "text"]
    headings = [b["reasoning"] for e in after[:80] if e["event"] == "message" and e["data"].get("node") == "model"
                for b in blocks(e) if b.get("type") == "reasoning" and b.get("reasoning")]
    entries = sum(len(re.findall(r"^\d+\. Read as:", text, re.M))
                  for n in pushes for text in message_texts(events[n]) if MARK in text)
    score = run / "score.json"
    return {
        "run": f"{run.parent.name}/{run.name}",
        "pushes": len(pushes),
        "entries": entries,
        "answers": [match.groups() for text in texts for match in ANSWER.finditer(text)],
        "trials_after_push": sum(1 for e in after if e["event"] == "tool_started"
                                 and e["data"]["tool_name"] == "evaluate_cad_hypothesis"),
        "headings_after_push": headings[:12],
        "iou": json.loads(score.read_text())["metrics"].get("mesh_iou") if score.exists() else None,
    }


if __name__ == "__main__":
    for sweep in map(Path, sys.argv[1:]):
        for events in sorted(sweep.glob("*/*/events.jsonl")):
            print(json.dumps(report(events.parent), indent=1))
