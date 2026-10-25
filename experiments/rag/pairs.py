"""Readings and the builds made under them, from finished runs.

    python -m experiments.rag.pairs OUT.jsonl SWEEP_DIR...

A SWEEP_DIR is a run_parallel.sh --out folder, one run per <dataset>/<part>.
Each run gives two pairs: its first reading with features and the last build
made under it, and the reading in force when the scored build was made.
"""

import json
import sys
from pathlib import Path

import yaml

from experiments.rag.schema import Pair, append_row


def _readings(attempts: Path) -> list[tuple[float, Path]]:
    """Accepted interpretations, oldest first; rejected ones save no interpretation.json."""
    saved = attempts.glob("round_*/interpretation/*/interpretation.json")
    return sorted((path.stat().st_mtime, path) for path in saved)


def _governing(readings: list[tuple[float, Path]], built: float) -> Path | None:
    """The reading in force when a build was written."""
    before = [path for saved, path in readings if saved <= built]
    return before[-1] if before else None


def run_pairs(run: Path) -> list[Pair]:
    score = json.loads((run / "score.json").read_text())
    config = yaml.safe_load((run / ".hydra/config.yaml").read_text())
    attempts = run / "workspace/attempts"
    readings = _readings(attempts)
    builds = sorted((path.stat().st_mtime, path) for path in attempts.glob("round_*/coding/*/output.step"))
    first = next((path for _, path in readings if json.loads(path.read_text())["features"]), None)
    found = {}
    if first is not None:
        under_first = [path for built, path in builds if _governing(readings, built) == first]
        if under_first:
            found["first"] = (first, under_first[-1])
    final = Path(score["pred_step"]) if score.get("pred_step") else None
    if final is not None and final.is_file():
        if (reading := _governing(readings, final.stat().st_mtime)) is not None:
            found["final"] = (reading, final)
    model, dataset, part = config["model"]["model"], run.parent.name, run.name
    return [
        Pair(
            key=f"{model}/{dataset}/{part}/{which}",
            model=model,
            dataset=dataset,
            part=part,
            which=which,
            drawing=str(run / "workspace/inputs/view_drawing.png"),
            gt=score["target_step"],
            interpretation=str(reading),
            build=str(build),
        )
        for which, (reading, build) in found.items()
    ]


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.unlink(missing_ok=True)
    for sweep in map(Path, sys.argv[2:]):
        for run in sorted(sweep.glob("*/*/score.json")):
            pairs = run_pairs(run.parent)
            for pair in pairs:
                append_row(out, pair)
            print(run.parent.relative_to(sweep), [pair.which for pair in pairs])
