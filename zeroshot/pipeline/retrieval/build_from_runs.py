"""Register the features of finished runs, keyed by crops of their evidence.

    python -m zeroshot.pipeline.retrieval.build_from_runs DB RUN_DIR... [--min-f1 F]

A RUN_DIR is one sample's artifact folder (workspace/, score.json, .hydra/).
Only the best-scoring run of each sample is registered; ditech runs never are.
A run with no target (workflow=interpretation) has no score and is kept.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from zeroshot.pipeline.retrieval.evidence import feature_crops
from zeroshot.pipeline.retrieval.index import RegionIndex, RoleCrop


def _score(run: Path) -> float | None:
    try:
        return json.loads((run / "score.json").read_text())["metrics"]["eccv_mean_f1"]
    except (OSError, KeyError, TypeError, ValueError):
        return None


def _validated(run: Path) -> bool:
    """Whether the interpretation left is one that validation accepted and saved."""
    left = run / "workspace/interpretation.json"
    if not left.is_file():
        return False
    accepted = run.glob(
        "workspace/attempts/round_*/interpretation/*/interpretation.json"
    )
    return any(saved.read_bytes() == left.read_bytes() for saved in accepted)


def ranked_runs(
    runs: list[Path], min_f1: float
) -> dict[str, list[tuple[float | None, Path, str]]]:
    """Per sample id: (F1, run folder, model), best first; ditech is left out."""
    ranked: dict[str, list[tuple[float | None, Path, str]]] = {}
    for run in runs:
        if not _validated(run):
            continue
        config = yaml.safe_load((run / ".hydra/config.yaml").read_text())
        f1 = _score(run)
        scored = config["sample"]["target_step_path"] is not None
        if scored and (f1 is None or f1 < min_f1):
            continue
        if any(
            "ditech" in sheet["file"] for sheet in config["sample"]["drawing"]["sheets"]
        ):
            continue
        sample = str(config["sample"]["sample_id"])
        ranked.setdefault(sample, []).append((f1, run, config["model"]["model"]))
    return {
        sample: sorted(found, key=lambda run: run[0] or 0.0, reverse=True)
        for sample, found in ranked.items()
    }


def _readable(
    sample: str, ranked: list[tuple[float | None, Path, str]]
) -> tuple[float | None, Path, str, dict[str, tuple[str, list[RoleCrop]]]] | None:
    """The best run whose interpretation this reader understands."""
    for f1, run, model in ranked:
        interpretation = json.loads((run / "workspace/interpretation.json").read_text())
        try:
            features = feature_crops(interpretation, run / "workspace")
        except (KeyError, OSError, TypeError, ValueError) as error:  # older schemas
            print(f"{sample}: skipped {run} ({error!r})")
            continue
        if features:
            return f1, run, model, features
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("db", type=Path)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--min-f1", type=float, default=0.6)
    args = parser.parse_args()
    index = RegionIndex(args.db)
    for sample, ranked in sorted(ranked_runs(args.runs, args.min_f1).items()):
        if (found := _readable(sample, ranked)) is None:
            continue
        f1, run, model, features = found
        for name, (description, crops) in features.items():
            meta = {"model": model, "eccv_mean_f1": f1}
            index.add(sample, str(run), name, description, crops, meta)
        print(f"{sample} F1 {f1} {model}: {len(features)} features from {run}")


if __name__ == "__main__":
    main()
