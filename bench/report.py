#!/usr/bin/env python3
"""Aggregate + compare scored result dirs into an ours-vs-baseline table.

Reads each fixture's result.json (written by evaluate.py) from one or more
labelled run dirs and emits:
  - per-run aggregate: CAD Score, shape, topology, validity rate, counts
  - a per-fixture side-by-side table (CAD Score per run)
  - a "paired" aggregate over the fixtures common to all runs (apples-to-apples)
Outputs report.json + report.md (under results/ by default).

Aggregate CAD Score INCLUDES zeros from invalid/missing fixtures (the
CADGenBench leaderboard convention), so a run cannot inflate its mean by only
averaging its successes.

Usage:
    python bench/report.py --ours results/ours_noblend_v1 \
                           --baseline results/20260705_xxxx_qwen2.5vl:32b
    python bench/report.py --run ours=results/ours_noblend_v1 \
                           --run baseline=results/<...>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_run(run_dir: Path) -> dict:
    """Per-fixture metrics from result.json files under run_dir."""
    fixtures = {}
    for fx in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        rj = fx / "result.json"
        if not rj.exists():
            continue
        try:
            d = json.loads(rj.read_text())
        except Exception:
            continue
        gt = d.get("gt_metrics") or {}
        topo = d.get("topology_metrics") or {}
        val = d.get("validation") or {}
        fixtures[fx.name] = {
            "status": d.get("status", "missing"),
            "cad_score": _f(d.get("cad_score")) or 0.0,
            "shape": _f(gt.get("shape_similarity_score")),
            "topology": _f(topo.get("score")),
            "is_valid": bool(val.get("is_valid")),
        }
    return fixtures


def aggregate(fixtures: dict, ids: list[str] | None = None) -> dict:
    ids = ids if ids is not None else list(fixtures)
    rows = [fixtures[i] for i in ids if i in fixtures]
    n = len(rows)
    if n == 0:
        return {"n": 0}
    shape_vals = [r["shape"] for r in rows if r["shape"] is not None]
    topo_vals = [r["topology"] for r in rows if r["topology"] is not None]
    n_valid = sum(1 for r in rows if r["status"] == "valid")
    n_missing = sum(1 for r in rows if r["status"] == "missing")
    n_invalid = sum(1 for r in rows if r["status"] == "invalid")
    return {
        "n": n,
        "cad_score_mean": round(mean(r["cad_score"] for r in rows), 4),
        "shape_mean_over_valid": round(mean(shape_vals), 4) if shape_vals else None,
        "topology_mean_over_valid": round(mean(topo_vals), 4) if topo_vals else None,
        "validity_rate": round(n_valid / n, 4),
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "n_missing": n_missing,
    }


def build_report(runs: dict[str, Path]) -> dict:
    loaded = {label: load_run(d) for label, d in runs.items()}
    # union + intersection of fixture ids
    all_ids: set[str] = set()
    for fx in loaded.values():
        all_ids |= set(fx)
    common_ids = (
        set.intersection(*(set(fx) for fx in loaded.values())) if loaded else set()
    )

    per_run = {label: aggregate(fx) for label, fx in loaded.items()}
    per_run_paired = {
        label: aggregate(fx, sorted(common_ids)) for label, fx in loaded.items()
    }

    per_fixture = {}
    for uuid in sorted(all_ids):
        per_fixture[uuid] = {
            label: (fx.get(uuid, {}).get("cad_score") if uuid in fx else None)
            for label, fx in loaded.items()
        }

    return {
        "runs": {label: str(d) for label, d in runs.items()},
        "per_run_aggregate": per_run,
        "paired_aggregate": {
            "n_common": len(common_ids),
            "by_run": per_run_paired,
        },
        "per_fixture_cad_score": per_fixture,
        "_detail": {label: fx for label, fx in loaded.items()},
    }


def to_markdown(rep: dict) -> str:
    labels = list(rep["runs"])
    lines = ["# Pseudo-CADGenBench: ours vs LLM baseline", ""]
    lines.append(
        "Metric = CADGenBench CAD Score. Our fixtures have no interface "
        "jig sub-volumes, so CAD Score = shape 0.4 / topology 0.2 "
        "renormalized => shape 2/3, topology 1/3. Invalid/missing => 0."
    )
    lines.append("")
    lines.append("## Runs")
    for label, d in rep["runs"].items():
        lines.append(f"- **{label}**: `{d}`")
    lines.append("")

    def agg_table(agg_by_run: dict, title: str, n_note: str = ""):
        out = [f"## {title} {n_note}".rstrip(), ""]
        cols = ["metric"] + labels
        out.append("| " + " | ".join(cols) + " |")
        out.append("|" + "|".join(["---"] * len(cols)) + "|")
        rows = [
            ("n", "n"),
            ("CAD Score (mean, incl. 0s)", "cad_score_mean"),
            ("shape (mean over valid)", "shape_mean_over_valid"),
            ("topology (mean over valid)", "topology_mean_over_valid"),
            ("validity rate", "validity_rate"),
            ("n_valid", "n_valid"),
            ("n_invalid", "n_invalid"),
            ("n_missing", "n_missing"),
        ]
        for disp, key in rows:
            cells = [disp]
            for label in labels:
                v = agg_by_run.get(label, {}).get(key)
                cells.append("-" if v is None else str(v))
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
        return out

    lines += agg_table(
        rep["per_run_aggregate"], "Per-run aggregate (all fixtures in each run)"
    )
    paired = rep["paired_aggregate"]
    lines += agg_table(
        paired["by_run"],
        "Paired aggregate (fixtures common to all runs)",
        f"— n_common={paired['n_common']}",
    )

    lines.append("## Per-fixture CAD Score")
    lines.append("")
    cols = ["fixture"] + labels
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for uuid, row in rep["per_fixture_cad_score"].items():
        cells = [uuid[:8]]
        for label in labels:
            v = row.get(label)
            cells.append("-" if v is None else f"{v:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ours", type=Path, default=None, help="ours run dir")
    parser.add_argument("--baseline", type=Path, default=None, help="baseline run dir")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL=DIR",
        help="add a labelled run (repeatable)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=C.RESULTS_DIR / "report",
        help="output path stem (writes .json + .md)",
    )
    args = parser.parse_args()

    runs: dict[str, Path] = {}
    if args.ours:
        runs["ours"] = args.ours if args.ours.is_absolute() else C.REPO / args.ours
    if args.baseline:
        runs["baseline"] = (
            args.baseline if args.baseline.is_absolute() else C.REPO / args.baseline
        )
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"--run expects LABEL=DIR, got {spec!r}")
        label, path = spec.split("=", 1)
        p = Path(path)
        runs[label] = p if p.is_absolute() else C.REPO / p
    if not runs:
        raise SystemExit("Provide at least one run (--ours/--baseline/--run).")
    for label, d in runs.items():
        if not d.is_dir():
            raise SystemExit(f"Run dir for {label} not found: {d}")

    rep = build_report(runs)
    out_stem = args.out if args.out.is_absolute() else C.REPO / args.out
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    (out_stem.with_suffix(".json")).write_text(json.dumps(rep, indent=2))
    md = to_markdown(rep)
    (out_stem.with_suffix(".md")).write_text(md)
    print(md)
    print(
        f"\n[report] wrote {out_stem.with_suffix('.json')} and {out_stem.with_suffix('.md')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
