#!/usr/bin/env python
"""Build a self-contained cadrille-2D training bundle from renderer output.

Reads (png, graph.json) pairs, serializes each graph to the compact target text
(serialize.graph_to_text), and writes a bundle dir:

    <out>/
      images/<file_name>.png         # staged copy of each drawing
      train.jsonl  val.jsonl         # {"image","target","file_name","variant"}
      meta.json                      # provenance + token/char stats

The bundle is path-relative and self-contained, so `train.py --data-root <out>` can
consume it as-is. CPU-only. Run from this directory for the sibling `serialize` import.

Sources (any combination):
  --manifest <dataset>/manifest.jsonl   # uses png + scan_png + graph
  --glob '<dataset>/*.graph.json'        # pairs each with sibling .png/.scan.png
"""

import os
import json
import glob
import shutil
import argparse
from serialize import graph_to_text, PROMPT, SER_VERSION


def stage(png, gjson, file_name, variant, out, rows):
    if not (png and os.path.isfile(png) and os.path.isfile(gjson)):
        return
    graph = json.load(open(gjson))
    target = graph_to_text(graph)
    dst = os.path.join(out, "images", f"{file_name}.png")
    shutil.copyfile(png, dst)
    rows.append(
        {
            "image": f"images/{file_name}.png",
            "target": target,
            "file_name": file_name,
            "variant": variant,
        }
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="dataset/manifest.jsonl")
    ap.add_argument("--glob", default=None)
    ap.add_argument("--out", default="data_bundle")
    ap.add_argument("--include-scan", action="store_true", default=True)
    ap.add_argument("--no-scan", dest="include_scan", action="store_false")
    ap.add_argument(
        "--val-frac",
        type=float,
        default=0.0,
        help="fraction held out for val; 0 -> val=train (smoke/overfit)",
    )
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    rows = []

    if args.manifest and os.path.isfile(args.manifest):
        for line in open(args.manifest):
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            pid = m.get("part_id") or os.path.basename(m["graph"]).split(".")[0]
            stage(m.get("png"), m["graph"], f"{pid}__clean", "clean", args.out, rows)
            if args.include_scan and m.get("scan_png"):
                stage(
                    m.get("scan_png"),
                    m["graph"],
                    f"{pid}__scan",
                    "scan",
                    args.out,
                    rows,
                )

    if args.glob:
        for gjson in sorted(glob.glob(args.glob)):
            base = (
                gjson[: -len(".graph.json")]
                if gjson.endswith(".graph.json")
                else os.path.splitext(gjson)[0]
            )
            pid = os.path.basename(base)
            stage(base + ".png", gjson, f"{pid}__clean", "clean", args.out, rows)
            if args.include_scan and os.path.isfile(base + ".scan.png"):
                stage(base + ".scan.png", gjson, f"{pid}__scan", "scan", args.out, rows)

    assert rows, "no (png, graph.json) pairs found"

    # split
    if args.val_frac <= 0:
        train, val = rows, rows
    else:
        k = max(1, int(len(rows) * args.val_frac))
        val, train = rows[:k], rows[k:] or rows[:k]

    with open(os.path.join(args.out, "train.jsonl"), "w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "val.jsonl"), "w") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    chars = [len(r["target"]) for r in rows]
    meta = {
        "ser_version": SER_VERSION,
        "prompt": PROMPT,
        "n": len(rows),
        "n_train": len(train),
        "n_val": len(val),
        "variants": sorted({r["variant"] for r in rows}),
        "target_chars": {
            "min": min(chars),
            "max": max(chars),
            "mean": sum(chars) // len(chars),
        },
    }
    json.dump(
        meta,
        open(os.path.join(args.out, "meta.json"), "w"),
        indent=2,
        ensure_ascii=False,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"\nbundle -> {args.out}  ({len(rows)} samples)")


if __name__ == "__main__":
    main()
