"""build_dataset.py — bundle AMVDG graphs + GT CadQuery into one SFT jsonl.

For every paired uuid (has both a graph.json and a GT cadquery.py) emit one record:
    {"id", "input_text" (serialized graph), "target_code" (GT CadQuery),
     "n_tok_input", "n_tok_target", "n_tok_total"}
into `<out>/all.jsonl`, plus `<out>/invalid_extent.jsonl` quarantine records and a
token/extent report in `<out>/stats.json`. The input text is produced by the frozen
`scripts/train3d/serialize.graph_to_text` (do not reimplement).

There is NO train/val split here: the split is the Zero-To-CAD **source split**, so you run
this once per split and train_sft consumes the two bundles separately (`--train`/`--val`):
  python build_dataset.py --graph-dir experiments/dataset_z2c_train \
      --code-dir experiments/stage_z2c_train --out experiments/data_z2c_train
  python build_dataset.py --graph-dir experiments/dataset_z2c_val \
      --code-dir experiments/stage_z2c_val   --out experiments/data_z2c_val

The instruction/prompt is NOT part of a record — train_sft wraps input_text with its own
prompt + the chat template, so the same jsonl serves any prompt. Run in the `drawing2cad` env.
"""
import argparse
import glob
import json
import math
import os
import random
import statistics as st
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))  # serialize.py is a sibling in train3d/
from serialize import (CANON_QUANT, extent_ratio, quantization_extent, serialize_3d,
                       serialized_scale, text_to_struct)  # noqa: E402


def dist(xs: list[float]):
    xs2 = sorted(xs)
    p = lambda q: xs2[min(len(xs2) - 1, int(q / 100 * len(xs2)))]
    return {"min": min(xs), "median": int(st.median(xs)), "mean": round(st.mean(xs), 1),
            "p90": p(90), "p95": p(95), "p99": p(99), "max": max(xs)}


def ratio_dist(xs: list[float]):
    xs2 = sorted(x for x in xs if x != float("inf"))
    if not xs2:
        return {"p50": None, "p90": None, "p99": None, "max": None,
                "n_nonfinite": len(xs)}
    p = lambda q: xs2[min(len(xs2) - 1, int(q / 100 * len(xs2)))]
    return {"p50": p(50), "p90": p(90), "p99": p(99), "max": max(xs2),
            "n_nonfinite": len(xs) - len(xs2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", type=str, default="experiments/dataset_z2c_val",
                        help="AMVDG graph dir (*.graph.json)")
    parser.add_argument("--code-dir", type=str, default="experiments/stage_z2c_val",
                        help="GT CadQuery dir (*.cadquery.py)")
    parser.add_argument("--out", type=str, default="experiments/data_z2c_val")
    # --- serialization (serialize_3d): covered-hidden suppression is ALWAYS on
    #     (lossless); these knobs are the only variables, and must match on infer.py ---
    parser.add_argument("--quant", type=int, default=CANON_QUANT,
                        help=f"signed integer-grid coordinate quantization, N magnitude bins "
                             f"per sign (default {CANON_QUANT}, yielding "
                             f"[-{CANON_QUANT - 1}, {CANON_QUANT - 1}]); "
                             f"0 = off (ablation). infer.py "
                             f"must use the SAME value.")
    parser.add_argument("--hid-dropout", type=float, default=0.0,
                        help="Probability of dropping a feature's redundant hidden lines "
                             "in views where it is visible elsewhere (sim-to-real robustness). "
                             "NOTE: baked ONCE here (static draw); per-epoch dynamic "
                             "augmentation would require re-serializing in the train loop.")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for --hid-dropout")
    parser.add_argument("--workers", type=int, default=16,
                        help="thread pool size: per-record work is file I/O + the "
                             "fast (Rust) tokenizer, both of which release the GIL, "
                             "so threads (not processes) parallelize this well")
    parser.add_argument("--extent-warn-ratio", type=float, default=2.0,
                        help="keep but report records whose serialized 2D extent / "
                             "max 3D bbox side exceeds this value (default 2)")
    parser.add_argument("--extent-fail-ratio", type=float, default=5.0,
                        help="quarantine records above this ratio instead of writing "
                             "them to all.jsonl (default 5)")
    parser.add_argument("--absolute-bbox-mm", type=float, default=None,
                        help="optional independent domain gate: quarantine parts whose "
                             "largest bbox side exceeds this many mm")
    args = parser.parse_args()
    if args.extent_warn_ratio < 0 or args.extent_fail_ratio <= args.extent_warn_ratio:
        parser.error("require 0 <= --extent-warn-ratio < --extent-fail-ratio")
    if args.absolute_bbox_mm is not None and args.absolute_bbox_mm <= 0:
        parser.error("--absolute-bbox-mm must be positive")

    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("ADSKAILab/Zero-To-CAD-Qwen3-VL-2B", trust_remote_code=True)
    tok_name = getattr(tok, "name_or_path", "?")
    ntok = lambda s: len(tok(s)["input_ids"])

    def record_seed(sid):
        # per-record deterministic seed, NOT a single rng shared/threaded across
        # records: hash() is salted per process (not reproducible, see
        # scripts/renderer/batch_dataset.py's scan_augment for the same fix), and a
        # shared sequential Random's draw order would depend on thread scheduling.
        return zlib.crc32(f"{args.seed}:{sid}".encode()) & 0xffffffff

    def process_one(gp):
        sid = os.path.basename(gp)[:-len(".graph.json")]
        cp = os.path.join(args.code_dir, f"{sid}.cadquery.py")
        if not os.path.exists(cp):
            return ("SKIP", sid, None)
        try:
            with open(gp) as f:
                graph = json.load(f)
            txt = serialize_3d(graph, quant=args.quant,
                               hid_dropout=args.hid_dropout,
                               rng=random.Random(record_seed(sid)))
        except Exception as e:
            return ("SKIP", f"{sid}:serfail:{e}", None)
        scale = serialized_scale(txt)
        if scale is None:
            # quant=0 ablations have no grid/ext header.  Parsing the emitted
            # text measures the exact post-suppression/dropout geometry rather
            # than a subtly different reconstruction from the source graph.
            stxt = text_to_struct(txt)
            bbox = stxt["bbox"]
            ext = quantization_extent(stxt)
            ratio = extent_ratio(ext, bbox)
        else:
            bbox, ext, ratio = scale
        reasons = []
        if ratio > args.extent_fail_ratio:
            reasons.append("extent_ratio")
        bbox_vals = [float(x) for x in bbox]
        bbox_max = (max((abs(x) for x in bbox_vals), default=float("inf"))
                    if all(math.isfinite(x) for x in bbox_vals) else float("inf"))
        if args.absolute_bbox_mm is not None and bbox_max > args.absolute_bbox_mm:
            reasons.append("absolute_bbox_mm")
        audit = {"id": sid, "reasons": reasons,
                 "bbox": [x if math.isfinite(x) else None for x in bbox_vals],
                 "extent": ext if math.isfinite(ext) else None,
                 "extent_ratio": ratio if math.isfinite(ratio) else None,
                 "graph": gp}
        if reasons:
            return ("QUARANTINE", sid, audit)
        with open(cp) as f:
            code = f.read()
        ti, tt = ntok(txt), ntok(code)
        return ("OK", sid, ({"id": sid, "input_text": txt, "target_code": code,
                             "n_tok_input": ti, "n_tok_target": tt,
                             "n_tok_total": ti + tt}, audit))

    graphs = sorted(glob.glob(os.path.join(args.graph_dir, "*.graph.json")))
    records, skipped, quarantined, extent_audits = [], [], [], []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for kind, key, payload in tqdm(ex.map(process_one, graphs), total=len(graphs)):
            if kind == "SKIP":
                skipped.append(key)
            elif kind == "QUARANTINE":
                quarantined.append(payload)
                extent_audits.append(payload)
            else:
                record, audit = payload
                records.append(record)
                extent_audits.append(audit)

    records.sort(key=lambda r: r["id"])
    with open(os.path.join(args.out, "all.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.out, "invalid_extent.jsonl"), "w") as f:
        for r in sorted(quarantined, key=lambda x: x["id"]):
            f.write(json.dumps(r, allow_nan=False) + "\n")

    inp = [r["n_tok_input"] for r in records]
    tgt = [r["n_tok_target"] for r in records]
    tot = [r["n_tok_total"] for r in records]
    if not records:
        raise RuntimeError("no valid paired records after serialization/quarantine; "
                           "see invalid_extent.jsonl and source paths")
    caps = {}
    for cap in (2048, 3072, 4096, 6144, 8192, 12288, 16384):
        over = sum(1 for x in tot if x + 40 > cap)
        caps[cap] = {"pairs_over": over, "pct_over": round(100 * over / len(tot), 1),
                     "pairs_kept": len(tot) - over}
    mnt = {cap: round(100 * sum(1 for x in tgt if x <= cap) / len(tgt), 1)
           for cap in (768, 1024, 1280, 1536)}
    ratios = [a["extent_ratio"] if a["extent_ratio"] is not None else float("inf")
              for a in extent_audits]
    warn_ids = [a["id"] for a in extent_audits
                if a["extent_ratio"] is not None
                and args.extent_warn_ratio < a["extent_ratio"] <= args.extent_fail_ratio]
    stats = {"n_records": len(records), "n_skipped": len(skipped),
             "n_extent_warn": len(warn_ids),
             "n_extent_quarantined": len(quarantined),
             "graph_dir": args.graph_dir, "code_dir": args.code_dir, "tokenizer": tok_name,
             "transforms": {"drop_covered": True, "quant": args.quant,
                            "hid_dropout": args.hid_dropout, "seed": args.seed},
             "extent_policy": {"warn_ratio": args.extent_warn_ratio,
                               "fail_ratio": args.extent_fail_ratio,
                               "absolute_bbox_mm": args.absolute_bbox_mm},
             "extent_ratio": ratio_dist(ratios),
             "extent_warning_ids": warn_ids,
             "extent_quarantined_ids": [a["id"] for a in quarantined],
             "input_tokens": dist(inp), "target_tokens": dist(tgt), "total_tokens": dist(tot),
             "seq_cap_coverage": caps, "max_new_tokens_target_coverage_pct": mnt}
    json.dump(stats, open(os.path.join(args.out, "stats.json"), "w"), indent=2)

    print(f"tokenizer: {tok_name}")
    print(f"records {len(records)}  skipped {len(skipped)}  "
          f"extent_warn {len(warn_ids)}  extent_quarantined {len(quarantined)}")
    print(f"INPUT  tokens {stats['input_tokens']}")
    print(f"TARGET tokens {stats['target_tokens']}")
    print(f"TOTAL  tokens {stats['total_tokens']}")
    print("seq cap -> pairs kept / over:")
    for cap, d in caps.items():
        print(f"  {cap:5d}: keep {d['pairs_kept']:3d}  over {d['pairs_over']:3d} ({d['pct_over']}%)")
    print("max_new_tokens target coverage %:", mnt)
    print(f"wrote {args.out}/all.jsonl + invalid_extent.jsonl + stats.json")


if __name__ == "__main__":
    main()
