"""build_dataset.py — bundle AMVDG graphs + GT CadQuery into one SFT jsonl.

For every paired uuid (has both a graph.json and a GT cadquery.py) emit one record:
    {"id", "input_text" (serialized graph), "target_code" (GT CadQuery),
     "n_tok_input", "n_tok_target", "n_tok_total"}
into `<out>/all.jsonl`, plus a token-length report in `<out>/stats.json`. The input text
is produced by the frozen `scripts/train3d/serialize.graph_to_text` (do not reimplement).

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
import os
import random
import statistics as st
import sys

from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))  # serialize.py is a sibling in train3d/
from serialize import serialize_3d, CANON_QUANT  # noqa: E402


def dist(xs: list[float]):
    xs2 = sorted(xs)
    p = lambda q: xs2[min(len(xs2) - 1, int(q / 100 * len(xs2)))]
    return {"min": min(xs), "median": int(st.median(xs)), "mean": round(st.mean(xs), 1),
            "p90": p(90), "p95": p(95), "p99": p(99), "max": max(xs)}


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
                        help=f"integer-grid coordinate quantization, N levels "
                             f"(default {CANON_QUANT}); 0 = off (ablation). infer.py "
                             f"must use the SAME value.")
    parser.add_argument("--hid-dropout", type=float, default=0.0,
                        help="TRAIN-ONLY augmentation: probability of dropping a "
                             "feature's redundant hidden lines in views where it is "
                             "visible elsewhere (sim-to-real robustness). Leave 0 for "
                             "val/infer. NOTE: baked ONCE here (static draw); per-epoch "
                             "dynamic augmentation would require re-serializing in the "
                             "train loop.")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for --hid-dropout")
    args = parser.parse_args()
    rng = random.Random(args.seed)

    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("ADSKAILab/Zero-To-CAD-Qwen3-VL-2B", trust_remote_code=True)
    tok_name = getattr(tok, "name_or_path", "?")
    ntok = lambda s: len(tok(s)["input_ids"])

    records, skipped = [], []
    for gp in sorted(glob.glob(os.path.join(args.graph_dir, "*.graph.json"))):
        sid = os.path.basename(gp)[:-len(".graph.json")]
        cp = os.path.join(args.code_dir, f"{sid}.cadquery.py")
        if not os.path.exists(cp):
            skipped.append(sid); continue
        try:
            txt = serialize_3d(json.load(open(gp)), quant=args.quant,
                               hid_dropout=args.hid_dropout, rng=rng)
        except Exception as e:
            skipped.append(f"{sid}:serfail:{e}"); continue
        code = open(cp).read()
        ti, tt = ntok(txt), ntok(code)
        records.append({"id": sid, "input_text": txt, "target_code": code,
                        "n_tok_input": ti, "n_tok_target": tt, "n_tok_total": ti + tt})

    records.sort(key=lambda r: r["id"])
    with open(os.path.join(args.out, "all.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    inp = [r["n_tok_input"] for r in records]
    tgt = [r["n_tok_target"] for r in records]
    tot = [r["n_tok_total"] for r in records]
    caps = {}
    for cap in (2048, 3072, 4096, 6144, 8192, 12288, 16384):
        over = sum(1 for x in tot if x + 40 > cap)
        caps[cap] = {"pairs_over": over, "pct_over": round(100 * over / len(tot), 1),
                     "pairs_kept": len(tot) - over}
    mnt = {cap: round(100 * sum(1 for x in tgt if x <= cap) / len(tgt), 1)
           for cap in (768, 1024, 1280, 1536)}
    stats = {"n_records": len(records), "n_skipped": len(skipped),
             "graph_dir": args.graph_dir, "code_dir": args.code_dir, "tokenizer": tok_name,
             "transforms": {"drop_covered": True, "quant": args.quant,
                            "hid_dropout": args.hid_dropout, "seed": args.seed},
             "input_tokens": dist(inp), "target_tokens": dist(tgt), "total_tokens": dist(tot),
             "seq_cap_coverage": caps, "max_new_tokens_target_coverage_pct": mnt}
    json.dump(stats, open(os.path.join(args.out, "stats.json"), "w"), indent=2)

    print(f"tokenizer: {tok_name}")
    print(f"records {len(records)}  skipped {len(skipped)}")
    print(f"INPUT  tokens {stats['input_tokens']}")
    print(f"TARGET tokens {stats['target_tokens']}")
    print(f"TOTAL  tokens {stats['total_tokens']}")
    print("seq cap -> pairs kept / over:")
    for cap, d in caps.items():
        print(f"  {cap:5d}: keep {d['pairs_kept']:3d}  over {d['pairs_over']:3d} ({d['pct_over']}%)")
    print("max_new_tokens target coverage %:", mnt)
    print(f"wrote {args.out}/all.jsonl + stats.json")


if __name__ == "__main__":
    main()
