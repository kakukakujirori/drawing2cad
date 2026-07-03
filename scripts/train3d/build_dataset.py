"""build_dataset.py — bundle AMVDG graphs + GT CadQuery into an SFT jsonl.

For every paired uuid (has both a graph.json and a GT cadquery.py) emit one record:
    {"id", "input_text" (g2 serialization), "target_code" (GT CadQuery),
     "n_tok_input", "n_tok_target", "n_tok_total"}
plus a deterministic train/val split and a token-length report. The g2 text is
produced by the frozen `scripts/amvdg/serialize_g2.graph_to_g2` (do not reimplement).

The instruction/prompt is NOT baked into input_text — `train_sft.py` wraps input_text
with `serialize_g2.PROMPT` + the model chat template, so the same jsonl serves any
prompt/checkpoint. Token counts use the target checkpoint's tokenizer when available.

Run in any env with `transformers` (e.g. py312 / drawing2cad-train):
  python build_dataset.py --graph-dir experiments/dataset_z2c \
      --code-dir experiments/stage_z2c --out scripts/train3d/data_z2c
"""
import os
import sys
import json
import glob
import random
import argparse
import statistics as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "amvdg"))
from serialize_g2 import graph_to_g2, PROMPT  # noqa: E402


def load_tokenizer(name):
    from transformers import AutoTokenizer
    for lfo in (True, False):
        try:
            return AutoTokenizer.from_pretrained(name, local_files_only=lfo, trust_remote_code=True)
        except Exception:
            continue
    return None


def dist(xs):
    xs2 = sorted(xs)
    p = lambda q: xs2[min(len(xs2) - 1, int(q / 100 * len(xs2)))]
    return {"min": min(xs), "median": int(st.median(xs)), "mean": round(st.mean(xs), 1),
            "p90": p(90), "p95": p(95), "p99": p(99), "max": max(xs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="experiments/dataset_z2c")
    ap.add_argument("--code-dir", default="experiments/stage_z2c")
    ap.add_argument("--out", default="scripts/train3d/data_z2c")
    ap.add_argument("--tokenizer", default="ADSKAILab/Zero-To-CAD-Qwen3-VL-2B",
                    help="for token counts; falls back to Qwen2-VL-2B then char/3.5")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tok = load_tokenizer(args.tokenizer)
    if tok is None:
        tok = load_tokenizer("Qwen/Qwen2-VL-2B-Instruct")
    tok_name = getattr(tok, "name_or_path", "char/3.5-approx") if tok else "char/3.5-approx"
    ntok = (lambda s: len(tok(s)["input_ids"])) if tok else (lambda s: int(len(s) / 3.5))

    graphs = sorted(glob.glob(os.path.join(args.graph_dir, "*.graph.json")))
    records, skipped = [], []
    for gp in graphs:
        sid = os.path.basename(gp)[:-len(".graph.json")]
        cp = os.path.join(args.code_dir, f"{sid}.cadquery.py")
        if not os.path.exists(cp):
            skipped.append(sid); continue
        try:
            g2 = graph_to_g2(json.load(open(gp)))
        except Exception as e:
            skipped.append(f"{sid}:g2fail:{e}"); continue
        code = open(cp).read()
        ti, tt = ntok(g2), ntok(code)
        records.append({"id": sid, "input_text": g2, "target_code": code,
                        "n_tok_input": ti, "n_tok_target": tt, "n_tok_total": ti + tt})

    records.sort(key=lambda r: r["id"])
    rng = random.Random(args.seed)
    order = records[:]
    rng.shuffle(order)
    n_val = int(round(args.val_frac * len(order)))
    if args.val_frac > 0:
        n_val = max(1, n_val)
    val_ids = {r["id"] for r in order[:n_val]}
    train = [r for r in records if r["id"] not in val_ids]
    val = [r for r in records if r["id"] in val_ids]

    def write(path, rows):
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    write(os.path.join(args.out, "train.jsonl"), train)
    write(os.path.join(args.out, "val.jsonl"), val)
    write(os.path.join(args.out, "all.jsonl"), records)

    inp = [r["n_tok_input"] for r in records]
    tgt = [r["n_tok_target"] for r in records]
    tot = [r["n_tok_total"] for r in records]
    caps = {}
    for cap in (2048, 3072, 4096, 6144, 8192):
        over = sum(1 for x in tot if x + 40 > cap)
        caps[cap] = {"pairs_over": over, "pct_over": round(100 * over / len(tot), 1),
                     "pairs_kept": len(tot) - over}
    mnt = {}
    for cap in (768, 1024, 1280, 1536):
        mnt[cap] = round(100 * sum(1 for x in tgt if x <= cap) / len(tgt), 1)
    stats = {"n_records": len(records), "n_train": len(train), "n_val": len(val),
             "n_skipped": len(skipped), "tokenizer": tok_name, "prompt": PROMPT,
             "input_tokens": dist(inp), "target_tokens": dist(tgt), "total_tokens": dist(tot),
             "seq_cap_coverage": caps, "max_new_tokens_target_coverage_pct": mnt,
             "val_ids": sorted(val_ids)}
    json.dump(stats, open(os.path.join(args.out, "stats.json"), "w"), indent=2)

    print(f"tokenizer: {tok_name}")
    print(f"records {len(records)}  train {len(train)}  val {len(val)}  skipped {len(skipped)}")
    print(f"INPUT  tokens {stats['input_tokens']}")
    print(f"TARGET tokens {stats['target_tokens']}")
    print(f"TOTAL  tokens {stats['total_tokens']}")
    print("seq cap -> pairs kept / over:")
    for cap, d in caps.items():
        print(f"  {cap:5d}: keep {d['pairs_kept']:3d}  over {d['pairs_over']:3d} ({d['pct_over']}%)")
    print("max_new_tokens target coverage %:", mnt)
    print(f"wrote {args.out}/{{train,val,all}}.jsonl + stats.json")


if __name__ == "__main__":
    main()
