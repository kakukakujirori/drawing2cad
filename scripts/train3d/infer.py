"""infer.py — AMVDG graph JSON -> CadQuery code (+ executed STEP) with a trained 3D-leg model.

Batch inference twin of the train_sft eval hook: for each AMVDG JSON it serializes the graph
(serialize.graph_to_text), wraps it in the SAME PROMPT + chat template train_sft uses, greedy-
generates CadQuery, and writes `{stem}.py`. It then execs that code in an isolated timeout'd
subprocess (eval_cq-style containment) and exports the `result` solid to `{stem}.step`.

`stem` = input filename minus `.graph.json` / `.json` (so `<uuid>.graph.json` -> `<uuid>.py` +
`<uuid>.step`), which is exactly what eval_cq.py pairs against the GT `{uuid}.step`.

Checkpoint:
  * a dir with adapter_config.json  -> LoRA: load `base_model_name_or_path`, apply the adapter.
  * anything else (full ckpt dir / HF id) -> loaded directly as a causal LM.

Usage:
  CUDA_VISIBLE_DEVICES=1 python infer.py --ckpt experiments/train3d/lora_v4/final \
      --input experiments/dataset_z2c_val --out experiments/train3d/lora_v4/preds_full
"""
import argparse
import json
import os
import sys
import glob
import time
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # siblings
import train_sft                      # load_model, chat_ids, PROMPT (single source of truth)
from train_sft import iter_batched_generate   # batched decode (infer.py; in-training eval uses the native Trainer loop)
from serialize import serialize_3d, CANON_QUANT
from eval_cq import _shape_from_globals, imap_isolated


# --------------------------------------------------------------- model loading
def load_infer_model(ckpt, dtype=torch.bfloat16):
    """Return (model, tokenizer). A dir with adapter_config.json is a PEFT adapter on top
    of its recorded base; otherwise `ckpt` is loaded directly."""
    adapter_cfg = os.path.join(ckpt, "adapter_config.json")
    if os.path.isdir(ckpt) and os.path.exists(adapter_cfg):
        base = json.load(open(adapter_cfg))["base_model_name_or_path"]
        model, _ = train_sft.load_model(base, dtype)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ckpt)
        tok_src = base
    else:
        model, _ = train_sft.load_model(ckpt, dtype)
        tok_src = ckpt
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model.eval()
    model.config.use_cache = True
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model, tok


# --------------------------------------------------------------- isolated exec -> STEP
def _export_one(code, out_step, q):
    """Child process: exec predicted code, export the result solid to STEP."""
    out = {"exec_ok": False, "step_written": False, "error_type": None}
    try:
        import cadquery as cq
        ns = {"__name__": "__main__", "cq": cq}
        exec(code, ns)
        out["exec_ok"] = True
        shape = _shape_from_globals(ns)
        if shape is None:
            out["error_type"] = "no_result_object"; q.put(out); return
        cq.exporters.export(shape, out_step)
        out["step_written"] = True
        out["error_type"] = "ok"
    except Exception as e:
        et, msg = type(e).__name__, str(e)
        for key in ("GC_MakeArcOfCircle", "Standard_ConstructionError",
                    "BRep_API", "StdFail_NotDone"):
            if key in msg:
                et = f"Kernel:{key}"; break
        out["error_type"] = et
        out["trace"] = traceback.format_exc()[-400:]
    q.put(out)


# --------------------------------------------------------------- io helpers
def collect_inputs(path):
    if os.path.isfile(path):
        return [path]
    # prefer the *.graph.json convention; fall back to bare *.json only if a dir has none
    # (avoids sweeping in sidecar metadata like `_build_results.json`).
    files = glob.glob(os.path.join(path, "*.graph.json")) or glob.glob(os.path.join(path, "*.json"))
    return sorted(files)


def stem_of(path):
    b = os.path.basename(path)
    for suf in (".graph.json", ".json"):
        if b.endswith(suf):
            return b[:-len(suf)]
    return os.path.splitext(b)[0]


# --------------------------------------------------------------- main
@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="LoRA adapter dir / full ckpt dir / HF id")
    parser.add_argument("--input", required=True, help="one AMVDG JSON, or a dir of *.graph.json/*.json")
    parser.add_argument("--out", required=True, help="output dir for {stem}.py + {stem}.step")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--prompt", default=train_sft.PROMPT,
                    help='instruction prepended to the graph text (--prompt "" to drop it)')
    parser.add_argument("--timeout", type=float, default=30.0, help="per-sample exec timeout (s)")
    parser.add_argument("--batch-size", type=int, default=16, help="max prompts per generate() call")
    parser.add_argument("--max-batch-tokens", type=int, default=48000,
                        help="cap on padded tokens (longest×count) per batch — bounds decode memory. "
                             "Measured on a 24 GB A5000 (2B bf16 + LoRA, new=1024): peak ≈ 4.3 GB + "
                             "0.21 MB/token, i.e. 48000→~14.6 GB (safe default), 64000→~18 GB. Raise "
                             "toward 64000 only if the GPU is dedicated.")
    parser.add_argument("--workers", type=int, default=0,
                        help="parallel exec workers (0 = auto = min(8, cpu_count))")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--quant", type=int, default=CANON_QUANT,
                        help=f"coordinate quantization levels (default {CANON_QUANT}); "
                             f"MUST match the value build_dataset used for the training "
                             f"data, else train/inference coordinate systems diverge. "
                             f"0 = off.")
    args = parser.parse_args()

    inputs = collect_inputs(args.input)
    if args.limit:
        inputs = inputs[:args.limit]
    os.makedirs(args.out, exist_ok=True)
    model, tok = load_infer_model(args.ckpt)
    tok.padding_side = "left"          # decoder-only: left-pad so gen starts at a shared offset
    dev = next(model.parameters()).device
    workers = args.workers or min(8, os.cpu_count() or 1)
    print(f"loaded {args.ckpt} | {len(inputs)} inputs -> {args.out}", file=sys.stderr)

    # ---- pass 1: serialize + tokenize every prompt (collect serialize failures as rows) ----
    prompts, serialize_err, t0 = [], {}, time.time()
    for path in inputs:
        stem = stem_of(path)
        try:
            text = serialize_3d(json.load(open(path)), quant=args.quant)
        except Exception as e:
            serialize_err[stem] = f"serialize:{type(e).__name__}"
            continue
        user = f"{args.prompt}\n\n{text}" if args.prompt else text
        ids = train_sft.chat_ids(tok, [{"role": "user", "content": user}], True)  # list[int]
        prompts.append((stem, ids))
    prompts.sort(key=lambda x: len(x[1]))     # length-sorted → minimal padding within a batch

    # ---- pass 2: batched greedy generation on the single GPU (shared with the eval hook) ----
    codes = {}
    n_gen = 0
    for stem, code in iter_batched_generate(model, tok, prompts, args.max_new_tokens,
                                            args.batch_size, args.max_batch_tokens, dev):
        codes[stem] = code
        with open(os.path.join(args.out, f"{stem}.py"), "w") as f:   # .py ALWAYS written
            f.write(code)
        n_gen += 1
        if n_gen % args.batch_size == 0 or n_gen == len(prompts):
            print(f"  [gen {n_gen}/{len(prompts)}] {time.time()-t0:.0f}s", file=sys.stderr)

    # ---- pass 3: isolated exec -> {stem}.step, parallel across CPU workers ----
    exec_tasks = [(stem, (code, os.path.join(args.out, f"{stem}.step")))
                  for stem, code in codes.items()]
    exec_res, done = {}, 0
    print(f"exec {len(exec_tasks)} preds with {workers} workers", file=sys.stderr)
    for stem, res in imap_isolated(exec_tasks, _export_one, args.timeout, workers):
        exec_res[stem] = res if res is not None else {
            "exec_ok": False, "step_written": False, "error_type": "timeout"}
        done += 1
        if done % 20 == 0:
            print(f"  [exec {done}/{len(exec_tasks)}] {time.time()-t0:.0f}s", file=sys.stderr)

    # ---- assemble rows in original input order ----
    rows, err_hist = [], {}
    n_exec_ok = n_step = 0
    for path in inputs:
        stem = stem_of(path)
        if stem in serialize_err:
            et = serialize_err[stem]
            rows.append({"id": stem, "gen": False, "error_type": et})
            err_hist["serialize_error"] = err_hist.get("serialize_error", 0) + 1
            continue
        res = exec_res.get(stem, {"exec_ok": False, "step_written": False,
                                  "error_type": "crash_no_result"})
        n_exec_ok += res["exec_ok"]; n_step += res["step_written"]
        err_hist[res["error_type"]] = err_hist.get(res["error_type"], 0) + 1
        rows.append({"id": stem, "gen": True, "exec_ok": res["exec_ok"],
                     "step_written": res["step_written"], "error_type": res["error_type"]})

    n = len(inputs)
    summary = {"n": n, "gen_n": sum(r["gen"] for r in rows),
               "exec_ok": n_exec_ok, "exec_ok_rate": round(n_exec_ok / n, 4) if n else 0,
               "step_written": n_step, "step_rate": round(n_step / n, 4) if n else 0,
               "error_hist": dict(sorted(err_hist.items(), key=lambda x: -x[1])),
               "runtime_s": round(time.time() - t0, 1), "ckpt": args.ckpt, "input": args.input}
    with open(os.path.join(args.out, "infer_summary.json"), "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
