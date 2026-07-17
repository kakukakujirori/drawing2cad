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
    python infer.py \
        --ckpt experiments/train3d/[yyyy-mm-dd_hh-mm-ss]/best \
        --input experiments/dataset_z2c_val \
        --out experiments/train3d/[yyyy-mm-dd_hh-mm-ss]/preds_full
"""
import argparse
import gc
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
from serialize import serialize_3d, CANON_QUANT, coordinate_tokenize_text, coordinate_quant_from_vocab
from eval_cq import _shape_from_globals, imap_isolated


# --------------------------------------------------------------- model loading
def _tok_source(ckpt: str) -> str:
    """Tokenizer/base source for `ckpt`: the LoRA base if it's an adapter dir, else `ckpt`."""
    adapter_cfg = os.path.join(ckpt, "adapter_config.json")
    if os.path.isdir(ckpt) and os.path.exists(adapter_cfg):
        # Coordinate-token adapters save their expanded tokenizer alongside the adapter.
        if any(os.path.exists(os.path.join(ckpt, name)) for name in
               ("tokenizer.json", "tokenizer_config.json", "added_tokens.json")):
            return ckpt
        return json.load(open(adapter_cfg))["base_model_name_or_path"]
    return ckpt


def load_tokenizer(ckpt: str):
    """Load only the tokenizer (cheap) so prompts can be sized before the weights are loaded."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(_tok_source(ckpt), trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_infer_model(ckpt: str, dtype: torch.dtype = torch.bfloat16, device_map=None):
    """Return (model, tokenizer). A dir with adapter_config.json is a PEFT adapter on top
    of its recorded base; otherwise `ckpt` is loaded directly.

    device_map=None -> whole model on cuda:0 (fast, the common path). device_map="auto" ->
    accelerate shards the layers across every visible GPU so a prompt too large for one GPU
    still fits (weights + KV cache spread across their combined memory)."""
    adapter_cfg = os.path.join(ckpt, "adapter_config.json")
    tok = load_tokenizer(ckpt)
    if os.path.isdir(ckpt) and os.path.exists(adapter_cfg):
        base = json.load(open(adapter_cfg))["base_model_name_or_path"]
        model, _ = train_sft.load_model(base, dtype, device_map=device_map)
        if model.get_input_embeddings().num_embeddings != len(tok):
            model.resize_token_embeddings(len(tok))
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ckpt)
    else:
        model, _ = train_sft.load_model(ckpt, dtype, device_map=device_map)
    model.eval()
    model.config.use_cache = True
    if device_map is None and torch.cuda.is_available():
        model = model.to("cuda")          # sharded models are already placed by accelerate
    return model, tok


# ---- GPU memory model (measured on a 24 GB A5000, this 2B bf16 + LoRA ckpt, sdpa): generate()
# peak ≈ MODEL_GB + MB_PER_TOKEN/1000 × padded prompt tokens in the batch. One GPU tops out near
# 82k tokens; longer prompts need the model sharded over GPUs (load_infer_model device_map="auto").
MB_PER_TOKEN = 0.214      # slope of generate() peak vs prompt length
MODEL_GB = 4.5            # resident weights (2B bf16 + LoRA)
MEM_SAFE = 0.85           # fraction of free VRAM a batch may use (headroom for spikes)


def _free_gb(dev: int) -> float:
    return torch.cuda.mem_get_info(dev)[0] / 1e9


def _tokens_for_gb(gb: float) -> int:
    """How many padded prompt tokens of generate() peak fit in `gb` GB."""
    return max(0, int(gb * 1000 / MB_PER_TOKEN))


def _model_devices(model) -> list[int]:
    """CUDA ordinals the model occupies (one entry unless device_map sharded it)."""
    dm = getattr(model, "hf_device_map", None)
    if dm:
        return sorted({d for d in dm.values() if isinstance(d, int)})
    d = next(model.parameters()).device
    return [d.index if d.index is not None else 0]


def _input_device(model) -> torch.device:
    """Device generate() expects its input_ids on (the input-embedding shard)."""
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


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
        for key in ("GC_MakeArcOfCircle", "Standard_ConstructionError", "BRep_API", "StdFail_NotDone"):
            if key in msg:
                et = f"Kernel:{key}"; break
        out["error_type"] = et
        out["error_msg"] = msg[:300]        # persisted per-row + surfaced on stderr (was dropped)
        out["trace"] = traceback.format_exc()[-800:]
    q.put(out)


# --------------------------------------------------------------- io helpers
def collect_inputs(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    # prefer the *.graph.json convention; fall back to bare *.json only if a dir has none
    # (avoids sweeping in sidecar metadata like `_build_results.json`).
    files = glob.glob(os.path.join(path, "*.graph.json")) or glob.glob(os.path.join(path, "*.json"))
    return sorted(files)


def stem_of(path: str) -> str:
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
    parser.add_argument("--max-new-tokens", type=int, default=train_sft.DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--prompt", default=train_sft.PROMPT,
                    help='instruction prepended to the graph text (--prompt "" to drop it)')
    parser.add_argument("--timeout", type=float, default=30.0, help="per-sample exec timeout (s)")
    parser.add_argument("--batch-size", type=int, default=16, help="max prompts per generate() call")
    parser.add_argument("--max-batch-tokens", type=int, default=0,
                        help="cap on padded tokens (longest×count) per generate() batch. "
                             "0 = auto from free VRAM (peak ≈ 0.214 MB/token, measured). A batch "
                             "that OOMs is halved and retried; a solo prompt that still won't fit "
                             "is recorded as an oom_generate failure (attempted, never skipped).")
    parser.add_argument("--shard", choices=("auto", "on", "off"), default="auto",
                        help="spread the model across all visible GPUs (device_map=auto) so prompts "
                             "too long for one GPU still run. auto = shard only the prompts that "
                             "exceed one GPU's ~82k-token capacity, and only if >1 GPU is visible.")
    parser.add_argument("--workers", type=int, default=0,
                        help="parallel exec workers (0 = auto = min(8, cpu_count))")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--quant", type=int, default=CANON_QUANT,
                        help=f"signed coordinate-grid magnitude bins per sign "
                             f"(default {CANON_QUANT}, yielding "
                             f"[-{CANON_QUANT - 1}, {CANON_QUANT - 1}]); "
                             f"MUST match the value build_dataset used for the training "
                             f"data, else train/inference coordinate systems diverge. "
                             f"0 = off.")
    parser.add_argument("--coord-tokens", choices=("auto", "on", "off"), default="auto",
                        help="coordinate-token serialization. auto (default) detects the expanded "
                             "checkpoint tokenizer; on/off are diagnostic overrides")
    args = parser.parse_args()

    inputs = collect_inputs(args.input)
    if args.limit:
        inputs = inputs[:args.limit]
    os.makedirs(args.out, exist_ok=True)
    workers = args.workers or min(8, os.cpu_count() or 1)
    n_gpu = torch.cuda.device_count()
    tok = load_tokenizer(args.ckpt)         # tokenizer first, so prompts can be sized pre-load
    tok.padding_side = "left"               # decoder-only: left-pad so gen starts at a shared offset
    coord_quant = coordinate_quant_from_vocab(tok.get_vocab())
    use_coord_tokens = coord_quant is not None if args.coord_tokens == "auto" else args.coord_tokens == "on"
    if use_coord_tokens and coord_quant is None:
        raise ValueError("--coord-tokens on requires a checkpoint tokenizer with coordinate tokens")
    if use_coord_tokens and args.quant != coord_quant:
        raise ValueError(
            f"coordinate-token checkpoint uses quant={coord_quant}, but infer received "
            f"--quant {args.quant}")
    print(f"loaded tokenizer for {args.ckpt} | {len(inputs)} inputs -> {args.out}", file=sys.stderr)

    # ---- pass 1: serialize + tokenize every prompt (collect serialize failures as rows) ----
    prompts, serialize_err, t0 = [], {}, time.time()
    for path in inputs:
        stem = stem_of(path)
        try:
            text = serialize_3d(json.load(open(path)), quant=args.quant)
            if use_coord_tokens:
                text = coordinate_tokenize_text(text)
        except Exception as e:
            serialize_err[stem] = f"serialize:{type(e).__name__}"
            continue
        user = f"{args.prompt}\n\n{text}" if args.prompt else text
        ids = train_sft.chat_ids(tok, [{"role": "user", "content": user}], True)  # list[int]
        prompts.append((stem, ids))
    prompts.sort(key=lambda x: len(x[1]))     # length-sorted → minimal padding within a batch

    # ---- pass 2: greedy generation. Prompts that fit on one GPU (<= cap1) run there; longer
    #      ones run through a model sharded across every visible GPU so they still fit. ----
    max_len = max((len(ids) for _, ids in prompts), default=0)
    cap1 = _tokens_for_gb((_free_gb(0) - MODEL_GB) * MEM_SAFE) if torch.cuda.is_available() else max_len
    can_shard = n_gpu > 1 and args.shard != "off"
    normal = [p for p in prompts if len(p[1]) <= cap1]
    oversize = [p for p in prompts if len(p[1]) > cap1]
    if oversize and not can_shard:
        print(f"[infer] {len(oversize)} prompt(s) exceed one GPU's ~{cap1} tok capacity and "
              f"sharding is off/unavailable ({n_gpu} GPU, --shard {args.shard}); they will be "
              f"attempted and may OOM-fail.", file=sys.stderr)

    codes, gen_fail = {}, {}

    def _run_phase(subset, device_map, label: str):
        if not subset:
            return
        model, mtok = load_infer_model(args.ckpt, device_map=device_map)
        mtok.padding_side = "left"
        dev = _input_device(model)
        devs = _model_devices(model)
        budget = args.max_batch_tokens or _tokens_for_gb(sum(_free_gb(d) for d in devs) * MEM_SAFE)
        print(f"[infer] phase {label}: {len(subset)} prompt(s) | GPU {devs} | "
              f"batch budget ~{budget} tok", file=sys.stderr)
        n = 0
        for stem, code in iter_batched_generate(model, mtok, subset, args.max_new_tokens,
                                                args.batch_size, budget, dev):
            if code is None:                    # generation OOM'd even solo (see OOM guard)
                gen_fail[stem] = "oom_generate"
            else:
                codes[stem] = code
                with open(os.path.join(args.out, f"{stem}.py"), "w") as f:
                    f.write(code)
            n += 1
            if n % args.batch_size == 0 or n == len(subset):
                print(f"  [gen {label} {n}/{len(subset)}] {time.time()-t0:.0f}s", file=sys.stderr)
        del model, mtok
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not oversize:                            # common case: a single single-GPU pass
        _run_phase(normal, None, "1-gpu")
    elif not can_shard:                         # can't shard -> one pass, OOM-guarded
        _run_phase(normal + oversize, None, "1-gpu")
    else:                                       # fast bulk on 1 GPU, then the tail sharded
        _run_phase(normal, None, "1-gpu")
        _run_phase(oversize, "auto", "sharded")

    # ---- pass 3: isolated exec -> {stem}.step, parallel across CPU workers ----
    exec_tasks = [(stem, (code, os.path.join(args.out, f"{stem}.step")))
                  for stem, code in codes.items()]
    exec_res, done, err_sample = {}, 0, {}
    print(f"exec {len(exec_tasks)} preds with {workers} workers", file=sys.stderr)
    for stem, res in imap_isolated(exec_tasks, _export_one, args.timeout, workers):
        exec_res[stem] = res if res is not None else {
            "exec_ok": False, "step_written": False, "error_type": "timeout"}
        et = exec_res[stem].get("error_type")
        if et not in ("ok", None) and et not in err_sample:   # keep 1 msg+trace per failure class
            err_sample[et] = (stem, exec_res[stem].get("error_msg", ""),
                              exec_res[stem].get("trace", ""))
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
        if stem in gen_fail:                          # generation attempted but OOM'd (never skipped)
            et = gen_fail[stem]
            rows.append({"id": stem, "gen": False, "error_type": et})
            err_hist[et] = err_hist.get(et, 0) + 1
            continue
        res = exec_res.get(stem, {"exec_ok": False, "step_written": False,
                                  "error_type": "crash_no_result"})
        n_exec_ok += res["exec_ok"]; n_step += res["step_written"]
        err_hist[res["error_type"]] = err_hist.get(res["error_type"], 0) + 1
        row = {"id": stem, "gen": True, "exec_ok": res["exec_ok"],
               "step_written": res["step_written"], "error_type": res["error_type"]}
        if res.get("error_msg"):
            row["error_msg"] = res["error_msg"]
        rows.append(row)

    n = len(inputs)
    summary = {"n": n, "gen_n": sum(r["gen"] for r in rows),
               "exec_ok": n_exec_ok, "exec_ok_rate": round(n_exec_ok / n, 4) if n else 0,
               "step_written": n_step, "step_rate": round(n_step / n, 4) if n else 0,
               "error_hist": dict(sorted(err_hist.items(), key=lambda x: -x[1])),
               "runtime_s": round(time.time() - t0, 1), "ckpt": args.ckpt, "input": args.input}
    with open(os.path.join(args.out, "infer_summary.json"), "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)
    print(json.dumps(summary, indent=2))

    # ---- surface failures on stderr: one representative message per error class ----
    for et, (stem, msg, _tr) in err_sample.items():
        print(f"[infer] error '{et}' (e.g. {stem}): {msg}", file=sys.stderr)

    # A run that generated code but produced ZERO execs is almost always an
    # environment fault (e.g. CadQuery/OCP import), not the model. Shout, dump a
    # full sample trace, and exit non-zero so callers (run_ours.py, CI) can't
    # mistake a broken env for a benign "0 fixtures" result.
    n_gen = summary["gen_n"]
    if n_gen and n_exec_ok == 0:
        top_et = max(err_hist.items(), key=lambda kv: kv[1])[0]
        _, top_msg, top_tr = err_sample.get(top_et, ("", "", ""))
        env_like = (top_et in ("ImportError", "ModuleNotFoundError", "OSError")
                    or any(k in (top_msg or "").lower()
                           for k in ("import", "undefined symbol", "cannot open shared")))
        why = ("ENVIRONMENT fault (e.g. CadQuery/OCP import) — NOT the model"
               if env_like else "every exec failed — check the generated code / geometry")
        print("\n" + "=" * 72, file=sys.stderr)
        print(f"[infer] ALL {n_gen} exec attempts FAILED, 0 STEP written -> {why}.", file=sys.stderr)
        print(f"[infer] dominant error: {top_et} ({err_hist[top_et]}/{n_gen})", file=sys.stderr)
        if top_msg:
            print(f"[infer] message: {top_msg}", file=sys.stderr)
        if top_tr:
            print(f"[infer] sample trace:\n{top_tr}", file=sys.stderr)
        print("=" * 72, file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
