"""train_sft.py — text-only SFT for the AMVDG->CadQuery (AMVDG->3D) leg.

Input  = g2 serialization of an AMVDG graph (text only; NO drawing image — this is the
         experiment that tests whether the IR is *sufficient*).
Output = CadQuery Python code (GT = Zero-to-CAD cadquery_file).
Start  = ADSKAILab/Zero-To-CAD-Qwen3-VL-2B (image->CadQuery SFT; same base + output as us),
         swappable via --ckpt (loads exactly that ckpt — no silent fallback to another model).

Recipe (see scripts/train3d/README.md for the cadrille / Zero-to-CAD diff table):
  * bf16, completion-only loss (mask the prompt, train only the answer span),
  * LoRA (default) or full fine-tuning (--full), vision tower frozen (text-only),
  * single-GPU (`python train_sft.py ...`) or DDP (`torchrun --nproc_per_node=2 ...` /
    `accelerate launch ...`) — HF Trainer picks up the world size automatically,
  * over-length pairs are FILTERED, never truncated (truncation would corrupt the g2
    input or cut the target); the dropped count is reported.
  * eval hook: greedy-generate a few val graphs, then shell out to eval_cq.py in the
    CadQuery env for validity + translation-aligned voxel IoU + bbox-mm error.

Smoke: python train_sft.py --smoke  (LoRA, ~40 steps, one A5000, overfits by design).
"""
import argparse
import importlib
import json
import os
import sys
import subprocess
import time

import torch
from torch.utils.data import Dataset
from transformers import (AutoConfig, AutoTokenizer, AutoModelForCausalLM,
                          Trainer, TrainingArguments, TrainerCallback)

DEFAULT_CKPT = "ADSKAILab/Zero-To-CAD-Qwen3-VL-2B"

# Task instruction prepended to the graph text in the user turn. cadrille uses the 3-word
# "Generate cadquery code"; the chat template's assistant-turn start is the code-start marker,
# so no custom special token is added. Override with --prompt (e.g. --prompt "" to ablate).
PROMPT = "Generate cadquery code from this multi-view drawing graph; assign the solid to `result`."
DRAWING2CAD_PY = os.environ.get(
    "DRAWING2CAD_PY", "/home/ryotaro/miniforge3/envs/drawing2cad/bin/python")
EVAL_CQ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_cq.py")


# --------------------------------------------------------------- model loading
def load_model(ckpt, dtype):
    """Load exactly `ckpt` as a causal LM.
    Qwen3-VL loads text-only fine (no images)."""
    kw = dict(torch_dtype=dtype, attn_implementation="sdpa", trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(ckpt, trust_remote_code=True)
    arch = (cfg.architectures or [""])[0]
    if "ForConditionalGeneration" in arch:
        mod = importlib.import_module("transformers")
        model = getattr(mod, arch).from_pretrained(ckpt, **kw)
    else:
        model = AutoModelForCausalLM.from_pretrained(ckpt, **kw)
    return model, ckpt


def chat_ids(tok, msgs, gen_prompt, return_tensors=None):
    """apply_chat_template across transformers versions: 4.x returns a bare id list,
    5.x returns a BatchEncoding (and nests the list per conversation)."""
    out = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=gen_prompt,
                                  return_tensors=return_tensors)
    if hasattr(out, "keys"):
        out = out["input_ids"]
    if return_tensors is None and len(out) and isinstance(out[0], list):
        out = out[0]
    return out


def build_labels(tok, input_text, target_code, max_len):
    """Tokenize one conversation; mask everything but the assistant answer (completion-only).
    Returns (input_ids, labels) or None if it exceeds max_len (filter, don't truncate)."""
    user = f"{PROMPT}\n\n{input_text}" if PROMPT else input_text
    msgs_p = [{"role": "user", "content": user}]
    msgs_f = msgs_p + [{"role": "assistant", "content": target_code}]
    pids = chat_ids(tok, msgs_p, True)
    fids = chat_ids(tok, msgs_f, False)
    if len(fids) > max_len:
        return None
    # locate answer span: prompt is a prefix of the full sequence
    p = len(pids)
    if fids[:p] != pids:                      # tokenizer edge case -> substring search
        p = 0
        for i in range(len(fids) - len(pids) + 1):
            if fids[i:i + len(pids)] == pids:
                p = i + len(pids); break
    labels = [-100] * p + fids[p:]
    return fids, labels


class SFTDataset(Dataset):
    def __init__(self, records: list[dict[str, str]], tok, max_len: int):
        self.ex, self.dropped = [], 0
        for r in records:
            built = build_labels(tok, r["input_text"], r["target_code"], max_len)
            if built is None:
                self.dropped += 1; continue
            ids, labels = built
            self.ex.append({"input_ids": ids, "labels": labels})

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i: int):
        return self.ex[i]


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        m = max(len(b["input_ids"]) for b in batch)
        ii, ll, am = [], [], []
        for b in batch:
            n = m - len(b["input_ids"])
            ii.append(b["input_ids"] + [self.pad_id] * n)
            ll.append(b["labels"] + [-100] * n)
            am.append([1] * len(b["input_ids"]) + [0] * n)
        return {"input_ids": torch.tensor(ii), "labels": torch.tensor(ll),
                "attention_mask": torch.tensor(am)}


# --------------------------------------------------------------- eval hook
@torch.no_grad()
def generate_preds(model, tok, records, out_dir, max_new_tokens):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    prev_cache = model.config.use_cache
    model.config.use_cache = True            # training left this False (grad-ckpt); KV cache = fast decode
    dev = next(model.parameters()).device
    for r in records:
        user = f"{PROMPT}\n\n{r['input_text']}" if PROMPT else r["input_text"]
        ids = chat_ids(tok, [{"role": "user", "content": user}], True,
                       return_tensors="pt").to(dev)
        if ids.shape[1] > 8192:            # skip pathologically long inputs in the probe
            continue
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                             use_cache=True, pad_token_id=tok.pad_token_id)
        txt = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        with open(os.path.join(out_dir, f"{r['id']}.py"), "w") as f:
            f.write(txt)
    model.config.use_cache = prev_cache
    model.train()


def run_eval_cq(pred_dir: str, gt_dir: str, out_json: str, limit: int = 0):
    """Shell out to eval_cq.py in the CadQuery env; return the aggregate dict or None."""
    if not os.path.exists(DRAWING2CAD_PY):
        print(f"[eval] skip: {DRAWING2CAD_PY} not found", file=sys.stderr)
        return None
    cmd = [DRAWING2CAD_PY, EVAL_CQ, "--pred-dir", pred_dir, "--gt-dir", gt_dir,
           "--out", out_json]
    if limit:
        cmd += ["--limit", str(limit)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        print(r.stdout[-1500:]); print(r.stderr[-500:], file=sys.stderr)
        return json.load(open(out_json)).get("aggregate") if os.path.exists(out_json) else None
    except Exception as e:
        print(f"[eval] eval_cq failed: {e}", file=sys.stderr)
        return None


class EvalCadCallback(TrainerCallback):
    def __init__(self, model, tok, val_records, work_dir, gt_dir, max_new_tokens, n_eval):
        self.model, self.tok = model, tok
        self.val = val_records[:n_eval]
        self.work_dir, self.gt_dir = work_dir, gt_dir
        self.max_new_tokens = max_new_tokens

    def _do(self, state, tag):
        if not self.val:
            return
        pdir = os.path.join(self.work_dir, f"preds_{tag}")
        generate_preds(self.model, self.tok, self.val, pdir, self.max_new_tokens)
        agg = run_eval_cq(pdir, self.gt_dir,
                          os.path.join(self.work_dir, f"eval_{tag}.json"))
        if agg:
            print(f"[eval @ {tag}] valid_rate={agg['valid_rate']} "
                  f"mean_iou={agg['mean_iou']} median_iou={agg['median_iou']} "
                  f"mean_bbox_err_mm={agg['mean_max_bbox_err_mm']}")

    def on_train_end(self, args, state, control, **kw):
        self._do(state, f"step{int(state.global_step)}")


# --------------------------------------------------------------- main
def main():
    global PROMPT  # --prompt overrides it; build_labels/generate_preds read the module global
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="experiments/data_z2c_train",
                        help="train bundle: a dir with all.jsonl, or a jsonl path")
    parser.add_argument("--val", default="experiments/data_z2c_val",
                        help="val bundle (Zero-To-CAD val source split): dir with all.jsonl, or jsonl")
    parser.add_argument("--gt-dir", default="experiments/stage_z2c_val",
                        help="GT STEP dir for the val split (eval hook execs preds against these)")
    parser.add_argument("--prompt", default=PROMPT,
                        help='task instruction prepended to the graph text (--prompt "" to drop it)')
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--out", default="scripts/train3d/runs/sft")
    parser.add_argument("--full", action="store_true", help="full fine-tune (default: LoRA)")
    parser.add_argument("--train-vision", action="store_true", help="don't freeze vision tower")
    parser.add_argument("--max-len", type=int, default=8192)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=None, help="default 2e-4 LoRA / 1e-4 full")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--optim", default="adamw_torch",
                    help="adamw_torch (default) | adafactor (fits full-FT of 2B on one 24GB A5000)")
    parser.add_argument("--no-grad-ckpt", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--n-eval", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0, help="cap train records (smoke/overfit)")
    parser.add_argument("--smoke", action="store_true",
                    help="LoRA, 40 steps, max-len 3072, limit 24 — pipe/loss sanity")
    args = parser.parse_args()
    PROMPT = args.prompt

    if args.smoke:
        args.max_steps = args.max_steps if args.max_steps > 0 else 40
        args.max_len = min(args.max_len, 3072)
        args.limit = args.limit or 24
        args.n_eval = min(args.n_eval, 4)
    lr = args.lr if args.lr is not None else (1e-4 if args.full else 2e-4)

    def load_jsonl(path: str) -> list[dict]:
        if os.path.isdir(path):
            path = os.path.join(path, "all.jsonl")
        return [json.loads(l) for l in open(path)]

    train = load_jsonl(args.train)
    val = load_jsonl(args.val)
    if args.limit:
        train = sorted(train, key=lambda r: r["n_tok_total"])[:args.limit]  # short = fast smoke

    tok = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    ds = SFTDataset(train, tok, args.max_len)
    print(f"train examples kept {len(ds)} (dropped over-len {ds.dropped}) "
          f"| val {len(val)} | max_len {args.max_len}")

    model, used = load_model(args.ckpt, torch.bfloat16)
    print(f"loaded {used}")
    if not args.train_vision:
        nfz = 0
        for n, p in model.named_parameters():
            if ".visual." in n or n.startswith("visual."):
                p.requires_grad_(False); nfz += 1
        print(f"froze {nfz} vision params")
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    if not args.full:
        from peft import LoraConfig, get_peft_model
        lc = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                        "gate_proj", "up_proj", "down_proj"],
                        task_type="CAUSAL_LM")
        if not args.no_grad_ckpt:
            model.enable_input_require_grads()
        model = get_peft_model(model, lc)
        model.print_trainable_parameters()

    world = int(os.environ.get("WORLD_SIZE", "1"))
    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=False,  # enabled manually above (PEFT input-grad handshake)
        # LoRA adapters land on the frozen vision attn too (never run text-only) -> DDP
        # would flag them as unused; allow it for the 2-GPU path (no-op single-GPU).
        ddp_find_unused_parameters=(world > 1),
        dataloader_num_workers=2, optim=args.optim)

    # probe on the shortest val graphs (fast + fits memory; full eval = eval_cq on all preds)
    val_probe = sorted(val, key=lambda r: r["n_tok_total"])
    cb = EvalCadCallback(model, tok, val_probe, args.out, args.gt_dir,
                         args.max_new_tokens, args.n_eval)
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=Collator(tok.pad_token_id),
        callbacks=[cb],
    )

    t0 = time.time()
    out = trainer.train()
    print(f"train_runtime {time.time()-t0:.0f}s  final_loss {out.training_loss:.4f}")
    if hasattr(trainer.state, "log_history"):
        losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
        if losses:
            print(f"loss[0]={losses[0]:.4f} loss[-1]={losses[-1]:.4f} "
                  f"(min {min(losses):.4f})")
    trainer.save_model(os.path.join(args.out, "final"))


if __name__ == "__main__":
    main()
