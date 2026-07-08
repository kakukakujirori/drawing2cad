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
  * eval hook: periodically (every `eval_every_epochs`, plus at train end) greedy-generate
    a FIXED, seeded RANDOM val subset (`eval_val_n`, default 48; 0 = full val) — batched via
    the SAME code path as infer.py (iter_batched_generate: length-sort + left-pad) — then shell
    out to eval_cq.py in the CadQuery env for validity + translation-aligned voxel IoU + bbox-mm
    error. A folded scalar (`best_metric`, default `mean_iou_incl_fail` = mean IoU over the whole
    subset counting failed/invalid preds as 0) is logged to TensorBoard and drives best-model
    checkpointing to `<out>/best/` (+ `<out>/best_meta.json`). rank0-only (DDP-safe).

Smoke: python train_sft.py --smoke  (LoRA, ~40 steps, one A5000, overfits by design).

Config: args are an `ExpConfig` dataclass parsed by HfArgumentParser, so every CLI flag
(dash or underscore form) can also be set from a JSON/YAML file via `--config run.yaml`
(CLI flags override file values). The fully resolved config is dumped to `<out>/config.json`
and metrics stream to TensorBoard under `<out>/logs` (or wandb with `--report-to wandb`).
"""
import importlib
import json
import os
import sys
import subprocess
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import (AutoConfig, AutoTokenizer, AutoModelForCausalLM,
                          HfArgumentParser, Trainer, TrainingArguments,
                          TrainerCallback, set_seed)

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


# ------------------------------------------------ shared batched generation
# Single source of truth for length-sorted, left-padded batched greedy decode.
# Used BOTH by infer.py (dedicated-GPU batch inference) and the in-training eval
# hook below, so the two never drift. The caller owns model/tokenizer state
# (model.eval(), use_cache=True, tok.padding_side="left", grad-ckpt disabled) —
# this function only forms batches and decodes.
def _make_batches(items, batch_size, max_tokens):
    """items: (stem, ids) pairs, sorted by len(ids) asc. Greedy grouping capped by count AND
    by padded-token budget (longest_in_batch × count) so a few very long prompts fall into
    smaller batches instead of blowing up padding/memory. A single over-budget prompt runs solo."""
    batch = []
    for stem, ids in items:
        would = batch + [(stem, ids)]
        longest = max(len(x[1]) for x in would)
        if batch and (len(would) > batch_size or longest * len(would) > max_tokens):
            yield batch
            batch = [(stem, ids)]
        else:
            batch = would
    if batch:
        yield batch


@torch.no_grad()
def iter_batched_generate(model, tok, prompts, max_new_tokens, batch_size, max_batch_tokens,
                          device):
    """prompts: list[(stem, ids_list[int])]. Yields (stem, decoded_code) per prompt.
    Length-sorts, left-pad-batches (_make_batches), greedy-decodes on `device`.
    Precondition (caller-set): model.eval(), model.config.use_cache=True,
    tok.padding_side='left', gradient checkpointing disabled."""
    prompts = sorted(prompts, key=lambda x: len(x[1]))
    for batch in _make_batches(prompts, batch_size, max_batch_tokens):
        enc = tok.pad({"input_ids": [ids for _, ids in batch]},
                      padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             use_cache=True, pad_token_id=tok.pad_token_id)
        plen = enc["input_ids"].shape[1]      # left-padded → shared prompt offset for the batch
        for j, (stem, _) in enumerate(batch):
            yield stem, tok.decode(gen[j][plen:], skip_special_tokens=True)


# --------------------------------------------------------------- eval hook
def run_eval_cq(pred_dir: str, gt_dir: str, out_json: str, limit: int = 0):
    """Shell out to eval_cq.py in the CadQuery env; return the loaded JSON dict
    ({"aggregate","config","rows"}) or None. We need per-row data (not just the
    aggregate) so the folded IoU metric can count failed/invalid preds as 0."""
    if not os.path.exists(DRAWING2CAD_PY):
        print(f"[eval] skip: {DRAWING2CAD_PY} not found", file=sys.stderr)
        return None
    cmd = [DRAWING2CAD_PY, EVAL_CQ, "--pred-dir", pred_dir, "--gt-dir", gt_dir,
           "--out", out_json]
    if limit:
        cmd += ["--limit", str(limit)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        print(r.stdout[-1500:]); print(r.stderr[-500:], file=sys.stderr)
        return json.load(open(out_json)) if os.path.exists(out_json) else None
    except Exception as e:
        print(f"[eval] eval_cq failed: {e}", file=sys.stderr)
        return None


def eval_metrics(rows, n_subset):
    """Fold eval_cq per-row output into scalars over the WHOLE subset (n_subset),
    counting failed/invalid/missing preds as IoU 0 — so validity and geometry live
    in one number. `rows` are only the preds eval_cq scored (⊆ subset); any shortfall
    is padded with zeros. Never trust eval_cq's aggregate mean_iou here: it averages
    only scored rows, hiding failures."""
    import statistics
    ious = [(r.get("iou") or 0.0) for r in rows]
    valids = [1.0 if r.get("valid") else 0.0 for r in rows]
    while len(ious) < n_subset:               # unscored/missing preds -> 0
        ious.append(0.0); valids.append(0.0)
    n = max(n_subset, len(ious)) or 1
    return {
        "mean_iou_incl_fail": round(sum(ious) / n, 4),
        "median_iou": round(statistics.median(ious), 4),
        "valid_rate": round(sum(valids) / n, 4),
        "n_scored": len(rows),
        "n_subset": n_subset,
    }


class EvalCadCallback(TrainerCallback):
    """Periodic in-training eval: greedy-generate a fixed val subset (batched, shared
    with infer.py), score via eval_cq.py, log folded metrics to TensorBoard, and keep
    a best-model checkpoint. Runs only on rank0 (DDP-safe); other ranks block at the
    next allreduce until rank0 finishes, so no desync. `trainer` is attached post-init
    (save_model needs it)."""

    def __init__(self, model, tok, val_records, work_dir, gt_dir, max_new_tokens,
                 cap_len, batch_size, max_batch_tokens, eval_every_epochs, best_metric):
        self.model, self.tok = model, tok
        self.val = list(val_records)
        self.work_dir, self.gt_dir = work_dir, gt_dir
        self.max_new_tokens = max_new_tokens
        self.cap_len = cap_len
        self.batch_size = batch_size
        self.max_batch_tokens = max_batch_tokens
        self.eval_every_epochs = float(eval_every_epochs)
        self.best_metric = best_metric
        self.trainer = None
        self.best_value = None
        self.best_step = self.best_epoch = None
        self.history = []
        self.last_eval_step = -1
        self.last_eval_epoch = 0.0

    def _generate(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        m, tok = self.model, self.tok
        was_training = m.training
        prev_cache = m.config.use_cache
        prev_pad = tok.padding_side
        gc_on = bool(getattr(m, "is_gradient_checkpointing", False))
        m.eval()
        if gc_on:                              # use_cache is silently forced off under grad-ckpt
            m.gradient_checkpointing_disable()
        m.config.use_cache = True
        tok.padding_side = "left"
        dev = next(m.parameters()).device
        prompts = []
        for r in self.val:
            user = f"{PROMPT}\n\n{r['input_text']}" if PROMPT else r["input_text"]
            ids = chat_ids(tok, [{"role": "user", "content": user}], True)
            if len(ids) > self.cap_len:        # over-cap: stub .py so it scores as a failure
                with open(os.path.join(out_dir, f"{r['id']}.py"), "w") as f:
                    f.write(f"# prompt too long: {len(ids)} tok > cap {self.cap_len}\n")
                continue
            prompts.append((r["id"], ids))
        try:
            for stem, code in iter_batched_generate(
                    m, tok, prompts, self.max_new_tokens, self.batch_size,
                    self.max_batch_tokens, dev):
                with open(os.path.join(out_dir, f"{stem}.py"), "w") as f:
                    f.write(code)
        finally:                               # always restore training state
            tok.padding_side = prev_pad
            m.config.use_cache = prev_cache
            if gc_on:
                m.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
            if was_training:
                m.train()

    def _do(self, state):
        if not self.val:
            return
        step = int(state.global_step)
        if step == self.last_eval_step:        # avoid a duplicate at train-end
            return
        self.last_eval_step = step
        epoch = round(float(state.epoch or 0.0), 4)
        tag = f"step{step}"
        pdir = os.path.join(self.work_dir, f"preds_{tag}")
        t0 = time.time()
        self._generate(pdir)
        gen_s = time.time() - t0
        data = run_eval_cq(pdir, self.gt_dir, os.path.join(self.work_dir, f"eval_{tag}.json"))
        if not data:
            return
        m = eval_metrics(data.get("rows", []), len(self.val))
        m["eval_gen_s"] = round(gen_s, 1)
        m["eval_total_s"] = round(time.time() - t0, 1)
        print(f"[eval @ {tag} epoch {epoch}] " + " ".join(f"{k}={v}" for k, v in m.items()))
        if self.trainer is not None:           # TensorBoard curves (eval/<metric>)
            self.trainer.log({f"eval/{k}": v for k, v in m.items()
                              if isinstance(v, (int, float))})
        rec = {"step": step, "epoch": epoch, **m}
        self.history.append(rec)
        cur = m.get(self.best_metric)
        if cur is not None and (self.best_value is None or cur > self.best_value):
            self.best_value, self.best_step, self.best_epoch = cur, step, epoch
            best_dir = os.path.join(self.work_dir, "best")
            if self.trainer is not None:
                self.trainer.save_model(best_dir)      # LoRA -> adapter; full -> full weights
            else:
                self.model.save_pretrained(best_dir)
            print(f"[eval] new best {self.best_metric}={cur} -> {best_dir}")
        with open(os.path.join(self.work_dir, "best_meta.json"), "w") as f:
            json.dump({"step": self.best_step, "epoch": self.best_epoch,
                       "metric_name": self.best_metric, "metric_value": self.best_value,
                       "history": self.history}, f, indent=2)

    def on_step_end(self, args, state, control, **kw):
        if not state.is_world_process_zero or self.eval_every_epochs <= 0:
            return
        ep = float(state.epoch or 0.0)
        if ep - self.last_eval_epoch + 1e-9 >= self.eval_every_epochs:
            self.last_eval_epoch = ep
            self._do(state)

    def on_train_end(self, args, state, control, **kw):
        if state.is_world_process_zero:
            self._do(state)


# --------------------------------------------------------------- config
@dataclass
class ExpConfig:
    """All experiment knobs. HfArgumentParser exposes each as a CLI flag (dash or
    underscore form) and can seed them from a JSON/YAML file (see parse_config)."""
    train: str = field(default="experiments/data_z2c_train",
                       metadata={"help": "train bundle: a dir with all.jsonl, or a jsonl path"})
    val: str = field(default="experiments/data_z2c_val",
                     metadata={"help": "val bundle (Z2C val source split): dir with all.jsonl, or jsonl"})
    gt_dir: str = field(default="experiments/stage_z2c_val",
                        metadata={"help": "GT STEP dir for the val split (eval hook execs preds against these)"})
    prompt: str = field(default=PROMPT,
                        metadata={"help": 'instruction prepended to the graph text (--prompt "" to drop it)'})
    ckpt: str = DEFAULT_CKPT
    out: Optional[str] = field(default=None, metadata={
        "help": "run dir (default: experiments/train3d/<YYYY-MM-DD_HH-MM-SS>)"})
    full: bool = field(default=False, metadata={"help": "full fine-tune (default: LoRA)"})
    train_vision: bool = field(default=False, metadata={"help": "don't freeze vision tower"})
    max_len: int = 8192
    bs: int = 1
    grad_accum: int = 8
    lr: Optional[float] = field(default=None, metadata={"help": "default 2e-4 LoRA / 1e-4 full"})
    epochs: float = 3.0
    max_steps: int = -1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lora_r: int = 16
    optim: str = field(default="adamw_torch",
                       metadata={"help": "adamw_torch | adafactor (fits full-FT of 2B on one 24GB A5000)"})
    no_grad_ckpt: bool = False
    max_new_tokens: int = 1024
    eval_val_n: int = field(default=48, metadata={
        "help": "in-training eval subset size, sampled ONCE from val with a fixed seed "
                "(identical across evals and across same-seed runs). 0 = full val set."})
    eval_seed: Optional[int] = field(default=None, metadata={
        "help": "seed for the eval-subset sample (default: reuse --seed)"})
    eval_every_epochs: float = field(default=1.0, metadata={
        "help": "run the eval hook every N epochs (float ok; checked at on_step_end via "
                "state.epoch). <=0 disables periodic eval (train-end eval still runs)."})
    eval_batch_size: int = field(default=8, metadata={
        "help": "prompts per generate() call in the eval hook (conservative vs infer's 16)"})
    eval_max_batch_tokens: int = field(default=24000, metadata={
        "help": "padded-token budget (longest×count) per eval batch. Conservative default "
                "(~half infer's 48000) because optimizer state + params are GPU-resident during "
                "training; ~7-8 GB peak leaves LoRA plenty of headroom on a 24 GB A5000."})
    best_metric: str = field(default="mean_iou_incl_fail", metadata={
        "help": "scalar tracked for best-model checkpointing (higher=better): "
                "mean_iou_incl_fail (default; folds validity+geometry, failures=0) | "
                "valid_rate | median_iou"})
    n_eval: int = field(default=-1, metadata={
        "help": "DEPRECATED alias of --eval_val_n (kept for backward compat; if >=0 it "
                "overrides eval_val_n with a warning)"})
    limit: int = field(default=0, metadata={"help": "cap train records (smoke/overfit)"})
    smoke: bool = field(default=False,
                        metadata={"help": "LoRA, 40 steps, max-len 3072, limit 24 — pipe/loss sanity"})
    seed: int = 42
    report_to: str = field(default="tensorboard",
                           metadata={"help": "tensorboard (default) | wandb | none"})
    run_name: Optional[str] = field(default=None, metadata={"help": "TensorBoard/wandb run label"})
    wandb_project: str = field(default="drawing2cad-3d",
                               metadata={"help": "wandb project (only when --report-to wandb)"})


def _git_hash():
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                           cwd=os.path.dirname(os.path.abspath(__file__)), timeout=5)
        return r.stdout.strip() or None
    except Exception:
        return None


DEFAULT_OUT_ROOT = "experiments/train3d"


def resolve_out(cli_out: Optional[str], world: int, is_main: bool) -> str:
    """Run dir. `--out` overrides; otherwise experiments/train3d/<timestamp>.
    Under DDP every rank runs main() independently, so rank0 picks the timestamp
    and shares it via a rendezvous file (keyed by the shared torchrun run id) —
    without this each rank would stamp a different second and split the run."""
    if cli_out:
        return cli_out
    if world <= 1:
        return os.path.join(DEFAULT_OUT_ROOT, time.strftime("%Y-%m-%d_%H-%M-%S"))
    import tempfile
    rid = os.environ.get("TORCHELASTIC_RUN_ID") or os.environ.get("MASTER_PORT") or "run"
    marker = os.path.join(tempfile.gettempdir(), f"train3d_out.{rid}")
    if is_main:
        out = os.path.join(DEFAULT_OUT_ROOT, time.strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(out, exist_ok=True)
        with open(marker, "w") as f:
            f.write(out)
        return out
    for _ in range(600):                 # wait (≤60 s) for rank0 to choose
        if os.path.exists(marker):
            return open(marker).read().strip()
        time.sleep(0.1)
    raise RuntimeError("timed out waiting for rank0 to resolve --out")


def parse_config() -> ExpConfig:
    """Parse CLI into ExpConfig; a `--config FILE.{json,yaml}` seeds defaults that CLI
    flags then override (set_defaults keeps CLI precedence)."""
    argv = sys.argv[1:]
    cfg_path = None
    if "--config" in argv:
        i = argv.index("--config"); cfg_path = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    parser = HfArgumentParser(ExpConfig)
    if cfg_path:
        if cfg_path.endswith((".yaml", ".yml")):
            import yaml
            data = yaml.safe_load(open(cfg_path)) or {}
        else:
            data = json.load(open(cfg_path))
        parser.set_defaults(**data)
    (cfg,) = parser.parse_args_into_dataclasses(args=argv)
    return cfg


# --------------------------------------------------------------- main
def main():
    global PROMPT  # --prompt overrides it; build_labels/generate_preds read the module global
    args = parse_config()
    PROMPT = args.prompt
    set_seed(args.seed)

    if args.n_eval >= 0:                       # deprecated alias -> eval_val_n (stay truthful in dump)
        print(f"[warn] --n_eval is deprecated; mapping {args.n_eval} -> --eval_val_n",
              file=sys.stderr)
        args.eval_val_n = args.n_eval
    if args.smoke:
        args.max_steps = args.max_steps if args.max_steps > 0 else 40
        args.max_len = min(args.max_len, 3072)
        args.limit = args.limit or 24
        args.eval_val_n = args.eval_val_n if args.eval_val_n in (0,) else min(args.eval_val_n, 4)
        if args.eval_every_epochs == 1.0:      # smoke runs ~13 epochs; don't eval every one
            args.eval_every_epochs = 4.0
    lr = args.lr if args.lr is not None else (1e-4 if args.full else 2e-4)

    world = int(os.environ.get("WORLD_SIZE", "1"))
    is_main = int(os.environ.get("RANK", "0")) == 0
    args.out = resolve_out(args.out, world, is_main)
    os.makedirs(args.out, exist_ok=True)
    if is_main:
        print(f"run dir: {args.out}")
    if is_main:  # dump the fully-resolved config (survives a crash mid-train)
        resolved = {**asdict(args), "lr_resolved": lr, "world_size": world,
                    "git_hash": _git_hash(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with open(os.path.join(args.out, "config.json"), "w") as f:
            json.dump(resolved, f, indent=2)

    report = args.report_to
    if report == "tensorboard":                # events under <out>/logs (tf5 reads this env)
        os.environ["TENSORBOARD_LOGGING_DIR"] = os.path.join(args.out, "logs")
    elif report == "wandb":                    # keep wandb fully local/offline on this box
        os.environ.setdefault("WANDB_MODE", "offline")
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

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
        seed=args.seed,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to=("none" if report == "none" else [report]),
        run_name=args.run_name,
        remove_unused_columns=False,
        gradient_checkpointing=False,  # enabled manually above (PEFT input-grad handshake)
        # LoRA adapters land on the frozen vision attn too (never run text-only) -> DDP
        # would flag them as unused; allow it for the 2-GPU path (no-op single-GPU).
        ddp_find_unused_parameters=(world > 1),
        dataloader_num_workers=2, optim=args.optim)

    # fixed, seeded RANDOM val subset for the eval hook (avoids the shortest-graph bias of
    # the old probe). Sampled once here so the subset is identical across every eval in a run
    # AND across runs sharing the same seed. eval_val_n=0 -> the whole val set.
    import random
    eval_seed = args.eval_seed if args.eval_seed is not None else args.seed
    n_sub = len(val) if args.eval_val_n <= 0 else min(args.eval_val_n, len(val))
    val_eval = (list(val) if n_sub >= len(val)
                else random.Random(eval_seed).sample(val, n_sub))
    if is_main:
        print(f"eval subset: {len(val_eval)}/{len(val)} (seed {eval_seed}) "
              f"every {args.eval_every_epochs} epoch(s) | best_metric={args.best_metric}")
    cb = EvalCadCallback(model, tok, val_eval, args.out, args.gt_dir,
                         max_new_tokens=args.max_new_tokens, cap_len=args.max_len,
                         batch_size=args.eval_batch_size,
                         max_batch_tokens=args.eval_max_batch_tokens,
                         eval_every_epochs=args.eval_every_epochs,
                         best_metric=args.best_metric)
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=Collator(tok.pad_token_id),
        callbacks=[cb],
    )
    cb.trainer = trainer                       # save_model / TensorBoard logging need it

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
