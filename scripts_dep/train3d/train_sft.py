"""train_sft.py — text-only SFT for the AMVDG->CadQuery (AMVDG->3D) leg.

Input  = Serialization of an AMVDG graph (text only; NO drawing image — this is the
         experiment that tests whether the IR is *sufficient*).
Output = CadQuery Python code (GT = Zero-to-CAD cadquery_file).
Start  = ADSKAILab/Zero-To-CAD-Qwen3-VL-2B (image->CadQuery SFT; same base + output as us),
         swappable via --ckpt (loads exactly that ckpt — no silent fallback to another model).

Recipe (see scripts/train3d/README.md for the cadrille / Zero-to-CAD diff table):
  * bf16, completion-only loss (mask the prompt, train only the answer span),
  * LoRA (default) or full fine-tuning (--full), vision tower frozen (text-only),
  * launched via torchrun (`torchrun --nproc_per_node=N ...`; N=1 single-GPU, N=2 DDP) —
    device selection is the launcher's job, never done in-process (see the note above imports),
  * over-length pairs are FILTERED, never truncated (truncation would corrupt the g2
    input or cut the target); the dropped count is reported.
  * eval: Trainer's NATIVE distributed eval loop (GenEvalTrainer) greedy-generates a FIXED,
    seeded RANDOM val subset (`eval_val_n`, default 48; 0 = full val) every `eval_every_epochs`
    epochs, or every `eval_every_steps` optimizer steps when that's set (replaces the epoch
    cadence entirely) — the subset is SHARDED across ranks (both GPUs generate), gathered, then
    scored by eval_cq.py (validity + translation-aligned voxel IoU + bbox-mm) inside compute_metrics.
    The folded scalar (`best_metric`, default `mean_iou_incl_fail` = mean IoU over the whole
    subset counting failed/invalid preds as 0) streams to TensorBoard as `eval_*` and drives
    native best-model checkpointing (save_strategy="steps" @ eval cadence, so EVERY eval saves a
    full HF checkpoint -- not just new-bests -- + load_best_model_at_end). The final save lands in
    `<out>/<best_metric>_<value>_step<N>/` (self-describing name); `<out>/best` is kept as a
    symlink to it so infer.py's fixed `--ckpt <out>/best` keeps working. `<out>/latest` is a
    symlink to the most-recently-saved `<out>/checkpoint-<N>/`, kept current live DURING training
    by `LatestLinkCallback` (survives a crash, unlike best/final which are only written after
    trainer.train() returns) -- resume a crashed/stopped run with `--resume_from auto` (or an
    explicit `<out>/checkpoint-<N>` path) + the SAME `--out`. `save_total_limit` (default 2) caps
    HF checkpoints (`<out>/checkpoint-<N>/`, full optimizer/scheduler/rng state) to latest+best.
    No rank0-only callback / manual barrier: the eval loop keeps the ranks in lockstep by design.

Smoke: python train_sft.py --smoke  (LoRA, ~40 steps, one A5000, overfits by design).

Config: args are an `ExpConfig` dataclass parsed by HfArgumentParser, so every CLI flag
(dash or underscore form) can also be set from a JSON/YAML file via `--config run.yaml`
(CLI flags override file values). The fully resolved config is dumped to `<out>/config.json`
and metrics stream to TensorBoard under `<out>/logs` (or wandb with `--report-to wandb`).
"""

from typing import Iterable
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict

# Device selection is delegated to the launcher (torchrun / accelerate launch), NOT done
# in-process: a bare `python train_sft.py` on a >1-GPU box makes HF Trainer fall back to
# torch.nn.DataParallel and silently scale the DataLoader batch to bs*n_gpu (which then
# feeds the shared-logits Collator cross-example batches). The supported launch is always
# `torchrun --nproc_per_node=N` (N GPUs; N=1 for single-GPU) — under it every rank sees
# LOCAL_RANK, so n_gpu=1 and DataParallel never triggers. Pick specific GPUs with
# CUDA_VISIBLE_DEVICES in the SHELL (it composes with torchrun). The n_gpu>1 tripwire in
# main() hard-fails a bare-python multi-GPU run with these instructions instead of limping
# into DataParallel. (Replaces the old cuInit-order-fragile in-process CUDA_VISIBLE_DEVICES
# rewrite; see troubleshoot.md for that saga.)

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from serialize import coordinate_tokens, coordinate_tokenize_text, serialized_quant


def _sac_context_fn():
    """Selective-AC policy: retain attention kernels and recompute MLP/cheap ops.

    Saving all MLP matmuls OOMs a 15k sequence on the 24 GB A5000, so this deliberately uses the
    memory-conservative end of SAC's tradeoff. Kept top-level because torch checkpoint invokes
    context_fn during every checkpointed layer forward. Ops absent from torch are simply omitted.
    """
    from torch.utils.checkpoint import (
        CheckpointPolicy,
        create_selective_checkpoint_contexts,
    )

    aten = torch.ops.aten
    save_ops = []
    for path in (
        "_scaled_dot_product_flash_attention.default",
        "_scaled_dot_product_efficient_attention.default",
        "_flash_attention_forward.default",
        "_efficient_attention_forward.default",
    ):
        op = aten
        try:
            for part in path.split("."):
                op = getattr(op, part)
        except AttributeError:
            continue
        save_ops.append(op)
    save_ops = set(save_ops)

    def policy_fn(ctx, op, *args, **kwargs):
        return (
            CheckpointPolicy.MUST_SAVE
            if op in save_ops
            else CheckpointPolicy.PREFER_RECOMPUTE
        )

    return create_selective_checkpoint_contexts(policy_fn)


def gradient_checkpointing_kwargs(mode: str) -> dict:
    if mode == "full":
        return {"use_reentrant": False}
    if mode == "sac":
        return {"use_reentrant": False, "context_fn": _sac_context_fn}
    raise ValueError(f"unknown --grad_ckpt_mode {mode!r}; expected full or sac")


# Task instruction prepended to the graph text in the user turn. cadrille uses the 3-word
# "Generate cadquery code"; the chat template's assistant-turn start is the code-start marker,
# so no custom special token is added. Override with --prompt (e.g. --prompt "" to ablate).
# infer.py reads this as its own --prompt default (train_sft.PROMPT); kept as an immutable
# module constant -- the per-run value is threaded explicitly, not written back here.
PROMPT = "Generate cadquery code from this multi-view drawing graph; assign the solid to `result`."
DEFAULT_MAX_NEW_TOKENS = 1024
# eval_cq.py needs cadquery/FreeCAD and is subprocessed rather than imported: OCC has real
# C-layer hang failure modes (see eval_cq.py's imap_isolated), and the periodic in-training
# eval hook can't be allowed to take a multi-day run down with it. Defaults to this process's
# own interpreter (verified to already have cadquery/FreeCAD); DRAWING2CAD_PY overrides it.
DRAWING2CAD_PY = os.environ.get("DRAWING2CAD_PY", sys.executable)
EVAL_CQ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_cq.py")


# --------------------------------------------------------------- model loading
def _resolve_attn(attn: str) -> str:
    """Pick the attention backend. 'auto' => flash_attention_2 if the flash-attn
    package is importable (best for long context / 16k), else 'sdpa'. Note SDPA
    already dispatches to a *memory-efficient O(n)* kernel for masked training, so
    16k fits without flash-attn — flash_attention_2 is a speed/headroom upgrade
    that needs `pip install flash-attn`."""
    if attn != "auto":
        return attn
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except Exception:
        return "sdpa"


def load_model(ckpt, dtype, attn="auto", device_map=None):
    """Load exactly `ckpt` as a causal LM.
    Qwen3-VL loads text-only fine (no images).
    device_map: None -> caller places the model (training / single-GPU inference); pass "auto"
    to let accelerate shard the layers across every visible GPU (big-model inference)."""
    impl = _resolve_attn(attn)
    print(f"[load_model] attn_implementation={impl}", file=sys.stderr)
    kw = dict(torch_dtype=dtype, attn_implementation=impl, trust_remote_code=True)
    if device_map is not None:
        kw["device_map"] = device_map
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
    out = tok.apply_chat_template(
        msgs,
        tokenize=True,
        add_generation_prompt=gen_prompt,
        return_tensors=return_tensors,
    )
    if hasattr(out, "keys"):
        out = out["input_ids"]
    if return_tensors is None and len(out) and isinstance(out[0], list):
        out = out[0]
    return out


def build_labels(tok, input_text, target_code, max_len, prompt=PROMPT):
    """Tokenize one conversation; mask everything but the assistant answer (completion-only).
    Returns (input_ids, labels, prompt_len) or None if it exceeds max_len (filter, don't truncate)."""
    user = f"{prompt}\n\n{input_text}" if prompt else input_text
    msgs_p = [{"role": "user", "content": user}]
    msgs_f = msgs_p + [{"role": "assistant", "content": target_code}]
    pids = chat_ids(tok, msgs_p, True)
    fids = chat_ids(tok, msgs_f, False)
    if len(fids) > max_len:
        return None
    # locate the answer span via longest common prefix. pids/fids are tokenized as whole
    # strings, not per-message, so BPE can re-merge the trailing prompt token once the
    # assistant content follows right after (e.g. "...assistant\n" alone vs
    # "...assistant\n\nimport ...") -- an exact `fids[:len(pids)] == pids` check fails on
    # ~0.25% of the noblend train corpus for exactly this reason (verified: always a
    # 1-token gap right at the boundary, never a deep mismatch). The previous fallback
    # (substring search, else silently p=0) both trained those examples on the whole
    # sequence unmasked AND forced the collator's logits window to the full length --
    # the actual root cause of a 9+ GiB single-example OOM at long max_len.
    p = 0
    for a, b in zip(pids, fids):
        if a != b:
            break
        p += 1
    labels = [-100] * p + fids[p:]
    return fids, labels, p


class SFTDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, str]],
        tok,
        max_len: int,
        workers: int = 16,
        prompt: str = PROMPT,
        coord_tokens: bool = False,
    ):
        # per-record work is the fast (Rust) chat-template/tokenizer, which releases the
        # GIL, so a thread pool parallelizes this well (same fix as build_dataset.py's
        # tokenizer loop) instead of tokenizing all records one at a time up front.
        self.ex, self.dropped = [], 0

        def _one(r):
            text = (
                coordinate_tokenize_text(r["input_text"])
                if coord_tokens
                else r["input_text"]
            )
            return build_labels(tok, text, r["target_code"], max_len, prompt=prompt)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for built in tqdm(
                ex.map(_one, records), total=len(records), desc="tokenizing"
            ):
                if built is None:
                    self.dropped += 1
                    continue
                ids, labels, p = built
                self.ex.append({"input_ids": ids, "labels": labels, "prompt_len": p})

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i: int):
        return self.ex[i]


class Collator:
    """Right-pads a batch AND restricts the model's final vocab-projection (lm_head) to
    the trailing span that actually needs logits, via `logits_to_keep` (supported by the
    Qwen3-VL forward). Completion targets are short (median ~380 tok, p99 ~850) versus
    the up-to-16k-token prompt, so materializing full-sequence logits — then upcasting
    them to fp32 for the loss, per HF's ForCausalLMLoss — is the dominant OOM source at
    long max_len (confirmed: 16k-token example OOMs at ~21GB with full logits, fits in
    ~10GB restricted to the completion span). See train_sft.py module docstring / README
    for the derivation; the alignment below must exactly match transformers'
    ForCausalLMLoss internal pad+shift or the loss silently trains on the wrong targets.
    `max_completion_cap` additionally hard-bounds the window regardless of prompt_len
    (see comment in __call__ -- this is what a real prompt-boundary mis-detection OOM'd on).
    """

    def __init__(self, pad_id, max_completion_cap=4096):
        self.pad_id = pad_id
        self.max_completion_cap = max_completion_cap

    def __call__(self, batch):
        m = max(len(b["input_ids"]) for b in batch)
        min_p = min(b["prompt_len"] for b in batch)
        # hard cap independent of prompt_len, so a future prompt-boundary bug (build_labels
        # returning a too-small p for some example) can't silently re-open the full-sequence-
        # logits OOM this once caused. The corpus-wide max completion is 3018 tok (verified
        # over all 85,903 kept examples), well under the default 4096 -- so under normal
        # operation this branch is unreachable. If it fires, something upstream is newly
        # broken; raise instead of truncating-and-warning, since a print to stderr is easy to
        # miss in a multi-day DDP log and the alternative (silently train on a truncated
        # target) is exactly the silent-corruption failure mode this whole fix exists to kill.
        # Raise the cap deliberately via --max_completion_cap if completions have genuinely
        # grown longer than this corpus; don't let this exception train you into raising it
        # reflexively without checking why first.
        uncapped = m - min_p + 1
        if uncapped > self.max_completion_cap:
            pairs = [(b["prompt_len"], len(b["input_ids"])) for b in batch]
            raise RuntimeError(
                f"[Collator] completion window {uncapped} exceeds max_completion_cap="
                f"{self.max_completion_cap} (min prompt_len={min_p}, max total len={m}, "
                f"len(batch)={len(batch)}, per-example (prompt_len, total)={pairs}). "
                f"If len(batch)==1 the example's own prompt boundary is mis-detected -- "
                f"investigate it before raising the cap. If len(batch)>1 the window is a "
                f"cross-example quantity: a long-prompt and a short-prompt example were "
                f"batched together, which the shared trailing window can't cover under the "
                f"cap -- check for an unintended effective batch size (e.g. Trainer's "
                f"DataParallel fallback scales it to bs*n_gpu when >1 GPU is visible "
                f"outside torchrun)."
            )
        k = min(
            m, uncapped
        )  # trailing hidden-state positions covering every example's completion
        ii, ll, am = [], [], []
        for b in batch:
            n = m - len(b["input_ids"])
            ii.append(b["input_ids"] + [self.pad_id] * n)
            am.append([1] * len(b["input_ids"]) + [0] * n)
            full = b["labels"] + [-100] * n  # length m
            ll.append(
                [-100] + full[m - k + 1 :]
            )  # length k, aligned to the kept window
        return {
            "input_ids": torch.tensor(ii),
            "labels": torch.tensor(ll),
            "attention_mask": torch.tensor(am),
            "logits_to_keep": k,
        }


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
def iter_batched_generate(
    model,
    tok,
    prompts,
    max_new_tokens: int,
    batch_size: int,
    max_batch_tokens: int,
    device: torch.device,
) -> Iterable[tuple[str, str | None]]:
    """prompts: list[(stem, ids_list[int])]. Yields (stem, decoded_code) per prompt, or
    (stem, None) when generation OOMs for that prompt even on its own — attempt-then-fail,
    nothing is dropped before it is tried. Length-sorts, left-pad-batches (_make_batches),
    greedy-decodes on `device` (the model's input-embedding device; the model may be sharded
    across GPUs). On a CUDA OOM the offending batch is halved and requeued (shorter half first);
    a solo batch that still OOMs yields None.
    Precondition (caller-set): model.eval(), model.config.use_cache=True,
    tok.padding_side='left', gradient checkpointing disabled."""
    prompts = sorted(prompts, key=lambda x: len(x[1]))
    queue = list(_make_batches(prompts, batch_size, max_batch_tokens))
    while queue:
        batch = queue.pop(0)
        enc = tok.pad(
            {"input_ids": [ids for _, ids in batch]}, padding=True, return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        try:
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tok.pad_token_id,
            )
        except Exception as e:  # only CUDA OOM is recoverable; re-raise the rest
            if "out of memory" not in str(e).lower():
                raise
            del enc
            torch.cuda.empty_cache()
            if len(batch) > 1:  # too big together -> split and retry
                mid = (len(batch) + 1) // 2
                queue[:0] = [batch[:mid], batch[mid:]]
            else:  # does not fit even alone -> generation failure
                stem, ids = batch[0]
                print(
                    f"[gen] OOM: '{stem}' ({len(ids)} tok) does not fit even solo -> failure",
                    file=sys.stderr,
                )
                yield stem, None
            continue
        plen = enc["input_ids"].shape[
            1
        ]  # left-padded → shared prompt offset for the batch
        for j, (stem, _) in enumerate(batch):
            yield stem, tok.decode(gen[j][plen:], skip_special_tokens=True)
        del enc, gen


# ---------------------------------------------- eval (native distributed generate-eval)
def run_eval_cq(pred_dir: str, gt_dir: str, out_json: str, limit: int = 0):
    """Shell out to eval_cq.py in the CadQuery env; return the loaded JSON dict
    ({"aggregate","config","rows"}) or None. We need per-row data (not just the
    aggregate) so the folded IoU metric can count failed/invalid preds as 0."""
    py = DRAWING2CAD_PY
    if not os.path.exists(py):  # sys.executable symlink can churn mid env-rebuild
        py = shutil.which("python3") or shutil.which("python") or py
    if not os.path.exists(py):
        print(
            f"[eval] skip: no python interpreter found (tried {DRAWING2CAD_PY})",
            file=sys.stderr,
        )
        return None
    cmd = [py, EVAL_CQ, "--pred-dir", pred_dir, "--gt-dir", gt_dir, "--out", out_json]
    if limit:
        cmd += ["--limit", str(limit)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        print(r.stdout[-1500:])
        print(r.stderr[-500:], file=sys.stderr)
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
    while len(ious) < n_subset:  # unscored/missing preds -> 0
        ious.append(0.0)
        valids.append(0.0)
    n = max(n_subset, len(ious)) or 1
    return {
        "mean_iou_incl_fail": round(sum(ious) / n, 4),
        "median_iou": round(statistics.median(ious), 4),
        "valid_rate": round(sum(valids) / n, 4),
        "n_scored": len(rows),
        "n_subset": n_subset,
    }


class GenEvalDataset(Dataset):
    """Prompt-only eval examples for GENERATION. Each item = the user-turn prompt ids plus a
    single-element `labels` holding the example's index into `self.ids` — so compute_metrics can
    map each (gathered, possibly reordered/padding-duplicated) prediction back to its uuid WITHOUT
    depending on gather order. Over-cap prompts (> cap_len) are EXCLUDED here (they'd OOM
    generate()); the caller keeps counting them in n_subset, so they score as failures (IoU 0)."""

    def __init__(self, records, tok, cap_len, prompt=PROMPT, coord_tokens=False):
        self.ids, self.ex, self.dropped = [], [], 0
        kept = []
        for r in records:
            text = (
                coordinate_tokenize_text(r["input_text"])
                if coord_tokens
                else r["input_text"]
            )
            user = f"{prompt}\n\n{text}" if prompt else text
            ids = chat_ids(tok, [{"role": "user", "content": user}], True)
            if len(ids) > cap_len:
                self.dropped += 1
                continue
            kept.append((r["id"], ids))
        kept.sort(
            key=lambda t: len(t[1])
        )  # length-sort -> homogeneous fixed-size eval batches
        for rid, ids in kept:  # labels = index into self.ids (gather-order-proof)
            self.ex.append({"input_ids": ids, "labels": [len(self.ids)]})
            self.ids.append(rid)

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        return self.ex[i]


class GenCollator:
    """LEFT-pad prompts for decoder-only generation; pass the index-carrying `labels` through."""

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        m = max(len(b["input_ids"]) for b in batch)
        ii, am, ll = [], [], []
        for b in batch:
            n = m - len(b["input_ids"])
            ii.append([self.pad_id] * n + b["input_ids"])  # left pad
            am.append([0] * n + [1] * len(b["input_ids"]))
            ll.append(b["labels"])  # length-1, uniform -> no padding
        return {
            "input_ids": torch.tensor(ii),
            "attention_mask": torch.tensor(am),
            "labels": torch.tensor(ll),
        }


class GenEvalTrainer(Trainer):
    """Trainer whose eval loop GENERATES (greedy) instead of a forward pass, so the native
    distributed evaluation_loop shards the eval set across ranks (both GPUs generate), gathers the
    predictions, and calls compute_metrics — no rank0-only callback, no manual barrier/log/save, no
    collective desync. Generation runs on the UNWRAPPED `self.model` (same pattern as
    Seq2SeqTrainer) so it issues no DDP collective. The eval dataset is length-sorted (GenEvalDataset)
    and batched at a FIXED per_device_eval_batch_size through the standard accelerate-prepared loader,
    which EQUALIZES batch counts across ranks. (A custom token-budget batch_sampler does not — it left
    the ranks with unequal batch counts, desyncing the per-batch gather into a 30-min NCCL timeout.)
    Peak eval memory ~= eval_batch_size * longest prompt, so keep eval_batch_size modest at max_len 16k."""

    def __init__(
        self,
        *args,
        gen_collator=None,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        grad_ckpt_min_tokens=0,
        grad_ckpt_kwargs=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._gen_collator = gen_collator
        self._max_new_tokens = max_new_tokens
        self._grad_ckpt_min_tokens = grad_ckpt_min_tokens
        self._grad_ckpt_kwargs = grad_ckpt_kwargs or {"use_reentrant": False}
        self._gc_modules = [
            m for m in self.model.modules() if hasattr(m, "gradient_checkpointing")
        ]
        self._last_gc_active = bool(
            getattr(self.model, "is_gradient_checkpointing", False)
        )

    def _set_gc_active(self, active, seq_len=None):
        """Toggle only layer flags; keep the configured checkpoint function/input-grad hook."""
        if active == self._last_gc_active:
            return
        for module in self._gc_modules:
            module.gradient_checkpointing = active
        self._last_gc_active = active
        if self.is_world_process_zero():
            print(
                f"[grad-ckpt] {'ON' if active else 'OFF'} at step {self.state.global_step} "
                f"seq_len={seq_len}"
            )

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self._grad_ckpt_min_tokens > 0:
            # Inputs are still CPU tensors here, before Trainer._prepare_inputs. Padded sequence
            # length is the conservative memory criterion and adds no CUDA synchronization.
            seq_len = int(inputs["input_ids"].shape[-1])
            self._set_gc_active(seq_len >= self._grad_ckpt_min_tokens, seq_len)
        return super().training_step(
            model, inputs, num_items_in_batch=num_items_in_batch
        )

    def get_eval_dataloader(self, eval_dataset=None):
        ds = self.eval_dataset if eval_dataset is None else eval_dataset
        dl = DataLoader(
            ds,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=self._gen_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        return self.accelerator.prepare(
            dl
        )  # accelerate shards + EQUALIZES batches across ranks

    def evaluate(self, *args, **kwargs):
        # Generation needs grad-ckpt OFF + KV cache ON, else an 8k-prefill decode is O(n^2)
        # (~minutes/batch instead of seconds). Training leaves gradient checkpointing enabled,
        # and while it is enabled this model forces use_cache=False in its forward -- so the
        # generate() use_cache kwarg is ignored. Bracket the whole eval: disable grad-ckpt + turn
        # the cache on, then restore for training (forward-only eval never needs grad-ckpt). This
        # mirrors the setup the old rank0 eval hook did, now hoisted to the native eval boundary.
        gc_on = bool(getattr(self.model, "is_gradient_checkpointing", False))
        prev_cache = self.model.config.use_cache
        if gc_on:
            self.model.gradient_checkpointing_disable()
        self.model.config.use_cache = True
        try:
            return super().evaluate(*args, **kwargs)
        finally:
            self.model.config.use_cache = prev_cache
            if gc_on:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=self._grad_ckpt_kwargs
                )
                self._last_gc_active = True

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        idx = inputs["labels"]  # [B,1] dataset index -> rides through the gather
        if prediction_loss_only:
            return (None, None, idx)
        with torch.no_grad():
            gen = self.model.generate(  # UNWRAPPED model: no DDP forward collective
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.processing_class.pad_token_id,
            )
        comp = gen[
            :, inputs["input_ids"].shape[1] :
        ].contiguous()  # strip shared left-pad prompt
        return (None, comp, idx)


class LatestLinkCallback(TrainerCallback):
    """Re-points `<out>/latest` at the checkpoint `_save_checkpoint` JUST wrote, on every save
    (save_strategy="steps" now saves every eval, not just new-bests -- see main()). This is the
    ONLY thing in this file that updates live, DURING training: the best-name dir + `<out>/best`
    symlink (see main(), after trainer.train() returns) are a post-hoc summary of a run that
    finished, so they don't exist if the process crashes first -- which is exactly when a resume
    pointer is needed. `on_save` fires after the checkpoint dir is fully written (Trainer calls it
    right after `_save_checkpoint`), so the target always exists at symlink-creation time; the one
    gap is a crash mid-symlink-swap, an inherent limit of any live pointer, not worth guarding."""

    def __init__(self, out_dir):
        self.out_dir = out_dir

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control
        target = f"checkpoint-{state.global_step}"  # matches Trainer._save_checkpoint exactly
        link = os.path.join(self.out_dir, "latest")
        if os.path.islink(link):
            os.remove(link)
        elif os.path.exists(link):
            shutil.rmtree(link)
        os.symlink(target, link)
        return control


def make_compute_metrics(trainer, eval_ids, gt_dir, out_dir, tok, n_subset):
    """compute_metrics for GenEvalTrainer. Runs on EVERY rank with the identical gathered
    predictions (accelerate.gather_for_metrics), so it needs no manual broadcast/barrier: each rank
    writes id-keyed .py files (main -> <out>/preds_step{N} for the artifact trail, others -> a
    throwaway tmp it deletes) and scores them via eval_cq.py, producing an identical metric dict.
    The scalars are prefixed `eval_` by Trainer and drive native best-model checkpointing."""

    def compute_metrics(eval_pred):
        import numpy as np

        preds = np.asarray(eval_pred.predictions)
        idx = np.asarray(eval_pred.label_ids).reshape(-1)
        preds = np.where(
            preds >= 0, preds, tok.pad_token_id
        )  # eval loop pads the gather with -100
        codes = tok.batch_decode(preds, skip_special_tokens=True)
        step = int(trainer.state.global_step)
        is_main = trainer.accelerator.is_main_process
        pdir = (
            os.path.join(out_dir, f"preds_step{step}")
            if is_main
            else tempfile.mkdtemp(prefix="eval_preds_")
        )
        os.makedirs(pdir, exist_ok=True)
        for j, code in zip(idx.tolist(), codes):
            with open(os.path.join(pdir, f"{eval_ids[int(j)]}.py"), "w") as f:
                f.write(code)
        data = run_eval_cq(pdir, gt_dir, os.path.join(pdir, f"eval_step{step}.json"))
        if not is_main:
            shutil.rmtree(pdir, ignore_errors=True)
        # ALWAYS return the metric keys (eval_metrics of [] folds to all-zeros) even when eval_cq
        # could not score (e.g. cadquery/FreeCAD broken) -- otherwise metric_for_best_model raises
        # KeyError in _determine_best_metric and takes the whole training run down.
        rows = data.get("rows", []) if data else []
        return {
            k: v
            for k, v in eval_metrics(rows, n_subset).items()
            if isinstance(v, (int, float))
        }

    return compute_metrics


# --------------------------------------------------------------- config
@dataclass
class ExpConfig:
    """All experiment knobs. HfArgumentParser exposes each as a CLI flag (dash or
    underscore form) and can seed them from a JSON/YAML file (see parse_config)."""

    train: str = field(
        default="experiments/data_z2c_train",
        metadata={"help": "train bundle: a dir with all.jsonl, or a jsonl path"},
    )
    val: str = field(
        default="experiments/data_z2c_val",
        metadata={
            "help": "val bundle (Z2C val source split): dir with all.jsonl, or jsonl"
        },
    )
    gt_dir: str = field(
        default="experiments/stage_z2c_val",
        metadata={
            "help": "GT STEP dir for the val split (eval hook execs preds against these)"
        },
    )
    prompt: str = field(
        default=PROMPT,
        metadata={
            "help": 'instruction prepended to the graph text (--prompt "" to drop it)'
        },
    )
    ckpt: str = "ADSKAILab/Zero-To-CAD-Qwen3-VL-2B"
    out: str | None = field(
        default=None,
        metadata={
            "help": "run dir (default: experiments/train3d/<YYYY-MM-DD_HH-MM-SS>)"
        },
    )
    full: bool = field(
        default=False, metadata={"help": "full fine-tune (default: LoRA)"}
    )
    train_vision: bool = field(
        default=False, metadata={"help": "don't freeze vision tower"}
    )
    max_len: int = 16384  # 8192
    max_completion_cap: int = field(
        default=4096,
        metadata={
            "help": "hard cap on the collator's completion-logits window (logits_to_keep), "
            "independent of prompt_len; bounds worst-case logits memory if a future "
            "tokenizer/prompt-boundary edge case ever widens it (see Collator)"
        },
    )
    tok_workers: int = field(
        default=16,
        metadata={
            "help": "thread pool size for SFTDataset's up-front tokenization pass"
        },
    )
    attn: str = field(
        default="auto",
        metadata={
            "help": "attention backend: auto (flash_attention_2 if installed else sdpa) "
            "| sdpa | flash_attention_2 | eager"
        },
    )
    bs: int = 1
    grad_accum: int = 8
    lr: float | None = field(
        default=None, metadata={"help": "default 2e-4 LoRA / 1e-4 full"}
    )
    epochs: float = 3.0
    max_steps: int = -1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lora_r: int = 16
    optim: str = field(
        default="adamw_torch",
        metadata={
            "help": "adamw_torch | adafactor (fits full-FT of 2B on one 24GB A5000)"
        },
    )
    no_grad_ckpt: bool = False
    grad_ckpt_min_tokens: int = field(
        default=0,
        metadata={
            "help": "dynamic activation-checkpoint threshold: with grad checkpointing enabled, "
            "disable it for padded sequences shorter than N tokens. 0 keeps checkpointing "
            "enabled for every batch (current behavior). Calibrate against peak VRAM first."
        },
    )
    grad_ckpt_mode: str = field(
        default="full",
        metadata={
            "help": "activation checkpoint policy: full (current whole-layer recomputation) | "
            "sac (experimental selective AC; retain recognized attention ops)"
        },
    )
    train_sampling_strategy: str = field(
        default="group_by_length",
        metadata={
            "help": "training sampler: group_by_length (default; reduces padding and DDP rank skew) "
            "| random | sequential"
        },
    )
    ddp_find_unused_parameters: bool = field(
        default=False,
        metadata={
            "help": "enable DDP's per-step unused-parameter graph traversal. Default false because "
            "text-only training freezes the unused vision tower after LoRA insertion. It is "
            "forced on by --train_vision."
        },
    )
    torch_compile: bool = field(
        default=False,
        metadata={
            "help": "opt in to torch.compile training. Keep off for the production recipe until a "
            "fixed-length-bucket benchmark confirms that recompilation does not erase gains."
        },
    )
    torch_compile_backend: str | None = field(
        default=None,
        metadata={
            "help": "torch.compile backend (default: PyTorch/Accelerate default, normally inductor)"
        },
    )
    torch_compile_mode: str | None = field(
        default=None,
        metadata={
            "help": "torch.compile mode, e.g. default | reduce-overhead | max-autotune"
        },
    )
    coord_tokens: bool = field(
        default=False,
        metadata={
            "help": "replace quantized geometry coordinates with N dedicated magnitude tokens, "
            "where N is read from the bundle's PART grid=N header. "
            "Opt-in representation experiment; learns only the added embedding rows under LoRA."
        },
    )
    track_token_throughput: bool = field(
        default=False,
        metadata={
            "help": "log non-padding tokens/second. Off by default because distributed token counting "
            "adds a small collective on each micro-step; enable for short throughput A/B runs."
        },
    )
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    eval_val_n: int = field(
        default=48,
        metadata={
            "help": "in-training eval subset size, sampled ONCE from val with a fixed seed "
            "(identical across evals and across same-seed runs). 0 = full val set."
        },
    )
    eval_seed: int | None = field(
        default=None,
        metadata={"help": "seed for the eval-subset sample (default: reuse --seed)"},
    )
    eval_every_epochs: float = field(
        default=1.0,
        metadata={
            "help": "run the eval hook every N epochs (float ok; converted to optimizer steps via "
            "steps_per_epoch). Ignored when --eval_every_steps is set. <=0 disables periodic "
            "eval (train-end eval still runs)."
        },
    )
    eval_every_steps: int | None = field(
        default=None,
        metadata={
            "help": "run the eval hook every N optimizer steps; when set this REPLACES the "
            "epoch-derived cadence entirely (0 or negative explicitly disables periodic "
            "eval, regardless of --eval_every_epochs). Default None keeps the epoch cadence."
        },
    )
    eval_batch_size: int = field(
        default=4,
        metadata={
            "help": "per-device eval GENERATION batch size (fixed). Peak eval memory ~= this * longest "
            "prompt, so keep it modest at --max_len 16k (4 fits LoRA on a 24 GB A5000; lower for "
            "full-FT). The eval subset is length-sorted so batches stay homogeneous."
        },
    )
    eval_max_batch_tokens: int = field(
        default=24000,
        metadata={
            "help": "DEPRECATED / unused since the native distributed eval — fixed --eval_batch_size now "
            "governs eval batching (a token-budget batch_sampler desynced the DDP gather). Kept "
            "only so pre-existing --config files don't error."
        },
    )
    best_metric: str = field(
        default="mean_iou_incl_fail",
        metadata={
            "help": "scalar tracked for best-model checkpointing (higher=better): "
            "mean_iou_incl_fail (default; folds validity+geometry, failures=0) | "
            "valid_rate | median_iou"
        },
    )
    save_total_limit: int = field(
        default=2,
        metadata={
            "help": "max HF-native checkpoints (<out>/checkpoint-<step>/, full optimizer/scheduler/rng "
            "state -- a save now happens every eval, not just new-bests, so resume can restart "
            "from the LATEST step) kept on disk; the most-recent AND the best are both always "
            "protected on top of this count and collapse to 1 dir when they're the same "
            "checkpoint. Default 2 = exactly latest+best. Raise for more resume history (costs "
            "disk -- only meaningfully large under --full; a LoRA checkpoint is ~tens of MB). "
            "<=0 = unlimited (HF default; unbounded growth under frequent step-eval)."
        },
    )
    resume_from: str | None = field(
        default=None,
        metadata={
            "help": "resume training from a checkpoint dir (optimizer/scheduler/rng/step state, not "
            "just weights) -- pass 'auto' to resume the LATEST checkpoint under --out (same "
            "convention as <out>/latest), or an explicit <out>/checkpoint-<N> path. Requires "
            "--out to point at the ORIGINAL run's dir (resume continues writing into it, it "
            "does not fork a new one) and the same --ckpt/--full/--lora_r/--train_vision/--optim "
            "(a mismatch is warned about, not blocked -- see main()). Needs periodic eval "
            "enabled (checkpointing is driven by the eval cadence); default None = fresh start."
        },
    )
    n_eval: int = field(
        default=-1,
        metadata={
            "help": "DEPRECATED alias of --eval_val_n (kept for backward compat; if >=0 it "
            "overrides eval_val_n with a warning)"
        },
    )
    limit: int = field(
        default=0, metadata={"help": "cap train records (smoke/overfit)"}
    )
    train_min_tokens: int = field(
        default=0,
        metadata={
            "help": "benchmark-only: keep records whose precomputed n_tok_total is at least N"
        },
    )
    train_max_tokens: int = field(
        default=0,
        metadata={
            "help": "benchmark-only: keep records whose precomputed n_tok_total is at most N; 0=off"
        },
    )
    smoke: bool = field(
        default=False,
        metadata={"help": "LoRA, 40 steps, max-len 3072, limit 24 — pipe/loss sanity"},
    )
    seed: int = 42
    report_to: str = field(
        default="tensorboard", metadata={"help": "tensorboard (default) | wandb | none"}
    )
    run_name: str | None = field(
        default=None, metadata={"help": "TensorBoard/wandb run label"}
    )
    wandb_project: str = field(
        default="drawing2cad-3d",
        metadata={"help": "wandb project (only when --report-to wandb)"},
    )


def _git_hash():
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            timeout=5,
        )
        return r.stdout.strip() or None
    except Exception:
        return None


DEFAULT_OUT_ROOT = "experiments/train3d"


def resolve_out(cli_out: str | None, world: int, is_main: bool) -> str:
    """Run dir. `--out` overrides; otherwise experiments/train3d/<timestamp>.
    Under DDP every rank runs main() independently, so rank0 picks the timestamp
    and shares it via a rendezvous file (keyed by the shared torchrun run id) —
    without this each rank would stamp a different second and split the run.

    Race-condition fix: rank0 writes via a temp file + os.replace (atomic on POSIX)
    so rank1 never sees a truncated (empty) marker. A stale marker from a previous
    run with the same TORCHELASTIC_RUN_ID is deleted first; rank1 additionally
    skips empty reads as a belt-and-suspenders guard."""
    if cli_out:
        return cli_out
    if world <= 1:
        return os.path.join(DEFAULT_OUT_ROOT, time.strftime("%Y-%m-%d_%H-%M-%S"))
    import tempfile

    rid = (
        os.environ.get("TORCHELASTIC_RUN_ID") or os.environ.get("MASTER_PORT") or "run"
    )
    marker = os.path.join(tempfile.gettempdir(), f"train3d_out.{rid}")
    if is_main:
        # Remove stale marker from a previous run so rank1 doesn't read an old value
        # before rank0 writes the new one.
        try:
            os.remove(marker)
        except FileNotFoundError:
            pass
        out = os.path.join(DEFAULT_OUT_ROOT, time.strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(out, exist_ok=True)
        # Atomic write: tmp file + os.replace prevents rank1 from ever reading
        # a truncated (0-byte) marker — the old open("w") truncated the file
        # before writing, creating a window where rank1 could read ''.
        tmp = marker + ".tmp"
        with open(tmp, "w") as f:
            f.write(out)
        os.replace(tmp, marker)  # atomic on POSIX
        return out
    for _ in range(600):  # wait (≤60 s) for rank0 to choose
        if os.path.exists(marker):
            val = open(marker).read().strip()
            if val:  # skip empty reads (stale truncated file)
                return val
        time.sleep(0.1)
    raise RuntimeError("timed out waiting for rank0 to resolve --out")


ARCH_ARGS = ["ckpt", "full", "lora_r", "train_vision", "optim", "coord_tokens"]


def warn_resume_arch_drift(resume_from: str, args: "ExpConfig") -> None:
    """--resume_from restores optimizer/scheduler/adapter state INTO whatever model
    load_model()+get_peft_model() just built from the CURRENT CLI args -- it does not restore
    the args themselves. A drifted --ckpt/--full/--lora_r/--train_vision/--optim either crashes
    deep inside optimizer/adapter loading (shape mismatch) or, worse, loads onto the wrong
    shapes without erroring. config.json (dumped by the ORIGINAL run at its own startup, see
    main()) sits one directory up from any <out>/checkpoint-<N> or <out>/latest path -- that's
    always true here since _save_checkpoint places checkpoint dirs directly under output_dir."""
    cfg_path = os.path.join(
        os.path.dirname(os.path.normpath(resume_from)), "config.json"
    )
    if not os.path.exists(cfg_path):
        print(
            f"[warn] --resume_from {resume_from}: no sibling config.json found at {cfg_path} "
            f"-- cannot check for architecture drift.",
            file=sys.stderr,
        )
        return
    orig = json.load(open(cfg_path))
    drift = {
        k: (orig.get(k), getattr(args, k))
        for k in ARCH_ARGS
        if orig.get(k) != getattr(args, k)
    }
    if drift:
        print(
            f"[warn] --resume_from {resume_from}: architecture args differ from the original "
            f"run's config.json ({cfg_path}): {drift} -- optimizer/adapter state may fail to "
            f"load, or load onto mismatched shapes without erroring. Verify this is intentional.",
            file=sys.stderr,
        )


def parse_config() -> ExpConfig:
    """Parse CLI into ExpConfig; a `--config FILE.{json,yaml}` seeds defaults that CLI
    flags then override (set_defaults keeps CLI precedence)."""
    argv = sys.argv[1:]
    cfg_path = None
    if "--config" in argv:
        i = argv.index("--config")
        cfg_path = argv[i + 1]
        argv = argv[:i] + argv[i + 2 :]
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
    args = parse_config()
    set_seed(args.seed)

    world = int(os.environ.get("WORLD_SIZE", "1"))
    is_main = int(os.environ.get("RANK", "0")) == 0

    if args.n_eval >= 0:  # deprecated alias -> eval_val_n (stay truthful in dump)
        print(
            f"[warn] --n_eval is deprecated; mapping {args.n_eval} -> --eval_val_n",
            file=sys.stderr,
        )
        args.eval_val_n = args.n_eval
    if args.smoke:
        args.max_steps = args.max_steps if args.max_steps > 0 else 40
        args.max_len = min(args.max_len, 3072)
        args.limit = args.limit or 24
        args.eval_val_n = (
            args.eval_val_n if args.eval_val_n in (0,) else min(args.eval_val_n, 4)
        )
        if args.eval_every_epochs == 1.0:  # smoke runs ~13 epochs; don't eval every one
            args.eval_every_epochs = 4.0
    lr = args.lr if args.lr is not None else (1e-4 if args.full else 2e-4)

    args.out = resolve_out(args.out, world, is_main)
    os.makedirs(args.out, exist_ok=True)
    if is_main:
        print(f"run dir: {args.out}")

    # resolve --resume_from BEFORE the config.json dump below overwrites it -- the drift check
    # needs the ORIGINAL run's config.json, and "auto" needs an existing checkpoint-N to find.
    # get_last_checkpoint is a deterministic local-filesystem glob, safe to call on every rank.
    resume = args.resume_from
    if resume == "auto":
        resume = get_last_checkpoint(args.out)
        if is_main and resume is None:
            print(
                f"[warn] --resume_from auto: no checkpoint-* found under {args.out}; "
                f"starting fresh.",
                file=sys.stderr,
            )
    if resume and is_main:
        print(f"[resume] resuming from {resume}")
        warn_resume_arch_drift(resume, args)

    if is_main:  # dump the fully-resolved config (survives a crash mid-train)
        resolved = {
            **asdict(args),
            "lr_resolved": lr,
            "world_size": world,
            "git_hash": _git_hash(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "resume_from_resolved": resume,
        }
        with open(os.path.join(args.out, "config.json"), "w") as f:
            json.dump(resolved, f, indent=2)

    report = args.report_to
    if report == "tensorboard":  # events under <out>/logs (tf5 reads this env)
        os.environ["TENSORBOARD_LOGGING_DIR"] = os.path.join(args.out, "logs")
    elif report == "wandb":  # keep wandb fully local/offline on this box
        os.environ.setdefault("WANDB_MODE", "offline")
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    def load_jsonl(path: str) -> list[dict]:
        if os.path.isdir(path):
            path = os.path.join(path, "all.jsonl")
        return [json.loads(line) for line in open(path)]

    train = load_jsonl(args.train)
    val = load_jsonl(args.val)
    if args.train_min_tokens:
        train = [r for r in train if r["n_tok_total"] >= args.train_min_tokens]
    if args.train_max_tokens:
        train = [r for r in train if r["n_tok_total"] <= args.train_max_tokens]
    if args.limit:
        train = sorted(train, key=lambda r: r["n_tok_total"])[
            : args.limit
        ]  # short = fast smoke

    tok = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
    coord_token_ids = []
    if args.coord_tokens:
        grids = {serialized_quant(r["input_text"]) for r in train + val}
        if None in grids or len(grids) != 1:
            raise ValueError(
                f"--coord_tokens requires one shared nonzero PART grid=N across train+val; "
                f"found {sorted(str(x) for x in grids)}"
            )
        coord_quant = grids.pop()
        coord_vocab = coordinate_tokens(coord_quant)
        added = tok.add_tokens(coord_vocab, special_tokens=True)
        coord_token_ids = tok.convert_tokens_to_ids(coord_vocab)
        if len(set(coord_token_ids)) != len(coord_vocab):
            raise RuntimeError(
                f"coordinate tokens did not map to {len(coord_vocab)} unique tokenizer ids"
            )
        print(
            f"coordinate tokens: quant={coord_quant}, added {added}, "
            f"ids {min(coord_token_ids)}.."
            f"{max(coord_token_ids)}"
        )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    ds = SFTDataset(
        train,
        tok,
        args.max_len,
        workers=args.tok_workers,
        prompt=args.prompt,
        coord_tokens=args.coord_tokens,
    )
    print(
        f"train examples kept {len(ds)} (dropped over-len {ds.dropped}) "
        f"| val {len(val)} | max_len {args.max_len}"
    )
    if is_main and len(ds):
        lengths = sorted(len(ex["input_ids"]) for ex in ds.ex)

        def percentile(q):
            return lengths[min(len(lengths) - 1, round(q * (len(lengths) - 1)))]

        print(
            f"train tokens/example mean {sum(lengths) / len(lengths):.0f} "
            f"p50 {percentile(0.50)} p90 {percentile(0.90)} p99 {percentile(0.99)}; "
            "Trainer will also report aggregate train_tokens_per_second when supported"
        )

    model, used = load_model(args.ckpt, torch.bfloat16, attn=args.attn)
    print(f"loaded {used}")
    if args.coord_tokens:
        model.resize_token_embeddings(len(tok))
    if not args.train_vision:
        nfz = 0
        for n, p in model.named_parameters():
            if ".visual." in n or n.startswith("visual."):
                p.requires_grad_(False)
                nfz += 1
        print(f"froze {nfz} vision params")
    gc_kwargs = gradient_checkpointing_kwargs(args.grad_ckpt_mode)
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
        model.config.use_cache = False
    if not args.full:
        from peft import LoraConfig, get_peft_model

        lc = LoraConfig(
            r=args.lora_r,
            lora_alpha=2 * args.lora_r,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            task_type="CAUSAL_LM",
            trainable_token_indices=(coord_token_ids or None),
        )
        if not args.no_grad_ckpt:
            model.enable_input_require_grads()
        model = get_peft_model(model, lc)

    # PEFT inserts fresh trainable LoRA tensors after the initial vision freeze above. Freeze the
    # vision namespace again so text-only DDP has no deliberately-unused trainable parameters and
    # can safely skip find_unused_parameters' autograd-graph traversal on every micro-step.
    if not args.train_vision:
        nfz_after, trainable_visual = 0, []
        for n, p in model.named_parameters():
            if ".visual." in n or n.startswith("visual."):
                if p.requires_grad:
                    nfz_after += 1
                    p.requires_grad_(False)
                if p.requires_grad:
                    trainable_visual.append(n)
        if trainable_visual:
            raise RuntimeError(
                f"text-only run still has trainable vision parameters: "
                f"{trainable_visual[:8]}"
            )
        if is_main:
            print(
                f"froze {nfz_after} vision params introduced or re-enabled after adapter setup"
            )
    if not args.full:
        model.print_trainable_parameters()

    # --- fixed, seeded RANDOM val subset for eval (identical across evals AND same-seed runs;
    # eval_val_n=0 -> whole val). Built BEFORE TrainingArguments so the eval cadence (in optimizer
    # steps) can be derived from the requested epoch fraction. ---
    import random

    eval_seed = args.eval_seed if args.eval_seed is not None else args.seed
    n_sub = len(val) if args.eval_val_n <= 0 else min(args.eval_val_n, len(val))
    val_eval = (
        list(val) if n_sub >= len(val) else random.Random(eval_seed).sample(val, n_sub)
    )
    n_subset = len(val_eval)  # denominator: over-cap prompts count as failures
    gen_eval_ds = GenEvalDataset(
        val_eval,
        tok,
        cap_len=args.max_len,
        prompt=args.prompt,
        coord_tokens=args.coord_tokens,
    )
    eval_ids = list(gen_eval_ds.ids)

    # optimizer steps/epoch = ceil(per-rank samples / bs) // grad_accum (matches HF's own count)
    steps_per_epoch = max(
        1, math.ceil(len(ds) / max(world, 1) / max(args.bs, 1)) // args.grad_accum
    )
    if args.eval_every_steps is not None:  # explicit step cadence -> REPLACES epochs
        eval_steps = args.eval_every_steps if args.eval_every_steps > 0 else None
        cadence_desc = f"every {eval_steps} steps (--eval_every_steps)"
    else:
        eval_steps = (
            max(1, round(args.eval_every_epochs * steps_per_epoch))
            if args.eval_every_epochs > 0
            else None
        )
        cadence_desc = f"every {args.eval_every_epochs} epoch(s) = {eval_steps} steps"
    eval_on = eval_steps is not None and len(gen_eval_ds) > 0
    if is_main:
        print(
            f"eval subset: {len(gen_eval_ds)}/{len(val)} usable "
            f"(+{gen_eval_ds.dropped} over-cap -> counted as failures) seed {eval_seed} | "
            + (
                f"{cadence_desc} | best=eval_{args.best_metric}"
                if eval_on
                else "periodic eval DISABLED"
            )
        )

    # Available in recent Transformers and harmlessly omitted on older installs. This scans the
    # already-tokenized in-memory dataset once and adds a length-aware throughput metric; examples/s
    # alone is misleading when sequences span hundreds to 16k tokens.
    throughput_args = {}
    if "include_tokens_per_second" in TrainingArguments.__dataclass_fields__:
        throughput_args["include_tokens_per_second"] = True
    if (
        args.track_token_throughput
        and "include_num_input_tokens_seen" in TrainingArguments.__dataclass_fields__
    ):
        throughput_args["include_num_input_tokens_seen"] = "non_padding"
    ddp_find_unused = args.ddp_find_unused_parameters or args.train_vision
    if is_main:
        print(
            f"train optimizations: train_sampling_strategy={args.train_sampling_strategy} "
            f"ddp_find_unused_parameters={ddp_find_unused} "
            f"torch_compile={args.torch_compile} "
            f"compile_backend={args.torch_compile_backend} "
            f"compile_mode={args.torch_compile_mode} "
            f"coord_tokens={args.coord_tokens} "
            f"track_token_throughput={args.track_token_throughput}"
        )

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.eval_batch_size,
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
        # native distributed generate-eval + best-model checkpointing (GenEvalTrainer):
        eval_strategy=("steps" if eval_on else "no"),
        eval_steps=eval_steps,
        # save EVERY eval (not just new-bests): load_best_model_at_end still finds the best via
        # best_global_step (tracked regardless of save_strategy), and this additionally leaves a
        # checkpoint at the true latest step for --resume_from / LatestLinkCallback. save_steps
        # MUST equal eval_steps, not just a multiple of it -- if the best landed on an eval step
        # that wasn't ALSO a save step, load_best_model_at_end would hunt for a checkpoint that
        # was never written. (When eval is off there's nothing to resume from either way.)
        save_strategy=("steps" if eval_on else "no"),
        save_steps=(
            eval_steps if eval_on else 500
        ),  # 500 = HF default; inert under strategy="no"
        save_total_limit=args.save_total_limit,  # protects latest + best; see field help
        metric_for_best_model=(f"eval_{args.best_metric}" if eval_on else None),
        greater_is_better=True,
        load_best_model_at_end=eval_on,
        report_to=("none" if report == "none" else [report]),
        run_name=args.run_name,
        remove_unused_columns=False,
        gradient_checkpointing=False,  # enabled manually above (PEFT input-grad handshake)
        train_sampling_strategy=args.train_sampling_strategy,
        ddp_find_unused_parameters=ddp_find_unused,
        torch_compile=args.torch_compile,
        torch_compile_backend=args.torch_compile_backend,
        torch_compile_mode=args.torch_compile_mode,
        dataloader_num_workers=2,
        optim=args.optim,
        **throughput_args,
    )
    # Tripwire: n_gpu>1 means a bare `python train_sft.py` on a multi-GPU box (no launcher set
    # LOCAL_RANK), so HF Trainer is about to fall back to DataParallel and scale the DataLoader
    # batch to bs*n_gpu, feeding the shared-logits Collator cross-example batches. Fail here with
    # instructions instead of limping on. Device selection is the launcher's job (see the note
    # atop this module); there is no in-process CUDA_VISIBLE_DEVICES rewrite anymore.
    if targs.n_gpu > 1:
        raise RuntimeError(
            f"n_gpu={targs.n_gpu}: bare-python multi-GPU run -> Trainer would use DataParallel and "
            f"scale the DataLoader batch to bs*n_gpu. Launch with a distributed launcher instead: "
            f"`torchrun --nproc_per_node={targs.n_gpu} scripts/train3d/train_sft.py ...` (N=1 for "
            f"single-GPU); pick specific GPUs with CUDA_VISIBLE_DEVICES in the shell."
        )

    trainer = GenEvalTrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        eval_dataset=(gen_eval_ds if eval_on else None),
        data_collator=Collator(
            tok.pad_token_id, max_completion_cap=args.max_completion_cap
        ),
        processing_class=tok,
        gen_collator=GenCollator(tok.pad_token_id),
        max_new_tokens=args.max_new_tokens,
        grad_ckpt_min_tokens=(0 if args.no_grad_ckpt else args.grad_ckpt_min_tokens),
        grad_ckpt_kwargs=gc_kwargs,
    )
    if eval_on:  # compute_metrics closes over the live trainer
        trainer.compute_metrics = make_compute_metrics(
            trainer, eval_ids, args.gt_dir, args.out, tok, n_subset
        )
        trainer.add_callback(
            LatestLinkCallback(args.out)
        )  # live <out>/latest, survives a crash

    t0 = time.time()
    out = trainer.train(resume_from_checkpoint=resume)
    print(f"train_runtime {time.time() - t0:.0f}s  final_loss {out.training_loss:.4f}")
    steps_per_second = out.metrics.get("train_steps_per_second", 0)
    seen = getattr(trainer.state, "num_input_tokens_seen", 0)
    initial_seen = getattr(trainer, "_initial_num_input_tokens_seen", 0)
    runtime = out.metrics.get("train_runtime", 0)
    if is_main and steps_per_second:
        msg = f"optimizer_step_seconds {1 / steps_per_second:.3f}"
        if runtime and seen > initial_seen:
            msg += (
                f"  non_padding_tokens_per_second {(seen - initial_seen) / runtime:.1f}"
            )
        print(msg)
    if hasattr(trainer.state, "log_history"):
        losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
        if losses:
            print(
                f"loss[0]={losses[0]:.4f} loss[-1]={losses[-1]:.4f} "
                f"(min {min(losses):.4f})"
            )
    # with load_best_model_at_end the in-memory model IS the best eval checkpoint -> save it under
    # a self-describing name (<best_metric>_<value>_step<N>) so `ls <out>` shows what the best
    # checkpoint scored without opening config.json/TensorBoard, and point the stable <out>/best
    # symlink at it (relative target, so moving the whole run dir doesn't break it) -- infer.py /
    # README / bench.py all hardcode --ckpt <out>/best, so that fixed path must keep resolving.
    # <out>/latest is kept current throughout training by LatestLinkCallback, not here -- nothing
    # to do for it at this point. Without eval, nothing was ever scored -> save <out>/final instead.
    has_best = eval_on and trainer.state.best_metric is not None
    if has_best:
        # best_metric/best_global_step are set atomically by Trainer._determine_best_metric
        # whenever save_strategy is steps/epoch/best (all true here) -> best_global_step cannot
        # be None in this branch.
        best_step = trainer.state.best_global_step
        best_name = (
            f"{args.best_metric}_{trainer.state.best_metric:.4f}_step{best_step}"
        )
    else:
        best_name = "final"
    save_dir = os.path.join(args.out, best_name)
    trainer.save_model(save_dir)
    if is_main:
        if has_best:
            best_link = os.path.join(args.out, "best")
            # unlike checkpoint-<N>/ (rotated by HF's own save_total_limit), best_name/final dirs
            # are OUR OWN trainer.save_model() dumps outside that mechanism -- across a resumed
            # run (main() invoked again against the same --out), a stale best_name dir from the
            # earlier invocation would otherwise never get cleaned up. Remove it once superseded.
            old_target = os.readlink(best_link) if os.path.islink(best_link) else None
            if os.path.islink(best_link):
                os.remove(best_link)
            elif os.path.exists(best_link):
                shutil.rmtree(best_link)
            if old_target and old_target != best_name:
                old_dir = os.path.join(args.out, old_target)
                if os.path.isdir(old_dir):
                    shutil.rmtree(old_dir, ignore_errors=True)
            os.symlink(best_name, best_link)
            print(
                f"saved best model -> {save_dir} (symlinked as {best_link}; eval_{args.best_metric}="
                f"{trainer.state.best_metric:.4f} @ step {best_step}); latest step's checkpoint "
                f"-> {os.path.join(args.out, 'latest')}"
            )
        else:
            print(f"saved model -> {save_dir}")


if __name__ == "__main__":
    main()
