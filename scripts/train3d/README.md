# AMVDG → 3D CAD learning harness (`scripts/train3d/`)

Trains and evaluates the **back leg** of the drawing2cad pipeline: `AMVDG graph (serialized text) → CadQuery code → 3D solid`. Text-only by design — **no drawing PNG is fed** — so a working leg is evidence the AMVDG IR is *sufficient* to recover 3D.

- **Input** = `scripts/train3d/serialize.graph_to_text(graph)` (part-frame mm, each axis bbox-min = 0; `Ck` tags mark the same 3D feature across views).
- **Target** = GT CadQuery (`experiments/stage_z2c_{train,val}/{uuid}.cadquery.py`, assigns `result`).
- **Start ckpt** = [`ADSKAILab/Zero-To-CAD-Qwen3-VL-2B`](https://huggingface.co/ADSKAILab/Zero-To-CAD-Qwen3-VL-2B) (image→CadQuery SFT; same base *and* same output format as us), swappable via `--ckpt`.
- **Data** = the two per-split bundles from **Data Preparation step 4** (top-level README):
  `experiments/data_z2c_train` (≈2825, Zero-To-CAD train split) and `experiments/data_z2c_val`
  (≈274, val split). Each is `all.jsonl` + `invalid_extent.jsonl` + `stats.json`; the SFT train/val sets **are** these
  source splits (no extra split is carved inside a bundle).

## Files
| file | what |
|---|---|
| `audit_extent.py` | audit an existing bundle's `ext / max(bbox_3d)` distribution; optionally trace each warning/quarantine to the source view/primitive/failure mode with `--graph-dir`. |
| `build_dataset.py` | one graph-dir + code-dir → `all.jsonl` (`{id,input_text,target_code,n_tok_*}`) + `invalid_extent.jsonl` quarantine + `stats.json` token/extent report. Run once per split — see top-level README step 4. |
| `serialize.py` | this leg's AMVDG-graph ↔ model-input text codec (`graph_to_text` + inverse for round-trip). `python serialize.py GRAPH.json` prints it; `--check 'GLOB'` validates round-trip + cross-view consistency dataset-wide. |
| `train_sft.py` | text-only SFT (bf16, completion-only loss, LoRA/full, 1-/2-GPU); `--train`/`--val` bundles. **Periodic in-training eval** (every `--eval_every_epochs` epochs, or every `--eval_every_steps` steps when set — that replaces the epoch cadence — plus train-end): batched greedy-generate a fixed seeded RANDOM val subset (`--eval_val_n`, default 48; 0 = full val) via the *shared* `iter_batched_generate` (same code path as `infer.py`), score with `eval_cq.py`, log folded metrics to TensorBoard, and save a full HF checkpoint EVERY eval on the `--best_metric` scalar: the final save lands in a self-describing `<out>/<best_metric>_<value>_step<N>/`, with `<out>/best` symlinked to it (fixed path for `--ckpt`) and `<out>/latest` symlinked to the most-recently-saved `<out>/checkpoint-<N>/` (kept current live during training, so it survives a crash — see `--resume_from`). `--save_total_limit` (default 2) caps HF checkpoints (`<out>/checkpoint-<N>/`, full optimizer/scheduler/rng state) to latest+best. Args are an `ExpConfig` dataclass (`HfArgumentParser`): every flag is also settable from a `--config FILE.{json,yaml}` (CLI overrides file); resolved config → `<out>/config.json`, metrics → TensorBoard `<out>/logs` (or `--report-to wandb`, offline). |
| `infer.py` | batch inference: AMVDG JSON → CadQuery `{stem}.py` + executed `{stem}.step`, with a trained ckpt (LoRA adapter dir *or* full ckpt / HF id). Batched generation is the shared `train_sft.iter_batched_generate` (length-sort + left-pad); isolated timeout'd exec; writes `infer_summary.json` (exec/step rates + error hist). Naming pairs with GT `{uuid}.step` so `eval_cq.py` scores its `.py` outputs directly. |
| `eval_cq.py` | isolated exec → validity + **translation-aligned voxel IoU (abs mm)** + bbox-mm error → JSON. |

## Run

### Training
```bash
# bundles come from Data Preparation step 4 (top-level README):
#   experiments/data_z2c_train  +  experiments/data_z2c_val
#
# LAUNCH POLICY: always via torchrun (`--nproc_per_node=N`; N=1 single-GPU, N=2 DDP). Device
# selection is the launcher's job — a bare `python train_sft.py` on a >1-GPU box trips HF Trainer
# into a DataParallel fallback (batch silently scaled to bs*n_gpu); the script hard-fails that with
# instructions. Pick specific GPUs with CUDA_VISIBLE_DEVICES in the shell (it composes with
# torchrun), e.g. `CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 ...`.

# 1. SFT smoke — LoRA, 40 steps, one A5000 (proves the pipe + loss drop)
torchrun --nproc_per_node=1 scripts/train3d/train_sft.py --smoke \
    --train experiments/data_z2c_train \
    --val experiments/data_z2c_val \
    --out experiments/train3d/smoke
#   NOTE: --smoke clamps --max_len to 3072. Bundles built with longer inputs (e.g.
#   data_z2c_val_noblend, median ~3.2k input tokens) then have 0 usable eval examples ->
#   "periodic eval DISABLED", even with --eval_every_steps set. To smoke-test the eval/
#   checkpoint machinery itself, skip --smoke and pass its knobs manually with a --max_len
#   that fits the bundle instead, e.g.:
#   torchrun --nproc_per_node=1 scripts/train3d/train_sft.py \
#       --max_steps 40 --limit 24 --max_len 16384 --eval_val_n 4 --eval_every_steps 10 \
#       --train experiments/data_z2c_train_noblend --val experiments/data_z2c_val_noblend \
#       --out experiments/train3d/smoke_stepeval

# 2. real single-GPU LoRA run.
#    --max_len 16384: hard parts (faces median 127) serialize to ~10k tokens; at the old
#      8192 cap ~84% are FILTERED out of training (build_labels drops, never truncates).
#      16k lets them in; --attn auto uses flash_attention_2 if `pip install flash-attn`,
#      else sdpa (already mem-efficient O(n), so 16k fits one 24 GB A5000 with grad-ckpt).
#    --out is optional: omit it and the run lands in experiments/train3d/<YYYY-MM-DD_HH-MM-SS>/
torchrun --nproc_per_node=1 scripts/train3d/train_sft.py \
    --train experiments/data_z2c_train \
    --val experiments/data_z2c_val \
    --epochs 3 \
    --max_len 16384 \
    --attn auto \
    # --eval_every_steps 1000

# 3. two-GPU DDP (LoRA recommended; --full optional): torchrun handles world size.
#    DDP = throughput (each GPU a full replica on different samples), not memory relief;
#    16k LoRA already fits ONE 24 GB A5000, so no sharding needed.
#    Length grouping is enabled by default, and the unused-parameter traversal is disabled after
#    the text-only vision tower (including any LoRA tensors) is frozen. The run log prints both
#    resolved settings plus token-length percentiles and optimizer-step seconds.
torchrun --nproc_per_node=2 scripts/train3d/train_sft.py \
    --train experiments/data_z2c_train \
    --val experiments/data_z2c_val \
    --epochs 3 \
    --bs 1 \
    --grad-accum 8 \
    --max_len 16384 \
    --attn auto \
    --grad_ckpt_min_tokens 3000 \
    --coord_tokens \
    # --eval_every_steps 1000
# Throughput ablations (use the same data/seed/step count and compare after warmup):
#   --track_token_throughput         # adds non-padding tokens/s; small per-microstep DDP collective
#   --train_sampling_strategy random
#   --ddp_find_unused_parameters
#   --torch_compile                 # experimental; watch graph-break/recompile logs
#   --grad_ckpt_min_tokens 3000     # no checkpoint below N; calibrated for LoRA on 24GB A5000
#   --grad_ckpt_mode sac             # research option; no measured gain with FlashAttention-2 here
#   --coord_tokens                   # N learned magnitude tokens, where bundle header grid=N
#   NOTE: full-FT of a 2B with AdamW replicates ~22 GB of optimizer state PER GPU under plain DDP -> OOMs a 24 GB A5000. To fit "modest full-FT" pick ONE of:
#     --optim adafactor            # ~no 2nd-moment state; full-FT fits one 24 GB GPU
#     accelerate launch --fsdp ... # shard AdamW state across the 2 A5000s

# 3b. drive train_sft entirely from a config file (CLI flags still override it):
torchrun --nproc_per_node=1 scripts/train3d/train_sft.py --config my_run.yaml --epochs 5
#   watch it:  tensorboard --logdir <out>/logs   (resolved config -> <out>/config.json)

# In-training eval + best-model checkpointing (NATIVE distributed eval; knobs all optional):
#   --eval_val_n 48        fixed val subset size, sampled ONCE with a fixed seed (identical across
#                          evals and same-seed runs); 0 = whole val set. Over-cap prompts (>--max_len)
#                          are excluded from generation but still counted as failures (IoU 0).
#   --eval_seed N          seed for that sample (default: reuse --seed).
#   --eval_every_epochs 1  evaluate every N epochs (float ok; mapped to Trainer eval_steps);
#                          <=0 disables in-training eval + best-model tracking. Ignored when
#                          --eval_every_steps is set.
#   --eval_every_steps N   evaluate every N optimizer steps -- REPLACES --eval_every_epochs
#                          entirely when set (0 or negative explicitly disables periodic eval).
#                          Use this for a long run where a fixed step cadence is easier to reason
#                          about than a fraction of a (possibly huge) epoch.
#   --eval_batch_size 4    per-device eval generation batch size (fixed). Peak eval mem ~= this *
#                          longest prompt -> keep modest at --max_len 16k (4 fits LoRA on 24 GB;
#                          lower for full-FT). The subset is length-sorted so batches are homogeneous.
#   --best_metric mean_iou_incl_fail   scalar tracked for best-model saving (higher=better):
#                          mean_iou_incl_fail (default; mean IoU over the WHOLE subset with
#                          failed/invalid preds counted as 0 — folds validity+geometry) |
#                          valid_rate | median_iou.  (logged with an `eval_` prefix.)
#   --save_total_limit 2   max HF checkpoints (<out>/checkpoint-<step>/, full optimizer/scheduler/
#                          rng state) kept on disk; a save now happens on EVERY eval (not just new
#                          bests), so the most-recent AND the best are both always protected on top
#                          of this count (collapsing to 1 dir when they're the same checkpoint).
#                          Default 2 = exactly latest+best. Raise for more resume history -- costs
#                          disk, but only meaningfully so under --full (a LoRA checkpoint is tens
#                          of MB; a full-FT AdamW checkpoint of this 2B model is ~20 GB). <=0 = unlimited.
#   --resume_from PATH     resume a crashed/stopped run: optimizer/scheduler/rng/step state, not
#                          just weights. Pass 'auto' to pick up <out>/latest's target automatically,
#                          or an explicit <out>/checkpoint-<N> path. MUST be combined with the SAME
#                          --out as the original run (resume continues writing into it) and the
#                          same --ckpt/--full/--lora_r/--train_vision/--optim (drift is warned
#                          about, not blocked). Needs periodic eval enabled -- checkpointing is
#                          driven by the eval cadence, so a no-eval run has nothing to resume from.
#   (--n_eval is a DEPRECATED alias of --eval_val_n; --eval_max_batch_tokens is now unused.)
# The subset is SHARDED across ranks (both GPUs generate) by the native evaluation_loop, gathered,
# then scored by eval_cq (against --gt_dir) inside compute_metrics — no rank0-only callback, no manual
# barrier (this is what fixed the DDP eval desync/hang). Each eval writes <out>/preds_step{N}/ +
# <out>/eval_step{N}.json (these accumulate once per eval too — frequent step-eval multiplies them,
# no rotation knob yet); metrics stream to TensorBoard as `eval_<metric>` and drive native
# checkpointing (save_strategy="steps" @ eval cadence + load_best_model_at_end) -> best adapter/
# weights land in a self-describing <out>/<best_metric>_<value>_step<N>/ at train end, with
# <out>/best kept as a symlink to it; <out>/latest tracks the most-recently-saved checkpoint LIVE
# throughout training (LatestLinkCallback), so it's there even if the process crashes before
# reaching train end. (If eval_cq can't score — e.g. cadquery broken — metrics fold to 0 and
# training continues instead of crashing.)

# 3c. resuming a crashed/stopped run (same --out, same architecture flags as the original):
torchrun --nproc_per_node=1 scripts/train3d/train_sft.py \
    --train experiments/data_z2c_train --val experiments/data_z2c_val \
    --epochs 3 --max_len 16384 --attn auto \
    --out experiments/train3d/<the-original-run-dir> \
    # --eval_every_steps 1000 \
    --resume_from auto
#   Also acceptable: --resume_from experiments/train3d/<run>/checkpoint-<N> (explicit step).
#   Use the SAME --nproc_per_node as the original run too: RNG state is saved per-rank
#   (rng_state_<i>.pth under DDP), so a world_size change silently skips RNG restore (warns,
#   doesn't fail) -- data-shuffling order then diverges from what it would've been uninterrupted.
#   A mismatched --ckpt/--full/--lora_r/--train_vision/--optim vs the original run's config.json
#   prints a [warn] (architecture drift can crash deep inside optimizer/adapter loading, or worse,
#   load onto mismatched shapes silently) but does not block the run -- read the warning.
```

### Throughput calibration (2026-07-12, 2 x RTX A5000 24 GB)

`group_by_length` intentionally creates long-to-short cycles, so the first steps of a group took
33--45 seconds and later steps took 2--8 seconds. Controlled runs on the same 50 optimizer steps:

| setting | optimizer sec/step | loss |
|---|---:|---:|
| old: random + DDP unused traversal | 13.158 | 0.3249 |
| default: grouped + no traversal | **12.346** | 0.3251 |

Optional activation-checkpoint calibration found that disabling checkpointing at 5.2k tokens OOMed
after optimizer-state allocation; `--grad_ckpt_min_tokens 3000` completed a real 25-step DDP run at
11.364 sec/step (about 8% faster than the all-checkpoint window). This threshold is specific to the
current LoRA/model/A5000 setup. Keep the default `0` for maximum memory safety, and recalibrate after
changing batch size, LoRA/full fine-tuning, model, attention backend, or GPU.

The current attention-only SAC policy completed a 15k sample but matched full checkpointing at 6.329
sec/step: FlashAttention-2's external kernel is not exposed to the current ATen policy, so SAC behaves
like full recomputation. Keep `--grad_ckpt_mode full` unless profiling a different backend.

`torch.compile` is implemented but not recommended for this variable-length run. On a fixed ~3k
five-step probe it took 4.695 sec/step versus 0.952 without compile, with graph breaks in DDP logging,
FlashAttention masking, and checkpointing, plus recompiles for 2828 vs 2829 total length and differing
completion windows. It needs strict bucketing on both total sequence length and `logits_to_keep` before
another production comparison.

### Coordinate-token experiment

`--coord_tokens` maps quantized magnitudes 0--1023 to one token each; negative coordinates use the
same magnitude token plus a sign. Only primitive coordinates/radii and DIM span positions change.
Angles, bbox/ext, exact mm dimensions, feature values, and target CadQuery remain numeric. Existing
`all.jsonl` bundles are transformed structurally at load time, so they do not need rebuilding.
The coordinate vocabulary size is read from the numeric bundle's `PART ... grid=N` header; train and val must have one shared N. The N learned tokens are saved with the checkpoint tokenizer. Inference recovers N from that tokenizer and fails fast if `infer --quant` differs, so 256/512/1024/2048 grids can be tested without creating an inconsistent vocabulary. `CANON_QUANT=1024` remains only the build/infer default. The numerically equal `max_new_tokens=1024` is unrelated: it is the CadQuery completion-generation budget, shared by train-time eval and infer.

On 1,000 real `data_z2c_train_noblend` inputs this reduced tokenizer input length from 5,422,465 to 3,584,219 tokens (**33.9%**). A two-GPU LoRA smoke verified DDP training and checkpoint round-trip;
PEFT stores only the 1,024 added embedding rows alongside LoRA. Use a separate run because the new embeddings must be learned:

```bash
torchrun --nproc_per_node=2 scripts/train3d/train_sft.py \
    --train experiments/data_z2c_train_noblend \
    --val experiments/data_z2c_val_noblend \
    --epochs 3 --bs 1 --grad-accum 8 --max_len 16384 --attn auto \
    --coord_tokens
```

`infer.py` auto-detects the expanded tokenizer saved with that adapter and applies the same coordinate
serialization. `--coord-tokens on|off` exists only as a diagnostic override.

### Inference and Evaluation

```bash
# 4. batch-infer a trained ckpt over the val AMVDG graphs -> {uuid}.py + {uuid}.step
python scripts/train3d/infer.py \
    --ckpt experiments/train3d/<run>/best \
    --input experiments/dataset_z2c_val \
    --out experiments/train3d/<run>/preds_full
#   --ckpt takes a LoRA adapter dir (adapter_config.json -> loads its base) or a full ckpt/HF id.
#   (train_sft saves the best eval checkpoint under <out>/<best_metric>_<value>_step<N>/ and
#   symlinks it as <out>/best/ -- that fixed path is what --ckpt above points at; a no-eval run
#   saves <out>/final/ instead, no symlink since nothing was ever scored.)
#   Generation is BATCHED (length-sorted, left-padded): --batch-size (default 16) prompts per
#   generate() call, capped by --max-batch-tokens (default 0 = auto from free VRAM; peak ≈
#   0.214 MB/token measured, so one 24 GB A5000 holds ~82k prompt tokens). Prompts longer than
#   one GPU's capacity are run through a model SHARDED across all visible GPUs (device_map=auto,
#   --shard auto|on|off) so they still fit — expose both cards (CUDA_VISIBLE_DEVICES=0,1) to
#   enable it. A batch that OOMs is halved and retried; a solo prompt that still won't fit is
#   recorded as an `oom_generate` failure (attempted, never skipped). exec runs in parallel
#   (--workers, default min(8,cpu)). Greedy is bf16-batched so individual outputs differ run-to-run
#   vs bs=1, but aggregate exec/IoU rates match.

# 5. evaluate a dir of predicted {id}.py against GT {id}.step (full val = infer.py preds)
#    THIS prints + writes the eval numbers (valid_rate, mean/median IoU, bbox-mm err); step 4 only
#    produces the preds. eval_cq re-execs each {id}.py itself, so run 4 then 5.
python scripts/train3d/eval_cq.py \
    --pred-dir experiments/train3d/<run>/preds_full \
    --gt-dir experiments/stage_z2c_val \
    --out experiments/train3d/<run>/eval_full.json
#   Scoring runs --workers in parallel (default min(8,cpu)); each worker is pinned to 1 numeric
#   thread (no BLAS oversubscription) and --timeout defaults to 120 s (headroom for single-thread
#   64³ voxelization), so results are invariant to --workers. (eval_cq globs *.py, so infer.py's
#   co-located *.step are ignored — same dir is fine.)
# GT-code self-IoU sanity (expect mean IoU ~1.0):
python scripts/train3d/eval_cq.py \
    --pred-dir experiments/stage_z2c_val \
    --gt-dir experiments/stage_z2c_val \
    --ids-file <uuids> --limit 20 --out self_iou.json
```

The instruction prepended to the graph text is `train_sft.PROMPT` (a short cadrille-style line;
`train_sft` and `infer.py` share it; override with `--prompt`, e.g. `--prompt ""` to ablate it —
the chat template's assistant-turn start is the code-start marker, so no special token is added).

Measured results — token stats, GT-corpus self-IoU, and each run's val metrics — live in `research/research-log_3d.md`, not here.

## Design notes / recipe deltas

We **reference** cadrille and Zero-to-CAD recipes but **do not reuse their code**.

| knob | cadrille (Qwen2-VL-2B) | Zero-to-CAD (Qwen3-VL-2B) | **ours (this harness)** | why |
|---|---|---|---|---|
| input | 4-view render → 2×2 grid, video path | 8 views 256² | **serialized-graph text only** | test IR sufficiency; no pixels |
| base | Qwen2-VL-2B-Instruct | Qwen3-VL-2B-Instruct | **Zero-To-CAD-Qwen3-VL-2B** (loads exactly, no fallback) | warm-start on same output format |
| prompt | `"Generate cadquery code"` (chat template) | — | **short cadrille-style line, `--prompt`-able** | chat template's assistant turn = code-start marker; no special token |
| tf version | **4.50.3 pinned** (subclass hooks) | 4.57.3 | **5.12.1** (`chat_ids` shim) | merged `drawing2cad` env |
| tuning | full FT | full FT | **LoRA (default) / full** | 2×A5000: LoRA fits easily, full is tight |
| optimizer | AdamW | AdamW | AdamW (`adamw_torch`) | — |
| lr / sched | 2e-4 / cosine, warmup 1000 | 1e-4 / cosine, warmup 0.03 | **2e-4 LoRA · 1e-4 full / cosine, warmup 0.03** | adopt Zero-to-CAD for full (same base) |
| weight decay | 0.01 | 0.0 | **0.0** | follow Zero-to-CAD |
| eff. batch | ~30–32 | 16 (1×16 DDP) | 8 (bs1×ga8, tune up) | 24 GB budget |
| max seq len | dynamic pad (no cap) | 4096 | **8192 (filter over-len)** | long serialized inputs; ~90% fit |
| loss mask | completion-only, hard-coded ids | (n/a in card) | **completion-only via chat-template prefix length** | tokenizer-agnostic, no magic ids |
| gen | greedy, `max_new_tokens=768` | greedy, 4096 | greedy, **1024** | 99.6% target coverage |
| result var | `r` | `result` | accept **`result`→`r`→last shape** | GT uses `result` |
| **eval IoU** | mesh boolean, **unit-box normalized** | voxel 64³, **rot-aligned, normalized** | **voxel 64³, translation-aligned, ABSOLUTE mm** + bbox-mm err | scale is the novelty; input↔GT differ by translation only |
| exec isolation | `Process`+join(3 s) | — | **`Process`+join(30 s)+terminate** | same containment, longer for heavy Z2C parts |

**What is portable vs rewritten from cadrille.** Portable *ideas*: completion-only masking,
separate-process exec-with-timeout, `isValid`/`Volume` validity gate. Rewritten: cadrille's
`Cadrille(Qwen2VLForConditionalGeneration)` subclass, its point-cloud `FourierPointEncoder`,
the `pixel_values_videos` collator, and the hard-coded `151644/77091/151645` label search are
**all Qwen2-VL / multimodal specific** and unused here — the text-only leg is a plain causal-LM
SFT on Qwen3-VL, so none of that code transfers. cadrille's `evaluate.py` IoU is deliberately
*replaced* (its unit-box rescale discards the absolute size this project exists to recover).

**Checkpoint (`ADSKAILab/Zero-To-CAD-Qwen3-VL-2B`).** `Qwen3VLForConditionalGeneration`, base
`Qwen/Qwen3-VL-2B-Instruct`, **Apache-2.0**, 4.0 GB; loads text-only in the `drawing2cad` env
(forward-with-loss + greedy generate verified). Its card publishes the SFT recipe adopted above
and an eval protocol (voxel IoU 64³, success-rate) that we follow — but *un-normalized*.

## RL feasibility memo (design only — NOT implemented; user decides later)

cadrille's RL stage (paper 2505.22914; code unreleased) rewards `10·IoU − 10·invalid`, GRPO-family
(Dr.CPPO), with hard-example mining (keep prompts whose SFT IoU < 7.5/10). The reward needs **only
a mesh**, not a target sequence — a perfect fit for our synthetic pipeline (every graph has a GT
STEP, and `eval_cq.py` already returns exactly that reward signal). To build it ourselves:

- **Trainer**: TRL `GRPOTrainer`. `reward_funcs` = a Python callable wrapping `eval_cq.py`'s
  per-sample validity + translation-aligned voxel IoU (→ `10·IoU − 10·(1−valid)`); prompts = the
  serialized graph text; no reference completions. Needs `pip install trl` (not yet added).
- **Generation throughput (no vLLM in this env)**: GRPO samples `G` (8–16) completions/prompt with
  HF `.generate`. On one A5000, batched 2B decode ≈ 30–60 tok/s/seq; a ~500-tok completion ≈ 10–20 s,
  so a group of 8 ≈ 1–3 min/prompt/step. Reward exec adds ~2–10 s/completion (isolated CadQuery),
  parallelizable across CPU (32 cores). → order **hundreds of prompt-steps/day/GPU**, i.e. slow.
- **2×A5000 plan**: GPU0 = actor train (LoRA GRPO, bf16, grad-ckpt), GPU1 = a `.generate` sampler,
  CPU pool = reward. Realistic: **~1–3k prompt-updates/day**. Enough for hard-example RL on the
  dev set, not for from-scratch RL.
- **Main risks**: (1) reward hacking — un-normalized voxel IoU can be gamed by a giant box
  overlapping the GT bbox; clamp with the validity gate + bbox-mm penalty. (2) exec latency/hangs
  dominating step time (mitigated by the process-timeout already in `eval_cq.py`). (3) KL / entropy
  collapse on a tiny prompt set → keep the SFT model as a frozen reference, low LR, small `G`.
  (4) long prompts (median ~3 k tok) make each generated group memory-heavy — cap prompt length.
- **Verdict**: **feasible but slow** on 2×A5000; ~1–2 days to wire TRL GRPO + `eval_cq` reward +
  a sampler loop. Best sequenced **after** SFT saturates and the noise-injection study says how
  much headroom RL must recover. Recommend SFT-first; revisit RL then.
