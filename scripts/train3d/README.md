# AMVDG → 3D CAD learning harness (`scripts/train3d/`)

Trains and evaluates the **back leg** of the drawing2cad pipeline: `AMVDG graph (serialized text) → CadQuery code → 3D solid`. Text-only by design — **no drawing PNG is fed** — so a working leg is evidence the AMVDG IR is *sufficient* to recover 3D.

- **Input** = `scripts/train3d/serialize.graph_to_text(graph)` (part-frame mm, each axis bbox-min = 0; `Ck` tags mark the same 3D feature across views).
- **Target** = GT CadQuery (`experiments/stage_z2c_{train,val}/{uuid}.cadquery.py`, assigns `result`).
- **Start ckpt** = [`ADSKAILab/Zero-To-CAD-Qwen3-VL-2B`](https://huggingface.co/ADSKAILab/Zero-To-CAD-Qwen3-VL-2B) (image→CadQuery SFT; same base *and* same output format as us), swappable via `--ckpt`.
- **Data** = the two per-split bundles from **Data Preparation step 3** (top-level README):
  `experiments/data_z2c_train` (≈2825, Zero-To-CAD train split) and `experiments/data_z2c_val`
  (≈274, val split). Each is `all.jsonl` + `stats.json`; the SFT train/val sets **are** these
  source splits (no extra split is carved inside a bundle).

## Files
| file | what |
|---|---|
| `build_dataset.py` | one graph-dir + code-dir → `all.jsonl` (`{id,input_text,target_code,n_tok_*}`) + `stats.json` token report. Run once per split — see top-level README step 3. |
| `serialize.py` | this leg's AMVDG-graph ↔ model-input text codec (`graph_to_text` + inverse for round-trip). `python serialize.py GRAPH.json` prints it; `--check 'GLOB'` validates round-trip + cross-view consistency dataset-wide. |
| `train_sft.py` | text-only SFT (bf16, completion-only loss, LoRA/full, 1-/2-GPU); `--train`/`--val` bundles. **Periodic in-training eval** (every `--eval_every_epochs`, plus train-end): batched greedy-generate a fixed seeded RANDOM val subset (`--eval_val_n`, default 48; 0 = full val) via the *shared* `iter_batched_generate` (same code path as `infer.py`), score with `eval_cq.py`, log folded metrics to TensorBoard, and keep a **best-model checkpoint** (`<out>/best/` + `<out>/best_meta.json`) on the `--best_metric` scalar. Args are an `ExpConfig` dataclass (`HfArgumentParser`): every flag is also settable from a `--config FILE.{json,yaml}` (CLI overrides file); resolved config → `<out>/config.json`, metrics → TensorBoard `<out>/logs` (or `--report-to wandb`, offline). |
| `infer.py` | batch inference: AMVDG JSON → CadQuery `{stem}.py` + executed `{stem}.step`, with a trained ckpt (LoRA adapter dir *or* full ckpt / HF id). Batched generation is the shared `train_sft.iter_batched_generate` (length-sort + left-pad); isolated timeout'd exec; writes `infer_summary.json` (exec/step rates + error hist). Naming pairs with GT `{uuid}.step` so `eval_cq.py` scores its `.py` outputs directly. |
| `eval_cq.py` | isolated exec → validity + **translation-aligned voxel IoU (abs mm)** + bbox-mm error → JSON. |

## Run
```bash
# bundles come from Data Preparation step 3 (top-level README):
#   experiments/data_z2c_train  +  experiments/data_z2c_val

# 1. SFT smoke — LoRA, 40 steps, one A5000 (proves the pipe + loss drop)
python scripts/train3d/train_sft.py --smoke \
    --train experiments/data_z2c_train \
    --val experiments/data_z2c_val \
    --out experiments/train3d/smoke

# 2. real single-GPU LoRA run.
#    --max_len 16384: hard parts (faces median 127) serialize to ~10k tokens; at the old
#      8192 cap ~84% are FILTERED out of training (build_labels drops, never truncates).
#      16k lets them in; --attn auto uses flash_attention_2 if `pip install flash-attn`,
#      else sdpa (already mem-efficient O(n), so 16k fits one 24 GB A5000 with grad-ckpt).
#    --out is optional: omit it and the run lands in experiments/train3d/<YYYY-MM-DD_HH-MM-SS>/
python scripts/train3d/train_sft.py \
    --train experiments/data_z2c_train \
    --val experiments/data_z2c_val \
    --epochs 3 \
    --max_len 16384 \
    --attn auto

# 3. two-GPU DDP (LoRA recommended; --full optional): torchrun handles world size.
#    DDP = throughput (each GPU a full replica on different samples), not memory relief;
#    16k LoRA already fits ONE 24 GB A5000, so no sharding needed.
torchrun --nproc_per_node=2 scripts/train3d/train_sft.py \
    --train experiments/data_z2c_train \
    --val experiments/data_z2c_val \
    --epochs 3 \
    --bs 1 \
    --grad-accum 8 \
    --max_len 16384 \
    --attn auto
#   NOTE: full-FT of a 2B with AdamW replicates ~22 GB of optimizer state PER GPU under plain DDP -> OOMs a 24 GB A5000. To fit "modest full-FT" pick ONE of:
#     --optim adafactor            # ~no 2nd-moment state; full-FT fits one 24 GB GPU
#     accelerate launch --fsdp ... # shard AdamW state across the 2 A5000s
#   LoRA (default) fits comfortably and is the recommended path on this box.

# 3b. drive train_sft entirely from a config file (CLI flags still override it):
python scripts/train3d/train_sft.py --config my_run.yaml --epochs 5
#   watch it:  tensorboard --logdir <out>/logs   (resolved config -> <out>/config.json)

# In-training eval + best-model checkpointing (knobs, all optional):
#   --eval_val_n 48        fixed val subset size, sampled ONCE with a fixed seed (identical
#                          across evals and across same-seed runs); 0 = whole val set.
#   --eval_seed N          seed for that sample (default: reuse --seed).
#   --eval_every_epochs 1  run the eval hook every N epochs (float ok); <=0 = train-end only.
#   --eval_batch_size 8    prompts per generate() call in the hook (conservative vs infer's 16).
#   --eval_max_batch_tokens 24000   padded-token budget/batch (~7-8 GB peak; half infer's 48000
#                          because optimizer state + params are GPU-resident during training).
#   --best_metric mean_iou_incl_fail   scalar tracked for best-model saving (higher=better):
#                          mean_iou_incl_fail (default; mean IoU over the WHOLE subset with
#                          failed/invalid preds counted as 0 — folds validity+geometry) |
#                          valid_rate | median_iou.
#   (--n_eval is a DEPRECATED alias of --eval_val_n; still honored with a warning.)
# Each eval writes <out>/preds_step{N}/ + <out>/eval_step{N}.json (scored by eval_cq against
# --gt_dir); the best checkpoint lands in <out>/best/ (LoRA adapter or full weights) with
# <out>/best_meta.json = {step, epoch, metric_name, metric_value, history:[per-eval records]}.
# Metrics stream to TensorBoard as train/eval/<metric>. Under DDP only rank0 evaluates/saves.

# 4. batch-infer a trained ckpt over the val AMVDG graphs -> {uuid}.py + {uuid}.step
python scripts/train3d/infer.py \
    --ckpt experiments/train3d/lora_v4/final \
    --input experiments/dataset_z2c_val \
    --out experiments/train3d/lora_v4/preds_full
#   --ckpt takes a LoRA adapter dir (adapter_config.json -> loads its base) or a full ckpt/HF id.
#   Generation is BATCHED (length-sorted, left-padded): --batch-size (default 16) prompts per
#   generate() call, capped by --max-batch-tokens (default 48000, ~14.6 GB peak on a 24 GB A5000;
#   ~0.21 MB/token so 64000≈18 GB) so long prompts form smaller batches; exec runs in parallel
#   (--workers, default min(8,cpu)). Greedy is bf16-batched so individual outputs differ run-to-run
#   vs bs=1, but aggregate exec/IoU rates match.

# 5. evaluate a dir of predicted {id}.py against GT {id}.step (full val = infer.py preds)
#    THIS prints + writes the eval numbers (valid_rate, mean/median IoU, bbox-mm err); step 4 only
#    produces the preds. eval_cq re-execs each {id}.py itself, so run 4 then 5.
python scripts/train3d/eval_cq.py \
    --pred-dir experiments/train3d/lora_v4/preds_full \
    --gt-dir experiments/stage_z2c_val \
    --out experiments/train3d/lora_v4/eval_full.json
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
