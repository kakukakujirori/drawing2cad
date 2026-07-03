# AMVDG → 3D CAD learning harness (`scripts/train3d/`)

Trains and evaluates the **back leg** of the drawing2cad pipeline:
`AMVDG graph (g2 text) → CadQuery code → 3D solid`. Text-only by design — **no drawing
PNG is fed** — so a working leg is evidence the AMVDG IR is *sufficient* to recover 3D.

- **Input** = `scripts/amvdg/serialize_g2.graph_to_g2(graph)` (g2 text; part-frame mm,
  each axis bbox-min = 0; `Ck` tags mark the same 3D feature across views).
- **Target** = GT CadQuery (`experiments/stage_z2c/{uuid}.cadquery.py`, assigns `result`).
- **Start ckpt** = [`ADSKAILab/Zero-To-CAD-Qwen3-VL-2B`](https://huggingface.co/ADSKAILab/Zero-To-CAD-Qwen3-VL-2B)
  (image→CadQuery SFT; same base *and* same output format as us), swappable via `--ckpt`.
- Pairs: 274 uuids that have **both** a `dataset_z2c/*.graph.json` and a `stage_z2c/*.cadquery.py`.

## Files
| file | env | what |
|---|---|---|
| `build_dataset.py` | any w/ `transformers` | graph.json → g2 → `{id,input_text,target_code,n_tok_*}` jsonl + train/val split + token report |
| `train_sft.py` | `drawing2cad-train` | text-only SFT (bf16, completion-only loss, LoRA/full, 1-/2-GPU), eval hook → `eval_cq.py` |
| `eval_cq.py` | `drawing2cad` | isolated exec → validity + **translation-aligned voxel IoU (abs mm)** + bbox-mm error → JSON |
| `data_z2c/` | — | built bundle: `train.jsonl` (247) · `val.jsonl` (27) · `all.jsonl` (274) · `stats.json` |

## Environments
Two envs, split by dependency (recorded per the repo constraint on new envs):
- **`drawing2cad-train`** — *created here* by `conda create -n drawing2cad-train --clone py312`
  then `pip install peft`. Gives torch 2.9.1+cu128 / transformers **4.57.3** (== the ckpt's
  `transformers_version`, so Qwen3-VL loads natively) / accelerate / datasets / peft 0.19.1.
  Used by `train_sft.py` (+ `build_dataset.py`). No cadquery here → eval is a subprocess.
- **`drawing2cad`** — existing (conda-forge FreeCAD 1.0.2 + cadquery 2.8.0 + trimesh 4.12 +
  numpy + scipy). Used by `eval_cq.py`. No torch. `manifold3d` absent, but the voxel-IoU
  path never calls a trimesh boolean, so it is not needed.

`train_sft.py` shells out to `eval_cq.py` via `$DRAWING2CAD_PY`
(default `/home/ryotaro/miniforge3/envs/drawing2cad/bin/python`).

## Run
```bash
# 1. build the jsonl bundle + token report  (py312 or drawing2cad-train)
python scripts/train3d/build_dataset.py \
    --graph-dir experiments/dataset_z2c --code-dir experiments/stage_z2c \
    --out scripts/train3d/data_z2c

# 2a. SFT smoke — LoRA, 40 steps, one A5000 (proves the pipe + loss drop)
CUDA_VISIBLE_DEVICES=1 python scripts/train3d/train_sft.py --smoke \
    --data scripts/train3d/data_z2c --gt-dir experiments/stage_z2c --out runs/smoke

# 2b. real single-GPU LoRA run (max-len defaults to 8192, the decided cap)
CUDA_VISIBLE_DEVICES=1 python scripts/train3d/train_sft.py \
    --data scripts/train3d/data_z2c --epochs 3 --out runs/lora

# 2c. two-GPU DDP (LoRA or --full):        torchrun handles world size
torchrun --nproc_per_node=2 scripts/train3d/train_sft.py --full \
    --data scripts/train3d/data_z2c --epochs 3 --bs 1 --grad-accum 8 --out runs/full
#   NOTE: full-FT of a 2B with AdamW replicates ~22 GB of optimizer state PER GPU under
#   plain DDP -> OOMs a 24 GB A5000. To fit "modest full-FT" pick ONE of:
#     --optim adafactor            # ~no 2nd-moment state; full-FT fits one 24 GB GPU
#     accelerate launch --fsdp ... # shard AdamW state across the 2 A5000s
#   LoRA (default) fits comfortably and is the recommended path on this box.

# 3. evaluate any dir of predicted {id}.py against GT {id}.step  (drawing2cad env)
python scripts/train3d/eval_cq.py --pred-dir runs/lora/preds_stepN \
    --gt-dir experiments/stage_z2c --out runs/lora/metrics.json
# GT-code self-IoU sanity (expect mean IoU ~1.0):
python scripts/train3d/eval_cq.py --pred-dir experiments/stage_z2c \
    --gt-dir experiments/stage_z2c --ids-file <uuids> --limit 20 --out self_iou.json
```

## Measured (this session)

**GT corpus (274 paired, tokenizer = Zero-To-CAD-Qwen3-VL-2B):**
- target CadQuery: median **463** tok (max 1067); `result` is the final variable in **273/274**
  (2 files also bind a stray `r`); imports are stdlib only (`cadquery`, `math`,
  `types.SimpleNamespace`, `dataclasses`, `itertools`, `copy`) → `exec` is self-contained.
- g2 **input** is the length driver. v1 measured median **4066** tok / max 42811 (41% fit a
  4096 cap) → prompted the **g2.1 slimming** in `serialize_g2.py` (hid≡vis duplicate rows
  merged as `vh`, id column dropped, DIM refs → measured spans). After g2.1:
  median **2913** tok, p90 7520, p95 12849, max 35335.

**`max_new_tokens` / truncation policy (decided):**
- `--max-new-tokens = 1024` (covers **99.6%** of targets; 1280 → 100%). cadrille's 768 would
  clip ~5% of our targets (it already truncated 6/49 on its own data).
- Over-length pairs are **filtered, not truncated** — truncating the front would corrupt the
  g2 input, truncating the back would cut the target. **Decided: `--max-len 8192`** (now the
  default), which with g2.1 keeps **246/274 (90%)**; 4096 keeps 169, 6144 keeps 228. The
  dropped count is printed every run. LoRA is the main path (user decision 2026-07-03);
  the FSDP full-FT reference run is deferred until the dataset is scaled up.

**`eval_cq.py` self-IoU sanity + GT-corpus executability (GT code vs GT STEP, all 274):**
exec_ok **98.2%** (269/274), strict-valid (isValid ∧ Vol>0 ∧ watertight) **90.2%**, mean voxel
IoU **0.996** / median **1.0** / 97.5% ≥ 0.9, mean max-bbox-err **0.0002 mm**. The 5 non-exec =
4 heavy OCC parts over the 30 s timeout + 1 API edge case. Confirms (a) the
exec→tessellate→voxel-IoU→bbox pipeline is correct at scale, and (b) the training targets are
clean. (Run sharded ×8 to parallelize; a single serial pass is ~40 min.)

**SFT smoke (LoRA r16, 40 steps, one A5000, 24 shortest examples, 242 s):** untrained
forward-loss ≈1.49 → first logged optimizer step 0.35 → **0.06** (overfits 24 samples by
design). The **eval hook ran end-to-end** on 4 unseen val graphs (generate → subprocess
`eval_cq`): **valid_rate 1.0, mean voxel IoU 0.24, median 0.19, mean bbox err 13.5 mm** —
i.e. even a 40-step overfit emits valid CadQuery that partly generalizes to held-out graphs,
so the full load → completion-loss → step → greedy-generate → `eval_cq` loop is proven.

## Design notes / recipe deltas

We **reference** cadrille and Zero-to-CAD recipes but **do not reuse their code**.

| knob | cadrille (Qwen2-VL-2B) | Zero-to-CAD (Qwen3-VL-2B) | **ours (this harness)** | why |
|---|---|---|---|---|
| input | 4-view render → 2×2 grid, video path | 8 views 256² | **g2 text only** | test IR sufficiency; no pixels |
| base | Qwen2-VL-2B-Instruct | Qwen3-VL-2B-Instruct | **Zero-To-CAD-Qwen3-VL-2B** (→ base fallback) | warm-start on same output format |
| tf version | **4.50.3 pinned** (subclass hooks) | 4.57.3 | **4.57.3** | Qwen3-VL needs ≥4.57; matches py312 |
| tuning | full FT | full FT | **LoRA (default) / full** | 2×A5000: LoRA fits easily, full is tight |
| optimizer | AdamW | AdamW | AdamW (`adamw_torch`) | — |
| lr / sched | 2e-4 / cosine, warmup 1000 | 1e-4 / cosine, warmup 0.03 | **2e-4 LoRA · 1e-4 full / cosine, warmup 0.03** | adopt Zero-to-CAD for full (same base) |
| weight decay | 0.01 | 0.0 | **0.0** | follow Zero-to-CAD |
| eff. batch | ~30–32 | 16 (1×16 DDP) | 8 (bs1×ga8, tune up) | 24 GB budget |
| max seq len | dynamic pad (no cap) | 4096 | **8192 (filter over-len)** | long g2 inputs; g2.1 slim → 90% fit (see above) |
| loss mask | completion-only, hard-coded ids | (n/a in card) | **completion-only via chat-template prefix length** | tokenizer-agnostic, no magic ids |
| gen | greedy, `max_new_tokens=768` | greedy, 4096 | greedy, **1024** | 99.6% target coverage |
| result var | `r` | `result` | accept **`result`→`r`→last shape** | GT uses `result` |
| **eval IoU** | mesh boolean, **unit-box normalized** | voxel 64³, **rot-aligned, normalized** | **voxel 64³, translation-aligned, ABSOLUTE mm** + bbox-mm err | scale is the novelty; g2↔GT differ by translation only |
| exec isolation | `Process`+join(3 s) | — | **`Process`+join(30 s)+terminate** | same containment, longer for heavy Z2C parts |

**What is portable vs rewritten from cadrille.** Portable *ideas*: completion-only masking,
separate-process exec-with-timeout, `isValid`/`Volume` validity gate. Rewritten: cadrille's
`Cadrille(Qwen2VLForConditionalGeneration)` subclass, its point-cloud `FourierPointEncoder`,
the `pixel_values_videos` collator, and the hard-coded `151644/77091/151645` label search are
**all Qwen2-VL / multimodal specific** and unused here — the text-only leg is a plain causal-LM
SFT on Qwen3-VL, so none of that code transfers. cadrille's `evaluate.py` IoU is deliberately
*replaced* (its unit-box rescale discards the absolute size this project exists to recover).

**Checkpoint verification (`ADSKAILab/Zero-To-CAD-Qwen3-VL-2B`).** Exists (16 files, 4.0 GB);
`Qwen3VLForConditionalGeneration`, base `Qwen/Qwen3-VL-2B-Instruct`, `transformers_version
4.57.3`, **Apache-2.0**. Loads text-only in `drawing2cad-train` (forward-with-loss + greedy
generate verified, 4.4 GB inference). Its card publishes the SFT recipe adopted above and an
eval protocol (voxel IoU 64³, success-rate) that we follow — but *un-normalized*.

## RL feasibility memo (design only — NOT implemented; user decides later)

cadrille's RL stage (paper 2505.22914; code unreleased) rewards `10·IoU − 10·invalid`, GRPO-family
(Dr.CPPO), with hard-example mining (keep prompts whose SFT IoU < 7.5/10). The reward needs **only
a mesh**, not a target sequence — a perfect fit for our synthetic pipeline (every g2 has a GT STEP,
and `eval_cq.py` already returns exactly that reward signal). To build it ourselves:

- **Trainer**: TRL `GRPOTrainer`. `reward_funcs` = a Python callable wrapping `eval_cq.py`'s
  per-sample validity + translation-aligned voxel IoU (→ `10·IoU − 10·(1−valid)`); prompts = g2
  text; no reference completions. Needs `pip install trl` into `drawing2cad-train` (not yet added).
- **Generation throughput (no vLLM — py312/train env has none)**: GRPO samples `G` (8–16)
  completions/prompt with HF `.generate`. On one A5000, greedy-ish 2B decode ≈ 30–60 tok/s/seq
  batched; a ~500-tok completion ≈ 10–20 s, so a group of 8 ≈ 1–3 min/prompt/step even batched.
  Reward exec adds ~2–10 s/completion (isolated CadQuery), parallelizable across CPU (32 cores).
  → order **hundreds of prompt-steps/day/GPU**, i.e. slow. vLLM would ~5–10× generation but is
  absent and non-trivial to add for a custom-loaded VL checkpoint.
- **2×A5000 plan**: GPU0 = actor train (LoRA GRPO, bf16, grad-ckpt), GPU1 = a `.generate` sampler,
  CPU pool = reward. Realistic: **~1–3k prompt-updates/day**. Enough for hard-example RL on the
  274-part dev set (and later a few-k-part slice), not for from-scratch RL.
- **Main risks**: (1) reward hacking — un-normalized voxel IoU can be gamed by a giant box
  overlapping the GT bbox; clamp with the validity gate + bbox-mm penalty. (2) exec latency/hangs
  dominating step time (mitigated by the process-timeout already in `eval_cq.py`). (3) KL / entropy
  collapse on a tiny prompt set → keep the SFT model as a frozen reference, low LR, small `G`.
  (4) long g2 prompts (median 4 k tok) make each generated group memory-heavy — cap prompt length.
- **Verdict**: **feasible but slow** on 2×A5000; ~1–2 days of engineering to wire TRL GRPO +
  `eval_cq` reward + a sampler loop. Best sequenced **after** SFT saturates and the §D-1 noise-
  injection study says how much headroom RL must recover. Recommend SFT-first; revisit RL then.
```
