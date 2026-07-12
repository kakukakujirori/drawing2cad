# `bench/` — our pseudo-CADGenBench

A local, CADGenBench-grade benchmark built from a **complexity-matched sample of
the Zero-to-CAD (Z2C) validation split**, to test whether our AMVDG→3D route
beats a strong zero-shot vision-LLM baseline. Two candidate systems are scored
with **CADGenBench's own CAD Score** against our own GT STEP files:

- **(i) Ours** — GT AMVDG graph → trained model → CadQuery → STEP.
  Using the **GT** AMVDG is intentional: it measures the ceiling of our route *"if the 2D leg were perfect"* — an upper-bound / possibility check.
- **(ii) LLM/agent baseline** — the **`agy` CLI agent** (Antigravity CLI) run headless, one prompt per fixture, input = the rendered drawing PNG. It writes and executes a CadQuery script and exports a STEP. See `run_baseline_agy.py`.

Metric = **CADGenBench CAD Score**, computed locally via `cadgenbench evaluate`.


## Layout

```bash
bench/
  common.py            # shared paths/helpers
  build_fixtures.py    # select N parts from a source set (--step-dir/--graph-dir) w/ BOTH
                       #   GT STEP and drawing+graph -> data/inputs/<uuid>/{description.yaml,
                       #   drawing.png}, data/gt/<uuid>/ground_truth.step, data/manifest.json
                       #   (manifest records seed, band, source, per-fixture OCC face counts)
  run_ours.py          # infer.py on the AMVDG graphs (drawing2cad env, GPU) ->
                       #   results/ours_noblend_v1/<uuid>/output.step
  run_baseline_agy.py  # agy CLI agent, one headless prompt/fixture, backend=cadquery ->
                       #   results/agy_<modelslug>/<uuid>/output.step  (CURRENT, resumable)
  run_baseline_ollama.py # LEGACY: cadgenbench baseline agent + ollama vision model ->
                       #   results/<timestamp>_<model_slug>/<uuid>/output.step
  evaluate.py          # cadgenbench evaluate over a results dir -> per-fixture result.json + run_summary.json
  report.py            # aggregate ours-vs-baseline -> results/report.{json,md}
  data/                # generated (gitignored)
  results/             # generated (gitignored)
```


## How to run

### Prepare Benchmark Data

The data is extracted from Zero-to-CAD validation set, with model complexity matching the original CADGenBench.

```bash
python scripts/renderer/select_zero_to_cad.py \
    --n 65 \
    --stage_dir experiments/stage_z2c_bench \
    --split validation \
    --seed 0 \
    --min-faces 38 \
    --max-faces 1200 \
    --stratify \
    --bin-edges 55,76,100,153,219,348,460,650,900

python scripts/renderer/batch_dataset.py \
    --step_dir experiments/stage_z2c_bench \
    --out_dir experiments/dataset_z2c_bench

python scripts/train3d/build_dataset.py \
    --graph-dir experiments/dataset_z2c_bench \
    --code-dir experiments/stage_z2c_bench \
    --out experiments/data_z2c_bench

# Build fixtures
python bench/build_fixtures.py \
    --step-dir experiments/stage_z2c_bench \
    --graph-dir experiments/dataset_z2c_bench
```

### Model Inference and Evaluation

```bash
# Ours.
python bench/run_ours.py \
    --ckpt experiments/train3d/[yyyy-mm-dd_hh-mm-ss]/best \
    --run-name [yyyy-mm-dd_hh-mm-ss] \
    --quant 1024  # Align with the training config

# Gemini. FIRST sign in once: run `agy` interactively (see "Baselines").
python bench/run_baseline_agy.py --dry-run --limit 1   # inspect cmd+prompt, run nothing
python bench/run_baseline_agy.py --limit 1             # smoke (needs authed agy)
python bench/run_baseline_agy.py --all --model <model-id>

# Score a results dir with CADGenBench's evaluator (writes result.json + run_summary.json).
python bench/evaluate.py bench/results/ours_noblend_v1
python bench/evaluate.py bench/results/agy_<modelslug>

# Side-by-side report -> results/report.{json,md}
python bench/report.py \
    --ours bench/results/[yyyy-mm-dd_hh-mm-ss] \
    --baseline bench/results/agy_<modelslug>
```


## Baselines

### Current: the `agy` CLI agent (`run_baseline_agy.py`)

The baseline is the **`agy` CLI agent** (Antigravity CLI, `~/.local/bin/agy`), run **headless, one prompt per fixture**. For each fixture the script makes a clean workdir, copies the drawing PNG in, and runs:

```
agy -p "<prompt>" --dangerously-skip-permissions --add-dir <workdir>   (cwd=workdir)
```

The prompt tells agy to read `drawing.png` (multi-view, dims in mm), write a
**CadQuery** script (`model.py`), execute it with a **specified interpreter**,
and export the solid to `output.step`. After agy exits, the script verifies
`output.step` and copies it into `results/agy_<modelslug>/<uuid>/`.
Per-fixture stdout/exit go to `<uuid>/agy.log`; the workdir is kept under
`<uuid>/work/`.

- **Kernel choice = CadQuery**, for fairness: our route emits CadQuery→OCC STEP,
  so the baseline must use the same kernel or the shape comparison is confounded.
  The prompt also pins the executing interpreter to `--cadquery-python` (default
  the **cadgenbench venv**, which has CadQuery), so both routes go through the
  *same* CadQuery/OCC install.
- **Auth required.** agy is **not signed in** out of the box. Run `agy` once
  interactively to sign in (and `agy models` to list model ids) **before** using
  this script. A headless run against an unauthenticated CLI just errors; the
  script detects this and reports `unauth` per fixture rather than hanging.
- **`--dangerously-skip-permissions` is required** for headless file-write/code-
  exec (otherwise agy blocks on approval prompts). ⚠️ In some harness auto-modes
  this flag is **denied by a classifier**, so the real run must be launched from
  a shell where the user controls agy directly.
- **Bounded.** `--print-timeout` (seconds, sent to agy as `Ns`, default 900) plus
  a hard `--wall-timeout` (default print-timeout + 120 s) so no fixture hangs.
- **Resumable.** The run dir is **date-less** (`results/agy_<modelslug>/`), so
  re-running the same command **continues where it left off**: fixtures that
  already have an `output.step` are skipped (override with `--force`). If agy hits
  a **usage / rate limit** mid-run, the loop **stops cleanly** (detected in agy's
  output) instead of burning the rest on errors — just re-run later to finish the
  remaining fixtures. `run_meta.json` tracks cumulative progress across resumes.
- **Dry run.** `--dry-run` prints the exact command + prompt per fixture and
  invokes nothing — use it to inspect before a real (costly) run.

### Legacy: ollama + CADGenBench baseline agent (`run_baseline_ollama.py`)

Superseded by agy; kept for reference/optional use. It drives CADGenBench's own reference agent with a **local ollama vision model** (`qwen2.5vl:32b`, Q4_K_M, ~23 GB VRAM — the strongest vision model fitting one 24 GB A5000) via LiteLLM, backend forced to cadquery. It needs a **private ollama server pinned to a free GPU** (the shared systemd ollama on `:11434` spreads the model across GPUs and offloads to CPU → unusably slow):

```bash
CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=127.0.0.1:11435 \
  OLLAMA_MODELS=/usr/share/ollama/.ollama/models OLLAMA_KEEP_ALIVE=30m \
  ollama serve &
```

`OLLAMA_MODELS` is **required** (else the user server reads an empty store and every request 404s). `run_baseline_ollama.py` defaults `OLLAMA_API_BASE` to that server.


## CAD Score, as computed here

CADGenBench's generation CAD Score is a weighted mean over the axes actually present: **shape 0.4 / interface 0.4 / topology 0.2**, renormalized over present axes; **0** for an invalid or missing candidate. Our fixtures carry **no interface jig sub-volumes**, so only **shape + topology** are present and the effective weights are **shape 2/3, topology 1/3**. This is a documented, honest deviation from the full 3-axis score — the interface axis is simply not available for Z2C parts.

- shape = surface-distance F1 + volume-IoU (rigid-aligned to GT).
- topology = fuzzy Betti-number (b0,b1,b2) agreement.
- `report.py` reports the mean CAD Score **including zeros** from invalid/missing fixtures (the leaderboard convention), plus a **paired** aggregate over the fixtures common to both runs.


## Estimated full-run time (50 fixtures, this box)

- **Ours:** inference is batched (~6 s/part amortized) + eval (~15 s/part / N workers). Roughly **10–20 min** total.
- **agy baseline:** dominated by the agent loop (read image → write → execute →
  fix). Per-fixture bounded by `--print-timeout` (default 900 s) + margin; budget
  several hours for 50 fixtures sequentially. Kick it off deliberately, and only
  after signing into agy.
