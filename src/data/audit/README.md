# GT solid audit

Rule-based B-rep trust audit for a staged Zero-To-CAD directory (a folder of
`{uuid}.step` + `{uuid}.cadquery.py` pairs, e.g. anything under
`/disk2/drawing2cad_experiments/stage_z2c_*`). Built because the training
corpus is 100% synthetic/procedurally generated and nothing in the pipeline
had checked, beyond `shape.isValid()`, whether the resulting solids are
actually sound.

For every pair it: executes the GT CadQuery program locally, runs the same
geometry check battery on both the executed shape and the shipped STEP,
diffs the two, and traces whether each chained solid-modifying call in the
program actually changed the shape. See the module docstrings for the full
rationale and the exact OCCT/CadQuery APIs used:

- `solid_checks.py` — the check battery (soft OCCT Boolean-argument
  self-interference, hard downstream mesh validity, non-manifold edges,
  open/unsolidified boundaries, kernel tolerance blow-up, micro edges/faces,
  sampled wall thickness, connected-component count) and the HARD_INVALID /
  SOFT_SUSPECT / OK / UNAUDITABLE severity classifier.
- `op_trace.py` — monkeypatch-based tracer: did each `cut`/`union`/`fillet`/...
  call in the GT program actually change the shape it was chained onto.
- `gt_audit.py` — the multiprocessing CLI that runs both over a directory,
  with checkpointed/resumable output and two-layer timeout handling for
  pathological samples.

## Quick start

```bash
python src/data/audit/gt_audit.py \
    --stage-dir data/z2c_train/target \
    --out-dir data/z2c_train/target_audit \
    --workers 30 \
    --task-timeout-s 45
```

A `tqdm` progress bar runs for the whole audit. Safe to re-run with the same
`--out-dir`: already-completed uuids (by `results.jsonl` line) are skipped,
so a Ctrl-C or crash only costs whatever was in flight. `--limit N` audits
only the first N (seed-shuffled, not a raw directory-listing prefix) pairs —
useful for a quick look before committing to a full run.

## Output (in `--out-dir`)

- `results.jsonl` — one full record per uuid (both shapes' check batteries,
  the divergence between them, the op-contribution trace, the final verdict).
  The STEP always gets the full battery; the executed code always gets the
  full topology and downstream-mesh battery, but its sampled wall-thickness
  pass runs only when the code's shape diverges from the STEP — a matching
  signature means the two describe the same solid, so the STEP-side thickness
  sample already covers it.
  The `code_thickness_audited` field records whether that pass ran for the code
  shape. This is the source of truth; everything else is derived from it via
  `aggregate()`, which you can re-run standalone with `--report-only` (e.g.
  after loosening a threshold) without re-scanning the whole corpus.
- `stats.json` — severity counts/rates, a reason-code histogram, the
  BRepCheck defect taxonomy summed across every sample, and percentile
  summaries (thickness, tolerance, aspect ratio, divergence).
- `hard_invalid.txt` / `soft_suspect.txt` / `ok.txt` / `unauditable.txt` —
  one `uuid\tstep_path\tcode_path` line per sample in that tier (tab-separated,
  `#`-commented header; `cut -f2` for a plain file list). Each file is named
  for the exact `severity` field value it lists, so the file name is 1:1 with
  `results.jsonl` and with what the training gate selects on (below). The four
  files partition the corpus (`ok.txt` = `results.jsonl` minus the other
  three). They are a **human-facing convenience view for eyeballing, not a
  pipeline input** — `results.jsonl` stays the source of truth, and the
  consuming filter reads it directly rather than these text files.

## Consuming the audit (training / eval gate)

The audit is a gate, not just a report. Both the SFT dataloader
(`src/data/factory.py`) and the validation benchmark
(`src/evaluation/generate.py`) filter samples through
`src.data.audit.gate.gate_present_ids`, driven by the `audit` block in
`configs/data/z2c.yaml`:

```yaml
audit:
  train_dir: .../data/z2c_train/target_audit   # a gt_audit.py --out-dir
  val_dir:   .../data/z2c_val/target_audit
  allow_reasons: [bop_self_interference, thin_wall, noop_operation, micro_edge,
                  micro_face, tolerance_bloat, kernel_small_edge]
  allow_hard: false
  allow_unauditable: false
# audit: null   # disable the gate entirely
```

A sample is kept iff: `ok` (always) · `soft_suspect` **and every one of its
reason codes is in `allow_reasons`** (set-subset — one disallowed reason drops
it) · `hard_invalid` only if `allow_hard` · `unauditable` only if
`allow_unauditable`. The gate is fail-closed: `results.jsonl` must cover the
whole staged corpus, so **any** present (rendered + manifest) sample missing
from it raises immediately — a count mismatch means the audit is incomplete or
mis-pointed, and is never silently absorbed by shrinking the corpus. Because it
selects on reason codes, this expresses cutoffs the flat `soft_suspect.txt`
cannot — e.g. keep
`thin_wall`/`noop_operation` but exclude `gt_mismatch` (the one soft reason
that corrupts the SFT label, since the executed code and the shipped STEP then
describe different objects). Changing the policy re-tunes training with no
re-render, since it runs at dataloader-build time.

## Severity tiers

- **hard_invalid** → `hard_invalid.txt`. The shape itself is broken or cannot
  be consumed by the evaluation geometry path: invalid/non-watertight mesh,
  non-manifold or open/unsolidified boundary, fragmentation into >1 disjoint
  solid, zero/negative volume, or STEP/code load/execute failure. Excluded by
  default (kept only with `allow_hard`). `mesh_not_watertight` uses the same
  CadQuery tessellation + processed-Trimesh check as the evaluator.
- **soft_suspect** → `soft_suspect.txt`. A technically valid solid with a smell:
  OCCT Boolean-argument self-interference (`bop_self_interference`), thin walls,
  micro edges/faces, kernel tolerance blow-up, a no-op operation in the GT
  program, or the executed code diverging from the shipped STEP
  (`gt_mismatch` — the SFT target text and the rendered input drawing would
  describe different objects; on the one sample seen with real divergence in
  the smoke test it did not correspond to any other flag, so it is worth
  checking on its own). These are review/downweight candidates, not
  automatic exclusions — the right cutoff is a training-policy call this
  tool deliberately does not make for you.
- **unauditable** → `unauditable.txt`. The harness itself didn't finish
  (timeout or an unexpected exception), not a claim that the data is bad.
  Excluded by default (kept only with `allow_unauditable`).

**Caveat on `thin_wall`**: it is a sampled ray-cast proxy (see
`sample_wall_thickness`'s docstring), not an exact thickness field. It will
false-positive on legitimate acute corners/wedges, so treat its prevalence
as an upper bound worth spot-checking, not a precise count.

## Tests

```bash
python -m pytest tests/data/audit/ -v
```

Constructs known-good/known-bad shapes directly (self-intersecting bowtie,
open shell, degenerately-touching solids, thin plate, micro chamfer) rather
than depending on any staged directory, so they run anywhere cadquery is
installed.
