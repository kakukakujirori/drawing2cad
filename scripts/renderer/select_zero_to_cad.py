#!/usr/bin/env python
"""Stage a DIVERSE random sample of **Zero-To-CAD-1m** seed solids as STEP files.

Zero-To-CAD (`ADSKAILab/Zero-To-CAD-1m`, Apache-2.0) ships each part already as a
binary `step_file` (OpenCASCADE STEP) plus its generating `cadquery_file` (the GT
program) — NO STL->STEP conversion is needed (unlike the README's older note). This
script pulls `step_file` bytes out of the locally-cached HF dataset and writes one
`{uuid}.step` per part into <stage_dir>, mirroring select_fusion360_recon.py so the
same `batch_dataset.py <stage_dir> <out_dir>` renders them.

It ALSO writes `{uuid}.cadquery.py` (the GT CadQuery program) next to each STEP, so the
stage dir is a paired (seed STEP -> AMVDG) + (GT 3D program) source for the future
AMVDG->3D leg. Pass --no-code to skip.

Diversity: Zero-To-CAD has no sub-step grouping (cf. Fusion360). We random-sample across
the split and keep a face-count band [--min-faces, --max-faces] to drop trivial solids
(a few faces = a bare block/cylinder) and cap giant B-reps that make HLR slow.

Needs the `datasets` library (NOT in the `drawing2cad` FreeCAD env — run this with an env
that has it, e.g. `py312`). Only `batch_dataset.py` needs FreeCAD.

Usage:
  python scripts/renderer/select_zero_to_cad.py <N> <stage_dir> \
      [--split validation] [--seed 0] [--min-faces 6] [--max-faces 200] [--no-code]
Then (in the drawing2cad env):
  python scripts/renderer/batch_dataset.py <stage_dir> <out_dir>
"""
import os, sys, random, argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int, help="number of parts to stage")
    ap.add_argument("stage_dir", type=str)
    ap.add_argument("--dataset", default="ADSKAILab/Zero-To-CAD-1m")
    ap.add_argument("--split", default="validation",
                    help="validation/test (10k each) or train (~980k); default validation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-faces", type=int, default=6)
    ap.add_argument("--max-faces", type=int, default=200)
    ap.add_argument("--no-code", action="store_true", help="don't also write {uuid}.cadquery.py")
    a = ap.parse_args()

    from datasets import load_dataset
    # select only the light columns — a bare row also carries 8 PNGs + an STL,
    # so pulling those per-row would be ~1000x slower.
    cols = ["uuid", "num_faces", "step_file"] + ([] if a.no_code else ["cadquery_file"])
    ds = load_dataset(a.dataset, split=a.split).select_columns(cols)
    n_total = len(ds)

    rng = random.Random(a.seed)
    order = list(range(n_total))
    rng.shuffle(order)

    os.makedirs(a.stage_dir, exist_ok=True)
    for old in os.listdir(a.stage_dir):
        if old.endswith(".step") or old.endswith(".cadquery.py"):
            os.unlink(os.path.join(a.stage_dir, old))

    kept = scanned = skipped_band = skipped_empty = 0
    for idx in order:
        if kept >= a.n:
            break
        row = ds[idx]
        scanned += 1
        nf = row["num_faces"]
        if nf is None or nf < a.min_faces or nf > a.max_faces:
            skipped_band += 1
            continue
        sf = row["step_file"]
        if not sf:
            skipped_empty += 1
            continue
        uid = row["uuid"]
        with open(os.path.join(a.stage_dir, uid + ".step"), "wb") as f:
            f.write(sf)
        if not a.no_code:
            code = row.get("cadquery_file")
            if code:
                try:
                    txt = code.decode("utf-8") if isinstance(code, (bytes, bytearray)) else str(code)
                    with open(os.path.join(a.stage_dir, uid + ".cadquery.py"), "w") as f:
                        f.write(txt)
                except Exception:
                    pass
        kept += 1
        if kept % 50 == 0:
            print("  staged %d/%d (scanned %d)" % (kept, a.n, scanned), flush=True)

    print("split=%s total=%d scanned=%d staged=%d (band[%d,%d] skip=%d, empty=%d) -> %s (seed=%d)"
          % (a.split, n_total, scanned, kept, a.min_faces, a.max_faces,
             skipped_band, skipped_empty, a.stage_dir, a.seed), flush=True)
    if kept < a.n:
        print("WARNING: only staged %d of requested %d (band too tight or split exhausted)"
              % (kept, a.n), flush=True)


if __name__ == "__main__":
    main()
