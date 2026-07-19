#!/usr/bin/env python
"""Stage a DIVERSE random sample of **Zero-To-CAD-1m** seed solids as STEP files.

Zero-To-CAD (`ADSKAILab/Zero-To-CAD-1m`, Apache-2.0) ships each part already as a
binary `step_file` (OpenCASCADE STEP) plus its generating `cadquery_file` (the GT
program). This script pulls `step_file` bytes out of the locally-cached HF dataset
and writes one `{uuid}.step` per part into <stage_dir>.

It ALSO writes `{uuid}.cadquery.py` (the GT CadQuery program) next to each STEP, so the
stage dir is a paired (seed STEP -> AMVDG) + (GT 3D program) source for AMVDG->3D.

We random-sample across the split and keep a face-count band [--min-faces, --max-faces] to drop trivial solids
(a few faces = a bare block/cylinder) and cap giant B-reps that make HLR slow.

By default the band is sampled IID (plain shuffle-and-take), which mirrors the split's
natural (heavily right-skewed) num_faces distribution: the vast majority of parts are
simple, so a wide --max-faces barely changes the realized sample composition. Pass
--stratify to instead draw a complexity-balanced sample: the band is cut into --n-bins
num_faces bins and a capacity-aware equal allocation ("water-filling": bins whose pool
can't fill an equal share are fully drained, the shortfall flows to the remaining bins)
spreads --n across them, so the complex tail is represented far above its natural
frequency (bounded by how many such parts actually exist in the split).

Usage:
  python scripts/renderer/select_zero_to_cad.py --n 100000 --stage_dir <dir> \
      [--split train] [--seed 0] [--min-faces 6] [--max-faces 400] [--no-code]
  # complexity-balanced instead of plain IID:
  python scripts/renderer/select_zero_to_cad.py --n 100000 --stage_dir <dir> \
      --split train --max-faces 400 --stratify [--n-bins 7] [--bin-edges 30,50,90,145,200,285]
  # preview the plan (pool sizes / allocation) without touching stage_dir:
  python scripts/renderer/select_zero_to_cad.py --n 100000 --stage_dir <dir> --stratify --dry-run
Then (in the drawing2cad env):
  python scripts/renderer/batch_dataset.py <stage_dir> <out_dir>
"""
import argparse
import json
import os
import random

import numpy as np
from datasets import load_dataset


def make_bin_edges(min_faces, max_faces, n_bins):
    """Interior num_faces edges: n_bins log-spaced bins covering [min_faces, max_faces].

    Log spacing (not the exact percentiles of any one split) so the same --n-bins
    default is reasonable for any --min-faces/--max-faces the caller picks. water_fill
    below is what actually corrects for the real, non-log-uniform pool-size skew
    across bins -- it does not depend on the edges being "well chosen"."""
    if n_bins < 2:
        return []
    lo = max(min_faces, 1)
    raw = np.geomspace(lo, max_faces, n_bins + 1)[1:-1]
    edges = sorted(set(int(round(e)) for e in raw))
    return [e for e in edges if min_faces < e < max_faces]


def water_fill(pool_sizes, n_total):
    """Capacity-aware equal allocation across bins.

    Split n_total as evenly as possible across bins, except a bin whose pool is
    smaller than its equal share is capped at its full pool and the shortfall is
    redistributed evenly across the remaining (uncapped) bins, iterated to a fixed
    point. Terminates in <= len(pool_sizes) rounds (each round either finishes or
    strictly shrinks the active set)."""
    n_bins = len(pool_sizes)
    alloc = [0] * n_bins
    active = list(range(n_bins))
    remaining = n_total
    while active and remaining > 0:
        share = remaining / len(active)
        capped = [b for b in active if pool_sizes[b] <= share]
        if not capped:
            base = remaining // len(active)
            extra = remaining - base * len(active)
            for i, b in enumerate(active):
                alloc[b] += base + (1 if i < extra else 0)
            remaining = 0
            break
        for b in capped:
            alloc[b] = pool_sizes[b]
            remaining -= pool_sizes[b]
        active = [b for b in active if b not in capped]
    return alloc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, help="number of parts to stage")
    parser.add_argument("--stage_dir", type=str)
    parser.add_argument("--split", default="validation", help="validation/test (10k each) or train (~980k); default validation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-faces", type=int, default=6)
    parser.add_argument("--max-faces", type=int, default=200)
    parser.add_argument("--no-code", action="store_true", help="don't also write {uuid}.cadquery.py")
    parser.add_argument("--stratify", action="store_true",
                         help="complexity-balanced sampling across num_faces bins instead of plain IID "
                              "(default: off => legacy behavior, unchanged file output for a given --seed)")
    parser.add_argument("--n-bins", type=int, default=7,
                         help="number of log-spaced num_faces bins for --stratify (ignored if --bin-edges given)")
    parser.add_argument("--bin-edges", type=str, default=None,
                         help="comma-separated interior num_faces edges for --stratify, "
                              "e.g. '30,50,90,145,200,285' (overrides --n-bins)")
    parser.add_argument("--dry-run", action="store_true",
                         help="print the pool/allocation plan and exit without touching --stage_dir")
    args = parser.parse_args()

    if os.path.isdir(args.stage_dir):
        raise FileExistsError(f"{args.stage_dir=} already exists. Remove it first.")
    os.makedirs(args.stage_dir)

    # select only the light columns -- a bare row also carries 8 PNGs + an STL,
    # so pulling those per-row would be much slower.
    cols = ["uuid", "num_faces", "step_file"] + ([] if args.no_code else ["cadquery_file"])
    ds = load_dataset("ADSKAILab/Zero-To-CAD-1m", split=args.split).select_columns(cols)
    n_total = len(ds)

    rng = random.Random(args.seed)
    bin_bounds = None
    bin_of_kept_idx = {}

    if args.stratify or args.dry_run:
        nf_all = np.asarray(ds["num_faces"], dtype=np.float64)  # cheap: Arrow columnar slice, no step_file decode
        in_band = (nf_all >= args.min_faces) & (nf_all <= args.max_faces)
        pool_total = int(in_band.sum())
        print("pool: split=%s band=[%d,%d] -> %d/%d parts eligible"
              % (args.split, args.min_faces, args.max_faces, pool_total, n_total))

    if args.stratify:
        if args.bin_edges:
            edges = sorted(int(e) for e in args.bin_edges.split(","))
        else:
            edges = make_bin_edges(args.min_faces, args.max_faces, args.n_bins)
        bin_bounds = [args.min_faces] + edges + [args.max_faces]

        bin_indices = []
        for i in range(len(bin_bounds) - 1):
            lo, hi = bin_bounds[i], bin_bounds[i + 1]
            last = i == len(bin_bounds) - 2
            mask = (nf_all >= lo) & ((nf_all <= hi) if last else (nf_all < hi))
            bin_indices.append(np.nonzero(mask)[0].tolist())
        pool_sizes = [len(b) for b in bin_indices]
        alloc = water_fill(pool_sizes, args.n)

        print("stratified plan (%d bins, requested=%d):" % (len(bin_indices), args.n))
        for i in range(len(bin_indices)):
            lo, hi = bin_bounds[i], bin_bounds[i + 1]
            close = "]" if i == len(bin_indices) - 1 else ")"
            print("  bin[%4d,%4d%s  pool=%7d  alloc=%7d" % (lo, hi, close, pool_sizes[i], alloc[i]))
        total_alloc = sum(alloc)
        print("  -> %d parts will be staged%s"
              % (total_alloc, "" if total_alloc == args.n else " (< requested %d: bins exhausted)" % args.n))

        order = []
        for i, idxs in enumerate(bin_indices):
            chosen = rng.sample(idxs, alloc[i])
            for idx in chosen:
                bin_of_kept_idx[idx] = i
            order.extend(chosen)
        rng.shuffle(order)
    else:
        order = list(range(n_total))
        rng.shuffle(order)

    if args.dry_run:
        print("--dry-run: exiting without touching %s" % args.stage_dir)
        return

    bin_kept_counts = [0] * (len(bin_bounds) - 1) if bin_bounds else None
    kept = scanned = skipped_band = skipped_empty = 0
    for idx in order:
        if kept >= args.n:
            break
        row = ds[idx]
        scanned += 1
        nf = row["num_faces"]
        if nf is None or nf < args.min_faces or nf > args.max_faces:
            skipped_band += 1
            continue
        sf = row["step_file"]
        if not sf:
            skipped_empty += 1
            continue
        uid = row["uuid"]
        with open(os.path.join(args.stage_dir, uid + ".step"), "wb") as f:
            f.write(sf)
        if not args.no_code:
            code = row.get("cadquery_file")
            if code:
                try:
                    txt = code.decode("utf-8") if isinstance(code, (bytes, bytearray)) else str(code)
                    with open(os.path.join(args.stage_dir, uid + ".cadquery.py"), "w") as f:
                        f.write(txt)
                except Exception:
                    pass
        kept += 1
        if bin_kept_counts is not None and idx in bin_of_kept_idx:
            bin_kept_counts[bin_of_kept_idx[idx]] += 1
        if kept % 50 == 0:
            print("  staged %d/%d (scanned %d)" % (kept, args.n, scanned), flush=True)

    print("split=%s total=%d scanned=%d staged=%d (band[%d,%d] skip=%d, empty=%d) -> %s (seed=%d, stratify=%s)"
          % (args.split, n_total, scanned, kept, args.min_faces, args.max_faces,
             skipped_band, skipped_empty, args.stage_dir, args.seed, args.stratify), flush=True)
    if bin_kept_counts is not None:
        for i, c in enumerate(bin_kept_counts):
            lo, hi = bin_bounds[i], bin_bounds[i + 1]
            close = "]" if i == len(bin_kept_counts) - 1 else ")"
            print("  achieved bin[%4d,%4d%s  staged=%7d" % (lo, hi, close, c))
    if kept < args.n:
        print("WARNING: only staged %d of requested %d (band too tight, bins exhausted, or split exhausted)"
              % (kept, args.n), flush=True)

    # Self-documenting provenance: downstream tools (bench/build_fixtures.py) read
    # this instead of guessing/hardcoding the selection params that produced stage_dir.
    params = {
        "split": args.split, "seed": args.seed, "n_requested": args.n, "n_staged": kept,
        "min_faces": args.min_faces, "max_faces": args.max_faces, "stratify": args.stratify,
        "n_bins": args.n_bins if (args.stratify and not args.bin_edges) else None,
        "bin_edges_arg": args.bin_edges,
        "bin_bounds": bin_bounds,
    }
    with open(os.path.join(args.stage_dir, "_select_params.json"), "w") as f:
        json.dump(params, f, indent=2)


if __name__ == "__main__":
    main()
