#!/usr/bin/env python
"""Stage a DIVERSE random sample of Fusion360 'reconstruction' STEP solids via symlinks.

The reconstruction set has many construction-sequence sub-steps per part; the renderer's
sorted()[:N] would take all sub-steps of a few low-numbered parts (not diverse). Instead:
group by part ({partid}_{hash}), take each part's FINAL (lexicographically-max = most-complete)
state, then random.sample N parts -> one diverse, geometry-rich drawing per part. Real solids
have bores along varied axes, so circles spread across views.

Usage:  select_recon.py <reconstruction_dir> <N> <stage_dir> [seed]
  e.g.  select_recon.py /path/Fusion360Gallery/r1.0.1/reconstruction 2500 experiments/stage_recon 0
Then:   python scripts/renderer/batch_dataset.py <stage_dir> <out_dir>
"""
import os, glob, random, sys, collections

recon_dir = sys.argv[1]
N = int(sys.argv[2])
stage = sys.argv[3]
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0

files = glob.glob(os.path.join(recon_dir, "*.step"))
groups = collections.defaultdict(list)
for f in files:
    flds = os.path.basename(f)[:-5].split("_")       # strip .step
    key = "_".join(flds[:2]) if len(flds) >= 2 else flds[0]
    groups[key].append(f)
finals = sorted(max(v) for v in groups.values())     # final/most-complete state per part
random.seed(seed)
sel = finals if N >= len(finals) else random.sample(finals, N)

os.makedirs(stage, exist_ok=True)
for old in glob.glob(os.path.join(stage, "*.step")):
    os.unlink(old)
for f in sel:
    os.symlink(os.path.abspath(f), os.path.join(stage, os.path.basename(f)))
print("step files=%d  distinct parts=%d  finals=%d  selected=%d -> %s (seed=%d)"
      % (len(files), len(groups), len(finals), len(sel), stage, seed))
