#!/usr/bin/env python3
"""strip_blends.py — produce blend-FREE CadQuery targets for the 3D leg.

The winning recipe (noblend_v1) trains on GT CadQuery with **fillet/chamfer edge
blends removed**: the AMVDG IR does not encode edge blends, and asking the model
to emit them is the single largest source of exec failures (blend ops throw in
OCC when the model guesses edges/values). Dropping blends lifted exec 74%->96% /
valid ->88% at ~zero geometric cost (blends are sub-voxel). See research-log_3d
2026-07-04. This script reproduces `experiments/stage_*_noblend` deterministically.

Transform (AST, matches the original `ast.unparse` output byte-for-byte on the
common case): for any `X.fillet(...)` / `X.chamfer(...)` call, replace the call
with its base solid, peeling back the selector chain that only existed to feed
the blend (`.edges/.faces/.vertices/.wires(...)`). Mid-chain blends keep their
downstream ops. Everything else is preserved; the file is re-emitted with
`ast.unparse` (one statement per line), so blend-free files are reformatted too.

Usage:
    python scripts/train3d/strip_blends.py \
        --in-dir experiments/stage_z2c_train \
        --out-dir experiments/stage_z2c_train_noblend
    # then point build_dataset --code-dir at --out-dir (gt_dir stays the original
    # WITH-blend stage: blends are sub-voxel, so eval GT is unaffected).
"""
from __future__ import annotations

import argparse
import ast
import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

BLEND = {"fillet", "chamfer"}
# selector / stack-navigation methods that only exist to pick the edges a blend
# acts on — peeled back so `X.edges().first().fillet(r)` reduces to `X`.
SELECTOR = {"edges", "faces", "vertices", "wires", "solids", "shells",
            "first", "last", "item", "end", "all"}


class StripBlend(ast.NodeTransformer):
    """Replace every `<base>.<selectors...>.fillet|chamfer(...)` with `<base>`."""

    def __init__(self):
        self.n = 0

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)                      # transform inner calls first
        if isinstance(node.func, ast.Attribute) and node.func.attr in BLEND:
            recv = node.func.value
            while (isinstance(recv, ast.Call) and isinstance(recv.func, ast.Attribute)
                   and recv.func.attr in SELECTOR):
                recv = recv.func.value
            self.n += 1
            return recv
        return node


def strip_source(src: str) -> tuple[str, int]:
    """(blend-free source via ast.unparse, n_blends_removed). Raises on parse error."""
    tree = ast.parse(src)
    tr = StripBlend()
    tree = tr.visit(tree)
    ast.fix_missing_locations(tree)
    # no trailing newline: matches the original `ast.unparse` output byte-for-byte
    return ast.unparse(tree), tr.n


def _process_one(p, out_dir):
    name = os.path.basename(p)
    src = open(p).read()
    try:
        out, nb = strip_source(src)
    except SyntaxError as e:
        return ("FAIL", name, str(e))
    # only rewrite (unparse) files we actually changed; keep blend-free ones verbatim
    with open(os.path.join(out_dir, name), "w") as f:
        f.write(out if nb > 0 else src)
    return ("OK", name, nb)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True, help="stage dir with *.cadquery.py")
    ap.add_argument("--out-dir", required=True, help="dest dir for blend-free *.cadquery.py")
    ap.add_argument("--glob", default="*.cadquery.py")
    ap.add_argument("--workers", type=int, default=32,
                    help="thread pool size: per-file work is almost entirely file "
                         "open/read/write (the ast parse/transform itself is a few "
                         "hundred microseconds), which releases the GIL, so threads "
                         "parallelize this well")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.in_dir, args.glob)))
    n_ok = n_blend = n_stripped = n_fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        jobs = ex.map(_process_one, paths, [args.out_dir] * len(paths))
        for kind, name, payload in tqdm(jobs, total=len(paths)):
            if kind == "FAIL":
                print(f"[skip] {name}: parse error ({payload})", file=sys.stderr)
                n_fail += 1
            else:
                n_ok += 1
                n_blend += payload
                n_stripped += (payload > 0)
    print(f"[strip_blends] {n_ok} written ({n_stripped} had blends, {n_blend} blend "
          f"ops removed), {n_fail} parse-failed -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
