#!/usr/bin/env python
"""SOLD2 (Self-supervised Occlusion-aware Line Detection, CVPR'21) via kornia, CPU.

A 3rd, learned line detector — occlusion-aware, so a fair check on dashed/hidden edges.
Heavier than M-LSD, so we run at 800px long-side (CPU) and scale segments back.
"""

import os
import json
import argparse
import time
import cv2
import torch
from score_primitives import score
from kornia.feature import SOLD2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="experiments/detector_bench")
    ap.add_argument("--long", type=int, default=800)
    a = ap.parse_args()
    graph = os.path.join(a.dir, "part.graph.json")
    sold2 = SOLD2(pretrained=True).eval()
    rows = []
    for variant in ("clean", "scan"):
        img = cv2.imread(os.path.join(a.dir, f"{variant}.png"), cv2.IMREAD_GRAYSCALE)
        H, W = img.shape
        s = a.long / max(H, W)
        im = cv2.resize(img, (int(W * s), int(H * s)))
        t = torch.from_numpy(im).float()[None, None] / 255.0
        t0 = time.time()
        with torch.no_grad():
            seg = (
                sold2(t)["line_segments"][0].cpu().numpy()
            )  # [N,2,2], (row,col)=(y,x) per endpoint
        lines = [
            [float(p0[1] / s), float(p0[0] / s), float(p1[1] / s), float(p1[0] / s)]
            for p0, p1 in seg
        ]
        sc = score(graph, {"lines": lines, "circles": []})
        rows.append((variant, len(lines), sc["lines"], time.time() - t0))
        json.dump(
            {"n_lines": len(lines), "lines": sc["lines"]},
            open(os.path.join(a.dir, f"sold2_{variant}.json"), "w"),
            indent=2,
        )
    print(f"\n{'variant':6} {'#det':>5}  line R(vis/hid/all)  line P   sec")
    for variant, ndet, ln, sec in rows:
        lr = f"{ln['visible']}/{ln['hidden']}/{ln['all']}"
        print(f"{variant:6} {ndet:>5}  {lr:20} {str(ln['precision']):7} {sec:.0f}")


if __name__ == "__main__":
    main()
