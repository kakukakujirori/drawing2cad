#!/usr/bin/env python
"""M-LSD (learned line segment detector, CPU) scored against the renderer GT.

Uses the M-LSD weights shipped via controlnet_aux (MobileNetV2 backbone). We pull the
RAW segment coordinates (pred_lines) rather than the drawn line-map, scale to image
space, and score. M-LSD is trained at 512px for *dominant structural* lines, so we
also try a higher input resolution to give the dense CAD wireframe a fair chance.
"""
import os, json, argparse
import cv2, numpy as np
from score_primitives import score
from controlnet_aux import MLSDdetector
from controlnet_aux.mlsd.utils import pred_lines

def load_model():
    for repo in ("lllyasviel/Annotators", "lllyasviel/ControlNet"):
        try:
            det = MLSDdetector.from_pretrained(repo)
            det.model.cpu().eval()
            return det
        except Exception as e:
            print(f"  (from_pretrained {repo} failed: {e})")
    raise RuntimeError("could not load M-LSD weights")

def detect(det, img_rgb, res, thr_v=0.05, thr_d=5.0):
    lines = pred_lines(img_rgb, det.model, [res, res], thr_v, thr_d)  # coords in original image space
    return [[float(x1), float(y1), float(x2), float(y2)] for (x1, y1, x2, y2) in lines]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="experiments/detector_bench")
    a = ap.parse_args()
    graph = os.path.join(a.dir, "part.graph.json")
    det = load_model()
    print("M-LSD loaded (CPU)")
    rows = []
    for variant in ("clean", "scan"):
        img = cv2.imread(os.path.join(a.dir, f"{variant}.png"))           # BGR
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for res in (512, 1280):
            lines = detect(det, rgb, res)
            s = score(graph, {"lines": lines, "circles": []})
            rows.append((variant, f"M-LSD@{res}", len(lines), s["lines"]))
            json.dump({"res": res, "n_lines": len(lines), "lines": s["lines"]},
                      open(os.path.join(a.dir, f"mlsd_{variant}_{res}.json"), "w"), indent=2)
    print(f"\n{'variant':6} {'method':12} {'#det':>5}  line R(vis/hid/all)  line P")
    for variant, name, ndet, ln in rows:
        lr = f"{ln['visible']}/{ln['hidden']}/{ln['all']}"
        print(f"{variant:6} {name:12} {ndet:>5}  {lr:20} {ln['precision']}")

if __name__ == "__main__":
    main()
