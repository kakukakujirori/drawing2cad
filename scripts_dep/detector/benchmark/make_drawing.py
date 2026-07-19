#!/usr/bin/env python
"""Generate a controlled synthetic engineering-drawing testbed with exact GT primitives.

The point is a *fair* probe of geometry detectors: the real geometry (outline edges,
holes, bore) is drawn at the SAME stroke weight as adversarial clutter (dimension lines,
extension lines, arrowheads, dashed hidden edges, dimension text), so a detector cannot
cheat on stroke width alone. GT lists ONLY the real geometry, so we can score both
recall (did it find the real lines/circles) and precision (did it report clutter as geometry).

Emits  clean.png, scan.png (blur+noise+rotation), gt.json  into --out.
CPU/PIL only; no GPU, no network.
"""

import os
import json
import math
import argparse
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

W, H = 1400, 1000
OUT_W = 2  # outline stroke
DIM_W = 2  # dimension/extension stroke (SAME as outline on purpose -> adversarial)
HID_W = 2


def font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def arrow(d, x, y, ang, a=10):
    bx, by = x - a * math.cos(ang), y - a * math.sin(ang)
    px, py = -math.sin(ang) * a * 0.35, math.cos(ang) * a * 0.35
    d.polygon([(x, y), (bx + px, by + py), (bx - px, by - py)], fill=0)


def dashed(d, p1, p2, w, dash=10, gap=7):
    x1, y1 = p1
    x2, y2 = p2
    L = math.hypot(x2 - x1, y2 - y1)
    n = int(L // (dash + gap))
    ux, uy = (x2 - x1) / L, (y2 - y1) / L
    for i in range(n + 1):
        s = i * (dash + gap)
        e = min(s + dash, L)
        d.line(
            [(x1 + ux * s, y1 + uy * s), (x1 + ux * e, y1 + uy * e)], fill=0, width=w
        )


def build():
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)
    gt_lines, gt_circles = [], []

    # ---- REAL geometry (this is the GT) ----
    # outer plate outline (front view)
    rect = [(220, 250), (760, 250), (760, 640), (220, 640)]
    for i in range(4):
        a, b = rect[i], rect[(i + 1) % 4]
        d.line([a, b], fill=0, width=OUT_W)
        gt_lines.append([*a, *b])
    # a step edge inside (real internal line)
    d.line([(220, 470), (760, 470)], fill=0, width=OUT_W)
    gt_lines.append([220, 470, 760, 470])
    # central bore + concentric ring
    cx, cy = 490, 360
    for r in (120, 95):
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=0, width=OUT_W)
        gt_circles.append([cx, cy, r])
    # 4 bolt holes
    for hx, hy in [(330, 320), (650, 320), (330, 400), (650, 400)]:
        r = 26
        d.ellipse([hx - r, hy - r, hx + r, hy + r], outline=0, width=OUT_W)
        gt_circles.append([hx, hy, r])

    # ---- CLUTTER (NOT in GT): the stuff that breaks naive detectors ----
    # dashed hidden edges (fragment line detectors)
    dashed(d, (300, 250), (300, 640), HID_W)
    dashed(d, (220, 360), (760, 360), HID_W)
    # centerlines through the bore
    d.line([(cx - 150, cy), (cx + 150, cy)], fill=0, width=1)
    d.line([(cx, cy - 150), (cx, cy + 150)], fill=0, width=1)
    # horizontal dimension below (extension + dim line + arrows + text)
    yd = 700
    for xx in (220, 760):
        d.line([(xx, 644), (xx, yd + 6)], fill=0, width=DIM_W)  # extension lines
    d.line([(220, yd), (760, yd)], fill=0, width=DIM_W)  # dim line
    arrow(d, 220, yd, math.pi)
    arrow(d, 760, yd, 0)
    d.text((480, yd - 22), "540", fill=0, font=font(22))
    # vertical dimension left
    xd = 150
    for yy in (250, 640):
        d.line([(214, yy), (xd - 6, yy)], fill=0, width=DIM_W)
    d.line([(xd, 250), (xd, 640)], fill=0, width=DIM_W)
    arrow(d, xd, 250, -math.pi / 2)
    arrow(d, xd, 640, math.pi / 2)
    d.text((xd - 40, 430), "390", fill=0, font=font(22))
    # diameter callout on the bore (leader line + text) -> clutter line through a circle
    lx, ly = (
        cx + 120 * math.cos(math.radians(40)),
        cy + 120 * math.sin(math.radians(40)),
    )
    d.line([(lx, ly), (lx + 90, ly + 70)], fill=0, width=DIM_W)
    d.text((lx + 95, ly + 58), "Ø240", fill=0, font=font(22))
    # title block (lots of straight clutter lines)
    d.rectangle([1020, 860, 1380, 980], outline=0, width=DIM_W)
    d.line([(1020, 905), (1380, 905)], fill=0, width=1)
    d.line([(1200, 860), (1200, 905)], fill=0, width=1)
    d.text((1030, 870), "PART-001  STEEL", fill=0, font=font(18))

    return img, {"image_size": [W, H], "lines": gt_lines, "circles": gt_circles}


def scan_aug(img, seed=0):
    rng = np.random.default_rng(seed)
    im = img.rotate(
        rng.uniform(-1.5, 1.5),
        resample=Image.BICUBIC,
        fillcolor=245,
        center=(W / 2, H / 2),
    )
    im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.6, 1.1)))
    a = np.asarray(im).astype(np.float32)
    a = a * 0.92 + 12  # ink not pure black, paper not pure white
    a += rng.normal(0, rng.uniform(4, 9), a.shape)  # sensor noise
    m = rng.random(a.shape) < 0.0006
    a[m] = rng.integers(20, 90, m.sum())  # speckle
    f = rng.uniform(0.6, 0.8)
    h, w = a.shape  # resolution loss
    small = Image.fromarray(a.clip(0, 255).astype(np.uint8)).resize(
        (int(w * f), int(h * f)), Image.BILINEAR
    )
    return small.resize((w, h), Image.BILINEAR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/detector_bench")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    img, gt = build()
    img.convert("RGB").save(os.path.join(a.out, "clean.png"))
    scan_aug(img, 0).convert("RGB").save(os.path.join(a.out, "scan.png"))
    json.dump(gt, open(os.path.join(a.out, "gt.json"), "w"), indent=1)
    print(f"wrote clean.png / scan.png / gt.json -> {a.out}")
    print(
        f"GT: {len(gt['lines'])} real lines, {len(gt['circles'])} real circles "
        f"(+ dimension/hidden/centerline/text clutter NOT in GT)"
    )


if __name__ == "__main__":
    main()
