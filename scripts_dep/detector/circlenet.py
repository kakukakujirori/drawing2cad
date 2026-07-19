#!/usr/bin/env python
"""CircleNet detector spike — center-heatmap + continuous-radius regression on our GT.
Goal: cut the classical flood/starve (2425 / 6-18) to ~real circle count AND give a
NON-vacuous over-determined scale on real/101 (the decisive "is real tractable" number).

Subcommands (audit + labeltest run WITHOUT torch; train/eval need torch):
  python detector.py labeltest                      # local: validate label-gen on Bracket/Flange (no torch)
  python detector.py audit   <root>                 # server: report graph.json schema + circle stats (no torch)
  python detector.py train   <root> <outdir> [--steps N --tile 512 --bs 8]
  python detector.py eval    <ckpt> <real_png> [--known 28,40,50,...]   # real-drawing scale test
"""

import sys
import os
import json
import glob
import math
import argparse
import numpy as np

# ---------------- schema-tolerant circle extraction (handles poc + official renderer) ----------------
# CircleNet targets DETECTABLE holes/bosses. Sub-pixel circles can't be localized, and
# huge-radius "arcs" (near-straight edges the classifier rounded to a circle) are not holes
# and — left in — smear a single tile's gaussian heatmap target across the whole tile.
# Clamp to a sane band (env-tunable) so the training targets and the eval GT stay consistent.
RMIN_PX = float(os.environ.get("CIRCLE_RMIN_PX", "3"))
RMAX_PX = float(os.environ.get("CIRCLE_RMAX_PX", "600"))


def circles_from_graph(g):
    """-> list of (cx, cy, r_px, visible_bool) in full-image pixel coords (radius-banded)."""
    out = []
    views = g.get("views") or []
    for v in views if isinstance(views, list) else views.values():
        prims = v.get("primitives") or v.get("entities") or []
        for p in prims:
            t = (p.get("type") or p.get("kind") or "").lower()
            if t not in ("circle", "arc", "ellipse"):
                continue
            c = p.get("center") or p.get("c")
            if not c:
                bb = p.get("bbox_px") or p.get("bbox")
                if bb and len(bb) == 4:
                    c = [(bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2]
                else:
                    continue
            r = p.get("r_px") or p.get("radius") or p.get("r")
            if r is None:
                bb = p.get("bbox_px") or p.get("bbox")
                if bb and len(bb) == 4:
                    r = (abs(bb[2] - bb[0]) + abs(bb[3] - bb[1])) / 4
                else:
                    continue
            role = p.get("visibility") or p.get("line_role") or "visible"
            vis = str(role).lower().startswith("vis")
            if RMIN_PX <= float(r) <= RMAX_PX:
                out.append((float(c[0]), float(c[1]), float(r), vis))
    return out


def find_pairs(root):
    pairs = []
    for gj in glob.glob(
        os.path.join(root, "**", "*.graph.json"), recursive=True
    ) + glob.glob(os.path.join(root, "*.graph.json")):
        stem = gj[: -len(".graph.json")]
        for ext in (".png", ".scan.png", ".jpg"):
            if os.path.exists(stem + ext):
                pairs.append((stem + ext, gj))
                break
    # dedup
    seen = set()
    uniq = []
    for p in pairs:
        if p[1] in seen:
            continue
        seen.add(p[1])
        uniq.append(p)
    return uniq


# ---------------- target builder (numpy; no torch) ----------------
def gaussian2d(h, w, cy, cx, sigma):
    ys = np.arange(h)[:, None]
    xs = np.arange(w)[None, :]
    return np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * sigma * sigma))


# Radius is regressed in LOG space. Linear L1 over a 3..600px radius range — dominated by the
# many small holes — made the head regress to the mean, so large bores were detected at the
# center but with a too-small radius and rejected by the radius tolerance (r>30px recall ~0).
# log(r) makes the loss penalize RELATIVE error, balancing small and large circles.
def r_encode(r_px):
    return math.log(max(float(r_px), 1.0))


def r_decode(t):
    return math.exp(float(t))


def build_targets(circ_in_tile, tile, stride=4):
    hw = tile // stride
    hm = np.zeros((hw, hw), np.float32)
    rmap = np.zeros((hw, hw), np.float32)
    cmask = np.zeros(
        (hw, hw), np.float32
    )  # center-only (1 cell/circle) -> for eval GT counting
    rmask = np.zeros(
        (hw, hw), np.float32
    )  # smeared 3x3 -> radius supervision (robust to peak jitter)
    for cx, cy, r, vis in circ_in_tile:
        gx, gy = cx / stride, cy / stride
        if not (0 <= gx < hw and 0 <= gy < hw):
            continue
        sigma = max(1.0, (r / stride) * 0.3)
        hm = np.maximum(hm, gaussian2d(hw, hw, gy, gx, sigma))
        iy, ix = int(round(gy)), int(round(gx))
        iy = min(max(iy, 0), hw - 1)
        ix = min(max(ix, 0), hw - 1)
        hm[iy, ix] = 1.0  # exact positive at center (focal pos branch keys on ==1)
        cmask[iy, ix] = 1.0
        # concentric circles share a center -> keep OUTER (max) radius; inner come from dim layer.
        # smear radius target over 3x3 so the radius head is learned in a neighborhood, not 1 pixel
        # (the single-pixel version collapsed to a constant ~18px at inference).
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = iy + dy, ix + dx
                if 0 <= yy < hw and 0 <= xx < hw:
                    rmap[yy, xx] = max(rmap[yy, xx], r_encode(r))
                    rmask[yy, xx] = 1.0
    return hm, rmap, cmask, rmask


def crop_circles(circ, x0, y0, tile):
    out = []
    for cx, cy, r, vis in circ:
        if x0 <= cx < x0 + tile and y0 <= cy < y0 + tile:
            out.append((cx - x0, cy - y0, r, vis))
    return out


# ---------------- LABELTEST (local, no torch) ----------------
def labeltest():
    import cv2

    R = os.environ.get(
        "CIRCLENET_DATA", "poc/dataset"
    )  # holds {Bracket,Flange}.{png,graph.json}
    for name in ("Bracket", "Flange"):
        g = json.load(open(f"{R}/{name}.graph.json"))
        circ = circles_from_graph(g)
        print(
            f"\n{name}: {len(circ)} circles; r_px range {min(c[2] for c in circ):.0f}..{max(c[2] for c in circ):.0f}; "
            f"visible {sum(c[3] for c in circ)}/{len(circ)}"
        )
        img = cv2.imread(f"{R}/{name}.png", cv2.IMREAD_GRAYSCALE)
        H, W = img.shape
        # center a tile on the densest circle cluster
        cxs = np.array([c[0] for c in circ])
        cys = np.array([c[1] for c in circ])
        tile = 640
        x0 = int(np.clip(cxs.mean() - tile / 2, 0, W - tile))
        y0 = int(np.clip(cys.mean() - tile / 2, 0, H - tile))
        ct = crop_circles(circ, x0, y0, tile)
        hm, rmap, cmask, rmask = build_targets(ct, tile, stride=4)
        npos = int(cmask.sum())
        ncenters = len(
            {(int(round(cx / 4)), int(round(cy / 4))) for cx, cy, r, vis in ct}
        )
        print(
            f"  tile@({x0},{y0}) {tile}px -> {len(ct)} circles ({ncenters} distinct centers; "
            f"{len(ct) - ncenters} concentric collapsed); heatmap peak={hm.max():.2f}; "
            f"{npos} center targets; outer-radii(px)={[round(r_decode(r)) for r in rmap[cmask > 0]]}"
        )
        assert npos == ncenters, (
            "center-target count must equal distinct circle centers"
        )
        assert hm.max() > 0.85 or len(ct) == 0, (
            "heatmap should peak near 1 at a center (sub-grid center => <1)"
        )
        # viz
        viz = cv2.cvtColor(img[y0 : y0 + tile, x0 : x0 + tile], cv2.COLOR_GRAY2BGR)
        for cx, cy, r, vis in ct:
            cv2.circle(viz, (int(cx), int(cy)), int(r), (0, 0, 255), 2)
        hmU = cv2.resize((hm * 255).astype(np.uint8), (tile, tile))
        hmU = cv2.applyColorMap(hmU, cv2.COLORMAP_JET)
        cv2.imwrite(f"{R}/labeltest_{name}.png", np.hstack([viz, hmU]))
    print(
        "\nLABELTEST OK -> circles extract, heatmap+radius targets build correctly (saved labeltest_*.png)"
    )


def audit(root):
    pairs = find_pairs(root)
    print(f"root={root}  pairs(png+graph)={len(pairs)}")
    if not pairs:
        print("NO PAIRS FOUND — check layout")
        return
    ncs = []
    for png, gj in pairs[:200]:
        try:
            ncs.append(len(circles_from_graph(json.load(open(gj)))))
        except Exception as e:
            print("  parse fail", gj, e)
    ncs = np.array(ncs)
    print(
        f"circles/part: mean {ncs.mean():.1f} median {np.median(ncs):.0f} min {ncs.min()} max {ncs.max()} (n={len(ncs)})"
    )
    # show one schema sample
    g = json.load(open(pairs[0][1]))
    v = (g.get("views") or [{}])[0]
    p = (v.get("primitives") or [{}])[0] if (v.get("primitives")) else {}
    print("sample primitive keys:", list(p.keys()))
    print("first image:", pairs[0][0])


# ---------------- torch model + train/eval ----------------
def _torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    return torch, nn, F


def build_model():
    torch, nn, F = _torch()

    def cbr(i, o, s=1, d=1):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, s, d, dilation=d), nn.BatchNorm2d(o), nn.ReLU(True)
        )

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.e1 = cbr(1, 32, 2)
            s.e2 = cbr(32, 64, 2)
            s.e3 = cbr(64, 128, 1)
            s.e4 = cbr(128, 128, 1)  # stride 4, RF~23px
            # Dilated context cascade (stride-1, 'same'): grows the receptive field from ~23px to
            # ~270px WITHOUT extra downsampling, so the center cell of a LARGE bore can see its rim.
            # With RF~23 the center of any r>~11px hole sees only blank interior -> r>30px recall 0
            # (verified: stratified recall fell off exactly as r passed RF/2). Final dil=1 conv fuses
            # the multi-dilation features to suppress gridding artifacts. RF~271px covers r up to ~135px.
            s.d1 = cbr(128, 128, 1, 2)
            s.d2 = cbr(128, 128, 1, 4)
            s.d3 = cbr(128, 128, 1, 8)
            s.d4 = cbr(128, 128, 1, 16)
            s.fuse = cbr(128, 128, 1, 1)
            s.hm = nn.Conv2d(128, 1, 1)
            s.rr = nn.Conv2d(128, 1, 1)
            # CenterNet focal-loss prior: start the heatmap at low confidence (p0~0.1) so training
            # raises the few true peaks instead of first having to crush random-init activations
            # everywhere (the observed cause of "maxp stuck ~0.2, recall 0" at low step counts).
            nn.init.constant_(s.hm.bias, -2.19)

        def forward(s, x):
            x = s.e1(x)
            x = s.e2(x)
            x = s.e3(x)
            x = s.e4(x)
            x = s.d1(x)
            x = s.d2(x)
            x = s.d3(x)
            x = s.d4(x)
            x = s.fuse(x)
            return s.hm(x), s.rr(x)

    return Net()


class DS:
    def __init__(s, pairs, tile, stride=4, aug=True):
        s.pairs = pairs
        s.tile = tile
        s.stride = stride
        s.aug = aug
        s.cache = {}

    def __len__(s):
        return len(s.pairs) * 8

    def sample(s, idx):
        import cv2

        k = idx % len(s.pairs)
        png, gj = s.pairs[k]
        # cache decoded image + parsed circles: a full cv2.imread per sample (1800x1273) was
        # starving the GPU (~1 step/s). The dataset fits in RAM, so read each part once.
        if k in s.cache:
            img, circ = s.cache[k]
        else:
            img = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            circ = circles_from_graph(json.load(open(gj)))
            s.cache[k] = (img, circ)
        H, W = img.shape
        t = s.tile
        if W < t or H < t:
            img = cv2.copyMakeBorder(
                img, 0, max(0, t - H), 0, max(0, t - W), cv2.BORDER_CONSTANT, value=255
            )
            H, W = img.shape
        # bias crop toward a circle
        if circ and np.random.rand() < 0.8:
            cx, cy, _, _ = circ[np.random.randint(len(circ))]
            x0 = int(np.clip(cx - t / 2 + np.random.randint(-t // 4, t // 4), 0, W - t))
            y0 = int(np.clip(cy - t / 2 + np.random.randint(-t // 4, t // 4), 0, H - t))
        else:
            x0 = np.random.randint(0, W - t + 1)
            y0 = np.random.randint(0, H - t + 1)
        crop = img[y0 : y0 + t, x0 : x0 + t].astype(np.float32)
        ct = crop_circles(circ, x0, y0, t)
        if s.aug:
            crop = scan_aug(crop)
        hm, rmap, cmask, rmask = build_targets(ct, t, s.stride)
        return crop[None] / 255.0, hm[None], rmap[None], cmask[None], rmask[None]


def scan_aug(im):
    import cv2

    if np.random.rand() < 0.5:
        im = cv2.GaussianBlur(im, (0, 0), np.random.uniform(0.5, 1.8))
    if np.random.rand() < 0.5:
        im = im + np.random.normal(0, np.random.uniform(3, 15), im.shape).astype(
            np.float32
        )
    if np.random.rand() < 0.4:  # line-weight jitter via morphology
        k = np.ones((2, 2), np.uint8)
        im = cv2.erode(im, k) if np.random.rand() < 0.5 else cv2.dilate(im, k)
    if np.random.rand() < 0.4:  # downscale-upscale (resolution loss)
        f = np.random.uniform(0.5, 0.9)
        h, w = im.shape
        im = cv2.resize(cv2.resize(im, (int(w * f), int(h * f))), (w, h))
    if np.random.rand() < 0.3:
        im = np.clip(
            (im - 128) * np.random.uniform(0.7, 1.3) + 128 + np.random.uniform(-20, 20),
            0,
            255,
        )
    return np.clip(im, 0, 255)


def train(root, outdir, steps, tile, bs, lr=2e-3):
    torch, nn, F = _torch()
    seed = int(os.environ.get("CIRCLENET_SEED", "0"))
    np.random.seed(seed)
    torch.manual_seed(seed)  # reproducible split + sampling
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pairs = find_pairs(root)
    assert pairs, "no data"
    np.random.shuffle(pairs)
    nval = max(1, len(pairs) // 10)
    val = pairs[:nval]
    tr = pairs[nval:]
    print(
        f"parts: train {len(tr)} val {len(val)}; device {dev}; tile {tile} bs {bs} steps {steps}"
    )
    ds = DS(tr, tile)
    net = build_model().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    os.makedirs(outdir, exist_ok=True)
    # persist the held-out split so a full-image (per-drawing) eval scores the SAME parts
    json.dump(
        {
            "val": [os.path.basename(p[0]) for p in val],
            "train": [os.path.basename(p[0]) for p in tr],
        },
        open(os.path.join(outdir, "split.json"), "w"),
    )

    def batch(dset, n):
        xs = []
        hs = []
        rs = []
        cs = []
        rms = []
        while len(xs) < n:
            s = dset.sample(np.random.randint(len(dset)))
            if s is None:
                continue
            xs.append(s[0])
            hs.append(s[1])
            rs.append(s[2])
            cs.append(s[3])
            rms.append(s[4])
        T = lambda a: torch.tensor(np.stack(a)).float().to(dev)
        return T(xs), T(hs), T(rs), T(cs), T(rms)

    net.train()
    for it in range(steps):
        x, hm, rm, cmk, rmk = batch(ds, bs)
        phm, prr = net(x)
        p = torch.sigmoid(phm).clamp(1e-4, 1 - 1e-4)
        # CenterNet penalty-reduced focal loss (handles the 4-positives-in-16384 imbalance that
        # plain BCE collapsed on -> "predict 0 everywhere"). pos: target==1; neg: weighted (1-target)^4.
        pos = (hm == 1).float()
        neg = 1.0 - pos
        lpos = -((1 - p) ** 2 * torch.log(p) * pos).sum()
        lneg = -(((1 - hm) ** 4) * (p**2) * torch.log(1 - p) * neg).sum()
        npos = pos.sum().clamp(min=1)
        lh = (lpos + lneg) / npos
        lr_ = F.l1_loss(prr * rmk, rm * rmk, reduction="sum") / (rmk.sum() + 1e-6)
        loss = (
            lh + 4.0 * lr_
        )  # log-radius L1 has smaller magnitude than the old linear L1; reweight up
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 50 == 0 or it == steps - 1:
            print(
                f"it {it:4d} loss {loss.item():.4f} (hm {lh.item():.4f} r {lr_.item():.3f}) maxp {float(p.max()):.2f}"
            )
    torch.save(net.state_dict(), os.path.join(outdir, "circlenet.pt"))
    print("saved", os.path.join(outdir, "circlenet.pt"))
    # synthetic held-out eval
    rec, prec, mae = eval_synth(net, val, tile, dev)
    print(
        f"\nSYNTH held-out: recall {rec:.2f} precision {prec:.2f} radius-MAE {mae:.1f}px"
    )


def decode(phm, prr, stride=4, thr=0.2, topk=200):
    torch, nn, F = _torch()
    import torch.nn.functional as F

    hmp = torch.sigmoid(phm)
    pool = F.max_pool2d(hmp, 3, 1, 1)
    peaks = ((hmp == pool) & (hmp > thr)).nonzero()  # (b,1,y,x)
    Hh, Ww = prr.shape[2], prr.shape[3]
    out = []
    for b, _, y, x in peaks.tolist():
        y0, y1 = max(0, y - 1), min(Hh, y + 2)
        x0, x1 = max(0, x - 1), min(Ww, x + 2)
        r = r_decode(
            prr[b, 0, y0:y1, x0:x1].mean()
        )  # 3x3 mean in log space (matches smeared radius training)
        out.append((b, x * stride, y * stride, r, float(hmp[b, 0, y, x])))
    return out


def eval_synth(net, val, tile, dev):
    torch, nn, F = _torch()
    net.eval()
    TP = FP = FN = 0
    errs = []
    maxp = 0.0
    ds = DS(val, tile, aug=False)
    with torch.no_grad():
        for i in range(len(val) * 4):
            s = ds.sample(i)
            if s is None:
                continue
            x = torch.tensor(s[0][None]).float().to(dev)
            phm, prr = net(x)
            maxp = max(maxp, float(torch.sigmoid(phm).max()))
            det = decode(phm, prr)
            # GT circles in this crop from targets
            gt = []
            rm = s[2][0]
            mk = s[3][0]
            ys, xs = np.where(mk > 0)
            for y, x_ in zip(ys, xs):
                gt.append((x_ * 4, y * 4, r_decode(rm[y, x_])))
            used = set()
            for _, dx, dy, dr, sc in det:
                m = None
                for j, (gx, gy, gr) in enumerate(gt):
                    if j in used:
                        continue
                    if math.hypot(dx - gx, dy - gy) < max(8, 0.3 * gr):
                        m = j
                        break
                if m is not None:
                    TP += 1
                    used.add(m)
                    errs.append(abs(dr - gt[m][2]))
                else:
                    FP += 1
            FN += len(gt) - len(used)
    rec = TP / (TP + FN + 1e-6)
    prec = TP / (TP + FP + 1e-6)
    mae = float(np.mean(errs)) if errs else float("nan")
    print(
        f"(eval debug: max predicted heatmap peak = {maxp:.2f}; TP {TP} FP {FP} FN {FN})"
    )
    return rec, prec, mae


def eval_real(ckpt, real_png, known):
    torch, nn, F = _torch()
    import cv2

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_model().to(dev)
    net.load_state_dict(torch.load(ckpt, map_location=dev))
    net.eval()
    img = cv2.imread(real_png, cv2.IMREAD_GRAYSCALE)
    H, W = img.shape
    tile = 640
    ov = 80
    dets = []
    maxp = 0.0
    with torch.no_grad():
        for y0 in range(0, max(1, H - tile) + 1, tile - ov):
            for x0 in range(0, max(1, W - tile) + 1, tile - ov):
                crop = img[y0 : y0 + tile, x0 : x0 + tile].astype(np.float32)
                if crop.shape != (tile, tile):
                    crop = cv2.copyMakeBorder(
                        crop,
                        0,
                        tile - crop.shape[0],
                        0,
                        tile - crop.shape[1],
                        cv2.BORDER_CONSTANT,
                        value=255,
                    )
                x = torch.tensor(crop[None, None] / 255.0).float().to(dev)
                phm, prr = net(x)
                maxp = max(maxp, float(torch.sigmoid(phm).max()))
                for _, dx, dy, dr, sc in decode(phm, prr, thr=0.25):
                    dets.append((x0 + dx, y0 + dy, dr, sc))
    print(f"(real eval debug: max predicted heatmap peak across tiles = {maxp:.2f})")
    # simple dedup
    dets = sorted(dets, key=lambda d: -d[3])
    keep = []
    for d in dets:
        if all(math.hypot(d[0] - k[0], d[1] - k[1]) > 15 for k in keep):
            keep.append(d)
    det_r = sorted(d[2] for d in keep)
    print(f"REAL {real_png}: {len(keep)} circles; radii px {[round(r) for r in det_r]}")
    exp_r = sorted(set(v / 2 for v in known))
    TOL = 0.06

    def matched(s):
        return sum(
            1 for r in det_r if min(abs(r - s * e) / (s * e) for e in exp_r) < TOL
        )

    if det_r and exp_r:
        cap = sorted({r / e for r in det_r for e in exp_r})
        nm, s = max(((matched(x), x) for x in cap))
        nulls = [matched(x) for x in np.linspace(min(cap), max(cap), 60)]
        nullm = sum(nulls) / len(nulls)
        print(
            f"scale: best px_per_mm={s:.3f} matched {nm}/{len(det_r)}; NULL mean {nullm:.1f}; signal {nm - nullm:.1f}"
        )
        print(
            "=>",
            "NON-VACUOUS (real tractable)"
            if (nm - nullm) >= 4 and len(det_r) < 150
            else "still weak",
        )


def _detect_full(net, img, dev, tile=640, ov=80, thr=0.25):
    """Tile a full drawing, decode circles per tile, dedup -> [(x,y,r,score)] in image px."""
    torch, nn, F = _torch()
    import cv2

    H, W = img.shape
    dets = []
    with torch.no_grad():
        for y0 in range(0, max(1, H - tile) + 1, tile - ov):
            for x0 in range(0, max(1, W - tile) + 1, tile - ov):
                crop = img[y0 : y0 + tile, x0 : x0 + tile].astype(np.float32)
                if crop.shape != (tile, tile):
                    crop = cv2.copyMakeBorder(
                        crop,
                        0,
                        tile - crop.shape[0],
                        0,
                        tile - crop.shape[1],
                        cv2.BORDER_CONSTANT,
                        value=255,
                    )
                x = torch.tensor(crop[None, None] / 255.0).float().to(dev)
                phm, prr = net(x)
                for _, dx, dy, dr, sc in decode(phm, prr, thr=thr):
                    dets.append((x0 + dx, y0 + dy, dr, sc))
    dets = sorted(dets, key=lambda d: -d[3])
    keep = []
    for d in dets:
        if all(
            math.hypot(d[0] - k[0], d[1] - k[1]) > max(10, 0.4 * d[2]) for k in keep
        ):
            keep.append(d)
    return keep


def _labeled_region(g, aff=None, pad=25.0):
    """Bbox (x0,y0,x1,y1) of ALL ortho-view GT primitives — the region that carries labels.
    The isometric view + third-angle symbol have real circles but NO GT, so detections there
    must be neither TP nor FP. Restrict scoring to this region for a fair precision."""
    xs = []
    ys = []
    for gv in g.get("views", []):
        for p in gv.get("primitives", []):
            bb = p.get("bbox_px")
            if bb and len(bb) == 4:
                xs += [bb[0], bb[2]]
                ys += [bb[1], bb[3]]
    if not xs:
        return None
    box = [min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad]
    if aff:
        cs = [
            _apply_aff((box[0], box[1]), aff),
            _apply_aff((box[2], box[1]), aff),
            _apply_aff((box[2], box[3]), aff),
            _apply_aff((box[0], box[3]), aff),
        ]
        box = [
            min(c[0] for c in cs),
            min(c[1] for c in cs),
            max(c[0] for c in cs),
            max(c[1] for c in cs),
        ]
    return box


def _in_region(d, box):
    return box is None or (box[0] <= d[0] <= box[2] and box[1] <= d[1] <= box[3])


def _gt_centers(circ, stride=4, visible_only=False):
    """Dedup GT circles to distinct centers (outer radius) — matches build_targets' 1-target-
    per-center, outer-radius convention so a counterbore counts once, not N times."""
    by = {}
    for cx, cy, r, vis in circ:
        if visible_only and not vis:
            continue
        k = (int(round(cx / stride)), int(round(cy / stride)))
        if k not in by or r > by[k][2]:
            by[k] = (cx, cy, r, vis)
    return list(by.values())


def evalfull(
    ckpt, root, split_json=None, use_scan=False, thr=0.25, rtol=0.35, dimunion=False
):
    """Per-drawing circle recall/precision/radius-MAE on full held-out images (clean or scan).
    This is the number comparable to the detector-benchmark 'circle recall' column."""
    torch, nn, F = _torch()
    import cv2

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_model().to(dev)
    net.load_state_dict(torch.load(ckpt, map_location=dev))
    net.eval()
    pairs = find_pairs(root)
    if split_json and os.path.exists(split_json):
        val = set(json.load(open(split_json))["val"])
        pairs = [p for p in pairs if os.path.basename(p[0]) in val]
    print(
        f"evalfull: {len(pairs)} held-out drawings; scan={use_scan} thr={thr} dimunion={dimunion} dev={dev}"
    )
    TP = FP = FN = 0
    errs = []
    per_recall = []
    ndraw = 0
    BANDS = [(3, 8), (8, 15), (15, 30), (30, 80), (80, 600)]
    btp = {b: 0 for b in BANDS}
    bn = {b: 0 for b in BANDS}
    for png, gj in pairs:
        ip = png
        if use_scan:
            sp = png[:-4] + ".scan.png" if png.endswith(".png") else png
            if os.path.exists(sp):
                ip = sp
        img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        g = json.load(open(gj))
        circ = _gt_centers(circles_from_graph(g))
        aff = (
            ((g.get("source") or {}).get("scan_affine") or g.get("scan_affine"))
            if use_scan
            else None
        )
        if aff:
            circ = [
                (*_apply_aff((cx, cy), aff), r * aff.get("scale", 1.0), vis)
                for cx, cy, r, vis in circ
            ]
        if not circ:
            continue
        ndraw += 1
        region = _labeled_region(g, aff)
        dets = [
            d for d in _detect_full(net, img, dev, thr=thr) if _in_region(d, region)
        ]
        if (
            dimunion
        ):  # union with circles recovered from diameter dimensions (OCR fallback)
            anchors = [
                (x, y, vmm)
                for (x, y, vmm) in _dim_anchors(g, aff)
                if _in_region((x, y), region)
            ]
            # recover this drawing's scale s (px per mm-of-radius) from anchors that already have a
            # nearby detection: s = median(det_r_px / (value_mm/2)). Uses only detector outputs +
            # dim text — no GT radius. Robust because within a drawing the true scale is constant.
            cal = [
                nd / (vmm / 2.0)
                for ax, ay, vmm in anchors
                if vmm > 0
                for nd in [
                    min(
                        (
                            dr
                            for dx, dy, dr, sc in dets
                            if math.hypot(dx - ax, dy - ay) < 15
                        ),
                        default=None,
                    )
                ]
                if nd is not None
            ]
            s = float(np.median(cal)) if cal else None
            if s:  # APPEND-ONLY: place a missed dimensioned bore (none detected nearby) at value*scale
                for ax, ay, vmm in anchors:
                    r_px = (vmm / 2.0) * s
                    if not (RMIN_PX <= r_px <= RMAX_PX):
                        continue
                    if any(
                        math.hypot(dx - ax, dy - ay) < max(10, 0.4 * max(r_px, dr))
                        for dx, dy, dr, sc in dets
                    ):
                        continue
                    dets.append((ax, ay, r_px, 1.0))
        used = set()
        tp = 0
        for dx, dy, dr, sc in dets:
            m = None
            best = 1e9
            for j, (gx, gy, gr, _) in enumerate(circ):
                if j in used:
                    continue
                d = math.hypot(dx - gx, dy - gy)
                if d < max(8, 0.3 * gr) and abs(dr - gr) <= rtol * gr and d < best:
                    best = d
                    m = j
            if m is not None:
                tp += 1
                used.add(m)
                errs.append(abs(dr - circ[m][2]))
            else:
                FP += 1
        TP += tp
        FN += len(circ) - tp
        for j, (gx, gy, gr, _) in enumerate(circ):
            for b in BANDS:
                if b[0] <= gr < b[1]:
                    bn[b] += 1
                    btp[b] += 1 if j in used else 0
        per_recall.append(tp / max(1, len(circ)))
    rec = TP / (TP + FN + 1e-9)
    prec = TP / (TP + FP + 1e-9)
    mae = float(np.mean(errs)) if errs else float("nan")
    mrec = float(np.mean(per_recall)) if per_recall else 0.0
    print(f"  drawings={ndraw}  GT circles(centers)={TP + FN}  detections-matched={TP}")
    print(
        f"  micro recall {rec:.3f}  precision {prec:.3f}  radius-MAE {mae:.1f}px  per-drawing-mean recall {mrec:.3f}"
    )
    print("  stratified recall by GT radius:")
    for b in BANDS:
        rb = btp[b] / bn[b] if bn[b] else float("nan")
        print(f"    r {b[0]:>3}-{b[1]:<3} px  recall {rb:.2f} ({btp[b]}/{bn[b]})")
    return {
        "recall": rec,
        "precision": prec,
        "radius_mae_px": mae,
        "per_drawing_recall": mrec,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "drawings": ndraw,
        "stratified": {f"{b[0]}-{b[1]}": (btp[b], bn[b]) for b in BANDS},
    }


def _apply_aff(pt, aff):
    cx, cy = aff["center_px"]
    rot = math.radians(aff["rotation_deg"])
    sc = aff["scale"]
    x, y = pt[0] - cx, pt[1] - cy
    ca, sa = math.cos(-rot), math.sin(-rot)
    return (
        (x * ca - y * sa) * sc + cx + aff["tx"],
        (x * sa + y * ca) * sc + cy + aff["ty"],
    )


def _dim_anchors(g, aff=None):
    """Diameter-dimension anchors = (center_px, value_mm) per diameter dim — the 'OCR fallback'
    source. center = the referenced circle primitive (= leader/centerline association); value_mm
    = the dimension text (= what OCR reads). Both are oracle here (read from GT) so we measure the
    fallback's ceiling; swap the body for a pixel-OCR + geometric binder for the real pipeline.
    NB: we do NOT convert mm->px with the graph's `px_per_mm` — that field is unreliable (only ~16%
    of circles satisfy r_px==r_mm*px_per_mm). Within a drawing r_px/r_mm is constant, so the caller
    recovers the scale empirically (OrthoSolve-style) from dim+detection pairs."""
    cid = {}
    for v in g.get("views", []):
        for p in v.get("primitives", []):
            if p.get("type") in ("circle", "arc"):
                cid[(v.get("name"), p["id"])] = (p["center"][0], p["center"][1])
    out = []
    for d in g.get("annotations", g.get("dimensions", [])):
        if str(d.get("kind", d.get("type"))).lower() != "diameter":
            continue
        val = d.get("value")
        view = d.get("view")
        if val is None:
            continue
        cen = next(
            (cid[(view, r)] for r in d.get("refs", []) if (view, r) in cid), None
        )
        if cen is None:
            continue
        x, y = cen
        if aff:
            x, y = _apply_aff((x, y), aff)
        out.append((x, y, abs(float(val))))
    return out


if __name__ == "__main__":
    a = sys.argv
    if len(a) < 2:
        print(__doc__)
        sys.exit(0)
    cmd = a[1]
    if cmd == "labeltest":
        labeltest()
    elif cmd == "audit":
        audit(a[2])
    elif cmd == "train":
        ap = argparse.ArgumentParser()
        ap.add_argument("root")
        ap.add_argument("outdir")
        ap.add_argument("--steps", type=int, default=600)
        ap.add_argument("--tile", type=int, default=512)
        ap.add_argument("--bs", type=int, default=8)
        n = ap.parse_args(a[2:])
        train(n.root, n.outdir, n.steps, n.tile, n.bs)
    elif cmd == "eval":
        ap = argparse.ArgumentParser()
        ap.add_argument("ckpt")
        ap.add_argument("real_png")
        ap.add_argument("--known", default="12,28,35,38,40,50,60,61,82,188,200")
        n = ap.parse_args(a[2:])
        eval_real(n.ckpt, n.real_png, [float(x) for x in n.known.split(",")])
    elif cmd == "evalfull":
        ap = argparse.ArgumentParser()
        ap.add_argument("ckpt")
        ap.add_argument("root")
        ap.add_argument("--split", default=None)
        ap.add_argument("--scan", action="store_true")
        ap.add_argument("--thr", type=float, default=0.25)
        ap.add_argument("--dimunion", action="store_true")
        n = ap.parse_args(a[2:])
        evalfull(n.ckpt, n.root, n.split, n.scan, n.thr, dimunion=n.dimunion)
    else:
        print("unknown", cmd)
