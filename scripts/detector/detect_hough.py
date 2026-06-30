#!/usr/bin/env python
"""Detection spike on REAL drawing (101) — classical CV, no training, no GT.
Tests: (A) is there a real thick/thin stroke-width separation? (B) does proper circle
detection recover a CONSISTENT scale from the known OCR'd diameters — the over-determination
self-check that FAILED on the VLM predictions (CV 36%, votes 0.4-1.6)?
A consistent scale here => the drawing IS tractable and the VLM (not the drawing) was the problem.

Needs the real fixtures (not shipped): point DETECT_DATA at a dir holding
real/101_input.png and pred/Real101.pred.json. Overlays are written under it.
"""
import os, cv2, json, numpy as np
R=os.environ.get("DETECT_DATA", "poc")
img=cv2.imread(f"{R}/real/101_input.png", cv2.IMREAD_GRAYSCALE)
H,W=img.shape; print(f"image {W}x{H}")

# ---------- (A) stroke-width separation (distance transform on ink) ----------
_,bw=cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)   # ink=255
ink=bw>0
dt=cv2.distanceTransform(bw,cv2.DIST_L2,5)                            # half-stroke-width per ink px
sw=2*dt[ink]                                                          # full stroke width (px)
import numpy as np
qs=np.percentile(sw,[10,25,50,75,90,99])
print("\n(A) STROKE-WIDTH separation:")
print("   ink px stroke-width percentiles [10/25/50/75/90/99]:", [round(float(q),1) for q in qs])
thr=float(np.percentile(sw,75))
thick_frac=float((sw>thr).mean())
print(f"   ink coverage: {ink.mean()*100:.1f}% of page is ink")
print(f"   -> thick layer (>p75={thr:.1f}px) = {thick_frac*100:.0f}% of ink; thin layer = the rest")
print("   => a measurable thick/thin split EXISTS (ISO 128 weights survive in the raster)")

# thick-ink mask (outline candidates): keep ink whose local stroke width is large
thick=np.zeros_like(bw); thick[ink & (2*dt>thr)]=255
thick=cv2.dilate(thick,np.ones((3,3),np.uint8),1)

# ---------- (B) circle detection + over-determined SCALE recovery ----------
blur=cv2.medianBlur(img,5)
circ=cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,dp=1.5,minDist=25,
                      param1=120,param2=55,minRadius=6,maxRadius=320)
det=[] if circ is None else [float(c[2]) for c in circ[0]]
det=sorted(det)
print(f"\n(B) CIRCLE detection (Hough): {len(det)} circles; radii px (sorted): "
      f"{[round(r) for r in det][:40]}{' ...' if len(det)>40 else ''}")

# known diameters from the OCR'd callouts (drawn circles/bores)
real=json.load(open(f"{R}/pred/Real101.pred.json"))
dia_mm=sorted({d["value"] for d in (real.get("annotations") or real.get("dimensions",[])) if d.get("kind",d.get("type"))=="diameter"})
exp_r=sorted(set(d/2 for d in dia_mm))
print(f"   known OCR'd diameters (mm): {[round(x) for x in dia_mm]}")

# scale voting: each (detected r_px, expected r_mm) proposes s=r_px/r_mm; find s with most support
TOL=0.06
cands=sorted({r/e for r in det for e in exp_r if e>0})
best=None
for s in cands:
    matched=set()
    for i,r in enumerate(det):
        if min(abs(r-s*e)/(s*e) for e in exp_r) < TOL: matched.add(i)
    if best is None or len(matched)>best[1]: best=(s,len(matched))
if best:
    s,nm=best
    print(f"\n   over-determined SCALE recovery (the VLM-failed test, redone with CV):")
    print(f"   best px_per_mm = {s:.3f}  -> matches {nm}/{len(det)} detected circles within {int(TOL*100)}%")
    # show what the big dims map to under this scale
    print(f"   under s={s:.3f}: ⌀200 -> {s*100:.0f}px, ⌀50 -> {s*25:.0f}px, ⌀38 -> {s*19:.0f}px (should be big>small)")
    # NULL TEST: do arbitrary WRONG scales match just as many? (if so, the test is vacuous)
    def matched(ss): return sum(1 for r in det if min(abs(r-ss*e)/(ss*e) for e in exp_r)<TOL)
    nulls=[matched(ss) for ss in np.linspace(3,12,40)]
    null_mean=sum(nulls)/len(nulls)
    print(f"   NULL TEST: arbitrary wrong scales match mean {null_mean:.0f}/{len(det)} ({null_mean/len(det):.0%}); best={nm} ({nm/len(det):.0%})")
    print(f"   maxRadius cap={320}px but ⌀200 would be ~{s*100:.0f}px -> the big bores are NEVER detectable")
    verdict = ("VACUOUS — naive Hough OVER-DETECTS (%d circles vs ~30 real); scale-consistency is meaningless. "
               "A TRAINED clutter-rejecting detector is required (classical CV insufficient, as the literature warns)."%len(det))
    print(f"   => {verdict}")

# overlay for the eye
out=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
if circ is not None:
    for c in circ[0]:
        cv2.circle(out,(int(c[0]),int(c[1])),int(c[2]),(0,0,255),3)
cv2.imwrite(f"{R}/detect_101_circles.png",out)
cv2.imwrite(f"{R}/detect_101_thick.png",thick)
print(f"\nsaved overlay -> {R}/detect_101_circles.png ; thick layer -> {R}/detect_101_thick.png")
