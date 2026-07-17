#!/usr/bin/env python
"""(a) Improved CLASSICAL circle detection on real/101: contour + radial-consistency fit
(not Hough). Tests whether better classical CV cuts the 2425 false-positive flood down to
~real count, catches the big bores, and makes the over-determined scale test NON-vacuous.

Needs the real fixtures (not shipped): point DETECT_DATA at a dir holding
real/101_input.png and pred/Real101.pred.json."""
import os, cv2, json, numpy as np
from collections import defaultdict
R=os.environ.get("DETECT_DATA", "poc")
img=cv2.imread(f"{R}/real/101_input.png",cv2.IMREAD_GRAYSCALE)
_,bw=cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# close small gaps so broken circle strokes form a contour
bw=cv2.morphologyEx(bw,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8),iterations=1)
cnts,_=cv2.findContours(bw,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
print(f"contours: {len(cnts)}")

cands=[]
for c in cnts:
    if len(c)<20: continue
    pts=c[:,0,:].astype(np.float64)
    cx,cy=pts.mean(0)
    rad=np.hypot(pts[:,0]-cx,pts[:,1]-cy)
    r=rad.mean()
    if r<8 or r>800: continue
    cvr=rad.std()/r                       # radial consistency: 0 = perfect circle
    # angular coverage: a real circle outline spans most angles (not a corner arc)
    ang=np.arctan2(pts[:,1]-cy,pts[:,0]-cx)
    cover=len(np.unique((ang*18/np.pi).astype(int)))/36.0   # fraction of 36 angular bins hit
    if cvr<0.06 and cover>0.7:            # near-perfect circle, mostly complete
        cands.append((cx,cy,r))
print(f"circle-like contours (radial cvr<0.06, coverage>0.7): {len(cands)}")

# dedup: same center (within 12px); keep distinct radii (>6% apart) -> concentric rings survive
groups=defaultdict(list)
for cx,cy,r in cands: groups[(round(cx/12),round(cy/12))].append((cx,cy,r))
circles=[]
for g in groups.values():
    g.sort(key=lambda t:t[2]); kept=[]
    for cx,cy,r in g:
        if not kept or abs(r-kept[-1][2])/r>0.06: kept.append((cx,cy,r))
    circles+=kept
det=sorted(r for _,_,r in circles)
print(f"after dedup: {len(circles)} circles; radii px: {[round(r) for r in det]}")

# scale recovery + NULL test
real=json.load(open(f"{R}/pred/Real101.pred.json"))
exp_r=sorted(set(d["value"]/2 for d in (real.get("annotations") or real.get("dimensions",[])) if d.get("kind",d.get("type"))=="diameter"))
print(f"known ⌀ (mm): {[round(e*2) for e in exp_r]}")
TOL=0.06
def matched(s): return sum(1 for r in det if min(abs(r-s*e)/(s*e) for e in exp_r)<TOL)
cap=sorted({r/e for r in det for e in exp_r})
best=max(((matched(s),s) for s in cap),default=(0,0))
nm,s=best
nulls=[matched(x) for x in np.linspace(min(cap),max(cap),60)] if cap else [0]
null_mean=sum(nulls)/len(nulls)
print(f"\nbest px_per_mm={s:.3f} -> {nm}/{len(det)} matched ({nm/max(len(det),1):.0%})")
print(f"NULL (arbitrary scales): mean {null_mean:.1f}/{len(det)} ({null_mean/max(len(det),1):.0%}), max {max(nulls)}")
signal = nm - null_mean
print(f"signal = best - null_mean = {signal:.1f}")
if det:
    print(f"under best s: ⌀200->{s*100:.0f}px ⌀188->{s*94:.0f}px ⌀12->{s*6:.0f}px (max detected r={max(det):.0f})")
    big=[r for r in det if r>s*40]   # bores bigger than ⌀80
    print(f"large circles detected (r> {s*40:.0f}px ~ ⌀80+): {[round(r) for r in big]}")
verdict = ("NON-VACUOUS — classical contour-fit gives a real scale signal" if signal>=4 and len(det)<150
           else "still vacuous/over-detected — needs trained detector" if len(det)>=150
           else "weak signal — trained detector likely needed")
print(f"=> {verdict}")

# overlay
out=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
for cx,cy,r in circles: cv2.circle(out,(int(cx),int(cy)),int(r),(0,0,255),3)
cv2.imwrite(f"{R}/detect_101_contour.png",out)
print(f"saved overlay -> {R}/detect_101_contour.png")
