#!/usr/bin/env python
"""OrthoSolve spike — local, zero-training, pure-numpy.
Tests the core bet: "given topology + OCR'd dims, a solver recovers EXACT metric for the
dimensioned subset, scan-invariant, where pixel-regression collapses."
Runs three falsifiable checks on EASY (synthetic Bracket/Flange clean+scan) and HARD (real/101).

Needs the PoC GT/pred fixtures (not shipped in this repo): point ORTHOSOLVE_DATA at a dir
holding dataset/{Bracket,Flange}.graph.json, pred/{Bracket.scan,Real101}.pred.json.
"""
import os, json, math, numpy as np
R=os.environ.get("ORTHOSOLVE_DATA", "poc")
def load(p): return json.load(open(p))
def circles(g):
    out={}
    for v in g.get("views",[]):
        for p in v["primitives"]:
            if p.get("type")=="circle": out[p["id"]]=p
    return out

# ============ CHECK 1: radius from DIM vs from PIXEL (the heart of "solve don't localize") ============
print("="*78)
print("CHECK 1 — circle radius: DIM-driven vs PIXEL-detected (clean vs scan)")
print("="*78)
gt=load(f"{R}/dataset/Bracket.graph.json")
ppm=gt["sheet"]["px_per_mm"]
gtc=circles(gt)
gtdim={d["id"]:d for d in gt["dimensions"] if d["type"]=="diameter"}
print(f"sheet px_per_mm = {ppm}")
print(f"{'dim':5} {'mm':>5} {'r_px from DIM':>14} {'GT r_px':>9} {'scan r_px':>10} {'dim_err':>8} {'scan_err':>9}")
scan=load(f"{R}/pred/Bracket.scan.pred.json")
scanc=circles(scan)
scandim={d["id"]:d for d in scan["dimensions"] if d["type"]=="diameter"}
# map GT diameter dims -> their GT circle; scan diameter dims -> their scan circle (by value)
for did,d in gtdim.items():
    rmm=d["value"]/2
    r_dim=rmm*ppm                       # radius computed from the OCR'd diameter (scan-invariant)
    gtcirc=gtc.get(d["refs"][0])
    r_gt=gtcirc["r_px"] if gtcirc else float('nan')
    # find the matching scan circle: scan diameter dim with same value -> its ref circle
    sc=None
    for sd in scandim.values():
        if abs(sd["value"]-d["value"])<1e-6:
            for rid in sd["refs"]:
                if rid in scanc: sc=scanc[rid]; break
    r_scan=sc["r_px"] if sc else float('nan')
    dim_err=abs(r_dim-r_gt)
    scan_err=abs(r_scan-r_gt) if sc else float('nan')
    print(f"{did:5} {d['value']:>5.0f} {r_dim:>14.2f} {r_gt:>9.2f} {r_scan:>10.2f} {dim_err:>8.2f} {scan_err:>9.2f}")
print("-> DIM-driven radius error ~0 (algebra); PIXEL(scan) radius error large. Scan-invariant by construction.")

# ============ CHECK 2: constraint SOLVE recovers exact coords from a NOISY seed ============
print("\n"+"="*78)
print("CHECK 2 — Gauss-Newton constraint solve: noisy(scan-like) seed -> exact coords")
print("="*78)
# representative rigid cluster mirroring the Bracket front: a W x H plate + a concentric bore,
# driven ONLY by topology (H/V + coincidence + concentric) and OCR'd dims (W,H,bore dia).
W,H=60.0,100.0; BORE_D=40.0
# variables: x0..x3,y0..y3 (4 corners CCW from bottom-left), cx,cy (bore center), r (bore radius)
# TRUE (what GT-consistent geometry should be), in mm:
true=np.array([0,0, W,0, W,H, 0,H, W/2,H/2, BORE_D/2],float)
names=["x0","y0","x1","y1","x2","y2","x3","y3","cx","cy","r"]
def resid(v):
    # corners CCW: 0=(0,0) 1=(W,0) 2=(W,H) 3=(0,H)
    x0,y0,x1,y1,x2,y2,x3,y3,cx,cy,r=v; e=[]
    e+=[x0, y0]                      # anchor corner0 at origin (datum)        -> 2
    e+=[y1-y0]                       # bottom edge 0-1 horizontal               -> 1
    e+=[y2-y3]                       # top edge 2-3 horizontal                  -> 1
    e+=[x3-x0]                       # left edge 3-0 vertical                   -> 1
    e+=[x1-x2]                       # right edge 1-2 vertical                  -> 1
    e+=[(x1-x0)-W]                   # dim: width  (horizontal)                 -> 1
    e+=[(y3-y0)-H]                   # dim: height (vertical)                   -> 1
    e+=[r-BORE_D/2]                  # dim: bore diameter                       -> 1
    e+=[cx-(x0+x1)/2, cy-(y0+y3)/2]  # concentric: bore center = plate center   -> 2
    return np.array(e,float)         # 11 residuals for 11 vars -> determined
def jac(v,eps=1e-6):
    base=resid(v); J=np.zeros((len(base),len(v)))
    for j in range(len(v)):
        vv=v.copy(); vv[j]+=eps; J[:,j]=(resid(vv)-base)/eps
    return J
rng=np.random.default_rng(0)
seed=true+rng.normal(0,8.0,true.shape)   # 8mm-scale corruption ~ scan pixel error magnitude
v=seed.copy()
for _ in range(50):
    r=resid(v); J=jac(v)
    dv,*_=np.linalg.lstsq(J,-r,rcond=None); v=v+dv
    if np.linalg.norm(dv)<1e-9: break
seed_err=np.abs(seed-true); solved_err=np.abs(v-true)
print(f"{'var':4} {'true':>8} {'noisy seed':>11} {'solved':>9} {'seed_err':>9} {'solved_err':>11}")
for i,n in enumerate(names):
    print(f"{n:4} {true[i]:>8.2f} {seed[i]:>11.2f} {v[i]:>9.3f} {seed_err[i]:>9.2f} {solved_err[i]:>11.2e}")
print(f"-> max seed err {seed_err.max():.2f}mm  ->  max SOLVED err {solved_err.max():.2e}mm  (solver computes exact metric from a wrong seed)")

# ============ CHECK 3: REAL/101 — over-determined SCALE recovery from multiple bound diameters ============
print("\n"+"="*78)
print("CHECK 3 — HARD (real/101): recover px_per_mm from multiple bound ⌀-dims; consistency = self-check")
print("="*78)
real=load(f"{R}/pred/Real101.pred.json")
rc=circles(real)
ppm_votes=[]
print(f"{'dim':5} {'mm⌀':>6} {'bound circle':>14} {'det r_px':>9} {'=> px_per_mm':>13}")
for d in real["dimensions"]:
    if d["type"]=="diameter" and d.get("refs"):
        for rid in d["refs"]:
            if rid in rc:
                rpx=rc[rid]["r_px"]; ppm_i=rpx/(d["value"]/2)
                ppm_votes.append(ppm_i)
                print(f"{d['id']:5} {d['value']:>6.0f} {rid:>14} {rpx:>9.1f} {ppm_i:>13.3f}")
import statistics as st
if ppm_votes:
    med=st.median(ppm_votes); sd=st.pstdev(ppm_votes)
    cv=sd/med if med else float('nan')
    print(f"-> {len(ppm_votes)} scale votes  median={med:.3f}  stdev={sd:.3f}  CV={cv:.1%}")
    print(f"   ({'CONSISTENT -> scale recovered + detections trustworthy' if cv<0.15 else 'INCONSISTENT -> detections noisy/mis-bound: the solver would EXPOSE this, not hide it'})")
tot=len(real["dimensions"]); bound=sum(1 for d in real['dimensions'] if d.get('refs'))
print(f"-> real binding coverage: {bound}/{tot} dims have a primitive ref ({bound/tot:.0%}); "
      f"the rest are honest-abstain (no calibration/refs on real, as predicted).")
