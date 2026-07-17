#!/usr/bin/env python3
"""Score an extractor-predicted intra-view graph against renderer ground truth.

Usage:
  python score.py PRED.json GT.graph.json [--scan]
Coords are global image pixels. With --scan, GT coords are mapped through gt['scan_affine']
(clean-px -> scan-px) before matching, because the prediction was made on the scan image.
Prints a metrics report and writes PRED.json.metrics.json next to PRED.

Design notes (so the metric is meaningful on REAL drawings, not just synthetic):
  * CURVES are pooled: circle + arc are matched together by center+radius. A real
    extractor (or VLM) constantly confuses a full circle with a near-full arc and a
    closed bore with an open fillet; penalising that as a miss measures labelling
    noise, not geometry recovery. The circle/arc LABEL agreement is reported
    separately as a non-gating `type_agree_rate`.
  * LINES are reported split by visibility: `visible` is the headline (the silhouette
    a real drawing actually shows), `hidden` and `all` are secondary. No extractor —
    and no human reading a scan — reproduces every dashed hidden segment, so visible
    recall is the honest bar.
  * LINKAGE (dim -> primitive binding) resolves through curve pairs (circle OR arc)
    and covers diameter AND radius dims.
"""
import json, sys, math

C_TOL = 35.0     # curve center match (px)
R_TOL = 18.0     # curve radius match (px), or 25% of r
L_TOL = 35.0     # line endpoint match (px)
V_ABS = 0.6      # dim value abs tol (mm)
V_REL = 0.02     # dim value rel tol

def load(p): return json.load(open(p))

def affine(aff):
    rot=math.radians(aff["rotation_deg"]); s=aff["scale"]
    cx,cy=aff["center_px"]; tx,ty=aff["tx"],aff["ty"]
    cos,sin=math.cos(rot),math.sin(rot)
    def f(pt):
        x,y=pt[0]-cx,pt[1]-cy
        x,y=s*(cos*x - sin*y), s*(sin*x + cos*y)
        return [x+cx+tx, y+cy+ty]
    return f

def gt_prims(gt, xf):
    out=[]
    for v in gt["views"]:
        for p in v["primitives"]:
            q=dict(p); q["view"]=v["name"]
            if p["type"]=="line":
                q["p1"]=xf(p["p1"]); q["p2"]=xf(p["p2"])
            else:
                q["center"]=xf(p["center"])
            out.append(q)
    return out

def pred_prims(pred):
    out=[]
    for v in pred.get("views",[]):
        for p in v.get("primitives",[]):
            q=dict(p); q.setdefault("line_role","visible"); q["view"]=v.get("name","?")
            out.append(q)
    return out

def d(a,b): return math.hypot(a[0]-b[0],a[1]-b[1])

def _vis(p): return p.get("line_role", p.get("visibility","visible"))
def _dkind(d): return d.get("kind", d.get("type"))   # annotation kind (v0.2 'kind' / v0 'type')
def _anns(g): return g.get("annotations", g.get("dimensions", []))

def match_curves(gt, pr, vis=None):
    """Pool circle+arc; match by center+radius. Returns tp, n_gt, n_pred, pairs, type_ok."""
    G=[p for p in gt if p["type"] in ("circle","arc") and (vis is None or _vis(p)==vis) and "center" in p]
    P=[p for p in pr if p["type"] in ("circle","arc") and (vis is None or _vis(p)==vis) and "center" in p]
    used=set(); tp=0; pairs=[]; typ=0
    for g in G:
        gr=g.get("r_px",0); best=None; bd=1e9
        for i,p in enumerate(P):
            if i in used: continue
            if d(g["center"],p["center"])<C_TOL and abs(gr-p.get("r_px",0))<max(R_TOL,0.25*gr):
                dist=d(g["center"],p["center"])
                if dist<bd: bd=dist; best=i
        if best is not None:
            used.add(best); tp+=1; pairs.append((g,P[best]))
            if P[best].get("type")==g.get("type"): typ+=1
    return tp,len(G),len(P),pairs,typ

def match_lines(gt, pr, vis=None):
    G=[p for p in gt if p["type"]=="line" and (vis is None or _vis(p)==vis)]
    P=[p for p in pr if p["type"]=="line" and (vis is None or _vis(p)==vis)]
    used=set(); tp=0
    for g in G:
        for i,p in enumerate(P):
            if i in used: continue
            a=(d(g["p1"],p.get("p1",[0,0]))<L_TOL and d(g["p2"],p.get("p2",[0,0]))<L_TOL)
            b=(d(g["p1"],p.get("p2",[0,0]))<L_TOL and d(g["p2"],p.get("p1",[0,0]))<L_TOL)
            if a or b: used.add(i); tp+=1; break
    return tp,len(G),len(P)

def prf(tp,ng,npd):
    P=tp/npd if npd else 0.0; R=tp/ng if ng else 0.0
    F=2*P*R/(P+R) if (P+R) else 0.0
    return {"P":round(P,3),"R":round(R,3),"F1":round(F,3),"tp":tp,"gt":ng,"pred":npd}

def val_match(gv,pv):
    return abs(gv-pv)<=max(V_ABS, V_REL*abs(gv))

def match_dims(gt,pred):
    G=_anns(gt); P=_anns(pred)
    used=set(); pairs=[]
    for g in G:
        best=None
        for i,p in enumerate(P):
            if i in used: continue
            if "value" in p and val_match(g["value"],p.get("value",1e9)):
                if best is None or (_dkind(p)==_dkind(g)): best=i
        if best is not None: used.add(best); pairs.append((g,P[best]))
    return pairs,len(G),len(P)

def main():
    pred_p, gt_p = sys.argv[1], sys.argv[2]
    scan = "--scan" in sys.argv[3:]
    pred=load(pred_p); gt=load(gt_p)
    _aff=(gt.get("source") or {}).get("scan_affine") or gt.get("scan_affine")
    xf=(affine(_aff) if (scan and _aff) else (lambda x:list(x)))
    G=gt_prims(gt,xf); Pr=pred_prims(pred)

    # curves (pooled circle+arc): visible headline + all
    ctp_v,cng_v,cnp_v,_,_       = match_curves(G,Pr,"visible")
    ctp_a,cng_a,cnp_a,cpairs,typ= match_curves(G,Pr,None)
    type_agree = round(typ/len(cpairs),3) if cpairs else None

    # lines split by visibility
    ltp_v,lng_v,lnp_v = match_lines(G,Pr,"visible")
    ltp_h,lng_h,lnp_h = match_lines(G,Pr,"hidden")
    ltp_a,lng_a,lnp_a = match_lines(G,Pr,None)

    # dims
    dpairs,dng,dnp=match_dims(gt,pred)
    val_ok=sum(1 for g,p in dpairs if val_match(g["value"],p["value"]))
    type_ok=sum(1 for g,p in dpairs if _dkind(p)==_dkind(g))
    dia_pairs=[(g,p) for g,p in dpairs if _dkind(g)=="diameter"]
    dia_mae=round(sum(abs(g["value"]-p["value"]) for g,p in dia_pairs)/len(dia_pairs),3) if dia_pairs else None

    # linkage (binding) over diameter+radius dims, resolved through curve pairs
    pid2gid={p.get("id"):g.get("id") for g,p in cpairs}
    link_ok=0; link_tot=0
    for g,p in dpairs:
        if _dkind(g) not in ("diameter","radius"): continue
        link_tot+=1
        mapped={pid2gid.get(r) for r in (p.get("refs") or [])}
        if set(g.get("refs",[])) & mapped: link_ok+=1

    m={
      "pred": pred_p, "gt": gt_p, "scan": scan,
      "primitives": {
        "curve": {"visible": prf(ctp_v,cng_v,cnp_v), "all": prf(ctp_a,cng_a,cnp_a),
                  "type_agree_rate": type_agree, "_note": "circle+arc pooled by center+radius"},
        "line":  {"visible": prf(ltp_v,lng_v,lnp_v), "hidden": prf(ltp_h,lng_h,lnp_h),
                  "all": prf(ltp_a,lng_a,lnp_a)},
      },
      "dimensions": {
        "count": prf(len(dpairs),dng,dnp),
        "value_exact_rate": round(val_ok/len(dpairs),3) if dpairs else None,
        "type_acc": round(type_ok/len(dpairs),3) if dpairs else None,
        "diameter_value_MAE_mm": dia_mae,
      },
      "linkage": {"correct": link_ok, "total": link_tot,
                  "acc": round(link_ok/link_tot,3) if link_tot else None,
                  "_note": "dim->primitive binding (diameter+radius), via curve pairs"},
    }
    json.dump(m, open(pred_p+".metrics.json","w"), indent=2)
    print(json.dumps(m, indent=2))

if __name__=="__main__": main()
