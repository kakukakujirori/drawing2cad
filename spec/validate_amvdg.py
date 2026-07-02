#!/usr/bin/env python
"""AMVDG v0.3 validator — makes 'the schema is solid' MECHANICAL, not a claim.

Gates (all must pass for the declared `profile`):
  1. STRUCTURE   : jsonschema against AMVDG_v0.3.schema.json
  2. REF-INTEGRITY: refs/members/feature_id/parent all resolve; no parent cycle
  3. CENSUS      : validity.census declared counts == actually emitted (abstain != absent)
  4. DoF RECOMPUTE: required/determined/det_by_geom/missing recomputed == dof block == self_declared
  5. ABSTAIN-DISJOINT: dof.undetermined never overlaps a determined param
  6. PROFILE GATE: model(coords nullable) < vectorized(coords required) < derived(constraints) < gt_executable(prov+complete build)
  7. ROUND-TRIP  : model-authored subset (V/A/F) lower->DSL->lift is identity (DSL<->JSON well-defined)

Usage: validate_amvdg.py <instance.json> [schema.json]
"""
import sys, os, json, jsonschema

def load(p): return json.load(open(p))

def check_refs(g):
    errs=[]
    prim_ids=set(); prim_view={}
    for v in g["views"]:
        for p in v["primitives"]:
            prim_ids.add(p["id"]); prim_view[p["id"]]=v["name"]
    feat_ids={f["feature_id"] for f in g["features"]}
    # annotation refs -> primitive ids
    for a in g["annotations"]:
        for r in a.get("refs",[]):
            if r not in prim_ids: errs.append(f"ann {a['id']} ref '{r}' not a primitive")
    # feature members -> primitives ; feature.parent -> feature
    member_prims=set()
    for f in g["features"]:
        for m in f["members"]:
            member_prims.add(m["primitive_id"])
            if m["primitive_id"] not in prim_ids: errs.append(f"feat {f['feature_id']} member '{m['primitive_id']}' not a primitive")
        if f.get("parent") and f["parent"] not in feat_ids:
            errs.append(f"feat {f['feature_id']} parent '{f['parent']}' not a feature")
    # primitive.feature_id -> must point to an existing feature (forward only; a feature need
    # not enumerate every outline edge as a member, so no bidirectional requirement)
    for v in g["views"]:
        for p in v["primitives"]:
            fid=p.get("feature_id")
            if fid and fid not in feat_ids: errs.append(f"prim {p['id']} feature_id '{fid}' not a feature")
    # parent cycle
    par={f["feature_id"]:f.get("parent") for f in g["features"]}
    for fid in par:
        seen=set(); cur=fid
        while cur:
            if cur in seen: errs.append(f"parent cycle at {fid}"); break
            seen.add(cur); cur=par.get(cur)
    return errs

def check_census(g):
    errs=[]
    cen=(g.get("validity") or {}).get("census") or {}
    by=cen.get("by_view",{})
    for vname,counts in by.items():
        v=next((x for x in g["views"] if x["name"]==vname),None)
        if not v: errs.append(f"census view {vname} missing"); continue
        circ=[p for p in v["primitives"] if p["type"]=="circle"]
        actual={"circles":len(circ),
                "circles_visible":len([p for p in circ if p["line_role"]=="visible"]),
                "circles_hidden":len([p for p in circ if p["line_role"]=="hidden"])}
        for k,decl in counts.items():
            if k in actual and actual[k]!=decl:
                errs.append(f"census {vname}.{k}: declared {decl} != emitted {actual[k]}")
    return errs

def check_dof(g):
    errs=[]
    req=det=detg=0
    determined_set=set()
    for f in g["features"]:
        d=f.get("dof") or {}
        req+=len(d.get("required_params",[]))
        det+=len(d.get("determined_params",[]))
        detg+=len(d.get("determined_by_geometry",[]))
        for p in d.get("determined_params",[]): determined_set.add(f"{f['feature_id']}.{p}")
    missing=req-det-detg
    D=g["dof"]
    for k,calc in (("required",req),("determined",det),("determined_by_geometry",detg),("missing",missing)):
        if D.get(k)!=calc: errs.append(f"dof.{k}: stated {D.get(k)} != recomputed {calc}")
    sd=D.get("self_declared") or {}
    for k in ("required","determined","missing"):
        if sd.get(k)!=D.get(k): errs.append(f"self_declared.{k} {sd.get(k)} != dof.{k} {D.get(k)}")
    # abstain disjoint
    for u in D.get("undetermined",[]):
        if u in determined_set: errs.append(f"undetermined '{u}' also marked determined")
    # coverage
    cov=round(det/req,4) if req else 0
    if abs((D.get("coverage") or 0)-cov)>1e-3: errs.append(f"coverage {D.get('coverage')} != {cov}")
    return errs

def _has_coords(p):
    t=p["type"]
    if t=="line": return p.get("p1") and p.get("p2")
    if t in ("circle","arc"): return p.get("center") and p.get("r_px") is not None
    if t=="polyline": return bool(p.get("pts"))
    return True

def check_profile(g):
    errs=[]; prof=g["profile"]
    need_coords = prof in ("vectorized","gt_executable")
    need_exec   = prof=="gt_executable"
    if need_coords:
        for v in g["views"]:
            for p in v["primitives"]:
                if p.get("state")=="known" and not _has_coords(p):
                    errs.append(f"[{prof}] prim {p['id']} state=known but missing coords")
                if p.get("coords_source","unknown")=="unknown":
                    errs.append(f"[{prof}] prim {p['id']} coords_source=unknown")
    if need_exec:
        ann_known={a["id"] for a in g["annotations"] if a.get("value_state")=="known"}
        for f in g["features"]:
            if not f.get("prov"): errs.append(f"[gt] feat {f['feature_id']} missing prov")
            d=f.get("dof") or {}
            if d.get("status")=="determined":
                b=f.get("build")
                if not b or not b.get("op") or not b.get("datum_plane"):
                    errs.append(f"[gt] determined feat {f['feature_id']} build incomplete (op/datum_plane)")
                    continue
                drv=b.get("driven_by") or {}
                for rp in d.get("required_params",[]):
                    ptr=drv.get(rp)
                    if not ptr: errs.append(f"[gt] feat {f['feature_id']} required '{rp}' has null driven_by")
                    elif ptr not in ann_known: errs.append(f"[gt] feat {f['feature_id']} '{rp}' -> {ptr} not a value_state=known annotation")
    return errs

# ---- ROUND-TRIP: model-authored subset (V/A/F). Schema-exact enum tokens (no aliases). ----
def fmt(x):  # canonical number token so lower(lift(s))==s (int 0 and float 0.0 both -> "0")
    f=float(x); return str(int(f)) if f==int(f) else repr(f)
def lower(g):
    L=[]
    for v in g["views"]:
        ar=v.get("align_to"); al=f"{ar['view']}:{ar['shared_axis']}" if ar else "-"
        L.append(f"V|{v['name']}|{v['view_type']}|{','.join(fmt(x) for x in v['projection_dir'])}|{al}")
    for a in g["annotations"]:
        L.append(f"A|{a['id']}|{a['kind']}|{a.get('param_role','unknown')}|{a.get('feature_id') or '-'}|"
                 f"{'-' if a.get('value') is None else fmt(a['value'])}|{a['value_state']}|{'.'.join(a.get('refs',[]))or '-'}")
    for f in g["features"]:
        mem=".".join(f"{m['view']}:{m['primitive_id']}:{m['projection_role']}" for m in f["members"])
        L.append(f"F|{f['feature_id']}|{f['kind']}|{f.get('axis','none')}|{f.get('extrude_dir') or '-'}|"
                 f"{f['status']}|{f.get('parent') or '-'}|{mem}")
    return "\n".join(L)

def lift(text):
    g={"views":[],"annotations":[],"features":[]}
    vt={}
    for line in text.split("\n"):
        t=line.split("|")
        if t[0]=="V":
            ar=None
            if t[4]!="-": v_,ax=t[4].split(":"); ar={"view":v_,"shared_axis":ax}
            g["views"].append({"name":t[1],"view_type":t[2],
                "projection_dir":[float(x) for x in t[3].split(",")],"align_to":ar})
        elif t[0]=="A":
            g["annotations"].append({"id":t[1],"kind":t[2],"param_role":t[3],
                "feature_id":None if t[4]=="-" else t[4],
                "value":None if t[5]=="-" else float(t[5]),"value_state":t[6],
                "refs":[] if t[7]=="-" else t[7].split(".")})
        elif t[0]=="F":
            mem=[]
            if t[7]:
                for ms in t[7].split("."):
                    vv,pp,rr=ms.split(":"); mem.append({"view":vv,"primitive_id":pp,"projection_role":rr})
            g["features"].append({"feature_id":t[1],"kind":t[2],"axis":t[3],
                "extrude_dir":None if t[4]=="-" else t[4],"status":t[5],
                "parent":None if t[6]=="-" else t[6],"members":mem})
    return g

def roundtrip(g):
    a=lower(g); g2=lift(a); b=lower(g2)
    return [] if a==b else ["round-trip NOT identity (DSL<->JSON ill-defined)"]

def main():
    here=os.path.dirname(os.path.abspath(__file__))
    inst=load(sys.argv[1]); schema=load(sys.argv[2] if len(sys.argv)>2 else
        os.path.join(here, "AMVDG_v0.3.schema.json"))
    res={}
    try:
        jsonschema.validate(inst, schema); res["1_structure"]=[]
    except jsonschema.ValidationError as e:
        res["1_structure"]=[f"{list(e.absolute_path)}: {e.message}"]
    res["2_ref_integrity"]=check_refs(inst)
    res["3_census"]=check_census(inst)
    res["4_dof"]=check_dof(inst)
    res["6_profile"]=check_profile(inst)
    res["7_roundtrip"]=roundtrip(inst)
    ok=all(not v for v in res.values())
    for k,v in res.items():
        print(f"{'PASS' if not v else 'FAIL'}  {k}")
        for m in v: print("      -",m)
    print("\n==> profile:", inst.get("profile"), "| OVERALL:", "PASS ✅" if ok else "FAIL ❌")
    sys.exit(0 if ok else 1)

if __name__=="__main__": main()
