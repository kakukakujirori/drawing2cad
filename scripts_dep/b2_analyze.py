#!/usr/bin/env python
"""(a)+(b) baseline analysis for B2-cadrille over all generation fixtures.
(a) validity/failure-mode rates. (b) scale-collapse metric (generated geometry
clusters at cadrille's normalized ~200-unit box, ignoring real mm dimensions)."""
import os, re, json, glob, statistics as st, sys

d = sys.argv[1] if len(sys.argv) > 1 else "experiments/b2_cadrille/whole268"
gen = {r["id"]: r for r in json.load(open(os.path.join(d, "summary.json")))}
ex = {r["id"]: r for r in json.load(open(os.path.join(d, "exec_summary.json")))}
ids = sorted(gen, key=int)
N = len(ids)

# (a) rates
parses = sum(gen[i]["parses"] for i in ids)
executed = sum(bool(ex.get(i, {}).get("executed_ok")) for i in ids)
watertight = sum(bool(ex.get(i, {}).get("watertight")) for i in ids)


def fail_cat(i):
    if ex.get(i, {}).get("watertight"):
        return "valid"
    err = (ex.get(i, {}) or {}).get("err") or ""
    if not gen[i]["parses"] or "was never closed" in err or "SyntaxError" in err:
        return "truncation/syntax"
    if "StdFail" in err or "BRep" in err:
        return "invalid_brep"
    if "timeout" in err:
        return "timeout"
    if ex.get(i, {}).get("executed_ok"):
        return "non_watertight"
    return "other:" + err[:40]


from collections import Counter
cats = Counter(fail_cat(i) for i in ids)

# (b) scale collapse: max extent of executed solids + extrude-constant histogram
extents = [max(ex[i]["bbox"]) for i in ids if ex.get(i, {}).get("bbox")]
mags = []
for i in ids:
    code = open(os.path.join(d, f"{i}.py")).read()
    for m in re.findall(r"(?:extrude|cylinder)\(\s*(-?\d+(?:\.\d+)?)", code):
        mags.append(abs(float(m)))

print(f"=== (a) B2-cadrille baseline on {N} generation fixtures (feed={os.path.basename(d)}) ===")
print(f"  gen parses     : {parses}/{N} ({parses/N:.0%})")
print(f"  executed (->solid): {executed}/{N} ({executed/N:.0%})")
print(f"  watertight (VALID): {watertight}/{N} ({watertight/N:.0%})")
print(f"  failure modes  : {dict(cats)}")
print(f"=== (b) scale-collapse metric ===")
if extents:
    cov = st.pstdev(extents) / st.mean(extents) if st.mean(extents) else 0
    near200 = sum(180 <= e <= 220 for e in extents)
    print(f"  generated max-extent over {len(extents)} valid solids:")
    print(f"    min/med/max = {min(extents):.0f}/{st.median(extents):.0f}/{max(extents):.0f}  "
          f"mean={st.mean(extents):.1f}  CoV={cov:.3f}")
    print(f"    within [180,220] (~200 norm box): {near200}/{len(extents)} ({near200/len(extents):.0%})")
if mags:
    near200m = sum(180 <= m <= 220 for m in mags)
    print(f"  extrude/cylinder magnitudes over {len(mags)} ops: "
          f"med={st.median(mags):.0f}  ==200: {mags.count(200.0)}  in[180,220]: {near200m} ({near200m/len(mags):.0%})")
print("  -> interpretation: real mechanical parts span widely (tens..hundreds mm); cadrille")
print("     emits a ~constant normalized box => absolute dimensions are ignored. Under")
print("     CADGenBench's scale-sensitive alignment (rot+trans, no scale norm), shape/")
print("     interface ~ 0 even when watertight.")
