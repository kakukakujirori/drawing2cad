"""Hole-edge doubts gathered per hole and scored against the geometric truth, on holes whose
build agrees with the claim.  python hole_score2.py ORACLE.jsonl FOLDER"""
import collections, json, sys
from pathlib import Path
from experiments.rag.grounded import _trees, doubt
from experiments.rag.trees import Tree

oracle = {(r["key"], r["id"]): r for r in map(json.loads, open(sys.argv[1]))}
folder = Path(sys.argv[2])
qs = [q for q in map(json.loads, open(folder / "questions.jsonl")) if q["kind"] == "hole edge"]
trees = _trees(folder)
holes = collections.defaultdict(list)
for q in qs:
    holes[(q["key"], q["id"].split("_")[0])].append(q)
usable = {h: v for h, v in holes.items() if h in oracle and oracle[h]["build"] == oracle[h]["claim"]}
wrong = [h for h in usable if oracle[h]["truth"] != oracle[h]["claim"]]
all_wrong = sum(1 for h, r in oracle.items() if r["build"] == r["claim"] and r["truth"] != r["claim"])
print(f"== {folder.name}: {len(usable)} usable holes asked of {len(holes)}; claim wrong on {len(wrong)} of them ({all_wrong} in all usable holes)")
for reader in ("sol", "luna"):
    path = folder / f"answers_{reader}.jsonl"
    if not path.exists():
        continue
    ans = {(a["key"], a["id"]): a["answers"] or {} for a in map(json.loads, open(path))}
    flagged, listed = 0, []
    right = 0
    for h, hq in usable.items():
        cons = [Tree.model_validate(trees[str(q["tree_fold"])]["hole edge"]).conclude(ans.get((q["key"], q["id"]), {}), q["hidden_drawn"]) for q in hq]
        if any(doubt(q, c) for q, c in zip(hq, cons)):
            flagged += 1
            ok = oracle[h]["truth"] != oracle[h]["claim"]
            right += ok
            listed.append(("RIGHT" if ok else "wrong", h[0].split("/")[2], h[1], oracle[h]["claim"], oracle[h]["truth"], cons))
    print(f"  {reader}: holes doubted {right}/{flagged}, claim-wrong holes caught {right}/{len(wrong)} (of {all_wrong} overall)")
    for row in listed:
        print("     ", *row)
