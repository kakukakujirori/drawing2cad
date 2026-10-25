"""Hole doubts scored against the geometric truth, on holes whose build agrees with the claim.
    python hole_score.py ORACLE.jsonl FOLDER [FOLDER...]"""
import collections, json, sys
from pathlib import Path
from experiments.rag.grounded import _trees, doubt
from experiments.rag.trees import Tree

oracle = {(r["key"], r["id"]): r for r in map(json.loads, open(sys.argv[1]))}
for folder in map(Path, sys.argv[2:]):
    qs = [q for q in map(json.loads, open(folder / "questions.jsonl")) if q["kind"] == "hole band"]
    trees = _trees(folder)
    usable = [q for q in qs if (q["key"], q["id"]) in oracle and oracle[(q["key"], q["id"])]["build"] == q["claim"]]
    wrong = [q for q in usable if oracle[(q["key"], q["id"])]["truth"] != q["claim"]]
    print(f"== {folder.name}: {len(usable)}/{len(qs)} holes usable, claim wrong on {len(wrong)}")
    for reader in ("sol", "luna"):
        path = folder / f"answers_{reader}.jsonl"
        if not path.exists():
            continue
        ans = {(a["key"], a["id"]): a["answers"] or {} for a in map(json.loads, open(path))}
        c = collections.Counter()
        listed = []
        for q in usable:
            truth = oracle[(q["key"], q["id"])]["truth"]
            con = Tree.model_validate(trees[str(q["tree_fold"])]["hole band"]).conclude(ans.get((q["key"], q["id"]), {}), q["hidden_drawn"])
            c["decided"] += con != "undecided"
            c["right when decided"] += con == truth
            if doubt(q, con):
                c["doubts"] += 1
                c["doubts right"] += truth != q["claim"]
                listed.append(("RIGHT" if truth != q["claim"] else "wrong", q["part"], q["id"], q["claim"], con, truth))
        print(f"  {reader}: doubts {c['doubts right']}/{c['doubts']}, claim-wrong holes caught {c['doubts right']}/{len(wrong)}, "
              f"decided {c['decided']}/{len(usable)} ({c['right when decided']} match the truth)")
        if "--list" in sys.argv[0] or len(sys.argv) > 0:
            for row in listed:
                print("     ", *row)
