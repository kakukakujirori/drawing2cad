"""Per-hole doubt rules over hole-edge conclusions, scored against the geometric truth (strict)
and against any judged error on the hole's feature (lenient)."""
import collections, json, sys
from pathlib import Path
from experiments.rag.grounded import _trees
from experiments.rag.schema import Judged, read_rows
from experiments.rag.trees import Tree

A = Path("/home/ryotaro/my_works/drawing2cad/outputs/20261024_rag_pilot/analysis_100")
judged = {r.key: r for r in read_rows(A / "judged.jsonl", Judged) if r.judgement is not None}
RULES = {
    "any no edge": lambda c: c.count("no edge") >= 1,
    "half or more": lambda c: c.count("no edge") >= max(1, len(c) / 2),
    "two, or all": lambda c: c.count("no edge") >= 2 or (c and all(x == "no edge" for x in c)),
}
for oracle_path, folder in zip(sys.argv[1::2], sys.argv[2::2]):
    oracle = {(r["key"], r["id"]): r for r in map(json.loads, open(oracle_path))}
    folder = Path(folder)
    qs = [q for q in map(json.loads, open(folder / "questions.jsonl")) if q["kind"] == "hole edge"]
    trees = _trees(folder)
    holes = collections.defaultdict(list)
    for q in qs:
        holes[(q["key"], q["id"].split("_")[0])].append(q)
    usable = {h: v for h, v in holes.items() if h in oracle and oracle[h]["build"] == oracle[h]["claim"]}
    n_wrong = sum(1 for h, r in oracle.items() if r["build"] == r["claim"] and r["truth"] != r["claim"])
    print(f"== {folder.name} ({n_wrong} claim-wrong usable holes)")
    for reader in ("sol", "luna"):
        ans = {(a["key"], a["id"]): a["answers"] or {} for a in map(json.loads, open(folder / f"answers_{reader}.jsonl"))}
        for name, rule in RULES.items():
            flagged = strict = lenient = 0
            for h, hq in usable.items():
                cons = [Tree.model_validate(trees[str(q["tree_fold"])]["hole edge"]).conclude(ans.get((q["key"], q["id"]), {}), q["hidden_drawn"]) for q in hq]
                if not rule(cons):
                    continue
                flagged += 1
                strict += oracle[h]["truth"] != oracle[h]["claim"]
                feature = hq[0]["feature"]
                lenient += oracle[h]["truth"] != oracle[h]["claim"] or any(
                    e.feature in feature.split(" + ") for e in judged[h[0]].judgement.errors)
            print(f"  {reader:4s} {name:13s} doubted {flagged:2d}: through/blind wrong {strict:2d}, any error {lenient:2d}; caught {strict}/{n_wrong}")
