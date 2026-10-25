"""The final set on one split: hole edges gathered per hole by hole_doubt, scored against the
geometric truth; cavity walls gathered per cavity, scored against the judged errors.
    python final_eval.py ORACLE HOLE_FOLDER WALL_FOLDER"""
import collections, json, sys
from pathlib import Path
from experiments.rag.grounded import CONTRADICTED_BY, _trees, doubt, hole_doubt
from experiments.rag.schema import Judged, read_rows
from experiments.rag.trees import Tree

A = Path("/home/ryotaro/my_works/drawing2cad/outputs/20261024_rag_pilot/analysis_100")
judged = {r.key: r for r in read_rows(A / "judged.jsonl", Judged) if r.judgement is not None}
oracle = {(r["key"], r["id"]): r for r in map(json.loads, open(sys.argv[1]))}
hole_dir, wall_dir = Path(sys.argv[2]), Path(sys.argv[3])


def any_error(key, feature):
    return any(e.feature in feature.split(" + ") for e in judged[key].judgement.errors)


for reader in ("sol", "luna"):
    usage, keys = [0, 0, 0], set()
    # Holes
    qs = [q for q in map(json.loads, open(hole_dir / "questions.jsonl")) if q["kind"] == "hole edge"]
    keys |= {q["key"] for q in qs}
    trees = _trees(hole_dir)
    ans = {(a["key"], a["id"]): a for a in map(json.loads, open(hole_dir / f"answers_{reader}.jsonl"))}
    holes = collections.defaultdict(list)
    for q in qs:
        holes[(q["key"], q["id"].split("_")[0])].append(q)
        a = ans.get((q["key"], q["id"]))
        if a:
            usage[0] += len(a["usage"]); usage[1] += sum(u["input_tokens"] for u in a["usage"]); usage[2] += sum(u["output_tokens"] for u in a["usage"])
    h = collections.Counter()
    for hid, hq in holes.items():
        cons = [Tree.model_validate(trees[str(q["tree_fold"])]["hole edge"]).conclude((ans.get((q["key"], q["id"])) or {}).get("answers") or {}, q["hidden_drawn"]) for q in hq]
        if not hole_doubt(cons):
            continue
        o = oracle.get(hid)
        h["doubted"] += 1
        h["unscored"] += o is None or o["build"] != o["claim"]
        h["wrong claim"] += bool(o and o["build"] == o["claim"] and o["truth"] != o["claim"])
        h["any error"] += bool(o and o["build"] == o["claim"] and o["truth"] != o["claim"]) or any_error(hid[0], hq[0]["feature"])
    wrong_total = sum(1 for r in oracle.values() if r["build"] == r["claim"] and r["truth"] != r["claim"])
    # Cavity walls
    qs = [q for q in map(json.loads, open(wall_dir / "questions.jsonl")) if q["kind"] == "cavity wall"]
    keys |= {q["key"] for q in qs}
    trees = _trees(wall_dir)
    ans = {(a["key"], a["id"]): a for a in map(json.loads, open(wall_dir / f"answers_{reader}.jsonl"))}
    cav = {}
    for q in qs:
        a = ans.get((q["key"], q["id"]))
        if not a or not a.get("answers"):
            continue
        usage[0] += len(a["usage"]); usage[1] += sum(u["input_tokens"] for u in a["usage"]); usage[2] += sum(u["output_tokens"] for u in a["usage"])
        c = Tree.model_validate(trees[str(q["tree_fold"])]["cavity wall"]).conclude(a["answers"], q["hidden_drawn"])
        if doubt(q, c):
            cav[(q["key"], q["id"].split("_")[0])] = q
    scope = {qq for qq, _ in CONTRADICTED_BY["cavity wall"]}
    strict = sum(any(e.feature in q["feature"].split(" + ") and e.decision.question in scope for e in judged[k].judgement.errors) for (k, _), q in cav.items())
    lenient = sum(any_error(k, q["feature"]) for (k, _), q in cav.items())
    in_scope = sum(1 for k in keys for e in judged[k].judgement.errors if e.decision.question in scope)
    n = len(keys)
    print(f"  {reader}: holes doubted {h['doubted']} (wrong through/blind {h['wrong claim']}, any error {h['any error']}, "
          f"unscored {h['unscored']}; caught {h['wrong claim']}/{wrong_total}); cavities doubted {len(cav)} "
          f"(in-scope error {strict}, any error {lenient}; {in_scope} in-scope errors in all); "
          f"per reading {usage[0] / n:.1f} calls, {usage[1] / n:,.0f} in, {usage[2] / n:,.0f} out")
