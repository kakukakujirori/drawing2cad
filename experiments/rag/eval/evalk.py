"""Score sets of mark kinds from answer folders against the judged errors.

    python evalk.py LABEL=FOLDER:kind,kind ... [--list]

For each reader: per kind hits/doubts and decided/questions, the judged errors in scope of
those kinds that some doubt matches, and cost per reading."""
import collections, json, sys
from pathlib import Path
from experiments.rag.grounded import CONTRADICTED_BY, doubt, _trees
from experiments.rag.schema import Judged, read_rows
from experiments.rag.trees import Tree

A = Path("/home/ryotaro/my_works/drawing2cad/outputs/20261024_rag_pilot/analysis_100")
judged = {r.key: r for r in read_rows(A / "judged.jsonl", Judged) if r.judgement is not None}
show = "--list" in sys.argv
configs = []
for arg in sys.argv[1:]:
    if arg.startswith("--"):
        continue
    label, rest = arg.split("=", 1)
    parts = []
    for piece in rest.split("+"):
        folder, kinds = piece.split(":")
        parts.append((Path(folder), tuple(kinds.split(","))))
    configs.append((label, parts))

for reader in ("sol", "luna"):
    print(f"\n== {reader}")
    for label, parts in configs:
        table, caught, listed, usage, keys = collections.defaultdict(collections.Counter), {}, [], [0, 0, 0], set()
        flagged = {}
        scope = {q for _, kinds in parts for k in kinds for q, _ in CONTRADICTED_BY[k]}
        for folder, kinds in parts:
            qs = [q for q in map(json.loads, open(folder / "questions.jsonl")) if q["kind"] in kinds]
            keys |= {json.loads(l)["key"] for l in open(folder / "questions.jsonl")}
            path = folder / f"answers_{reader}.jsonl"
            ans = {(a["key"], a["id"]): a for a in map(json.loads, open(path))} if path.exists() else {}
            trees = _trees(folder)
            for q in qs:
                t = table[q["kind"]]
                t["questions"] += 1
                a = ans.get((q["key"], q["id"]))
                if not a or not a.get("answers"):
                    continue
                usage[0] += len(a["usage"]); usage[1] += sum(u["input_tokens"] for u in a["usage"]); usage[2] += sum(u["output_tokens"] for u in a["usage"])
                c = Tree.model_validate(trees[str(q["tree_fold"])][q["kind"]]).conclude(a["answers"], q["hidden_drawn"])
                t["decided"] += c != "undecided"
                if not doubt(q, c):
                    continue
                t["doubts"] += 1
                match = [e for e in judged[q["key"]].judgement.errors if e.feature in q["feature"].split(" + ")
                         and any(e.decision.question == eq and (r is None or e.decision.read == r) for eq, r in CONTRADICTED_BY[q["kind"]])]
                t["hits"] += bool(match)
                for e in match:
                    caught[(q["key"].split("/")[2], e.feature, e.decision.question)] = True
                listed.append(("HIT " if match else "miss", q["part"], q["id"], q["feature"], c))
                any_error = any(e.feature in q["feature"].split(" + ") for e in judged[q["key"]].judgement.errors)
                flagged.setdefault((q["key"], q["feature"]), [False, False])
                flagged[(q["key"], q["feature"])][0] |= bool(match)
                flagged[(q["key"], q["feature"])][1] |= any_error
        errors = [(k.split("/")[2], e.feature, e.decision.question) for k in sorted(keys) for e in judged[k].judgement.errors if e.decision.question in scope]
        kinds = [k for _, ks in parts for k in ks]
        hits, doubts = sum(table[k]["hits"] for k in kinds), sum(table[k]["doubts"] for k in kinds)
        print(f"{label:10s} all {hits}/{doubts}, caught {len(caught)}/{len(errors)} {sorted(caught)}")
        print(f"    features flagged {len(flagged)}: in-scope error {sum(v[0] for v in flagged.values())}, "
              f"any judged error {sum(v[1] for v in flagged.values())}")
        for k in kinds:
            print(f"    {k:13s} {table[k]['hits']}/{table[k]['doubts']} (decided {table[k]['decided']}/{table[k]['questions']})")
        print(f"    per reading {usage[0] / len(keys):.1f} calls, {usage[1] / len(keys):,.0f} in, {usage[2] / len(keys):,.0f} out")
        if show:
            for row in sorted(listed, key=lambda r: r[1:]):
                print("     ", *row)
