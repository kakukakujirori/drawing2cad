"""Unit tests for the grounded trees: drawn situations with known answers, told in
words to Sol acting as the reader, and run through each tree. Trees that conclude
wrongly are given their failures and rewritten, and tested again.

    python -m experiments.rag.tree_tests TREES.json OUT.json [ROUNDS]

A situation passes when the tree's conclusion is one it allows; "undecided" is
allowed only where the lines cannot settle the claim, or where it raises no doubt.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from string import Template

from langchain_core.messages import HumanMessage

from experiments.rag.judge import sol
from experiments.rag.trees import HIDDEN, MARKS, MAX_GROUNDED_NODES, Tree, _ask_tree

OBSERVE = Path(__file__).with_name("tree_observer.md").read_text()
REPAIR = Path(__file__).with_name("tree_repair.md").read_text()

# (name, hidden lines drawn, what is drawn, the conclusions allowed)
SCENARIOS: dict[str, list[tuple[str, bool, str, set[str]]]] = {
    "hole band": [
        ("through", True, "In the view across the hole, two dashed lines run along the band, one near each long edge, from the "
         "part's outline on one side to its outline on the other, without a break. In the view along the axis, a circle "
         "sits inside the red outline.", {"through"}),
        ("blind, flat bottom", True, "Two dashed lines run along the band near its long edges, starting at the part's outline "
         "on the left. About halfway across, they end at a short dashed line drawn across the band that joins them. From "
         "there to the part's outline on the right, nothing is drawn inside the band.", {"blind"}),
        ("blind, drill point", True, "Two dashed lines run along the band near its long edges from the part's outline on the "
         "left. About halfway across, two short dashed lines slant inward from their ends and meet at a point on the "
         "band's centre. Beyond that point nothing is drawn in the band.", {"blind"}),
        ("through, counterbore", True, "From the part's outline on the left, two dashed lines run just outside the band's "
         "long edges, a little wider than the band, to a short dashed line across. From the ends of that line, two dashed "
         "lines continue along the band's long edges to the part's outline on the right.", {"through"}),
        ("through, countersink seen from the narrow end", True, "From the part's outline on the right, two dashed lines run "
         "along the band's long edges to a short dashed line drawn across the band, about a third of the way across. From "
         "the ends of that line, two dashed lines slant outward, leave the band, and reach the part's outline on the left, "
         "where they are much farther apart than the band is wide.", {"through"}),
        ("through, into a cavity", True, "Two dashed lines run along the band's long edges from the part's outline on "
         "the left to a long dashed line that crosses the band and continues far beyond both of its long edges; that line "
         "is one side of a large dashed rectangle, a cavity, that reaches nearly to the outline on the right. Inside the "
         "rectangle nothing more is drawn in the band.", {"through"}),
        ("blind, at a boss's junction", True, "Two dashed lines run along the band's long edges from the boss's end "
         "outline on the left to a solid line crossing the band where the boss meets a plate; that line runs far beyond "
         "both long edges. Past it the band carries on over the plate to the plate's far outline, with nothing drawn "
         "inside the band.", {"blind"}),
        ("through, past a junction", True, "Two dashed lines run along the band's long edges from the outline on the "
         "left, cross a solid line where a boss meets a plate, and go on to the part's outline on the right.",
         {"through"}),
        ("blind, another feature's line beyond", True, "Two dashed lines run along the band's long edges from the part's "
         "outline at the bottom and end at a short dashed line joining them, about two thirds of the way up. Above it, a "
         "long dashed line of another feature crosses the band and continues far beyond both edges; the part's outline at "
         "the top is solid and unbroken. Between the short joining line and the top outline, no lines run along the band.",
         {"blind"}),
        ("blind, a cavity line just beyond", True, "Two dashed lines run along the band's long edges from the part's "
         "outline at the bottom and end at a short dashed line joining them, which stays between the band's long edges. "
         "Slightly further up, a long dashed line of a recess crosses the whole band and runs far past both of its "
         "edges.", {"blind"}),
        ("blind, opening on the far side", True, "In the view along the axis, the circle inside the red outline is dashed. In "
         "the view across, two dashed lines start at the part's outline on the right and stop at a short dashed line across "
         "the band, leaving a clear gap before the outline on the left.", {"blind"}),
        ("through, with a centre line", True, "Two dashed lines run along the band's long edges from outline to outline; a "
         "thin dash-dot centre line runs along the middle of the band and past the outlines.", {"through"}),
        ("no hidden lines drawn", False, "In the view across the hole, no lines run along the band; only the part's outline "
         "crosses its two ends. In the view along the axis, a solid circle sits inside the red outline.", {"undecided"}),
    ],
    "face bars": [
        ("as read", True, "The body's two long outline edges, thick and solid, run exactly along the two red bars over the "
         "body's whole length.", {"as read"}),
        ("larger", True, "The body's thick solid outline edges parallel to the bars lie outside both red bars, each beyond its "
         "bar by about the distance between the bars.", {"beyond the bars"}),
        ("smaller", True, "Both thick outline edges of the body parallel to the bars lie between the two red bars, with a "
         "clear gap between each edge and its bar.", {"short of the bars"}),
        ("chamfered corners", True, "The body's thick outline edges run along the two red bars; at each end they turn through "
         "a short slanted segment into the end outlines.", {"as read"}),
        ("a boss beyond one bar", True, "The body's thick outline edges run along the red bars. Over the middle third of the "
         "length, a boss's outline sticks out beyond one bar and comes back to it.", {"as read", "undecided"}),
        ("a dimension line on a bar", True, "A thin extension line runs exactly along one red bar. The body's thick outline "
         "edges both lie outside the bars.", {"beyond the bars"}),
        ("one side beyond", True, "One thick outline edge of the body lies along its red bar; the other lies well beyond its "
         "bar, away from the space between the bars.", {"beyond the bars", "undecided"}),
        ("a hidden face", True, "One face of the body is hidden behind a flange and is drawn as a dashed line along its red "
         "bar; the other face is a thick solid line along the other red bar.", {"as read"}),
    ],
    "wall line": [
        ("hidden wall", True, "A separate dashed line runs along the red line over its whole length, a little inside the "
         "part's solid outline; at both ends it meets dashed lines that run across to the outline.", {"wall drawn"}),
        ("no wall", True, "Nothing runs along the red line. The part's solid outline runs straight beside it, close but "
         "clearly separate.", {"no wall"}),
        ("draft", True, "Nothing runs along the red line except the part's solid outline, which slants slightly: it lies "
         "beside the red line at one end and crosses it near the other end.", {"no wall", "undecided"}),
        ("chamfer", True, "Nothing runs along the red line over most of its length; near one end a short slanted solid segment "
         "joins the outline at a corner.", {"no wall", "undecided"}),
        ("visible wall", True, "A separate solid line runs along the red line over its whole length; at both ends solid lines "
         "run across from it to the part's outline.", {"wall drawn"}),
        ("a hole's lines across", True, "A short pair of dashed lines from a hole crosses the red line at one place; otherwise "
         "nothing runs along the red line, and the outline beside it is straight.", {"no wall", "undecided"}),
    ],
    "body region": [
        ("solid with holes", True, "Inside the tint, the only lines are three small solid circles of holes and their thin "
         "centre lines. In the other views, short pairs of dashed lines show the holes.", {"solid", "undecided"}),
        ("hollow box", True, "Inside the tint, a dashed closed outline follows the body's outline all around, a short "
         "distance inside it. In another view, a similar dashed inset outline follows the body's outline.", {"hollow"}),
        ("thin-walled shell", True, "Along the body's outline, inside the tint, run pairs of close parallel solid lines; in "
         "another view the same close pairs follow the outline.", {"hollow"}),
        ("a recess only", True, "Inside the tint, a dashed rectangle covers about a quarter of the body near one corner and "
         "does not follow the body's outline; in another view two dashed lines show it reaching only partway into the "
         "body.", {"solid", "undecided"}),
        ("no hidden lines drawn", False, "Inside the tint nothing is drawn but thin centre lines.", {"solid", "undecided"}),
    ],
    "body edges": [
        ("as read", True, "Every red band has a thick solid line running inside it over its whole length, and every row of "
         "orange dots has a dashed line running along it. No other thick line runs near the marks.", {"as read"}),
        ("larger", True, "Inside the red bands nothing is drawn. Outside them, parallel to each band and clearly apart, the "
         "body's thick solid outline runs all around, farther from the body's middle than the bands.", {"drawn larger"}),
        ("smaller", True, "Inside the red bands nothing is drawn. The body's thick solid outline runs parallel to the bands "
         "but inside them, nearer the body's middle, with white between each line and its band.", {"drawn smaller"}),
        ("thicker in one direction", True, "Along the two short end bands, thick solid lines run inside the bands. The two "
         "long bands are empty; the body's long outline edges run parallel to them, well outside, so the drawn body is "
         "wider than the marks across that direction.", {"drawn larger", "undecided"}),
        ("chamfered corners", True, "Thick solid lines run inside all the red bands. Where two bands meet at a sharp "
         "corner, the drawn outline leaves the bands and cuts the corner with a short slanted line.", {"as read"}),
        ("draft", True, "Thick solid lines run inside most bands. One drawn outline line lies inside its band at one end "
         "and drifts out to lie just beside the band, touching it, at the other end.", {"as read", "undecided"}),
        ("a dimension line on a band", True, "A thin line runs inside one red band; the other bands are empty. The body's "
         "thick solid outline runs outside all the bands, parallel to them and clearly apart.", {"drawn larger"}),
        ("other features' lines cross", True, "Thick solid lines run inside all the red bands. Two dashed lines of a hole "
         "cross one band and continue on both sides of it.", {"as read"}),
        ("no hidden lines drawn", False, "There are only red bands, no dots. A thick solid line runs inside every band.",
         {"as read"}),
    ],
    "cavity edges": [
        ("hidden walls", True, "A dashed line runs along every row of orange dots over its whole length, and the dashed "
         "lines meet at the corners where the rows meet.", {"walls drawn"}),
        ("no walls", True, "No line runs along any row of orange dots. The part's thick solid outline runs beside the "
         "outer rows, close to them but with clear white between.", {"no walls"}),
        ("draft", True, "No dashed line runs along the dots. The only line near them is the part's solid outline, which "
         "slants: beside one row of dots at one end, crossing it near the other end.", {"no walls", "undecided"}),
        ("open to the surface", True, "Where the marks are red bands, thick solid lines run inside them; dashed lines run "
         "along every row of orange dots.", {"walls drawn"}),
        ("a hollow of another size", True, "No line runs along the orange dots. Well inside them, a dashed rectangle is "
         "drawn, clearly smaller than the rectangle the dots outline.", {"no walls", "undecided"}),
        ("a hole's lines across", True, "A short pair of dashed lines from a hole crosses one row of dots; otherwise "
         "nothing runs along the dots.", {"no walls", "undecided"}),
        ("no hidden lines drawn", False, "There are only red bands, no dots, at the cavity's opening; thick solid lines "
         "run inside them.", {"walls drawn", "undecided"}),
    ],
    "body interior": [
        ("solid with holes", True, "Inside the tint, the only lines are hole circles and dashed hole sides; each lies "
         "inside a red band or along orange dots. Thin centre lines cross the tint.", {"solid"}),
        ("hollow box", True, "Inside the tint, a dashed closed outline follows the body's outline all round, a short "
         "distance inside it; no band or dots cover it. In another view a similar unmarked dashed outline is drawn.",
         {"hollow"}),
        ("thin-walled shell", True, "Along the body's outline, inside the tint, run pairs of close parallel solid lines; "
         "the outer line of each pair lies in a red band, the inner one is unmarked. The other view shows the same.",
         {"hollow"}),
        ("an unmarked recess", True, "Inside the tint, a small dashed rectangle near one corner is unmarked; it does not "
         "follow the body's outline. Every other line lies in a band or along dots.", {"solid", "undecided"}),
        ("the reading's own cavity", True, "Inside the tint, a dashed closed outline set in from the body's outline lies "
         "along a row of orange dots over its whole length. No unmarked lines are inside the tint.", {"solid"}),
        ("only thin lines unmarked", True, "Inside the tint, thin dimension and extension lines cross unmarked; every "
         "thick or dashed line lies in a band or along dots.", {"solid"}),
        ("no hidden lines drawn", False, "Inside the tint, only red bands appear, each with a thick solid line inside it.",
         {"solid", "undecided"}),
    ],
    "cavity wall": [
        ("hidden wall", True, "A dashed line runs along the row of orange dots over its whole length.", {"wall drawn"}),
        ("visible wall", True, "A thick solid line runs inside the red band over its whole length.", {"wall drawn"}),
        ("no wall", True, "Nothing runs along the row of orange dots. The part's thick outline runs parallel to it a "
         "little way off, with clear white between.", {"no wall"}),
        ("empty band", True, "Nothing is drawn inside the red band. A thick solid line runs parallel to it, clearly "
         "apart.", {"no wall"}),
        ("thin wall under the outline", True, "The row of orange dots lies against the inner edge of the part's thick "
         "solid outline, partly covered by it. No separate dashed line runs along the dots.", {"no wall", "undecided"}),
        ("a crossing line", True, "A short pair of dashed lines from a hole crosses the row of dots near its middle; "
         "otherwise nothing runs along the dots.", {"no wall", "undecided"}),
        ("part of the length", True, "A dashed line runs along the first third of the row of dots and then turns away; "
         "along the rest of the row nothing is drawn.", {"undecided"}),
        ("draft", True, "No dashed line runs along the dots. The part's solid outline slants across them, beside the "
         "dots at one end and crossing them at the other.", {"no wall", "undecided"}),
        ("a dashed line close beside", True, "A dashed line runs parallel to the row of dots over its whole length, so "
         "close that it touches the dots in places.", {"wall drawn", "undecided"}),
        ("no hidden lines drawn", False, "A thick solid line runs inside the red band over its whole length.",
         {"wall drawn"}),
    ],
    "hole edge": [
        ("hidden side", True, "A dashed line runs along the row of orange dots over its whole length.", {"edge drawn"}),
        ("visible side", True, "A thick solid line runs inside the red band over its whole length.", {"edge drawn"}),
        ("no line", True, "Nothing runs along the row of orange dots; the nearest line runs parallel to it with clear "
         "white between.", {"no edge"}),
        ("a missing bottom", True, "The row of orange dots runs across the hole's width; no line runs along it, and the "
         "hole's two dashed side lines run on past it.", {"no edge"}),
        ("a centre line along", True, "Only a thin line of long and short dashes runs along the row of orange dots.",
         {"no edge", "undecided"}),
        ("a crossing line", True, "A dashed line of another hole crosses the row of dots near its middle; otherwise "
         "nothing runs along the dots.", {"no edge", "undecided"}),
        ("part of the length", True, "A dashed line runs along the first third of the row of dots and then stops at a "
         "short line across; along the rest of the row nothing is drawn.", {"undecided", "no edge"}),
        ("under a thick outline", True, "The row of orange dots lies against a thick solid outline, partly covered by "
         "it; no separate dashed line runs along the dots.", {"no edge", "undecided"}),
        ("no hidden lines drawn", False, "A thick solid line runs inside the red band over its whole length.",
         {"edge drawn"}),
    ],
    "body extent": [
        ("as read", True, "Each of the two red bands has a thick solid line running inside it over its whole length.",
         {"as read"}),
        ("larger", True, "Both red bands are empty. The body's thick outline runs parallel to each band, outside it and "
         "clearly apart, so the drawn body is wider than the marks.", {"drawn larger"}),
        ("smaller", True, "Both red bands are empty. The body's thick outline runs parallel to each band, inside the "
         "space between the bands and clearly apart, so the drawn body is narrower than the marks.", {"drawn smaller"}),
        ("one side beyond", True, "A thick solid line runs inside one red band. The other band is empty; the body's "
         "outline on that side runs parallel to it, well outside.", {"drawn larger", "undecided"}),
        ("a neighbour beyond the ends", True, "Thick solid lines run inside both red bands. Near one end, the curved "
         "outline of a fillet joining another feature sweeps outward beyond one band.", {"as read"}),
        ("chamfered corner", True, "Thick solid lines run inside both red bands; at their ends the drawn outline turns "
         "through a short slanted piece.", {"as read"}),
        ("draft", True, "Thick solid lines run inside both bands, except that one drawn line slants slightly: inside its "
         "band at one end, just touching its edge at the other.", {"as read", "undecided"}),
        ("a dimension line on a band", True, "A thin line runs inside one red band; the other band is empty. The body's "
         "thick outline runs parallel to both bands, outside them and clearly apart.", {"drawn larger"}),
        ("a hidden side", True, "A thick solid line runs inside the red band on one side; a dashed line runs along the "
         "row of orange dots on the other side.", {"as read"}),
        ("an inner step", True, "Thick solid lines run inside both red bands. A second thick outline, of a step or "
         "raised face, runs parallel to each band inside the space between them, clearly apart.", {"as read"}),
        ("the body goes on", True, "A thick solid line runs inside the red band on one side. On the other side nothing "
         "runs along the row of orange dots; there the drawn outline carries on past the dots and slants away from the "
         "other side.", {"drawn larger", "undecided"}),
        ("no hidden lines drawn", False, "Thick solid lines run inside both red bands over their whole length.",
         {"as read"}),
    ],
}


def observe(model, mark: str, tree: Tree, hidden: bool, scene: str) -> dict[str, str]:
    asks = [node for node in tree.nodes if node.id != HIDDEN]
    text = Template(OBSERVE).substitute(
        mark=MARKS[mark].description, scene=scene + ("" if hidden else " Hidden lines are not drawn in this drawing."),
        questions="\n".join(f"{node.id}: {node.ask} [{' / '.join(node.answers)}]" for node in asks))
    raw = model.invoke([HumanMessage(content=text)]).text
    return json.loads(raw[raw.index("{"):raw.rindex("}") + 1])["answers"]


def _path(tree: Tree, answers: dict[str, str], hidden: bool) -> list[str]:
    nodes, steps, node = {n.id: n for n in tree.nodes}, [], tree.root
    while node in nodes:
        answer = ("yes" if hidden else "no") if node == HIDDEN else answers.get(node)
        steps.append(f"{node}={answer}")
        node = nodes[node].answers.get(answer, "=undecided")
    return steps


def test(model, mark: str, tree: Tree) -> list[dict]:
    def one(case):
        name, hidden, scene, allowed = case
        answers = observe(model, mark, tree, hidden, scene)
        conclusion = tree.conclude(answers, hidden)
        return {"name": name, "scene": scene, "hidden": hidden, "allowed": sorted(allowed), "answers": answers,
                "path": _path(tree, answers, hidden), "conclusion": conclusion, "ok": conclusion in allowed}

    with ThreadPoolExecutor(8) as pool:
        return list(pool.map(one, SCENARIOS[mark]))


def repair(model, mark: str, tree: Tree, results: list[dict]) -> Tree:
    spec = MARKS[mark]

    def told(r):
        return (f"- {r['name']}: {r['scene']}{'' if r['hidden'] else ' Hidden lines are not drawn.'} Answers: "
                f"{json.dumps(r['answers'])}. Path: {' > '.join(r['path'])}. Concluded {r['conclusion']}; "
                f"should be {' or '.join(r['allowed'])}.")

    text = Template(REPAIR).substitute(
        mark=spec.description, conclusions=", ".join(spec.conclusions),
        confusers="\n".join(f"- {line}" for line in spec.confusers),
        tree=tree.model_dump_json(exclude={"question"}, indent=1),
        failures="\n".join(told(r) for r in results if not r["ok"]),
        passes="\n".join(f"- {r['name']}: {r['scene']}" for r in results if r["ok"]) or "(none)",
        max_nodes=MAX_GROUNDED_NODES)
    return _ask_tree(model, text, mark, MAX_GROUNDED_NODES)


if __name__ == "__main__":
    trees = json.loads(Path(sys.argv[1]).read_text())
    out, rounds = Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 2
    model = sol("medium")
    log = []

    def work(job: tuple[str, str]) -> tuple[str, str, Tree]:
        parts_fold, mark = job
        tree = Tree.model_validate(trees[parts_fold][mark])
        for n in range(rounds + 1):
            results = test(model, mark, tree)
            failed = [r["name"] + " -> " + r["conclusion"] for r in results if not r["ok"]]
            log.append({"fold": parts_fold, "mark": mark, "round": n, "results": results})
            print(f"fold {parts_fold} {mark} round {n}: {len(results) - len(failed)}/{len(results)} pass; failed {failed}", flush=True)
            if not failed or n == rounds:
                return parts_fold, mark, tree
            tree = repair(model, mark, tree, results)
        return parts_fold, mark, tree

    with ThreadPoolExecutor(8) as pool:
        for parts_fold, mark, tree in pool.map(work, [(f, m) for f in trees for m in trees[f]]):
            trees[parts_fold][mark] = tree.model_dump()
    out.write_text(json.dumps(trees, indent=1))
    out.with_suffix(".log.json").write_text(json.dumps(log, indent=1))
