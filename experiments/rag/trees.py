"""Decision trees that read a property of a feature from what is drawn: Sol writes
one per question from the judged mistakes, and a program follows a reader's
observations through it to a conclusion.

    python -m experiments.rag.trees JUDGED.jsonl OUT.json
    python -m experiments.rag.trees simplify TREES.json OUT.json
    python -m experiments.rag.trees grounded JUDGED.jsonl OUT.json [MARK...]
    python -m experiments.rag.trees merge OUT.json TREES.json...

Parts fall in two folds. Each fold's trees are written from the mistakes of its
own parts' first readings and are applied to the other fold's parts. `simplify`
has Sol rewrite each tree into short questions that reach the same conclusions.
`grounded` writes one tree per kind of mark: a reading's claim drawn onto the
views, so that every question points at the place it asks about. `merge` gathers
trees files into one, later files replacing a fold's trees of the same name: the
questions folder's trees_grounded_tested.json is made this way.
"""

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from string import Template

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, model_validator

from experiments.rag.judge import sol
from experiments.rag.schema import Judged, read_rows
from zeroshot.pipeline.retrieval.mistakes import QUESTIONS

PROMPT = Path(__file__).with_name("trees.md").read_text()
SIMPLIFY = Path(__file__).with_name("trees_simple.md").read_text()
GROUNDED = Path(__file__).with_name("grounded_trees.md").read_text()
# Questions whose answers a drawing settles for a feature alone.
TREE_QUESTIONS = ["through or blind", "solid or hollow", "open or closed", "edge"]
MAX_NODES = 8
MAX_SIMPLE_NODES = 12
TRIES = 3
HIDDEN = "hidden"  # the node the program answers: are hidden lines drawn?
MAX_GROUNDED_NODES = 10


class Mark(BaseModel):
    """A kind of claim drawn onto the views, and what a tree over it concludes."""

    description: str
    conclusions: tuple[str, ...]
    confusers: tuple[str, ...]
    # The judged mistakes it learns from: (question, the reading's answer or None for any).
    cases: tuple[tuple[str, str | None], ...]


MARKS = {
    "hole band": Mark(
        description=(
            "In a view across a hole, slot or pocket, a red band as wide as the hole, centred on it and running along "
            "its axis across the whole part, from the part's outline on one side to its outline on the other. The band "
            "does not say where the hole starts or ends: it shows only where to look. In a view along the axis, a red "
            "outline marks the hole's mouth."),
        conclusions=("through", "blind"),
        confusers=(
            "A short line across the band where the hole's width changes is not a bottom: past it the hole goes on, "
            "narrower or wider, as at a counterbore, step or countersink, where the wider part may be drawn as two "
            "slanted lines.",
            "A blind drilled hole may end in two slanted lines meeting at a point instead of a line straight across.",
            "Lines of other features, another hole, a recess floor or a rib, may cross or overlap the band; only the two "
            "lines running along the band near its long edges belong to the hole.",
            "A hole that opens into a cavity ends at the cavity's outline, not at the part's far outline; it still "
            "passes through the wall it is in.",
            "Where the two lines along the band end, look at the line they end at: a hole's bottom stays between the "
            "band's long edges, while a cavity's outline runs on well past both of them. A cavity line may cross the band "
            "just beyond a hole's bottom; the hole ends at the first line, not at the cavity.",
            "A solid line where the hole's body meets another body, such as a boss meeting a plate, may cross the band "
            "and run past both its long edges; past it the part goes on as material, with nothing drawn along the band. "
            "Two lines stopping at such a line show a hole that stops there; a cavity's outline instead closes around an "
            "empty space.",
            "Where hidden lines are not drawn, a hole inside the material does not show in a view across it.",
        ),
        cases=(("through or blind", None),),
    ),
    "face bars": Mark(
        description=(
            "In a view where two opposite faces of a body are seen edge-on, two red bars across the view at the positions "
            "the reading gives those faces, such as a plate's front and back, or a body's top and bottom."),
        conclusions=("as read", "beyond the bars", "short of the bars"),
        confusers=(
            "Chamfers and rounds at the body's edges make its outline meet a bar with a short slanted or curved piece.",
            "Features attached to the body, a boss, flange or rib, extend beyond its faces; judge the body's own outline, "
            "the lines running parallel to the bars.",
            "Dimension and extension lines are thin and may line up with a bar; outlines are thick.",
            "A face hidden behind other material is drawn dashed.",
        ),
        cases=(("thickness", None), ("how far", None)),
    ),
    "wall line": Mark(
        description=(
            "In a view where the inner face of a cavity's wall is seen edge-on, a red line where the reading puts that "
            "face, over the cavity's extent."),
        conclusions=("wall drawn", "no wall"),
        confusers=(
            "An outline drawn twice close together can come from a draft or taper, a slanted outline seen at both ends, "
            "or from a chamfer, instead of from a wall.",
            "The inner face of a wall hidden behind material is dashed; seen through an opening it is solid.",
            "A thin wall's inner face lies close to the outline; look for a separate line, not the outline itself.",
            "Dashed lines of holes and recesses inside the body run only over those features, not over the whole cavity.",
        ),
        cases=(("solid or hollow", "hollow"), ("open or closed", None)),
    ),
    "body region": Mark(
        description="In a view across a body the reading calls solid, a light red tint over that body.",
        conclusions=("solid", "hollow"),
        confusers=(
            "Dashed lines inside a body may come from holes, slots and recesses; a cavity shows as a closed inset outline "
            "that follows the body's own outline.",
            "Thin walls show as pairs of close parallel lines along the outline, in more than one view.",
            "Webs or ribs of a hollow body show as lines crossing the inset outline.",
        ),
        cases=(("solid or hollow", "solid"),),
    ),
    # The edge marks: a program builds the part as the reading describes it and draws the built part's
    # edges onto the drawing, so the marks follow the part's real outline instead of a box.
    "body edges": Mark(
        description=(
            "A program built the part as the reading describes it. On the drawing it marks the edges one body of that "
            "built part has in this view: a translucent red band along each edge seen directly, where the drawing should "
            "have a solid line, and a row of orange dots along each edge hidden behind material, where it should have a "
            "dashed line. The enlarged area is also shown again without the marks."),
        conclusions=("as read", "drawn larger", "drawn smaller"),
        confusers=(
            "A drawn line of the same edge runs inside its band or along its dots, at most a stroke's width off; a line "
            "running beside the band with clear white between them is a different edge.",
            "Chamfers and rounds make a drawn outline turn a corner with a short slanted or curved piece where the bands "
            "meet at a sharp corner, or the other way round; that is not a size difference.",
            "A draft or taper makes a drawn outline slant slightly across a straight band: on the band at one end, just "
            "beside it at the other.",
            "Lines of other features, holes, recesses and other bodies, cross the marks or run near them; judge only the "
            "lines along the marks and the body's own outline.",
            "Dimension, extension and centre lines are thin and may run along a band; outlines are thick.",
            "A hidden edge right behind a visible one is covered by the solid line and not drawn on its own.",
        ),
        cases=(("thickness", None), ("how far", None)),
    ),
    "cavity edges": Mark(
        description=(
            "A program built the part as the reading describes it. On the drawing it marks the edges one cavity, recess "
            "or void of that built part has in this view: a translucent red band along each edge seen directly, where "
            "the drawing should have a solid line, and a row of orange dots along each edge hidden behind material, "
            "where it should have a dashed line. The enlarged area is also shown again without the marks."),
        conclusions=("walls drawn", "no walls"),
        confusers=(
            "A drawn line of the same edge runs inside its band or along its dots, at most a stroke's width off; a line "
            "running beside them with clear white between is a different edge.",
            "An outline drawn twice close together can come from a draft or taper, a slanted outline seen at both ends, "
            "or from a chamfer, instead of from a cavity's wall.",
            "A thin wall's edges lie close to the part's outline; look for a separate line along the dots, not the "
            "outline itself.",
            "Dashed lines of holes and other recesses run over those features only, and may cross the dots.",
            "Dashed lines near the dots but clearly apart from them show a hollow of another size or place, not these "
            "walls.",
        ),
        cases=(("solid or hollow", "hollow"), ("open or closed", None)),
    ),
    "body interior": Mark(
        description=(
            "A program built the part as the reading describes it, and the reading calls one body solid. On the drawing "
            "a light red tint covers that body's box, and every edge the built part has inside the tint is marked: a "
            "translucent red band where the drawing should have a solid line, a row of orange dots where it should have "
            "a dashed line. A line inside the tint that no band or dots cover is drawn on the sheet but missing from the "
            "reading. The enlarged area is also shown again without the marks."),
        conclusions=("solid", "hollow"),
        confusers=(
            "Unmarked lines may belong to a hole, slot or recess the reading misses or places differently; a hollow body "
            "shows as a closed outline set in from the body's outline and following it.",
            "Thin walls show as pairs of close parallel lines along the outline, in more than one view.",
            "Dimension, extension and centre lines are thin and are not edges.",
            "The tint is a box and may cover parts of neighbouring features; their lines are marked where the built part "
            "has them.",
        ),
        cases=(("solid or hollow", "solid"),),
    ),
    "cavity wall": Mark(
        description=(
            "A program built the part as the reading describes it. On the drawing it marks one wall of one cavity, "
            "recess or void of that built part, as the single line that wall makes in this view: a translucent red band "
            "where the built part shows the wall's edge directly and the drawing should have a solid line, or a row of "
            "orange dots where the edge is hidden behind material and the drawing should have a dashed line. The "
            "enlarged area is also shown again without the mark."),
        conclusions=("wall drawn", "no wall"),
        confusers=(
            "A drawn line of the wall runs inside the band or along the dots, at most a stroke's width off, over most "
            "of the mark's length; a line running beside it with clear white between is a different edge.",
            "A thick outline right next to a row of dots can cover them; that outline is the part's outer face, not the "
            "wall, unless a separate line also runs along the dots.",
            "An outline drawn twice close together can come from a draft or taper, a slanted outline seen at both ends, "
            "or from a chamfer, instead of from a wall.",
            "Lines of holes, recesses and other features may cross the mark or run along a short piece of it; only a "
            "line following the mark over most of its length counts.",
            "A line that follows the mark over only part of its length shows a wall of another size, or another "
            "feature's edge.",
        ),
        cases=(("solid or hollow", "hollow"), ("open or closed", None)),
    ),
    "hole edge": Mark(
        description=(
            "A program built the part as the reading describes it. On the drawing it marks one edge of one hole, slot "
            "or pocket of that built part, as the single line that edge makes in this view: a translucent red band "
            "where the built part shows the edge directly and the drawing should have a solid line, or a row of orange "
            "dots where the edge is hidden behind material and the drawing should have a dashed line. A long side of a "
            "hole may be marked one half at a time. The enlarged area is also shown again without the mark."),
        conclusions=("edge drawn", "no edge"),
        confusers=(
            "A drawn line of the edge runs inside the band or along the dots, at most a stroke's width off, over most of "
            "the mark's length; a line running beside it with clear white between is a different edge.",
            "Lines of other holes, recesses and features may cross the mark or run along a short piece of it; only a "
            "line following the mark over most of its length counts.",
            "A line that follows the mark over only part of its length shows an edge of another length, or another "
            "feature's edge.",
            "A centre line runs along a hole's middle and is thin, with long and short dashes; it is not an edge.",
            "A thick outline right next to a row of dots can cover them; it is another edge unless a separate line also "
            "runs along the dots.",
        ),
        cases=(("through or blind", None),),
    ),
    "body extent": Mark(
        description=(
            "A program built the part as the reading describes it. On the drawing it marks the two outermost edges one "
            "body of that built part has across one direction in this view, one on each side: a translucent red band "
            "where the edge is seen directly and the drawing should have a solid line, or a row of orange dots where it "
            "is hidden and the drawing should have a dashed line. The enlarged area is also shown again without the "
            "mark."),
        conclusions=("as read", "drawn larger", "drawn smaller"),
        confusers=(
            "A drawn line of the same edge runs inside its band or along its dots, at most a stroke's width off; a line "
            "running beside the mark with clear white between is a different edge.",
            "Lines of neighbouring features, fillets, supports, flanges or bosses joined to the body, may run beyond the "
            "marks; judge the body's own outline running alongside the marks.",
            "A body may have more outlines than its outermost ones, steps or raised faces running parallel inside; when a "
            "drawn line fills the band or follows the dots, that is the outline the mark stands for.",
            "Chamfers and rounds make a drawn outline turn a corner with a short slanted or curved piece; that is not a "
            "size difference.",
            "A draft or taper makes a drawn outline slant slightly across a straight band: on it at one end, just beside "
            "it at the other.",
            "Dimension, extension and centre lines are thin and may run along a band; outlines are thick.",
        ),
        cases=(("thickness", None), ("how far", None)),
    ),
}


def conclusions_of(question: str) -> tuple[str, ...]:
    return MARKS[question].conclusions if question in MARKS else QUESTIONS[question][0]


class Node(BaseModel):
    id: str
    ask: str
    answers: dict[str, str] = Field(min_length=2)  # answer -> node id, or "=conclusion"


class Tree(BaseModel):
    question: str
    applies: str
    root: str
    nodes: list[Node] = Field(min_length=1, max_length=MAX_SIMPLE_NODES)

    @model_validator(mode="after")
    def _well_formed(self) -> "Tree":
        nodes = {node.id: node for node in self.nodes}
        conclusions = {*conclusions_of(self.question), "undecided"}
        if self.root not in nodes:
            raise ValueError(f"root {self.root!r} is not a node")
        for node in self.nodes:
            if node.id == HIDDEN and set(node.answers) != {"yes", "no"}:
                raise ValueError(f"node {HIDDEN!r} takes the answers yes and no")
            for target in node.answers.values():
                if target.startswith("=") and target[1:] not in conclusions:
                    raise ValueError(f"{node.id}: {target!r} is not one of {sorted(conclusions)}")
                if not target.startswith("=") and target not in nodes:
                    raise ValueError(f"{node.id}: {target!r} is not a node")
        # Every node reachable from the root, and no path loops back.
        seen, path = set(), set()

        def walk(node_id: str) -> None:
            if node_id in path:
                raise ValueError(f"the tree loops through {node_id!r}")
            if node_id in seen:
                return
            seen.add(node_id)
            path.add(node_id)
            for target in nodes[node_id].answers.values():
                if not target.startswith("="):
                    walk(target)
            path.discard(node_id)

        walk(self.root)
        if unreached := nodes.keys() - seen:
            raise ValueError(f"nodes {sorted(unreached)} cannot be reached from the root")
        return self

    def conclude(self, observations: dict[str, str], hidden_drawn: bool) -> str:
        """The conclusion the observations lead to; an answer off the tree is undecided."""
        nodes = {node.id: node for node in self.nodes}
        node = nodes[self.root]
        while True:
            answer = ("yes" if hidden_drawn else "no") if node.id == HIDDEN else observations.get(node.id)
            target = node.answers.get(answer)
            if target is None:
                return "undecided"
            if target.startswith("="):
                return target[1:]
            node = nodes[target]


def fold(part: str) -> int:
    return int(hashlib.sha256(part.encode()).hexdigest(), 16) % 2


def cases(judged: list[Judged], question: str, parts_fold: int, read: str | None = None) -> list[str]:
    """The first readings' mistakes on a question in one fold's parts, if the reading answered `read`."""
    found = []
    for row in judged:
        if row.judgement is None or not row.key.endswith("/first") or fold(row.key.split("/")[2]) != parts_fold:
            continue
        for error in row.judgement.errors:
            if error.decision.question == question and read in (None, error.decision.read):
                found.append(f"- Read: {error.reading} Truth: {error.truth} Decided by: {error.cue}")
    return found


def _ask_tree(model, text: str, question: str, max_nodes: int) -> Tree:
    messages = [HumanMessage(content=text)]
    for _ in range(TRIES):
        answer = model.invoke(messages)
        raw = answer.text
        try:
            tree = Tree.model_validate({**json.loads(raw[raw.index("{"):raw.rindex("}") + 1]), "question": question})
            if len(tree.nodes) > max_nodes:
                raise ValueError(f"at most {max_nodes} nodes")
            return tree
        except ValueError as error:
            messages += [answer, HumanMessage(content=f"Fix your answer and give the whole JSON again: {str(error)[:800]}")]
    raise RuntimeError(f"no valid tree for {question}")


def write_tree(model, question: str, found: list[str]) -> Tree:
    answers = QUESTIONS[question][0]
    text = Template(PROMPT).substitute(question=question, meaning=QUESTIONS[question][1], answers=", ".join(answers),
                                       first=answers[0], cases="\n".join(found) or "(none)")
    return _ask_tree(model, text, question, MAX_NODES)


def write_grounded(model, mark: str, found: list[str]) -> Tree:
    spec = MARKS[mark]
    text = Template(GROUNDED).substitute(
        mark=spec.description, conclusions=", ".join(spec.conclusions),
        confusers="\n".join(f"- {line}" for line in spec.confusers), cases="\n".join(found) or "(none)",
        max_nodes=MAX_GROUNDED_NODES)
    return _ask_tree(model, text, mark, MAX_GROUNDED_NODES)


def _write_grounded_all(judged_path: Path, out: Path, only: list[str] | None = None) -> None:
    """Every fold's tree for every mark, or rewrite only the named marks' trees in `out`."""
    judged = list(read_rows(judged_path, Judged))
    model = sol()
    jobs = [(parts_fold, mark) for parts_fold in (0, 1) for mark in (only or MARKS)]

    def work(job: tuple[int, str]) -> Tree:
        found = [line for q, read in MARKS[job[1]].cases for line in cases(judged, q, job[0], read)]
        tree = write_grounded(model, job[1], found)
        print(f"fold {job[0]} {job[1]}: {len(found)} cases, {len(tree.nodes)} nodes", flush=True)
        return tree

    with ThreadPoolExecutor(len(jobs)) as pool:
        written = list(pool.map(work, jobs))
    trees: dict[str, dict] = json.loads(out.read_text()) if only and out.exists() else {}
    for (parts_fold, mark), tree in zip(jobs, written, strict=True):
        trees.setdefault(str(parts_fold), {})[mark] = tree.model_dump()
    out.write_text(json.dumps(trees, indent=1))


def simplify(model, tree: Tree) -> Tree:
    text = Template(SIMPLIFY).substitute(question=tree.question, answers=", ".join(QUESTIONS[tree.question][0]),
                                         tree=tree.model_dump_json(exclude={"question"}, indent=1), max_nodes=MAX_SIMPLE_NODES)
    return _ask_tree(model, text, tree.question, MAX_SIMPLE_NODES)


def _simplify_all(source: Path, out: Path) -> None:
    trees = json.loads(source.read_text())
    model = sol()
    jobs = [(parts_fold, q) for parts_fold, by_question in trees.items() for q in by_question]
    with ThreadPoolExecutor(len(jobs)) as pool:
        rewritten = list(pool.map(lambda job: simplify(model, Tree.model_validate(trees[job[0]][job[1]])), jobs))
    for (parts_fold, q), tree in zip(jobs, rewritten, strict=True):
        trees[parts_fold][q] = tree.model_dump()
        print(f"fold {parts_fold} {q}: {len(tree.nodes)} nodes", flush=True)
    out.write_text(json.dumps(trees, indent=1))


if __name__ == "__main__" and sys.argv[1] == "simplify":
    _simplify_all(Path(sys.argv[2]), Path(sys.argv[3]))
elif __name__ == "__main__" and sys.argv[1] == "merge":
    merged: dict[str, dict] = {}
    for source in sys.argv[3:]:
        for parts_fold, by_mark in json.loads(Path(source).read_text()).items():
            merged.setdefault(parts_fold, {}).update(by_mark)
    Path(sys.argv[2]).write_text(json.dumps(merged, indent=1))
    print({parts_fold: sorted(by_mark) for parts_fold, by_mark in merged.items()})
elif __name__ == "__main__" and sys.argv[1] == "grounded":
    _write_grounded_all(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:] or None)
elif __name__ == "__main__":
    judged = list(read_rows(sys.argv[1], Judged))
    model = sol()
    jobs = [(parts_fold, question) for parts_fold in (0, 1) for question in TREE_QUESTIONS]

    def work(job: tuple[int, str]) -> Tree:
        found = cases(judged, job[1], job[0])
        tree = write_tree(model, job[1], found)
        print(f"fold {job[0]} {job[1]}: {len(found)} cases, {len(tree.nodes)} nodes", flush=True)
        return tree

    with ThreadPoolExecutor(len(jobs)) as pool:
        written = list(pool.map(work, jobs))
    trees: dict[str, dict] = {}
    for (parts_fold, question), tree in zip(jobs, written, strict=True):
        trees.setdefault(str(parts_fold), {})[question] = tree.model_dump()
    Path(sys.argv[2]).write_text(json.dumps(trees, indent=1))
