import json
import os

from experiments.rag.build_db import mistakes
from experiments.rag.judge import problems
from experiments.rag.pairs import run_pairs
from experiments.rag.schema import Call, Comparison, Component, Error, Judged, Judgement, Pair


def _save(path, text, when):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    os.utime(path, (when, when))


def test_each_build_pairs_with_the_reading_in_force_when_it_was_built(tmp_path) -> None:
    run = tmp_path / "train_eccv" / "000001"
    attempts = run / "workspace/attempts/round_000"
    _save(attempts / "interpretation/000/interpretation.json", json.dumps({"features": []}), 10)
    _save(attempts / "interpretation/001/interpretation.json", json.dumps({"features": [{"name": "a"}]}), 20)
    _save(attempts / "coding/000/output.step", "", 25)
    _save(attempts / "coding/001/output.step", "", 30)
    _save(attempts / "interpretation/002/interpretation.json", json.dumps({"features": [{"name": "b"}]}), 40)
    _save(attempts / "coding/002/output.step", "", 50)
    _save(run / ".hydra/config.yaml", "model: {model: m}\n", 0)
    _save(run / "score.json", json.dumps({"pred_step": str(attempts / "coding/002/output.step"), "target_step": "gt.step"}), 60)
    pairs = {pair.which: pair for pair in run_pairs(run)}
    assert pairs["first"].interpretation.endswith("001/interpretation.json")
    assert pairs["first"].build.endswith("coding/001/output.step")
    assert pairs["final"].interpretation.endswith("002/interpretation.json")


def test_a_first_reading_error_persists_only_if_the_final_reading_keeps_it(tmp_path) -> None:
    reading = tmp_path / "interpretation.json"
    reading.write_text(json.dumps({"features": [{"name": "a", "description": "da"}, {"name": "b", "description": "db"}]}))
    pairs = {f"m/d/p/{which}": Pair(key=f"m/d/p/{which}", model="m", dataset="d", part="p", which=which, drawing="",
                                     gt="", interpretation=str(reading), build="") for which in ("first", "final")}

    def judged(which, features):
        errors = [Error(feature=f, decision={"question": "other"}, reading="r", truth="t", cue="c", components=["M1"]) for f in features]
        return Judged(key=f"m/d/p/{which}", judge="j", judgement=Judgement(components=[], errors=errors))

    comparisons = {key: Comparison(key=key, iou=0.5, rotation_index=0, images={}, components=[Component(id="M1", share=0.1)])
                   for key in pairs}
    rows = mistakes(pairs, comparisons, [judged("first", ["a", "b"]), judged("final", ["a"])])
    assert {(row.which, row.feature, row.persisted, row.share) for row in rows} == {
        ("first", "a", True, 0.1), ("first", "b", False, 0.1), ("final", "a", True, 0.1)}


def test_an_answer_must_call_every_piece_and_explain_reading_pieces() -> None:
    comparison = Comparison(key="k", iou=0.5, rotation_index=0, images={},
                            components=[Component(id="M1", share=0.1), Component(id="E1", share=0.2)])
    answer = Judgement(components=[Call(id="M1", features=["a"], cause="reading")], errors=[])
    found = problems(answer, comparison, {"a"})
    assert any("E1" in problem for problem in found) and any("no error" in problem for problem in found)


def test_keys_lose_the_frame_and_keep_the_kind() -> None:
    from experiments.rag.relevance import kinds, stripped

    text = "A R0.040 rounded slot at x=0.36 along +Y in the XZ plane, with a counterbore"
    assert stripped(text) == "A rounded slot at x= along in the plane, with a counterbore"
    assert kinds("sem_u_slot " + text) == {"slot", "round", "counterbore"}


def test_a_skin_from_nearly_coinciding_surfaces_is_no_piece() -> None:
    import numpy as np

    from experiments.rag.geometry import _one_grid, _pieces

    part = np.ones((20, 20, 20), bool)
    made = np.ones((19, 20, 20), bool)  # one voxel short on one side: a skin
    made[5:10, 5:10, 5:10] = False  # and a real hole
    part, made, start = _one_grid(part, np.array([0, 0, 0]), made, np.array([1, 0, 0]))
    assert list(start) == [0, 0, 0]
    pieces = _pieces(part & ~made, made, int(part.sum()), 1, "M")
    assert [(name, round(share, 4)) for name, share, _ in pieces] == [("M1", round(125 / 8000, 4))]


def test_a_missing_wall_thinner_than_the_margin_is_still_a_piece() -> None:
    import numpy as np

    from experiments.rag.geometry import _pieces

    part = np.zeros((30, 30, 30), bool)
    part[:, :, :2] = True  # a two-voxel plate
    part[:, :2, :] = True  # and a two-voxel wall standing on it
    made = part.copy()
    made[:, :2, 2:] = False  # the build lacks the wall
    pieces = _pieces(part & ~made, made, int(part.sum()), 3, "M")
    assert len(pieces) == 1 and pieces[0][2].sum() > 0.8 * (part & ~made).sum()


def test_voxels_fill_a_solid_and_keep_its_hole_empty() -> None:
    import manifold3d

    from experiments.rag.geometry import _occupancy

    plate = manifold3d.Manifold.cube((1.0, 1.0, 0.2)) - manifold3d.Manifold.cylinder(0.4, 0.2).translate((0.5, 0.5, -0.1))
    grid, first = _occupancy(plate, 0.05)
    assert grid.sum() * 0.05**3 == __import__("pytest").approx(plate.volume(), rel=0.15)
    assert not grid[10, 10].any() and grid[1, 1].any()


def test_a_checklist_shows_the_questions_most_often_wrong_on_the_same_kind() -> None:
    from experiments.rag.relevance import checklist
    from zeroshot.pipeline.retrieval.mistakes import Mistake

    def row(part, feature, question):
        answers = {"through or blind": ("through", "blind"), "edge": ("sharp", "round")}.get(question, (None, None))
        return Mistake(part=part, dataset="d", model="m", which="final", persisted=True, iou=0.5, feature=feature,
                       description="", decision={"question": question, "read": answers[0], "truth": answers[1]},
                       reading="", truth="", cue="")

    db = [row("q", "sem_bore", "edge"), row("a", "sem_hole", "through or blind"), row("b", "sem_holes", "through or blind"),
          row("c", "sem_plate", "edge"), row("d", "sem_plate", "edge"), row("e", "sem_plate", "edge"), row("f", "sem_hole", "other")]
    assert checklist(db, 0, [1, 2, 3, 4, 5, 6], 1, by_kind=False) == ["edge"]
    assert checklist(db, 0, [1, 2, 3, 4, 5, 6], 1, by_kind=True) == ["through or blind"]


def test_a_feature_raises_the_questions_of_the_kind_its_name_gives_or_else_its_description() -> None:
    from experiments.rag.reread import items

    reading = {"features": [
        {"name": "sem_main_shell", "description": "A housing with five holes"},
        {"name": "sem_mount_holes", "description": "Holes into the plate"},
        {"name": "sem_front_window", "description": "A rectangular opening"},
        {"name": "sem_base_plate", "description": "A flat plate"},
    ]}
    assert [(item["feature"], item["questions"]) for item in items(reading)] == [
        ("sem_mount_holes", ["through or blind", "edge"]), ("sem_front_window", ["through or blind", "edge"])]


def test_a_tree_follows_observations_and_the_drawing_settings_to_a_conclusion() -> None:
    import pytest

    from experiments.rag.trees import Tree

    tree = Tree(question="through or blind", applies="a hole", root="hidden", nodes=[
        {"id": "hidden", "ask": "Are hidden lines drawn?", "answers": {"yes": "n1", "no": "=undecided"}},
        {"id": "n1", "ask": "Do the dashed side lines reach the far outline?",
         "answers": {"yes": "=through", "no": "=blind", "cannot tell": "=undecided"}},
    ])
    assert tree.conclude({"n1": "no"}, hidden_drawn=True) == "blind"
    assert tree.conclude({"n1": "no"}, hidden_drawn=False) == "undecided"
    assert tree.conclude({"n1": "maybe"}, hidden_drawn=True) == "undecided"
    for bad in ({"yes": "=solid", "no": "=blind"}, {"yes": "n9", "no": "=blind"}, {"yes": "n1", "no": "=blind"}):
        with pytest.raises(ValueError):
            Tree(question="through or blind", applies="a hole", root="n1",
                 nodes=[{"id": "n1", "ask": "?", "answers": bad}])


def test_a_counterbore_and_its_bore_are_one_hole_through_the_bodies_around_it() -> None:
    import numpy as np

    from experiments.rag.marks import Body, Hole, material_along, stepped

    def hole(feature, width, claim, x=10.0):
        return Hole(feature=feature, center=(x, 0, 5), axis=(0, 1, 0), across=[((1, 0, 0), width), ((0, 0, 1), width)],
                    entry=0, end=10, claim=claim)

    merged = stepped([hole("sem_cbore", 8, "blind"), hole("sem_bore", 4, "through"), hole("sem_other", 4, "blind", x=30)])
    assert [(h.feature, h.claim, h.across[0][1]) for h in merged] == [
        ("sem_bore + sem_cbore", "through", 4), ("sem_other", "blind", 4)]
    plate = Body(feature="sem_plate", low=(0, 0, 0), high=(40, 3, 10), solid=True)
    support = Body(feature="sem_support", low=(25, 0, 0), high=(40, 20, 10), solid=True)
    lo, hi = material_along(merged[0], [plate, support], np.zeros(3), np.array([40.0, 20, 10]))
    assert (lo[1], hi[1]) == (0, 3)




def test_a_hole_is_doubted_when_two_of_its_edges_or_all_of_them_are_bare() -> None:
    from experiments.rag.grounded import hole_doubt

    assert hole_doubt(["no edge"]) and hole_doubt(["no edge", "no edge", "edge drawn", "edge drawn"])
    assert not hole_doubt(["no edge", "edge drawn"]) and not hole_doubt(["undecided", "edge drawn"])
