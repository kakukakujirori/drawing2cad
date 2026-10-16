import numpy as np
import pytest
from PIL import Image, ImageDraw

from zeroshot.pipeline.retrieval import index
from zeroshot.pipeline.retrieval.index import (
    ROTATIONS,
    RegionIndex,
    crop_region,
    standard_turns,
    turn,
)


def _fake_embed(images):
    pixels = np.stack(
        [
            np.asarray(i.convert("L").resize((16, 16)), np.float32).ravel()
            for i in images
        ]
    )
    pixels = 255 - pixels + 1
    return pixels / np.linalg.norm(pixels, axis=1, keepdims=True)


def _drawn(shape):
    image = Image.new("L", (64, 64), 255)
    draw = ImageDraw.Draw(image)
    if shape == "circle":
        draw.ellipse((16, 16, 48, 48), outline=0, width=3)
    elif shape == "bar":
        draw.rectangle((8, 28, 56, 36), fill=0)
    elif shape == "ell":
        draw.line((12, 8, 12, 52, 40, 52), fill=0, width=5)
    else:
        draw.line((8, 8, 56, 56), fill=0, width=3)
    return image


def test_search_ranks_by_every_view(tmp_path, monkeypatch):
    monkeypatch.setattr(index, "embed", _fake_embed)
    store = RegionIndex(tmp_path / "regions.sqlite")
    store.add(
        "p1",
        "gt",
        "boss",
        "a boss",
        [("front", _drawn("circle")), ("top", _drawn("bar"))],
    )
    store.add(
        "p2",
        "gt",
        "hole",
        "a hole",
        [("front", _drawn("circle")), ("top", _drawn("slash"))],
    )

    query = [("front", _drawn("circle")), ("top", _drawn("bar"))]
    first, second = store.search(query, k=2)
    assert (first.name, second.name) == ("boss", "hole")
    assert first.score > second.score
    assert [role for role, _ in first.crops] == ["front", "top"]


def test_crop_region_keeps_context_inside_the_page():
    page = Image.new("L", (100, 50), 255)
    assert crop_region(page, (40, 10, 60, 20)).size == (38, 28)
    assert crop_region(page, (0, 0, 10, 10)).size == (16, 16)


def test_a_missing_view_ranks_between_a_match_and_a_contradiction(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(index, "embed", _fake_embed)
    store = RegionIndex(tmp_path / "regions.sqlite")
    circle = ("front", _drawn("circle"))
    store.add("p1", "gt", "boss", "", [circle, ("top", _drawn("bar"))])
    store.add("p2", "gt", "hole", "", [circle, ("top", _drawn("slash"))])
    store.add("p3", "glm", "front only", "", [circle])

    ranked = store.search([circle, ("top", _drawn("bar"))], k=3)
    assert [r.name for r in ranked] == ["boss", "front only", "hole"]


def test_an_empty_index_finds_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(index, "embed", _fake_embed)
    assert (
        RegionIndex(tmp_path / "empty.sqlite").search([("front", _drawn("circle"))])
        == []
    )


def test_an_index_built_with_another_encoder_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(index, "embed", _fake_embed)
    monkeypatch.setattr(index, "ENCODER", "old-encoder")
    RegionIndex(tmp_path / "old.sqlite").add(
        "p0", "glm", "boss", "a boss", [("front", _drawn("circle"))]
    )
    monkeypatch.undo()
    with pytest.raises(ValueError, match="built with old-encoder"):
        RegionIndex(tmp_path / "old.sqlite")


def test_every_cube_rotation_moves_the_views_differently():
    assert len(ROTATIONS) == 24
    assert len({tuple(sorted(moves.items())) for moves in ROTATIONS}) == 24
    assert {role: (role, 0) for role in ROTATIONS[0]} in ROTATIONS


def test_a_part_drawn_in_another_orientation_is_found(tmp_path, monkeypatch):
    monkeypatch.setattr(index, "embed", _fake_embed)
    store = RegionIndex(tmp_path / "regions.sqlite")
    stored = {"front": _drawn("ell"), "top": _drawn("bar")}
    store.add("p1", "gt", "boss", "a boss", list(stored.items()))
    # The same part turned so that its front is drawn in the top view.
    moves = next(m for m in ROTATIONS if m["top"][0] == "front" and m["top"][1])
    query = [
        (role, turn(stored[target], -turns))
        for role, (target, turns) in moves.items()
        if target in stored
    ]

    (found,) = store.search(query, k=1)
    assert found.score > 0.999
    for (role, image), (match_role, match) in zip(query, found.crops, strict=True):
        assert match_role == role and match.tobytes() == image.tobytes()


def test_a_view_turned_on_the_page_is_turned_back_to_its_standard_frame():
    assert standard_turns("top", "+x", "+y") == 0
    # Rightwards is -y, which points down in the standard top view.
    assert standard_turns("top", "-y", "+x") == 3
