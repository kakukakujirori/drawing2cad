import json

import pytest
from PIL import Image, ImageDraw
from pydantic import ValidationError

from zeroshot.pipeline.retrieval import index
from zeroshot.pipeline.retrieval.index import RegionIndex, crop_region
from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.tools.errors import ToolFeedbackError
from zeroshot.pipeline.tools.search_semantic_features import (
    create_search_semantic_features_tool,
)
from zeroshot.pipeline.verification.render.orthographic import STANDARD_VIEW_FRAMES

FRONT, TOP, SLASH = [20, 20, 60, 60], [20, 100, 60, 130], [120, 20, 180, 80]


def _fake_embed(images):
    import numpy as np

    pixels = np.stack(
        [256 - np.asarray(i.resize((16, 16)), np.float32).ravel() for i in images]
    )
    return pixels / np.linalg.norm(pixels, axis=1, keepdims=True)


@pytest.fixture
def sheet(tmp_path, monkeypatch):
    monkeypatch.setattr(index, "embed", _fake_embed)
    page = Image.new("L", (200, 160), 255)
    draw = ImageDraw.Draw(page)
    draw.ellipse(FRONT, outline=0, width=3)
    draw.rectangle(TOP, fill=0)
    draw.line(SLASH, fill=0, width=3)
    (tmp_path / "inputs").mkdir()
    page.save(tmp_path / "inputs/sheet.png")
    store = RegionIndex(tmp_path / "regions.sqlite")
    crops = [("front", crop_region(page, FRONT)), ("top", crop_region(page, TOP))]
    store.add("000405", "glm", "sem_boss", "A boss rising from the floor.", crops)
    return tmp_path


def _items(boxes):
    return [
        {
            "image_path": "/work/inputs/sheet.png",
            "box_px": box,
            "role": role,
            "u_axis": STANDARD_VIEW_FRAMES[role][0],
            "v_axis": STANDARD_VIEW_FRAMES[role][1],
        }
        for role, box in boxes
    ]


def test_a_close_match_comes_with_its_description_and_a_saved_sheet(sheet):
    (text,) = create_search_semantic_features_tool(
        SandboxWorkdir(host_bind_dir=sheet), sheet / "regions.sqlite"
    ).invoke({"items": _items([("front", FRONT), ("top", TOP)])})
    assert text["text"] == (
        "A. A boss rising from the floor.\nCrops: /work/semantic_search/001/A.png"
    )
    assert Image.open(sheet / "semantic_search/001/A.png").size == (1024, 1024)
    record = json.loads((sheet / "semantic_search/001.json").read_text())
    assert record["matches"][0]["shown_as"] == "A" and len(record["items"]) == 2


def test_a_region_drawn_unlike_any_feature_gets_no_match(sheet):
    (text,) = create_search_semantic_features_tool(
        SandboxWorkdir(host_bind_dir=sheet), sheet / "regions.sqlite"
    ).invoke({"items": _items([("front", SLASH)])})
    assert text["text"].startswith("No semantic feature in the database matches")
    # Unshown matches are still recorded, for analysis.
    record = json.loads((sheet / "semantic_search/001.json").read_text())
    assert record["matches"][0]["shown_as"] is None


def test_a_box_outside_the_image_is_refused(sheet):
    tool = create_search_semantic_features_tool(
        SandboxWorkdir(host_bind_dir=sheet), sheet / "regions.sqlite"
    )
    with pytest.raises(ToolFeedbackError, match="box_px"):
        tool.invoke({"items": _items([("front", [150, 20, 260, 60])])})


def test_a_view_given_twice_is_refused(sheet):
    tool = create_search_semantic_features_tool(
        SandboxWorkdir(host_bind_dir=sheet), sheet / "regions.sqlite"
    )
    with pytest.raises(ValidationError, match="one box per view"):
        tool.invoke({"items": _items([("front", FRONT), ("front", TOP)])})


def test_axes_that_do_not_fit_the_view_are_refused(sheet):
    tool = create_search_semantic_features_tool(
        SandboxWorkdir(host_bind_dir=sheet), sheet / "regions.sqlite"
    )
    (item,) = _items([("front", FRONT)])
    with pytest.raises(ValidationError, match="axis pointing at the viewer"):
        tool.invoke({"items": [{**item, "u_axis": "+y"}]})
