from __future__ import annotations

import shutil
from pathlib import Path

import ezdxf
import pytest
from ezdxf.lldxf.tagwriter import TagCollector

from src.data.dxf import DXFPrimitiveConfig, DXFPrimitiveParser, VIEW_DIRECTIONS
from src.data.render.reformat_techdraw import UNSTAMPED_LAYER, stamp_dxf

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "unstamped_sample.dxf"


def _tags(entity) -> list[tuple[int, object]]:
    """Full DXF tag list for one entity, excluding layer (8) and linetype (6)
    -- the only two fields the stamper is allowed to change."""
    collector = TagCollector()
    entity.export_dxf(collector)
    return [(code, value) for code, value in collector.tags if code not in (6, 8)]


@pytest.fixture()
def stamped_dxf(tmp_path):
    target = tmp_path / FIXTURE.name
    shutil.copy(FIXTURE, target)

    before_doc = ezdxf.readfile(target)
    before = {
        entity.dxf.handle: (entity.dxftype(), entity.dxf.layer, _tags(entity))
        for entity in before_doc.modelspace()
    }

    config = DXFPrimitiveConfig(included_layers=(UNSTAMPED_LAYER,))
    counts = stamp_dxf(target, config)
    return target, before, counts


def test_stamp_preserves_geometry_and_handles(stamped_dxf) -> None:
    target, before, _counts = stamped_dxf
    after = {
        entity.dxf.handle: entity for entity in ezdxf.readfile(target).modelspace()
    }

    # Same set of entities: the stamper mutates in place, it never deletes,
    # adds, or recreates (recreation would assign fresh handles).
    assert set(after) == set(before)
    for handle, (dxftype, _layer_before, tags_before) in before.items():
        entity = after[handle]
        assert entity.dxftype() == dxftype
        assert _tags(entity) == tags_before, f"geometry changed for handle {handle}"


def test_stamp_assigns_view_layers_and_parses(stamped_dxf) -> None:
    target, before, counts = stamped_dxf

    # Sanity: this fixture is rich enough to actually split.
    assert sum(counts[view] for view in VIEW_DIRECTIONS) > 0

    parser = DXFPrimitiveParser()
    parsed = parser.parse(target)
    assert parsed.num_primitives > 0
    assert set(parsed.view_direction_ids.tolist()) == {0, 1, 2}


def test_stamp_leaves_center_marks_and_other_layers_untouched(stamped_dxf) -> None:
    target, before, _counts = stamped_dxf
    after = {
        entity.dxf.handle: entity for entity in ezdxf.readfile(target).modelspace()
    }

    for handle, (_dxftype, layer_before, _tags_before) in before.items():
        if layer_before != UNSTAMPED_LAYER:
            assert after[handle].dxf.layer == layer_before


def test_stamp_only_moves_entities_onto_known_view_layers(stamped_dxf) -> None:
    target, before, _counts = stamped_dxf
    after = {
        entity.dxf.handle: entity for entity in ezdxf.readfile(target).modelspace()
    }

    allowed = set(VIEW_DIRECTIONS) | {UNSTAMPED_LAYER}
    for handle, (_dxftype, layer_before, _tags_before) in before.items():
        if layer_before == UNSTAMPED_LAYER:
            assert after[handle].dxf.layer in allowed
