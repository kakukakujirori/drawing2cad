"""Every vocabulary in the contract, checked against the corpus it came from.

The enums were not designed and then justified; they are a census. These tests
re-run the census, so a vocabulary that stops covering the data fails here
rather than in a run. They also fail when the corpus grows a shape nobody has
described yet, which is the more useful direction.

The mapping runs one way only: the contract must cover what the corpus holds.
A member the corpus never exercises is allowed -- `inch` units and `assumed`
claims are real answers this dataset happens never to need.
"""

import collections
import json
from pathlib import Path

import ezdxf
import pytest
from ezdxf.tools import standards

from zeroshot.pipeline.messages.contracts import (
    _EXCLUDED_GEOMETRY,
    VIEW_FRAME,
    DrawnEntity,
    EdgeStyle,
    GeometryKind,
    View,
    edge_style_for_linetype,
)

_CORPUS = Path(__file__).parents[3] / "data" / "test_vlm"
_DRAWINGS = sorted((_CORPUS / "techdraw" / "dxf").glob("*.dxf"))
_TARGETS = sorted((_CORPUS / "target_step").glob("*.step"))

pytestmark = pytest.mark.skipif(
    not _DRAWINGS or not _TARGETS,
    reason="the sample corpus is not checked out",
)

# `INSERT` is the one entity that carries no geometry: it places the
# centre-mark annotation blocks SolidWorks writes.
_ANNOTATION_ENTITIES = {"INSERT"}

# Entities inherit `BYLAYER`, which is not a visibility. Only the two real
# linetypes are a claim about whether an edge is seen.
_INHERITED_LINETYPE = "BYLAYER"


def _drawn() -> tuple[collections.Counter, collections.Counter]:
    entities: collections.Counter = collections.Counter()
    linetypes: collections.Counter = collections.Counter()
    for path in _DRAWINGS:
        for entity in ezdxf.readfile(path).modelspace():
            if entity.dxftype() in _ANNOTATION_ENTITIES:
                continue
            entities[entity.dxftype()] += 1
            linetypes[entity.dxf.linetype] += 1
    return entities, linetypes


def _built() -> tuple[collections.Counter, collections.Counter]:
    from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_CurveType, GeomAbs_SurfaceType
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    def names(enum: object) -> dict[int, str]:
        return {
            getattr(enum, name): name.removeprefix("GeomAbs_")
            for name in dir(enum)
            if name.startswith("GeomAbs_")
        }

    surface_names, curve_names = names(GeomAbs_SurfaceType), names(GeomAbs_CurveType)
    surfaces: collections.Counter = collections.Counter()
    curves: collections.Counter = collections.Counter()
    for path in _TARGETS:
        reader = STEPControl_Reader()
        reader.ReadFile(str(path))
        reader.TransferRoots()
        shape = reader.OneShape()
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face = BRepAdaptor_Surface(TopoDS.Face(explorer.Current()))
            surfaces[surface_names[face.GetType()]] += 1
            explorer.Next()
        explorer = TopExp_Explorer(shape, TopAbs_EDGE)
        while explorer.More():
            edge = BRepAdaptor_Curve(TopoDS.Edge(explorer.Current()))
            curves[curve_names[edge.GetType()]] += 1
            explorer.Next()
    return surfaces, curves


def test_the_drawn_entity_vocabulary_covers_every_entity_in_the_drawings() -> None:
    entities, _ = _drawn()
    drawn = {member.value for member in DrawnEntity}
    uncovered = {
        name for name in entities if name.lower().removeprefix("lw") not in drawn
    }
    assert not uncovered, (
        f"the drawings hold entities the contract cannot name: {uncovered}"
    )


def test_every_linetype_ezdxf_defines_has_a_meaning() -> None:
    """The authority for what a linetype may be is the format, not this corpus:
    `linetype` names a row in the file's own LTYPE table, so the input side is
    open. What can be enumerated is the set ezdxf ships, and every one of those
    has to resolve to a drafting meaning rather than falling through."""
    shipped = [name for name, _description, _pattern in standards.linetypes()]
    assert shipped, "ezdxf defines no standard linetypes"

    unmapped = {
        name for name in shipped if edge_style_for_linetype(name) is EdgeStyle.OTHER
    }
    assert not unmapped, f"ezdxf linetypes with no meaning in the contract: {unmapped}"


def test_the_scale_suffixes_carry_the_meaning_of_their_base_pattern() -> None:
    """`CENTER`, `CENTER2` and `CENTERX2` are one pattern at three scales."""
    for base in ("CENTER", "DASHED", "PHANTOM", "DIVIDE", "DASHDOT", "DOT"):
        meanings = {
            edge_style_for_linetype(f"{base}{suffix}") for suffix in ("", "2", "X2")
        }
        assert len(meanings) == 1, f"{base} splits across {meanings}"
        assert meanings != {EdgeStyle.OTHER}, base


def test_an_inherited_linetype_is_not_mistaken_for_a_visible_edge() -> None:
    """`BYLAYER` is an instruction to look somewhere else, not a pattern.
    Reading it as a visible edge would invent linework that is not there."""
    for inherited in ("BYLAYER", "BYBLOCK"):
        assert edge_style_for_linetype(inherited) is EdgeStyle.OTHER


def test_every_linetype_in_the_drawings_resolves_to_a_meaning() -> None:
    _, linetypes = _drawn()
    unresolved = {
        name
        for name in linetypes
        if name != _INHERITED_LINETYPE
        and edge_style_for_linetype(name) is EdgeStyle.OTHER
    }
    assert not unresolved, (
        f"the drawings hold linetypes the contract cannot read: {unresolved}"
    )


def _claimable() -> set[str]:
    return {member.value.replace("_", "") for member in GeometryKind}


def test_every_occ_geometry_type_is_either_named_or_excluded() -> None:
    """The set of geometry a B-rep can hold is closed, so this is checkable
    without any corpus at all: every member of OCC's two enums must be a
    decision, not an oversight. Twenty files were not enough to make that
    decision -- `ellipse` was dropped on their evidence and the wider census
    found 34,712 of them -- so the enums are the authority and the census only
    informs which ones to carry."""
    from OCP.GeomAbs import GeomAbs_CurveType, GeomAbs_SurfaceType

    universe = {
        name.removeprefix("GeomAbs_")
        for enum in (GeomAbs_SurfaceType, GeomAbs_CurveType)
        for name in dir(enum)
        if name.startswith("GeomAbs_")
    }
    undecided = {
        name
        for name in universe
        if name.lower() not in _claimable() and name not in _EXCLUDED_GEOMETRY
    }
    assert not undecided, (
        f"OCC types the contract neither names nor excludes: {undecided}"
    )

    stale = {name for name in _EXCLUDED_GEOMETRY if name not in universe}
    assert not stale, f"excluded types OCC no longer has: {stale}"


def test_the_census_holds_no_type_the_contract_neither_names_nor_excludes() -> None:
    """`geometry_census.json` is the evidence behind those decisions, counted
    over 1000 ABC models. A type appearing there that is neither named nor
    excluded means the decision was never made for it."""
    census = json.loads((Path(__file__).parent / "geometry_census.json").read_text())
    for corpus, data in census["corpora"].items():
        seen = {
            name
            for group in ("surfaces", "curves")
            for name, count in data[group].items()
            if count
        }
        undecided = {
            name
            for name in seen
            if name.lower() not in _claimable() and name not in _EXCLUDED_GEOMETRY
        }
        assert not undecided, f"{corpus} holds undecided geometry: {undecided}"


def test_the_local_targets_hold_no_geometry_the_contract_cannot_name() -> None:
    """The corpus this pipeline is scored on, as a narrower second check."""
    surfaces, curves = _built()
    uncovered = {
        name
        for name in list(surfaces) + list(curves)
        if name.lower() not in _claimable() and name not in _EXCLUDED_GEOMETRY
    }
    assert not uncovered, (
        f"targets hold geometry the contract cannot claim: {uncovered}"
    )


def test_every_view_the_contract_names_has_a_frame() -> None:
    assert set(VIEW_FRAME) == set(View)
    axes = {axis for frame in VIEW_FRAME.values() for axis in frame}
    assert axes <= {"+x", "-x", "+y", "-y", "+z", "-z"}
    for view, frame in VIEW_FRAME.items():
        assert len({axis.lstrip("+-") for axis in frame}) == 3, view
