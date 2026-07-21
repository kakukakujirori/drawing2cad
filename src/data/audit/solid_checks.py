"""Rule-based B-rep sanity checks for CAD ground-truth solids.

Targets concrete trust concerns about a synthetic (Zero-To-CAD-1m-derived)
training corpus: zero/near-zero-volume material, micro edges/faces relative to
both kernel tolerance and part scale, self-intersection, thin walls, solids
fragmented into disjoint islands, and kernel tolerance blow-up (a proxy for
"the boolean kernel struggled here"). Operation-contribution tracing (did a
chained CadQuery call actually change the shape) lives in ``op_trace.py``
since it needs to execute source code, not just inspect a finished shape.

Every check takes a ``cadquery.Shape`` and returns a small immutable result.
``audit_shape`` runs the full battery and folds the results into a severity
verdict. ``gt_audit.py`` is the multiprocessing CLI that runs this over a
staged Zero-To-CAD directory of ``{uuid}.step`` / ``{uuid}.cadquery.py`` pairs.

Two things are deliberately NOT attempted here (see gt_audit.py docstring for
the full rationale):
  - Analytic/exhaustive local-thickness fields. ``sample_wall_thickness``
    below is a sampled Monte-Carlo ray-cast proxy, not a guarantee.
  - Thin-neck "kissing" contact WITHIN a single solid (two lobes joined by a
    genuinely non-degenerate but perilously thin bridge). Two solids touching
    at a degenerate (zero-area) edge or vertex do NOT stay one manifold
    TopoDS_SOLID after a boolean op -- OCC itself splits them into separate
    solids (verified empirically), so ``connected_components`` below already
    catches that case. A real thin *neck* is a non-degenerate connection and
    is only caught probabilistically by the thickness sampler.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import cadquery as cq
from OCP.BOPAlgo import BOPAlgo_ArgumentAnalyzer, BOPAlgo_CheckStatus
from OCP.BRep import BRep_Tool
from OCP.BRepCheck import BRepCheck_Analyzer, BRepCheck_Status
from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds
from OCP.TopAbs import (
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_SHELL,
    TopAbs_SOLID,
    TopAbs_VERTEX,
    TopAbs_WIRE,
)
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape


# OCCT's Precision::Confusion(): the tolerance freshly-built vertices/edges/
# faces carry. Empirically confirmed (BRep_Tool.Tolerance_s on a fresh box).
KERNEL_BASE_TOLERANCE_MM = 1e-7


class Severity(str, Enum):
    OK = "ok"
    SOFT_SUSPECT = "soft_suspect"
    HARD_INVALID = "hard_invalid"
    UNAUDITABLE = "unauditable"


@dataclass(frozen=True)
class Thresholds:
    """All cutoffs in one place, overridable from the CLI.

    See the module docstring for why several checks report BOTH a
    kernel-absolute and a part-relative reading rather than picking one.
    """

    # Absolute: matches src/evaluation/executor.py's volume_tolerance default.
    zero_volume_abs_mm3: float = 1e-6
    # Absolute: ~4 orders of magnitude above KERNEL_BASE_TOLERANCE_MM. Below
    # this the kernel is just carrying normal fp slop; above it, some boolean
    # step likely had to fatten a subshape's tolerance to close up.
    tolerance_bloat_abs_mm: float = 1e-3
    # Relative to bbox diagonal, so it means the same thing for a 5mm part
    # and a 5m part.
    small_edge_relative: float = 1e-3
    small_face_relative: float = 1e-3  # compared against sqrt(area)
    thin_wall_relative: float = 5e-3
    aspect_ratio_max: float = 500.0
    divergence_volume_relative: float = 0.01
    divergence_bbox_relative: float = 0.01
    thickness_samples: int = 300
    thickness_seed: int = 0
    tessellation_tolerance_mm: float = 0.1
    free_boundary_tolerance_mm: float = 1e-6


@dataclass(frozen=True)
class ShapeSignature:
    """Cheap, comparable fingerprint of a shape's gross geometry.

    Used both to detect a no-op CadQuery operation (op_trace.py) and to
    compare the executed-code shape against the shipped STEP (this module).
    """

    volume: float
    n_solids: int
    n_faces: int
    n_edges: int
    n_vertices: int
    bbox_extents: tuple[float, float, float]
    bbox_diagonal: float

    @classmethod
    def of(cls, shape: cq.Shape) -> "ShapeSignature":
        bbox_extents, bbox_diagonal = _safe_bbox(shape)
        return cls(
            volume=float(shape.Volume()),
            n_solids=len(shape.Solids()),
            n_faces=len(shape.Faces()),
            n_edges=len(shape.Edges()),
            n_vertices=len(shape.Vertices()),
            bbox_extents=bbox_extents,
            bbox_diagonal=bbox_diagonal,
        )

    def close_to(
        self, other: "ShapeSignature", *, volume_tol: float, extent_tol: float
    ) -> bool:
        # Counts first: a split-but-keep-both-halves or an added seam/imprint
        # face can change topology at ~constant volume and bbox, so volume+
        # bbox agreement alone is not sufficient to call two signatures the
        # same shape. Exact equality (not a tolerance) since these are counts.
        if (self.n_solids, self.n_faces, self.n_edges) != (
            other.n_solids,
            other.n_faces,
            other.n_edges,
        ):
            return False
        denom = max(abs(self.volume), abs(other.volume), 1e-12)
        if abs(self.volume - other.volume) / denom > volume_tol:
            return False
        return all(
            abs(a - b) <= extent_tol for a, b in zip(self.bbox_extents, other.bbox_extents)
        )


@dataclass(frozen=True)
class TopologyResult:
    brepcheck_valid: bool
    brepcheck_status_histogram: dict[str, int]
    self_intersects: bool
    kernel_flags_small_edge: bool
    has_open_boundary: bool
    free_boundary_edge_count: int
    non_manifold_edge_count: int
    max_tolerance_mm: float
    mean_tolerance_mm: float
    n_solids: int
    n_faces: int
    solid_volumes: list[float]
    n_small_edges_relative: int
    n_small_faces_relative: int
    volume: float
    bbox_extents: tuple[float, float, float]
    bbox_diagonal: float
    aspect_ratio: float


@dataclass(frozen=True)
class ThicknessResult:
    n_rays: int
    n_hits: int
    min_thickness_mm: float | None
    p05_thickness_mm: float | None
    median_thickness_mm: float | None
    min_thickness_relative: float | None


@dataclass(frozen=True)
class ShapeAudit:
    signature: ShapeSignature
    topology: TopologyResult
    thickness: ThicknessResult | None
    severity: Severity
    reasons: list[str]


def _iter_subshapes(shape, kind) -> list:
    out = []
    explorer = TopExp_Explorer(shape, kind)
    while explorer.More():
        out.append(explorer.Current())
        explorer.Next()
    return out


def _safe_bbox(shape: cq.Shape) -> tuple[tuple[float, float, float], float]:
    """(extents, diagonal), or zeros for a shape with no geometry at all.

    A genuinely empty result (e.g. a cut tool that fully enclosed the base --
    verified: 0 solids/faces/edges/vertices, no exception from the boolean op
    itself) has nothing to bound: ``shape.BoundingBox()`` raises
    Standard_ConstructionError("Bnd_Box is void") rather than returning a
    zero-size box. That should not blow away the rest of the audit for what
    is itself a real, reportable finding (empty_shape).
    """

    try:
        bb = shape.BoundingBox()
        return (bb.xlen, bb.ylen, bb.zlen), float(bb.DiagonalLength)
    except Exception:
        return (0.0, 0.0, 0.0), 0.0


def brepcheck_defect_histogram(shape: cq.Shape) -> tuple[bool, dict[str, int]]:
    """Walk BRepCheck_Analyzer per-subshape instead of trusting the bool.

    ``shape.isValid()`` collapses ~30 distinct BRepCheck_Status defect
    categories into one bool. Walking ``Result(subshape).Status()`` for every
    vertex/edge/wire/face/shell/solid gives the actual taxonomy, which is what
    makes it possible to see e.g. "23% of flagged parts are FreeEdge" instead
    of just "23% are invalid."
    """

    analyzer = BRepCheck_Analyzer(shape.wrapped)
    histogram: dict[str, int] = {}
    for kind in (
        TopAbs_VERTEX,
        TopAbs_EDGE,
        TopAbs_WIRE,
        TopAbs_FACE,
        TopAbs_SHELL,
        TopAbs_SOLID,
    ):
        for sub in _iter_subshapes(shape.wrapped, kind):
            result = analyzer.Result(sub)
            for status in result.Status():
                if status == BRepCheck_Status.BRepCheck_NoError:
                    continue
                name = str(status).rsplit(".", maxsplit=1)[-1]
                histogram[name] = histogram.get(name, 0) + 1
    return bool(analyzer.IsValid()), histogram


def check_self_intersection(shape: cq.Shape) -> tuple[bool, bool]:
    """Global 3D self-intersection + kernel-relative tiny-edge detection.

    BRepCheck_Analyzer does mostly local/parametric consistency checks; it
    does not reliably catch two non-adjacent parts of a shape overlapping in
    3D space (e.g. a botched union leaving interpenetrating material).
    BOPAlgo_ArgumentAnalyzer is what the boolean-op pipeline itself uses for
    that, so it is the authoritative source here. Confirmed empirically: a
    self-crossing (bowtie) profile extrusion is flagged with
    BOPAlgo_SelfIntersect in ~1-2ms; typical real parts take 5-200ms.

    Returns (self_intersects, kernel_flags_small_edge).
    """

    analyzer = BOPAlgo_ArgumentAnalyzer()
    analyzer.SetShape1(shape.wrapped)
    analyzer.SelfInterMode = True
    analyzer.SmallEdgeMode = True
    analyzer.Perform()
    self_intersects = False
    small_edge = False
    for result in analyzer.GetCheckResult():
        status = result.GetCheckStatus()
        if status == BOPAlgo_CheckStatus.BOPAlgo_SelfIntersect:
            self_intersects = True
        elif status == BOPAlgo_CheckStatus.BOPAlgo_TooSmallEdge:
            small_edge = True
    return self_intersects, small_edge


def check_open_boundary(
    shape: cq.Shape, *, tolerance: float = 1e-6
) -> tuple[bool, int]:
    """B-rep-level watertightness: any naked/free boundary edge at all.

    Purely topological (no tessellation tolerance involved), so it is a more
    reliable "is this actually a closed solid" signal than tessellate-then-
    check-mesh-is-watertight. An open shell (e.g. one face of a box missing)
    shows up as edges in GetClosedWires()/GetOpenWires(); a properly closed
    solid shows zero in both (verified empirically).
    """

    free_bounds = ShapeAnalysis_FreeBounds(shape.wrapped, tolerance)
    edge_count = len(_iter_subshapes(free_bounds.GetClosedWires(), TopAbs_EDGE))
    edge_count += len(_iter_subshapes(free_bounds.GetOpenWires(), TopAbs_EDGE))
    return edge_count > 0, edge_count


def check_non_manifold_edges(shape: cq.Shape) -> int:
    """Count edges shared by >2 faces, evaluated per top-level solid.

    A normal manifold edge has exactly 2 face ancestors. This is scoped to
    each individual TopAbs_SOLID (not the whole compound) because two
    genuinely disjoint solids that happen to share coincident boundary
    geometry (see module docstring) would otherwise look non-manifold at the
    compound level even though each solid on its own is fine; that case is
    already covered by n_solids != 1 in ``connected_components``.
    """

    total = 0
    for solid in shape.Solids():
        shape_map = TopTools_IndexedDataMapOfShapeListOfShape()
        TopExp.MapShapesAndAncestors_s(
            solid.wrapped, TopAbs_EDGE, TopAbs_FACE, shape_map
        )
        for i in range(1, shape_map.Extent() + 1):
            if shape_map.FindFromIndex(i).Extent() > 2:
                total += 1
    return total


def check_tolerance(shape: cq.Shape) -> tuple[float, float]:
    """Max/mean stored tolerance across all vertices/edges/faces.

    A tolerance far above KERNEL_BASE_TOLERANCE_MM is a proxy for "some
    boolean/fillet step had to fatten a subshape's tolerance to close up the
    result" -- a numerically-strained kernel operation that produced a
    technically-valid but geometrically-suspect result.
    """

    # BRep_Tool.Tolerance_s is overloaded on the concrete downcast type
    # (TopoDS_Face/Edge/Vertex) and rejects a base TopoDS_Shape at the
    # binding layer (verified: raises TypeError without the downcast), so
    # each kind needs its own explicit TopoDS.* cast.
    downcast = {
        TopAbs_VERTEX: TopoDS.Vertex,
        TopAbs_EDGE: TopoDS.Edge,
        TopAbs_FACE: TopoDS.Face,
    }
    values: list[float] = []
    for kind, cast in downcast.items():
        for sub in _iter_subshapes(shape.wrapped, kind):
            values.append(float(BRep_Tool.Tolerance_s(cast(sub))))
    if not values:
        return 0.0, 0.0
    return max(values), sum(values) / len(values)


def connected_components(shape: cq.Shape) -> list[float]:
    """Per-solid volume of every top-level TopAbs_SOLID in ``shape``.

    len() == 1 is the healthy case for a single-part Zero-To-CAD sample.
    >1 means the result fragmented into disjoint islands OR two lobes only
    touch along a degenerate (zero-area) edge/vertex -- OCC represents both
    as separate solids rather than one non-manifold solid (verified: two
    boxes unioned so they share only one vertical edge come back as
    n_solids=2, volume = sum of both, not one welded body).
    """

    return [float(solid.Volume()) for solid in shape.Solids()]


def small_features(
    shape: cq.Shape, *, bbox_diagonal: float, edge_relative: float, face_relative: float
) -> tuple[int, int]:
    """Count edges/faces small relative to the part's own bbox diagonal.

    Distinct from check_self_intersection's kernel_flags_small_edge: that one
    is the kernel's own (tolerance-relative) opinion; this one is "is this
    edge/face a visually/semantically negligible sliver of THIS part,"
    e.g. a 0.01mm edge on a 500mm part is kernel-fine but design-degenerate.
    """

    if bbox_diagonal <= 0:
        return 0, 0
    edge_floor = edge_relative * bbox_diagonal
    face_floor = face_relative * bbox_diagonal
    n_small_edges = sum(1 for e in shape.Edges() if e.Length() < edge_floor)
    n_small_faces = sum(1 for f in shape.Faces() if math.sqrt(max(f.Area(), 0.0)) < face_floor)
    return n_small_edges, n_small_faces


def sample_wall_thickness(
    shape: cq.Shape,
    *,
    n_samples: int = 300,
    seed: int = 0,
    tessellation_tolerance: float = 0.1,
) -> ThicknessResult:
    """Sampled Monte-Carlo local-thickness proxy via inward ray-casting.

    Tessellates once, area-weight-samples surface points, casts a ray from
    each point along -normal, and takes the first back-face hit distance as a
    local thickness estimate. This is the standard cheap approximation (not
    an analytic/exhaustive thickness field): it can miss thin features a
    sparse sample doesn't land near, and it can't tell a genuine thin wall
    from a normal-but-acute corner. Verified accurate on synthetic shapes
    (10mm box -> ~10.0, 0.2mm-thick plate -> ~0.1999) and fast (<15ms for 300
    rays on real ~100-face GT parts).
    """

    import numpy as np
    import trimesh

    if n_samples <= 0:
        return ThicknessResult(0, 0, None, None, None, None)

    vertices, faces = shape.tessellate(tessellation_tolerance)
    mesh = trimesh.Trimesh(
        vertices=[(v.x, v.y, v.z) for v in vertices], faces=faces, process=True
    )
    if len(mesh.faces) < 1:
        return ThicknessResult(n_samples, 0, None, None, None, None)

    points, face_idx = trimesh.sample.sample_surface(mesh, n_samples, seed=seed)
    normals = mesh.face_normals[face_idx]
    eps = 1e-6 * max(mesh.scale, 1e-9)
    origins = points - normals * eps
    directions = -normals
    locations, index_ray, _ = mesh.ray.intersects_location(
        origins, directions, multiple_hits=False
    )
    n_hits = len(locations)
    if n_hits == 0:
        return ThicknessResult(n_samples, 0, None, None, None, None)

    dists = np.linalg.norm(locations - origins[index_ray], axis=1)
    dists.sort()
    _, bbox_diag = _safe_bbox(shape)
    min_thickness = float(dists[0])
    return ThicknessResult(
        n_rays=n_samples,
        n_hits=int(n_hits),
        min_thickness_mm=min_thickness,
        p05_thickness_mm=float(dists[max(0, int(0.05 * len(dists)) - 1)]),
        median_thickness_mm=float(dists[len(dists) // 2]),
        min_thickness_relative=(min_thickness / bbox_diag if bbox_diag > 0 else None),
    )


def audit_topology(shape: cq.Shape, thresholds: Thresholds) -> TopologyResult:
    brepcheck_valid, histogram = brepcheck_defect_histogram(shape)
    self_intersects, kernel_small_edge = check_self_intersection(shape)
    has_open_boundary, free_edge_count = check_open_boundary(
        shape, tolerance=thresholds.free_boundary_tolerance_mm
    )
    non_manifold = check_non_manifold_edges(shape)
    max_tol, mean_tol = check_tolerance(shape)
    solid_volumes = connected_components(shape)
    extents, bbox_diag = _safe_bbox(shape)
    n_small_edges, n_small_faces = small_features(
        shape,
        bbox_diagonal=bbox_diag,
        edge_relative=thresholds.small_edge_relative,
        face_relative=thresholds.small_face_relative,
    )
    min_extent = max(min(extents), 1e-12)
    return TopologyResult(
        brepcheck_valid=brepcheck_valid,
        brepcheck_status_histogram=histogram,
        self_intersects=self_intersects,
        kernel_flags_small_edge=kernel_small_edge,
        has_open_boundary=has_open_boundary,
        free_boundary_edge_count=free_edge_count,
        non_manifold_edge_count=non_manifold,
        max_tolerance_mm=max_tol,
        mean_tolerance_mm=mean_tol,
        n_solids=len(solid_volumes),
        n_faces=len(shape.Faces()),
        solid_volumes=solid_volumes,
        n_small_edges_relative=n_small_edges,
        n_small_faces_relative=n_small_faces,
        volume=float(shape.Volume()),
        bbox_extents=extents,
        bbox_diagonal=bbox_diag,
        aspect_ratio=max(extents) / min_extent,
    )


def classify_severity(
    topology: TopologyResult,
    thickness: ThicknessResult | None,
    thresholds: Thresholds,
    *,
    extra_soft_reasons: Sequence[str] = (),
) -> tuple[Severity, list[str]]:
    """Fold check results into one verdict + auditable reason codes.

    HARD_INVALID: the shape itself is broken (self-intersecting, non-
    manifold, open, fragmented, zero/negative volume) -- these are the
    candidates for hard exclusion from training.
    SOFT_SUSPECT: the shape is a technically valid solid but has a smell
    (tolerance bloat, micro features, thin walls, extreme aspect) --
    candidates for review/downweighting, not automatic exclusion.
    """

    hard: list[str] = []
    if not topology.brepcheck_valid:
        hard.append("brepcheck_invalid")
    if topology.self_intersects:
        hard.append("self_intersection")
    if topology.non_manifold_edge_count > 0:
        hard.append("non_manifold_edge")
    if topology.has_open_boundary:
        hard.append("open_boundary")
    if topology.n_solids == 0:
        # n_solids==0 with real geometry present (n_faces>0) is a distinct,
        # important case from a genuinely empty result: it means the shape
        # root is a bare Compound/Shell that was never wrapped into a
        # TopAbs_SOLID -- confirmed on real Zero-To-CAD STEP files (a
        # TopAbs_COMPOUND of 2 open shells / 8 faces, 0 solids). Shape.Volume()
        # still returns a plausible-looking nonzero number for such shapes
        # (the divergence-theorem surface integral doesn't require a closed
        # boundary to produce *a* number), which is exactly why n_solids is
        # gated on above rather than trusting volume alone.
        hard.append("unsolidified_shell" if topology.n_faces > 0 else "empty_shape")
    elif topology.n_solids > 1:
        hard.append("disjoint_solids")
    if topology.n_solids >= 1 and (
        topology.volume <= thresholds.zero_volume_abs_mm3
    ):
        hard.append("zero_or_negative_volume")

    if hard:
        return Severity.HARD_INVALID, hard

    soft: list[str] = list(extra_soft_reasons)
    if topology.max_tolerance_mm > thresholds.tolerance_bloat_abs_mm:
        soft.append("tolerance_bloat")
    if topology.kernel_flags_small_edge:
        soft.append("kernel_small_edge")
    if topology.n_small_edges_relative > 0:
        soft.append("micro_edge")
    if topology.n_small_faces_relative > 0:
        soft.append("micro_face")
    if topology.aspect_ratio > thresholds.aspect_ratio_max:
        soft.append("extreme_aspect_ratio")
    if (
        thickness is not None
        and thickness.min_thickness_relative is not None
        and thickness.min_thickness_relative < thresholds.thin_wall_relative
    ):
        soft.append("thin_wall")

    return (Severity.SOFT_SUSPECT if soft else Severity.OK), soft


def audit_shape(
    shape: cq.Shape,
    thresholds: Thresholds | None = None,
    *,
    with_thickness: bool = True,
) -> ShapeAudit:
    """Run the full check battery on one shape and return a tiered verdict."""

    thresholds = thresholds or Thresholds()
    signature = ShapeSignature.of(shape)
    topology = audit_topology(shape, thresholds)
    thickness = None
    # Thickness sampling needs a tessellatable, non-degenerate solid; skip it
    # for shapes already hard-invalid at the topology level (garbage in,
    # garbage out -- and tessellate() can itself throw on some invalid shapes).
    hard_precheck = (
        not topology.brepcheck_valid
        or topology.self_intersects
        or topology.n_solids != 1
        or topology.volume <= thresholds.zero_volume_abs_mm3
    )
    if with_thickness and not hard_precheck:
        try:
            thickness = sample_wall_thickness(
                shape,
                n_samples=thresholds.thickness_samples,
                seed=thresholds.thickness_seed,
                tessellation_tolerance=thresholds.tessellation_tolerance_mm,
            )
        except Exception:
            thickness = None
    severity, reasons = classify_severity(topology, thickness, thresholds)
    return ShapeAudit(
        signature=signature,
        topology=topology,
        thickness=thickness,
        severity=severity,
        reasons=reasons,
    )


def compare_signatures(
    a: ShapeSignature, b: ShapeSignature, thresholds: Thresholds
) -> tuple[bool, dict[str, float]]:
    """Diff two signatures (typically: executed-code shape vs shipped STEP).

    Divergence here means the GT CadQuery program, executed locally, does not
    reproduce the shape the rest of the pipeline (2D-drawing rendering, IoU/CD
    scoring) actually uses -- a text-target/shape mismatch, not merely a
    "this one shape is invalid" defect. See gt_audit.py docstring for why this
    is scored as its own category rather than folded into the per-shape tiers.
    """

    volume_denom = max(abs(a.volume), abs(b.volume), 1e-12)
    volume_rel = abs(a.volume - b.volume) / volume_denom
    bbox_denom = max(a.bbox_diagonal, b.bbox_diagonal, 1e-12)
    bbox_rel = max(abs(x - y) for x, y in zip(a.bbox_extents, b.bbox_extents)) / bbox_denom
    diverges = (
        volume_rel > thresholds.divergence_volume_relative
        or bbox_rel > thresholds.divergence_bbox_relative
        or a.n_solids != b.n_solids
    )
    return diverges, {
        "volume_relative_diff": volume_rel,
        "bbox_relative_diff": bbox_rel,
        "n_faces_diff": a.n_faces - b.n_faces,
        "n_edges_diff": a.n_edges - b.n_edges,
        "n_solids_diff": a.n_solids - b.n_solids,
    }


__all__ = [
    "KERNEL_BASE_TOLERANCE_MM",
    "Severity",
    "Thresholds",
    "ShapeSignature",
    "TopologyResult",
    "ThicknessResult",
    "ShapeAudit",
    "brepcheck_defect_histogram",
    "check_self_intersection",
    "check_open_boundary",
    "check_non_manifold_edges",
    "check_tolerance",
    "connected_components",
    "small_features",
    "sample_wall_thickness",
    "audit_topology",
    "audit_shape",
    "classify_severity",
    "compare_signatures",
]
