"""Locate review-worthy structures in the built solid; never repair it."""

from collections import defaultdict
from itertools import combinations
from math import sqrt

import cadquery as cq
from OCP.BRepAdaptor import BRepAdaptor_Surface

# Review candidates, not validity limits. All size tests scale with the solid.
_SMALL_FRACTION = 0.1
_TOL = 1e-7  # OCCT's usual linear confusion, in mm.
_ANGULAR_TOL = 1e-9
_MAX_REGIONS = 4


def find_shape_warnings(shape) -> tuple[str, ...]:
    """Run the three structural checks on the built solid."""
    return (
        *_check_local_thin_lands(shape),
        *_check_shallow_steps(shape),
        *_check_same_surface_splits(shape),
    )


def _check_local_thin_lands(shape) -> tuple[str, ...]:
    """Find broad, unusually thin material between opposing planar faces.

    shortcut: curved thin walls are not covered; extend this check if they
    become common reconstruction failures.
    """
    regions = []
    for solid in shape.Solids():
        limit = _small_feature_limit(solid)
        if limit <= 0:
            continue
        faces = solid.Faces()
        planes = {i for i, face in enumerate(faces) if face.geomType() == "PLANE"}
        for i, j in combinations(sorted(planes), 2):
            a, b = faces[i], faces[j]
            n = a.normalAt()
            if n.dot(b.normalAt()) > -1 + _ANGULAR_TOL:
                continue
            depth = -(b.Center() - a.Center()).dot(n)
            if not _TOL < depth < limit:
                continue
            # Bring the trimmed faces onto one plane, not just their bounding boxes.
            overlapping = a.intersect(b.translate(n.multiply(depth)))
            for patch in overlapping.Faces():
                area = patch.Area()
                if depth >= _SMALL_FRACTION * sqrt(area):
                    continue
                prism = cq.Solid.extrudeLinear(
                    patch.outerWire(), patch.innerWires(), n.multiply(-depth)
                )
                # A close pair across a void, or with another cavity between, is not a wall.
                if abs(prism.cut(solid).Volume()) > max(
                    area * _TOL, abs(prism.Volume()) * 1e-6
                ):
                    continue
                regions.append(f"thickness {depth:.6g} mm at {_place(prism)}")
    return _summarize("Possible local thin lands", regions)


def _check_shallow_steps(shape) -> tuple[str, ...]:
    """Find small offsets joined by a planar or cylindrical step face."""
    regions = []
    for solid in shape.Solids():
        limit = _small_feature_limit(solid)
        if limit <= 0:
            continue
        faces = solid.Faces()
        planes = {i for i, face in enumerate(faces) if face.geomType() == "PLANE"}
        neighbours = [set() for _ in faces]
        for _edge, a, b in _shared_edges(faces):
            neighbours[a].add(b)
            neighbours[b].add(a)
        for i, face in enumerate(faces):
            kind = face.geomType()
            if kind not in {"PLANE", "CYLINDER"}:
                continue
            adjacent_planes = sorted(neighbours[i] & planes)
            for a, b in combinations(adjacent_planes, 2):
                n = faces[a].normalAt()
                # A step joins offset faces pointing the same way, unlike a thin wall.
                if n.dot(faces[b].normalAt()) < 1 - _ANGULAR_TOL:
                    continue
                if kind == "PLANE":
                    if abs(n.dot(face.normalAt())) > _ANGULAR_TOL:
                        continue
                else:
                    axis = (
                        BRepAdaptor_Surface(face.wrapped).Cylinder().Axis().Direction()
                    )
                    if not _parallel(axis, n.toDir()):
                        continue
                start = faces[a].Center().dot(n)
                end = faces[b].Center().dot(n)
                depth = abs(end - start)
                if not _TOL < depth < limit:
                    continue
                # Exclude a face that also extends beyond this pair of levels.
                bounds = cq.Plane((0, 0, 0), normal=n).toLocalCoords(face).BoundingBox()
                if (
                    bounds.zmin < min(start, end) - _TOL
                    or bounds.zmax > max(start, end) + _TOL
                ):
                    continue
                if face.Area() <= 10 * depth**2:
                    continue
                regions.append(f"depth {depth:.6g} mm at {_place(face)}")
                break
    return _summarize("Possible shallow steps", regions)


def _check_same_surface_splits(shape) -> tuple[str, ...]:
    """Find adjoining faces on the same plane or cylinder, excluding STEP seams."""
    regions = []
    for solid in shape.Solids():
        if solid.Area() <= 0 or solid.Volume() <= 0:
            continue
        faces = solid.Faces()
        for edge, a, b in _shared_edges(faces):
            if _same_domain(faces[a], faces[b]):
                if faces[a].geomType() == "CYLINDER" and edge.geomType() == "LINE":
                    axis = (
                        BRepAdaptor_Surface(faces[a].wrapped)
                        .Cylinder()
                        .Axis()
                        .Direction()
                    )
                    # STEP commonly stores a cylinder in halves along its meridians.
                    if _parallel(axis, edge.tangentAt().toDir()):
                        continue
                regions.append(
                    f"{faces[a].geomType().lower()} partition at {_place(edge)}"
                )
    return _summarize("Adjacent faces on the same analytic surface", regions)


def _parallel(a, b) -> bool:
    return abs(a.Dot(b)) >= 1 - _ANGULAR_TOL


def _same_domain(a, b) -> bool:
    """Only exact analytic supports, not merely tangent neighbouring faces."""
    kind = a.geomType()
    if kind != b.geomType():
        return False
    sa, sb = BRepAdaptor_Surface(a.wrapped), BRepAdaptor_Surface(b.wrapped)
    if kind == "PLANE":
        pa, pb = sa.Plane(), sb.Plane()
        n = pa.Axis().Direction()
        delta = pb.Location().XYZ().Subtracted(pa.Location().XYZ())
        return _parallel(n, pb.Axis().Direction()) and abs(delta.Dot(n.XYZ())) <= _TOL
    if kind == "CYLINDER":
        ca, cb = sa.Cylinder(), sb.Cylinder()
        axis = ca.Axis().Direction()
        delta = cb.Location().XYZ().Subtracted(ca.Location().XYZ())
        delta.Subtract(axis.XYZ().Multiplied(delta.Dot(axis.XYZ())))
        return (
            _parallel(axis, cb.Axis().Direction())
            and abs(ca.Radius() - cb.Radius()) <= _TOL
            and delta.Modulus() <= _TOL
        )
    return False


def _small_feature_limit(solid) -> float:
    area, volume = solid.Area(), solid.Volume()
    if area <= 0 or volume <= 0:
        return 0.0
    # For a broad uniform sheet 2V/A approaches its thickness, not its width.
    return _SMALL_FRACTION * 2 * volume / area


def _shared_edges(faces):
    edge_faces = defaultdict(list)
    for i, face in enumerate(faces):
        for edge in face.Edges():
            edge_faces[edge].append(i)
    for edge, indices in edge_faces.items():
        if len(indices) == 2:
            yield edge, *indices


def _place(shape) -> str:
    box = shape.BoundingBox()
    return ", ".join(
        f"{axis}={low + 0.0:.6g}..{high + 0.0:.6g}"
        for axis, low, high in (
            ("x", box.xmin, box.xmax),
            ("y", box.ymin, box.ymax),
            ("z", box.zmin, box.zmax),
        )
    )


def _summarize(label: str, regions: list[str]) -> tuple[str, ...]:
    if not regions:
        return ()
    shown = "; ".join(regions[:_MAX_REGIONS])
    more = (
        f"; {len(regions) - _MAX_REGIONS} more" if len(regions) > _MAX_REGIONS else ""
    )
    unit = "region" if len(regions) == 1 else "regions"
    return (f"{label} ({len(regions)} {unit}): {shown}{more}.",)
