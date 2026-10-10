"""What a built solid is made of, counted by kind.

Several places in the prompts tell the coder to model curves as curves, and
they can all fail the same way: told to be exact, a model samples its arcs
more finely rather than differently, and the solid comes out faceted. One
measured run reached 793 straight edges against a target of 119. Nothing in the
loop had ever put that number in front of the coder.

A census is that number, and it costs one read of a STEP already written.
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path


def _kinds(items: list, adaptor) -> Counter[str]:
    counted: Counter[str] = Counter()
    for item in items:
        name = str(adaptor(item.wrapped).GetType()).rsplit(".", 1)[-1]
        counted[name.removeprefix("GeomAbs_")] += 1
    return counted


def _by_kind(counted: Counter[str]) -> str:
    return ", ".join(f"{kind} {n}" for kind, n in counted.most_common())


# Smaller inner shells are numerical slivers, not cavities.
_MIN_VOID_MM3 = 0.01


@dataclass(frozen=True)
class Void:
    """A cavity sealed inside a solid: an inner shell no opening reaches."""

    volume: float
    bbox: tuple[float, float, float, float, float, float]

    def describe(self) -> str:
        # + 0.0 turns a rounded -0.0 into 0.0
        corners = ", ".join(f"{round(v, 1) + 0.0:.1f}" for v in self.bbox)
        return f"{self.volume:.1f} mm³ at [{corners}]"


@dataclass(frozen=True)
class ShapeCensus:
    """How many pieces a part is in, how big it is, and what bounds it."""

    solids: int
    volume: float
    extent: tuple[float, float, float]
    faces: Counter[str]
    edges: Counter[str]
    voids: tuple[Void, ...] = ()

    def _describe_extent(self) -> str:
        """The bounding box of the built solid, as the overall dimensions read.

        Two decimals, because a part that came out 98.75 wide against a printed
        100 is a real miss and `98.8` reads like agreement.
        """
        return "bbox " + " x ".join(f"{length:.2f}" for length in self.extent)

    def _describe_voids(self) -> list[str]:
        """Count sealed voids and place each one."""
        if not self.voids:
            return []
        where = ", ".join(void.describe() for void in self.voids)
        return [
            (
                f"sealed voids {len(self.voids)} (cavities no opening reaches, "
                f"bbox [xmin, ymin, zmin, xmax, ymax, zmax]: {where})"
            )
        ]

    def describe(self) -> str:
        """Give the volume and size, then count the faces and edges by kind.

        One line, in the report the coder reads every turn: `faces 85 (Cylinder
        42, Plane 33, ...)`. Kinds rather than a total, because a total says a
        part is complicated and a kind says a cylinder came out as a hundred
        flat strips.

        A part that is not one solid says so first. One solid is the normal
        case and goes unsaid.
        """
        return "; ".join(
            [
                *([f"solids {self.solids}"] if self.solids != 1 else []),
                f"volume {self.volume:.1f}",
                self._describe_extent(),
                *self._describe_voids(),
                *(
                    f"{label} {sum(counted.values())} ({_by_kind(counted)})"
                    for label, counted in (("faces", self.faces), ("edges", self.edges))
                    if counted
                ),
            ]
        )


def read_census(step_path: Path) -> ShapeCensus | None:
    """Read a STEP and count the volume, faces and edges of its solid.

    None when the file cannot be read as a shape; the caller is reporting a
    build that already succeeded, and a census that fails is not a reason to
    fail it.
    """
    # Imported here, as `verify_step` does, so that reading this module costs
    # nothing to a process that never builds anything.
    import cadquery as cq
    from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface

    try:
        # The whole compound, not `.val()`: a part that broke apart must be
        # counted whole, or the census silently reports one of its pieces.
        shape = cq.Compound.makeCompound(
            cq.importers.importStep(str(step_path)).vals()  # type: ignore[arg-type]
        )
        box = shape.BoundingBox()
        return ShapeCensus(
            solids=len(shape.Solids()),
            volume=shape.Volume(),
            extent=(box.xlen, box.ylen, box.zlen),
            faces=_kinds(shape.Faces(), BRepAdaptor_Surface),
            edges=_kinds(shape.Edges(), BRepAdaptor_Curve),
            voids=_voids(shape),
        )
    except Exception:  # noqa: BLE001 - e.g. an empty nested compound; a diagnostic must not stop verification
        return None


def _voids(shape) -> tuple[Void, ...]:
    """Every shell of each solid other than its outer one."""
    import cadquery as cq
    from OCP.BRepClass3d import BRepClass3d

    voids = []
    for solid in shape.Solids():
        outer = BRepClass3d.OuterShell_s(solid.wrapped)
        for shell in solid.Shells():
            if shell.wrapped.IsSame(outer):
                continue
            volume = cq.Solid.makeSolid(shell).Volume()
            if volume >= _MIN_VOID_MM3:
                box = shell.BoundingBox()
                bbox = (box.xmin, box.ymin, box.zmin, box.xmax, box.ymax, box.zmax)
                voids.append(Void(volume, bbox))
    return tuple(voids)
