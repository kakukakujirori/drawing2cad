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


def _by_kind_change(now: Counter[str], before: Counter[str]) -> str:
    changed = {kind: now[kind] - before[kind] for kind in now.keys() | before.keys()}
    return ", ".join(
        f"{kind} {delta:+d}"
        for kind, delta in sorted(changed.items(), key=lambda kv: (-abs(kv[1]), kv[0]))
        if delta
    )


@dataclass(frozen=True)
class ShapeCensus:
    """How many pieces a part is in, how big it is, and what bounds it."""

    solids: int
    volume: float
    faces: Counter[str]
    edges: Counter[str]

    def describe(self) -> str:
        """Give the volume, then count the faces and edges by kind.

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
                *(
                    f"{label} {sum(counted.values())} ({_by_kind(counted)})"
                    for label, counted in (("faces", self.faces), ("edges", self.edges))
                    if counted
                ),
            ]
        )

    def describe_change_from(self, previous: "ShapeCensus") -> str:
        """Give the same counts, and after each one its change since `previous`.

        An operation that built nothing shows `+0.0` volume and `+0` faces.
        That is what an accidental identity looks like from the outside. An
        operation that broke the part apart shows the solid count going up.
        """
        parts = [
            *(
                [f"solids {self.solids} ({self.solids - previous.solids:+d})"]
                if self.solids != 1 or previous.solids != 1
                else []
            ),
            f"volume {self.volume:.1f} ({self.volume - previous.volume:+.1f})",
        ]
        for label, now, before in (
            ("faces", self.faces, previous.faces),
            ("edges", self.edges, previous.edges),
        ):
            total, delta = sum(now.values()), sum(now.values()) - sum(before.values())
            kinds = _by_kind_change(now, before)
            parts.append(f"{label} {total} ({delta:+d}{f': {kinds}' if kinds else ''})")
        return "; ".join(parts)


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
        volume = shape.Volume()
    except (OSError, ValueError, RuntimeError, IndexError):
        # A build that reached here has already been accepted as one valid
        # solid, so this should not fire. It is here because a census is a
        # diagnostic: failing to count what was built must not turn a build
        # that worked into a run that crashed.
        return None

    return ShapeCensus(
        solids=len(shape.Solids()),
        volume=volume,
        faces=_kinds(shape.Faces(), BRepAdaptor_Surface),
        edges=_kinds(shape.Edges(), BRepAdaptor_Curve),
    )
