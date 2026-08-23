"""What a built solid is made of, counted by kind.

Several places in the prompts tell the coder to model curves as curves, and
they can all fail the same way: told to be exact, a model samples its arcs
more finely rather than differently, and the solid comes out faceted. One
measured run reached 793 straight edges against a target of 119. Nothing in the
loop had ever put that number in front of the coder.

A census is that number, and it costs one read of a STEP already written.
"""

from collections import Counter
from pathlib import Path


def _kinds(items: list, adaptor) -> Counter[str]:
    counted: Counter[str] = Counter()
    for item in items:
        name = str(adaptor(item.wrapped).GetType()).rsplit(".", 1)[-1]
        counted[name.removeprefix("GeomAbs_")] += 1
    return counted


def describe_shape(step_path: Path) -> str:
    """Count the faces and edges of a STEP by surface and curve kind.

    One line, in the report the coder reads every turn: `faces 85 (Cylinder 42,
    Plane 33, ...)`. Kinds rather than a total, because a total says a part is
    complicated and a kind says a cylinder came out as a hundred flat strips.

    Returns an empty string when the file cannot be read as a shape; the caller
    is reporting a build that already succeeded, and a census that fails is not
    a reason to fail it.
    """
    # Imported here, as `verify_step` does, so that reading this module costs
    # nothing to a process that never builds anything.
    import cadquery as cq
    from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface

    try:
        shape = cq.importers.importStep(str(step_path)).val()
    except (OSError, ValueError, RuntimeError, IndexError):
        # A build that reached here has already been accepted as one valid
        # solid, so this should not fire. It is here because a census is a
        # diagnostic: failing to count what was built must not turn a build
        # that worked into a run that crashed.
        return ""

    faces = _kinds(shape.Faces(), BRepAdaptor_Surface)
    edges = _kinds(shape.Edges(), BRepAdaptor_Curve)
    return "; ".join(
        f"{label} {sum(counted.values())} "
        f"({', '.join(f'{kind} {n}' for kind, n in counted.most_common())})"
        for label, counted in (("faces", faces), ("edges", edges))
        if counted
    )
