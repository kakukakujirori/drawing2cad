"""What a built solid is made of, counted by kind.

The coder is told to model curves as curves, and three separate places in the
prompts say so. On 001100 all three failed in the same direction: told to be
exact, the model sampled its arcs and splines into a 200-point polyline, and
the solid came out with 793 straight edges where the target has 119, 350 of
them under a millimetre long. Nothing in the loop ever put a number in front of
it, so there was nothing to notice.

A census is that number. It says what was built rather than what should have
been, and it costs one read of the STEP that has already been written.
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
