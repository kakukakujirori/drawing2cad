"""Execute a GT CadQuery program while tracing whether each chained solid op
actually changed the shape it was called on.

Monkeypatches ``cadquery.Workplane``'s solid-modifying methods rather than
parsing the source with ``ast``: GT programs in this corpus are not flat
one-liners (they route through builder classes -- ``self.build()`` methods,
loops emitting a list of cut tools, etc.), and patching at the API boundary
means tracing is correct regardless of that surrounding structure. Verified
end-to-end against a real staged GT program: correctly recorded contributing
extrude/chamfer/cut calls AND caught a real no-op ``union`` (added a rib
that was already fully inside the base solid -- shape signature identical
before/after).

The signature compared before/after each call is (volume, n_faces, n_edges,
n_solids) -- volume alone is not enough (a fillet changes shape at
~constant volume; a cut immediately undone by a union can cancel), so any
component moving is treated as a real contribution.
"""

from __future__ import annotations

import contextlib
import functools
from dataclasses import dataclass

import cadquery as cq

from src.data.audit.solid_checks import ShapeSignature


# Every cq.Workplane method that can change the solid it's chained onto.
# Sketch-only / selector / positioning calls (rect, center, faces, ...) are
# deliberately excluded: they don't claim to modify material, so "did it
# contribute" isn't a meaningful question for them.
SOLID_OPS: tuple[str, ...] = (
    "cut",
    "cutBlind",
    "cutEach",
    "cutThruAll",
    "union",
    "combine",
    "intersect",
    "extrude",
    "twistExtrude",
    "fillet",
    "chamfer",
    "shell",
    "loft",
    "revolve",
    "sweep",
    "split",
)


@dataclass(frozen=True)
class OpTraceEntry:
    index: int
    method: str
    contributed: bool | None  # None when before/after couldn't be resolved
    before: ShapeSignature | None
    after: ShapeSignature | None


def _val_or_none(workplane_or_shape) -> cq.Shape | None:
    try:
        if isinstance(workplane_or_shape, cq.Workplane):
            if not workplane_or_shape.objects:
                return None
            candidate = workplane_or_shape.val()
        else:
            candidate = workplane_or_shape
        return candidate if isinstance(candidate, cq.Shape) else None
    except Exception:
        return None


def _signature_or_none(shape: cq.Shape | None) -> ShapeSignature | None:
    if shape is None:
        return None
    try:
        return ShapeSignature.of(shape)
    except Exception:
        return None


def _same_signature(a: ShapeSignature | None, b: ShapeSignature | None) -> bool | None:
    if a is None or b is None:
        return None
    return a.close_to(b, volume_tol=1e-9, extent_tol=1e-9)


@contextlib.contextmanager
def _patched_workplane_ops(trace: list[OpTraceEntry]):
    """Wrap each SOLID_OPS method on cq.Workplane for the duration of the
    ``with`` block, appending one OpTraceEntry per call. Always restores the
    original methods on exit (including on exception) -- required because a
    persistent multiprocessing worker reuses the same interpreter/class
    across many samples, so a leaked patch would corrupt later samples.
    """

    originals: dict[str, object] = {}

    def make_wrapper(name: str, original):
        @functools.wraps(original)
        def wrapper(self, *args, **kwargs):
            before = _signature_or_none(_val_or_none(self))
            result = original(self, *args, **kwargs)
            after = _signature_or_none(_val_or_none(result))
            contributed = _same_signature(before, after)
            trace.append(
                OpTraceEntry(
                    index=len(trace),
                    method=name,
                    contributed=(None if contributed is None else not contributed),
                    before=before,
                    after=after,
                )
            )
            return result

        return wrapper

    try:
        for name in SOLID_OPS:
            original = getattr(cq.Workplane, name, None)
            if original is None:
                continue
            originals[name] = original
            setattr(cq.Workplane, name, make_wrapper(name, original))
        yield trace
    finally:
        for name, original in originals.items():
            setattr(cq.Workplane, name, original)


def _recover_shape(namespace: dict) -> cq.Shape | None:
    """Same convention as src/evaluation/executor.py: prefer a `result`/`r`
    name, else fall back to the last Workplane/Shape value bound in the
    executed namespace."""

    def unwrap(value: object) -> cq.Shape | None:
        if isinstance(value, cq.Workplane):
            return _val_or_none(value)
        if isinstance(value, cq.Shape):
            return value
        return None

    for name in ("result", "r"):
        if name in namespace:
            shape = unwrap(namespace[name])
            if shape is not None:
                return shape
    candidate = None
    for value in namespace.values():
        shape = unwrap(value)
        if shape is not None:
            candidate = shape
    return candidate


def trace_source(source: str) -> tuple[cq.Shape | None, list[OpTraceEntry], str | None]:
    """Exec ``source`` (a GT ``{uuid}.cadquery.py``) with op-tracing enabled.

    Returns (recovered_shape_or_None, trace_entries, error_or_None). Never
    raises for ordinary Python exceptions in ``source`` -- the caller (a
    multiprocessing worker) is expected to still want the partial trace and
    an error string rather than an unhandled exception. Does not itself
    contain crashes (SIGSEGV) or hangs; callers running this at scale should
    still wrap it in their own timeout/process isolation.
    """

    trace: list[OpTraceEntry] = []
    namespace: dict = {"__name__": "__main__", "cq": cq}
    error: str | None = None
    with _patched_workplane_ops(trace):
        try:
            exec(compile(source, "<gt-cadquery>", "exec"), namespace)
        except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
            error = f"{type(exc).__name__}: {' '.join(str(exc).split())}"[:500]
            return None, trace, error
    shape = _recover_shape(namespace)
    if shape is None:
        error = "no_result"
    return shape, trace, error


def noop_count(trace: list[OpTraceEntry]) -> int:
    return sum(1 for entry in trace if entry.contributed is False)


__all__ = ["SOLID_OPS", "OpTraceEntry", "trace_source", "noop_count"]
