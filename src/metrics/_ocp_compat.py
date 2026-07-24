"""Compatibility shim between CADGenBench and this environment's OCP binding.

CADGenBench calls the OpenCASCADE downcast helpers by their pybind static-method
spelling (``TopoDS.Face_s``), which is what the PyPI ``cadquery-ocp`` wheels
expose. This project installs OCP from conda-forge (``ocp 7.9.3.1``, via
``cadquery_ocp_proxy``), where the same helpers are bound without the ``_s``
suffix, so ``cadgenbench.eval.alignment`` raises ``AttributeError`` on the very
first tessellation.

Every other static call CADGenBench makes (``TopExp.MapShapes_s``,
``BRep_Tool.Triangulation_s``, ...) exists under both bindings; only the
``TopoDS`` downcasts differ. Aliasing those four names is therefore the whole
fix, and it is applied inside the isolated scoring subprocess only.
"""

from __future__ import annotations


_TOPODS_DOWNCASTS = ("Face", "Vertex", "Edge", "Shell", "Wire", "Solid", "Compound")


def ensure_ocp_static_aliases() -> None:
    """Give ``OCP.TopoDS.TopoDS`` the ``*_s`` aliases CADGenBench expects."""

    from OCP.TopoDS import TopoDS

    for name in _TOPODS_DOWNCASTS:
        alias = f"{name}_s"
        if not hasattr(TopoDS, alias) and hasattr(TopoDS, name):
            setattr(TopoDS, alias, getattr(TopoDS, name))


__all__ = ["ensure_ocp_static_aliases"]
