"""Measured target-domain constants and data contracts shared across this package.

These are *measurements* of the drawing domain the renderer reproduces
(SolidWorks-generated 3-view technical drawings plus three perspective renders
per part, as in data/test_vlm), not tunable settings -- they are deliberately
not exposed as Hydra config.  Runtime knobs (timeouts, which render_3d styles to
expose as feedback) live in zeroshot/configs instead.

Constants used by exactly one module live in that module: the scale ladder and
inter-view gaps in arrange.py, the center-mark dimensions in export_dxf.py.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path

# Sheet: A4 landscape.  DXF model space is laid out in sheet-mm.  Everything
# below describes this sheet, so a change of paper size is a change to this
# block alone -- arrange.py owns only the composition on top of it (view gaps,
# scale ladder, degeneracy floor).
SHEET_W_MM = 297.0
SHEET_H_MM = 210.0
# The 3-view cluster is centred here (measured GT cluster centre ~= (145, 106)).
SHEET_CENTER_MM = (SHEET_W_MM / 2.0, SHEET_H_MM / 2.0)  # (148.5, 105.0)
# Usable envelope the cluster must fit inside (this drives scale selection).
# GT clusters reach ~240x201 mm; keep a small margin.
ENVELOPE_W_MM = 255.0
ENVELOPE_H_MM = 192.0
# How far outside the sheet a placed view may still land before it is treated as
# a runaway projection rather than a drawing.
SHEET_MARGIN_MM = 10.0

# render_3d target raster size (GT PNGs are 1400x1000 RGB, white background).
RENDER3D_SIZE = (1400, 1000)


class _OutputPaths:
    """Shared view over a DTO whose fields are one optional output path each."""

    def as_mapping(self) -> Mapping[str, Path]:
        """Field name -> path, dropping the fields that hold no path.

        Consumers key these artifacts by name rather than by field, because
        which ones a run produces is decided at runtime.  A ``None`` field
        means "not produced", so it is absent here rather than present with an
        empty value -- callers can iterate the result without re-checking.
        """
        return {
            path_field.name: path
            for path_field in fields(self)  # type: ignore[arg-type]
            if (path := getattr(self, path_field.name)) is not None
        }


@dataclass(frozen=True)
class TechdrawPaths(_OutputPaths):
    """One output file per technical-drawing format.

    Mirrors the dataset contract (data/test_vlm/techdraw/{svg,dxf,pdf}).  This
    renderer currently writes ``dxf`` only; ``svg`` and ``pdf`` are part of the
    contract but not produced.
    """

    svg: Path | None = None
    dxf: Path | None = None
    pdf: Path | None = None

    @classmethod
    def from_path(cls, base_path: Path, stem: str) -> TechdrawPaths:
        """Lay the outputs out the way the dataset does: one format, one stem."""
        return cls(
            # svg=base_path / "svg" / f"{stem}.svg",
            dxf=base_path / "dxf" / f"{stem}.dxf",
            # pdf=base_path / "pdf" / f"{stem}.pdf",
        )

    @classmethod
    def flat(cls, base_path: Path) -> TechdrawPaths:
        """Lay the outputs out the way the sandbox does: format as filename.

        The dataset holds many samples, so it needs a directory per format and
        a stem per sample. A verification attempt holds one drawing, so the
        same layout would spend two path components saying nothing. See
        `Render3dPaths.flat` for why the difference is worth removing.
        """
        return cls(dxf=base_path.with_name(f"{base_path.name}.dxf"))


@dataclass(frozen=True)
class Render3dPaths(_OutputPaths):
    """One output PNG per perspective style.

    Field name == style name == subdirectory name.  These are the style names
    already used by zeroshot/configs and by the manifests' ``render3d_paths``
    keys, so there is no separate selectable-style list to drift out of sync.
    _render3d.py reads these fields directly and was adapted to the names below.
    """

    hlg_perspective: Path | None = None
    transparent_shaded_edges_perspective: Path | None = None
    hlg_translucent_faces_perspective: Path | None = None

    @classmethod
    def from_path(cls, base_path: Path, stem: str) -> Render3dPaths:
        """Lay the outputs out the way the dataset does: one style, one stem."""
        return cls(
            hlg_perspective=base_path / "hlg_perspective" / f"{stem}.png",
            transparent_shaded_edges_perspective=base_path
            / "transparent_shaded_edges_perspective"
            / f"{stem}.png",
            hlg_translucent_faces_perspective=base_path
            / "hlg_translucent_faces_perspective"
            / f"{stem}.png",
        )

    @classmethod
    def flat(cls, base_path: Path) -> Render3dPaths:
        """Lay the outputs out the way the sandbox does: style as filename.

        The same three styles reach a model twice: as the drawing's own renders
        under `inputs/`, and as the renders of what it built. Two layouts for
        one set of names leaves the second path to be guessed from the first,
        and a guess that is wrong costs tool calls to find out. One convention
        inside the sandbox is what removes the guess.

        The dataset keeps its own layout: it holds many samples per style, so
        the directory and the stem both carry something there.
        """
        return cls(
            hlg_perspective=base_path / "hlg_perspective.png",
            transparent_shaded_edges_perspective=base_path
            / "transparent_shaded_edges_perspective.png",
            hlg_translucent_faces_perspective=base_path
            / "hlg_translucent_faces_perspective.png",
        )
