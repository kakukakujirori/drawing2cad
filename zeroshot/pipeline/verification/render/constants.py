"""Measured target-domain constants and data contracts shared across this package.

These are *measurements* of the drawing domain the renderer reproduces
(SolidWorks-generated 3-view technical drawings plus three perspective renders
per part, as in data/test_vlm), not tunable settings -- they are deliberately
not exposed as Hydra config.  Runtime knobs such as render timeouts live in
zeroshot/configs instead.

Constants used by exactly one module live in that module: the degeneracy floor
in project.py.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from pathlib import Path

from zeroshot.pipeline.messages.contracts.drawings import View

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
class ProjectionPaths(_OutputPaths):
    """One output DXF per orthographic view a drawing asked for.

    One field per view the contract names, so a request carries the views it
    wants by leaving the rest empty.  Field name == view name == the layer
    written inside == the role the sheet is announced under, so nothing has to
    translate between them.
    """

    front: Path | None = None
    back: Path | None = None
    top: Path | None = None
    bottom: Path | None = None
    left: Path | None = None
    right: Path | None = None

    @classmethod
    def flat(cls, base_path: Path, views: Iterable[View]) -> ProjectionPaths:
        """View as filename, the way the sandbox names the perspectives."""
        return cls(**{view.value: base_path / f"{view.value}.dxf" for view in views})


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
