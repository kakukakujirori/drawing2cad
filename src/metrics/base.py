"""Protocol every evaluation metric follows.

Metrics are scored in two stages, mirroring how the evaluation pipeline is
already shaped:

``score``
    Runs once per sample, inside the isolated scoring subprocess, and returns
    the scalar columns this metric contributes to that sample's row.
``reduce``
    Runs once over every gathered row and returns the log-ready aggregates.

The split matters because the interesting aggregates are not plain means over
per-sample values: the ECCV summary multiplies a corpus-level valid ratio by a
mean taken over valid samples only, and the voxel IoU is averaged both with and
without failures. It also keeps metrics stateless, which is what makes them
safe to pickle across the ``spawn`` boundary and irrelevant to the rank
all-gather that :mod:`src.evaluation.generate` already performs.

Metric instances are frozen dataclasses holding their own parameters, so they
travel to the scoring worker by value. Heavy imports (OCC, cadgenbench,
trimesh) must happen inside ``score``, never at module import: this module is
imported by the training process, which must stay free of CAD kernels.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Final


#: Intermediate artifacts a metric can ask for. The evaluator produces exactly
#: the union of what the selected metrics require, so an unused artifact never
#: costs a tessellation or a subprocess.
PRED_MESH: Final = "pred_mesh"
GT_MESH: Final = "gt_mesh"
PRED_STEP: Final = "pred_step"
GT_STEP: Final = "gt_step"

ARTIFACTS: Final = frozenset({PRED_MESH, GT_MESH, PRED_STEP, GT_STEP})


class MetricSample:
    """Lazily loaded, memoized artifacts for one scored sample.

    Several metrics read the same mesh or STEP file, so each artifact is loaded
    at most once per sample. Accessors raise when the artifact was not produced,
    which can only happen if a metric reads something outside its ``requires``.
    """

    def __init__(
        self,
        sample_id: str,
        *,
        workdir: str | Path,
        pred_mesh_path: str | Path | None = None,
        gt_mesh_path: str | Path | None = None,
        pred_step_path: str | Path | None = None,
        gt_step_path: str | Path | None = None,
    ) -> None:
        self.sample_id = sample_id
        #: Scratch directory owned by this sample's evaluation. Metrics that
        #: must materialize intermediate files (a rescaled STEP, an aligned
        #: STEP) write them here, never next to the persisted prediction.
        self.workdir = Path(workdir)
        self._paths: dict[str, Path | None] = {
            PRED_MESH: None if pred_mesh_path is None else Path(pred_mesh_path),
            GT_MESH: None if gt_mesh_path is None else Path(gt_mesh_path),
            PRED_STEP: None if pred_step_path is None else Path(pred_step_path),
            GT_STEP: None if gt_step_path is None else Path(gt_step_path),
        }
        self._meshes: dict[str, Any] = {}

    def _path(self, artifact: str) -> Path:
        path = self._paths.get(artifact)
        if path is None:
            raise KeyError(
                f"artifact {artifact!r} was not produced for sample "
                f"{self.sample_id!r}; is it listed in the metric's `requires`?"
            )
        if not path.is_file():
            raise FileNotFoundError(f"missing {artifact} artifact: {path}")
        return path

    def pred_step(self) -> Path:
        return self._path(PRED_STEP)

    def gt_step(self) -> Path:
        return self._path(GT_STEP)

    def _mesh(self, artifact: str) -> Any:
        if artifact not in self._meshes:
            import numpy as np
            import trimesh

            path = self._path(artifact)
            with np.load(path, allow_pickle=False) as archive:
                vertices = np.asarray(archive["vertices"], dtype=np.float64)
                faces = np.asarray(archive["faces"], dtype=np.int64)
            self._meshes[artifact] = trimesh.Trimesh(
                vertices=vertices, faces=faces, process=False
            )
        return self._meshes[artifact]

    def pred_mesh(self) -> Any:
        return self._mesh(PRED_MESH)

    def gt_mesh(self) -> Any:
        return self._mesh(GT_MESH)


class CADMetric(ABC):
    """One family of evaluation metrics.

    A family is the unit of selection in ``evaluation.metrics``: it owns every
    column that shares a per-sample computation. Splitting a family further
    would either repeat that computation or push family-specific logic into the
    artifact cache.
    """

    #: Name written in the config. Set automatically by ``register_metric``.
    name: ClassVar[str] = ""
    #: Artifacts :meth:`score` may read.
    requires: ClassVar[frozenset[str]] = frozenset()
    #: Columns :meth:`score` writes into the per-sample row. Declared so the
    #: evaluator can build a failed row without running the metric, and so key
    #: collisions between families are caught when metrics are built.
    row_keys: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def score(self, sample: MetricSample) -> Mapping[str, Any]:
        """Return this metric's columns for one successfully executed sample."""

    @abstractmethod
    def reduce(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        prefix: str,
    ) -> Mapping[str, float | int]:
        """Aggregate every row (including failures) into log-ready scalars."""

    def empty_row(self) -> dict[str, Any]:
        """Columns for a sample this metric could not score."""

        return dict.fromkeys(self.row_keys)


def stem(prefix: str) -> str:
    """Return the log-key prefix, with the separator handled in one place."""

    return f"{prefix}/" if prefix else ""


__all__ = [
    "ARTIFACTS",
    "CADMetric",
    "GT_MESH",
    "GT_STEP",
    "MetricSample",
    "PRED_MESH",
    "PRED_STEP",
    "stem",
]
