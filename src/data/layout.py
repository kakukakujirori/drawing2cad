"""Dataset-root layout: where a sample's files live, and which samples exist.

A split (``train`` / ``val``) is configured as a list of dataset roots, each of
which is a self-contained staged corpus produced by
``src/data/render/render_dataset.py``. Every root must carry the same four
subdirectories; the layout is checked up front so a typo in a path fails at
composition time rather than after model loading.

This module is also the only place that maps a sample id onto files, and the
only place that decides whether a sample exists at all. That decision is an
intersection, not an assertion: rendering is expected to lose parts -- OCC HLR
hangs in native code on the complex tail and the batch driver can only SIGKILL
it -- so a sample counts as available once its technical-drawing DXF, every
configured raster, and (where supervision is required) its CadQuery target are
all on disk. Partial samples are reported by :func:`take_inventory` and left
out; they are never an error, because a lost render is a property of the corpus
rather than a misconfiguration.

The one axis roots are allowed to differ on is whether ``target/`` holds
CadQuery sources alongside the STEP answers. A root without them (an external
benchmark shipping STEP only) can still be scored geometrically -- generation
plus IoU/Chamfer needs the STEP alone -- but has no labels, so it cannot supply
supervised cross entropy. ``has_code_targets`` is what lets the callers route
each root to the evaluations it can actually support.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REQUIRED_SUBDIRS = ("render_3d", "target", "target_audit", "techdraw")
AUDIT_SUBDIR = "target_audit"
TARGET_SUBDIR = "target"
TECHDRAW_DXF_SUBDIR = Path("techdraw") / "dxf"
CODE_TARGET_SUFFIX = ".cadquery.py"


def code_target_path(root: str | Path, sample_id: str) -> Path:
    """The CadQuery source that supervises a sample, when the root ships one."""
    return Path(root) / TARGET_SUBDIR / f"{sample_id}{CODE_TARGET_SUFFIX}"


def take_inventory(
    root: str | Path,
    *,
    image_dirs: Sequence[str | Path] = (),
    require_target: bool = False,
) -> tuple[tuple[str, ...], dict[str, Path]]:
    """Partition a root's samples into the complete ones and the rest.

    Candidates come from ``techdraw/dxf/*.dxf``: the drawing is the one artifact
    every consumer needs, and a part the renderer could not project at all
    leaves no DXF behind. Each candidate must then carry one raster per entry of
    ``image_dirs`` (the configured ``image_sources``) and, when
    ``require_target`` is set, its CadQuery label.

    Returns the usable ids sorted (so every rank derives an identical corpus
    without communicating), plus each unusable id mapped to the first required
    file found absent -- enough for the caller to report what it skipped and
    why.
    """
    root = Path(root)
    directories = tuple(Path(directory) for directory in image_dirs)
    complete: list[str] = []
    incomplete: dict[str, Path] = {}
    for sample_id in sorted(
        path.stem for path in (root / TECHDRAW_DXF_SUBDIR).glob("*.dxf")
    ):
        required = [root / directory / f"{sample_id}.png" for directory in directories]
        if require_target:
            required.append(code_target_path(root, sample_id))
        missing = next((path for path in required if not path.is_file()), None)
        if missing is None:
            complete.append(sample_id)
        else:
            incomplete[sample_id] = missing
    return tuple(complete), incomplete


@dataclass(frozen=True)
class DatasetRoot:
    """One validated dataset root and the facts callers dispatch on."""

    name: str
    path: Path
    has_code_targets: bool

    @property
    def audit_dir(self) -> Path:
        """Where ``gt_audit.py --out-dir`` wrote this root's verdicts."""
        return self.path / AUDIT_SUBDIR


def _resolve_one(value: str | Path, *, split: str) -> DatasetRoot:
    path = Path(value).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"[{split}] dataset root does not exist: {path}")
    missing = [name for name in REQUIRED_SUBDIRS if not (path / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"[{split}] dataset root {path} is missing {missing}; every root must "
            f"provide {list(REQUIRED_SUBDIRS)}"
        )
    target_dir = path / TARGET_SUBDIR
    has_code_targets = any(target_dir.glob(f"*{CODE_TARGET_SUFFIX}"))
    return DatasetRoot(
        name=path.name,
        path=path,
        has_code_targets=has_code_targets,
    )


def resolve_dataset_roots(
    value: str | Path | Sequence[str | Path], *, split: str
) -> tuple[DatasetRoot, ...]:
    """Validate every configured root for ``split`` and name it by basename.

    A bare string is accepted as a one-element list so a single-root config (and
    existing callers) keep working. Names come from the directory basename and
    must be unique, because they namespace metric keys and prediction
    directories -- two roots sharing a basename would silently collide there.
    """
    if isinstance(value, (str, Path)):
        values: list[str | Path] = [value]
    else:
        values = list(value)
    if not values:
        raise ValueError(f"[{split}] at least one dataset root is required")
    roots = tuple(_resolve_one(item, split=split) for item in values)
    names = [root.name for root in roots]
    if len(set(names)) != len(names):
        raise ValueError(
            f"[{split}] dataset root basenames must be unique (they namespace "
            f"metrics and prediction directories), got {names}"
        )
    return roots


__all__ = [
    "AUDIT_SUBDIR",
    "CODE_TARGET_SUFFIX",
    "DatasetRoot",
    "REQUIRED_SUBDIRS",
    "TARGET_SUBDIR",
    "TECHDRAW_DXF_SUBDIR",
    "code_target_path",
    "resolve_dataset_roots",
    "take_inventory",
]
