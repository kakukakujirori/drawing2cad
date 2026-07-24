"""Resolve and validate the on-disk layout of one or more dataset roots.

A split (``train`` / ``val``) is configured as a list of dataset roots, each of
which is a self-contained staged corpus produced by
``src/data/render/render_dataset.py``. Every root must carry the same four
subdirectories plus a manifest; the layout is checked up front so a typo in a
path fails at composition time rather than after model loading.

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
MANIFEST_NAME = "manifest.jsonl"
AUDIT_SUBDIR = "target_audit"
TARGET_SUBDIR = "target"
CODE_TARGET_SUFFIX = ".cadquery.py"


@dataclass(frozen=True)
class DatasetRoot:
    """One validated dataset root and the facts callers dispatch on."""

    name: str
    path: Path
    audit_dir: Path
    has_code_targets: bool


def _resolve_one(value: str | Path, *, split: str) -> DatasetRoot:
    path = Path(value).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"[{split}] dataset root does not exist: {path}")
    missing = [name for name in REQUIRED_SUBDIRS if not (path / name).is_dir()]
    if not (path / MANIFEST_NAME).is_file():
        missing.append(MANIFEST_NAME)
    if missing:
        raise FileNotFoundError(
            f"[{split}] dataset root {path} is missing {missing}; every root must "
            f"provide {list(REQUIRED_SUBDIRS)} and {MANIFEST_NAME}"
        )
    target_dir = path / TARGET_SUBDIR
    has_code_targets = any(target_dir.glob(f"*{CODE_TARGET_SUFFIX}"))
    return DatasetRoot(
        name=path.name,
        path=path,
        audit_dir=path / AUDIT_SUBDIR,
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
    "MANIFEST_NAME",
    "REQUIRED_SUBDIRS",
    "TARGET_SUBDIR",
    "resolve_dataset_roots",
]
