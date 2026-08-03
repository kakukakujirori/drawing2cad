"""Verbatim forks of the ECCV 2026 CAD Challenge's own evaluator.

``_step_brep`` samples labelled B-Rep point clouds and incidence matrices out
of a STEP file; ``_matching`` assigns entities between two of them and scores
the F1. Together they *are* the metric -- their sampling densities, meshing
deflections, entity caps, distance threshold and matching rule are the
leaderboard's -- so they are forked as blackboxes and not refactored, tuned or
tidied. :mod:`zeroshot.evaluation.metrics.eccv` is the part written here.

Importing this package does not load a CAD kernel: ``OCC`` and ``trimesh`` are
imported inside the functions that need them, so the parent process can hold
these names and still hand the actual work to a child.
"""

from ._matching import (
    chamfer,
    match_entities,
    match_incidence,
    match_or_empty,
)
from ._step_brep import (
    load_step_brep,
    normalize_to_reference_bbox,
    reference_frame,
)

__all__ = [
    "chamfer",
    "load_step_brep",
    "match_entities",
    "match_incidence",
    "match_or_empty",
    "normalize_to_reference_bbox",
    "reference_frame",
]
