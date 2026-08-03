"""GT-aware metrics for a single reconstructed solid.

Each metric is a plain function taking two STEP paths and returning its
columns. There is no registry and no metric base class: the evaluator scores
one fixed set offline, so the only thing a registry would buy is indirection.

Every function here reads the ground truth, which is why this package sits
outside ``zeroshot/pipeline/`` -- nothing the agent process imports may reach
it.
"""

from .eccv import score_eccv
from .voxel import score_voxel

__all__ = ["score_eccv", "score_voxel"]
