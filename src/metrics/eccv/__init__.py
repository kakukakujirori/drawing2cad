"""ECCV 2026 CAD Challenge metric family.

The pythonocc-dependent sampling lives in :mod:`._step_brep` and is imported
only inside the metric's ``score``, so importing this package from the training
process stays free of CAD kernels.
"""

from .metric import ECCVChallengeMetric

__all__ = ["ECCVChallengeMetric"]
