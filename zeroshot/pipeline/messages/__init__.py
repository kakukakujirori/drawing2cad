"""What the model sees: the artifacts a turn offers, and how they are presented.

`manifest` inventories the files a turn may show; `artifact` turns an inventory
into content blocks. The split exists because a manifest holds *host* paths and
`artifact` is the only thing allowed to translate them into something the model
receives, and nothing translates back: what the model is shown it also answers
in, and the run keeps that.
"""

from .artifact import ArtifactPresenter, drawing_for_model
from .contracts import (
    DrawingSource,
    FeatureGeometry,
    SemanticFeature,
    SemanticHypothesis,
    View,
    unread_sheet,
)
from .manifest import FeedbackManifest, InputManifest

__all__ = [
    "ArtifactPresenter",
    "DrawingSource",
    "FeatureGeometry",
    "FeedbackManifest",
    "InputManifest",
    "SemanticFeature",
    "SemanticHypothesis",
    "View",
    "drawing_for_model",
    "unread_sheet",
]
