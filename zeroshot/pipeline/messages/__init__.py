"""What the model sees: the artifacts a turn offers, and how they are presented.

`manifest` inventories the files a turn may show; `artifact` turns an inventory
into content blocks. The split exists because a manifest holds *host* paths and
`artifact` is the only thing allowed to translate them into something the model
receives, and nothing translates back: what the model is shown it also answers
in, and the run keeps that. `build_instruction` needs neither, which is why it
is a function rather than one of its methods.
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
from .prompt import (
    build_instruction,
    build_system_prompt,
    instruction_section,
    instruction_text,
    system_prompt_text,
)
from .prompts import PromptTemplate

__all__ = [
    "ArtifactPresenter",
    "DrawingSource",
    "FeatureGeometry",
    "FeedbackManifest",
    "InputManifest",
    "PromptTemplate",
    "SemanticFeature",
    "SemanticHypothesis",
    "View",
    "build_instruction",
    "build_system_prompt",
    "drawing_for_model",
    "instruction_section",
    "instruction_text",
    "system_prompt_text",
    "unread_sheet",
]
