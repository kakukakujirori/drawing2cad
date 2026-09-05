"""What the model sees: the artifacts a turn offers, and how they are presented.

`manifest` inventories the files a turn may show; `artifact` turns an inventory
into content blocks. The split exists because a manifest holds *host* paths and
`ArtifactPresenter` is the only thing allowed to translate them into something the
model receives. `build_instruction` needs neither, which is why it is a
function rather than one of its methods.
"""

from .artifact import ArtifactPresenter
from .contracts import (
    FeatureGeometry,
    SemanticFeature,
    SemanticHypothesis,
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
    "FeatureGeometry",
    "FeedbackManifest",
    "InputManifest",
    "PromptTemplate",
    "SemanticFeature",
    "SemanticHypothesis",
    "build_instruction",
    "build_system_prompt",
    "instruction_section",
    "instruction_text",
    "system_prompt_text",
]
