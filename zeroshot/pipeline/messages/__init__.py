"""What the model sees: the artifacts a turn offers, and how they are presented.

`manifest` inventories the files a turn may show; `builder` turns an inventory
into content blocks. The split exists because a manifest holds *host* paths and
`MessageBuilder` is the only thing allowed to translate them into something the
model receives.
"""

from .builder import MessageBuilder
from .manifest import FeedbackManifest, InputManifest
from .prompts import PromptTemplate

__all__ = [
    "FeedbackManifest",
    "InputManifest",
    "MessageBuilder",
    "PromptTemplate",
]
