"""What the model sees: the artifacts a turn offers, and how they are presented.

`manifest` inventories the files a turn may show; `builder` turns an inventory
into content blocks. The split exists because a manifest holds *host* paths and
`MessageBuilder` is the only thing allowed to translate them into something the
model receives. `build_instruction` needs neither, which is why it is a
function rather than one of its methods.
"""

from .builder import MessageBuilder, build_instruction, instruction_text
from .manifest import FeedbackManifest, InputManifest
from .prompts import PromptTemplate

__all__ = [
    "FeedbackManifest",
    "InputManifest",
    "MessageBuilder",
    "PromptTemplate",
    "build_instruction",
    "instruction_text",
]
