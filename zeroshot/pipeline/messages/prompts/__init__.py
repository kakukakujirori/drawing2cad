"""System prompts, kept as files because they are what experiments vary.

A prompt in Python would have to escape every brace a CadQuery snippet contains,
and a diff of one would be a diff of source. These are `.md` beside this module,
addressed by name.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from string import Template

_PROMPT_DIR = Path(__file__).parent


@dataclass(frozen=True)
class PromptTemplate:
    """A named prompt file, and the values it still needs to be rendered.

    `name` resolves against this package rather than the working directory:
    Hydra moves the process into the job's output directory before a run
    starts, so a relative path written in a config would not survive it. An
    absolute path is taken as given, which is enough to try a prompt kept
    outside the repository without adding a search-path setting.
    """

    name: str

    def __post_init__(self) -> None:
        # A missing prompt is a configuration error, so it fails while the
        # graph is being built rather than partway through a run.
        if not self.path.is_file():
            raise ValueError(f"prompt not found: {self.path}")

    @property
    def path(self) -> Path:
        candidate = Path(self.name)
        return candidate if candidate.is_absolute() else _PROMPT_DIR / f"{self.name}.md"

    @property
    def sha256(self) -> str:
        return sha256(self.path.read_bytes()).hexdigest()

    def render(self, **context: str) -> str:
        """Fill the `$name` placeholders, refusing to leave any unfilled.

        `$` rather than `{}` so a prompt can hold code without escaping, and
        `substitute` rather than `safe_substitute` so a value we forgot to pass
        raises here instead of reaching the model as the literal `$output_path`.
        Surrounding whitespace is dropped so that whether the file ends in a
        newline -- an editor's decision -- cannot change what the model is sent.
        """
        return (
            Template(self.path.read_text(encoding="utf-8")).substitute(context).strip()
        )
