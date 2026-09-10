from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from langchain_core.messages.content import (
    ContentBlock,
    create_image_block,
    create_text_block,
)

from zeroshot.pipeline.messages.manifest import FeedbackManifest
from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.stages.drawings.contracts import (
    DrawingSheet,
    DrawingSource,
    View,
)

# A pictorial is offered for context, never read: it fixes no axes, so nothing
# lifts a coordinate from one.
_PICTORIAL = frozenset({View.PERSPECTIVE, View.ISOMETRIC})

_MIME_TYPES: Mapping[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


def drawing_for_model(drawing: DrawingSource, workdir: SandboxWorkdir) -> DrawingSource:
    """The same drawing with every file addressed where the model can open it.

    A drawing the model then reads, answers about, and hands back stays in
    those addresses, so nothing carries a host path into a message by mistake.
    """
    return drawing.model_copy(
        update={
            "sheets": [
                sheet.model_copy(
                    update={"file": str(workdir.host_to_sandbox_path(sheet.file))}
                )
                for sheet in drawing.sheets
            ]
        }
    )


@dataclass(frozen=True)
class _SandboxSheet:
    """A `DrawingSheet` whose file is addressed where the model can open it.

    Nothing host-side is kept, so a host path cannot reach a message by being
    to hand.
    """

    name: str
    role: View
    label: str | None
    file: PurePosixPath

    @classmethod
    def of(cls, sheet: DrawingSheet, file: PurePosixPath) -> _SandboxSheet:
        return cls(
            name=sheet.name,
            role=sheet.role,
            label=sheet.label,
            file=file,
        )

    @property
    def mime_type(self) -> str | None:
        """What the file is as an image, or `None` when it is not one."""
        return _MIME_TYPES.get(self.file.suffix.lower())

    @property
    def is_raster(self) -> bool:
        return self.mime_type is not None


def _check_reachable(path: Path, workdir: SandboxWorkdir, key: str) -> None:
    if path.is_symlink():
        raise ValueError(f"{path} must not be a symlink")
    if not path.is_absolute():
        raise ValueError(f"{path} must be absolute")
    if not path.is_relative_to(workdir.host_bind_dir):
        raise ValueError(f"{path} must be under {workdir.host_bind_dir}")
    if not path.is_file():
        raise FileNotFoundError(f"{key} not found: {path}")


@dataclass(frozen=True)
class SandboxedArtifact:
    """A drawing as a message will show it: the files, named and addressed."""

    sheets: list[_SandboxSheet]
    workdir: SandboxWorkdir

    @classmethod
    def of(cls, drawing: DrawingSource, workdir: SandboxWorkdir) -> Self:
        """Every sheet the drawing holds, not only the ones a stage reads.

        The audit is offered the unsplit sheet whatever the reasoning stages
        worked from, so a split that went wrong is still there to point at.
        """
        found = []
        for sheet in drawing.sheets:
            held = Path(sheet.file)
            _check_reachable(held, workdir, sheet.name)
            found.append(_SandboxSheet.of(sheet, workdir.host_to_sandbox_path(held)))
        return cls(found, workdir)

    def read(self) -> list[_SandboxSheet]:
        """The sheets a stage reads, which is every one that is not a pictorial."""
        return [sheet for sheet in self.sheets if sheet.role not in _PICTORIAL]

    def listing(self) -> list[str]:
        described = [
            f"- {sheet.name} ({sheet.role.value}): {sheet.file}"
            for sheet in self.sheets
        ]
        if any(sheet.role is View.FULL_PAGE for sheet in self.sheets):
            described.append(
                "  A full page carries every view at once. They are not "
                "separated by layer or by file: tell them apart by where they "
                "sit on the page."
            )
        return described

    def images(self) -> list[ContentBlock]:
        """The rasters, attached. A vector sheet has no pixels to send.

        The bytes are fetched here rather than held: nothing is read for a
        message that only names its files.
        """
        blocks: list[ContentBlock] = []
        for sheet in self.sheets:
            if sheet.mime_type is None:
                continue
            data = self.workdir.sandbox_to_host_path(sheet.file).read_bytes()
            blocks.append(create_text_block(f"- {sheet.name}: {sheet.file}"))
            blocks.append(
                create_image_block(
                    base64=base64.b64encode(data).decode("ascii"),
                    mime_type=sheet.mime_type,
                )
            )
        return blocks

    def has_raster(self) -> bool:
        return any(sheet.is_raster for sheet in self.read())

    def has_vector(self) -> bool:
        return any(not sheet.is_raster for sheet in self.read())


@dataclass(frozen=True)
class ArtifactPresenter:
    """How the run's files are announced to whichever agent is addressed.

    What the run offers, not what it asks: the system prompt belongs to the
    agent, and one presenter is shared by all of them.
    """

    input_mode: Literal["path", "image"]
    feedback_mode: Literal["none", "path", "image"]

    def __post_init__(self) -> None:
        if self.input_mode not in {"path", "image"}:
            raise ValueError(f"invalid input_mode: {self.input_mode!r}")
        if self.feedback_mode not in {"none", "path", "image"}:
            raise ValueError(f"invalid feedback_mode: {self.feedback_mode!r}")


def build_feedback_message_blocks(
    manifest: FeedbackManifest,
    workdir: SandboxWorkdir,
    *,
    mode: Literal["none", "path", "image"],
    heading: str = "[Projected drawing]",
) -> list[ContentBlock]:
    """Present the views produced by an artifact verification.

    Blocks rather than a message, so the caller decides what carries them.
    A verification never becomes a turn anyone spoke.
    """
    if mode not in {"none", "path", "image"}:
        raise ValueError(f"invalid feedback mode: {mode!r}")

    if mode == "none":
        return []

    presented = (
        SandboxedArtifact.of(manifest.drawing, workdir)
        if manifest.drawing is not None
        else SandboxedArtifact([], workdir)
    )
    failed = [f"- {name}: unavailable ({why})" for name, why in manifest.errors.items()]
    if not presented.sheets and not failed:
        return []

    lines = [heading, *presented.listing(), *failed, ""]
    blocks: list[ContentBlock] = [create_text_block("\n".join(lines))]
    if mode == "image":
        blocks.extend(presented.images())
    return blocks
