from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from langchain_core.messages.content import (
    ContentBlock,
    create_image_block,
    create_text_block,
)

from zeroshot.pipeline.messages.contracts import (
    DrawingSheet,
    DrawingSource,
    View,
)
from zeroshot.pipeline.messages.manifest import FeedbackManifest, InputManifest
from zeroshot.pipeline.sandbox import SandboxWorkdir

_MIME_TYPES: Mapping[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


@dataclass(frozen=True)
class _SandboxSheet:
    """A `DrawingSheet` whose file is addressed where the model can open it.

    The same fields as the sheet it came from, so nothing is dropped on the
    way into a message; `file` is the one that changes, and it changes type.
    Nothing host-side is kept, so a host path cannot reach a message by being
    to hand.
    """

    role: View
    label: str | None
    detail_of: View | None
    file: PurePosixPath

    @classmethod
    def of(cls, sheet: DrawingSheet, file: PurePosixPath) -> _SandboxSheet:
        return cls(
            role=sheet.role,
            label=sheet.label,
            detail_of=sheet.detail_of,
            file=file,
        )

    @property
    def name(self) -> str:
        """What a message calls it: its label, or its role when it has none."""
        return self.label or self.role.value

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
class _Presented:
    """A drawing as a message will show it: the files, named and addressed."""

    sheets: list[_SandboxSheet]
    workdir: SandboxWorkdir

    @classmethod
    def of(cls, drawing: DrawingSource, workdir: SandboxWorkdir) -> _Presented:
        """Every sheet the drawing holds, not only the ones a stage reads.

        The audit is offered the unsplit sheet whatever the reasoning stages
        worked from, so a split that went wrong is still there to point at.
        """
        found = []
        for sheet in drawing.sheets:
            _check_reachable(sheet.file, workdir, sheet.name)
            found.append(
                _SandboxSheet.of(sheet, workdir.host_to_sandbox_path(sheet.file))
            )
        return cls(found, workdir)

    def listing(self) -> list[str]:
        """One line per file, plus what an unsplit sheet needs saying about it.

        Keyed on the role, so a drawing that was separated into views and one
        that was not are announced the same way.
        """
        described = [f"- {sheet.name}: {sheet.file}" for sheet in self.sheets]
        if any(sheet.role is View.UNKNOWN for sheet in self.sheets):
            described.append(
                "  A sheet named for no view of its own carries every view at "
                "once. They are not separated by layer or by file: tell them "
                "apart by where they sit on the sheet."
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
        return any(sheet.is_raster for sheet in self.sheets)

    def has_vector(self) -> bool:
        return any(not sheet.is_raster for sheet in self.sheets)


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

    def build_input_message_blocks(
        self,
        manifest: InputManifest,
        workdir: SandboxWorkdir,
    ) -> list[ContentBlock]:
        presented = _Presented.of(manifest.drawing, workdir)
        lines = ["[Input drawing]", *presented.listing()]
        lines.append(
            "The coordinate system of the 2D drawing is as follows: "
            f"{manifest.drawing.frame_sentence()}."
        )
        # What the format affords, said where the format is known. A stage's
        # guidelines describe the job; only this message knows whether the job
        # is done by reading a file or by measuring an image.
        if presented.has_raster():
            lines.append(
                "Sheets given as images carry no curve definitions. Measure "
                "them with code -- OpenCV, numpy -- rather than by eye, and "
                "convert what you measure into the drawing's own units before "
                "reporting it."
            )
        if presented.has_vector():
            lines.append(
                "Sheets given as DXF state every curve outright. Read the "
                "definitions with a library such as `ezdxf` and carry the "
                "numbers across unchanged; linetype is what tells you whether "
                "an edge is visible or hidden."
            )
        lines.append("")

        blocks: list[ContentBlock] = [create_text_block("\n".join(lines))]
        if self.input_mode == "image":
            blocks.extend(presented.images())
        return blocks

    def build_feedback_message_blocks(
        self,
        manifest: FeedbackManifest,
        workdir: SandboxWorkdir,
    ) -> list[ContentBlock]:
        """What a verification drew of the solid it built.

        Blocks rather than a message, so the caller decides what carries them.
        A verification never becomes a turn anyone spoke.
        """
        if self.feedback_mode == "none":
            return []

        presented = (
            _Presented.of(manifest.drawing, workdir)
            if manifest.drawing is not None
            else _Presented([], workdir)
        )
        failed = [
            f"- {name}: unavailable ({why})" for name, why in manifest.errors.items()
        ]
        if not presented.sheets and not failed:
            return []

        lines = ["[Projected drawing]", *presented.listing(), *failed, ""]
        blocks: list[ContentBlock] = [create_text_block("\n".join(lines))]
        if self.feedback_mode == "image":
            blocks.extend(presented.images())
        return blocks
