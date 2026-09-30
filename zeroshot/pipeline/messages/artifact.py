from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
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
from zeroshot.pipeline.stages.interpretation.contracts import (
    PICTORIAL_VIEWS,
    DrawingView,
    View,
)

_MIME_TYPES: Mapping[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


def drawing_for_model(
    drawing: Sequence[DrawingView], workdir: SandboxWorkdir
) -> list[DrawingView]:
    """The same views with every file addressed where the model can open it.

    Views the model then reads, answers about, and hands back stay in those
    addresses, so nothing carries a host path into a message by mistake.
    """
    return [
        view.model_copy(update={"file": str(workdir.host_to_sandbox_path(view.file))})
        for view in drawing
    ]


@dataclass(frozen=True)
class _SandboxSheet:
    """A `DrawingView` addressed where the model can open it.

    Nothing host-side is kept, so a host path cannot reach a message by being
    to hand.
    """

    name: str
    role: View
    file: PurePosixPath

    @classmethod
    def of(cls, sheet: DrawingView, file: PurePosixPath) -> _SandboxSheet:
        return cls(
            name=sheet.name,
            role=sheet.role,
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
    def of(cls, sheets: Sequence[DrawingView], workdir: SandboxWorkdir) -> Self:
        """Every sheet the drawing holds, not only the ones a stage reads.

        The audit is offered the unsplit sheet whatever the reasoning stages
        worked from, so a split that went wrong is still there to point at.
        """
        found = []
        for sheet in sheets:
            held = Path(sheet.file)
            _check_reachable(held, workdir, sheet.name)
            found.append(_SandboxSheet.of(sheet, workdir.host_to_sandbox_path(held)))
        return cls(found, workdir)

    def read(self) -> list[_SandboxSheet]:
        """The sheets a stage reads, which is every one that is not a pictorial."""
        return [sheet for sheet in self.sheets if sheet.role not in PICTORIAL_VIEWS]

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

    input: Literal["path", "image"]
    output_renders: Literal["path", "image"] = "path"
    overlay: Literal["none", "path", "image"] = "path"
    unmatched: Literal["path", "image"] = "path"
    intermediates: Literal["none", "path", "image"] = "path"

    def __post_init__(self) -> None:
        if self.input not in {"path", "image"}:
            raise ValueError(f"invalid input: {self.input!r}")
        for name in ("overlay", "intermediates"):
            if getattr(self, name) not in {"none", "path", "image"}:
                raise ValueError(f"invalid {name}: {getattr(self, name)!r}")
        for name in ("output_renders", "unmatched"):
            if getattr(self, name) not in {"path", "image"}:
                raise ValueError(f"invalid {name}: {getattr(self, name)!r}")


def build_feedback_message_blocks(
    artifacts: FeedbackManifest | Mapping[str, Path],
    workdir: SandboxWorkdir,
    *,
    mode: Literal["none", "path", "image"],
    heading: str = "[Projected drawing]",
) -> list[ContentBlock]:
    """Present a description followed by named files and optional images.

    Render manifests include view roles and failures; annotated comparison
    images only need names and paths.

    Blocks rather than a message, so the caller decides what carries them.
    A verification never becomes a turn anyone spoke.
    """
    if mode not in {"none", "path", "image"}:
        raise ValueError(f"invalid feedback mode: {mode!r}")

    if mode == "none":
        return []

    paths: Mapping[str, Path]
    if isinstance(artifacts, FeedbackManifest):
        presented = SandboxedArtifact.of(artifacts.drawing, workdir)
        failed = [
            f"- {name}: unavailable ({why})" for name, why in artifacts.errors.items()
        ]
        if not presented.sheets and not failed:
            return []
        lines = [heading, *presented.listing(), *failed, ""]
        blocks: list[ContentBlock] = [create_text_block("\n".join(lines))]
        if mode == "image":
            blocks.extend(presented.images())
        paths = {
            f"{sheet.name} PNG": png
            for sheet in artifacts.drawing
            if Path(sheet.file).suffix.lower() == ".dxf"
            and (png := Path(sheet.file).with_suffix(".png")).is_file()
        }
    else:
        if not artifacts:
            return []
        blocks = [create_text_block(heading)]
        paths = artifacts

    for name, path in paths.items():
        _check_reachable(path, workdir, name)
        blocks.append(
            create_text_block(f"{name}: {workdir.host_to_sandbox_path(path)}")
        )
        if mode == "image":
            blocks.append(
                create_image_block(
                    base64=base64.b64encode(path.read_bytes()).decode("ascii"),
                    mime_type=_MIME_TYPES[path.suffix.lower()],
                )
            )
    return blocks
