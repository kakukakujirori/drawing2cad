"""Validate and render a DrawingSource being edited in the workspace."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Literal

from langchain_core.messages.content import ContentBlock, create_text_block
from pydantic import ValidationError

from zeroshot.pipeline.drawing.dxf import export_sheet, rasterize_dxf
from zeroshot.pipeline.messages.artifact import (
    DrawingSource,
    FeedbackManifest,
    View,
    build_feedback_message_blocks,
)
from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.stages.drawings.contracts import (
    DrawingSheet,
    DrawnEntity,
    unread_sheet,
)
from zeroshot.pipeline.verification.attempts import AttemptStore

_BOUNDED_PARAMETERS = {
    DrawnEntity.LINE: ("start", "end"),
    DrawnEntity.ARC: ("start", "end"),
    DrawnEntity.CIRCLE: ("center",),
    DrawnEntity.ELLIPSE: ("start", "end"),
    DrawnEntity.POLYLINE: ("vertices",),
}


@dataclass(frozen=True)
class DrawingVerificationResult:
    """One immutable attempt to read and draw ``drawing.json``."""

    attempt_id: str
    attempt_dir: PurePosixPath
    drawing: DrawingSource | None
    errors: tuple[str, ...] = ()

    @property
    def confirmed(self) -> bool:
        """Whether parsing, structural checks, and rendering all succeeded."""
        return self.drawing is not None and not self.errors


class DrawingVerifier:
    """Turn the current drawing JSON into view files and visual feedback."""

    def __init__(
        self,
        workdir: SandboxWorkdir,
        attempt_store: AttemptStore,
        feedback_presentation_mode: Literal["none", "path", "image"],
        source_filename: str = "drawing.json",
    ) -> None:
        source = PurePosixPath(source_filename)
        if source.is_absolute() or len(source.parts) != 1 or source.suffix != ".json":
            raise ValueError("source_filename must be a JSON file basename")
        self.workdir = workdir
        self.attempt_store = attempt_store
        self.feedback_presentation_mode = feedback_presentation_mode
        self.source_filename = source_filename
        self._baseline: DrawingSource | None = None
        self._built: tuple[DrawingVerificationResult, FeedbackManifest] | None = None
        self._built_from_digest: str | None = None
        self._last_feedback_result: DrawingVerificationResult | None = None

    @property
    def source_path(self) -> Path:
        return self.workdir.host_bind_dir / self.source_filename

    def reset(self, baseline: DrawingSource) -> None:
        """Reset drawing work to a baseline selected by the workflow."""
        if self.source_path.is_symlink():
            raise ValueError(f"{self.source_filename} must not be a symlink")
        self.source_path.write_text(
            baseline.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        self._baseline = baseline
        self._built = None
        self._built_from_digest = None
        self._last_feedback_result = None

    def _source_digest(self) -> str | None:
        if self.source_path.is_symlink() or not self.source_path.is_file():
            return None
        return sha256(self.source_path.read_bytes()).hexdigest()

    def _validate_files(self, drawing: DrawingSource) -> list[str]:
        errors: list[str] = []
        for sheet in drawing.sheets:
            try:
                path = self.workdir.sandbox_to_host_path(sheet.file)
            except ValueError as error:
                errors.append(f"{sheet.name}: {error}")
                continue
            if path.is_symlink():
                errors.append(f"{sheet.name}: {sheet.file} must not be a symlink")
            elif not path.is_file():
                errors.append(f"{sheet.name}: {sheet.file} was not found")
            elif not path.resolve().is_relative_to(
                self.workdir.host_bind_dir.resolve()
            ):
                errors.append(f"{sheet.name}: {sheet.file} escapes the workspace")
        return errors

    def _validate_against_baseline(self, drawing: DrawingSource) -> list[str]:
        baseline = self._baseline
        if baseline is None:
            return ["the drawing round has not been prepared"]

        errors: list[str] = []
        current = {sheet.name: sheet for sheet in drawing.sheets}
        for original in (sheet for sheet in baseline.sheets if sheet.crop_of is None):
            held = current.get(original.name)
            if held is None:
                errors.append(f"the supplied sheet {original.name} must be retained")
            elif held.file != original.file:
                errors.append(
                    f"the supplied sheet {original.name} must keep file {original.file}"
                )
            elif held.role != original.role or held.crop_of != original.crop_of:
                errors.append(
                    f"the supplied sheet {original.name} must keep its role and origin"
                )
        return errors

    @staticmethod
    def _validate_sheet_bounds(sheet: DrawingSheet) -> list[str]:
        """Keep a cropped view's drawn coordinates inside its physical extent.

        Curve centres and spline controls may legitimately lie outside a
        clipped view, so only coordinates that locate drawn points are checked.
        """
        if sheet.crop_of is None:
            return []

        u0, v0, u1, v1 = sheet.crop_of.box
        width, height = u1 - u0, v1 - v0
        errors: list[str] = []

        for entry in sheet.evidence:
            parameters = {item.name.value: item.values for item in entry.parameters}
            for name in _BOUNDED_PARAMETERS.get(entry.entity, ()):
                values = parameters[name]
                for index, (u, v) in enumerate(
                    zip(values[0::2], values[1::2], strict=True)
                ):
                    if not (-1e-3 <= u <= width + 1e-3) or not (
                        -1e-3 <= v <= height + 1e-3
                    ):
                        member = name if len(values) == 2 else f"{name}[{index}]"
                        errors.append(
                            f"{sheet.name}: {entry.name}.{member} ({u:g}, {v:g}) "
                            f"is outside this cropped view's {width:g} x "
                            f"{height:g} mm local bounds"
                        )
        return errors

    @staticmethod
    def _view_filename(sheet_name: str, role: View) -> str:
        if role in {
            View.FRONT,
            View.BACK,
            View.TOP,
            View.BOTTOM,
            View.LEFT,
            View.RIGHT,
        }:
            return role.value
        return sheet_name.removeprefix("sheet_")

    def verify(
        self,
    ) -> tuple[DrawingVerificationResult, FeedbackManifest]:
        """Validate and render once for each distinct source in this round."""
        digest = self._source_digest()
        if self._built is not None and digest == self._built_from_digest:
            return self._built

        attempt_id, attempt_dir, sandbox_attempt_dir = self.attempt_store.issue(
            "drawing"
        )
        frozen_source = attempt_dir / self.source_filename
        errors: list[str] = []
        drawing: DrawingSource | None = None
        sheets = []

        if self.source_path.is_symlink():
            errors.append(f"{self.source_filename} must not be a symlink")
        elif not self.source_path.is_file():
            errors.append(f"{self.source_filename} was not found")
        else:
            payload_bytes = self.source_path.read_bytes()
            frozen_source.write_bytes(payload_bytes)
            try:
                payload = payload_bytes.decode("utf-8")
            except UnicodeDecodeError as error:
                errors.append(f"{self.source_filename} must be valid UTF-8: {error}")
            else:
                try:
                    drawing = DrawingSource.model_validate_json(payload)
                except ValidationError as error:
                    errors.append(str(error))

        if drawing is not None:
            errors.extend(self._validate_against_baseline(drawing))
            errors.extend(self._validate_files(drawing))
            for sheet in drawing.sheets:
                errors.extend(self._validate_sheet_bounds(sheet))
                if not sheet.evidence:
                    continue
                name = self._view_filename(sheet.name, sheet.role)
                dxf_path = attempt_dir / f"{name}.dxf"
                try:
                    export_sheet(sheet, dxf_path)
                    png_path = rasterize_dxf(dxf_path)
                    sheets.append(unread_sheet(sheet.name, sheet.role, png_path))
                except Exception as error:  # noqa: BLE001 - reported to the agent
                    reason = f"{type(error).__name__}: {error}"
                    errors.append(f"{sheet.name}: {reason}")
            if not sheets:
                errors.append("the drawing has no transcribed evidence to render")

        result = DrawingVerificationResult(
            attempt_id=attempt_id,
            attempt_dir=sandbox_attempt_dir,
            drawing=drawing,
            errors=tuple(errors),
        )
        manifest = FeedbackManifest(
            verification_id=attempt_id,
            drawing=DrawingSource(sheets=sheets) if sheets else None,
            errors={},
        )
        self._built = result, manifest
        self._built_from_digest = digest
        return result, manifest

    @property
    def confirmed(self) -> bool:
        result = self._last_feedback_result
        return result is not None and result.confirmed

    @property
    def accepted_drawing(self) -> DrawingSource | None:
        result = self._last_feedback_result
        return result.drawing if result is not None and result.confirmed else None

    def feedback(self) -> list[ContentBlock]:
        """Return validation failures or the generated views to the agent."""
        result, manifest = self.verify()
        self._last_feedback_result = result
        blocks: list[ContentBlock] = [
            create_text_block(
                f"{self.workdir.sandbox_bind_dir / self.source_filename} was "
                f"structurally checked in {result.attempt_dir}."
            )
        ]
        if result.errors:
            blocks.append(
                create_text_block(json.dumps({"errors": result.errors}, indent=2))
            )

        blocks.extend(
            build_feedback_message_blocks(
                manifest,
                self.workdir,
                mode=self.feedback_presentation_mode,
                heading="[Transcribed drawing]",
            )
        )
        return blocks
