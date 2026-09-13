"""Validate a complete interpretation and report its dimension calibration."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any

from langchain_core.messages.content import ContentBlock, create_text_block

from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.stages.drawings.contracts import DrawingSource
from zeroshot.pipeline.stages.interpretation.contracts import (
    DrawingInterpretation,
    View,
)
from zeroshot.pipeline.stages.interpretation.validate import validate_interpretation
from zeroshot.pipeline.verification.attempts import AttemptStore


@dataclass(frozen=True)
class InterpretationVerificationResult:
    attempt_id: str
    attempt_dir: PurePosixPath
    interpretation: DrawingInterpretation | None
    reports: dict[str, dict[str, Any]]
    errors: tuple[str, ...] = ()

    @property
    def confirmed(self) -> bool:
        return self.interpretation is not None and not self.errors


class InterpretationVerifier:
    def __init__(
        self,
        workdir: SandboxWorkdir,
        attempt_store: AttemptStore,
        input_artifact: DrawingSource,
        source_filename: str = "interpretation.json",
        *,
        dxf_mm_per_unit: Mapping[str, float] | None = None,
    ) -> None:
        source = PurePosixPath(source_filename)
        if source.is_absolute() or len(source.parts) != 1 or source.suffix != ".json":
            raise ValueError("source_filename must be a JSON file basename")
        self.workdir = workdir
        self.attempt_store = attempt_store
        self.source_filename = source_filename
        self.dxf_mm_per_unit = dxf_mm_per_unit
        self._original_files = {
            workdir.host_to_sandbox_path(sheet.file)
            for sheet in input_artifact.sheets
            if sheet.crop_of is None
        }
        self._built: InterpretationVerificationResult | None = None
        self._built_from_digest: str | None = None
        self._last_feedback_result: InterpretationVerificationResult | None = None

    @property
    def source_path(self) -> Path:
        return self.workdir.host_bind_dir / self.source_filename

    def reset(self, baseline: DrawingInterpretation | None) -> None:
        """Create the first artifact from scratch; seed revisions in full."""
        if self.source_path.is_symlink():
            raise ValueError(f"{self.source_filename} must not be a symlink")
        if baseline is None:
            self.source_path.unlink(missing_ok=True)
        else:
            self.source_path.write_text(
                baseline.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
        self._built = None
        self._built_from_digest = None
        self._last_feedback_result = None

    def _file_path(self, file: str) -> Path:
        path = self.workdir.sandbox_to_host_path(file)
        if path.is_symlink() or not path.resolve().is_relative_to(
            self.workdir.host_bind_dir.resolve()
        ):
            raise ValueError(f"{file} must stay inside the workspace without symlinks")
        if not path.is_file():
            raise ValueError(f"{file} is not a regular file")
        return path

    def source_digest(self) -> str | None:
        """Watch image contents too: fixing a crop need not change the JSON."""
        if self.source_path.is_symlink() or not self.source_path.is_file():
            return None
        payload = self.source_path.read_bytes()
        digest = sha256(payload)
        try:
            data = json.loads(payload)
        except (ValueError, UnicodeDecodeError):
            return digest.hexdigest()
        views = data.get("views", []) if isinstance(data, dict) else []
        if not isinstance(views, list):
            return digest.hexdigest()
        for view in views:
            if not isinstance(view, dict) or not isinstance(view.get("file"), str):
                continue
            try:
                fingerprint = sha256(
                    self._file_path(view["file"]).read_bytes()
                ).hexdigest()
            except (ValueError, OSError) as error:
                fingerprint = str(error)
            digest.update(json.dumps([view["file"], fingerprint]).encode("utf-8"))
        return digest.hexdigest()

    def verify(self) -> InterpretationVerificationResult:
        digest = self.source_digest()
        if self._built is not None and digest == self._built_from_digest:
            return self._built
        attempt_id, attempt_dir, sandbox_attempt_dir = self.attempt_store.issue(
            "interpretation"
        )
        interpretation = None
        reports: dict[str, dict[str, Any]] = {}
        errors: list[str] = []
        try:
            if self.source_path.is_symlink():
                raise ValueError(f"{self.source_filename} must not be a symlink")
            payload = self.source_path.read_bytes()
            (attempt_dir / "_interpretation_raw.json").write_bytes(payload)
            submitted = DrawingInterpretation.model_validate_json(payload)
            retained = {
                self.workdir.host_to_sandbox_path(self._file_path(view.file))
                for view in submitted.views
                if view.role == View.FULL_PAGE
            }
            missing = self._original_files - retained
            if missing:
                raise ValueError(
                    "Retain each original input file as a FULL_PAGE view: "
                    + ", ".join(map(str, sorted(missing)))
                )
            interpretation, reports = validate_interpretation(
                submitted,
                workdir=self.workdir,
                dxf_mm_per_unit=self.dxf_mm_per_unit,
            )
            enriched = interpretation.model_dump_json(indent=2) + "\n"
            (attempt_dir / self.source_filename).write_text(enriched, encoding="utf-8")
            # Replace only after validation and a complete write; preserve edits on failure.
            with TemporaryDirectory(dir=self.source_path.parent) as temporary:
                replacement = Path(temporary) / self.source_filename
                replacement.write_text(enriched, encoding="utf-8")
                replacement.replace(self.source_path)
        except Exception as error:  # noqa: BLE001 - return submission errors to the agent
            errors.append(f"{type(error).__name__}: {error}")
        (attempt_dir / "_interpretation_validation_log.json").write_text(
            json.dumps({"errors": errors, "reports": reports}, indent=2) + "\n",
            encoding="utf-8",
        )
        self._built = InterpretationVerificationResult(
            attempt_id=attempt_id,
            attempt_dir=sandbox_attempt_dir,
            interpretation=interpretation,
            reports=reports,
            errors=tuple(errors),
        )
        self._built_from_digest = self.source_digest()
        return self._built

    @property
    def confirmed(self) -> bool:
        result = self._last_feedback_result
        return (
            result is not None
            and result is self._built
            and result.confirmed
            and self.source_digest() == self._built_from_digest
        )

    @property
    def accepted_interpretation(self) -> DrawingInterpretation | None:
        result = self._last_feedback_result
        return result.interpretation if result is not None and self.confirmed else None

    def feedback(self) -> list[ContentBlock]:
        result = self.verify()
        self._last_feedback_result = result
        summaries = {}
        for name, report in result.reports.items():
            total = len(report.get("measurements", []))
            inliers = len(report.get("inliers", []))
            summaries[name] = {
                "status": report["status"],
                "scale": report.get("scale"),
                "inliers": f"{inliers}/{total}",
                "outliers": report.get("outliers", []),
            }
        return [
            create_text_block(
                f"[Interpretation verification]\n"
                f"{self.workdir.sandbox_bind_dir / self.source_filename}: "
                f"{'valid' if result.confirmed else 'invalid'}.\n"
                + json.dumps({"errors": result.errors, "reports": summaries})
            )
        ]
