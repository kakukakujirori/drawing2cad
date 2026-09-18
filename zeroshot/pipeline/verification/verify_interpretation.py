"""Validate a complete interpretation and report its dimension calibration."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any

from langchain_core.messages.content import ContentBlock, create_text_block
from pydantic import ValidationError

from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.stages._base.validate import LocatedError
from zeroshot.pipeline.stages.interpretation.contracts import (
    ORTHOGRAPHIC_VIEWS,
    PICTORIAL_VIEWS,
    DrawingInterpretation,
    DrawingView,
    View,
)
from zeroshot.pipeline.stages.interpretation.validate import validate_interpretation
from zeroshot.pipeline.verification.attempts import AttemptStore
from zeroshot.pipeline.verification.error_locations import file_errors, semantic_errors


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
        input_artifact: Sequence[DrawingView],
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
        self._original_views = [
            view.model_copy(
                update={"file": str(workdir.host_to_sandbox_path(view.file))},
                deep=True,
            )
            for view in input_artifact
        ]
        self._built: InterpretationVerificationResult | None = None
        self._built_from_digest: str | None = None
        self._last_feedback_result: InterpretationVerificationResult | None = None

    @property
    def source_path(self) -> Path:
        return self.workdir.host_bind_dir / self.source_filename

    def reset(self, baseline: DrawingInterpretation) -> None:
        """Seed the artifact in full: the input registered, or last round's answer."""
        if self.source_path.is_symlink():
            raise ValueError(f"{self.source_filename} must not be a symlink")
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
        except (ValueError, UnicodeDecodeError, RecursionError):
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
        validated_interpretation = None
        reports: dict[str, dict[str, Any]] = {}
        errors: list[str] = []
        payload = b""
        try:
            if self.source_path.is_symlink():
                raise ValueError(f"{self.source_filename} must not be a symlink")
            payload = self.source_path.read_bytes()
            (attempt_dir / "_interpretation_raw.json").write_bytes(payload)
            submitted_interpretation = DrawingInterpretation.model_validate_json(
                payload
            )
            by_name = {
                view.name: (index, view)
                for index, view in enumerate(submitted_interpretation.views)
            }
            for original in self._original_views:
                index, view = by_name.get(original.name, (None, None))
                if view is None and original.role in PICTORIAL_VIEWS:
                    continue
                if (
                    view is None
                    or self._file_path(view.file) != self._file_path(original.file)
                    or view.role != original.role
                    or not view.region.matches_bounds(original.region)
                ):
                    raise LocatedError.at(
                        ("views",) if index is None else ("views", index),
                        "Retain each original input file with its registered name, "
                        f"role and full-file region: {original.name} ({original.role}), "
                        f"{original.file}. Dimensions may be added.",
                    )
            if any(
                original.role is View.FULL_PAGE for original in self._original_views
            ) and not any(
                view.role in ORTHOGRAPHIC_VIEWS
                for view in submitted_interpretation.views
            ):
                raise LocatedError.at(
                    ("views",),
                    "A full_page input is an unsplit page. Add at least one DrawingView "
                    "with an orthographic role (front, back, top, bottom, left or right). "
                    "For a single-view drawing only showing the front or the top, "
                    "add a new DrawingView that uses the same file with a full-file region.",
                )
            validated_interpretation, reports = validate_interpretation(
                submitted_interpretation,
                workdir=self.workdir,
                dxf_mm_per_unit=self.dxf_mm_per_unit,
            )
            enriched = validated_interpretation.model_dump_json(indent=2) + "\n"
            (attempt_dir / self.source_filename).write_text(enriched, encoding="utf-8")
            # Replace only after validation and a complete write; preserve edits on failure.
            with TemporaryDirectory(dir=self.source_path.parent) as temporary:
                replacement = Path(temporary) / self.source_filename
                replacement.write_text(enriched, encoding="utf-8")
                replacement.replace(self.source_path)
        except ValidationError as error:
            errors.extend(file_errors(error, self.source_filename, payload))
        except Exception as error:  # noqa: BLE001 - return submission errors to the agent
            errors.extend(semantic_errors(error, self.source_filename, payload))
        (attempt_dir / "_interpretation_validation_log.json").write_text(
            json.dumps({"errors": errors, "reports": reports}, indent=2) + "\n",
            encoding="utf-8",
        )
        self._built = InterpretationVerificationResult(
            attempt_id=attempt_id,
            attempt_dir=sandbox_attempt_dir,
            interpretation=validated_interpretation,
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
