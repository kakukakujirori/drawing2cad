"""Validate an audit file and let its author review immutable evidence images."""

import json
from pathlib import Path, PurePosixPath

from langchain_core.messages.content import ContentBlock, create_text_block
from pydantic import ValidationError

from zeroshot.pipeline.stages._base.error_locations import file_errors
from zeroshot.pipeline.stages._base.validate import SubmissionValidationError
from zeroshot.pipeline.stages.audit.contracts import AuditReport
from zeroshot.pipeline.stages.audit.validate import validate_audit_report
from zeroshot.pipeline.stages.contracts import ReconstructionSnapshot
from zeroshot.pipeline.verification import AttemptStore
from zeroshot.pipeline.workflow.evidence import EvidenceMode, render_evidence


class AuditVerifier:
    def __init__(
        self,
        attempt_store: AttemptStore,
        source_filename: str = "audit.json",
        *,
        evidence_mode: EvidenceMode = "mark",
    ) -> None:
        source = PurePosixPath(source_filename)
        if source.is_absolute() or len(source.parts) != 1 or source.suffix != ".json":
            raise ValueError("source_filename must be a JSON file basename")
        self.attempt_store = attempt_store
        self.source_filename = source_filename
        self.evidence_mode = evidence_mode
        self._snapshot: ReconstructionSnapshot | None = None
        self._checked: bytes | None = None
        self._accepted: AuditReport | None = None
        self.evidence_crops: dict[str, list[str]] = {}

    @property
    def source_path(self) -> Path:
        return self.attempt_store.workdir.host_bind_dir / self.source_filename

    def reset(self, snapshot: ReconstructionSnapshot) -> None:
        if self.source_path.is_symlink():
            raise ValueError(f"{self.source_path} must not be a symlink")
        self.source_path.unlink(missing_ok=True)
        self._snapshot = snapshot
        self._checked = None
        self._accepted = None
        self.evidence_crops = {}

    def _contents(self) -> bytes | None:
        path = self.source_path
        return path.read_bytes() if path.is_file() and not path.is_symlink() else None

    @property
    def confirmed(self) -> bool:
        return self._accepted is not None and self._contents() == self._checked

    @property
    def accepted_report(self) -> AuditReport | None:
        return self._accepted if self.confirmed else None

    def feedback(self) -> list[ContentBlock]:
        contents = self._contents()
        self._checked = contents
        self._accepted = None
        self.evidence_crops = {}
        _, directory, sandbox_directory = self.attempt_store.issue("audit")
        if contents is not None:
            (directory / self.source_filename).write_bytes(contents)
        error = None
        try:
            if self._snapshot is None:
                raise SubmissionValidationError("the audit round is not prepared")
            if contents is None:
                raise SubmissionValidationError(
                    f"Write the complete report to {self.source_filename}"
                )
            report = AuditReport.model_validate_json(contents)
            validate_audit_report(report, self._snapshot, self.attempt_store)
            crops = {
                finding.name: render_evidence(
                    finding,
                    directory / finding.name,
                    self.attempt_store.workdir,
                    mode=self.evidence_mode,
                )
                for finding in report.findings
            }
            self._accepted = report
            self.evidence_crops = crops
        except ValidationError as invalid:
            error = "\n".join(
                file_errors(invalid, self.source_filename, contents or b"")
            )
        except Exception as invalid:  # noqa: BLE001 - feed file/render failures back
            error = str(invalid)
        (directory / "_audit_validation_log.json").write_text(
            json.dumps(
                {"error": error, "evidence_crops": self.evidence_crops}, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        text = f"[Audit verification]\nAttempt: {sandbox_directory}\n"
        if error is not None:
            text += f"{self.source_filename}: invalid.\n{error}"
        else:
            text += f"{self.source_filename}: valid.\n"
            for name, paths in self.evidence_crops.items():
                text += f"{name}: {', '.join(paths)}\n"
            if self.evidence_crops:
                text += "Check these evidence images with load_image against the findings and source drawing."
        return [create_text_block(text)]
