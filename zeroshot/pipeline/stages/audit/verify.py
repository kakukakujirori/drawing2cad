"""Validate an audit file and let its author review immutable evidence images."""

import json
from pathlib import Path, PurePosixPath
from typing import Any

from langchain_core.messages.content import ContentBlock, create_text_block
from pydantic import ValidationError

from zeroshot.pipeline.stages._base.error_locations import file_errors
from zeroshot.pipeline.stages._base.validate import SubmissionValidationError
from zeroshot.pipeline.stages.audit.contracts import AuditReport
from zeroshot.pipeline.stages.audit.validate import validate_audit_report
from zeroshot.pipeline.stages.coding.verify import unmatched_items
from zeroshot.pipeline.stages.contracts import ReconstructionSnapshot
from zeroshot.pipeline.verification import AttemptStore
from zeroshot.pipeline.workflow.evidence import EvidenceMode, render_evidence

_DISMISSAL_CHECK = (
    "[Dismissal check]\n"
    "These reviews dismiss extra items: {items}. An extra item is an edge or "
    "material the build draws but the input view lacks.\n"
    "Whether a view omits a line depends on the input drawing alone. An input "
    "drawing omits a line only when the 3D shape is clear without it, so a real "
    "omission leaves no doubt. Keep a dismissal only if the rest of the input "
    "drawing shows that this edge or material must exist, and the view only "
    "leaves it out. In the disposition, name what in the drawing supports this.\n"
    "Never infer an omission from the build or from trials. When no trial "
    "removes an extra item, the trials are wrong too, not the drawing.\n"
    "Without such evidence, report the item as a defect in a finding; the coder "
    "decides the correction."
)


def dismissal_check_text(items: list[str]) -> str:
    return _DISMISSAL_CHECK.format(items=", ".join(items))


def dismissed_extras(
    report: AuditReport, snapshot: ReconstructionSnapshot
) -> list[str]:
    """Reviews that dismiss an edge or material the input view does not show.

    An "extra" cluster lies beyond the comparison margin of every input line,
    or outside the input silhouette, so the input is blank where it is.
    """
    verification = snapshot.verification
    items: dict[str, dict[str, Any]] = unmatched_items(
        verification.drawing_diff_report if verification else None
    )
    return [
        key
        for key, review in report.concern_reviews.items()
        if review.finding_name is None
        and items.get(key, {}).get("direction") == "extra"
    ]


class AuditVerifier:
    def __init__(
        self,
        attempt_store: AttemptStore,
        source_filename: str = "audit.json",
        *,
        evidence_mode: EvidenceMode = "mark",
        require_drawing_diff_reviews: bool = False,
    ) -> None:
        source = PurePosixPath(source_filename)
        if source.is_absolute() or len(source.parts) != 1 or source.suffix != ".json":
            raise ValueError("source_filename must be a JSON file basename")
        self.attempt_store = attempt_store
        self.source_filename = source_filename
        self.evidence_mode = evidence_mode
        self.require_drawing_diff_reviews = require_drawing_diff_reviews
        self._snapshot: ReconstructionSnapshot | None = None
        self._checked: bytes | None = None
        self._accepted: AuditReport | None = None
        self._error: str | None = None
        self.evidence_renders: dict[str, list[str]] = {}

    @property
    def source_path(self) -> Path:
        return self.attempt_store.workdir.host_bind_dir / self.source_filename

    @property
    def snapshot(self) -> ReconstructionSnapshot | None:
        """The coding snapshot under audit."""
        return self._snapshot

    def reset(self, snapshot: ReconstructionSnapshot) -> None:
        if self.source_path.is_symlink():
            raise ValueError(f"{self.source_path} must not be a symlink")
        self.source_path.unlink(missing_ok=True)
        self._snapshot = snapshot
        self._checked = None
        self._accepted = None
        self._error = None
        self.evidence_renders = {}

    def _contents(self) -> bytes | None:
        path = self.source_path
        return path.read_bytes() if path.is_file() and not path.is_symlink() else None

    @property
    def confirmed(self) -> bool:
        return not self.blockers

    @property
    def blockers(self) -> list[str]:
        """Why the file as it is now cannot be the answer, by its latest verification."""
        verified = self._accepted is not None or self._error is not None
        if not verified or self._contents() != self._checked:
            return [f"{self.source_filename} has not been verified as it is now"]
        return [self._error] if self._error is not None else []

    @property
    def accepted_report(self) -> AuditReport | None:
        return self._accepted if self.confirmed else None

    @property
    def dismissed_extras(self) -> list[str]:
        """Extra items the verified audit.json dismisses without a finding."""
        if self._accepted is None or self._snapshot is None:
            return []
        return dismissed_extras(self._accepted, self._snapshot)

    def feedback(self) -> list[ContentBlock]:
        contents = self._contents()
        self._checked = contents
        self._accepted = None
        self.evidence_renders = {}
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
            validate_audit_report(
                report,
                self._snapshot,
                self.attempt_store,
                require_drawing_diff_reviews=self.require_drawing_diff_reviews,
            )
            renders = {
                finding.name: render_evidence(
                    finding,
                    directory / finding.name,
                    self.attempt_store.workdir,
                    mode=self.evidence_mode,
                )
                for finding in report.findings
            }
            self._accepted = report
            self.evidence_renders = renders
        except ValidationError as invalid:
            error = "\n".join(
                file_errors(invalid, self.source_filename, contents or b"")
            )
        except Exception as invalid:  # noqa: BLE001 - feed file/render failures back
            error = str(invalid)
        self._error = error
        (directory / "_audit_validation_log.json").write_text(
            json.dumps(
                {"error": error, "evidence_renders": self.evidence_renders}, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        text = f"[Audit verification]\nAttempt: {sandbox_directory}\n"
        if error is not None:
            text += f"{self.source_filename}: invalid.\n{error}"
        else:
            text += f"{self.source_filename}: valid.\n"
            for name, paths in self.evidence_renders.items():
                text += f"{name}: {', '.join(paths)}\n"
            if self.evidence_renders:
                text += "Check these evidence images with load_image against the findings and source drawing."
        blocks: list[ContentBlock] = [create_text_block(text)]
        if dismissed := self.dismissed_extras:
            blocks.append(create_text_block(dismissal_check_text(dismissed)))
        return blocks
