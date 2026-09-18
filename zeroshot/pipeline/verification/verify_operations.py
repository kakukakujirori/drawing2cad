"""Validate the complete operation plan a stage writes to its workspace file."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

from langchain_core.messages.content import ContentBlock, create_text_block
from pydantic import ValidationError

from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.stages._base.validate import SubmissionValidationError
from zeroshot.pipeline.stages.interpretation.contracts import DrawingInterpretation
from zeroshot.pipeline.stages.operations.contracts import OperationPlan
from zeroshot.pipeline.stages.operations.validate import validate_operations
from zeroshot.pipeline.verification.attempts import AttemptStore
from zeroshot.pipeline.verification.error_locations import file_errors, semantic_errors


class OperationPlanVerifier:
    """Parse the workspace plan and check it against the round's interpretation.

    The stage answers with ticket responses alone, so this file is the only
    place the plan itself exists until integration adopts it.
    """

    def __init__(
        self,
        workdir: SandboxWorkdir,
        attempt_store: AttemptStore,
        source_filename: str = "operations.json",
    ) -> None:
        source = PurePosixPath(source_filename)
        if source.is_absolute() or len(source.parts) != 1 or source.suffix != ".json":
            raise ValueError("source_filename must be a JSON file basename")
        self.workdir = workdir
        self.attempt_store = attempt_store
        self.source_filename = source_filename
        self._interpretation: DrawingInterpretation | None = None
        self._accepted: OperationPlan | None = None
        self._checked: bytes | None = None

    @property
    def source_path(self) -> Path:
        return self.workdir.host_bind_dir / self.source_filename

    def reset(
        self,
        baseline: OperationPlan | None,
        interpretation: DrawingInterpretation,
    ) -> None:
        """Seed the preceding round's plan; the first round has none to seed."""
        if self.source_path.is_symlink():
            raise ValueError(f"{self.source_filename} must not be a symlink")
        if baseline is None:
            self.source_path.unlink(missing_ok=True)
        else:
            self.source_path.write_text(
                baseline.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
        self._interpretation = interpretation
        self._accepted = None
        self._checked = None

    def _contents(self) -> bytes | None:
        path = self.source_path
        return path.read_bytes() if path.is_file() and not path.is_symlink() else None

    @property
    def confirmed(self) -> bool:
        return self._accepted is not None and self._contents() == self._checked

    @property
    def accepted_plan(self) -> OperationPlan | None:
        return self._accepted if self.confirmed else None

    def feedback(self) -> list[ContentBlock]:
        contents = self._contents()
        self._checked = contents
        self._accepted = None
        _, attempt_dir, _ = self.attempt_store.issue("operations")
        if contents is not None:
            (attempt_dir / self.source_filename).write_bytes(contents)

        error = None
        try:
            if self._interpretation is None:
                raise SubmissionValidationError("the operations round is not prepared")
            if contents is None:
                raise SubmissionValidationError(
                    f"Write the complete plan to {self.source_filename}"
                )
            plan = OperationPlan.model_validate_json(contents)
            validate_operations(plan, self._interpretation)
            self._accepted = plan
        except ValidationError as invalid:
            error = "\n".join(
                file_errors(invalid, self.source_filename, contents or b"")
            )
        except Exception as invalid:  # noqa: BLE001 - return errors to the agent
            error = "\n".join(
                semantic_errors(invalid, self.source_filename, contents or b"")
            )
        (attempt_dir / "_operations_validation_log.json").write_text(
            json.dumps({"error": error}, indent=2) + "\n", encoding="utf-8"
        )

        return [
            create_text_block(
                f"[Operation plan verification]\n"
                f"{self.workdir.sandbox_bind_dir / self.source_filename}: "
                + (f"invalid.\n{error}" if error else "valid.")
            )
        ]
