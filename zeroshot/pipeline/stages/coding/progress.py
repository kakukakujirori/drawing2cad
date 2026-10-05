"""Rank verified builds by their drawing match and tell the coder how they compare."""

import math
from pathlib import PurePosixPath
from statistics import fmean
from typing import Any

from langchain_core.messages.content import ContentBlock, create_text_block

from zeroshot.pipeline.stages.coding.verify import OutputVerifier, VerifyOutputResult
from zeroshot.pipeline.verification.run_cadquery import ExecutionStatus


class ProgressOutputVerifier(OutputVerifier):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._previous_candidate: VerifyOutputResult | None = None
        self._best_candidate: tuple[float, VerifyOutputResult] | None = None
        self._progress_text = ""

    def reset(self) -> None:
        super().reset()
        self._previous_candidate = None
        self._best_candidate = None
        self._progress_text = ""

    def _candidate_score(self, report: VerifyOutputResult) -> tuple[float | None, str]:
        """Require the whole registered drawing and the existing submission checks."""
        execution = report.exec_report
        if (
            execution is None
            or execution.status != ExecutionStatus.VERIFIED
            or execution.returncode != 0
            or execution.source is None
            or execution.step_path is None
            or not execution.step_path.is_file()
            or execution.step_path.is_symlink()
            or report.host_verification_dir is None
            or report.sandbox_verification_dir is None
            or not (report.host_verification_dir / self.source_filename).is_file()
            or (report.host_verification_dir / self.source_filename).is_symlink()
            or self.operations is None
        ):
            return None, "STEP or OperationPlan checks are not satisfied"
        if faults := self._program_faults(report):
            return None, "submission is blocked: " + " ".join(faults)
        try:
            saved_source = (
                report.host_verification_dir / self.source_filename
            ).read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return None, "the saved candidate program is unreadable"
        if saved_source != execution.source:
            return None, "the saved candidate differs from the measured program"
        if self.interpretation is None:
            return None, "the interpreted views are unavailable"
        frames = self.interpretation.view_frames()
        expected = {
            view.name: view for view in self.interpretation.views if view.role in frames
        }
        diffs = report.drawing_diff_report or {}
        if not expected or not set(expected).issubset(diffs):
            return None, "not all registered views have drawing comparisons"
        scores = []
        for name in sorted(expected):
            diff = diffs[name]
            alignment = diff.alignment
            stats = diff.stats
            view = expected[name]
            if (
                diff.error
                or diff.drawing_path != self.workdir.sandbox_to_host_path(view.file)
                or view.scale is None
                or not math.isfinite(view.scale)
                or view.scale <= 0
                or alignment is None
                or alignment.status != "ok"
                or alignment.H_drawing_to_projection is None
                or alignment.diagnostics.get("scale_calibration", {}).get("status")
                != "applied"
                or stats.get("comparison_status") != "ok"
                or stats.get("provisional") is not False
                or stats.get("outside_count") != 0
                or stats.get("material_error", {}).get("status") != "ok"
            ):
                return (
                    None,
                    f"{name} is uncalibrated, provisional, or incompletely observed",
                )
            score = stats.get("match_score")
            if type(score) not in (int, float) or not math.isfinite(score) or score < 0:
                return None, f"{name} has no finite drawing comparison score"
            scores.append(score)
        return fmean(scores), ""

    @property
    def best_report(self) -> VerifyOutputResult | None:
        """The best eligible build of this invocation."""
        return self._best_candidate[1] if self._best_candidate is not None else None

    def candidate_path(self, report: VerifyOutputResult) -> str:
        assert report.sandbox_verification_dir is not None
        return str(
            PurePosixPath(report.sandbox_verification_dir) / self.source_filename
        )

    def feedback(self) -> list[ContentBlock]:
        blocks = super().feedback()
        report = self._last_feedback_report
        assert report is not None
        execution = report.exec_report
        if execution is None or execution.status != ExecutionStatus.VERIFIED:
            return blocks  # The build report already says why.
        if report is not self._previous_candidate:
            score, reason = self._candidate_score(report)
            previous = self._previous_candidate
            previous_score = self._candidate_score(previous)[0] if previous else None
            best = self._best_candidate

            if score is None:
                text = f"Overall comparison unavailable: {reason}."
            elif best is None:
                self._best_candidate = (score, report)
                text = "First fully comparable candidate recorded as the baseline."
            elif score < best[0]:
                self._best_candidate = (score, report)
                text = (
                    "Overall drawing match improved. Use this candidate as the next "
                    "baseline; verify the selected defect in the latest target view "
                    "and check other views. A numeric improvement does not confirm "
                    "that defect is fixed."
                )
            elif previous_score is not None and score < previous_score:
                text = "Overall match improved versus the previous candidate; " + (
                    "it matches the saved best."
                    if score == best[0]
                    else "the saved best remains better."
                )
            elif previous_score is not None and score > previous_score:
                text = "Overall match worsened versus the previous candidate; inspect the latest views before adopting it."
            else:
                text = "No improvement over the saved best candidate was measured."

            if previous is not None and previous.sandbox_verification_dir is not None:
                text += f"\nPrevious candidate: {self.candidate_path(previous)}"
            if report.sandbox_verification_dir is not None:
                text += f"\nCurrent candidate: {self.candidate_path(report)}"
            if self._best_candidate is not None:
                text += "\nBest saved eligible candidate: " + self.candidate_path(
                    self._best_candidate[1]
                )
            # ponytail: best is scoped to this invocation; persist it only if
            # checkpoint-resume needs comparisons spanning coder invocations.
            self._previous_candidate = report
            self._progress_text = text
        return [
            create_text_block("[Candidate progress]\n" + self._progress_text),
            *blocks,
        ]
