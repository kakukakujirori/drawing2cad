from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.zeroshot.contracts import interpretation, view
from zeroshot.pipeline.messages.artifact import ArtifactPresenter
from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.stages.coding.middleware import CodingMiddleware
from zeroshot.pipeline.stages.coding.progress import ProgressOutputVerifier
from zeroshot.pipeline.stages.coding.stage import CodingStage
from zeroshot.pipeline.stages.coding.verify import VerifyOutputResult
from zeroshot.pipeline.stages.tickets.verify import TicketVerifier
from zeroshot.pipeline.stages.types import PipelineStage
from zeroshot.pipeline.verification.attempts import AttemptStore
from zeroshot.pipeline.verification.drawing_diff.align import AlignmentResult
from zeroshot.pipeline.verification.run_cadquery import (
    CadQueryExecutionReport,
    ExecutionStatus,
)
from zeroshot.pipeline.verification.run_drawing_diff import DrawingDiffReport

SOURCE = "result = object()\n"


def test_best_candidate_eligibility_cache_and_stage_baseline(tmp_path, monkeypatch):
    with SandboxWorkdir(host_bind_dir=tmp_path) as workdir:
        verifier = ProgressOutputVerifier(
            executor=None,
            workdir=workdir,
            renderer=None,
            diff_drawer=None,
            artifact_presenter=ArtifactPresenter(input="path"),
            attempt_store=AttemptStore(workdir, round_source=lambda: 0),
        )
        verifier.interpretation = interpretation(
            views=[
                view(role, scale=0.1, file=f"inputs/{role}.png")
                for role in ("front", "top", "right")
            ]
        )
        build_count = 0
        scores = [0.7, 0.7, 0.7]
        patches = {}

        def build():
            nonlocal build_count
            build_count += 1
            identifier, host, sandbox = verifier.attempt_store.issue("coding")
            source = verifier.source_path.read_text()
            (host / "model.py").write_text(source)
            step = host / "output.step"
            step.write_text("mock verified STEP")
            diffs = {
                name: DrawingDiffReport(
                    drawing_path=tmp_path
                    / "inputs"
                    / f"{name.removeprefix('view_')}.png",
                    projection_path=host / f"{name}.png",
                    alignment=AlignmentResult(
                        backend="directional_chamfer",
                        model="similarity",
                        status="ok",
                        H_drawing_to_projection=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        diagnostics={"scale_calibration": {"status": "applied"}},
                    ),
                    stats={
                        "match_score": score,
                        "bounded_chamfer_drawing_px": score * 20,
                        "comparison_status": "ok",
                        "provisional": False,
                        "outside_count": 0,
                        "material_error": {"status": "ok"},
                    },
                )
                for name, score in zip(
                    ("view_front", "view_top", "view_right"), scores, strict=True
                )
            }
            first = diffs["view_front"]
            if patches:
                diffs["view_front"] = replace(first, **patches)
            return VerifyOutputResult(
                verification_id=identifier,
                host_verification_dir=host,
                sandbox_verification_dir=str(sandbox),
                exec_report=CadQueryExecutionReport(
                    status=ExecutionStatus.VERIFIED,
                    source=source,
                    returncode=0,
                    step_path=step,
                ),
                drawing_diff_report=diffs,
            )

        monkeypatch.setattr(verifier, "_build", build)
        middleware = CodingMiddleware(verifier, fingerprint=verifier.source_digest)
        assert middleware.report_existing() == []  # Empty initial coder workspace.

        def attempt(values, source=SOURCE):
            nonlocal scores
            scores = values
            verifier.source_path.write_text(source + f"# trial {build_count}\n")
            return verifier.feedback()[0]["text"]

        assert "baseline (match score 7.00 px)" in attempt([7.0, 7.0, 7.0])
        before = build_count
        assert "baseline" in verifier.feedback()[0]["text"]
        assert build_count == before  # Repeated feedback uses the same saved build.
        improved = attempt([8.0, 4.0, 3.0])  # One view worse; the whole drawing better.
        assert "Overall match improved by 2.00 px over the saved best" in improved
        assert (
            "Best saved eligible candidate: /work/attempts/round_000/coding/001/model.py"
            in improved
        )
        worse = attempt([8.0, 8.0, 8.0])
        assert "Worsened by 3.00 px versus the previous candidate" in worse
        assert (
            "Previous candidate: /work/attempts/round_000/coding/001/model.py" in worse
        )
        # Changes inside the margin are not measurable and keep the saved best.
        assert "No measurable change: within 0.50 px" in attempt([8.0, 8.0, 8.3])
        near_best = attempt([5.0, 5.0, 4.4])
        assert "Improved by 3.30 px over the previous candidate" in near_best
        assert "but not by 0.50 px over the saved best (5.00 px)" in near_best
        assert "round_000/coding/001/model.py" in near_best.split("Best saved")[1]
        best = verifier._best_candidate
        assert best is not None
        saved_model = best[1].host_verification_dir / "model.py"
        saved_source = saved_model.read_text(encoding="utf-8")
        saved_model.write_text(
            saved_source + "# changed after measurement\n", encoding="utf-8"
        )
        assert verifier._candidate_score(best[1])[0] is None
        saved_model.write_text(saved_source, encoding="utf-8")
        assert verifier._candidate_score(best[1])[0] == best[0]

        for stats in (
            {"match_score": float("nan")},
            {"match_score": True},
            {"comparison_status": "uncertain"},
            {"provisional": True},
            {"outside_count": 1},
        ):
            patches = {
                "stats": {**best[1].drawing_diff_report["view_front"].stats, **stats}
            }
            assert "Overall comparison unavailable" in attempt([0.1, 0.1, 0.1])
            assert verifier._best_candidate is best
        patches = {
            "alignment": replace(
                first_alignment := best[1].drawing_diff_report["view_front"].alignment,
                diagnostics={},
            )
        }
        assert "uncalibrated" in attempt([0.1, 0.1, 0.1])
        patches = {"alignment": replace(first_alignment, status="uncertain")}
        assert "provisional" in attempt([0.1, 0.1, 0.1])
        patches = {"drawing_path": tmp_path / "a_different_input.png"}
        assert "Overall comparison unavailable" in attempt([0.1, 0.1, 0.1])
        patches = {}
        incomplete = replace(
            best[1],
            drawing_diff_report={
                "view_front": best[1].drawing_diff_report["view_front"]
            },
        )
        assert verifier._candidate_score(incomplete)[0] is None
        with_optional = replace(
            best[1],
            drawing_diff_report={
                **best[1].drawing_diff_report,
                "view_full_page": DrawingDiffReport(
                    drawing_path=tmp_path / "page.png",
                    projection_path=None,
                    error="unregistered projection role",
                ),
            },
        )
        assert verifier._candidate_score(with_optional)[0] == best[0]
        failed = replace(
            best[1],
            exec_report=replace(best[1].exec_report, status=ExecutionStatus.FAILED),
        )
        assert verifier._candidate_score(failed)[0] is None

        # A new invocation gets an existing-file baseline once; validation retries
        # reuse it, and the first valid answer is already allowed by the middleware.
        verifier.source_path.write_text(SOURCE)
        scores = [0.7, 0.7, 0.7]
        received = []
        agent = SimpleNamespace(
            invoke=lambda state, config: received.append(state) or {}
        )
        instructions = SimpleNamespace(build=lambda *args, **kwargs: "instructions")
        stage = CodingStage(
            agent,
            instructions,
            verifier,
            TicketVerifier(lambda: None),
            middleware,
            False,
        )
        state = {
            "reconstruction": SimpleNamespace(
                snapshots=[
                    SimpleNamespace(
                        last_completed_stage=PipelineStage.INTERPRETATION,
                        interpretation=verifier.interpretation,
                    )
                ]
            )
        }
        before = build_count
        stage.run(state, {})
        assert build_count == before + 1
        assert "First fully comparable" in received[-1]["messages"][-1].text
        assert middleware._artifact_reasons() == []
        stage.run({**state, "stage_validation_error": "answer missing a ticket"}, {})
        assert build_count == before + 1
        assert len(received[-1]["messages"]) == 1
        assert verifier._best_candidate is not best  # Fresh-invocation scope.
        assert "S=" not in improved and "C/20" not in improved


@pytest.mark.parametrize("margin", [-0.1, float("nan"), float("inf")])
def test_match_margin_must_be_finite_and_non_negative(tmp_path, margin):
    with (
        SandboxWorkdir(host_bind_dir=tmp_path) as workdir,
        pytest.raises(ValueError, match="match_margin_px"),
    ):
        ProgressOutputVerifier(
            executor=None,
            workdir=workdir,
            renderer=None,
            diff_drawer=None,
            artifact_presenter=ArtifactPresenter(input="path"),
            attempt_store=AttemptStore(workdir, round_source=lambda: 0),
            match_margin_px=margin,
        )
