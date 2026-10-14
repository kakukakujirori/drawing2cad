"""An audit that dismisses what only the build draws is questioned, and refused once."""

import sys
from dataclasses import replace
from functools import partial
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from tests.zeroshot.chat_models import ScriptedChatModel, tool_call
from tests.zeroshot.contracts import bootstrap_review
from tests.zeroshot.prompt_paths import ROLE_PATHS
from tests.zeroshot.workflow.test_reconstruction_workflow import _completed_run
from zeroshot.pipeline.messages.artifact import ArtifactPresenter
from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.stages._base.prompt import StageInstructions
from zeroshot.pipeline.stages.audit.contracts import AuditReport, ConcernReview
from zeroshot.pipeline.stages.audit.stage import create_audit_stage
from zeroshot.pipeline.stages.audit.verify import dismissed_extras
from zeroshot.pipeline.stages.interpretation.contracts import ProjectionMethod
from zeroshot.pipeline.tools.cad_hypothesis import TRIALS_DIRNAME
from zeroshot.pipeline.verification import AttemptStore
from zeroshot.pipeline.verification.run_drawing_diff import DrawingDiffReport
from zeroshot.pipeline.workflow import create_agent

DISMISSED = ConcernReview(finding_name=None, disposition="A drawing convention.")


def _listed(*directions: str):
    history = _completed_run()
    snapshot = history.snapshots[-1]
    groups = [
        {
            "direction": direction,
            "kind": "lines",
            "unit": "px",
            "size_px": 60,
            "box_px": [1, 2, 3, 4],
            "color": "red",
        }
        for direction in directions
    ]
    assert snapshot.verification is not None
    snapshot.verification = replace(
        snapshot.verification,
        drawing_diff_report={
            "view_front": DrawingDiffReport(
                drawing_path=Path("front.png"),
                projection_path=None,
                stats={"unmatched": groups},
            )
        },
    )
    return history


def _report(**reviews: ConcernReview) -> AuditReport:
    return AuditReport(
        ticket_reviews=bootstrap_review(),
        concern_reviews={
            f"drawing_diff.view_front.{key[-1]}": review
            for key, review in reviews.items()
        },
        findings=[],
    )


def test_only_a_dismissed_extra_item_is_questioned():
    history = _listed("extra", "missing", "extra")
    linked = ConcernReview(finding_name="find_bore", disposition="The bore.")
    dismissed = _report(item1=DISMISSED, item2=DISMISSED)
    # Without its finding: only the review matters here.
    audit = dismissed.model_copy(
        update={
            "concern_reviews": dismissed.concern_reviews
            | {"drawing_diff.view_front.3": linked}
        }
    )

    assert dismissed_extras(audit, history.snapshots[-1]) == [
        "drawing_diff.view_front.1"
    ]


def _run(history, report: AuditReport, script: tuple[str, ...]) -> tuple:
    """Run the audit stage on a model that writes `report` and answers as scripted."""
    with SandboxWorkdir() as workdir:

        @tool
        def write_report() -> str:
            """Write the complete audit report."""
            (workdir.host_bind_dir / "audit.json").write_text(report.model_dump_json())
            return "written"

        answer = tool_call("AuditSubmission", {"accepted": True}, "submit")
        steps = {
            "write": tool_call("write_report", {}, "write"),
            "answer": answer,
            # An answer with another call is refused before any check.
            "write and answer": AIMessage(
                content="",
                tool_calls=[
                    *tool_call("write_report", {}, "w2").tool_calls,
                    *answer.tool_calls,
                ],
            ),
        }
        model = ScriptedChatModel(responses=tuple(steps[step] for step in script))
        context = {
            "coding_output_path": "/work/model.py",
            "reconstruction_path": "/work/reconstruction.json",
        }
        stage = create_audit_stage(
            partial(
                create_agent,
                role="output_auditor",
                model=model,
                max_turns=len(script) + 1,
                announce_turns=False,
                checkpointer=False,
                response_format_strategy="tool",
            ),
            tools=[write_report],
            role_path=ROLE_PATHS["output_auditor"],
            instructions=StageInstructions(
                [],
                "path",
                context,
                workdir,
                projection_method=ProjectionMethod.THIRD_ANGLE,
            ),
            prompt_context=context,
            attempt_store=AttemptStore(workdir, lambda: 0),
            artifact_presenter=ArtifactPresenter(input="path"),
            sandbox_runner=SandboxRunner(Path(sys.executable), default_timeout_s=10),
            diff_drawer_config=None,
            trial_store=AttemptStore(workdir, lambda: 0, root_dirname=TRIALS_DIRNAME),
        )
        result = stage.run({"reconstruction": history}, {})
    return result, model.received_messages


def test_the_check_is_reported_and_refuses_an_answer_once():
    report = _report(item1=DISMISSED)
    result, received = _run(_listed("extra"), report, ("write", "answer", "answer"))

    assert result["audit_report"] == report
    checked = received[1][-1].text
    assert "[Dismissal check]" in checked
    assert "drawing_diff.view_front.1" in checked
    # The second answer is accepted: the refusal is not repeated.
    assert len(received) == 3
    refused = received[2][-1].text
    assert refused.startswith("Answer refused once:")
    assert "[Dismissal check]" in refused


def test_an_answer_the_write_check_refuses_does_not_use_up_the_question():
    report = _report(item1=DISMISSED)
    script = ("write", "write and answer", "answer", "answer")
    result, received = _run(_listed("extra"), report, script)

    assert result["audit_report"] == report
    assert len(received) == 4
    assert received[3][-1].text.startswith("Answer refused once:")
