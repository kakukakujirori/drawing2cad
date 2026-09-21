"""The auditor reviews file-derived evidence before completing the current audit."""

import json
from functools import partial

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from PIL import Image

from tests.zeroshot.chat_models import (
    ScriptedChatModel,
    tool_call,
    unanswered_tool_calls,
)
from tests.zeroshot.contracts import bootstrap_review
from tests.zeroshot.workflow.test_reconstruction_workflow import (
    _completed_run,
    _ref,
    _report,
)
from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.stages._base.prompt import StageInstructions
from zeroshot.pipeline.stages.audit.contracts import (
    AuditRegion,
    AuditReport,
    AuditSubmission,
)
from zeroshot.pipeline.stages.audit.stage import create_audit_stage
from zeroshot.pipeline.stages.audit.verify import AuditVerifier
from zeroshot.pipeline.verification import AttemptStore
from zeroshot.pipeline.workflow import create_agent
from zeroshot.pipeline.workflow.lifecycle import open_next_round
from zeroshot.pipeline.workflow.middleware import VerifyOnWriteMiddleware


def _answer(strategy, *, accepted=True):
    payload = {"accepted": accepted}
    if strategy == "provider":
        return AIMessage(content=json.dumps(payload))
    return tool_call("AuditSubmission", payload, "submit_audit")


def test_audit_stage_uses_its_configured_filename_for_writing_and_archiving():
    history = _completed_run()
    report = AuditReport(
        ticket_reviews=bootstrap_review(), concern_reviews={}, findings=[]
    )
    with SandboxWorkdir() as workdir:

        @tool
        def write_report() -> str:
            """Write the complete audit report."""
            (workdir.host_bind_dir / "review.json").write_text(report.model_dump_json())
            return "written"

        model = ScriptedChatModel(
            responses=(tool_call("write_report", {}, "write"), _answer("tool"))
        )
        context = {
            "coding_output_path": "/work/model.py",
            "reconstruction_path": "/work/reconstruction.json",
        }
        stage = create_audit_stage(
            partial(
                create_agent,
                role="output_auditor",
                model=model,
                max_turns=3,
                announce_turns=False,
                checkpointer=False,
                response_format_strategy="tool",
            ),
            tools=[write_report],
            system_prompt_path=None,
            instructions=StageInstructions([], "path", context, workdir),
            prompt_context=context,
            attempt_store=AttemptStore(workdir, lambda: 0),
            audit_filename="review.json",
        )
        result = stage.run({"reconstruction": history}, {})
        assert result["audit_report"] == report
        instruction = model.received_messages[0][-1].text
        assert "`/work/review.json`" in instruction
        assert "`audit.json`" not in instruction
        assert "review.json: valid." in model.received_messages[1][-1].text
        assert (
            workdir.host_bind_dir / "attempts/round_000/audit/000/review.json"
        ).is_file()
        assert not (workdir.host_bind_dir / "audit.json").exists()


@pytest.mark.parametrize("strategy", ["tool", "provider"])
def test_write_validate_review_rewrite_and_confirm_only_current_attempt(strategy):
    history = _completed_run()
    snapshot = history.snapshots[-1]
    report = _report(target=_ref("coding", "ret_hole"))
    report.findings[0].evidence = [AuditRegion(file="input.png", box=(0, 0, 10, 10))]
    corrected = report.model_copy(deep=True)
    corrected.findings[0].evidence = [
        AuditRegion(file="input.png", box=(10, 10, 20, 20))
    ]
    viewed = []
    with SandboxWorkdir() as workdir:
        Image.new("RGB", (20, 20), "white").save(workdir.host_bind_dir / "input.png")
        verifier = AuditVerifier(AttemptStore(workdir, lambda: snapshot.round))
        verifier.reset(snapshot)

        @tool
        def write_report(correct: bool) -> str:
            """Write the proposed audit file."""
            verifier.source_path.write_text(
                (corrected if correct else report).model_dump_json()
            )
            return "written"

        @tool
        def load_image(path: str) -> str:
            """Read an evidence crop."""
            with Image.open(workdir.sandbox_to_host_path(path)) as image:
                viewed.append((path, image.size))
            return "viewed"

        attempt0 = "/work/attempts/round_000/audit/000"
        attempt1 = "/work/attempts/round_000/audit/001"
        crop = "/find_shape_mismatch/evidence_0.png"
        model = ScriptedChatModel(
            responses=(
                tool_call("write_report", {"correct": False}, "write0"),
                tool_call("load_image", {"path": attempt0 + crop}, "view0"),
                tool_call("write_report", {"correct": True}, "write1"),
                tool_call("load_image", {"path": attempt1 + crop}, "view1"),
                _answer(strategy, accepted=False),
            )
        )
        middleware = VerifyOnWriteMiddleware(
            verifier, require_feedback_before_submit=True
        )
        agent = create_agent(
            role="output_auditor",
            model=model,
            tools=[write_report, load_image],
            output_schema=AuditSubmission,
            response_format_strategy=strategy,
            max_turns=8,
            model_retries=1,
            announce_turns=False,
            checkpointer=False,
            extra_middleware=[middleware],
        )
        result = agent.invoke(
            {"messages": [HumanMessage("Write and review audit.json.")]},
            context=snapshot,
        )
        assert result["structured_response"] == AuditSubmission(accepted=False)
        assert verifier.accepted_report == corrected
        assert viewed == [(attempt0 + crop, (10, 10)), (attempt1 + crop, (10, 10))]
        assert (
            "Check these crops with load_image" in model.received_messages[1][-1].text
        )
        assert unanswered_tool_calls(result["messages"]) == []
        next_round = open_next_round(history, corrected, verifier.evidence_crops)
        assert next_round.snapshots[-1].open_tickets[0].evidence_crops == [
            attempt1 + crop
        ]
        assert not (workdir.host_bind_dir / "tickets").exists()
        first = workdir.sandbox_to_host_path(attempt0) / "audit.json"
        assert AuditReport.model_validate_json(first.read_text()) == report

        verifier.source_path.write_text("{}")
        assert not verifier.confirmed
        verifier.feedback()
        assert verifier.accepted_report is None
        assert verifier.evidence_crops == {}


def test_invalid_report_and_crop_failure_cannot_leave_an_accepted_audit(monkeypatch):
    snapshot = _completed_run().snapshots[-1]
    with SandboxWorkdir() as workdir:
        verifier = AuditVerifier(AttemptStore(workdir, lambda: 0))
        verifier.reset(snapshot)
        report = AuditReport(
            ticket_reviews=bootstrap_review(),
            concern_reviews={},
            findings=[],
        )
        verifier.source_path.write_text(report.model_dump_json())
        assert "valid." in str(verifier.feedback())
        assert verifier.confirmed
        verifier.source_path.write_text("{}")
        assert "invalid" in str(verifier.feedback())
        assert not verifier.confirmed

        report = _report(target=_ref("coding", "ret_hole"))
        report.findings[0].evidence = [
            AuditRegion(file="input.png", box=(0, 0, 10, 10))
        ]
        Image.new("RGB", (20, 20)).save(workdir.host_bind_dir / "input.png")
        verifier.source_path.write_text(report.model_dump_json())

        def broken_crop(*args):
            raise OSError("render failed")

        monkeypatch.setattr(
            "zeroshot.pipeline.stages.audit.verify.crop_evidence", broken_crop
        )
        assert "render failed" in str(verifier.feedback())
        assert verifier.accepted_report is None


@pytest.mark.parametrize("strategy", ["tool", "provider"])
def test_completion_cannot_skip_feedback_or_submit_changed_invalid_bytes(strategy):
    snapshot = _completed_run().snapshots[-1]
    report = AuditReport(
        ticket_reviews=bootstrap_review(),
        concern_reviews={},
        findings=[],
    )
    with SandboxWorkdir() as workdir:
        verifier = AuditVerifier(AttemptStore(workdir, lambda: 0))
        verifier.reset(snapshot)
        verifier.source_path.write_text(report.model_dump_json())
        # The file is valid but no verification feedback has been delivered yet.
        middleware = VerifyOnWriteMiddleware(
            verifier, require_feedback_before_submit=True
        )
        from langchain.agents.middleware import ModelResponse
        from langchain_core.messages import ToolMessage

        acknowledgement = AuditSubmission(accepted=True)

        def respond(_request):
            message = _answer(strategy)
            return ModelResponse(
                result=[
                    message,
                    *(
                        [
                            ToolMessage(
                                "Submitted",
                                tool_call_id="submit_audit",
                                name="AuditSubmission",
                            )
                        ]
                        if strategy == "tool"
                        else []
                    ),
                ],
                structured_response=acknowledgement,
            )

        refused = middleware.wrap_model_call(None, respond)
        assert refused.structured_response is None
        assert "audit.json: valid." in refused.result[-1].text
        assert (
            middleware.wrap_model_call(None, respond).structured_response
            == acknowledgement
        )

        verifier.source_path.write_text("{}")
        assert middleware.wrap_model_call(None, respond).structured_response is None
        assert verifier.accepted_report is None
