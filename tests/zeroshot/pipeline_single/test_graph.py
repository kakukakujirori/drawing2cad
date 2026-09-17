import sys
from functools import partial
from pathlib import Path

from langchain_core.messages import AIMessage
from PIL import Image

import zeroshot.pipeline_single.graph as graph_module
from tests.zeroshot.chat_models import ScriptedChatModel
from zeroshot.pipeline.messages.artifact import ArtifactPresenter
from zeroshot.pipeline.messages.manifest import InputManifest, register_view
from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.stages.interpretation.contracts import View
from zeroshot.pipeline.verification.run_cadquery import ExecutionStatus
from zeroshot.pipeline.verification.verify_output import VerifyOutputResult
from zeroshot.pipeline.workflow import create_agent
from zeroshot.pipeline_single.contracts import (
    CodingReport,
    Finding,
    SingleAuditReport,
)


class _StubVerifier:
    def __init__(self, workdir: SandboxWorkdir, **_: object) -> None:
        self.source_path = workdir.host_bind_dir / "model.py"
        self.confirmed = True
        self.builds = 0

    def reset(self) -> None:
        pass

    def verify(self):
        self.builds += 1
        return VerifyOutputResult(
            verification_id=f"{self.builds:03d}", status=ExecutionStatus.VERIFIED
        ), None

    def feedback(self):
        return []


def _answer(model) -> AIMessage:
    return AIMessage(content=model.model_dump_json())


def test_a_rejected_audit_sends_its_findings_to_a_second_coding_round(monkeypatch):
    monkeypatch.setattr(graph_module, "OutputVerifier", _StubVerifier)
    monkeypatch.setattr(
        graph_module, "compact_transcript", lambda thread, **_: thread[-1:]
    )
    finding = Finding(
        observation="hole missing", evidence=["front view"], revision_request="add it"
    )
    coder = ScriptedChatModel(
        responses=(
            _answer(CodingReport(summary="plate")),
            _answer(CodingReport(summary="plate with hole")),
        )
    )
    auditor = ScriptedChatModel(
        responses=(_answer(SingleAuditReport(accepted=False, findings=[finding])),)
    )
    common = {"announce_turns": False, "model_retries": 0, "max_turns": 5}
    with SandboxWorkdir() as workdir:
        image = workdir.host_bind_dir / "drawing.png"
        Image.new("RGB", (20, 20), "white").save(image)
        result = graph_module.create_single_graph(
            coding_agent_builder=partial(
                create_agent, role="coder", model=coder, **common
            ),
            audit_agent_builder=partial(
                create_agent, role="output_auditor", model=auditor, **common
            ),
            compact_between_stages=coder,
            sandbox_runner=SandboxRunner(
                python_executable=Path(sys.executable), default_timeout_s=10
            ),
            sandbox_workdir=workdir,
            artifact_presenter=ArtifactPresenter(
                input_mode="path", feedback_mode="none"
            ),
            input_manifest=InputManifest(
                sample_id="test",
                drawing=[register_view("view_input", View.FULL_PAGE, image)],
            ),
        ).invoke({})

    assert result["round"] == 1
    assert result["stage_submission"] == {"summary": "plate with hole"}
    assert len(auditor.received_messages) == 1
    assert any("hole missing" in m.text for m in coder.received_messages[1])
