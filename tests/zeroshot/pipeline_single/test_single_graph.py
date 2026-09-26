import sys
from functools import partial
from pathlib import Path

from langchain_core.messages import AIMessage
from PIL import Image

import zeroshot.pipeline_single.graph as graph_module
from tests.zeroshot.chat_models import ScriptedChatModel, tool_call
from zeroshot.pipeline.messages.artifact import ArtifactPresenter
from zeroshot.pipeline.messages.manifest import InputManifest, register_view
from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.stages._base.prompt import PromptTemplate
from zeroshot.pipeline.stages.coding.verify import VerifyOutputResult
from zeroshot.pipeline.stages.interpretation.contracts import View
from zeroshot.pipeline.verification.render.orthographic import STANDARD_VIEW_FRAMES
from zeroshot.pipeline.verification.run_cadquery import (
    CadQueryExecutionReport,
    ExecutionStatus,
)
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

    def verify(self) -> VerifyOutputResult:
        self.builds += 1
        return VerifyOutputResult(
            verification_id=f"{self.builds:03d}",
            exec_report=CadQueryExecutionReport(status=ExecutionStatus.VERIFIED),
        )

    def feedback(self):
        return []


def _answer(model) -> AIMessage:
    return AIMessage(content=model.model_dump_json())


def _graph(workdir: SandboxWorkdir, coder, auditor):
    common = {"announce_turns": False, "model_retries": 0, "max_turns": 5}
    image = workdir.host_bind_dir / "drawing.png"
    Image.new("RGB", (20, 20), "white").save(image)
    return graph_module.create_single_graph(
        coding_agent_builder=partial(create_agent, role="coder", model=coder, **common),
        audit_agent_builder=partial(
            create_agent, role="output_auditor", model=auditor, **common
        ),
        compact_between_stages=coder,
        sandbox_runner=SandboxRunner(
            python_executable=Path(sys.executable), default_timeout_s=30
        ),
        sandbox_workdir=workdir,
        artifact_presenter=ArtifactPresenter(input_mode="path", feedback_mode="path"),
        input_manifest=InputManifest(
            sample_id="test",
            drawing=[register_view("view_input", View.FULL_PAGE, image)],
        ),
    )


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
    with SandboxWorkdir() as workdir:
        result = _graph(workdir, coder, auditor).invoke({})

    assert result["round"] == 1
    assert result["stage_submission"] == {"summary": "plate with hole"}
    assert result["verification"] == {"verification_id": "002", "status": "VERIFIED"}
    assert len(auditor.received_messages) == 1
    assert any("hole missing" in m.text for m in coder.received_messages[1])


PROGRAM = """\
import cadquery as cq

result = cq.Workplane("XY").box(30, 20, 10, centered=False)
"""


def test_a_written_program_reaches_coder_and_auditor_in_six_views(monkeypatch):
    """The real verifier builds and draws what the coder wrote with its shell."""
    monkeypatch.setattr(
        graph_module, "compact_transcript", lambda thread, **_: thread[-1:]
    )
    write = tool_call(
        "run_shell", {"command": f"cat > /work/model.py <<'EOF'\n{PROGRAM}EOF"}, "write"
    )
    coder = ScriptedChatModel(responses=(write, _answer(CodingReport(summary="box"))))
    auditor = ScriptedChatModel(
        responses=(_answer(SingleAuditReport(accepted=True, findings=[])),)
    )
    with SandboxWorkdir() as workdir:
        result = _graph(workdir, coder, auditor).invoke({})
        projection_dir = (
            workdir.host_bind_dir / "attempts/round_000/coding/000/projection"
        )
        drawn = {path.name for path in projection_dir.iterdir()}

    assert drawn == {
        f"{view}.{suffix}" for view in STANDARD_VIEW_FRAMES for suffix in ("dxf", "png")
    }
    assert result["verification"] == {"verification_id": "000", "status": "VERIFIED"}
    before, after = (
        "\n".join(m.text for m in call) for call in coder.received_messages
    )
    attempt = "/work/attempts/round_000/coding/000"
    for view in STANDARD_VIEW_FRAMES:
        assert f"{attempt}/projection/{view}.png" not in before
        assert f"{attempt}/projection/{view}.png" in after
    (audit,) = auditor.received_messages
    audit_text = "\n".join(m.text for m in audit)
    assert f"Built artifacts are in {attempt}" in audit_text
    assert "Verification status: VERIFIED" in audit_text
    # The auditor reads the view names in the same frames as the coder.
    assert PromptTemplate(graph_module._COORDINATE_FRAMES).render() in audit_text
