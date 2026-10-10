"""The coder's two files: interpretation.json is verified first and gates the build."""

import json
import shutil
import sys
from functools import partial
from hashlib import sha256
from pathlib import Path

import pytest
from langchain_core.messages.content import create_text_block
from langchain_core.tools import tool

from tests.zeroshot.chat_models import ScriptedChatModel, tool_call
from tests.zeroshot.prompt_paths import ROLE_PATHS
from tests.zeroshot.verification.test_interpretation_validation import dxf_case
from tests.zeroshot.verification.test_verify_interpretation import _case
from tests.zeroshot.workflow.test_graph import _artifact_presenter, _stub_verification
from zeroshot.pipeline.messages.artifact import drawing_for_model
from zeroshot.pipeline.messages.manifest import register_view
from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.stages._base.prompt import StageInstructions
from zeroshot.pipeline.stages.coding.middleware import (
    CodingMiddleware,
    CodingTrialMiddleware,
    FreshCodingMiddleware,
)
from zeroshot.pipeline.stages.coding.stage import create_coding_stage
from zeroshot.pipeline.stages.coding.workspace import CodingWorkspaceVerifier
from zeroshot.pipeline.stages.interpretation.contracts import (
    DrawingInterpretation,
    View,
)
from zeroshot.pipeline.stages.tickets.contracts import TicketAnswers
from zeroshot.pipeline.verification.attempts import AttemptStore
from zeroshot.pipeline.workflow.components.agent import create_agent
from zeroshot.pipeline.workflow.lifecycle import start_reconstruction


class _Output:
    """A build that only records what it was asked to build against."""

    source_filename = "model.py"

    def __init__(self, workdir: SandboxWorkdir) -> None:
        self.source_path = workdir.host_bind_dir / "model.py"
        self.interpretation: DrawingInterpretation | None = None
        self.builds: list[DrawingInterpretation | None] = []

    def source_digest(self) -> str | None:
        path = self.source_path
        return sha256(path.read_bytes()).hexdigest() if path.is_file() else None

    def reset(self) -> None:
        pass

    def feedback(self):
        self.builds.append(self.interpretation)
        return [create_text_block("[Execution result]")]

    @property
    def blockers(self) -> list[str]:
        return []


def _workspace(tmp_path):
    verifier, candidate, seed = _case(tmp_path)
    output = _Output(verifier.workdir)
    workspace = CodingWorkspaceVerifier(verifier, output)  # type: ignore[arg-type]
    workspace.reset(seed, ["ticket_initial"])
    output.source_path.write_text("result = object()\n")
    return workspace, output, candidate


def _text(blocks) -> str:
    return "\n".join(block["text"] for block in blocks)


def test_the_build_waits_for_a_valid_interpretation(tmp_path):
    workspace, output, candidate = _workspace(tmp_path)

    gated = _text(workspace.feedback())
    assert "[Interpretation verification]" in gated
    assert "[Build skipped]" in gated
    assert output.builds == []
    assert not workspace.interpretation_ready
    assert any("is built only after" in b for b in workspace.blockers)

    workspace.interpretation.source_path.write_text(candidate.model_dump_json())
    built = _text(workspace.feedback())
    assert "[Execution result]" in built
    assert len(output.builds) == 1 and output.builds[0] is not None
    assert workspace.interpretation_ready
    assert workspace.blockers == []
    assert built.endswith(
        "[Pending answer requirements]\n"
        "- TicketAnswers.responses: one answer for each open ticket: ticket_initial\n"
    )


def test_only_a_change_of_views_rebuilds_an_unchanged_program(tmp_path):
    workspace, output, candidate = _workspace(tmp_path)
    workspace.interpretation.source_path.write_text(candidate.model_dump_json())
    workspace.feedback()

    accepted = json.loads(workspace.interpretation.source_path.read_text())
    accepted["features"][0]["description"] = "A reworded description."
    workspace.interpretation.source_path.write_text(json.dumps(accepted))
    reworded = _text(workspace.feedback())
    assert "[Interpretation verification]" in reworded
    assert len(output.builds) == 1

    accepted["views"][1]["role"] = "top"
    accepted["views"][1]["u_axis"], accepted["views"][1]["v_axis"] = "+x", "+y"
    workspace.interpretation.source_path.write_text(json.dumps(accepted))
    workspace.feedback()
    rebuilt_on = output.builds[-1]
    assert len(output.builds) == 2 and rebuilt_on is not None
    assert rebuilt_on.views[1].role is View.TOP


def test_an_answer_needs_features(tmp_path):
    workspace, output, candidate = _workspace(tmp_path)
    workspace.interpretation.source_path.write_text(
        candidate.model_copy(update={"features": []}).model_dump_json()
    )
    pending = _text(workspace.feedback())

    assert output.builds  # The build does not wait for features.
    (blocker,) = workspace.blockers
    assert blocker.startswith("interpretation.json lists no features.")
    assert "- interpretation.json lists no features." in pending


def _stage(builder, workdir, drawing, **flags):
    return create_coding_stage(
        builder,
        tools=[],
        role_path=ROLE_PATHS["coder"],
        instructions=StageInstructions(drawing, "path", _PATHS, workdir),
        prompt_context={},
        attempt_store=AttemptStore(workdir, lambda: 0),
        sandbox_runner=SandboxRunner(
            python_executable=Path(sys.executable), default_timeout_s=1
        ),
        diff_drawer_config=None,
        artifact_presenter=_artifact_presenter(),
        **flags,
    )


@pytest.mark.parametrize("fresh", [False, True])
@pytest.mark.parametrize("reminder", [False, True])
def test_the_flags_choose_the_coder_middleware_in_one_place(tmp_path, fresh, reminder):
    built = {}

    def builder(**kwargs):
        built.update(kwargs)
        return object()

    stage = _stage(
        partial(builder, max_turns=5),
        SandboxWorkdir(tmp_path),
        [],
        fresh_memory=fresh,
        coding_trial_reminder=reminder,
    )
    kinds = [type(middleware) for middleware in built["extra_middleware"]]
    assert kinds == [
        FreshCodingMiddleware if fresh else CodingMiddleware,
        *([CodingTrialMiddleware] if reminder else []),
    ]
    assert stage.middleware is built["extra_middleware"][0]


_PATHS = {"coding_output_path": "/work/model.py", "verification_dir": "/work/attempts"}


def _coder(model, max_turns):
    return partial(
        create_agent,
        role="coder",
        model=model,
        checkpointer=False,
        max_turns=max_turns,
        announce_turns=False,
        response_format_strategy="tool",
    )


def test_an_answer_waits_for_a_written_verified_interpretation(tmp_path, monkeypatch):
    _stub_verification(monkeypatch)
    verifier, candidate, _ = _case(tmp_path)
    drawing = [register_view("view_page", View.FULL_PAGE, tmp_path / "source.png")]
    response = {
        "stage_report": {"concerns": {}},
        "responses": {
            "ticket_initial": "Read view_front and localized sem_pin_upper_left."
        },
    }

    @tool("write_interpretation")
    def write_interpretation() -> str:
        """Write the complete candidate artifact."""
        verifier.source_path.write_text(candidate.model_dump_json())
        return "written"

    model = ScriptedChatModel(
        responses=(
            tool_call("TicketAnswers", response, "premature"),
            tool_call("write_interpretation", {}, "write"),
            tool_call("TicketAnswers", response, "accepted"),
        )
    )
    stage = create_coding_stage(
        _coder(model, 5),
        tools=[write_interpretation],
        role_path=ROLE_PATHS["coder"],
        instructions=StageInstructions(drawing, "path", _PATHS, verifier.workdir),
        prompt_context={},
        attempt_store=verifier.attempt_store,
        sandbox_runner=SandboxRunner(
            python_executable=Path(sys.executable), default_timeout_s=1
        ),
        diff_drawer_config=None,
        artifact_presenter=_artifact_presenter(),
    )
    run = start_reconstruction(
        "run_test",
        "Reconstruct the part.",
        drawing_for_model(drawing, verifier.workdir),
    )
    result = stage.run({"reconstruction": run}, {})

    assert result["stage_submission"] == TicketAnswers.model_validate(response)
    accepted = stage.workspace.interpretation.accepted_interpretation
    assert accepted is not None
    assert (
        DrawingInterpretation.model_validate_json(
            stage.workspace.interpretation.source_path.read_bytes()
        )
        == accepted
    )
    messages = [message.text for message in result["coding_state"]["messages"]]
    refusal = next(text for text in messages if "Answer refused" in text)
    assert "full_page input is an unsplit page" in refusal
    assert "[Build skipped]" in refusal
    assert any('"inliers": "3/3"' in text for text in messages)
    assert "DrawingInterpretation JSON schema" in model.received_messages[0][-1].text
    assert "calculate_drawing_scale" in model.bound_tool_name_history[0]
    assert "[Native DXF coordinates]" not in model.received_messages[0][-1].text
    assert len(model.received_messages) == 3


def test_dxf_metadata_reaches_the_coder_and_its_interpretation_validates(
    tmp_path, monkeypatch
):
    _stub_verification(monkeypatch)
    candidate = dxf_case(tmp_path)
    drawing = [
        register_view("view_front", View.FRONT, tmp_path / "front.dxf", ("+x", "+z"))
    ]
    workdir = SandboxWorkdir(tmp_path)
    data = candidate.model_dump()
    data["views"].append(
        {
            **data["views"][0],
            "name": "view_detail",
            "role": "front",
            "file": "/work/detail.dxf",
            "dimensions": [],
        }
    )
    candidate = DrawingInterpretation.model_validate(data)
    response = {
        "stage_report": {"concerns": {}},
        "responses": {
            "ticket_initial": "Interpreted view_front and view_detail in millimetres."
        },
    }

    @tool("write_dxf_interpretation")
    def write_dxf_interpretation() -> str:
        """Write a DXF interpretation and an additional drawing file."""
        shutil.copyfile(tmp_path / "front.dxf", tmp_path / "detail.dxf")
        (tmp_path / "interpretation.json").write_text(candidate.model_dump_json())
        return "written"

    model = ScriptedChatModel(
        responses=(
            tool_call("write_dxf_interpretation", {}, "write"),
            tool_call("TicketAnswers", response, "accepted"),
        )
    )
    stage = create_coding_stage(
        _coder(model, 4),
        tools=[write_dxf_interpretation],
        role_path=ROLE_PATHS["coder"],
        instructions=StageInstructions(drawing, "path", _PATHS, workdir),
        prompt_context={},
        attempt_store=AttemptStore(workdir, lambda: 0),
        sandbox_runner=SandboxRunner(
            python_executable=Path(sys.executable), default_timeout_s=1
        ),
        diff_drawer_config=None,
        artifact_presenter=_artifact_presenter(),
    )
    run = start_reconstruction(
        "run_dxf", "Reconstruct the part.", drawing_for_model(drawing, workdir)
    )
    result = stage.run({"reconstruction": run}, {})
    assert result["stage_submission"] == TicketAnswers.model_validate(response)
    text = model.received_messages[0][-1].text
    assert "The original DXFs below are already registered" in text
    assert "box_uv is what you read out of the file" in text
    metadata = json.loads(text.splitlines()[-1])
    assert metadata["originals"] == [
        {
            "view": "view_front",
            "file": "/work/front.dxf",
            "box_mm": pytest.approx([10.0, 20.0, 111.6, 96.2]),
        }
    ]
    interpretation = stage.workspace.interpretation
    accepted = interpretation.accepted_interpretation
    assert accepted is not None and len(accepted.views) == 2
    assert accepted.views[0].region.box_uv == pytest.approx((10.0, 20.0, 111.6, 96.2))
    reports = interpretation.verify().reports
    assert reports["view_front"]["status"] == "native_dxf"
    assert reports["view_detail"]["status"] == "native_dxf"
