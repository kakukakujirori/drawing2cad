import json
import shutil
from functools import partial

import pytest
from langchain_core.tools import tool

from tests.zeroshot.chat_models import ScriptedChatModel, tool_call
from tests.zeroshot.verification.test_interpretation_validation import dxf_case
from tests.zeroshot.verification.test_verify_interpretation import _case
from zeroshot.pipeline.messages.artifact import drawing_for_model
from zeroshot.pipeline.messages.manifest import register_view
from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.stages._base.prompt import StageInstructions
from zeroshot.pipeline.stages.interpretation.contracts import (
    DrawingInterpretation,
    View,
)
from zeroshot.pipeline.stages.interpretation.stage import create_interpretation_stage
from zeroshot.pipeline.stages.tickets.contracts import TicketAnswers
from zeroshot.pipeline.verification.attempts import AttemptStore
from zeroshot.pipeline.workflow.components.agent import create_agent
from zeroshot.pipeline.workflow.lifecycle import start_reconstruction


def test_stage_requires_written_verified_json_before_ticket_submission(tmp_path):
    verifier, candidate, _ = _case(tmp_path)
    drawing = [register_view("view_page", View.FULL_PAGE, tmp_path / "source.png")]
    response = {
        "stage_report": {
            "concerns": {},
            "dimension_checks": None,
            "unticketed_changes": {},
        },
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
    stage = create_interpretation_stage(
        builder=partial(
            create_agent,
            role="interpreter",
            model=model,
            checkpointer=False,
            max_turns=5,
            announce_turns=False,
            response_format_strategy="tool",
        ),
        tools=[write_interpretation],
        system_prompt_path=None,
        instructions=StageInstructions(drawing, "path", {}, verifier.workdir),
        prompt_context={},
        attempt_store=verifier.attempt_store,
    )
    run = start_reconstruction(
        "run_test",
        "Reconstruct the part.",
        drawing_for_model(drawing, verifier.workdir),
    )
    result = stage.run({"reconstruction": run}, {})
    assert result["stage_submission"] == TicketAnswers.model_validate(response)
    assert stage.interpretation_verifier.accepted_interpretation is not None
    messages = result["interpretation_state"]["messages"]
    assert any("not ready to submit" in message.text for message in messages)
    assert any('"inliers": "3/3"' in message.text for message in messages)
    assert "DrawingInterpretation JSON schema" in model.received_messages[0][-1].text
    assert "calculate_drawing_scale" in model.bound_tool_name_history[0]
    assert "[Native DXF coordinates]" not in model.received_messages[0][-1].text

    assert (
        DrawingInterpretation.model_validate_json(
            stage.interpretation_verifier.source_path.read_bytes()
        )
        == stage.interpretation_verifier.accepted_interpretation
    )
    assert len(model.received_messages) == 3
    # One for the premature answer, naming the view the seed still lacks, and
    # one for the write that followed.
    verifications = [
        message.text
        for message in messages
        if "[Interpretation verification]" in message.text
    ]
    assert len(verifications) == 2
    assert "full_page input is an unsplit page" in verifications[0]
    assert "valid." in verifications[1]


def _dxf_input(tmp_path):
    candidate = dxf_case(tmp_path)
    drawing = [
        register_view("view_front", View.FRONT, tmp_path / "front.dxf", ("+x", "+z"))
    ]
    workdir = SandboxWorkdir(tmp_path)
    return candidate, drawing, workdir


def test_dxf_metadata_reaches_model_and_written_artifact_validates(tmp_path):
    candidate, drawing, workdir = _dxf_input(tmp_path)
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
        "stage_report": {
            "concerns": {},
            "dimension_checks": None,
            "unticketed_changes": {},
        },
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
    stage = create_interpretation_stage(
        builder=partial(
            create_agent,
            role="interpreter",
            model=model,
            checkpointer=False,
            max_turns=4,
            announce_turns=False,
            response_format_strategy="tool",
        ),
        tools=[write_dxf_interpretation],
        system_prompt_path=None,
        instructions=StageInstructions(drawing, "path", {}, workdir),
        prompt_context={},
        attempt_store=AttemptStore(workdir, lambda: 0),
    )
    run = start_reconstruction(
        "run_dxf", "Reconstruct the part.", drawing_for_model(drawing, workdir)
    )
    result = stage.run({"reconstruction": run}, {})
    assert result["stage_submission"] == TicketAnswers.model_validate(response)
    text = model.received_messages[0][-1].text
    assert "The original DXFs below are already registered" in text
    assert "using role full_page" not in text
    assert "box_uv is what you read out of the file" in text
    metadata = json.loads(text.splitlines()[-1])
    assert metadata["originals"] == [
        {
            "view": "view_front",
            "file": "/work/front.dxf",
            "box_mm": pytest.approx([10.0, 20.0, 111.6, 96.2]),
        }
    ]
    accepted = stage.interpretation_verifier.accepted_interpretation
    assert accepted is not None and len(accepted.views) == 2
    assert accepted.views[0].role == View.FRONT
    assert accepted.views[0].region.box_uv == pytest.approx((10.0, 20.0, 111.6, 96.2))
    reports = stage.interpretation_verifier.verify().reports
    assert reports["view_front"]["status"] == "native_dxf"
    assert reports["view_detail"]["status"] == "native_dxf"
    data = accepted.model_dump()
    data["views"][0]["region"]["box_uv"] = [0, 0, 50, 50]
    stage.interpretation_verifier.source_path.write_text(json.dumps(data))
    rejected = stage.interpretation_verifier.verify()
    assert not rejected.confirmed
    assert "full-file region" in rejected.errors[0]
