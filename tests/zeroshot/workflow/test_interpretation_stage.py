import json
import shutil
from functools import partial
from unittest.mock import Mock

import pytest
from langchain_core.tools import tool

from tests.zeroshot.chat_models import ScriptedChatModel, tool_call
from tests.zeroshot.verification.test_interpretation_validation import dxf_case
from tests.zeroshot.verification.test_verify_interpretation import _case
from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.stages._base.prompt import StageInstructions
from zeroshot.pipeline.stages.drawings.contracts import (
    DrawingSheet,
    DrawingSource,
    View,
    unread_sheet,
)
from zeroshot.pipeline.stages.interpretation.contracts import DrawingInterpretation
from zeroshot.pipeline.stages.interpretation.stage import create_interpretation_stage
from zeroshot.pipeline.stages.interpretation.submission import InterpretationSubmission
from zeroshot.pipeline.verification.attempts import AttemptStore
from zeroshot.pipeline.workflow.components.agent import create_agent
from zeroshot.pipeline.workflow.lifecycle import start_reconstruction


def test_stage_requires_written_verified_json_before_ticket_submission(tmp_path):
    verifier, candidate = _case(tmp_path)
    drawing = DrawingSource(
        sheets=[
            DrawingSheet(
                name="sheet_page",
                role=View.FULL_PAGE,
                crop_of=None,
                scale=1,
                file=str(tmp_path / "source.png"),
                evidence=[],
                dimensions=[],
            )
        ]
    )
    response = {
        "responses": [
            {
                "ticket_id": "ticket_initial",
                "stage": "interpretation",
                "summary": "Read view_front and localized sem_pin_upper_left.",
            }
        ]
    }

    @tool("write_interpretation")
    def write_interpretation() -> str:
        """Write the complete candidate artifact."""
        verifier.source_path.write_text(candidate.model_dump_json())
        return "written"

    model = ScriptedChatModel(
        responses=(
            tool_call("InterpretationSubmission", response, "premature"),
            tool_call("write_interpretation", {}, "write"),
            tool_call("InterpretationSubmission", response, "accepted"),
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
    run = start_reconstruction("run_test", "Reconstruct the part.", drawing)
    result = stage.run({"reconstruction": run}, {})
    assert result["stage_submission"] == InterpretationSubmission.model_validate(
        response
    )
    assert stage.verifier.accepted_interpretation is not None
    messages = result["interpretation_state"]["messages"]
    assert any("not ready to submit" in message.text for message in messages)
    assert any('"inliers": "3/3"' in message.text for message in messages)
    assert "DrawingInterpretation JSON schema" in model.received_messages[0][-1].text
    assert "calculate_drawing_scale" in model.bound_tool_name_history[0]
    assert "[Native DXF coordinates]" not in model.received_messages[0][-1].text

    assert (
        DrawingInterpretation.model_validate_json(
            stage.verifier.source_path.read_bytes()
        )
        == stage.verifier.accepted_interpretation
    )
    assert len(model.received_messages) == 3
    assert (
        sum("[Interpretation verification]" in message.text for message in messages)
        == 1
    )


def _dxf_input(tmp_path):
    candidate = dxf_case(tmp_path)
    data = candidate.model_dump()
    data["views"][0]["role"] = "full_page"
    drawing = DrawingSource(
        sheets=[unread_sheet("sheet_front", View.FULL_PAGE, tmp_path / "front.dxf")]
    )
    workdir = SandboxWorkdir(tmp_path)
    return DrawingInterpretation.model_validate(data), drawing, workdir


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
        "responses": [
            {
                "ticket_id": "ticket_initial",
                "stage": "interpretation",
                "summary": "Interpreted view_front and view_detail in millimetres.",
            }
        ]
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
            tool_call("InterpretationSubmission", response, "accepted"),
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
        dxf_mm_per_unit={"view_front": 25.4, "view_detail": 25.4},
    )
    run = start_reconstruction("run_dxf", "Reconstruct the part.", drawing)
    result = stage.run({"reconstruction": run}, {})
    assert result["stage_submission"] == InterpretationSubmission.model_validate(
        response
    )
    text = model.received_messages[0][-1].text
    assert "u = (x - origin_native[0]) * mm_per_unit" in text
    assert "v = (y - origin_native[1]) * mm_per_unit" in text
    metadata = json.loads(text.splitlines()[-1])
    assert metadata["originals"] == [
        {
            "view": "view_front",
            "file": "/work/front.dxf",
            "origin_native": [10, 20],
            "mm_per_unit": 25.4,
            "size_mm": pytest.approx([101.6, 76.2]),
        }
    ]
    assert metadata["configured_mm_per_unit"] == {
        "view_front": 25.4,
        "view_detail": 25.4,
    }
    accepted = stage.verifier.accepted_interpretation
    assert accepted is not None and len(accepted.views) == 2
    assert accepted.views[0].region.box_uv == pytest.approx((0, 0, 101.6, 76.2))
    reports = stage.verifier.verify().reports
    assert reports["view_front"]["status"] == "native_dxf"
    assert reports["view_detail"]["mm_per_unit"] == 25.4


@pytest.mark.parametrize("factors", [None, {"view_detail": 25.4}])
def test_dxf_missing_original_factor_fails_before_building_agent(tmp_path, factors):
    _, drawing, workdir = _dxf_input(tmp_path)
    builder = Mock(side_effect=AssertionError("agent construction must not start"))
    with pytest.raises(
        ValueError, match="DXF input sheet_front requires dxf_mm_per_unit"
    ):
        create_interpretation_stage(
            builder=builder,
            tools=[],
            system_prompt_path=None,
            instructions=StageInstructions(drawing, "path", {}, workdir),
            prompt_context={},
            attempt_store=AttemptStore(workdir, lambda: 0),
            dxf_mm_per_unit=factors,
        )
    builder.assert_not_called()
