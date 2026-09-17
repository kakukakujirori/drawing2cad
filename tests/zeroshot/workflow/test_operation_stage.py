"""The plan lives in the workspace file; the answer carries tickets alone."""

import json
from functools import partial

from langchain_core.tools import tool

from tests.zeroshot.chat_models import ScriptedChatModel, tool_call
from tests.zeroshot.contracts import interpretation, view
from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.stages._base.prompt import StageInstructions
from zeroshot.pipeline.stages.operations.contracts import (
    Operation,
    OperationPlan,
    OperationVerb,
)
from zeroshot.pipeline.stages.operations.stage import create_operation_stage
from zeroshot.pipeline.stages.tickets.contracts import TicketAnswers
from zeroshot.pipeline.stages.types import PipelineStage
from zeroshot.pipeline.verification.attempts import AttemptStore
from zeroshot.pipeline.verification.verify_operations import OperationPlanVerifier
from zeroshot.pipeline.workflow.components.agent import create_agent
from zeroshot.pipeline.workflow.lifecycle import (
    advance_reconstruction,
    start_reconstruction,
)

_ANSWER = {
    "responses": [
        {
            "ticket_id": "ticket_initial",
            "stage": "operations",
            "summary": "Established op_base.",
        }
    ]
}


def _plan(*, builds: str = "sem_feature_1") -> OperationPlan:
    return OperationPlan(
        proposal=[
            Operation(
                name="op_base",
                verb=OperationVerb.EXTRUDE,
                detail="Extrude the base.",
                depends_on=[],
                semantics=[builds],
            )
        ],
        rationale="The base is one extrusion.",
    )


def _interpreted_run(tmp_path):
    page = tmp_path / "front.dxf"
    page.write_text("0\nEOF\n")
    source = [view("front", file=str(page))]
    return source, advance_reconstruction(
        start_reconstruction("run_plan", "Reconstruct the part.", source),
        TicketAnswers.model_validate(
            {
                "responses": [
                    {
                        "ticket_id": "ticket_initial",
                        "stage": "interpretation",
                        "summary": "Established sem_feature_1.",
                    }
                ]
            }
        ),
        workspace_output=interpretation("a plate"),
    )


def _stage(tools, workdir, source, model):
    return create_operation_stage(
        partial(
            create_agent,
            role="operation_planner",
            model=model,
            checkpointer=False,
            max_turns=6,
            announce_turns=False,
            response_format_strategy="tool",
        ),
        tools=tools,
        system_prompt_path=None,
        instructions=StageInstructions(source, "path", {}, workdir),
        prompt_context={},
        attempt_store=AttemptStore(workdir, lambda: 0),
    )


def test_an_invalid_plan_is_refused_until_the_file_validates(tmp_path):
    source, run = _interpreted_run(tmp_path)
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    plans = iter([_plan(builds="sem_absent"), _plan()])

    @tool("write_plan")
    def write_plan() -> str:
        """Write the next candidate plan to the workspace."""
        (tmp_path / "operations.json").write_text(next(plans).model_dump_json())
        return "written"

    model = ScriptedChatModel(
        responses=(
            tool_call("write_plan", {}, "invalid"),
            tool_call("TicketAnswers", _ANSWER, "premature"),
            tool_call("write_plan", {}, "valid"),
            tool_call("TicketAnswers", _ANSWER, "accepted"),
        )
    )
    stage = _stage([write_plan], workdir, source, model)

    result = stage.run({"reconstruction": run}, {})

    assert result["stage_submission"] == TicketAnswers.model_validate(_ANSWER)
    assert stage.operation_verifier.accepted_plan == _plan()
    messages = result["operations_state"]["messages"]
    assert any("sem_absent" in message.text for message in messages)
    assert any("not ready to submit" in message.text for message in messages)
    assert "OperationPlan JSON schema" in model.received_messages[0][-1].text


def test_a_schema_error_points_at_its_key_in_the_plan_file(tmp_path):
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    verifier = OperationPlanVerifier(workdir, AttemptStore(workdir, lambda: 0))
    verifier.reset(None, interpretation("a plate"))
    data = _plan().model_dump(mode="json")
    data["proposal"][0]["revision_note"] = "x"
    text = json.dumps(data, indent=2)
    verifier.source_path.write_text(text)
    error = verifier.feedback()[0]["text"].splitlines()[-1]
    position, rest = error.split(" ", 1)
    _, line, column = position.split(":")
    assert text.splitlines()[int(line) - 1][int(column) - 1 :].startswith(
        '"revision_note"'
    )
    assert (
        rest == "$.proposal[0].revision_note (op_base): Extra inputs are not permitted"
    )


def test_an_unassigned_stage_answers_nothing_and_writes_no_plan(tmp_path):
    source, run = _interpreted_run(tmp_path)
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    run.snapshots[-1].open_tickets[0].assigned_stages = [PipelineStage.CODING]
    model = ScriptedChatModel(responses=())

    stage = _stage([], workdir, source, model)

    assert stage.run({"reconstruction": run}, {}) == {
        "stage_submission": TicketAnswers(responses=[])
    }
    assert model.received_messages == []
    # Round zero has no preceding plan to seed, so the model creates the file.
    assert not (tmp_path / "operations.json").exists()
