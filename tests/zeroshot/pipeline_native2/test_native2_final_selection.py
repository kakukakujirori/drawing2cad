"""Select submitted native rounds through the real agent loop, without a provider."""

import hashlib
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from tests.zeroshot.chat_models import ScriptedChatModel, tool_call
from tests.zeroshot.pipeline_native2.test_native2 import (
    _accepted_run,
    _config,
    _events,
    _idle,
    _truncated,
    _write,
)
from zeroshot.evaluation.run_scoring import native_step, sandbox_for_builds
from zeroshot.pipeline_native2 import runner


def _rounds():
    return (
        _write("result = 1", "c0"),
        AIMessage(content="Built the plate."),
        AIMessage(content="The hole is missing.\nVERDICT: REJECT"),
        _write("result = 2", "c1"),
        AIMessage(content="Added the hole."),
    )


def _run_dir(config):
    return Path(config.artifact_root) / config.sample.sample_id


@pytest.mark.parametrize("config_name", ["native2", "native2_prime"])
@pytest.mark.parametrize("enabled", [False, True])
def test_agents_are_told_the_submission_path_only_when_it_is_available(
    tmp_path, config_name, enabled
):
    config = _config(tmp_path, config_name)
    config.final_selection = enabled
    model, _ = _accepted_run(config)

    for messages in (model.received_messages[0], model.received_messages[2]):
        system = messages[0].text
        assert ("/work/submissions/round_NNN/model.py" in system) is enabled
        if enabled:
            assert "read-only" in system
        assert "final selection" not in system.lower()
        assert "SELECTED_ROUND" not in system


@pytest.mark.parametrize("chosen", [0, 1])
def test_selects_a_submitted_round_and_leaves_native_builds_to_the_agent(
    tmp_path, chosen
):
    config = _config(tmp_path)
    config.final_selection = True
    report = f"Round {chosen} preserves the main profile.\nSELECTED_ROUND: {chosen}"
    model = ScriptedChatModel(
        responses=(
            *_rounds(),
            AIMessage(content=report),
            AIMessage(content="A report."),
        )
    )
    result = runner.run(config, model=model)

    run_dir = _run_dir(config)
    source = f"result = {chosen + 1}\n"
    assert "status" not in result
    assert (run_dir / "workspace/model.py").read_text() == source
    assert not (run_dir / "workspace/attempts").exists()
    choice = json.loads((run_dir / "final_selection.json").read_text())
    assert choice == {"round": chosen, "reason": report}
    assert json.loads((run_dir / "run.json").read_text())["config"]["final_selection"]
    for number, author in enumerate(["Built the plate.", "Added the hole."]):
        directory = run_dir / f"workspace/submissions/round_{number:03d}"
        assert (directory / "model.py").read_text() == f"result = {number + 1}\n"
        assert json.loads((directory / "submission.json").read_text()) == {
            "round": number,
            "author_message": author,
            "build_report": [],
        }
    selection = model.received_messages[5]
    assert any(b["type"] == "image" for b in selection[1].content_blocks)
    assert "/work/submissions/round_000/model.py" in selection[1].text
    assert "/work/submissions/round_001/model.py" in selection[1].text
    assert not any("The hole is missing." in m.text for m in selection)
    assert selection[-1].text == "[turn 1/20]"
    assert model.bound_tool_name_history[-1] == ("run_shell", "load_image")
    assert (run_dir / "selection_messages.json").is_file()
    events = _events(config)
    assert [e["data"] for e in events if e["event"] == "final_selection"] == [choice]
    assert events[-1]["data"]["coder_rounds"] == 2
    assert (
        events[-1]["data"]["model_sha256"]
        == hashlib.sha256(source.encode()).hexdigest()
    )


@pytest.mark.parametrize("max_audits", [0, 1])
def test_one_candidate_needs_no_selector_call(tmp_path, max_audits):
    config = _config(tmp_path)
    config.final_selection = True
    config.max_audits = max_audits
    model = ScriptedChatModel(
        responses=(
            *_rounds()[:2],
            *([AIMessage(content="Matches.\nVERDICT: ACCEPT")] if max_audits else []),
            AIMessage(content="A report."),
        )
    )
    runner.run(config, model=model)

    run_dir = _run_dir(config)
    assert json.loads((run_dir / "final_selection.json").read_text())["round"] == 0
    assert not (run_dir / "selection_messages.json").exists()
    assert not any(
        e["event"] == "prompt" and e["data"]["agent"] == "final_selector"
        for e in _events(config)
    )


@pytest.mark.parametrize(
    "answer", ["No decision.", "SELECTED_ROUND: 99", "SELECTED_ROUND: -1"]
)
def test_invalid_choices_get_the_auditors_grace_turns(tmp_path, answer):
    config = _config(tmp_path)
    config.final_selection = True
    config.auditor_max_turns = 3
    config.retry_turns = 2
    config.max_stage_validation_retries = 1
    model = ScriptedChatModel(
        responses=(
            *_rounds(),
            _idle("s0"),
            _idle("s1"),
            AIMessage(content=answer),
            AIMessage(content="Earlier is closer.\n**SELECTED_ROUND:** 0"),
            AIMessage(content="A report."),
        )
    )
    result = runner.run(config, model=model)

    assert "status" not in result
    retry = model.received_messages[8]
    assert any(m.text == answer for m in retry)
    assert any("candidate round from [0, 1]" in m.text for m in retry)
    assert retry[-1].text.startswith("[turn 2/3]")
    assert [
        (e["data"]["agent"], e["data"]["total_turns"])
        for e in _events(config)
        if e["event"] == "stage_validation_retry"
    ] == [("final_selector", 3)]


@pytest.mark.parametrize("exhausted", [False, True])
def test_failed_selection_preserves_the_last_submission_and_stops(tmp_path, exhausted):
    config = _config(tmp_path)
    config.final_selection = True
    config.auditor_max_turns = 2
    config.max_stage_validation_retries = 0
    model = ScriptedChatModel(
        responses=(
            *_rounds(),
            *(
                [_idle("s0"), _idle("s1")]
                if exhausted
                else [AIMessage(content="Unsure.")]
            ),
        )
    )
    result = runner.run(config, model=model)

    status = "turn_limit_exceeded" if exhausted else "stage_validation_failed"
    assert result["status"] == status
    run_dir = _run_dir(config)
    assert (run_dir / "workspace/model.py").read_text() == "result = 2\n"
    assert not (run_dir / "final_selection.json").exists()
    assert (run_dir / "selection_messages.json").is_file()
    assert _events(config)[-1]["data"]["status"] == status


def test_selection_cannot_edit_the_candidates_or_the_workspace_program(tmp_path):
    config = _config(tmp_path)
    config.final_selection = True
    model = ScriptedChatModel(
        responses=(
            *_rounds(),
            _write("result = 99", "s0"),
            tool_call(
                "run_shell",
                {"command": "echo changed > submissions/round_000/model.py"},
                "s1",
            ),
            AIMessage(content="Earlier is closer.\nSELECTED_ROUND: 0"),
            AIMessage(content="A report."),
        )
    )
    runner.run(config, model=model)

    run_dir = _run_dir(config)
    assert (run_dir / "workspace/model.py").read_text() == "result = 1\n"
    assert (
        run_dir / "workspace/submissions/round_000/model.py"
    ).read_text() == "result = 1\n"
    assert (
        run_dir / "workspace/submissions/round_001/model.py"
    ).read_text() == "result = 2\n"
    results = [m for m in model.received_messages[7] if isinstance(m, ToolMessage)]
    assert len(results) == 2
    assert all(json.loads(m.content)["returncode"] != 0 for m in results)


def test_selector_output_limits_use_its_own_reminder(tmp_path):
    config = _config(tmp_path)
    config.final_selection = True
    model = ScriptedChatModel(
        responses=(
            *_rounds(),
            _truncated(),
            AIMessage(content="Earlier is closer.\nSELECTED_ROUND: 0"),
            AIMessage(content="A report."),
        )
    )
    runner.run(config, model=model)

    assert "SELECTED_ROUND: N" in model.received_messages[6][-1].text
    assert not any(e["event"] == "stage_validation_retry" for e in _events(config))


def test_an_unfinished_coder_round_is_not_a_candidate_or_a_selection_trigger(tmp_path):
    config = _config(tmp_path)
    config.final_selection = True
    config.max_turns = 2
    config.max_stage_validation_retries = 0
    model = ScriptedChatModel(responses=(*_rounds()[:4], _idle("unfinished")))
    result = runner.run(config, model=model)

    assert result["status"] == "turn_limit_exceeded"
    run_dir = _run_dir(config)
    assert [p.name for p in (run_dir / "workspace/submissions").iterdir()] == [
        "round_000"
    ]
    assert not (run_dir / "final_selection.json").exists()


def test_native2_prime_keeps_build_evidence_with_each_submission(tmp_path):
    config = _config(tmp_path, "native2_prime")
    config.final_selection = True
    model = ScriptedChatModel(
        responses=(
            *_rounds(),
            AIMessage(content="Earlier is closer.\nSELECTED_ROUND: 0"),
            AIMessage(content="A report."),
        )
    )
    runner.run(config, model=model)

    run_dir = _run_dir(config)
    for number in range(2):
        submission = json.loads(
            (
                run_dir / f"workspace/submissions/round_{number:03d}/submission.json"
            ).read_text()
        )
        assert submission["build_report"][0]["text"] == "[Execution result]"
        build = run_dir / f"workspace/attempts/round_{number:03d}/coding/000"
        assert (build / "model.py").read_text() == f"result = {number + 1}\n"
    assert "### Model frame" in model.received_messages[5][0].text


def test_native_scoring_builds_the_selected_program(tmp_path):
    import cadquery as cq

    config = _config(tmp_path)
    config.final_selection = True
    model = ScriptedChatModel(
        responses=(
            _write("import cadquery as cq; result = cq.Workplane().box(1, 2, 3)", "c0"),
            AIMessage(content="Built it."),
            AIMessage(content="Check the width.\nVERDICT: REJECT"),
            _write("import cadquery as cq; result = cq.Workplane().box(4, 2, 3)", "c1"),
            AIMessage(content="Changed the width."),
            AIMessage(content="Earlier width was correct.\nSELECTED_ROUND: 0"),
            AIMessage(content="A report."),
        )
    )
    runner.run(config, model=model)

    run_dir = _run_dir(config)
    step, errors, status = native_step(run_dir, sandbox_for_builds(run_dir))
    assert status == "VERIFIED", errors
    assert cq.importers.importStep(str(step)).val().Volume() == pytest.approx(6)


@pytest.mark.parametrize("answer", ["SELECTED_ROUND: 0.5", "SELECTED_ROUND: 0/1"])
def test_the_choice_must_be_an_integer(answer):
    assert runner._selected_round(answer) is None
