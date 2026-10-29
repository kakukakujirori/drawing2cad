"""Run the real coder-auditor loop without calling a provider."""

import json
import re
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from langchain_core.messages import AIMessage
from PIL import Image

from tests.zeroshot.chat_models import (
    ScriptedChatModel,
    tool_call,
    unanswered_tool_calls,
)
from zeroshot.evaluation.aggregate_run import collect, summarize
from zeroshot.evaluation.run_scoring import is_native_run
from zeroshot.pipeline_native2 import runner

_ROOT = Path(__file__).resolve().parents[3]


def _config(tmp_path, config_name="native2"):
    drawing = tmp_path / "drawing.png"
    Image.new("RGB", (16, 16), "white").save(drawing)
    with initialize_config_dir(
        config_dir=str(_ROOT / "zeroshot/configs"), version_base="1.3"
    ):
        return compose(
            config_name=config_name,
            overrides=[
                f"artifact_root={tmp_path / 'runs'}",
                "console=false",
                f"sample.drawing.sheets.0.file={drawing}",
            ],
        )


def _write(source, call_id):
    return tool_call("run_shell", {"command": f"echo '{source}' > model.py"}, call_id)


def _idle(call_id):
    return tool_call("run_shell", {"command": "true"}, call_id)


def _events(config):
    path = Path(config.artifact_root) / config.sample.sample_id / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_rejection_sends_the_report_to_the_coder_and_protects_its_program(tmp_path):
    config = _config(tmp_path)
    model = ScriptedChatModel(
        responses=(
            _write("result = 1", "c0"),
            AIMessage(content="Built the plate."),
            _write("result = 2", "a0"),
            AIMessage(content="The hole is missing.\n**VERDICT: REJECT**"),
            _write("result = 3", "c1"),
            AIMessage(content="Added the hole."),
            AIMessage(content="A report."),
        )
    )
    runner.run(config, model=model)

    run_dir = Path(config.artifact_root) / config.sample.sample_id
    assert (run_dir / "workspace/model.py").read_text() == "result = 3\n"
    auditor_first, revision = model.received_messages[2], model.received_messages[4]
    assert "VERDICT: ACCEPT" in auditor_first[0].text
    assert any(block.get("type") == "image" for block in auditor_first[1].content)
    assert "Built the plate." in auditor_first[1].content[-1]["text"]
    assert any("The hole is missing." in m.text for m in revision)
    events = [json.loads(line) for line in (run_dir / "events.jsonl").open()]
    names = [event["event"] for event in events]
    assert "model_restored" in names  # the auditor's edit was undone
    assert {"round": 0, "accepted": False} in [
        event["data"] for event in events if event["event"] == "audit_verdict"
    ]
    assert events[-1]["data"]["coder_rounds"] == 2
    assert (run_dir / "audit_messages.json").is_file()
    assert (run_dir / "reasoning_traj.md").read_text() == "A report.\n"
    assert is_native_run(run_dir)


def test_acceptance_ends_the_run(tmp_path):
    config = _config(tmp_path)
    model = ScriptedChatModel(
        responses=(
            _write("result = 1", "c0"),
            AIMessage(content="Built it."),
            AIMessage(content="Matches.\nVERDICT: ACCEPT"),
            AIMessage(content="A report."),
        )
    )
    runner.run(config, model=model)

    events = (Path(config.artifact_root) / "000364/events.jsonl").read_text()
    assert json.loads(events.splitlines()[-1])["data"]["coder_rounds"] == 1


def test_only_a_final_accept_verdict_is_accepted():
    assert runner.accepted("fine\nVERDICT: ACCEPT")
    assert not runner.accepted("fine")
    assert not runner.accepted("VERDICT: ACCEPT, no wait\nVERDICT: REJECT")
    assert runner.accepted("fine\n**Verdict:** ACCEPT")


def _truncated():
    return AIMessage(
        content=[{"type": "reasoning", "reasoning": "Unfinished."}],
        response_metadata={"finish_reason": "length", "id": "gen-cut"},
    )


def test_output_limits_are_retried_with_each_agents_reminder(tmp_path):
    config = _config(tmp_path)
    config.max_turns = 2
    config.auditor_max_turns = 1
    model = ScriptedChatModel(
        responses=(
            _truncated(),
            _write("result = 1", "c0"),
            AIMessage(content="Built it."),
            _truncated(),
            AIMessage(content="Matches.\nVERDICT: ACCEPT"),
            AIMessage(content="A report."),
        )
    )
    runner.run(config, model=model)

    assert "/work/model.py now" in model.received_messages[1][-1].text
    assert "verdict line" in model.received_messages[4][-1].text
    assert not any(e["event"] == "stage_validation_retry" for e in _events(config))


@pytest.mark.parametrize(
    ("max_turns", "retry_notice", "rewinds"),
    [(3, "[turn 2/3]", 1), (6, "[turn 4/6]", 0)],
)
def test_coder_grace_preserves_history_and_counts_as_one_round(
    tmp_path, max_turns, retry_notice, rewinds
):
    config = _config(tmp_path)
    config.max_audits = 0
    config.max_turns = max_turns
    config.retry_turns = 2
    model = ScriptedChatModel(
        responses=(
            _idle("c0"),
            _idle("c1"),
            AIMessage(content="Still measuring."),
            _write("result = 1", "c2"),
            AIMessage(content="Built it."),
            AIMessage(content="A report."),
        )
    )
    result = runner.run(config, model=model)

    assert "status" not in result
    retry = model.received_messages[3]
    assert any(m.text == "Still measuring." for m in retry)
    assert sum("turn set back" in m.text for m in retry) == rewinds
    assert retry[-1].text.startswith(retry_notice)
    assert not unanswered_tool_calls(result["messages"])
    events = _events(config)
    retries = [e["data"] for e in events if e["event"] == "stage_validation_retry"]
    assert [(r["agent"], r["retry"], r["total_turns"]) for r in retries] == [
        ("coder", 1, 3)
    ]
    assert events[-1]["data"]["coder_rounds"] == 1
    assert summarize(collect(Path(config.artifact_root))).budget_exhausted == 0


def test_an_unfinished_tool_turn_gets_grace_even_with_an_existing_model(tmp_path):
    config = _config(tmp_path)
    config.max_audits = 0
    config.max_turns = 2
    config.retry_turns = 1
    model = ScriptedChatModel(
        responses=(
            _write("result = 1", "c0"),
            _idle("c1"),
            AIMessage(content="Built it."),
            AIMessage(content="A report."),
        )
    )
    result = runner.run(config, model=model)

    assert "status" not in result
    assert not unanswered_tool_calls(result["messages"])
    retries = [e for e in _events(config) if e["event"] == "stage_validation_retry"]
    assert len(retries) == 1
    assert "budget ran out" in retries[0]["data"]["error"]


@pytest.mark.parametrize("retries", [0, 3])
@pytest.mark.parametrize("artifact", ["missing", "symlink"])
@pytest.mark.parametrize("unfinished_tool", [False, True])
def test_coder_grace_has_a_cap_and_keeps_the_failure_scoring_contract(
    tmp_path, retries, artifact, unfinished_tool
):
    config = _config(tmp_path)
    config.max_audits = 0
    config.max_turns = 3
    config.retry_turns = 2
    config.max_stage_validation_retries = retries
    last = _idle("c2") if unfinished_tool else AIMessage(content="Still measuring.")
    responses = [_idle("c0"), _idle("c1"), last]
    if artifact == "symlink":
        responses[0] = tool_call(
            "run_shell", {"command": "ln -s inputs/view_drawing.png model.py"}, "c0"
        )
    for index in range(retries):
        last = (
            _idle(f"r{index}-last")
            if unfinished_tool
            else AIMessage(content="Still measuring.")
        )
        responses.extend([_idle(f"r{index}"), last])
    model = ScriptedChatModel(responses=tuple(responses))
    result = runner.run(config, model=model)

    status = "turn_limit_exceeded" if unfinished_tool else "stage_validation_failed"
    assert result["status"] == status
    assert len(model.received_messages) == 3 + 2 * retries
    events = _events(config)
    retry_events = [e for e in events if e["event"] == "stage_validation_retry"]
    assert len(retry_events) == retries
    assert events[-2]["event"] == status
    assert events[-2]["data"]["turn_count"] == 3 + 2 * retries
    assert events[-1]["data"]["status"] == status
    run_dir = Path(config.artifact_root) / config.sample.sample_id
    assert "result = None" in (run_dir / "workspace/model.py").read_text()
    assert not (run_dir / "workspace/model.py").is_symlink()
    assert (run_dir / "workspace/inputs/view_drawing.png").read_bytes() == (
        tmp_path / "drawing.png"
    ).read_bytes()
    assert (run_dir / "messages.json").is_file()
    assert not (run_dir / "reasoning_traj.md").exists()
    assert summarize(collect(Path(config.artifact_root))).budget_exhausted == int(
        unfinished_tool
    )


def test_a_new_coder_round_gets_fresh_turns_and_retries(tmp_path):
    config = _config(tmp_path)
    config.max_turns = 3
    config.retry_turns = 2
    config.max_stage_validation_retries = 1
    model = ScriptedChatModel(
        responses=(
            _idle("c0"),
            _idle("c1"),
            AIMessage(content="Still measuring."),
            _write("result = 1", "c2"),
            AIMessage(content="Built it."),
            AIMessage(content="The hole is missing.\nVERDICT: REJECT"),
            tool_call("run_shell", {"command": "mv model.py previous.py"}, "c3"),
            _idle("c4"),
            AIMessage(content="Revising."),
            _write("result = 3", "c5"),
            AIMessage(content="Added the hole."),
            AIMessage(content="A report."),
        )
    )
    result = runner.run(config, model=model)

    assert "status" not in result
    assert model.received_messages[6][-1].text == "[turn 1/3]"
    assert any("The hole is missing." in m.text for m in model.received_messages[6])
    retries = [
        e["data"] for e in _events(config) if e["event"] == "stage_validation_retry"
    ]
    assert [(r["round"], r["retry"], r["total_turns"]) for r in retries] == [
        (0, 1, 3),
        (1, 1, 3),
    ]


def test_auditor_retries_a_missing_verdict_and_then_rejects_normally(tmp_path):
    config = _config(tmp_path)
    config.auditor_max_turns = 3
    config.retry_turns = 2
    model = ScriptedChatModel(
        responses=(
            _write("result = 1", "c0"),
            AIMessage(content="Built it."),
            _write("result = 99", "a0"),
            _idle("a1"),
            AIMessage(content="The hole is missing."),
            AIMessage(content="The hole is missing.\nVERDICT: REJECT"),
            _write("result = 3", "c1"),
            AIMessage(content="Added it."),
            AIMessage(content="A report."),
        )
    )
    result = runner.run(config, model=model)

    assert "status" not in result
    retry = model.received_messages[5]
    assert any(m.text == "The hole is missing." for m in retry)
    assert any("VERDICT: ACCEPT or VERDICT: REJECT" in m.text for m in retry[1:])
    assert retry[-1].text.startswith("[turn 2/3]")
    events = _events(config)
    assert [
        e["data"]["agent"] for e in events if e["event"] == "stage_validation_retry"
    ] == ["auditor"]
    assert any(e["event"] == "model_restored" for e in events)
    assert events[-1]["data"]["coder_rounds"] == 2


def test_an_incomplete_final_audit_preserves_the_coders_program_and_stops(tmp_path):
    config = _config(tmp_path)
    config.auditor_max_turns = 3
    config.retry_turns = 2
    config.max_stage_validation_retries = 1
    model = ScriptedChatModel(
        responses=(
            _write("result = 1", "c0"),
            AIMessage(content="Built it."),
            _write("result = 99", "a0"),
            _idle("a1"),
            AIMessage(content="Still checking."),
            _idle("a2"),
            AIMessage(content="Still checking."),
        )
    )
    result = runner.run(config, model=model)

    assert result["status"] == "stage_validation_failed"
    run_dir = Path(config.artifact_root) / config.sample.sample_id
    assert (run_dir / "workspace/model.py").read_text() == "result = 1\n"
    assert (run_dir / "audit_messages.json").is_file()
    events = _events(config)
    assert not any(e["event"] == "audit_verdict" for e in events)
    assert events[-1]["data"]["coder_rounds"] == 1


@pytest.mark.parametrize("role", ["coder", "auditor"])
@pytest.mark.parametrize("retries", [0, 3])
def test_an_incomplete_stage_stops_after_retries_even_with_turns_left(
    tmp_path, role, retries
):
    config = _config(tmp_path)
    config.max_stage_validation_retries = retries
    responses = []
    if role == "auditor":
        responses.extend([_write("result = 1", "c0"), AIMessage(content="Built it.")])
    responses.extend(AIMessage(content="Still checking.") for _ in range(retries + 1))
    model = ScriptedChatModel(responses=tuple(responses))
    result = runner.run(config, model=model)

    assert result["status"] == "stage_validation_failed"
    assert len(model.received_messages) == len(responses)
    events = _events(config)
    assert not any(e["event"] == "audit_verdict" for e in events)
    assert not any(e["event"] == "turn_limit_exceeded" for e in events)
    assert events[-2]["event"] == "stage_validation_failed"
    assert events[-2]["data"]["agent"] == role
    assert events[-2]["data"]["turn_count"] == retries + 1
    assert events[-1]["data"]["status"] == "stage_validation_failed"
    assert events[-1]["data"]["coder_rounds"] == 1
    run_dir = Path(config.artifact_root) / config.sample.sample_id
    expected = "result = 1\n" if role == "auditor" else "result = None"
    assert expected in (run_dir / "workspace/model.py").read_text()
    assert not (run_dir / "reasoning_traj.md").exists()


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("max_turns", 0),
        ("auditor_max_turns", 0),
        ("retry_turns", 0),
        ("max_stage_validation_retries", -1),
    ],
)
def test_invalid_turn_settings_fail_before_creating_a_workspace(
    tmp_path, setting, value
):
    config = _config(tmp_path)
    config[setting] = value
    with pytest.raises(ValueError):
        runner.run(config, model=ScriptedChatModel(responses=()))
    assert not Path(config.artifact_root).exists()


def test_the_auditor_reads_the_drawing_as_the_coder_does():
    """Its length and view conventions follow the prompts they were copied from."""
    staged = (
        _ROOT / "zeroshot/pipeline/stages/_base/prompts/reconstruction_context.md"
    ).read_text()
    (paragraph,) = [p for p in staged.split("\n\n") if p.startswith("Length")]
    auditor = (_ROOT / "zeroshot/pipeline_native2/prompts/auditor.md").read_text()
    assert paragraph.strip() in auditor
    coder = (_ROOT / "zeroshot/pipeline_native/prompts/generator.md").read_text()
    views = "### Which view is which?"
    assert auditor[auditor.index(views) :] == coder[coder.index(views) :]


def test_a_failed_audit_still_leaves_the_coders_program(tmp_path):
    config = _config(tmp_path)
    model = ScriptedChatModel(
        responses=(
            _write("result = 1", "c0"),
            AIMessage(content="Built it."),
            _write("result = 2", "a0"),
        )
    )
    with pytest.raises(AssertionError, match="ran out of responses"):
        runner.run(config, model=model)

    run_dir = Path(config.artifact_root) / config.sample.sample_id
    assert (run_dir / "workspace/model.py").read_text() == "result = 1\n"


def _accepted_run(config):
    model = ScriptedChatModel(
        responses=(
            _write("result = 1", "c0"),
            AIMessage(content="Built it."),
            AIMessage(content="Matches.\nVERDICT: ACCEPT"),
            AIMessage(content="A report."),
        )
    )
    runner.run(config, model=model)
    return model, Path(config.artifact_root) / config.sample.sample_id


def _texts(message):
    return [block.get("text", "") for block in message.content_blocks]


def test_native2_neither_builds_nor_describes_builds(tmp_path):
    model, run_dir = _accepted_run(_config(tmp_path))

    coder_system = model.received_messages[1][0].text
    assert coder_system == runner.PromptTemplate(runner._CODER_PROMPT).render(
        projection_placement=runner.describe_placement(
            runner.ProjectionMethod.THIRD_ANGLE
        )
    )
    assert not any(
        "[Execution result]" in text
        for messages in model.received_messages
        for message in messages
        for text in _texts(message)
    )
    assert not (run_dir / "workspace/attempts").exists()


def test_native2_prime_builds_each_written_program_and_reports_it_to_the_auditor(
    tmp_path,
):
    model, run_dir = _accepted_run(_config(tmp_path, "native2_prime"))

    coder, auditor = model.received_messages[1], model.received_messages[2]
    assert "### Automatic verification" in coder[0].text
    assert "### Model frame" in coder[0].text
    assert "### Model frame" in auditor[0].text
    assert "### Automatic verification" not in auditor[0].text
    # The build follows the coder's write and, as in the staged pipeline, the
    # announcement of its next turn.
    assert coder[-2].text == "[turn 2/60]"
    assert _texts(coder[-1])[0] == "[Execution result]"
    request = next(m for m in auditor if "now builds" in m.text)
    texts = _texts(request)
    report = texts.index("The pipeline's verification report describes this build.")
    assert texts[report + 1] == "[Execution result]"
    assert '"verification_id":"000"' in texts[report + 2]
    # The audit reuses that build of the unchanged program.
    attempts = run_dir / "workspace/attempts/round_000/coding"
    assert [path.name for path in attempts.iterdir()] == ["000"]


def test_native2_prime_takes_its_model_frame_from_the_staged_pipeline():
    frames = (
        _ROOT / "zeroshot/pipeline/stages/_base/prompts/coordinate_frames.md"
    ).read_text()
    native = (_ROOT / "zeroshot/pipeline_native2/prompts/model_frame.md").read_text()
    table = [line for line in native.splitlines() if line.startswith("|")]
    assert len(table) == 8
    assert "\n".join(table) in frames
    # Every sentence but the heading and the table's own introduction is copied.
    prose = [
        sentence
        for paragraph in native.split("\n\n")
        if not paragraph.startswith(("#", "|"))
        for sentence in re.split(r"(?<=\.) ", paragraph.strip())
        if not sentence.endswith(":")
    ]
    assert len(prose) == 4
    assert [sentence for sentence in prose if sentence not in frames] == []


def test_native2_prime_describes_builds_as_the_staged_coder_round_does():
    staged = (_ROOT / "zeroshot/pipeline/stages/coding/prompts/round.md").read_text()
    native = (
        _ROOT / "zeroshot/pipeline_native2/prompts/automatic_build.md"
    ).read_text()
    for shared in (
        "automatic verification",
        (
            "returns diagnostics and paths to STEP, orthographic PNG/DXF and "
            "perspective renders under"
        ),
        "This feedback finds misreadings faster than further measuring by hand.",
        "Verification uses no additional model turn.",
    ):
        assert shared in native and shared in staged
