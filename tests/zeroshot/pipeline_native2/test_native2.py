"""Run the real coder-auditor loop without calling a provider."""

import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from langchain_core.messages import AIMessage
from PIL import Image

from tests.zeroshot.chat_models import ScriptedChatModel, tool_call
from zeroshot.evaluation.run_scoring import is_native_run
from zeroshot.pipeline_native2 import runner

_ROOT = Path(__file__).resolve().parents[3]


def _config(tmp_path):
    drawing = tmp_path / "drawing.png"
    Image.new("RGB", (16, 16), "white").save(drawing)
    with initialize_config_dir(
        config_dir=str(_ROOT / "zeroshot/configs"), version_base="1.3"
    ):
        return compose(
            config_name="native2",
            overrides=[
                f"artifact_root={tmp_path / 'runs'}",
                "console=false",
                f"sample.drawing.sheets.0.file={drawing}",
            ],
        )


def _write(source, call_id):
    return tool_call("run_shell", {"command": f"echo '{source}' > model.py"}, call_id)


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
    assert "The hole is missing." in revision[-1].text
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


def test_a_missing_or_contradicted_verdict_rejects():
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
