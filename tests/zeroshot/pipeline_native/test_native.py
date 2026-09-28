"""Exercise the real native loop without calling a provider."""

import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from langchain.agents import create_agent
from langchain.agents.middleware import ToolErrorMiddleware
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from PIL import Image
from pydantic import PrivateAttr

from tests.zeroshot.chat_models import ScriptedChatModel, tool_call
from zeroshot.pipeline.workflow.middleware.stateless_reasoning import (
    StatelessReasoningMiddleware,
)
from zeroshot.pipeline_native import runner

_ROOT = Path(__file__).resolve().parents[3]


def _config(tmp_path):
    drawing = tmp_path / "drawing.png"
    Image.new("RGB", (16, 16), "white").save(drawing)
    with initialize_config_dir(
        config_dir=str(_ROOT / "zeroshot/configs"), version_base="1.3"
    ):
        config = compose(
            config_name="workflow/native",
            overrides=[
                f"artifact_root={tmp_path / 'runs'}",
                "console=false",
                f"sample.drawing.sheets.0.file={drawing}",
            ],
        )
    return config


def _events(config, name="events.jsonl"):
    path = Path(config.artifact_root) / config.sample.sample_id / name
    return [json.loads(line) for line in path.read_text().splitlines()]


class _ToolAwareModel(ScriptedChatModel):
    _request_tools: list[tuple[str, ...]] = PrivateAttr(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        super().bind_tools(tools, **kwargs)
        return self.bind(_native_tools=self.bound_tool_names)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._request_tools.append(kwargs.pop("_native_tools", ()))
        return super()._generate(messages, stop, run_manager, **kwargs)


def test_native_runs_two_tools_preserves_reasoning_and_requires_model_file(
    tmp_path, monkeypatch
):
    created = []
    assert runner.create_agent is create_agent

    def create_native_agent(**kwargs):
        created.append(kwargs)
        return create_agent(**kwargs)

    monkeypatch.setattr(runner, "create_agent", create_native_agent)
    config = _config(tmp_path)
    script = 'import cadquery as cq\npart = cq.Workplane("XY").box(1, 2, 3)\n'
    first = tool_call("load_image", {"image_path": "/work/missing.png"}, "bad-image")
    first.content = [
        {
            "type": "reasoning",
            "id": "rs_fixture",
            "summary": [
                {"type": "summary_text", "text": "Read the drawing."},
                {"type": "summary_text", "text": "Choose a box."},
            ],
        }
    ]
    retrospective = "I used the supplied drawing to choose the dimensions of a box."
    model = _ToolAwareModel(
        responses=(
            first,
            tool_call(
                "load_image", {"image_path": "/work/inputs/view_drawing.png"}, "image"
            ),
            tool_call(
                "run_shell",
                {"command": "cat > model.py <<'PY'\n" + script + "PY"},
                "write",
            ),
            AIMessage(
                content="Saved model.py.",
                additional_kwargs={"reasoning_content": "provider summary"},
                response_metadata={"finish_reason": "stop"},
            ),
            AIMessage(
                content=[
                    {"type": "reasoning", "reasoning": "retrospective summary"},
                    {"type": "text", "text": retrospective},
                ]
            ),
        )
    )
    result = runner.run(config, model=model)
    assert [tuple(type(m) for m in call["middleware"]) for call in created] == [
        (ToolErrorMiddleware, StatelessReasoningMiddleware),
        (StatelessReasoningMiddleware,),
    ]
    assert [[tool.name for tool in call["tools"]] for call in created] == [
        ["run_shell", "load_image"],
        [],
    ]
    assert all(call.get("response_format") is None for call in created)
    directory = Path(config.artifact_root) / config.sample.sample_id
    assert (directory / "workspace/model.py").read_text() == script
    assert {names for names in model.bound_tool_name_history} == {
        ("run_shell", "load_image")
    }
    assert len(model.received_messages) == 5
    assert model._request_tools == [("run_shell", "load_image")] * 4 + [()]
    assert any(
        getattr(message, "tool_call_id", "") == "bad-image"
        and "Cannot access image" in message.content
        for message in model.received_messages[1]
    )
    assert (
        result["messages"][-1].additional_kwargs["reasoning_content"]
        == "provider summary"
    )
    events = _events(config)
    serialized = json.dumps(events)
    assert events[-1]["event"] == "run_completed"
    assert "Read the drawing." in serialized and "Choose a box." in serialized
    assert "provider summary" in serialized
    assert "data:image/png;base64," not in serialized
    saved_generation = json.loads((directory / "messages.json").read_text())
    assert saved_generation[-1]["content"] == "Saved model.py."
    assert events[-1]["data"]["model_sha256"] == sha256(script.encode()).hexdigest()

    report_history = [
        message for message in model.received_messages[-1] if message.type != "system"
    ]
    assert report_history[:-1] == result["messages"]
    assert report_history[-1].type == "human"
    report_request = report_history[-1].content
    assert script in "\n".join(block["text"] for block in report_request)
    for name in ("interpreter", "planner", "coder"):
        assert name not in str(report_request).lower()
    for history in model.received_messages[:-1]:
        assert report_request not in [message.content for message in history]
        assert not any(
            word in str(message.content).lower()
            for message in history
            for word in (
                "retrospective",
                "reasoning_traj",
                "interpreter",
                "planner",
                "coder",
            )
        )
    assert (
        report_history[0].content[1]["image_url"]["url"].startswith("data:image/png;")
    )
    assert "retrospective summary" not in serialized
    assert (directory / "reasoning_traj.md").read_text().strip() == retrospective
    assert not (directory / "workspace/reasoning_traj.md").exists()
    report_messages = json.loads(
        (directory / "retrospective_messages.json").read_text()
    )
    assert [message["type"] for message in report_messages] == ["human", "ai"]
    assert report_messages[0]["content"] == report_request
    report_events = _events(config, "retrospective_events.jsonl")
    assert any(event["event"] == "prompt" for event in report_events)
    assert report_events[-1]["event"] == "retrospective_completed"
    assert "retrospective summary" in json.dumps(report_events)
    assert not any(event["event"] == "tools" for event in report_events)
    assert events[-1]["timestamp"] <= report_events[0]["timestamp"]
    assert not (directory / "workspace/attempts").exists()
    assert not (directory / "checkpoints.sqlite").exists()
    assert all("[turn " not in str(messages) for messages in model.received_messages)
    before = (directory / "events.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        runner.run(config, model=model)
    assert (directory / "events.jsonl").read_bytes() == before


def test_missing_model_is_recorded_without_a_reask(tmp_path):
    config = _config(tmp_path)
    model = ScriptedChatModel(responses=(AIMessage(content="Done."),))
    with pytest.raises(FileNotFoundError, match="model.py"):
        runner.run(config, model=model)
    assert len(model.received_messages) == 1
    assert _events(config)[-1]["event"] == "run_failed"
    assert (
        Path(config.artifact_root) / config.sample.sample_id / "messages.json"
    ).is_file()
    assert not (
        Path(config.artifact_root)
        / config.sample.sample_id
        / "retrospective_events.jsonl"
    ).exists()


class _BrokenStream(ScriptedChatModel):
    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content=[
                    {
                        "type": "reasoning",
                        "reasoning": "partial summary before disconnect",
                    }
                ]
            )
        )
        raise RuntimeError("fixture disconnect")


def test_stream_failure_leaves_partial_summary_with_console_off(tmp_path):
    config = _config(tmp_path)
    with pytest.raises(RuntimeError, match="fixture disconnect"):
        runner.run(config, model=_BrokenStream(responses=()))
    events = _events(config)
    assert events[-1]["event"] == "run_failed"
    assert "partial summary before disconnect" in json.dumps(events)
    assert not any(event["event"] == "run_completed" for event in events)
    assert not (
        Path(config.artifact_root)
        / config.sample.sample_id
        / "retrospective_events.jsonl"
    ).exists()


class _BrokenRetrospective(_ToolAwareModel):
    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        if self._response_index < len(self.responses):
            response = self._generate(messages, stop, run_manager, **kwargs)
            message = response.generations[0].message
            yield ChatGenerationChunk(
                message=AIMessageChunk(**message.model_dump(exclude={"type"}))
            )
            return
        self._request_tools.append(kwargs.pop("_native_tools", ()))
        self._received_messages.append(list(messages))
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content=[
                    {
                        "type": "reasoning",
                        "reasoning": "partial retrospective before disconnect",
                    },
                ]
            )
        )
        raise RuntimeError("retrospective disconnect")


def test_retrospective_failure_preserves_completed_generation_and_partial_report(
    tmp_path,
):
    config = _config(tmp_path)
    script = "result = None\n"
    model = _BrokenRetrospective(
        responses=(
            tool_call(
                "run_shell",
                {"command": "cat > model.py <<'PY'\n" + script + "PY"},
                "write",
            ),
            AIMessage(content="Saved model.py."),
        )
    )
    with pytest.raises(RuntimeError, match="retrospective disconnect"):
        runner.run(config, model=model)
    directory = Path(config.artifact_root) / config.sample.sample_id
    assert (directory / "workspace/model.py").read_text() == script
    saved = json.loads((directory / "messages.json").read_text())
    assert AIMessage.model_validate(saved[-1]).text == "Saved model.py."
    events = _events(config)
    assert events[-1]["event"] == "run_completed"
    assert events[-1]["data"]["model_sha256"] == sha256(script.encode()).hexdigest()
    assert not any(event["event"] == "run_failed" for event in events)
    report_events = _events(config, "retrospective_events.jsonl")
    assert report_events[-1]["event"] == "retrospective_failed"
    assert "partial retrospective before disconnect" in json.dumps(report_events)
    assert not any(
        event["event"] == "retrospective_completed" for event in report_events
    )
    assert model._request_tools == [("run_shell", "load_image")] * 2 + [()]
    assert len(model.received_messages) == 3


def test_native_imports_without_shared_runner_or_logging():
    script = """
import importlib.abc
import sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(
            fullname == blocked or fullname.startswith(blocked + ".")
            for blocked in ("zeroshot.pipeline.runner", "zeroshot.pipeline.event_logging")
        ):
            raise AssertionError("Unexpected pipeline dependency: " + fullname)
sys.meta_path.insert(0, Guard())
from zeroshot.pipeline_native.runner import run
from hydra import compose, initialize_config_dir
from hydra.utils import get_class
from pathlib import Path
with initialize_config_dir(config_dir=str(Path("zeroshot/configs").resolve()), version_base="1.3"):
    gpt = compose(config_name="workflow/native")
    assert gpt.model.model == "gpt-6-luna"
    get_class(gpt.model._target_)
    for model in (
        "gpt6_sol_codex",
        "gpt6_astra_codex",
        "opus5.5_openrouter",
        "sonnet5_openrouter",
        "glm5.3_flash_openrouter",
    ):
        cfg = compose(config_name="workflow/native", overrides=[f"model={model}"])
        get_class(cfg.model._target_)
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
