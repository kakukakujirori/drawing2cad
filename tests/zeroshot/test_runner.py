import json
import shlex
import sys
from collections.abc import Callable, Sequence
from inspect import cleandoc
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import PrivateAttr
from rich.console import Console

from zeroshot.pipeline.event_logging import ConsoleReporter
from zeroshot.pipeline.manifest import InputManifest
from zeroshot.pipeline.messages import MessageBuilder
from zeroshot.pipeline.runner import PipelineRunner
from zeroshot.pipeline.sandbox import SandboxRunner
from zeroshot.pipeline.verification import CadQueryExecutor, StepRenderer

VALID_BOX_SOURCE = """\
import cadquery as cq

result = cq.Workplane("XY").box(10, 20, 30)
"""


class _ScriptedChatModel(BaseChatModel):
    responses: tuple[AIMessage, ...]

    _response_index: int = PrivateAttr(default=0)
    _received_messages: list[list[BaseMessage]] = PrivateAttr(default_factory=list)
    _bound_tool_names: tuple[str, ...] = PrivateAttr(default=())

    @property
    def _llm_type(self) -> str:
        return "scripted-test-model"

    @property
    def received_messages(self) -> list[list[BaseMessage]]:
        return self._received_messages

    @property
    def bound_tool_names(self) -> tuple[str, ...]:
        return self._bound_tool_names

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        del tool_choice, kwargs

        if not all(isinstance(tool, BaseTool) for tool in tools):
            raise TypeError("This scripted model only accepts BaseTool instances")

        self._bound_tool_names = tuple(
            tool.name for tool in tools if isinstance(tool, BaseTool)
        )
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs

        self._received_messages.append(list(messages))
        try:
            response = self.responses[self._response_index]
        except IndexError as error:
            raise AssertionError("Scripted model ran out of responses") from error

        self._response_index += 1
        return ChatResult(
            generations=[ChatGeneration(message=response)],
        )


def _message_text(message: BaseMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return "\n".join(
        block["text"]
        for block in message.content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _write_text_command(filename: str, content: str) -> str:
    script = (
        "from pathlib import Path; "
        f"Path({filename!r}).write_text({content!r}, encoding='utf-8')"
    )
    return f"python -c {shlex.quote(script)}"


def _manifest_without_renders(tmp_path: Path, sample_id: str) -> InputManifest:
    dxf_path = tmp_path / f"{sample_id}.dxf"
    dxf_path.write_text("DXF_FIXTURE", encoding="utf-8")
    return InputManifest(
        sample_id=sample_id,
        dxf_path=dxf_path,
        render3d_paths={},
    )


def _message_builder_without_renders() -> MessageBuilder:
    return MessageBuilder(
        access_render3d="none",
        access_render3d_styles=(),
        feedback_render3d="none",
        feedback_render3d_styles=(),
    )


def _sandbox_runner() -> SandboxRunner:
    return SandboxRunner(
        python_executable=Path(sys.executable),
        default_timeout_s=30,
    )


def _renderer() -> StepRenderer:
    return StepRenderer(timeout_s=120.0)


def test_run_sample_stages_only_allowed_inputs_and_preserves_workdir(
    tmp_path: Path,
) -> None:
    dxf_path = tmp_path / "source.dxf"
    selected_render_path = tmp_path / "selected.png"
    hidden_render_path = tmp_path / "hidden.png"
    dxf_path.write_text("ORIGINAL_DXF", encoding="utf-8")
    selected_render_path.write_bytes(b"ALLOWED_RENDER")
    hidden_render_path.write_bytes(b"HIDDEN_RENDER")

    manifest = InputManifest(
        sample_id="sample-1",
        dxf_path=dxf_path,
        render3d_paths={
            "style-a": selected_render_path,
            "style-b": hidden_render_path,
        },
    )
    inspect_inputs = cleandoc(
        """
        from pathlib import Path

        dxf = Path('/work/inputs/techdraw.dxf')
        assert dxf.read_text() == 'ORIGINAL_DXF'
        assert Path('/work/inputs/style-a.png').read_bytes() == b'ALLOWED_RENDER'
        assert not Path('/work/inputs/style-b.png').exists()
        try:
            dxf.write_text('SANDBOX_MUTATION')
        except OSError:
            pass
        else:
            raise AssertionError('sandbox input must be read-only')
        Path('/work/scratch.txt').write_text('persisted')
        Path('/work/events.jsonl').write_text('FORGED')
        print('staged-ok')
        """
    )
    model = _ScriptedChatModel(
        responses=(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "run_shell",
                        "args": {"command": f"python -c {shlex.quote(inspect_inputs)}"},
                        "id": "call-inspect-inputs",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "run_shell",
                        "args": {"command": "cat /work/scratch.txt"},
                        "id": "call-read-scratch",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="done"),
        )
    )
    message_builder = MessageBuilder(
        access_render3d="path",
        access_render3d_styles=("style-a",),
        feedback_render3d="none",
        feedback_render3d_styles=(),
    )
    runner = PipelineRunner(
        model=model,
        message_builder=message_builder,
        sandbox_runner=SandboxRunner(
            python_executable=Path(sys.executable),
            default_timeout_s=10,
        ),
        artifact_root=tmp_path / "artifacts",
        renderer=_renderer(),
    )

    result = runner.run_sample(manifest)

    assert model.bound_tool_names == (
        "run_shell",
        "load_image",
        "verify_output",
    )
    assert len(model.received_messages) == 3

    initial_messages = model.received_messages[0]
    assert len(initial_messages) == 2
    initial_human_message = initial_messages[1]
    assert isinstance(initial_human_message, HumanMessage)
    initial_text = _message_text(initial_human_message)
    assert "/work/inputs/techdraw.dxf" in initial_text
    assert "/work/inputs/style-a.png" in initial_text
    assert "style-b" not in initial_text
    assert str(dxf_path) not in initial_text
    assert str(selected_render_path) not in initial_text
    assert str(hidden_render_path) not in initial_text

    messages = result["messages"]
    assert [type(message) for message in messages] == [
        *[type(message) for message in initial_messages],
        AIMessage,
        ToolMessage,
        AIMessage,
        ToolMessage,
        AIMessage,
    ]

    inspect_result = messages[3]
    assert isinstance(inspect_result, ToolMessage)
    assert inspect_result.tool_call_id == "call-inspect-inputs"
    assert isinstance(inspect_result.content, str)
    assert json.loads(inspect_result.content) == {
        "status": "COMPLETED",
        "returncode": 0,
        "stdout": "staged-ok\n",
        "stderr": "",
    }

    persistence_result = messages[5]
    assert isinstance(persistence_result, ToolMessage)
    assert persistence_result.tool_call_id == "call-read-scratch"
    assert isinstance(persistence_result.content, str)
    assert json.loads(persistence_result.content) == {
        "status": "COMPLETED",
        "returncode": 0,
        "stdout": "persisted",
        "stderr": "",
    }

    assert isinstance(messages[-1], AIMessage)
    assert messages[-1].content == "done"
    assert messages[-1].tool_calls == []

    assert dxf_path.read_text(encoding="utf-8") == "ORIGINAL_DXF"
    assert selected_render_path.read_bytes() == b"ALLOWED_RENDER"
    assert hidden_render_path.read_bytes() == b"HIDDEN_RENDER"

    sample_artifact_root = tmp_path / "artifacts" / "sample-1"
    saved_workdir = sample_artifact_root / "workspace"
    assert (saved_workdir / "inputs" / "techdraw.dxf").read_text(
        encoding="utf-8"
    ) == "ORIGINAL_DXF"
    assert (saved_workdir / "inputs" / "style-a.png").read_bytes() == b"ALLOWED_RENDER"
    assert not (saved_workdir / "inputs" / "style-b.png").exists()
    assert (saved_workdir / "scratch.txt").read_text(encoding="utf-8") == "persisted"
    assert (saved_workdir / "attempts").is_dir()

    assert (saved_workdir / "events.jsonl").read_text(encoding="utf-8") == "FORGED"

    event_log_path = sample_artifact_root / "events.jsonl"
    events = [
        json.loads(line)
        for line in event_log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event_index"] for event in events] == list(range(len(events)))
    assert events[0]["event"] == "run_started"
    assert events[-1]["event"] == "run_completed"
    assert {event["sample_id"] for event in events} == {"sample-1"}
    assert len({event["run_id"] for event in events}) == 1
    assert sum(event["event"] == "input" for event in events) == 1

    tool_events = [
        event
        for event in events
        if event["event"] in {"tool_started", "tool_finished"}
        and event["data"].get("tool_call_id")
        in {"call-inspect-inputs", "call-read-scratch"}
    ]
    assert [event["event"] for event in tool_events] == [
        "tool_started",
        "tool_finished",
        "tool_started",
        "tool_finished",
    ]

    final_verify_started = next(
        event
        for event in events
        if event["event"] == "tool_started"
        and event["data"]["tool_name"] == "verify_output"
    )
    assert final_verify_started["data"]["caller"] == "workflow"
    assert "output" not in {event["event"] for event in events}

    checkpoint_path = sample_artifact_root / "checkpoints.sqlite"
    assert checkpoint_path.is_file()
    with SqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
        checkpoints = list(checkpointer.list(None))

    assert checkpoints
    thread_ids = {
        checkpoint.config["configurable"]["thread_id"] for checkpoint in checkpoints
    }
    assert len(thread_ids) == 1
    assert thread_ids == {events[0]["run_id"]}


def test_run_sample_preserves_workdir_when_graph_fails(tmp_path: Path) -> None:
    manifest = _manifest_without_renders(tmp_path, "failed-run")
    model = _ScriptedChatModel(
        responses=(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "run_shell",
                        "args": {
                            "command": _write_text_command("failure.txt", "preserved")
                        },
                        "id": "call-write-before-failure",
                        "type": "tool_call",
                    }
                ],
            ),
        )
    )
    artifact_root = tmp_path / "artifacts"
    runner = PipelineRunner(
        model=model,
        message_builder=_message_builder_without_renders(),
        sandbox_runner=_sandbox_runner(),
        artifact_root=artifact_root,
        renderer=_renderer(),
    )

    with pytest.raises(AssertionError, match="ran out of responses"):
        runner.run_sample(manifest)

    sample_artifact_root = artifact_root / manifest.sample_id
    assert (sample_artifact_root / "workspace" / "failure.txt").read_text(
        encoding="utf-8"
    ) == "preserved"
    events = [
        json.loads(line)
        for line in (sample_artifact_root / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert events[-1]["event"] == "run_failed"


def test_run_sample_verifies_and_preserves_valid_cadquery_output(
    tmp_path: Path,
) -> None:
    manifest = _manifest_without_renders(tmp_path, "valid-box")
    model = _ScriptedChatModel(
        responses=(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "run_shell",
                        "args": {
                            "command": _write_text_command(
                                "model.py",
                                VALID_BOX_SOURCE,
                            )
                        },
                        "id": "call-write-model",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="done"),
        )
    )
    artifact_root = tmp_path / "artifacts"
    console_output = StringIO()
    runner = PipelineRunner(
        model=model,
        message_builder=_message_builder_without_renders(),
        sandbox_runner=_sandbox_runner(),
        artifact_root=artifact_root,
        renderer=_renderer(),
        console_reporter=ConsoleReporter(
            Console(
                file=console_output,
                color_system=None,
                force_terminal=False,
                highlight=False,
            )
        ),
    )

    result = runner.run_sample(manifest)

    report = result["last_verification"]
    assert report.status == "VERIFIED"
    assert report.verification_id == "000"
    rendered_console = console_output.getvalue()
    assert "[node] agent started — waiting for model" in rendered_console
    assert "tool call: run_shell" in rendered_console
    assert "done" in rendered_console
    assert "[verification]" in rendered_console
    assert "run completed" in rendered_console

    final_attempt = artifact_root / "valid-box" / "workspace" / "attempts" / "000"
    assert (final_attempt / "model.py").read_text(encoding="utf-8") == VALID_BOX_SOURCE
    CadQueryExecutor.verify_step(final_attempt / "output.step")

    events = [
        json.loads(line)
        for line in (artifact_root / "valid-box" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    verification = next(event for event in events if event["event"] == "verification")
    assert verification["data"]["report"]["status"] == "VERIFIED"
    assert verification["data"]["report"]["verification_id"] == "000"
    assert verification["data"]["report"]["source"]["omitted"] == "source"


def test_run_sample_repairs_model_after_intermediate_verification_failure(
    tmp_path: Path,
) -> None:
    manifest = _manifest_without_renders(tmp_path, "repair-box")
    model = _ScriptedChatModel(
        responses=(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "run_shell",
                        "args": {
                            "command": _write_text_command("model.py", "result = (")
                        },
                        "id": "call-write-invalid-model",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "verify_output",
                        "args": {},
                        "id": "call-verify-invalid-model",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "run_shell",
                        "args": {
                            "command": _write_text_command(
                                "model.py",
                                VALID_BOX_SOURCE,
                            )
                        },
                        "id": "call-repair-model",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="done"),
        )
    )
    artifact_root = tmp_path / "artifacts"
    runner = PipelineRunner(
        model=model,
        message_builder=_message_builder_without_renders(),
        sandbox_runner=_sandbox_runner(),
        artifact_root=artifact_root,
        renderer=_renderer(),
    )

    result = runner.run_sample(manifest)

    intermediate_result = result["messages"][5]
    assert isinstance(intermediate_result, ToolMessage)
    assert intermediate_result.tool_call_id == "call-verify-invalid-model"
    # verify_output answers in content blocks, so that a verified attempt can
    # hand its rendered views back through the same tool message.
    assert isinstance(intermediate_result.content, list)
    (block,) = intermediate_result.content
    intermediate_report = json.loads(block["text"])
    assert intermediate_report["status"] == "REJECTED"
    assert intermediate_report["verification_id"] == "000"

    final_report = result["last_verification"]
    assert final_report.status == "VERIFIED"
    assert final_report.verification_id == "001"

    attempts = artifact_root / "repair-box" / "workspace" / "attempts"
    assert (attempts / "000" / "model.py").read_text(encoding="utf-8") == "result = ("
    assert not (attempts / "000" / "output.step").exists()
    assert (attempts / "001" / "model.py").read_text(
        encoding="utf-8"
    ) == VALID_BOX_SOURCE
    CadQueryExecutor.verify_step(attempts / "001" / "output.step")
