import json
import shlex
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import PrivateAttr

from zeroshot.pipeline.manifest import InputManifest
from zeroshot.pipeline.messages import MessageBuilder
from zeroshot.pipeline.runner import PipelineRunner
from zeroshot.pipeline.sandbox import SandboxRunner


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
    inspect_inputs = "\n".join(
        [
            "from pathlib import Path",
            "dxf = Path('/work/inputs/techdraw.dxf')",
            "assert dxf.read_text() == 'ORIGINAL_DXF'",
            "assert Path('/work/inputs/style-a.png').read_bytes() == b'ALLOWED_RENDER'",
            "assert not Path('/work/inputs/style-b.png').exists()",
            "dxf.write_text('SANDBOX_MUTATION')",
            "Path('/work/scratch.txt').write_text('persisted')",
            "print('staged-ok')",
        ]
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

    saved_workdir = tmp_path / "artifacts" / "sample-1"
    assert (saved_workdir / "inputs" / "techdraw.dxf").read_text(
        encoding="utf-8"
    ) == "SANDBOX_MUTATION"
    assert (saved_workdir / "inputs" / "style-a.png").read_bytes() == b"ALLOWED_RENDER"
    assert not (saved_workdir / "inputs" / "style-b.png").exists()
    assert (saved_workdir / "scratch.txt").read_text(encoding="utf-8") == "persisted"
    assert (saved_workdir / "attempts").is_dir()

    checkpoint_path = tmp_path / "artifacts" / ".langgraph" / "sample-1.sqlite"
    assert checkpoint_path.is_file()
    with SqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
        checkpoints = list(checkpointer.list(None))

    assert checkpoints
    thread_ids = {
        checkpoint.config["configurable"]["thread_id"] for checkpoint in checkpoints
    }
    assert len(thread_ids) == 1
    assert next(iter(thread_ids)).startswith("sample-1:")
