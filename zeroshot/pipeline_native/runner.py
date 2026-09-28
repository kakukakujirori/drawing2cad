"""A single LangGraph agent, with no staged reconstruction machinery."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from hydra.utils import instantiate, to_absolute_path
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallRequest, ToolErrorMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from omegaconf import DictConfig, OmegaConf

from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.tools.errors import ToolFeedbackError
from zeroshot.pipeline.tools.load_image import create_load_image_tool
from zeroshot.pipeline.tools.run_shell import create_run_shell_tool
from zeroshot.pipeline.workflow.middleware.stateless_reasoning import (
    StatelessReasoningMiddleware,
)
from zeroshot.pipeline_native.event_logging import EventLog, write_json

_PROMPT = Path(__file__).parent / "prompts" / "generator.md"
_RETROSPECTIVE_PROMPT = _PROMPT.with_name("retrospective.md")


def _basename(value: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"Expected a filename component: {value!r}")
    return value


def _tool_error(error: Exception, request: ToolCallRequest) -> str:
    del request
    if isinstance(error, ToolFeedbackError):
        return str(error)
    raise error


def _git(*args: str) -> str | None:
    try:
        process = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return process.stdout.strip() if process.returncode == 0 else None


def run(config: DictConfig, *, model: BaseChatModel | None = None) -> dict[str, Any]:
    """Run once; retain failures and refuse to overwrite an earlier experiment."""
    if config.workflow.name != "native":
        raise ValueError("This entrypoint requires workflow=native")
    sample_id = _basename(str(config.sample.sample_id))
    sheets = list(config.sample.drawing.sheets)
    if not sheets:
        raise ValueError("At least one input PNG is required")
    names = [_basename(str(sheet.name)) for sheet in sheets]
    if len(set(names)) != len(names):
        raise ValueError("Input drawing names must be unique")
    sources = [Path(to_absolute_path(str(sheet.file))) for sheet in sheets]
    for source in sources:
        if source.suffix.lower() != ".png" or not source.is_file():
            raise ValueError(f"Input must be an existing PNG: {source}")

    run_dir = Path(to_absolute_path(str(config.artifact_root))) / sample_id
    workspace = run_dir / "workspace"
    if workspace.exists() or (run_dir / "events.jsonl").exists():
        raise FileExistsError(
            f"Run already exists; choose a new artifact_root: {run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    workspace.mkdir()  # Also reserves the destination against concurrent runs.

    with EventLog(run_dir / "events.jsonl", console=bool(config.console)) as log:
        run_id = str(uuid4())
        log.write("run_started", {"run_id": run_id, "sample_id": sample_id})
        try:
            status = _git("status", "--porcelain")
            metadata = {
                "run_id": run_id,
                "sample_id": sample_id,
                "config": OmegaConf.to_container(config, resolve=True),
                "git_commit": _git("rev-parse", "HEAD"),
                "git_dirty": None if status is None else bool(status),
                "versions": {
                    name: version(name)
                    for name in (
                        "langchain",
                        "langgraph",
                        "langchain-openai",
                        "langchain-openrouter",
                        "cadquery",
                    )
                },
                "inputs": [],
            }
            write_json(run_dir / "run.json", metadata)
            sandbox_settings = OmegaConf.to_container(
                config.sandbox_runner, resolve=True
            )
            sandbox_settings["python_executable"] = Path(
                to_absolute_path(
                    sandbox_settings["python_executable"] or sys.executable
                )
            )
            sandbox = SandboxRunner(**sandbox_settings)
            with SandboxWorkdir(
                workspace, read_only_subdirs=[PurePosixPath("inputs")]
            ) as workdir:
                inputs = workspace / "inputs"
                inputs.mkdir()
                load_image = create_load_image_tool(workdir)
                blocks: list[dict[str, Any]] = []
                for name, source in zip(names, sources, strict=True):
                    staged = inputs / f"{name}.png"
                    shutil.copyfile(source, staged)
                    sandbox_path = str(workdir.host_to_sandbox_path(staged))
                    images = load_image.invoke({"image_path": sandbox_path})
                    if not images[0]["image_url"]["url"].startswith("data:image/png;"):
                        raise ValueError(f"Input is not a PNG image: {source}")
                    blocks.extend(
                        [
                            {"type": "text", "text": f"Input drawing: {sandbox_path}"},
                            *images,
                        ]
                    )
                    metadata["inputs"].append(
                        {
                            "source": str(source),
                            "file": sandbox_path,
                            "sha256": hashlib.sha256(staged.read_bytes()).hexdigest(),
                        }
                    )

                prompt = _PROMPT.read_text(encoding="utf-8").strip()
                messages = [HumanMessage(content=blocks)]
                write_json(run_dir / "run.json", metadata)
                log.write("prompt", {"system": prompt, "messages": messages})
                if model is None:
                    model = instantiate(config.model)
                agent = create_agent(
                    model=model,
                    tools=[create_run_shell_tool(sandbox, workdir), load_image],
                    system_prompt=prompt,
                    middleware=[
                        ToolErrorMiddleware(on_error=_tool_error),
                        StatelessReasoningMiddleware(),
                    ],
                    name="generator",
                    checkpointer=False,
                ).with_config(recursion_limit=sys.maxsize)
                with agent.stream_events(
                    {"messages": messages}, version="v3"
                ) as stream:
                    for event in stream:
                        log.record_protocol(event)
                    result = stream.output
                if result is None:
                    raise RuntimeError("Agent ended without a final state")
                write_json(run_dir / "messages.json", result["messages"])
                output = workspace / "model.py"
                if (
                    output.is_symlink()
                    or not output.is_file()
                    or output.stat().st_size == 0
                ):
                    raise FileNotFoundError(
                        "Agent did not write a nonempty /work/model.py"
                    )
                log.write(
                    "run_completed",
                    {
                        "model_path": "workspace/model.py",
                        "model_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                    },
                )
        except BaseException as error:
            log.write(
                "run_failed",
                {
                    "error_type": type(error).__name__,
                    "error": str(error)[:4000],
                },
            )
            raise

    _write_retrospective(
        model, result["messages"], prompt, run_dir, bool(config.console)
    )
    return result


def _write_retrospective(
    model: BaseChatModel,
    messages: list[BaseMessage],
    system_prompt: str,
    run_dir: Path,
    console: bool,
) -> None:
    """Ask only after generation is saved; tools cannot modify its artifacts."""
    with EventLog(run_dir / "retrospective_events.jsonl", console=console) as log:
        log.write("retrospective_started", {})
        try:
            source = (run_dir / "workspace/model.py").read_text(encoding="utf-8")
            request = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": f"Final saved /work/model.py:\n```python\n{source}\n```",
                    },
                    {
                        "type": "text",
                        "text": _RETROSPECTIVE_PROMPT.read_text(
                            encoding="utf-8"
                        ).strip(),
                    },
                ]
            )
            log.write("prompt", {"system": system_prompt, "messages": [request]})
            agent = create_agent(
                model=model,
                tools=[],
                system_prompt=system_prompt,
                middleware=[StatelessReasoningMiddleware()],
                name="retrospective",
                checkpointer=False,
            )
            with agent.stream_events(
                {"messages": [*messages, request]}, version="v3"
            ) as stream:
                for event in stream:
                    log.record_protocol(event)
                output = stream.output
            if output is None:
                raise RuntimeError("Retrospective ended without a final state")
            response = output["messages"][-1]
            write_json(run_dir / "retrospective_messages.json", [request, response])
            if (
                not isinstance(response, AIMessage)
                or response.tool_calls
                or not response.text.strip()
            ):
                raise ValueError("Retrospective did not produce a text explanation")
            (run_dir / "reasoning_traj.md").write_text(
                response.text.strip() + "\n", encoding="utf-8"
            )
            log.write("retrospective_completed", {"path": "reasoning_traj.md"})
        except BaseException as error:
            log.write(
                "retrospective_failed",
                {"error_type": type(error).__name__, "error": str(error)[:4000]},
            )
            raise
