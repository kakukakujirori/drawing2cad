"""The native baseline plus an auditor: same two tools, no staged machinery
beyond the automatic builds Native 2' adds."""

from __future__ import annotations

import hashlib
import re
import sys
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from typing import Any, cast
from uuid import uuid4

from hydra.utils import instantiate, to_absolute_path
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ToolErrorMiddleware
from langchain.agents.middleware.types import InputAgentState
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.messages.content import create_image_block, create_text_block
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from zeroshot.pipeline.messages.artifact import ArtifactPresenter
from zeroshot.pipeline.models.image_history import INPUT_IMAGE_ID
from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.stages._base.prompt import PromptTemplate, describe_placement
from zeroshot.pipeline.stages.coding.verify import OutputVerifier
from zeroshot.pipeline.stages.interpretation.contracts import ProjectionMethod
from zeroshot.pipeline.tools.load_image import create_load_image_tool
from zeroshot.pipeline.tools.run_shell import create_run_shell_tool
from zeroshot.pipeline.verification import (
    AttemptStore,
    CadQueryExecutor,
    StepRenderer,
)
from zeroshot.pipeline.workflow.middleware import VerifyOnWriteMiddleware
from zeroshot.pipeline.workflow.middleware.stateless_reasoning import (
    StatelessReasoningMiddleware,
)
from zeroshot.pipeline_native.connection_retry import ConnectionRetryMiddleware
from zeroshot.pipeline_native.event_logging import (
    EventLog,
    has_run_completed,
    write_json,
)
from zeroshot.pipeline_native.runner import (
    _LENGTH_RETRY_PROMPT,
    _ON_EXISTING,
    _basename,
    _git,
    _prepare_workspace,
    _tool_error,
    _write_retrospective,
)
from zeroshot.pipeline_native.runner import _PROMPT as _CODER_PROMPT
from zeroshot.pipeline_native2.turn_budget import (
    StopReason,
    TurnBudgetMiddleware,
    TurnBudgetState,
)

_PROMPTS = Path(__file__).parent / "prompts"
# Tolerates Markdown around the verdict line, such as `**VERDICT:** ACCEPT`.
_VERDICT = re.compile(r"VERDICT\W*(ACCEPT|REJECT)", re.IGNORECASE)
_AUDIT_LENGTH_RETRY_PROMPT = (
    "Your response reached the output limit; give your report now, ending with "
    "the verdict line."
)
_BUILD_REPORT = "The pipeline's verification report describes this build."


def accepted(report: str) -> bool:
    """Whether the report's last verdict is ACCEPT."""
    verdicts = _VERDICT.findall(report)
    return bool(verdicts) and verdicts[-1].upper() == "ACCEPT"


def _has_model(path: Path) -> bool:
    return path.is_file() and not path.is_symlink() and path.stat().st_size > 0


def run(
    config: DictConfig, *, model: BaseChatModel | None = None
) -> dict[str, Any] | None:
    """Run one sample, returning None when a completed generation is skipped."""
    on_existing = config.get("on_existing", "fail")
    if on_existing not in _ON_EXISTING:
        raise ValueError(f"on_existing must be one of {_ON_EXISTING}: {on_existing!r}")
    sample_id = _basename(str(config.sample.sample_id))
    run_dir = Path(to_absolute_path(str(config.artifact_root))) / sample_id
    events_path = run_dir / "events.jsonl"
    if has_run_completed(events_path):
        if on_existing in {"skip", "retry"}:
            return None
        raise FileExistsError(f"Sample already ran: {run_dir}")
    if on_existing != "retry" and (
        (run_dir / "workspace").exists() or events_path.exists()
    ):
        raise FileExistsError(
            f"Incomplete run left behind; use on_existing=retry to redo: {run_dir}"
        )

    sheets = list(config.sample.drawing.sheets)
    if not sheets:
        raise ValueError("At least one input PNG is required")
    names = [_basename(str(sheet.name)) for sheet in sheets]
    if len(set(names)) != len(names):
        raise ValueError("Input drawing names must be unique")
    sources = [Path(to_absolute_path(str(sheet.file))) for sheet in sheets]
    projection_method = ProjectionMethod(str(config.sample.drawing.projection_method))

    max_side = config.sample.max_input_image_side
    if max_side < 1:
        raise ValueError("sample.max_input_image_side must be positive")
    for source in sources:
        if source.suffix.lower() != ".png" or not source.is_file():
            raise ValueError(f"Input must be an existing PNG: {source}")
        with Image.open(source) as image:
            width, height = image.size
        if max(width, height) > max_side:
            raise ValueError(
                f"Input image {source} is {width}x{height}; "
                f"sample.max_input_image_side={max_side} permits neither side to exceed "
                f"{max_side} pixels"
            )
    max_audits = int(config.max_audits)
    if max_audits < 0:
        raise ValueError("max_audits must be non-negative")
    retry_turns = int(config.retry_turns)
    max_stage_validation_retries = int(config.max_stage_validation_retries)
    if max_stage_validation_retries < 0:
        raise ValueError("max_stage_validation_retries must be non-negative")
    coder_budget = TurnBudgetMiddleware(int(config.max_turns), retry_turns)
    auditor_budget = TurnBudgetMiddleware(int(config.auditor_max_turns), retry_turns)

    workspace = _prepare_workspace(run_dir, names, sources, on_existing)

    with EventLog(events_path, console=bool(config.console)) as log:
        run_id = str(uuid4())
        log.write("run_started", {"run_id": run_id, "sample_id": sample_id})
        try:
            status = _git("status", "--porcelain")
            metadata = {
                # How `run_scoring.is_native_run` tells this layout apart.
                "pipeline": "native2",
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
            sandbox_settings = cast(
                dict[str, Any],
                OmegaConf.to_container(config.sandbox_runner, resolve=True),
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
                load_image = create_load_image_tool(workdir)
                blocks: list[str | dict[str, Any]] = []
                for name, source in zip(names, sources, strict=True):
                    staged = inputs / f"{name}.png"
                    sandbox_path = str(workdir.host_to_sandbox_path(staged))
                    images = load_image.invoke({"image_path": sandbox_path})
                    if not images[0]["image_url"]["url"].startswith("data:image/png;"):
                        raise ValueError(f"Input is not a PNG image: {source}")
                    images[0] = create_image_block(
                        url=images[0]["image_url"]["url"], id=INPUT_IMAGE_ID
                    )
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
                write_json(run_dir / "run.json", metadata)

                # Native 2': build and render the program as the staged pipeline
                # does, in the standard views, with no interpretation to compare.
                builds = (
                    OutputVerifier(
                        executor=CadQueryExecutor(sandbox_runner=sandbox),
                        workdir=workdir,
                        renderer=StepRenderer(
                            ignore_hidden_lines=bool(
                                OmegaConf.select(
                                    config,
                                    "sample.drawing.hidden_lines_omitted",
                                    default=False,
                                )
                            )
                        ),
                        diff_drawer=None,
                        artifact_presenter=ArtifactPresenter(
                            input="path", output_renders="path"
                        ),
                        attempt_store=AttemptStore(workdir, lambda: round_),
                        projection_view_mode="standard",
                    )
                    if config.auto_build
                    else None
                )
                placement = describe_placement(projection_method)
                prompt = PromptTemplate(_CODER_PROMPT).render(
                    projection_placement=placement
                )
                auditor_prompt = PromptTemplate(_PROMPTS / "auditor.md").render(
                    projection_placement=placement
                )
                if builds is not None:
                    frame = PromptTemplate(_PROMPTS / "model_frame.md").render()
                    automatic = PromptTemplate(_PROMPTS / "automatic_build.md").render()
                    prompt = f"{prompt}\n\n{automatic}\n\n{frame}"
                    auditor_prompt = f"{auditor_prompt}\n\n{frame}"
                if model is None:
                    model = instantiate(config.model)
                model_retries = int(config.model_retries)
                tools = [create_run_shell_tool(sandbox, workdir), load_image]
                output = workspace / "model.py"

                def converse(
                    role: str,
                    system: str,
                    history: list[AnyMessage],
                    request: HumanMessage,
                    budget: TurnBudgetMiddleware,
                    length_retry_prompt: str,
                ) -> tuple[list[AnyMessage], str | None]:
                    """Continue `history` with `request`; return a failure status or None."""
                    log.write(
                        "prompt",
                        {"agent": role, "system": system, "messages": [request]},
                    )
                    state: TurnBudgetState = {"messages": [*history, request]}
                    middleware: list[AgentMiddleware[Any, None]] = [
                        ToolErrorMiddleware(on_error=_tool_error),
                        budget,
                        ConnectionRetryMiddleware(
                            max_retries=model_retries,
                            role=role,
                            length_retry_prompt=length_retry_prompt,
                        ),
                        StatelessReasoningMiddleware(),
                        # After the turn budget, as in the staged pipeline.
                        *(
                            [VerifyOnWriteMiddleware(builds)]
                            if role == "coder" and builds is not None
                            else []
                        ),
                    ]
                    agent = create_agent(
                        model=model,
                        tools=tools,
                        system_prompt=system,
                        middleware=middleware,
                        name=role,
                        checkpointer=False,
                    ).with_config(recursion_limit=sys.maxsize)
                    for attempt in range(max_stage_validation_retries + 1):
                        # LangChain's input/output types omit middleware state fields.
                        with agent.stream_events(
                            cast(InputAgentState, state), version="v3"
                        ) as stream:
                            for event in stream:
                                log.record_protocol(event)
                            result = stream.output
                        if result is None:
                            raise RuntimeError(f"{role} ended without a final state")
                        state = cast(TurnBudgetState, result)
                        if state.get("stop_reason") is StopReason.BUDGET_EXHAUSTED:
                            error = "The turn budget ran out before your final answer."
                        elif role == "coder" and not _has_model(output):
                            error = "Write a nonempty /work/model.py, then give your final answer."
                        elif role == "auditor" and not _VERDICT.search(
                            state["messages"][-1].text
                        ):
                            error = "Finish your report with VERDICT: ACCEPT or VERDICT: REJECT."
                        else:
                            return state["messages"], None

                        if attempt == max_stage_validation_retries:
                            break
                        log.write(
                            "stage_validation_retry",
                            {
                                "agent": role,
                                "round": round_,
                                "retry": attempt + 1,
                                "total_turns": state["total_turns"],
                                "error": error,
                            },
                        )
                        state = {
                            **state,
                            "retry": True,
                            "messages": [
                                *state["messages"],
                                HumanMessage(content=error),
                            ],
                        }
                    exhausted = state.get("stop_reason") is StopReason.BUDGET_EXHAUSTED
                    failure_status = (
                        "turn_limit_exceeded"
                        if exhausted
                        else "stage_validation_failed"
                    )
                    log.write(
                        failure_status,
                        {
                            "agent": role,
                            "max_turns": budget.max_turns,
                            "turn_count": state["total_turns"],
                            "error": error,
                        },
                    )
                    return state["messages"], failure_status

                coder_messages: list[AnyMessage] = []
                audit_messages: list[AnyMessage] = []
                coder_request = HumanMessage(content=blocks)
                for round_ in range(max_audits + 1):
                    coder_messages, failure_status = converse(
                        "coder",
                        prompt,
                        coder_messages,
                        coder_request,
                        coder_budget,
                        _LENGTH_RETRY_PROMPT,
                    )
                    if failure_status:
                        break
                    if round_ == max_audits:
                        break

                    program = output.read_bytes()
                    request = PromptTemplate(_PROMPTS / "audit_request.md").render(
                        author_message=coder_messages[-1].text
                    )
                    # The coder's last build of this program, built again only if it
                    # failed transiently (a timeout or an infrastructure error).
                    build_report = (
                        [create_text_block(_BUILD_REPORT), *builds.feedback()]
                        if builds is not None
                        else []
                    )
                    try:
                        audit_messages, failure_status = converse(
                            "auditor",
                            auditor_prompt,
                            audit_messages,
                            HumanMessage(
                                content=[
                                    *([] if audit_messages else blocks),
                                    {"type": "text", "text": request},
                                    *build_report,
                                ]
                            ),
                            auditor_budget,
                            _AUDIT_LENGTH_RETRY_PROMPT,
                        )
                    finally:
                        # Even a failed audit leaves the coder's program to be scored.
                        if not _has_model(output) or output.read_bytes() != program:
                            output.unlink(missing_ok=True)
                            output.write_bytes(program)
                            log.write("model_restored", {"round": round_})
                    if failure_status:
                        break
                    report = next(
                        m.text
                        for m in reversed(audit_messages)
                        if isinstance(m, AIMessage)
                    )
                    verdict = accepted(report)
                    log.write("audit_verdict", {"round": round_, "accepted": verdict})
                    if verdict:
                        break
                    coder_request = HumanMessage(
                        content=PromptTemplate(_PROMPTS / "revision.md").render(
                            report=report
                        )
                    )

                write_json(run_dir / "messages.json", coder_messages)
                if audit_messages:
                    write_json(run_dir / "audit_messages.json", audit_messages)
                if failure_status and not _has_model(output):
                    output.unlink(missing_ok=True)
                    output.write_text(
                        f"# Failed: {failure_status} before /work/model.py was written.\n"
                        "import cadquery as cq\n"
                        "result = None\n",
                        encoding="utf-8",
                    )
                log.write(
                    "run_completed",
                    {
                        "model_path": "workspace/model.py",
                        "model_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                        "coder_rounds": round_ + 1,
                        **({"status": failure_status} if failure_status else {}),
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

    if not failure_status:
        _write_retrospective(
            model,
            coder_messages,
            prompt,
            run_dir,
            bool(config.console),
            model_retries=model_retries,
        )
    return {
        "messages": coder_messages,
        **({"status": failure_status} if failure_status else {}),
    }
