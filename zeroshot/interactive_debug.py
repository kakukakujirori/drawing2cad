"""Continue a submitted agent's conversation in a separate workspace.

    python -m zeroshot.interactive_debug --run RUN/checkpoints.sqlite \
        --stage coder --round 0

Enter one question per line; /reset starts a fresh branch from the same source,
and /quit or EOF exits. --prepare-only checks restoration without calling an API.
The original database and workspace are read only. No pipeline node is resumed.
"""

from __future__ import annotations

import argparse
import copy
import json
import readline  # noqa: F401 -- Enables cursor movement and editing in input().
import shutil
import sqlite3
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from itertools import count
from pathlib import Path, PurePosixPath
from string import Template
from typing import Any, cast

from hydra.utils import instantiate
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ToolErrorMiddleware
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from omegaconf import DictConfig, OmegaConf

from zeroshot.pipeline.event_logging import (
    AgentMessageTransformer,
    ConsoleReporter,
    JsonlEventWriter,
    RunEventTransformer,
)
from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.stages.coding.verify import OutputVerifier
from zeroshot.pipeline.stages.contracts import ReconstructionHistory
from zeroshot.pipeline.tools.calculate_drawing_scale import (
    create_calculate_drawing_scale_tool,
)
from zeroshot.pipeline.tools.load_image import create_load_image_tool
from zeroshot.pipeline.tools.render_step import create_render_step_tool
from zeroshot.pipeline.tools.run_shell import create_run_shell_tool
from zeroshot.pipeline.verification import (
    AttemptStore,
    CadQueryExecutor,
    DrawingDiffExecutor,
    StepRenderer,
)
from zeroshot.pipeline.workflow import CUSTOM_STATE_TYPES
from zeroshot.pipeline.workflow.components.agent import AgentState, _handle_tool_error
from zeroshot.pipeline.workflow.middleware import (
    ModelCallRetryMiddleware,
    PromptLogMiddleware,
    StatelessReasoningMiddleware,
    VerifyOnWriteMiddleware,
)

STAGES = {
    "interpreter": "interpretation",
    "interpretation": "interpretation",
    "planner": "operations",
    "operations": "operations",
    "coder": "coding",
    "coding": "coding",
    "auditor": "audit",
    "audit": "audit",
}
ASSET_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".svg", ".dxf", ".step", ".stp"}
INTRODUCTION = (
    "提出結果に対してユーザーからの質問が届きました。以下対話形式で回答してください。\n"
    "過去の提出内容は /work/reconstruction.json に保存されています。"
    "個別の提出用JSONやmodel.pyは置かれていません。必要なコードはprogram_sourceから取り出せます。"
)


@dataclass(frozen=True)
class SourceCheckpoint:
    thread_id: str
    checkpoint_id: str
    timestamp: str
    values: dict[str, Any]


def serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=list(CUSTOM_STATE_TYPES))


def select_checkpoint(
    database: Path,
    stage: str,
    round_number: int,
    checkpoint_id: str | None = None,
) -> SourceCheckpoint:
    """Choose integration before handover; audit keeps its pre-integration report."""
    selected = []
    previous: dict[str, dict[str, Any]] = {}
    serde = serializer()
    # Read-only SQLite also sees committed WAL entries without modifying the run.
    with sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "SELECT thread_id, checkpoint_id, type, checkpoint FROM checkpoints "
            "WHERE checkpoint_ns = '' ORDER BY checkpoint_id"
        )
        for thread, cid, kind, payload in rows:
            checkpoint = serde.loads_typed((kind, payload))
            values = checkpoint["channel_values"]
            before = previous.get(thread, {})
            previous[thread] = values
            history = values.get("reconstruction")
            agent_state = values.get(f"{stage}_state") or {}
            if history is None or not agent_state.get("messages"):
                continue
            snapshot = history.snapshots[-1]
            if snapshot.round != round_number:
                continue
            if checkpoint_id is not None:
                matches = cid == checkpoint_id
            elif stage == "audit":
                matches = (
                    values.get("audit_report") is not None
                    and "branch:to:integrate_audit_report" in values
                )
            else:
                matches = (
                    snapshot.last_completed_stage == stage
                    and "branch:to:integrate_stage_submission" in before
                    and values.get("stage_validation_error") is None
                    and values.get("stage_submission") is None
                )
            if matches:
                selected.append(SourceCheckpoint(thread, cid, checkpoint["ts"], values))
    if len(selected) != 1:
        ids = ", ".join(item.checkpoint_id for item in selected) or "none"
        raise ValueError(
            f"Expected one {stage} round {round_number} checkpoint; found {ids}. "
            "Use --checkpoint-id to select a particular saved state."
        )
    return selected[0]


def load_system_prompt(
    run_dir: Path, stage: str, config: DictConfig
) -> tuple[str, list[Path]]:
    """Return the system prompt and every file read to obtain its text."""
    shared = stage != "audit" and config.workflow.get("share_thread", False)
    events = run_dir / "events.jsonl"
    if events.is_file():
        with events.open() as handle:
            for line in handle:
                event = json.loads(line)
                namespace = event.get("namespace") or []
                owner = namespace[0].split(":", 1)[0] if namespace else None
                valid_owner = (
                    owner in {"interpretation", "operations", "coding"}
                    if shared
                    else owner == stage
                )
                system = event.get("data", {}).get("system")
                if event["event"] == "prompt" and valid_owner and system:
                    content = (
                        system.get("content", "")
                        if isinstance(system, dict)
                        else system
                    )
                    return SystemMessage(content=content).text, [events]

    # Resumed runs may omit the already-reported system prompt. Use their frozen
    # prompt files where available, otherwise identify the local-file fallback.
    frozen = run_dir.parent / "code" / "zeroshot" / "pipeline" / "stages"
    stages_dir = (
        frozen if frozen.is_dir() else Path(__file__).parent / "pipeline" / "stages"
    )
    paths = [stages_dir / "_base/prompts/reconstruction_context.md"]
    if not shared:
        paths.append(stages_dir / f"{stage}/prompts/role.md")
    text = "\n\n".join(path.read_text().strip() for path in paths)
    return Template(text).substitute(
        reconstruction_path="/work/reconstruction.json"
    ), paths


def strings(value: Any) -> Iterator[str]:
    """Walk JSON values for file references, without parsing prose as paths."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)


def prepare_workspace(
    source: SourceCheckpoint, source_dir: Path, destination: Path
) -> list[str]:
    """Copy only referenced visual artifacts; keep submission text in the record."""
    destination.mkdir(parents=True, exist_ok=False)
    history = source.values["reconstruction"].model_dump(mode="json")
    report = source.values.get("audit_report")
    report = report.model_dump(mode="json") if report is not None else None
    evidence = source.values.get("audit_evidence") or {}
    missing = []
    relative_paths = set()
    for name in strings([history, report, evidence]):
        if "\n" in name or Path(name).suffix.lower() not in ASSET_SUFFIXES:
            continue
        path = Path(name)
        if path.is_relative_to(source_dir):
            relative = path.relative_to(source_dir)
        elif path.is_relative_to("/work"):
            relative = path.relative_to("/work")
        elif path.is_relative_to("/tmp"):
            relative = Path("tmp") / path.relative_to("/tmp")
        else:
            continue
        if ".." in relative.parts:
            raise ValueError(f"Artifact path escapes the workspace: {name}")
        relative_paths.add(relative)
        if path.suffix.lower() == ".dxf":
            relative_paths.add(relative.with_suffix(".png"))
    for relative in sorted(relative_paths):
        original = source_dir / relative
        if not original.resolve().is_relative_to(source_dir.resolve()):
            raise ValueError(f"Artifact escapes the original workspace: {original}")
        if not original.is_file():
            missing.append(str(PurePosixPath("/work") / relative))
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)

    # Change path values, not paths embedded in the submitted program or prose.
    def sandbox_paths(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: sandbox_paths(child) for key, child in value.items()}
        if isinstance(value, list):
            return [sandbox_paths(child) for child in value]
        if isinstance(value, str) and value.startswith(str(source_dir) + "/"):
            return "/work/" + value.removeprefix(str(source_dir) + "/")
        return value

    def save_record(filename: str, value: Any) -> None:
        text = json.dumps(sandbox_paths(value), ensure_ascii=False, indent=2)
        (destination / filename).write_text(text + "\n")

    save_record("reconstruction.json", history)
    if report is not None:
        save_record("audit_report.json", report)
    return missing


def build_agent(
    config: DictConfig,
    stage: str,
    source: SourceCheckpoint,
    workdir: SandboxWorkdir,
    system: str,
    checkpointer: SqliteSaver,
):
    """Use the source model and current stage tools, without submission/turn limits."""
    settings = config.workflow[f"{stage}_agent_builder"]
    runner_config = cast(
        dict[str, Any], OmegaConf.to_container(config.sandbox_runner, resolve=True)
    )
    runner_config["python_executable"] = Path(runner_config["python_executable"])
    runner = SandboxRunner(**runner_config)
    tools = [create_run_shell_tool(runner, workdir), create_load_image_tool(workdir)]
    if stage == "interpretation":
        tools.append(create_calculate_drawing_scale_tool())
    middleware: list[AgentMiddleware[Any, None, Any]] = [
        ToolErrorMiddleware(on_error=_handle_tool_error),
        PromptLogMiddleware(settings.role),
        ModelCallRetryMiddleware(
            max_retries=settings.get("model_retries", 5), role=settings.role
        ),
        StatelessReasoningMiddleware(),
    ]
    if stage == "coding":
        snapshot = source.values["reconstruction"].snapshots[-1]
        diff_config = config.workflow.get("diff_drawer_config")
        renderer = StepRenderer()
        verifier = OutputVerifier(
            executor=CadQueryExecutor(sandbox_runner=runner),
            workdir=workdir,
            renderer=renderer,
            diff_drawer=DrawingDiffExecutor(
                **cast(
                    dict[str, Any], OmegaConf.to_container(diff_config, resolve=True)
                )
            )
            if diff_config
            else None,
            feedback_presentation_mode=config.artifact_presenter.feedback_mode,
            attempt_store=AttemptStore(workdir, lambda: snapshot.round),
            show_intermediate_returns=config.workflow.get(
                "show_intermediate_returns", True
            ),
        )
        # Interpretation paths may be host paths in older runs; point the new
        # verifier at the copy, never the original workspace.
        record = json.loads((workdir.host_bind_dir / "reconstruction.json").read_text())
        restored = ReconstructionHistory.model_validate(record).snapshots[-1]
        verifier.interpretation = restored.interpretation
        verifier.operations = restored.operations
        tools.append(
            create_render_step_tool(
                workdir,
                renderer,
                drawing_frames=lambda: (
                    verifier.interpretation.view_frames()
                    if verifier.interpretation is not None
                    else {}
                ),
            )
        )
        middleware.append(
            VerifyOnWriteMiddleware(verifier, fingerprint=verifier.source_digest)
        )
    return create_agent(
        model=instantiate(settings.model),
        tools=tools,
        system_prompt=system,
        middleware=middleware,
        state_schema=AgentState,
        response_format=None,
        checkpointer=checkpointer,
        name=settings.role,
    )


def converse(
    agent,
    source: SourceCheckpoint,
    stage: str,
    writer: JsonlEventWriter,
    reporter: ConsoleReporter,
    missing: list[str],
) -> bool:
    """Return True on /reset; each branch has its own graph thread and workspace."""
    initial = copy.deepcopy(source.values[f"{stage}_state"]["messages"])
    introduction = INTRODUCTION
    if source.values.get("audit_report") is not None:
        introduction += "\nAudit submission are saved at /work/audit_report.json."
    if missing:
        introduction += "\nMissing reference files are not available:\n" + "\n".join(
            missing
        )
    initial.append(HumanMessage(content=introduction))
    first = True
    # LangGraph requires a positive integer, so use the platform maximum rather
    # than imposing a research turn budget or its default small recursion limit.
    run_config = {
        "configurable": {"thread_id": "conversation"},
        "recursion_limit": sys.maxsize,
    }
    while True:
        try:
            question = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            return False
        if question == "/quit":
            return False
        if question == "/reset":
            return True
        if not question:
            continue
        if question.startswith("/"):
            print("Commands: /reset, /quit. Otherwise enter a question.")
            continue
        writer.write(
            {
                "event": "user_question",
                "timestamp_ms": int(datetime.now(UTC).timestamp() * 1000),
                "namespace": [],
                "data": {"text": question},
            }
        )
        messages = [*(initial if first else []), HumanMessage(content=question)]
        stream = agent.stream_events(
            {"messages": messages},
            config=run_config,
            version="v3",
            durability="sync",
            transformers=[
                partial(RunEventTransformer, sink=writer.write),
                AgentMessageTransformer,
            ],
        )
        for channel, item in stream.interleave(
            RunEventTransformer.CHANNEL, AgentMessageTransformer.CHANNEL
        ):
            if channel == RunEventTransformer.CHANNEL:
                # Keep the full log, but omit repeated input and routine node
                # notifications from the terminal. Errors remain visible.
                if item["event"] in {"input", "prompt"}:
                    continue
                if item["event"] in {"node_started", "node_finished"} and not item[
                    "data"
                ].get("error"):
                    continue
                reporter.render_event(item)
            else:
                reporter.render_model_item(item)
        first = False


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run", required=True, type=Path, help="Original checkpoints.sqlite"
    )
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--round", required=True, type=int, dest="round_number")
    parser.add_argument(
        "--checkpoint-id",
        help="Select a specific checkpoint, including an unsuccessful submission",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="New session directory (must not exist)"
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Restore and inspect without making model calls",
    )
    args = parser.parse_args(argv)
    database = args.run.resolve()
    stage = STAGES[args.stage]
    source = select_checkpoint(database, stage, args.round_number, args.checkpoint_id)
    run_dir = database.parent
    config = OmegaConf.load(run_dir / ".hydra/config.yaml")
    if not isinstance(config, DictConfig):
        raise TypeError("Run config must be a YAML mapping")
    system, prompt_files = load_system_prompt(run_dir, stage, config)
    started_at = datetime.now(UTC).astimezone().strftime("%Y%m%d_%H%M%S")
    session = (
        args.output_dir
        or run_dir / "dialogues" / f"{stage}_{args.round_number:03d}_{started_at}"
    )
    session = session.resolve()
    if session.is_relative_to(run_dir / "workspace"):
        parser.error("--output-dir must be outside the original workspace")
    session.mkdir(parents=True, exist_ok=False)
    (session / "system_prompt.txt").write_text(system)
    metadata = {
        "source_database": str(database),
        "source_thread_id": source.thread_id,
        "source_checkpoint_id": source.checkpoint_id,
        "source_timestamp": source.timestamp,
        "stage": stage,
        "round": args.round_number,
        "system_prompt_files": [str(path) for path in prompt_files],
        "message_count": len(source.values[f"{stage}_state"]["messages"]),
        "source_config": str(run_dir / ".hydra/config.yaml"),
    }
    (session / "session.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Checkpoint: {source.checkpoint_id}; {metadata['message_count']} messages")
    print("System prompt files: " + ", ".join(map(str, prompt_files)))
    print(f"Session: {session}")
    for index in count():
        branch = session / f"branch_{index:03d}"
        workspace = branch / "workspace"
        missing = prepare_workspace(source, run_dir / "workspace", workspace)
        (branch / "missing_artifacts.json").write_text(
            json.dumps(missing, indent=2) + "\n"
        )
        print(f"Workspace: {workspace}; missing artifacts: {len(missing)}")
        if args.prepare_only:
            return
        with (
            SandboxWorkdir(host_bind_dir=workspace) as workdir,
            SqliteSaver.from_conn_string(str(branch / "checkpoints.sqlite")) as saver,
            JsonlEventWriter(
                branch / "events.jsonl",
                run_id=f"dialogue_{index}",
                sample_id=run_dir.name,
            ) as writer,
        ):
            saver.serde = serializer()
            agent = build_agent(config, stage, source, workdir, system, saver)
            if not converse(agent, source, stage, writer, ConsoleReporter(), missing):
                return


if __name__ == "__main__":
    main()
