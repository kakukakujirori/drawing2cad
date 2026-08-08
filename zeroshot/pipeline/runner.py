from __future__ import annotations

import shutil
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast
from uuid import uuid4

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from zeroshot.pipeline.event_logging import (
    AgentMessageTransformer,
    ConsoleReporter,
    JsonlEventWriter,
    RunEventTransformer,
    has_run_completed,
)
from zeroshot.pipeline.messages import InputManifest, MessageBuilder
from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.verification import StepRenderer
from zeroshot.pipeline.workflow import CUSTOM_STATE_TYPES, ReconstructionState

# What the runner hands a graph: the run environment and the artifact contract.
# A graph's own settings are bound into the factory before it gets here, so
# swapping graphs never widens this signature.
GraphFactory = Callable[..., Any]

OnExisting = Literal["fail", "skip", "retry"]
_ON_EXISTING = ("fail", "skip", "retry")


def _clear_incomplete_run(sample_artifact_root: Path) -> None:
    """Remove what the runner writes, keeping what Hydra wrote for this job.

    Deleting the directory outright would take `.hydra/` and the job log with
    it, and those describe the run about to start.  Everything else is cleared
    by exclusion so a new artifact cannot survive a redo by being forgotten
    here.
    """
    if not sample_artifact_root.is_dir():
        return
    for entry in sample_artifact_root.iterdir():
        if entry.name == ".hydra" or entry.suffix == ".log":
            continue
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()


class PipelineRunner:
    WORKSPACE_DIRNAME = "workspace"

    def __init__(
        self,
        sandbox_runner: SandboxRunner,
        graph_factory: GraphFactory,
        message_builder: MessageBuilder,
        artifact_root: str | Path,
        renderer: StepRenderer,
        output_filename: str = "model.py",
        verification_dirname: PurePosixPath = PurePosixPath("attempts"),
        on_existing: OnExisting = "fail",
        console_reporter: ConsoleReporter | None = None,
    ) -> None:
        if on_existing not in _ON_EXISTING:
            raise ValueError(
                f"on_existing must be one of {_ON_EXISTING}: {on_existing!r}"
            )
        self.message_builder = message_builder
        self.sandbox_runner = sandbox_runner
        self.renderer = renderer
        self.artifact_root = Path(artifact_root)
        self.output_filename = output_filename
        self.verification_dirname = verification_dirname
        self.graph_factory = graph_factory
        self.on_existing = on_existing
        self.console_reporter = console_reporter

    def run_sample(self, manifest: InputManifest) -> ReconstructionState | None:
        """Run one sample, or return None when the policy skips it.

        The directory may already exist because Hydra writes its resolved config
        there before the job body runs, so `events.jsonl` is what says whether a
        run happened.  No policy ever *skips* an incomplete one: that would let
        a sweep report every sample as done while some produced nothing.
        """
        sample_artifact_root = self.artifact_root / manifest.sample_id
        events_path = sample_artifact_root / "events.jsonl"
        if has_run_completed(events_path):
            if self.on_existing in {"skip", "retry"}:
                return None
            raise FileExistsError(f"sample already ran: {sample_artifact_root}")
        if self.on_existing == "retry":
            _clear_incomplete_run(sample_artifact_root)
        elif events_path.exists():
            raise FileExistsError(
                f"incomplete run left behind, delete it to redo: {sample_artifact_root}"
            )
        sample_artifact_root.mkdir(parents=True, exist_ok=True)

        run_id = f"{manifest.sample_id}:{uuid4()}"
        checkpoint_path = sample_artifact_root / "checkpoints.sqlite"

        with ExitStack() as stack:
            # instantiate event loggers (available only in this block)
            event_writer = stack.enter_context(
                JsonlEventWriter(
                    sample_artifact_root / "events.jsonl",
                    run_id=run_id,
                    sample_id=manifest.sample_id,
                )
            )
            if self.console_reporter is not None:
                stack.enter_context(
                    self.console_reporter.run_context(
                        run_id=run_id,
                        sample_id=manifest.sample_id,
                    )
                )

            # instantiate sandbox directly in the persistent sample workspace
            workspace_path = sample_artifact_root / self.WORKSPACE_DIRNAME
            workspace_path.mkdir()
            workdir = stack.enter_context(SandboxWorkdir(host_bind_dir=workspace_path))

            # copy input files to sandbox
            input_dirname = PurePosixPath("inputs")
            workdir.read_only_subdirs.append(input_dirname)
            staged_manifest = self._stage_inputs(
                manifest=manifest,
                workdir=workdir,
                input_dirname=input_dirname,
            )

            # what this run offers the model; the graph decides how to open with it
            input_message = self.message_builder.build_input_message(
                manifest=staged_manifest,
                workdir=workdir,
            )

            # instantiate the graph
            checkpointer = stack.enter_context(
                SqliteSaver.from_conn_string(str(checkpoint_path))
            )
            checkpointer.serde = JsonPlusSerializer(
                allowed_msgpack_modules=list(CUSTOM_STATE_TYPES)
            )
            graph = self.graph_factory(
                sandbox_runner=self.sandbox_runner,
                sandbox_workdir=workdir,
                renderer=self.renderer,
                message_builder=self.message_builder,
                input_message=input_message,
                output_filename=self.output_filename,
                verification_dirname=self.verification_dirname,
                checkpointer=checkpointer,
            )

            # run the graph
            stream = graph.stream_events(
                ReconstructionState(messages=[]),
                config={"configurable": {"thread_id": run_id}},
                version="v3",
                durability="sync",
                transformers=[RunEventTransformer, AgentMessageTransformer],
            )
            channels = (
                (RunEventTransformer.CHANNEL, AgentMessageTransformer.CHANNEL)
                if self.console_reporter is not None
                else (RunEventTransformer.CHANNEL,)
            )
            for channel, item in stream.interleave(*channels):
                if channel == RunEventTransformer.CHANNEL:
                    event_writer.write(item)
                    if self.console_reporter is not None:
                        self.console_reporter.render_event(item)
                else:
                    assert self.console_reporter is not None
                    self.console_reporter.render_message(item)
            result = stream.output

        return cast(ReconstructionState, result)

    def _stage_inputs(
        self,
        manifest: InputManifest,
        workdir: SandboxWorkdir,
        input_dirname: str | PurePosixPath = "inputs",
    ) -> InputManifest:
        staged_input_dir = workdir.host_bind_dir / input_dirname
        staged_input_dir.mkdir(parents=True, exist_ok=False)

        staged_dxf_path = staged_input_dir / "techdraw.dxf"
        shutil.copyfile(manifest.dxf_path, staged_dxf_path)

        staged_render_paths: dict[str, Path] = {}

        if self.message_builder.access_render3d != "none":
            for style in self.message_builder.access_render3d_styles:
                source_path = manifest.render3d_paths[style]
                suffix = source_path.suffix or ".png"
                staged_path = staged_input_dir / f"{style}{suffix}"
                shutil.copyfile(source_path, staged_path)
                staged_render_paths[style] = staged_path

        return InputManifest(
            sample_id=manifest.sample_id,
            dxf_path=staged_dxf_path,
            render3d_paths=staged_render_paths,
        )
