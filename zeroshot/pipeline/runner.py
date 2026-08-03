from __future__ import annotations

import shutil
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from typing import cast
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.sqlite import SqliteSaver

from zeroshot.pipeline.event_logging import (
    ConsoleReporter,
    JsonlEventWriter,
    RunEventTransformer,
)
from zeroshot.pipeline.manifest import InputManifest
from zeroshot.pipeline.messages import MessageBuilder
from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.verification import StepRenderer
from zeroshot.pipeline.workflow import (
    ReconstructionState,
    create_reconstruction_graph,
)


class PipelineRunner:
    WORKSPACE_DIRNAME = "workspace"

    def __init__(
        self,
        model: BaseChatModel,
        message_builder: MessageBuilder,
        sandbox_runner: SandboxRunner,
        artifact_root: str | Path,
        renderer: StepRenderer,
        output_filename: str = "model.py",
        verification_dirname: PurePosixPath = PurePosixPath("attempts"),
        max_agent_turns: int = 30,
        console_reporter: ConsoleReporter | None = None,
    ) -> None:
        self.model = model
        self.message_builder = message_builder
        self.sandbox_runner = sandbox_runner
        self.renderer = renderer
        self.artifact_root = Path(artifact_root)
        self.output_filename = output_filename
        self.verification_dirname = verification_dirname
        self.max_agent_turns = max_agent_turns
        self.console_reporter = console_reporter

    def run_sample(self, manifest: InputManifest) -> ReconstructionState:
        sample_artifact_root = self.artifact_root / manifest.sample_id
        sample_artifact_root.mkdir(parents=True, exist_ok=False)

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

            # prepare initial messages
            initial_messages = self.message_builder.build_initial(
                manifest=staged_manifest,
                workdir=workdir,
                output_filename=self.output_filename,
                verification_dirname=self.verification_dirname,
            )

            # instantiate the graph
            checkpointer = stack.enter_context(
                SqliteSaver.from_conn_string(str(checkpoint_path))
            )
            graph = create_reconstruction_graph(
                model=self.model,
                sandbox_runner=self.sandbox_runner,
                sandbox_workdir=workdir,
                renderer=self.renderer,
                message_builder=self.message_builder,
                output_filename=self.output_filename,
                verification_dirname=self.verification_dirname,
                max_agent_turns=self.max_agent_turns,
                checkpointer=checkpointer,
            )

            # run the graph
            stream = graph.stream_events(
                ReconstructionState(messages=initial_messages),
                config={"configurable": {"thread_id": run_id}},
                version="v3",
                durability="sync",
                transformers=[RunEventTransformer],
            )
            channels = (
                ("run_events", "messages")
                if self.console_reporter is not None
                else ("run_events",)
            )
            for channel, item in stream.interleave(*channels):
                if channel == "run_events":
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
