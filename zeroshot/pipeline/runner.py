from __future__ import annotations

import shutil
from pathlib import Path
from typing import cast
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.sqlite import SqliteSaver

from zeroshot.pipeline.manifest import InputManifest
from zeroshot.pipeline.messages import MessageBuilder
from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.workflow import (
    create_reconstruction_graph,
    ReconstructionState,
)


class PipelineRunner:
    def __init__(
        self,
        model: BaseChatModel,
        message_builder: MessageBuilder,
        sandbox_runner: SandboxRunner,
        artifact_root: str | Path,
    ) -> None:
        self.model = model
        self.message_builder = message_builder
        self.sandbox_runner = sandbox_runner
        self.artifact_root = Path(artifact_root)

    def run_sample(self, manifest: InputManifest) -> ReconstructionState:
        checkpoint_path = (
            self.artifact_root / ".langgraph" / f"{manifest.sample_id}.sqlite"
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        with SandboxWorkdir() as workdir:
            staged_manifest = self._stage_inputs(
                manifest=manifest,
                workdir=workdir,
            )

            initial_messages = self.message_builder.build_initial(
                staged_manifest,
                workdir,
            )

            with SqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
                graph = create_reconstruction_graph(
                    model=self.model,
                    sandbox_runner=self.sandbox_runner,
                    sandbox_workdir=workdir,
                    checkpointer=checkpointer,
                )

                stream = graph.stream_events(
                    ReconstructionState(messages=initial_messages),
                    config={
                        "configurable": {"thread_id": f"{manifest.sample_id}:{uuid4()}"}
                    },
                    version="v3",
                    durability="sync",
                )
                for _ in stream:
                    pass
                result = stream.output

            # Save artifacts
            shutil.copytree(
                workdir.host_bind_dir,
                self.artifact_root / manifest.sample_id,
            )

        return cast(ReconstructionState, result)

    def _stage_inputs(
        self,
        manifest: InputManifest,
        workdir: SandboxWorkdir,
    ) -> InputManifest:
        staged_input_dir = workdir.host_bind_dir / "inputs"
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
