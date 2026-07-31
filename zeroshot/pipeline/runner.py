from __future__ import annotations

import shutil
from pathlib import Path
from typing import cast

from langchain_core.language_models import BaseChatModel

from zeroshot.pipeline.manifest import InputManifest
from zeroshot.pipeline.messages import MessageBuilder
from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.tools import (
    create_load_image_tool,
    create_run_shell_tool,
    create_verify_output_tool,
)
from zeroshot.pipeline.verification import CadQueryExecutor
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
        artifact_root: Path,
    ) -> None:
        self.model = model
        self.message_builder = message_builder
        self.sandbox_runner = sandbox_runner
        self.artifact_root = Path(artifact_root)

    def run_sample(self, manifest: InputManifest) -> ReconstructionState:
        sample_artifact_root = self.artifact_root / manifest.sample_id / "verifications"
        sample_artifact_root.mkdir(parents=True, exist_ok=True)

        executor = CadQueryExecutor(
            artifact_root=sample_artifact_root,
            sandbox_runner=self.sandbox_runner,
        )

        with SandboxWorkdir() as workdir:
            staged_manifest = self._stage_inputs(
                manifest=manifest,
                workdir=workdir,
            )

            initial_messages = self.message_builder.build_initial(
                staged_manifest,
                workdir,
            )

            tools = [
                create_run_shell_tool(self.sandbox_runner, workdir),
                create_load_image_tool(workdir),
                create_verify_output_tool(
                    executor=executor,
                    workdir=workdir,
                    render_views=False,
                ),
            ]

            agent_with_tools = self.model.bind_tools(tools)

            graph = create_reconstruction_graph(
                agent_with_tools=agent_with_tools,
                tools=tools,
            )

            result = graph.invoke(ReconstructionState(messages=initial_messages))

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
