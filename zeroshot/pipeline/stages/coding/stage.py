import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, cast

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages.content import create_text_block
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.pregel import Pregel
from omegaconf import DictConfig, OmegaConf

from zeroshot.pipeline.messages.artifact import ArtifactPresenter
from zeroshot.pipeline.messages.manifest import read_dxf_frame
from zeroshot.pipeline.sandbox import SandboxRunner
from zeroshot.pipeline.stages._base.prompt import (
    StageInstructions,
    build_system_prompt,
    schema_for_prompt,
)
from zeroshot.pipeline.stages.coding.middleware import (
    CodingMiddleware,
    CodingTrialMiddleware,
    FreshCodingMiddleware,
)
from zeroshot.pipeline.stages.coding.progress import (
    DEFAULT_MATCH_MARGIN_PX,
    ProgressOutputVerifier,
)
from zeroshot.pipeline.stages.coding.workspace import CodingWorkspaceVerifier
from zeroshot.pipeline.stages.interpretation.contracts import DrawingInterpretation
from zeroshot.pipeline.stages.interpretation.verify import InterpretationVerifier
from zeroshot.pipeline.stages.tickets.contracts import TicketAnswers
from zeroshot.pipeline.stages.tickets.verify import TicketVerifier
from zeroshot.pipeline.stages.types import PipelineStage
from zeroshot.pipeline.tools.calculate_drawing_scale import (
    create_calculate_drawing_scale_tool,
)
from zeroshot.pipeline.tools.render_step import create_render_step_tool
from zeroshot.pipeline.verification import (
    AttemptStore,
    CadQueryExecutor,
    DrawingDiffExecutor,
    StepRenderer,
)
from zeroshot.pipeline.workflow._config import _child_graph_config
from zeroshot.pipeline.workflow.lifecycle import interpretation_baseline
from zeroshot.pipeline.workflow.state import ReconstructionState, current_snapshot

type CompiledGraph = Pregel[Any, Any, Any, Any]
type AgentBuilder = partial[CompiledGraph]


@dataclass(frozen=True)
class CodingStage:
    agent: CompiledGraph
    instructions: StageInstructions
    workspace: CodingWorkspaceVerifier
    ticket_verifier: TicketVerifier
    middleware: CodingMiddleware
    input_after_compaction: bool
    dxf_context: str | None = None

    def run(self, state: ReconstructionState, config: RunnableConfig) -> dict[str, Any]:
        snapshot = current_snapshot(state)
        if snapshot.last_completed_stage is not None:
            raise RuntimeError("coding starts a round")

        # A validation retry continues this round's workspace and builds.
        retry = state.get("stage_validation_error") is not None
        if not retry:
            self.workspace.reset(
                interpretation_baseline(state["reconstruction"]),
                [ticket.ticket_id for ticket in snapshot.open_tickets],
            )
            self.middleware.reset()  # NOTE: must be after the workspace is seeded
        self.ticket_verifier.reset(state["reconstruction"])

        previous = state.get("coding_state") or {}
        instruction = self.instructions.build(
            state,
            PipelineStage.CODING,
            append_inputs=(not previous or self.input_after_compaction),
            interpretation_output_path=str(
                self.instructions.workdir.sandbox_bind_dir
                / self.workspace.interpretation.source_filename
            ),
            interpretation_schema=schema_for_prompt(DrawingInterpretation),
            feedback=(
                [create_text_block(self.dxf_context)]
                if self.dxf_context is not None and not retry
                else []
            ),
        )
        messages = [
            *list(previous.get("messages") or []),
            *self.middleware.opening(
                state, self.instructions, instruction, retry=retry
            ),
        ]
        result = self.agent.invoke(
            {
                **previous,
                "messages": messages,
            },
            config=_child_graph_config(config),
        )
        return {
            "coding_state": result,
            "stage_submission": result.get("structured_response"),
        }


def _dxf_context(instructions: StageInstructions) -> str | None:
    originals = []
    for view in instructions.input_artifact:
        if Path(view.file).suffix.lower() != ".dxf":
            continue

        file = instructions.workdir.host_to_sandbox_path(view.file)
        originals.append(
            {
                "view": view.name,
                "file": str(file),
                **read_dxf_frame(instructions.workdir.sandbox_to_host_path(file)),
            }
        )
    if not originals:
        return None
    return (
        "[Native DXF coordinates]\n"
        "The original DXFs below are already registered. Keep their names, "
        "files, roles and full-file regions; add dimension readings. "
        "A drawing unit is a millimetre and nothing is moved, so box_uv is "
        "what you read out of the file. box_mm is that file's full extent.\n"
        + json.dumps({"originals": originals})
    )


def create_coding_stage(
    coding_agent_builder: AgentBuilder,
    tools: Sequence[BaseTool],
    role_path: Path | None,
    instructions: StageInstructions,
    prompt_context: dict[str, str],
    attempt_store: AttemptStore,
    sandbox_runner: SandboxRunner,
    diff_drawer_config: dict[str, Any] | None,
    artifact_presenter: ArtifactPresenter,
    output_filename: str = "model.py",
    interpretation_filename: str = "interpretation.json",
    input_after_compaction: bool = False,
    fresh_memory: bool = False,
    coding_trial_reminder: bool = True,
    match_margin_px: float = DEFAULT_MATCH_MARGIN_PX,
) -> CodingStage:
    if isinstance(diff_drawer_config, DictConfig):
        diff_drawer_config = cast(
            dict[str, Any], OmegaConf.to_container(diff_drawer_config, resolve=True)
        )

    renderer = StepRenderer()
    workspace = CodingWorkspaceVerifier(
        InterpretationVerifier(
            workdir=instructions.workdir,
            attempt_store=attempt_store,
            input_artifact=instructions.input_artifact,
            source_filename=interpretation_filename,
        ),
        ProgressOutputVerifier(
            executor=CadQueryExecutor(sandbox_runner=sandbox_runner),
            workdir=instructions.workdir,
            renderer=renderer,
            diff_drawer=(
                DrawingDiffExecutor(**diff_drawer_config)
                if diff_drawer_config is not None
                else None
            ),
            artifact_presenter=artifact_presenter,
            attempt_store=attempt_store,
            source_filename=output_filename,
            match_margin_px=match_margin_px,
        ),
    )
    ticket_verifier = TicketVerifier()
    middleware_type = FreshCodingMiddleware if fresh_memory else CodingMiddleware
    coding_middleware = middleware_type(
        workspace,
        ticket_verifier=ticket_verifier,
        fingerprint=workspace.source_digest,
    )
    # Resolve frames on each call: they follow the latest valid interpretation.
    coding_tools = [
        *tools,
        create_calculate_drawing_scale_tool(),
        create_render_step_tool(
            instructions.workdir,
            renderer,
            drawing_frames=lambda: (
                workspace.output.interpretation.view_frames()
                if workspace.output.interpretation is not None
                else {}
            ),
        ),
    ]
    max_turns = coding_agent_builder.keywords["max_turns"]
    extra_middleware: list[AgentMiddleware[Any, None, Any]] = [coding_middleware]
    if coding_trial_reminder:
        extra_middleware.append(
            CodingTrialMiddleware(
                tool_names=[tool.name for tool in coding_tools],
                max_turns=max_turns,
                ready=lambda: workspace.interpretation_ready,
            )
        )
    coding_agent = coding_agent_builder(
        tools=coding_tools,
        system_prompt=build_system_prompt(
            role_path,
            prompt_context | {"max_turns": max_turns},
            TicketAnswers,
        ),
        output_schema=TicketAnswers,
        extra_middleware=extra_middleware,
    )
    return CodingStage(
        agent=coding_agent,
        instructions=instructions,
        workspace=workspace,
        ticket_verifier=ticket_verifier,
        middleware=coding_middleware,
        input_after_compaction=input_after_compaction,
        dxf_context=_dxf_context(instructions),
    )
