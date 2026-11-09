"""Choose among completed round submissions without rebuilding or rewriting them."""

import json
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.messages.content import create_text_block
from langchain_core.runnables import RunnableConfig

from zeroshot.pipeline.sandbox import SandboxRunner
from zeroshot.pipeline.stages._base.prompt import (
    PromptTemplate,
    StageInstructions,
    build_system_prompt,
    describe_placement,
)
from zeroshot.pipeline.stages._base.validate import SubmissionValidationError
from zeroshot.pipeline.stages.contracts import (
    FinalSelection,
    ReconstructionHistory,
    ReconstructionSnapshot,
)
from zeroshot.pipeline.stages.types import PipelineStage
from zeroshot.pipeline.tools import pick_named_tools
from zeroshot.pipeline.tools.calculate_drawing_scale import (
    create_calculate_drawing_scale_tool,
)
from zeroshot.pipeline.tools.fit_edge import create_fit_edge_tool
from zeroshot.pipeline.tools.load_image import create_load_image_tool
from zeroshot.pipeline.tools.run_shell import RunShellContext, create_run_shell_tool
from zeroshot.pipeline.verification import AttemptStore, ExecutionStatus
from zeroshot.pipeline.verification.run_cadquery import (
    CadQueryExecutor,
    StepVerificationError,
)
from zeroshot.pipeline.workflow._config import _child_graph_config
from zeroshot.pipeline.workflow.lifecycle import (
    restore_selected_model,
    save_reconstruction,
)
from zeroshot.pipeline.workflow.middleware import VerifyOnSubmitMiddleware
from zeroshot.pipeline.workflow.middleware.output_limit_budget import OutputLimitBudget
from zeroshot.pipeline.workflow.state import ReconstructionState


def _candidates(
    snapshots: Sequence[ReconstructionSnapshot],
    attempts: AttemptStore,
    output_filename: str,
) -> dict[int, Path]:
    """Only completed submissions with their original program and a valid STEP."""
    candidates = {}
    for snapshot in snapshots:
        verification = snapshot.verification
        if (
            snapshot.last_completed_stage is not PipelineStage.CODING
            or snapshot.program_source is None
            or verification is None
            or verification.verification_id is None
            or verification.exec_report is None
            or verification.exec_report.status is not ExecutionStatus.VERIFIED
            or verification.exec_report.returncode != 0
        ):
            continue
        directory = attempts.workdir.sandbox_to_host_path(
            attempts.sandbox_attempt_dir(
                snapshot.round, "coding", verification.verification_id
            )
        )
        program = directory / output_filename
        if not program.is_file() or program.is_symlink():
            continue
        try:
            source = program.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if source != snapshot.program_source:
            continue
        try:
            CadQueryExecutor.verify_step(directory / "output.step")
        except StepVerificationError:
            continue
        candidates[snapshot.round] = directory
    return candidates


def select_final_model(
    state: ReconstructionState,
    config: RunnableConfig,
    *,
    audit_agent_builder: partial[Any],
    instructions: StageInstructions,
    attempts: AttemptStore,
    sandbox_runner: SandboxRunner,
    tool_names: Sequence[str],
    history_path: Path,
    output_filename: str,
    interpretation_filename: str,
    output_limit_budget: OutputLimitBudget,
) -> dict[str, Any]:
    """Use the auditor's model and turn budget for a separate selection task."""
    history = state["reconstruction"]
    workdir = instructions.workdir
    candidates = _candidates(history.snapshots, attempts, output_filename)
    if not candidates:
        return {}  # There is no exported model to choose or restore.

    update = {}
    if len(candidates) == 1:
        choice = FinalSelection(
            round=next(iter(candidates)),
            reason="Only one completed round submission has its original program and a verified STEP.",
        )
    else:

        def validate(answer: FinalSelection, _context: RunShellContext) -> None:
            if answer.round not in candidates:
                raise SubmissionValidationError(
                    f"choose a verified candidate round from {sorted(candidates)}"
                )

        available = [
            create_run_shell_tool(sandbox_runner, workdir),
            create_load_image_tool(workdir),
            create_fit_edge_tool(workdir),
            create_calculate_drawing_scale_tool(),
        ]
        stages_dir = Path(__file__).parents[1] / "stages"
        agent = audit_agent_builder(
            role="final_selector",
            tools=pick_named_tools(
                available,
                [name for name in tool_names if name != "evaluate_cad_hypothesis"],
            ),
            system_prompt=build_system_prompt(
                stages_dir / "audit/prompts/select.md",
                instructions.prompt_context,
                FinalSelection,
            ),
            output_schema=FinalSelection,
            extra_middleware=[VerifyOnSubmitMiddleware(validate)],
            output_limit_budget=output_limit_budget,
        )
        listing = [
            {
                "round": round_number,
                "program": str(
                    workdir.host_to_sandbox_path(directory / output_filename)
                ),
                "step": str(workdir.host_to_sandbox_path(directory / "output.step")),
                "artifacts": str(workdir.host_to_sandbox_path(directory)),
            }
            for round_number, directory in candidates.items()
        ]
        message = HumanMessage(
            content_blocks=[
                create_text_block(
                    PromptTemplate(
                        stages_dir / "_base/prompts/coordinate_frames.md"
                    ).render(
                        projection_placement=describe_placement(
                            instructions.projection_method
                        )
                    )
                ),
                create_text_block("[Candidate submissions]\n" + json.dumps(listing)),
                *instructions.create_artifact_message().content_blocks,
            ]
        )
        protected = [
            workdir.host_bind_dir / output_filename,
            workdir.host_bind_dir / interpretation_filename,
            history_path,
        ]
        result = agent.invoke(
            {"messages": [message]},
            config=_child_graph_config(config),
            context=RunShellContext(
                read_only_files=tuple(
                    str(workdir.host_to_sandbox_path(path))
                    for path in protected
                    if path.is_file()
                )
            ),
        )
        choice = result.get("structured_response")
        if not isinstance(choice, FinalSelection):
            raise SubmissionValidationError(
                "final selector did not return a model choice"
            )
        validate(choice, RunShellContext(read_only_files=()))
        update["selection_state"] = result

    # Recheck the chosen immutable artifacts before making it the submission.
    if choice.round not in _candidates(
        [history.snapshots[choice.round]], attempts, output_filename
    ):
        raise SubmissionValidationError("the selected candidate is no longer available")
    selected = ReconstructionHistory.model_validate(
        {**dict(history), "final_selection": choice}
    )
    restore_selected_model(
        selected,
        workdir.host_bind_dir / output_filename,
        workdir.host_bind_dir / interpretation_filename,
    )
    save_reconstruction(history_path, selected)
    return {**update, "reconstruction": selected}
