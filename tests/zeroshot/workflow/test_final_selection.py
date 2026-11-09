import json
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path

import cadquery as cq
import pytest
from langchain_core.messages import AIMessage
from PIL import Image

from tests.zeroshot.chat_models import ScriptedChatModel, tool_call
from tests.zeroshot.contracts import interpretation
from tests.zeroshot.workflow.test_graph import (
    _accepted_audit,
    _audit_script,
    _coding_script,
    _graph,
    _stub_verification,
)
from tests.zeroshot.workflow.test_reconstruction_workflow import _completed_run, _report
from zeroshot.evaluation.run_scoring import latest_verified_step
from zeroshot.pipeline.messages.manifest import register_view
from zeroshot.pipeline.runner import _latest_program_source
from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.stages._base.prompt import StageInstructions
from zeroshot.pipeline.stages._base.validate import SubmissionValidationError
from zeroshot.pipeline.stages.coding.verify import VerifyOutputResult
from zeroshot.pipeline.stages.contracts import FinalSelection, ReconstructionHistory
from zeroshot.pipeline.stages.interpretation.contracts import ProjectionMethod, View
from zeroshot.pipeline.verification import AttemptStore, ExecutionStatus
from zeroshot.pipeline.verification.run_cadquery import CadQueryExecutionReport
from zeroshot.pipeline.workflow import create_agent
from zeroshot.pipeline.workflow.final_selection import select_final_model
from zeroshot.pipeline.workflow.lifecycle import (
    load_reconstruction,
    open_next_round,
    save_reconstruction,
)
from zeroshot.pipeline.workflow.middleware.output_limit_budget import OutputLimitBudget


def _history(workdir):
    history = _completed_run()
    history = _completed_run(open_next_round(history, _report()))
    attempts = AttemptStore(workdir, lambda: 1)
    snapshots = []
    for snapshot in history.snapshots:
        directory = attempts.host_root / f"round_{snapshot.round:03d}/coding/000"
        directory.mkdir(parents=True)
        source = f"import cadquery as cq\nresult = cq.Workplane().box(10, 10, {snapshot.round + 1})\n"
        (directory / "model.py").write_text(source)
        cq.exporters.export(
            cq.Workplane().box(10, 10, snapshot.round + 1),
            str(directory / "output.step"),
        )
        verification = VerifyOutputResult(
            verification_id="000",
            exec_report=CadQueryExecutionReport(
                status=ExecutionStatus.VERIFIED,
                returncode=0,
                step_path=directory / "output.step",
            ),
        )
        snapshots.append(
            snapshot.model_copy(
                update={
                    "program_source": source,
                    "verification": verification,
                    "interpretation": interpretation(
                        f"round {snapshot.round} geometry"
                    ),
                }
            )
        )
    path = workdir.host_bind_dir / "drawing.png"
    Image.new("RGB", (20, 20), "white").save(path)
    original = register_view("view_front", View.FRONT, path, axes=("+x", "+z"))
    return ReconstructionHistory.model_validate(
        {
            **dict(history),
            "snapshots": snapshots,
            "input_drawings": [
                original.model_copy(update={"file": "/work/drawing.png"})
            ],
        }
    ), attempts


def _select(
    history,
    attempts,
    model,
    *,
    tool_names=(),
    projection_method=ProjectionMethod.THIRD_ANGLE,
):
    workdir = attempts.workdir
    path = workdir.host_bind_dir / "reconstruction.json"
    save_reconstruction(path, history)
    (workdir.host_bind_dir / "model.py").write_text(
        history.snapshots[-1].program_source
    )
    (workdir.host_bind_dir / "interpretation.json").write_text(
        history.snapshots[-1].interpretation.model_dump_json()
    )
    instructions = StageInstructions(
        input_artifact=[
            view.model_copy(
                update={"file": str(workdir.sandbox_to_host_path(view.file))}
            )
            for view in history.input_drawings
        ],
        input_presentation_mode="path",
        prompt_context={
            "reconstruction_path": "/work/reconstruction.json",
            "drawing_skeleton": "",
        },
        workdir=workdir,
        projection_method=projection_method,
    )
    return select_final_model(
        {"reconstruction": history},
        {},
        audit_agent_builder=partial(
            create_agent,
            role="output_auditor",
            model=model,
            max_turns=3,
            model_call_retries=1,
            announce_turns=False,
        ),
        instructions=instructions,
        attempts=attempts,
        sandbox_runner=SandboxRunner(
            python_executable=Path(sys.executable), default_timeout_s=10
        ),
        tool_names=tool_names,
        history_path=path,
        output_filename="model.py",
        interpretation_filename="interpretation.json",
        output_limit_budget=OutputLimitBudget(3),
    )


def test_selecting_an_earlier_round_restores_the_pair_and_the_step_to_score(tmp_path):
    root = tmp_path / "run" / "workspace"
    root.mkdir(parents=True)
    with SandboxWorkdir(host_bind_dir=root) as workdir:
        history, attempts = _history(workdir)
        choice = FinalSelection(
            round=0, reason="The earlier model preserves the through opening."
        )
        result = _select(
            history,
            attempts,
            ScriptedChatModel(responses=(AIMessage(choice.model_dump_json()),)),
        )
        selected = result["reconstruction"]

        assert selected.snapshots == history.snapshots
        assert selected.final_selection == choice
        assert load_reconstruction(root / "reconstruction.json") == selected
        assert (root / "model.py").read_text() == history.snapshots[0].program_source
        assert json.loads(
            (root / "interpretation.json").read_text()
        ) == history.snapshots[0].interpretation.model_dump(mode="json")
        assert _latest_program_source(selected) == history.snapshots[0].program_source
        assert (
            latest_verified_step(root.parent)[0]
            == attempts.host_root / "round_000/coding/000/output.step"
        )


@pytest.mark.parametrize(
    "method", [ProjectionMethod.FIRST_ANGLE, ProjectionMethod.THIRD_ANGLE]
)
def test_selector_receives_the_shared_context_projection_rules_and_image_legend(
    tmp_path, method
):
    with SandboxWorkdir(host_bind_dir=tmp_path) as workdir:
        history, attempts = _history(workdir)
        model = ScriptedChatModel(
            responses=(
                AIMessage(
                    FinalSelection(
                        round=0, reason="Fits the drawing."
                    ).model_dump_json()
                ),
            )
        )

        _select(history, attempts, model, projection_method=method)

        messages = model.received_messages[0]
        system = messages[0].text
        human = messages[1].text
        assert "## Reconstruction workflow" in system
        assert "No further coding rounds will run" in system
        assert "magenta" in system.lower() and "green" in system.lower()
        assert "hatching" in system.lower()
        assert "## Coordinate frames" in human
        assert f"This drawing uses {method.replace('_', '-')} projection" in human
        assert "[Candidate submissions]" in human


def test_an_unknown_round_is_corrected_before_any_model_is_restored(tmp_path):
    with SandboxWorkdir(host_bind_dir=tmp_path) as workdir:
        history, attempts = _history(workdir)
        model = ScriptedChatModel(
            responses=(
                AIMessage(
                    FinalSelection(round=8, reason="Unknown round.").model_dump_json()
                ),
                AIMessage(
                    FinalSelection(
                        round=1, reason="This is an available candidate."
                    ).model_dump_json()
                ),
            )
        )
        result = _select(history, attempts, model)

        assert result["reconstruction"].final_selection.round == 1
        assert len(model.received_messages) == 2
        assert (
            "choose a verified candidate round from [0, 1]"
            in model.received_messages[1][-1].text
        )


def test_a_selector_without_an_answer_preserves_the_existing_submission(tmp_path):
    with SandboxWorkdir(host_bind_dir=tmp_path) as workdir:
        history, attempts = _history(workdir)
        model = ScriptedChatModel(
            responses=tuple(AIMessage("Still comparing.") for _ in range(3))
        )

        with pytest.raises(
            SubmissionValidationError, match="did not return a model choice"
        ):
            _select(history, attempts, model)

        assert load_reconstruction(tmp_path / "reconstruction.json") == history
        assert (tmp_path / "model.py").read_text() == history.snapshots[
            -1
        ].program_source


@pytest.mark.parametrize(
    "target", ["model.py", "attempts/round_000/coding/000/model.py"]
)
def test_selector_cannot_edit_the_workspace_program_or_candidates(tmp_path, target):
    with SandboxWorkdir(host_bind_dir=tmp_path) as workdir:
        history, attempts = _history(workdir)
        command = f"python -c \"open('/work/{target}', 'w').write('changed')\""
        choice = FinalSelection(
            round=0, reason="The original first candidate fits best."
        )
        model = ScriptedChatModel(
            responses=(
                tool_call("run_shell", {"command": command}, "write"),
                AIMessage(choice.model_dump_json()),
            )
        )
        result = _select(history, attempts, model, tool_names=("run_shell",))

        tool_result = next(
            m for m in result["selection_state"]["messages"] if m.type == "tool"
        )
        assert json.loads(tool_result.text)["returncode"] != 0
        assert (
            attempts.host_root / "round_000/coding/000/model.py"
        ).read_text() == history.snapshots[0].program_source


@pytest.mark.parametrize("valid_candidates", [0, 1])
def test_zero_or_one_valid_candidate_needs_no_model_call(tmp_path, valid_candidates):
    with SandboxWorkdir(host_bind_dir=tmp_path) as workdir:
        history, attempts = _history(workdir)
        for round_number in range(valid_candidates, 2):
            (
                attempts.host_root / f"round_{round_number:03d}/coding/000/output.step"
            ).unlink()
        model = ScriptedChatModel(responses=())
        result = _select(history, attempts, model)

        assert model.received_messages == []
        if valid_candidates:
            assert result["reconstruction"].final_selection.round == 0
        else:
            assert result == {}
            assert (
                load_reconstruction(tmp_path / "reconstruction.json").final_selection
                is None
            )


@pytest.mark.parametrize("enabled", [False, True])
def test_completed_graph_routes_to_selection_only_when_enabled(tmp_path, enabled):
    with SandboxWorkdir(host_bind_dir=tmp_path) as workdir:
        history, _ = _history(workdir)
        choice = FinalSelection(
            round=0, reason="The earlier model agrees with the source."
        )
        auditor = ScriptedChatModel(responses=(AIMessage(choice.model_dump_json()),))
        result = _graph(
            workdir,
            coder=ScriptedChatModel(responses=()),
            auditor=auditor,
            max_audit_reject_count=1,
            final_selection=enabled,
        ).invoke({"reconstruction": history})

        assert len(auditor.received_messages) == int(enabled)
        assert result["reconstruction"].final_selection == (choice if enabled else None)


def test_resuming_a_selected_run_restores_it_without_another_model_call(tmp_path):
    with SandboxWorkdir(host_bind_dir=tmp_path) as workdir:
        history, _ = _history(workdir)
        selected = ReconstructionHistory.model_validate(
            {
                **dict(history),
                "final_selection": FinalSelection(
                    round=0, reason="Chosen before resume."
                ),
            }
        )
        coder = ScriptedChatModel(responses=())
        auditor = ScriptedChatModel(responses=())
        result = _graph(
            workdir,
            coder=coder,
            auditor=auditor,
            final_selection=True,
        ).invoke({"reconstruction": selected})

        assert result["reconstruction"] == selected
        assert (tmp_path / "model.py").read_text() == selected.snapshots[
            0
        ].program_source
        assert json.loads(
            (tmp_path / "interpretation.json").read_text()
        ) == selected.snapshots[0].interpretation.model_dump(mode="json")
        assert coder.received_messages == auditor.received_messages == []


def test_early_audit_acceptance_also_finishes_with_selection(tmp_path):
    with SandboxWorkdir(host_bind_dir=tmp_path) as workdir:
        history, _ = _history(workdir)
        history = ReconstructionHistory.model_validate(
            {**dict(history), "snapshots": history.snapshots[:1]}
        )
        auditor = ScriptedChatModel(responses=_audit_script(_accepted_audit()))
        result = _graph(
            workdir,
            coder=ScriptedChatModel(responses=()),
            auditor=auditor,
            final_selection=True,
            max_audit_reject_count=2,
        ).invoke({"reconstruction": history})

        assert result["reconstruction"].final_selection.round == 0
        assert (
            len(auditor.received_messages) == 2
        )  # No choice to spend another agent call on.


def test_terminal_coding_submission_is_followed_by_selection(tmp_path, monkeypatch):
    with SandboxWorkdir(host_bind_dir=tmp_path) as workdir:
        completed, _ = _history(workdir)
        initial = ReconstructionHistory.model_validate(
            {**dict(completed), "snapshots": completed.snapshots[:1]}
        )
        pending = open_next_round(initial, _report())
        snapshot = completed.snapshots[1]
        report = replace(
            snapshot.verification,
            exec_report=replace(
                snapshot.verification.exec_report, source=snapshot.program_source
            ),
        )
        _stub_verification(monkeypatch, report)
        coder = ScriptedChatModel(responses=_coding_script("ticket_001_shape_mismatch"))
        choice = FinalSelection(round=0, reason="The refinement lost an opening.")
        auditor = ScriptedChatModel(responses=(AIMessage(choice.model_dump_json()),))

        result = _graph(
            workdir,
            coder=coder,
            auditor=auditor,
            final_selection=True,
            max_audit_reject_count=1,
        ).invoke({"reconstruction": pending})

        assert result["reconstruction"].snapshots[-1].round == 1
        assert result["reconstruction"].final_selection == choice
        assert len(auditor.received_messages) == 1
