import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from zeroshot import run_pipeline
from zeroshot.pipeline.messages.artifact import ArtifactPresenter
from zeroshot.pipeline.messages.manifest import InputManifest
from zeroshot.pipeline.stages.drawings.contracts import (
    DrawingSource,
    View,
    unread_sheet,
)
from zeroshot.pipeline.workflow import (
    ReconstructionState,
)
from zeroshot.pipeline.workflow.graph import create_reconstruction_graph


def _config(tmp_path: Path, dxf_path: Path, **overrides: Any) -> Any:
    values: dict[str, Any] = {
        "artifact_root": str(tmp_path / "artifacts"),
        "on_existing": "fail",
        "workflow": {
            "_target_": "zeroshot.pipeline.workflow.graph.create_reconstruction_graph",
            "_partial_": True,
            "max_audit_reject_count": 7,
        },
        "console": None,
        "artifact_presenter": {
            "_target_": "zeroshot.pipeline.messages.artifact.ArtifactPresenter",
            "input_mode": "path",
            "feedback_mode": "none",
        },
        "models": {},
        "model": {
            "_target_": (
                "langchain_core.language_models.fake_chat_models.FakeListChatModel"
            ),
            "responses": ["done"],
        },
        "sandbox_runner": {
            "python_executable": sys.executable,
            "default_timeout_s": 30.0,
            "max_stdout_bytes": 1000,
            "max_stderr_bytes": 2000,
        },
        "sample": {
            "sample_id": "sample-1",
            "drawing": {
                "sheets": [
                    {
                        "name": "sheet_drawing",
                        "role": "full_page",
                        "file": str(dxf_path),
                    }
                ]
            },
        },
    }
    values.update(overrides)
    return OmegaConf.create(values)


def test_run_composes_dependencies_and_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dxf_path = tmp_path / "input.dxf"
    dxf_path.write_text("DXF_FIXTURE", encoding="utf-8")
    artifact_root = tmp_path / "artifacts"
    captured: dict[str, Any] = {}

    class StubSandboxRunner:
        def __init__(self, **kwargs: Any) -> None:
            captured["sandbox_options"] = kwargs

    class StubPipelineRunner:
        def __init__(self, **kwargs: Any) -> None:
            captured["runner_options"] = kwargs

        def run_sample(self, manifest: InputManifest) -> ReconstructionState:
            captured["manifest"] = manifest
            return ReconstructionState()

    monkeypatch.setattr(run_pipeline, "SandboxRunner", StubSandboxRunner)
    monkeypatch.setattr(run_pipeline, "PipelineRunner", StubPipelineRunner)

    config = OmegaConf.create(
        {
            "artifact_root": str(artifact_root),
            "on_existing": "fail",
            "resume_from": str(tmp_path / "reconstruction.json"),
            "workflow": {
                "_target_": (
                    "zeroshot.pipeline.workflow.graph.create_reconstruction_graph"
                ),
                "_partial_": True,
                "max_audit_reject_count": 7,
            },
            "console": None,
            "artifact_presenter": {
                "_target_": "zeroshot.pipeline.messages.artifact.ArtifactPresenter",
                "input_mode": "path",
                "feedback_mode": "none",
            },
            "models": {},
            "model": {
                "_target_": (
                    "langchain_core.language_models.fake_chat_models.FakeListChatModel"
                ),
                "responses": ["done"],
            },
            "sandbox_runner": {
                "python_executable": sys.executable,
                "default_timeout_s": 30.0,
                "max_stdout_bytes": 1000,
                "max_stderr_bytes": 2000,
            },
            "sample": {
                "sample_id": "sample-1",
                "drawing": {
                    "sheets": [
                        {
                            "name": "sheet_drawing",
                            "role": "full_page",
                            "file": str(dxf_path),
                        }
                    ]
                },
            },
        }
    )

    result = run_pipeline.run(config)

    runner_options = captured["runner_options"]
    # No model reaches the runner: an agent carries its own, so the workflow
    # config is where `instantiate` finds one.
    assert "model" not in runner_options
    assert isinstance(runner_options["artifact_presenter"], ArtifactPresenter)
    assert isinstance(runner_options["sandbox_runner"], StubSandboxRunner)
    assert runner_options["artifact_root"] == artifact_root
    assert runner_options["console_reporter"] is None
    assert runner_options["resume_from"] == tmp_path / "reconstruction.json"
    graph_factory = runner_options["graph_factory"]
    assert graph_factory.func is create_reconstruction_graph
    assert graph_factory.keywords == {"max_audit_reject_count": 7}
    assert captured["sandbox_options"] == {
        "python_executable": Path(sys.executable),
        "default_timeout_s": 30.0,
        "max_stdout_bytes": 1000,
        "max_stderr_bytes": 2000,
    }
    assert captured["manifest"] == InputManifest(
        sample_id="sample-1",
        drawing=DrawingSource(
            sheets=[unread_sheet("sheet_drawing", View.FULL_PAGE, dxf_path)]
        ),
    )
    assert result is not None
    assert result == {}


def test_module_help_uses_hydra_entrypoint() -> None:
    repository_root = Path(__file__).parents[2]

    completed = subprocess.run(
        [sys.executable, "-m", "zeroshot.run_pipeline", "--help"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "run_pipeline is powered by Hydra" in completed.stdout
    assert "artifact_root:" in completed.stdout


@pytest.mark.parametrize("compaction", [None, False])
def test_a_shared_workflow_requires_configured_compaction(
    compaction: object,
) -> None:
    config = OmegaConf.create(
        {
            "workflow": {
                "share_thread": True,
                "compact_between_stages": compaction,
            }
        }
    )

    with pytest.raises(
        ValueError,
        match="share_thread=true requires.*compact_between_stages",
    ):
        run_pipeline._validate_workflow_config(config)


def test_a_skipped_sample_is_neither_recorded_nor_scored(
    tmp_path: Path, monkeypatch
) -> None:
    """`run` returning None means the sample already ran; nothing follows it."""
    called: list[str] = []
    monkeypatch.setattr(run_pipeline, "run", lambda config: None)
    monkeypatch.setattr(
        run_pipeline, "record_run", lambda *args: called.append("record")
    )
    monkeypatch.setattr(run_pipeline, "score", lambda config: called.append("score"))

    dxf_path = tmp_path / "input.dxf"
    dxf_path.write_text("DXF_FIXTURE", encoding="utf-8")
    config = _config(tmp_path, dxf_path)
    config.sample.target_step_path = str(tmp_path / "target.step")

    run_pipeline.main.__wrapped__(config)

    assert called == []


def test_a_run_that_raised_is_still_described_and_scored(
    tmp_path: Path, monkeypatch
) -> None:
    """What it produced before it raised is a result, not a hole.

    A sweep that leaves no `score.json` behind for a crashed sample cannot say
    afterwards whether the sample was hard or the pipeline broke.
    """
    called: list[str] = []

    def _raise(config) -> None:
        del config
        raise RuntimeError("stream finished without producing a message")

    monkeypatch.setattr(run_pipeline, "run", _raise)
    monkeypatch.setattr(
        run_pipeline, "record_run", lambda *args: called.append("record")
    )
    monkeypatch.setattr(run_pipeline, "score", lambda config: called.append("score"))

    dxf_path = tmp_path / "input.dxf"
    dxf_path.write_text("DXF_FIXTURE", encoding="utf-8")
    config = _config(tmp_path, dxf_path)
    config.sample.target_step_path = str(tmp_path / "target.step")
    (Path(config.artifact_root) / config.sample.sample_id).mkdir(parents=True)

    with pytest.raises(RuntimeError, match="stream finished"):
        run_pipeline.main.__wrapped__(config)

    assert called == ["record", "score"]


def test_a_run_is_recorded_before_it_is_scored(tmp_path: Path, monkeypatch) -> None:
    called: list[str] = []
    monkeypatch.setattr(run_pipeline, "run", lambda config: ReconstructionState())
    monkeypatch.setattr(
        run_pipeline, "record_run", lambda *args: called.append("record")
    )
    monkeypatch.setattr(run_pipeline, "score", lambda config: called.append("score"))

    dxf_path = tmp_path / "input.dxf"
    dxf_path.write_text("DXF_FIXTURE", encoding="utf-8")
    config = _config(tmp_path, dxf_path)
    config.sample.target_step_path = str(tmp_path / "target.step")

    run_pipeline.main.__wrapped__(config)

    assert called == ["record", "score"]
