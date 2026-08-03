import subprocess
import sys
from pathlib import Path
from typing import Any

from hydra.utils import instantiate
from langchain_core.language_models import BaseChatModel
from omegaconf import OmegaConf

from zeroshot import run_pipeline
from zeroshot.pipeline.manifest import InputManifest
from zeroshot.pipeline.messages import MessageBuilder
from zeroshot.pipeline.tools import VerifyOutputResult
from zeroshot.pipeline.verification import StepRenderer
from zeroshot.pipeline.workflow import ReconstructionState


def _unreachable_runner(**kwargs: Any) -> Any:
    raise AssertionError("the run must fail before a PipelineRunner is built")


def _config(tmp_path: Path, dxf_path: Path, **overrides: Any) -> Any:
    values: dict[str, Any] = {
        "artifact_root": str(tmp_path / "artifacts"),
        "console": None,
        "renderer": {
            "_target_": "zeroshot.pipeline.verification.StepRenderer",
            "timeout_s": 42.0,
        },
        "message_builder": {
            "_target_": "zeroshot.pipeline.messages.MessageBuilder",
            "access_render3d": "none",
            "access_render3d_styles": [],
            "feedback_render3d": "none",
            "feedback_render3d_styles": [],
        },
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
            "dxf_path": str(dxf_path),
            "render3d_paths": {},
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
            return ReconstructionState(
                messages=[],
                last_verification=VerifyOutputResult(status="VERIFIED"),
            )

    monkeypatch.setattr(run_pipeline, "SandboxRunner", StubSandboxRunner)
    monkeypatch.setattr(run_pipeline, "PipelineRunner", StubPipelineRunner)

    config = OmegaConf.create(
        {
            "artifact_root": str(artifact_root),
            "console": None,
            "message_builder": {
                "_target_": "zeroshot.pipeline.messages.MessageBuilder",
                "access_render3d": "none",
                "access_render3d_styles": [],
                "feedback_render3d": "none",
                "feedback_render3d_styles": [],
            },
            "model": {
                "_target_": (
                    "langchain_core.language_models.fake_chat_models.FakeListChatModel"
                ),
                "responses": ["done"],
            },
            "renderer": {
                "_target_": "zeroshot.pipeline.verification.StepRenderer",
                "timeout_s": 42.0,
            },
            "sandbox_runner": {
                "python_executable": sys.executable,
                "default_timeout_s": 30.0,
                "max_stdout_bytes": 1000,
                "max_stderr_bytes": 2000,
            },
            "sample": {
                "sample_id": "sample-1",
                "dxf_path": str(dxf_path),
                "render3d_paths": {},
            },
        }
    )

    result = run_pipeline.run(config)

    runner_options = captured["runner_options"]
    assert isinstance(runner_options["model"], BaseChatModel)
    assert isinstance(runner_options["message_builder"], MessageBuilder)
    assert isinstance(runner_options["sandbox_runner"], StubSandboxRunner)
    assert runner_options["artifact_root"] == artifact_root
    assert runner_options["console_reporter"] is None
    assert isinstance(runner_options["renderer"], StepRenderer)
    assert runner_options["renderer"].timeout_s == 42.0
    assert captured["sandbox_options"] == {
        "python_executable": Path(sys.executable),
        "default_timeout_s": 30.0,
        "max_stdout_bytes": 1000,
        "max_stderr_bytes": 2000,
    }
    assert captured["manifest"] == InputManifest(
        sample_id="sample-1",
        dxf_path=dxf_path,
        render3d_paths={},
    )
    assert result["last_verification"].status == "VERIFIED"


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


def test_shipped_config_actually_builds_a_renderer() -> None:
    """The renderer is optional in code, so only the config decides whether
    visual feedback happens at all.  A default that quietly omits it renders
    nothing while every test still passes."""
    config = OmegaConf.load(Path(run_pipeline.__file__).parent / "configs/default.yaml")

    assert config.get("renderer") is not None
    renderer = instantiate(config.renderer)
    assert isinstance(renderer, StepRenderer)
    assert renderer.timeout_s > 0


def test_null_renderer_falls_back_to_the_default_renderer(
    tmp_path: Path, monkeypatch
) -> None:
    """Rendering is not the visual-feedback switch, so it cannot be turned off.

    Hydra yields None for `renderer: null`; that must mean the default
    renderer, never a None that only surfaces once a candidate first verifies.
    """
    captured: dict[str, Any] = {}

    class StubPipelineRunner:
        def __init__(self, **kwargs: Any) -> None:
            captured["runner_options"] = kwargs

        def run_sample(self, manifest: InputManifest) -> ReconstructionState:
            return ReconstructionState(
                messages=[],
                last_verification=VerifyOutputResult(status="VERIFIED"),
            )

    monkeypatch.setattr(run_pipeline, "PipelineRunner", StubPipelineRunner)
    dxf_path = tmp_path / "input.dxf"
    dxf_path.write_text("DXF_FIXTURE", encoding="utf-8")

    run_pipeline.run(_config(tmp_path, dxf_path, renderer=None))

    renderer = captured["runner_options"]["renderer"]
    assert isinstance(renderer, StepRenderer)
    assert renderer.timeout_s == StepRenderer().timeout_s
