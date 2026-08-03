"""The CLI and the ``run_pipeline`` seam that reaches the scorer."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from zeroshot import run_pipeline

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "zeroshot" / "configs"


def _cli(*args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "zeroshot.evaluation.run_scoring", *map(str, args)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_defaults_to_the_submitted_attempt(
    make_run_dir, solids: dict[str, Path], tmp_path: Path
) -> None:
    """``action="store_true"`` defaults to False, inverting ``last_only``.

    That scored a solid the run had already replaced, which can only flatter
    the model.
    """

    run_dir = make_run_dir({"000": solids["box"], "001": None})
    output = tmp_path / "score.json"
    result = _cli(
        "--run-dir", run_dir, "--target-step", solids["box"], "--output", output
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["status"] == "NO_PREDICTION"


def test_cli_can_fall_back_to_an_earlier_attempt(
    make_run_dir, solids: dict[str, Path], tmp_path: Path
) -> None:
    run_dir = make_run_dir({"000": solids["box"], "001": None})
    output = tmp_path / "score.json"
    result = _cli(
        "--run-dir",
        run_dir,
        "--target-step",
        solids["box"],
        "--no-last-only",
        "--output",
        output,
    )

    assert result.returncode == 0, result.stderr
    document = json.loads(output.read_text())
    assert document["status"] == "OK"
    assert document["last_only"] is False


def test_cli_rejects_a_missing_target(make_run_dir, tmp_path: Path) -> None:
    result = _cli(
        "--run-dir", make_run_dir({}), "--target-step", tmp_path / "absent.step"
    )
    assert result.returncode == 2


def _config(tmp_path: Path, **overrides: object):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(
            config_name="default",
            overrides=[
                f"artifact_root={tmp_path}",
                *(f"{k}={v}" for k, v in overrides.items()),
            ],
        )


@pytest.fixture
def sample_run(make_run_dir, solids: dict[str, Path], tmp_path: Path) -> Path:
    """A finished run laid out where ``run_pipeline`` expects to find it."""

    run_dir = make_run_dir({"000": solids["box"]})
    placed = tmp_path / "000364"
    run_dir.rename(placed)
    return placed


def test_score_writes_the_report_into_the_run(
    sample_run: Path, solids: dict[str, Path], tmp_path: Path
) -> None:
    config = _config(tmp_path, **{"sample.target_step_path": solids["box"]})
    run_pipeline.score(config)

    document = json.loads((sample_run / "score.json").read_text())
    assert document["status"] == "OK"
    assert document["metrics"]["voxel_iou"] == 1.0


def test_a_null_target_skips_scoring(tmp_path: Path) -> None:
    config = _config(tmp_path, **{"sample.target_step_path": "null"})
    assert config.sample.target_step_path is None
