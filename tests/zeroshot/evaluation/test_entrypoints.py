"""The CLI and the ``run_pipeline`` seam that reaches the scorer."""

import json
import shutil
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


def _targets(tmp_path: Path, run_dir: Path, target: Path) -> Path:
    """Mark ``run_dir`` as a pipeline sample and file its target by sample id."""

    (run_dir / "events.jsonl").touch()
    targets = tmp_path / "targets"
    targets.mkdir(exist_ok=True)
    shutil.copyfile(target, targets / f"{run_dir.name}.step")
    return targets


def test_cli_defaults_to_the_last_attempt_that_built(
    make_run_dir, solids: dict[str, Path], tmp_path: Path
) -> None:
    """A run whose final attempt failed still reached a solid on the way.

    Scoring only the final attempt reports nothing for such a run, which reads
    afterwards as a sample the pipeline could not do at all.
    """

    run_dir = make_run_dir({"000": solids["box"], "001": None})
    targets = _targets(tmp_path, run_dir, solids["box"])
    result = _cli("--run-dir", run_dir, "--target-dir", targets)

    assert result.returncode == 0, result.stderr
    document = json.loads((run_dir / "score.json").read_text())
    assert document["status"] == "OK"
    assert document["last_only"] is False


def test_cli_can_score_the_submitted_attempt_alone(
    make_run_dir, solids: dict[str, Path], tmp_path: Path
) -> None:
    run_dir = make_run_dir({"000": solids["box"], "001": None})
    targets = _targets(tmp_path, run_dir, solids["box"])
    result = _cli("--run-dir", run_dir, "--target-dir", targets, "--last-only")

    assert result.returncode == 0, result.stderr
    assert json.loads((run_dir / "score.json").read_text())["status"] == "NO_PREDICTION"


def test_cli_skips_non_samples_and_samples_without_a_target(
    make_run_dir, solids: dict[str, Path], tmp_path: Path
) -> None:
    run_dir = make_run_dir({"000": solids["box"]})
    targets = _targets(tmp_path, run_dir, solids["box"])
    not_a_sample = tmp_path / "logs"
    untargeted = tmp_path / "017"
    for directory in (not_a_sample, untargeted):
        directory.mkdir()
    (untargeted / "events.jsonl").touch()
    result = _cli(
        "--run-dir", run_dir, not_a_sample, untargeted, "--target-dir", targets
    )

    assert result.returncode == 0, result.stderr
    assert json.loads((run_dir / "score.json").read_text())["status"] == "OK"
    assert not (not_a_sample / "score.json").exists()
    assert not (untargeted / "score.json").exists()
    assert sum(line.startswith("skip ") for line in result.stdout.splitlines()) == 2


def test_cli_rejects_a_missing_target_dir(make_run_dir, tmp_path: Path) -> None:
    result = _cli("--run-dir", make_run_dir({}), "--target-dir", tmp_path / "absent")
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
    assert document["metrics"]["mesh_iou"] == 1.0
    assert document["build_valid"] is True


def test_a_null_target_skips_scoring(tmp_path: Path) -> None:
    config = _config(tmp_path, **{"sample.target_step_path": "null"})
    assert config.sample.target_step_path is None
