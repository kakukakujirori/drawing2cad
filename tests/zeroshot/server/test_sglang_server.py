import subprocess
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from zeroshot.server import SGLangServer

CONFIG_DIR = Path(__file__).parents[3] / "zeroshot" / "server" / "configs"


def test_qwen_profile_builds_sglang_command() -> None:
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR.resolve()),
        version_base="1.3",
    ):
        config = compose(config_name="sglang")

    server = instantiate(config.server)

    assert isinstance(server, SGLangServer)
    assert server.command() == [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        "Qwen/Qwen3.6-35B-A3B-FP8",
        "--host",
        "127.0.0.1",
        "--port",
        "30000",
        "--tp",
        "2",
        "--mem-fraction-static",
        "0.85",
        "--reasoning-parser",
        "qwen3",
        "--tool-call-parser",
        "qwen3_coder",
        "--max-running-requests",
        "32",
        "--disable-cuda-graph",
    ]


def test_optional_flags_and_extra_args() -> None:
    server = SGLangServer(
        model_path="local/model",
        context_length=None,
        mem_fraction_static=None,
        trust_remote_code=True,
        extra_args=("--attention-backend", "triton"),
    )

    assert server.command()[-3:] == [
        "--trust-remote-code",
        "--attention-backend",
        "triton",
    ]


def test_serve_model_dry_run() -> None:
    repository_root = Path(__file__).parents[3]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "zeroshot.server.serve_model",
            "dry_run=true",
        ],
        cwd=repository_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "sglang.launch_server" in completed.stdout
    assert "--tool-call-parser qwen3_coder" in completed.stdout
