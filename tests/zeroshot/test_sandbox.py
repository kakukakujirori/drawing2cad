import shlex
import socket
import sys
from pathlib import Path

import pytest

from zeroshot.pipeline.sandbox import SandboxRunner, SandboxStatus


def _python_command(source: str) -> str:
    return f"python -c {shlex.quote(source)}"


@pytest.fixture
def sandbox_runner() -> SandboxRunner:
    return SandboxRunner(
        python_executable=Path(sys.executable),
        default_timeout_s=2.0,
    )


def test_sandbox_can_read_and_write_only_inside_work_dir(
    tmp_path: Path,
    sandbox_runner: SandboxRunner,
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "input.txt").write_text("allowed", encoding="utf-8")
    source = (
        "from pathlib import Path\n"
        "text = Path('input.txt').read_text()\n"
        "Path('output.txt').write_text(text.upper())\n"
        "print(text)\n"
    )

    result = sandbox_runner.run(
        command=_python_command(source),
        work_dir=work_dir,
    )

    assert result.status is SandboxStatus.COMPLETED
    assert result.returncode == 0
    assert result.stdout == "allowed\n"
    assert (work_dir / "output.txt").read_text(encoding="utf-8") == "ALLOWED"


def test_sandbox_cannot_read_gt_or_repository_outside_work_dir(
    tmp_path: Path,
    sandbox_runner: SandboxRunner,
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    fake_gt = tmp_path / "gt.step"
    fake_gt.write_text("secret GT", encoding="utf-8")
    repository_file = Path(__file__).resolve().parents[2] / "implementation_plan.md"

    for forbidden_path in (fake_gt, repository_file):
        source = (
            "from pathlib import Path\n"
            f"print(Path({str(forbidden_path)!r}).read_text())\n"
        )
        result = sandbox_runner.run(
            command=_python_command(source),
            work_dir=work_dir,
        )

        assert result.status is SandboxStatus.COMPLETED
        assert result.returncode != 0
        assert "secret GT" not in result.stdout


def test_sandbox_cannot_import_module_outside_work_dir(
    tmp_path: Path,
    sandbox_runner: SandboxRunner,
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    hidden_module = tmp_path / "hidden_module.py"
    hidden_module.write_text("VALUE = 'secret'\n", encoding="utf-8")
    source = (
        "import importlib.util\n"
        "spec = importlib.util.spec_from_file_location("
        f"'hidden_module', {str(hidden_module)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "print(module.VALUE)\n"
    )

    result = sandbox_runner.run(
        command=_python_command(source),
        work_dir=work_dir,
    )

    assert result.status is SandboxStatus.COMPLETED
    assert result.returncode != 0
    assert "secret" not in result.stdout


def test_child_process_cannot_escape_sandbox_filesystem(
    tmp_path: Path,
    sandbox_runner: SandboxRunner,
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    secret_file = tmp_path / "secret.txt"
    secret_file.write_text("host secret", encoding="utf-8")
    source = (
        "import subprocess\n"
        f"subprocess.run(['cat', {str(secret_file)!r}], check=True)\n"
    )

    result = sandbox_runner.run(
        command=_python_command(source),
        work_dir=work_dir,
    )

    assert result.status is SandboxStatus.COMPLETED
    assert result.returncode != 0
    assert "host secret" not in result.stdout


def test_sandbox_cannot_access_host_network(
    tmp_path: Path,
    sandbox_runner: SandboxRunner,
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    with socket.socket() as host_server:
        host_server.bind(("127.0.0.1", 0))
        host_server.listen(1)
        port = host_server.getsockname()[1]
        source = (
            "import socket\n"
            "socket.create_connection("
            f"('127.0.0.1', {port}), timeout=0.5"
            ")\n"
        )

        result = sandbox_runner.run(
            command=_python_command(source),
            work_dir=work_dir,
        )

    assert result.status is SandboxStatus.COMPLETED
    assert result.returncode != 0


def test_sandbox_times_out_infinite_loop(
    tmp_path: Path,
    sandbox_runner: SandboxRunner,
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    result = sandbox_runner.run(
        command=_python_command("while True:\n    pass\n"),
        work_dir=work_dir,
        timeout_s=0.1,
    )

    assert result.status is SandboxStatus.TIMEOUT
    assert result.returncode is None
