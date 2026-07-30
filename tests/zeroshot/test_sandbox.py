import shlex
import socket
import sys
from pathlib import Path

import pytest

from zeroshot.pipeline.sandbox import (
    SandboxRunner,
    SandboxStatus,
    SandboxWorkdir,
)


def _python_command(source: str) -> str:
    return f"python -c {shlex.quote(source)}"


@pytest.fixture
def sandbox_runner() -> SandboxRunner:
    return SandboxRunner(
        python_executable=Path(sys.executable),
        default_timeout_s=2.0,
    )


def test_owned_workdir_is_created_and_removed_after_context() -> None:
    with SandboxWorkdir() as workdir:
        host_bind_dir = workdir.host_bind_dir

        assert host_bind_dir.is_dir()
        assert workdir.sandbox_bind_dir.as_posix() == "/work"

    assert not host_bind_dir.exists()


def test_borrowed_workdir_is_not_removed_after_context(tmp_path: Path) -> None:
    host_bind_dir = tmp_path / "work"
    host_bind_dir.mkdir()

    with SandboxWorkdir(host_bind_dir=host_bind_dir) as workdir:
        assert workdir.host_bind_dir == host_bind_dir

    assert host_bind_dir.is_dir()


def test_borrowed_workdir_must_exist(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="host_bind_dir"):
        SandboxWorkdir(host_bind_dir=missing_dir)


def test_runner_uses_files_staged_directly_in_owned_workdir(
    sandbox_runner: SandboxRunner,
) -> None:
    with SandboxWorkdir() as workdir:
        (workdir.host_bind_dir / "input.txt").write_text("staged", encoding="utf-8")

        result = sandbox_runner.run(
            command="tr a-z A-Z < input.txt > output.txt",
            workdir=workdir,
        )

        assert result.status is SandboxStatus.COMPLETED
        assert result.returncode == 0
        assert (workdir.host_bind_dir / "output.txt").read_text(
            encoding="utf-8"
        ) == "STAGED"


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

    with SandboxWorkdir(host_bind_dir=work_dir) as sandbox_workdir:
        result = sandbox_runner.run(
            command=_python_command(source),
            workdir=sandbox_workdir,
        )

    assert result.status is SandboxStatus.COMPLETED
    assert result.returncode == 0
    assert result.stdout == "allowed\n"
    assert (work_dir / "output.txt").read_text(encoding="utf-8") == "ALLOWED"


def test_sandbox_preserves_files_between_runs(
    tmp_path: Path,
    sandbox_runner: SandboxRunner,
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    with SandboxWorkdir(host_bind_dir=work_dir) as sandbox_workdir:
        first_result = sandbox_runner.run(
            command="printf first > state.txt",
            workdir=sandbox_workdir,
        )
        second_result = sandbox_runner.run(
            command="cat state.txt",
            workdir=sandbox_workdir,
        )

    assert first_result.status is SandboxStatus.COMPLETED
    assert first_result.returncode == 0
    assert second_result.status is SandboxStatus.COMPLETED
    assert second_result.returncode == 0
    assert second_result.stdout == "first"


def test_sandbox_workdirs_are_isolated(
    tmp_path: Path,
    sandbox_runner: SandboxRunner,
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()

    with (
        SandboxWorkdir(host_bind_dir=first_dir) as first_workdir,
        SandboxWorkdir(host_bind_dir=second_dir) as second_workdir,
    ):
        write_result = sandbox_runner.run(
            command="printf secret > state.txt",
            workdir=first_workdir,
        )
        read_result = sandbox_runner.run(
            command="cat state.txt",
            workdir=second_workdir,
        )

    assert write_result.status is SandboxStatus.COMPLETED
    assert write_result.returncode == 0
    assert read_result.status is SandboxStatus.COMPLETED
    assert read_result.returncode != 0
    assert "secret" not in read_result.stdout


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
        with SandboxWorkdir(host_bind_dir=work_dir) as sandbox_workdir:
            result = sandbox_runner.run(
                command=_python_command(source),
                workdir=sandbox_workdir,
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

    with SandboxWorkdir(host_bind_dir=work_dir) as sandbox_workdir:
        result = sandbox_runner.run(
            command=_python_command(source),
            workdir=sandbox_workdir,
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

    with SandboxWorkdir(host_bind_dir=work_dir) as sandbox_workdir:
        result = sandbox_runner.run(
            command=_python_command(source),
            workdir=sandbox_workdir,
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

        with SandboxWorkdir(host_bind_dir=work_dir) as sandbox_workdir:
            result = sandbox_runner.run(
                command=_python_command(source),
                workdir=sandbox_workdir,
            )

    assert result.status is SandboxStatus.COMPLETED
    assert result.returncode != 0


def test_sandbox_times_out_infinite_loop(
    tmp_path: Path,
    sandbox_runner: SandboxRunner,
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    with SandboxWorkdir(host_bind_dir=work_dir) as sandbox_workdir:
        result = sandbox_runner.run(
            command=_python_command("while True:\n    pass\n"),
            workdir=sandbox_workdir,
            timeout_s=0.1,
        )

    assert result.status is SandboxStatus.TIMEOUT
    assert result.returncode is None
