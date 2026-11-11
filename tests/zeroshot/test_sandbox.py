import shlex
import socket
import sys
import tempfile
from pathlib import Path, PurePosixPath

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


def test_read_only_subdir_is_readable_but_not_writable_in_sandbox(
    tmp_path: Path,
    sandbox_runner: SandboxRunner,
) -> None:
    work_dir = tmp_path / "work"
    protected_dir = work_dir / "attempts"
    protected_dir.mkdir(parents=True)
    protected_file = protected_dir / "model.py"
    protected_file.write_text("original", encoding="utf-8")

    with SandboxWorkdir(
        host_bind_dir=work_dir,
        read_only_subdirs=[PurePosixPath("attempts")],
    ) as sandbox_workdir:
        read_result = sandbox_runner.run(
            command="cat attempts/model.py",
            workdir=sandbox_workdir,
        )
        overwrite_result = sandbox_runner.run(
            command="printf changed > attempts/model.py",
            workdir=sandbox_workdir,
        )
        create_result = sandbox_runner.run(
            command="touch attempts/new.txt",
            workdir=sandbox_workdir,
            read_only_files=["/work/attempts/model.py"],
        )
        normal_write_result = sandbox_runner.run(
            command="touch writable.txt",
            workdir=sandbox_workdir,
        )

    assert read_result.returncode == 0
    assert read_result.stdout == "original"
    assert overwrite_result.returncode != 0
    assert create_result.returncode != 0
    assert normal_write_result.returncode == 0
    assert protected_file.read_text(encoding="utf-8") == "original"
    assert not (protected_dir / "new.txt").exists()
    assert (work_dir / "writable.txt").is_file()


@pytest.mark.parametrize(
    "file", ["front.png", "views/nested/front.png", "/tmp/front.png"]
)
def test_read_only_files_apply_to_one_call_without_blocking_scratch(
    tmp_path, sandbox_runner, file
):
    workdir = SandboxWorkdir(tmp_path)
    path = workdir.sandbox_to_host_path(file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("registered pixels")
    for command in (
        f"printf changed > {file}",
        f"cp /dev/null {file}",
        f"touch replacement; mv replacement {file}",
        f"rm {file}",
        *(
            f"mv {parent.relative_to(tmp_path)} moved"
            for parent in path.parents
            if parent != tmp_path and parent.is_relative_to(tmp_path)
        ),
    ):
        result = sandbox_runner.run(command, workdir, read_only_files=[file])
        assert result.returncode != 0, command
        assert path.read_text() == "registered pixels"
    # /tmp and /work/tmp name the same file.
    if file.startswith("/tmp/"):
        result = sandbox_runner.run(
            "printf changed > /work/tmp/front.png", workdir, read_only_files=[file]
        )
        assert result.returncode != 0
    result = sandbox_runner.run(
        f"cat {file}; printf zoom > {file}.zoom", workdir, read_only_files=[file]
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "registered pixels"
    assert path.with_suffix(".png.zoom").read_text() == "zoom"
    # The next coder call can still revise the same path.
    result = sandbox_runner.run(f"printf revised > {file}", workdir)
    assert result.returncode == 0, result.stderr
    assert path.read_text() == "revised"


@pytest.mark.parametrize("alias", ["symlink", "parent_symlink", "hardlink", "missing"])
def test_read_only_files_reject_missing_files_and_writable_aliases(
    tmp_path, sandbox_runner, alias
):
    workdir = SandboxWorkdir(tmp_path)
    source = tmp_path / "front.png"
    source.write_text("pixels")
    target = tmp_path / "alias.png"
    if alias == "symlink":
        target.symlink_to(source)
    elif alias == "parent_symlink":
        (tmp_path / "views").symlink_to(tmp_path, target_is_directory=True)
        target = tmp_path / "views/front.png"
    elif alias == "hardlink":
        target.hardlink_to(source)
    with pytest.raises(ValueError, match="read-only file"):
        sandbox_runner.run(
            "touch command_ran",
            workdir,
            read_only_files=[workdir.host_to_sandbox_path(target)],
        )
    assert not (tmp_path / "command_ran").exists()


@pytest.mark.parametrize(
    "subdir",
    [
        PurePosixPath("."),
        PurePosixPath(".."),
        PurePosixPath("nested/attempts"),
        PurePosixPath("/work/attempts"),
    ],
)
def test_runner_rejects_non_basename_read_only_subdir(
    tmp_path: Path,
    sandbox_runner: SandboxRunner,
    subdir: PurePosixPath,
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    workdir = SandboxWorkdir(
        host_bind_dir=work_dir,
        read_only_subdirs=[subdir],
    )

    with pytest.raises(ValueError, match="read_only_subdirs"):
        sandbox_runner.run(command="true", workdir=workdir)


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


def test_sandbox_replaces_non_utf8_output(
    tmp_path: Path,
    sandbox_runner: SandboxRunner,
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    with SandboxWorkdir(host_bind_dir=work_dir) as sandbox_workdir:
        result = sandbox_runner.run(
            command=_python_command("import os; os.write(1, b'before\\xffafter')"),
            workdir=sandbox_workdir,
        )

    assert result.status is SandboxStatus.COMPLETED
    assert result.returncode == 0
    assert result.stdout == "before\ufffdafter"


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


def test_tmp_is_writable_and_survives_between_runs(
    sandbox_runner: SandboxRunner,
) -> None:
    with SandboxWorkdir() as workdir:
        written = sandbox_runner.run(
            _python_command("open('/tmp/scratch.txt', 'w').write('kept')"),
            workdir,
        )
        read_back = sandbox_runner.run("cat /tmp/scratch.txt", workdir)

    assert written.returncode == 0, written.stderr
    assert read_back.stdout == "kept"


def test_tempfile_works_inside_sandbox(sandbox_runner: SandboxRunner) -> None:
    result = sandbox_runner.run(
        _python_command(
            "import tempfile\n"
            "with tempfile.NamedTemporaryFile() as handle:\n"
            "    print(handle.name)"
        ),
        workdir=SandboxWorkdir(),
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("/tmp/")


def test_tmp_and_work_name_the_same_file(tmp_path: Path) -> None:
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)

    assert workdir.sandbox_to_host_path(
        "/tmp/crop.png"
    ) == workdir.sandbox_to_host_path("/work/tmp/crop.png")


def test_tmp_path_cannot_escape_the_work_dir(tmp_path: Path) -> None:
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)

    with pytest.raises(ValueError):
        workdir.sandbox_to_host_path("/tmp/../../etc/passwd")


def test_a_bind_dir_written_with_dotdot_maps_paths_written_either_way(
    tmp_path: Path,
) -> None:
    (tmp_path / "workspace").mkdir()
    (tmp_path / "other").mkdir()
    workdir = SandboxWorkdir(host_bind_dir=tmp_path / "other" / ".." / "workspace")
    with tempfile.NamedTemporaryFile(dir=workdir.host_bind_dir / "tmp") as saved:
        assert workdir.host_to_sandbox_path(saved.name).parent == PurePosixPath(
            "/work/tmp"
        )
    unnormalized = tmp_path / "other" / ".." / "workspace" / "model.py"
    assert workdir.host_to_sandbox_path(unnormalized) == PurePosixPath("/work/model.py")
    assert workdir.sandbox_to_host_path("/work/model.py") == (
        tmp_path / "workspace" / "model.py"
    )


def test_a_host_path_cannot_escape_the_work_dir(tmp_path: Path) -> None:
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)

    with pytest.raises(ValueError, match="must be inside"):
        workdir.host_to_sandbox_path(tmp_path / ".." / "etc" / "passwd")
