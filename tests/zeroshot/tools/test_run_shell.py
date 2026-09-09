import shlex
import sys
from pathlib import Path

import pytest

from zeroshot.pipeline.sandbox import (
    SandboxResult,
    SandboxRunner,
    SandboxStatus,
    SandboxWorkdir,
)
from zeroshot.pipeline.tools.run_shell import create_run_shell_tool


class StubSandboxRunner:
    def __init__(self, result: SandboxResult) -> None:
        self.result = result
        self.calls: list[tuple[str, SandboxWorkdir]] = []

    def run(
        self,
        command: str,
        workdir: SandboxWorkdir,
        timeout_s: float | None = None,
    ) -> SandboxResult:
        del timeout_s
        self.calls.append((command, workdir))
        return self.result


@pytest.fixture
def sandbox_runner() -> SandboxRunner:
    return SandboxRunner(Path(sys.executable), default_timeout_s=10)


def test_tool_schema_exposes_only_command(sandbox_runner: SandboxRunner):
    with SandboxWorkdir() as workdir:
        run_shell = create_run_shell_tool(sandbox_runner, workdir)

        assert run_shell.name == "run_shell"

        schema = run_shell.get_input_jsonschema()
        # schema = {
        #     "properties": {"command": {"title": "Command", "type": "string"}},
        #     "required": ["command"],
        #     "title": "run_shell",
        #     "type": "object"
        # }

        assert set(schema["properties"]) == {"command"}
        assert schema["required"] == ["command"]
        assert schema["properties"]["command"]["type"] == "string"


def test_tool_preserves_files_between_calls(sandbox_runner: SandboxRunner):
    with SandboxWorkdir() as workdir:
        run_shell = create_run_shell_tool(sandbox_runner, workdir)

        source = """\
from pathlib import Path
value = int(Path('value.txt').read_text())
print(value + 1)
"""

        first = run_shell.invoke({"command": "printf 41 > value.txt"})
        second = run_shell.invoke({"command": f"python -c {shlex.quote(source)}"})

    assert first["status"] == "COMPLETED"
    assert first["returncode"] == 0
    assert second["status"] == "COMPLETED"
    assert second["returncode"] == 0
    assert second["stdout"] == "42\n"


@pytest.mark.parametrize(
    ("sandbox_result", "expected"),
    [
        (
            SandboxResult(
                status=SandboxStatus.COMPLETED,
                returncode=7,
                stdout="diagnostic",
                stderr="command failed",
            ),
            {
                "status": "COMPLETED",
                "returncode": 7,
                "stdout": "diagnostic",
                "stderr": "command failed",
            },
        ),
        (
            SandboxResult(
                status=SandboxStatus.TIMEOUT,
                returncode=None,
                stdout="partial output",
                stderr="",
            ),
            {
                "status": "TIMEOUT",
                "returncode": None,
                "stdout": "partial output",
                "stderr": "",
            },
        ),
        (
            SandboxResult(
                status=SandboxStatus.INFRA_ERROR,
                returncode=None,
                stdout="",
                stderr="bwrap failed",
            ),
            {
                "status": "INFRA_ERROR",
                "returncode": None,
                "stdout": "",
                "stderr": "bwrap failed",
            },
        ),
    ],
    ids=["nonzero-exit", "timeout", "infra-error"],
)
def test_tool_returns_sandbox_result_as_mapping(
    tmp_path: Path,
    sandbox_result: SandboxResult,
    expected: dict[str, str | int | None],
) -> None:
    stub_runner = StubSandboxRunner(sandbox_result)

    with SandboxWorkdir(host_bind_dir=tmp_path) as workdir:
        run_shell = create_run_shell_tool(
            stub_runner,  # type: ignore[arg-type]
            workdir,
        )

        actual = run_shell.invoke({"command": "test command"})

        assert actual == expected
        assert stub_runner.calls == [("test command", workdir)]


@pytest.mark.parametrize("status", [SandboxStatus.COMPLETED, SandboxStatus.TIMEOUT])
def test_large_output_is_bounded_and_recoverable(
    tmp_path: Path, status: SandboxStatus
) -> None:
    stdout = "BEGIN\n" + "寸法 123\n" * 20_000 + "END\n"
    stderr = "Traceback\n" + "diagnostic\n" * 20_000 + "ValueError: bad shape\n"
    stub_runner = StubSandboxRunner(SandboxResult(status, 7, stdout, stderr))
    with SandboxWorkdir(host_bind_dir=tmp_path) as workdir:
        run_shell = create_run_shell_tool(stub_runner, workdir)  # type: ignore[arg-type]
        first = run_shell.invoke({"command": "verbose command"})
        second = run_shell.invoke({"command": "verbose command again"})

        assert first["status"] == status.value
        assert first["returncode"] == 7
        for stream, original in [("stdout", stdout), ("stderr", stderr)]:
            assert len(first[stream]) < 17_000
            assert first[stream].startswith(original[:100])
            assert first[stream].endswith(original[-100:])
            assert "Omitted" in first[stream]
            saved = list((tmp_path / "tmp").glob(f"shell_{stream}_*.txt"))
            assert len(saved) == 2
            assert all(p.read_text(encoding="utf-8") == original for p in saved)
            paths = [str(workdir.host_to_sandbox_path(p)) for p in saved]
            assert sum(path in first[stream] for path in paths) == 1
            assert sum(path in second[stream] for path in paths) == 1
            assert first[stream] != second[stream]
