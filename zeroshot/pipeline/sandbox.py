import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath


class SandboxStatus(Enum):
    COMPLETED = "COMPLETED"
    TIMEOUT = "TIMEOUT"
    INFRA_ERROR = "INFRA_ERROR"


@dataclass(frozen=True)
class SandboxResult:
    status: SandboxStatus
    returncode: int | None
    stdout: str
    stderr: str


class SandboxWorkdir:
    sandbox_bind_dir = PurePosixPath("/work")

    def __init__(self, host_bind_dir: Path | None = None):
        if host_bind_dir is None:
            self._tmpdir_context = tempfile.TemporaryDirectory(
                prefix="drawing2cad-sandbox-"
            )
            self.host_bind_dir = Path(self._tmpdir_context.name)
        else:
            self.host_bind_dir = host_bind_dir

        if not self.host_bind_dir.is_dir():
            raise FileNotFoundError(f"{self.host_bind_dir=} not found")

    def __enter__(self) -> "SandboxWorkdir":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if hasattr(self, "_tmpdir_context"):
            self._tmpdir_context.cleanup()


class SandboxRunner:
    def __init__(
        self,
        python_executable: Path,
        default_timeout_s: float,
        max_stdout_bytes: int = 1024 * 1024,
        max_stderr_bytes: int = 1024 * 1024,
    ) -> None:
        self.python_executable = python_executable
        self.default_timeout_s = default_timeout_s
        self.max_stdout_bytes = max_stdout_bytes
        self.max_stderr_bytes = max_stderr_bytes

        # input check
        try:
            subprocess.run(
                [python_executable, "--version"], check=True, capture_output=True
            )
        except (OSError, subprocess.CalledProcessError) as e:
            raise ValueError(f"Failed to run python executable: {e}")

        if default_timeout_s <= 0:
            raise ValueError(
                f"default_timeout_s must be positive: given {default_timeout_s}"
            )

        # bwrap availability check
        if shutil.which("bwrap") is None:
            raise RuntimeError(
                "bwrap is not found. Please install bwrap to use this sandbox."
            )

    def run(
        self,
        command: str,
        workdir: SandboxWorkdir,
        timeout_s: float | None = None,
    ) -> SandboxResult:

        # sanity check
        host_workdir = workdir.host_bind_dir.resolve()
        sandbox_workdir = workdir.sandbox_bind_dir
        py_path = self.python_executable.resolve()
        py_root_dir = py_path.parent.parent  # should include site-packages

        # fmt: off
        bwrap_command = [
            "bwrap",
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
            "--ro-bind", "/usr", "/usr",
            "--ro-bind", "/lib", "/lib",
            "--ro-bind", "/lib64", "/lib64",
            "--ro-bind", "/bin", "/bin",
            "--proc", "/proc",
            "--dev", "/dev",
            "--dir", "/cad-env",
            "--ro-bind", str(py_root_dir), "/cad-env",
            "--clearenv",
            "--setenv", "PATH", "/cad-env/bin:/usr/bin:/bin",
            "--setenv", "HOME", str(sandbox_workdir),
            "--setenv", "LANG", "C.UTF-8",
            "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
            "--bind", str(host_workdir), str(sandbox_workdir),
            "--chdir", str(sandbox_workdir),
            "--", "/bin/bash", "-c", command,
        ]
        # fmt: on

        try:
            ret = subprocess.run(
                bwrap_command,
                shell=False,
                capture_output=True,
                check=False,
                text=True,
                timeout=timeout_s or self.default_timeout_s,
            )
            stdout = ret.stdout[: self.max_stdout_bytes] if ret.stdout else ""
            stderr = ret.stderr[: self.max_stderr_bytes] if ret.stderr else ""
            return SandboxResult(
                status=SandboxStatus.COMPLETED,
                returncode=ret.returncode,
                stdout=stdout,
                stderr=stderr,
            )
        except subprocess.TimeoutExpired as e:
            stdout = _to_str(e.stdout)[: self.max_stdout_bytes]
            stderr = _to_str(e.stderr)[: self.max_stderr_bytes]
            return SandboxResult(
                status=SandboxStatus.TIMEOUT,
                returncode=None,
                stdout=stdout,
                stderr=stderr,
            )
        except OSError as e:
            return SandboxResult(
                status=SandboxStatus.INFRA_ERROR,
                returncode=None,
                stdout="",
                stderr=f"Sandbox infra error: {e}",
            )


def _to_str(data: str | bytes | None) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return data
