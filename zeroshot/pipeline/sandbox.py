import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


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


DEFAULT_LOG_LIMIT_BYTES = 1 * 1024 * 1024


class SandboxRunner:
    def __init__(
        self,
        python_executable: Path,
        default_timeout_s: float,
        max_stdout_bytes: int = DEFAULT_LOG_LIMIT_BYTES,
        max_stderr_bytes: int = DEFAULT_LOG_LIMIT_BYTES,
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
        work_dir: Path,
        timeout_s: float | None = None,
    ) -> SandboxResult:

        # sanity check
        work_dir = work_dir.resolve()
        if not work_dir.is_dir():
            raise ValueError(f"work_dir must be a directory: {work_dir}")

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
            "--setenv", "HOME", "/work",
            "--setenv", "LANG", "C.UTF-8",
            "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
            "--bind", str(work_dir), "/work",
            "--chdir", "/work",
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
