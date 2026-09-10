from inspect import cleandoc
from tempfile import NamedTemporaryFile

from langchain_core.tools import BaseTool, tool

from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir

_MAX_INLINE_CHARS = 16_000


def _inline_output(text: str, stream: str, workdir: SandboxWorkdir) -> str:
    """Keep large command output out of every subsequent model request."""
    if len(text) <= _MAX_INLINE_CHARS:
        return text

    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f"shell_{stream}_",
        suffix=".txt",
        dir=workdir.host_bind_dir / workdir.tmp_subdir,
        delete=False,
    ) as saved:
        saved.write(text)
        path = workdir.host_to_sandbox_path(saved.name)

    head = _MAX_INLINE_CHARS * 3 // 4
    tail = _MAX_INLINE_CHARS - head
    omitted = len(text) - _MAX_INLINE_CHARS
    return (
        text[:head] + f"\n\n[Omitted {omitted} characters from {stream}. "
        f"Captured output saved to {path}; read only the relevant sections.]\n\n"
        + text[-tail:]
    )


def create_run_shell_tool(
    sandbox_runner: SandboxRunner, workdir: SandboxWorkdir
) -> BaseTool:
    description = cleandoc(
        f"""Run a Bash command in the isolated {workdir.sandbox_bind_dir} directory.

        Each call starts a fresh process. Files under {workdir.sandbox_bind_dir} persist
        between calls, but shell variables and process state do not. The command can use
        bash commands, file read/write, and Python (CadQuery, ezdxf, Pillow, and NumPy included).

        Returns status, returncode, stdout, and stderr. Long stdout/stderr is saved
        to a file; the response contains its beginning, end, and file path. Filter
        searches and file reads to the information you need.
        """
    )

    @tool("run_shell", description=description)
    def run_shell(command: str) -> dict[str, str | int | None]:
        result = sandbox_runner.run(
            command=command,
            workdir=workdir,
        )

        return {
            "status": result.status.value,
            "returncode": result.returncode,
            "stdout": _inline_output(result.stdout, "stdout", workdir),
            "stderr": _inline_output(result.stderr, "stderr", workdir),
        }

    return run_shell
