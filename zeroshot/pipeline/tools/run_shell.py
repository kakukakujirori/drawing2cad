from inspect import cleandoc

from langchain_core.tools import BaseTool, tool

from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir


def create_run_shell_tool(sandbox_runner: SandboxRunner, workdir: SandboxWorkdir) -> BaseTool:
    description = cleandoc(
        f"""Run a Bash command in the isolated {workdir.sandbox_bind_dir} directory.

        Each call starts a fresh process. Files under {workdir.sandbox_bind_dir} persist
        between calls, but shell variables and process state do not. The command can use
        bash commands, file read/write, and Python (CadQuery, ezdxf, Pillow, and NumPy included).

        Returns status, returncode, stdout, and stderr.
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
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    return run_shell