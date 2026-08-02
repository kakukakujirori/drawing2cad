import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NoReturn


@dataclass(frozen=True, slots=True)
class SGLangServer:
    """Build and launch one SGLang OpenAI-compatible server process."""

    model_path: str
    python_executable: str = "python"
    host: str = "127.0.0.1"
    port: int = 30000
    tensor_parallel_size: int = 1
    context_length: int | None = None
    mem_fraction_static: float | None = None
    reasoning_parser: str | None = None
    tool_call_parser: str | None = None
    trust_remote_code: bool = False
    extra_args: Sequence[str] = ()

    def __post_init__(self) -> None:
        if not self.model_path:
            raise ValueError("model_path must not be empty")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be at least 1")

    def command(self) -> list[str]:
        command = [
            self.python_executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.model_path,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--tp",
            str(self.tensor_parallel_size),
        ]

        optional_arguments = (
            ("--context-length", self.context_length),
            ("--mem-fraction-static", self.mem_fraction_static),
            ("--reasoning-parser", self.reasoning_parser),
            ("--tool-call-parser", self.tool_call_parser),
        )
        for flag, value in optional_arguments:
            if value is not None:
                command.extend((flag, str(value)))

        if self.trust_remote_code:
            command.append("--trust-remote-code")

        command.extend(str(argument) for argument in self.extra_args)
        return command

    def launch(self) -> NoReturn:
        command = self.command()
        os.execvp(command[0], command)
