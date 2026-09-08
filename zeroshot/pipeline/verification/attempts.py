"""Issue immutable verification-attempt directories by round and stage."""

from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Literal

from zeroshot.pipeline.sandbox import SandboxWorkdir

type AttemptStage = Literal["drawing", "coding"]
type RoundSource = Callable[[], int]


def attempt_relative_path(
    round_number: int,
    stage: AttemptStage,
    attempt_id: str,
) -> PurePosixPath:
    """Address one attempt below the configured attempt root."""
    if round_number < 0:
        raise ValueError("round_number must be non-negative")
    if stage not in {"drawing", "coding"}:
        raise ValueError(f"unsupported attempt stage: {stage!r}")
    if not attempt_id.isdigit():
        raise ValueError(f"attempt_id must be numeric: {attempt_id!r}")
    return PurePosixPath(f"round_{round_number:03d}", stage, attempt_id)


class AttemptStore:
    """Own attempt layout and numbering, deriving the round from run state."""

    def __init__(
        self,
        workdir: SandboxWorkdir,
        round_source: RoundSource,
        root_dirname: PurePosixPath = PurePosixPath("attempts"),
    ) -> None:
        if (
            root_dirname.is_absolute()
            or len(root_dirname.parts) != 1
            or root_dirname.name in {"", ".", ".."}
        ):
            raise ValueError("attempt root must be a directory basename")

        self.workdir = workdir
        self.root_dirname = root_dirname
        self._round_source = round_source
        self.host_root = workdir.host_bind_dir / root_dirname
        if self.host_root.is_symlink():
            raise ValueError("attempt root must not be a symlink")
        self.host_root.mkdir(parents=True, exist_ok=True)

        # Verifiers write from the host. Agents may inspect, but never alter,
        # an attempt after it has been issued.
        if root_dirname not in workdir.read_only_subdirs:
            workdir.read_only_subdirs.append(root_dirname)

    @property
    def sandbox_root(self) -> PurePosixPath:
        return self.workdir.sandbox_bind_dir / self.root_dirname

    def issue(self, stage: AttemptStage) -> tuple[str, Path, PurePosixPath]:
        """Create and return the next attempt in the current round and stage."""
        round_number = self._round_source()
        relative = attempt_relative_path(round_number, stage, "000").parent
        stage_dir = self.host_root / relative
        if stage_dir.is_symlink():
            raise ValueError("attempt stage directory must not be a symlink")
        stage_dir.mkdir(parents=True, exist_ok=True)
        existing = [
            int(path.name) for path in stage_dir.iterdir() if path.name.isdigit()
        ]
        attempt_id = f"{max(existing, default=-1) + 1:03d}"
        attempt_dir = stage_dir / attempt_id
        attempt_dir.mkdir(exist_ok=False)
        return (
            attempt_id,
            attempt_dir,
            self.sandbox_root
            / attempt_relative_path(round_number, stage, attempt_id),
        )

    def sandbox_attempt_dir(
        self,
        round_number: int,
        stage: AttemptStage,
        attempt_id: str,
    ) -> PurePosixPath:
        return self.sandbox_root / attempt_relative_path(
            round_number, stage, attempt_id
        )
