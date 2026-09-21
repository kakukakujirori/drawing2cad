from pathlib import PurePosixPath

import pytest

from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.verification import AttemptStore


def test_attempts_are_numbered_independently_by_round_and_stage(
    tmp_path,
) -> None:
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    current_round = [0]
    store = AttemptStore(workdir, round_source=lambda: current_round[0])

    assert store.issue("interpretation")[0] == "000"
    assert store.issue("interpretation")[0] == "001"
    assert store.issue("coding")[0] == "000"
    assert store.issue("audit")[0] == "000"
    resumed = AttemptStore(workdir, round_source=lambda: current_round[0])
    assert resumed.issue("audit")[0] == "001"
    current_round[0] = 1
    assert resumed.issue("audit")[0] == "000"
    assert store.issue("interpretation")[0] == "000"
    assert store.sandbox_attempt_dir(1, "coding", "007") == PurePosixPath(
        "/work/attempts/round_001/coding/007"
    )
    assert store.latest_sandbox_attempt_dir("interpretation", 1) == PurePosixPath(
        "/work/attempts/round_001/interpretation/000"
    )
    assert store.latest_sandbox_attempt_dir("coding", 1) == PurePosixPath(
        "/work/attempts/round_000/coding/000"
    )


@pytest.mark.parametrize(
    "root",
    [PurePosixPath(""), PurePosixPath(".."), PurePosixPath("nested/attempts")],
)
def test_attempt_store_rejects_a_non_basename_root(
    tmp_path,
    root: PurePosixPath,
) -> None:
    with pytest.raises(ValueError, match="directory basename"):
        AttemptStore(
            SandboxWorkdir(host_bind_dir=tmp_path),
            round_source=lambda: 0,
            root_dirname=root,
        )
