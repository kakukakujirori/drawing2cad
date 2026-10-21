import json
import multiprocessing
from pathlib import Path
from typing import Any

from zeroshot import provenance


def _write(artifact_root: Path, sample_id: str, monkeypatch, **version: Any) -> None:
    monkeypatch.setattr(
        provenance,
        "code_version",
        lambda: {"git_commit": "aaa", "git_dirty": False, **version},
    )
    provenance.record_run(artifact_root, sample_id)


def _read(artifact_root: Path) -> dict[str, Any]:
    return json.loads(
        (artifact_root / provenance.MANIFEST_FILENAME).read_text(encoding="utf-8")
    )


def test_a_sweep_leaves_one_entry_listing_every_sample(
    tmp_path: Path, monkeypatch
) -> None:
    for sample_id in ("000775", "000364", "000405"):
        _write(tmp_path, sample_id, monkeypatch)

    versions = _read(tmp_path)["code_versions"]
    assert len(versions) == 1
    assert versions[0]["sample_ids"] == ["000364", "000405", "000775"]
    assert versions[0]["git_commit"] == "aaa"
    assert versions[0]["first_seen"] <= versions[0]["last_seen"]


def test_a_second_commit_gets_its_own_entry(tmp_path: Path, monkeypatch) -> None:
    """Which code produced which numbers is the manifest's whole job."""
    _write(tmp_path, "000364", monkeypatch)
    _write(tmp_path, "000405", monkeypatch, git_commit="bbb")

    versions = _read(tmp_path)["code_versions"]
    assert [(item["git_commit"], item["sample_ids"]) for item in versions] == [
        ("aaa", ["000364"]),
        ("bbb", ["000405"]),
    ]


def test_a_dirty_tree_is_not_folded_into_the_clean_commit(
    tmp_path: Path, monkeypatch
) -> None:
    _write(tmp_path, "000364", monkeypatch)
    _write(tmp_path, "000405", monkeypatch, git_dirty=True)

    versions = _read(tmp_path)["code_versions"]
    assert [item["git_dirty"] for item in versions] == [False, True]


def test_rerecording_a_sample_does_not_duplicate_it(
    tmp_path: Path, monkeypatch
) -> None:
    _write(tmp_path, "000364", monkeypatch)
    _write(tmp_path, "000364", monkeypatch)

    assert _read(tmp_path)["code_versions"][0]["sample_ids"] == ["000364"]


def _record_all(artifact_root: Path, sample_ids: list[str], start: Any) -> None:
    """Record samples the way one of several parallel runs would."""
    provenance.code_version = lambda: {"git_commit": "aaa", "git_dirty": False}
    start.wait()
    for sample_id in sample_ids:
        provenance.record_run(artifact_root, sample_id)


def test_parallel_runs_all_reach_the_manifest(tmp_path: Path) -> None:
    """None crashes on another's half-written file or drops another's sample."""
    context = multiprocessing.get_context("spawn")
    batches = [[f"{run}_{index:02d}" for index in range(20)] for run in "abcdefgh"]
    start = context.Barrier(len(batches))
    runs = [
        context.Process(target=_record_all, args=(tmp_path, batch, start))
        for batch in batches
    ]
    for run in runs:
        run.start()
    for run in runs:
        run.join(timeout=60)

    assert [run.exitcode for run in runs] == [0] * len(runs)
    versions = _read(tmp_path)["code_versions"]
    assert versions[0]["sample_ids"] == sorted(
        sample for batch in batches for sample in batch
    )


def test_code_version_reads_the_repository(tmp_path: Path) -> None:
    version = provenance.code_version()

    assert version["git_commit"] is not None
    assert len(version["git_commit"]) == 40
    assert isinstance(version["git_dirty"], bool)
