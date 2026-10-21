"""Which code version produced which samples under an artifact root.

Hydra already writes the resolved config and overrides into every sample
directory, so this records only what Hydra cannot.  Nothing here is read by the
pipeline.
"""

from __future__ import annotations

import fcntl
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MANIFEST_FILENAME = "run_manifest.json"
SCHEMA_VERSION = 1


def _git(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def code_version() -> dict[str, Any]:
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
    }


def record_run(artifact_root: Path, sample_id: str) -> dict[str, Any]:
    """Add `sample_id` to the manifest under the current code version.

    Samples run at the same commit merge into one entry, so a sweep leaves a
    single record; re-running part of a directory at a different commit leaves a
    second one rather than overwriting the first.
    """
    path = artifact_root / MANIFEST_FILENAME
    version = code_version()
    artifact_root.mkdir(parents=True, exist_ok=True)
    path.touch()
    with path.open("r+", encoding="utf-8") as handle:
        # Samples finishing at once in parallel processes take turns.
        fcntl.flock(handle, fcntl.LOCK_EX)
        text = handle.read()
        document: dict[str, Any] = (
            json.loads(text)
            if text
            else {"schema_version": SCHEMA_VERSION, "code_versions": []}
        )

        now = datetime.now(tz=UTC).isoformat()
        entries = document["code_versions"]
        entry = next((item for item in entries if _same_version(item, version)), None)
        if entry is None:
            entry = {**version, "first_seen": now, "last_seen": now, "sample_ids": []}
            entries.append(entry)
        entry["last_seen"] = now
        if sample_id not in entry["sample_ids"]:
            entry["sample_ids"] = sorted([*entry["sample_ids"], sample_id])

        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(document, indent=2) + "\n")
    return document


def _same_version(entry: dict[str, Any], version: dict[str, Any]) -> bool:
    return all(entry.get(key) == value for key, value in version.items())
