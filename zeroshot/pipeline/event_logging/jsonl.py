from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from types import TracebackType
from typing import IO, Any, Self

from zeroshot.pipeline.event_logging.projections import RunEvent, _safe_value


def has_run_completed(path: Path) -> bool:
    """Whether `path` holds a run that reached its `run_completed` event.

    A run that raised writes `run_failed` instead, and one killed outright
    writes neither, so both read as incomplete.
    """
    if not path.is_file():
        return False
    with path.open(encoding="utf-8") as handle:
        return any(json.loads(line).get("event") == "run_completed" for line in handle)


class JsonlEventWriter:
    """Incrementally write one JSON line per projected run event."""

    def __init__(self, path: Path, *, run_id: str, sample_id: str) -> None:
        self.path = path
        self.run_id = run_id
        self.sample_id = sample_id
        self._handle: IO[str] | None = None
        self._event_index = 0
        self._started_at = 0.0

    def __enter__(self) -> Self:
        self._handle = self.path.open("x", encoding="utf-8")
        self._started_at = perf_counter()
        self._write_synthetic("run_started", {})
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del traceback
        try:
            duration_ms = round((perf_counter() - self._started_at) * 1000)
            if exc_type is None:
                self._write_synthetic("run_completed", {"duration_ms": duration_ms})
            else:
                self._write_synthetic(
                    "run_failed",
                    {
                        "duration_ms": duration_ms,
                        "error_type": exc_type.__qualname__,
                        "error": _safe_value(str(exc_value)[:2000]),
                    },
                )
        finally:
            if self._handle is not None:
                self._handle.close()
                self._handle = None

    def write(self, event: RunEvent) -> None:
        if self._handle is None:
            raise RuntimeError("JsonlEventWriter is not open")
        record = {
            "schema_version": 1,
            "run_id": self.run_id,
            "sample_id": self.sample_id,
            "event_index": self._event_index,
            **event,
        }
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._handle.flush()
        self._event_index += 1

    def _write_synthetic(self, event: str, data: dict[str, Any]) -> None:
        self.write(
            RunEvent(
                event=event,
                timestamp_ms=int(datetime.now(tz=UTC).timestamp() * 1000),
                namespace=[],
                data=data,
            )
        )
