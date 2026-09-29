"""Durable, unprojected model/tool events for the native baseline."""

from __future__ import annotations

import json
import time
from hashlib import sha256
from pathlib import Path
from typing import Any, Self, TextIO

from langgraph.stream import ProtocolEvent
from pydantic_core import to_jsonable_python


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            name = key.casefold().replace("-", "_")
            if name in {
                "api_key",
                "apikey",
                "authorization",
                "proxy_authorization",
                "password",
                "secret",
                "token",
                "cookie",
                "set_cookie",
            } or name.endswith(("_api_key", "_password", "_secret", "_token")):
                result[key] = "<redacted>"
            elif name == "base64" and isinstance(item, str):
                result[key] = _omitted(item, "base64")
            else:
                result[key] = _redact(item)
        return result
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str) and value.startswith("data:image/"):
        return _omitted(value, "image_data_url")
    return value


def _omitted(value: str, kind: str) -> dict[str, str | int]:
    encoded = value.encode("utf-8")
    return {
        "omitted": kind,
        "size_bytes": len(encoded),
        "sha256": sha256(encoded).hexdigest(),
    }


def safe_value(value: Any) -> Any:
    """Keep complete messages, including provider fields, while hiding secrets/images."""
    return _redact(
        to_jsonable_python(value, bytes_mode="base64", inf_nan_mode="strings")
    )


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(safe_value(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def has_run_completed(path: Path) -> bool:
    """Check generation completion, tolerating partial records from interruption."""
    if not path.is_file():
        return False
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and record.get("event") == "run_completed":
                return True
    return False


class EventLog:
    def __init__(self, path: Path, *, console: bool = False) -> None:
        self.path = path
        self.console = console
        self._file: TextIO | None = None
        self._event_index = 0

    def __enter__(self) -> Self:
        self._file = self.path.open("x", encoding="utf-8")
        return self

    def __exit__(self, *args: object) -> None:
        if self._file is not None:
            self._file.close()

    def write(self, event: str, data: Any, **metadata: Any) -> dict[str, Any]:
        if self._file is None:
            raise RuntimeError("EventLog must be opened as a context manager")
        record = safe_value(
            {
                "timestamp": time.time_ns() // 1_000_000,
                **metadata,
                "schema_version": 1,
                "event_index": self._event_index,
                "event": event,
                "data": data,
            }
        )
        line = json.dumps(record, ensure_ascii=False)
        self._file.write(line + "\n")
        self._file.flush()
        self._event_index += 1
        if self.console:
            print(line, flush=True)
        return record

    def record_protocol(self, event: ProtocolEvent) -> dict[str, Any] | None:
        """Keep deltas and completions distinct; never duplicate whole graph states."""
        if event["method"] not in {"messages", "tools"}:
            return None
        params = event["params"]
        metadata = {
            key: value
            for key, value in event.items()
            if key not in {"type", "method", "params"}
        }
        metadata.update({key: value for key, value in params.items() if key != "data"})
        return self.write(event["method"], params["data"], **metadata)
