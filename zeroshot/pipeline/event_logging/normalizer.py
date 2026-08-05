from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
from typing import Any, TypedDict
from urllib.parse import urlsplit, urlunsplit

from langchain_core.messages import BaseMessage
from langgraph.stream import ProtocolEvent, StreamChannel, StreamTransformer
from pydantic_core import to_jsonable_python


class RunEvent(TypedDict):
    event: str
    timestamp_ms: int
    namespace: list[str]
    data: dict[str, Any]


_SECRET_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "access_token",
    "refresh_token",
    "password",
    "secret",
    "token",
}


def _summarize(value: str, kind: str) -> dict[str, str | int]:
    encoded = value.encode("utf-8")
    return {
        "omitted": kind,
        "size_bytes": len(encoded),
        "sha256": sha256(encoded).hexdigest(),
    }


def _is_secret_key(key: str) -> bool:
    key = key.casefold()
    return key in _SECRET_KEYS or key.endswith(
        ("_api_key", "_password", "_secret", "_token")
    )


def _redact(value: Any, field_name: str | None = None) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                "<redacted>" if _is_secret_key(key) else _redact(item, key.casefold())
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        if field_name in {"base64", "source"}:
            return _summarize(value, field_name)
        if value.startswith("data:image/"):
            return _summarize(value, "image_data_url")
        if value.startswith(("http://", "https://")):
            parsed = urlsplit(value)
            return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    return value


def _safe_value(value: Any) -> Any:
    if isinstance(value, BaseMessage):
        value = value.model_dump(
            mode="python",
            exclude={"additional_kwargs", "response_metadata", "artifact"},
        )
    serialized = to_jsonable_python(
        value,
        bytes_mode="base64",
        inf_nan_mode="strings",
    )
    return _redact(serialized)


class RunEventTransformer(StreamTransformer):
    """Project graph activity into compact, stable research events."""

    required_stream_modes = ("values", "tools", "updates", "tasks")

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self.events = StreamChannel[RunEvent]()
        self._input_written = False
        self._active_node: str | None = None
        self._tool_context: dict[str, tuple[str | None, str]] = {}

    def init(self) -> dict[str, StreamChannel[RunEvent]]:
        return {"run_events": self.events}

    def process(self, event: ProtocolEvent) -> bool:
        method = event["method"]
        data = event["params"]["data"]

        if method == "values" and not self._input_written:
            if isinstance(data, Mapping):
                messages = data.get("messages", [])
                self._push(
                    event,
                    "input",
                    {
                        "messages": [
                            _safe_value(message)
                            for message in messages
                            if isinstance(message, BaseMessage)
                        ]
                    },
                )
                self._input_written = True

        elif method == "tasks" and isinstance(data, Mapping):
            node = str(data.get("name", ""))
            started = "input" in data
            if started:
                self._active_node = node
                details = {
                    "node": node,
                    "triggers": _safe_value(data.get("triggers", [])),
                }
            else:
                error = data.get("error")
                details = {
                    "node": node,
                    "error": None if error is None else str(error),
                }
            self._push(
                event,
                "node_started" if started else "node_finished",
                details,
            )
            if not started and self._active_node == node:
                self._active_node = None

        elif method == "tools" and isinstance(data, Mapping):
            event_name = str(data.get("event", "tool-event")).replace("-", "_")
            call_id = str(data.get("tool_call_id", ""))
            payload = {
                str(key): _safe_value(value)
                for key, value in data.items()
                if key != "event"
            }
            if event_name == "tool_started":
                context = (
                    str(data.get("tool_name", "")),
                    "model" if self._active_node == "tools" else "workflow",
                )
                self._tool_context[call_id] = context
            else:
                context = self._tool_context.get(call_id, (None, "unknown"))
            payload.setdefault("tool_name", context[0])
            payload["caller"] = context[1]
            self._push(event, event_name, payload)
            if event_name in {"tool_finished", "tool_error"}:
                self._tool_context.pop(call_id, None)

        elif method == "updates" and isinstance(data, Mapping):
            for node, update in data.items():
                if node == "tools" or not isinstance(update, Mapping):
                    continue
                if "messages" in update:
                    self._push(
                        event,
                        "message",
                        {
                            "node": str(node),
                            "messages": [
                                _safe_value(message) for message in update["messages"]
                            ],
                        },
                    )
                if "last_verification" in update:
                    self._push(
                        event,
                        "verification",
                        {
                            "node": str(node),
                            "report": _safe_value(update["last_verification"]),
                        },
                    )
                # Whether the agent finished or ran out of turns is only in
                # graph state, so without this no offline reader can tell.
                if "stop_reason" in update:
                    self._push(
                        event,
                        "stop_reason",
                        {
                            "node": str(node),
                            "reason": _safe_value(update["stop_reason"]),
                        },
                    )

        return True

    def _push(
        self,
        source: ProtocolEvent,
        event: str,
        data: dict[str, Any],
    ) -> None:
        self.events.push(
            RunEvent(
                event=event,
                timestamp_ms=source["params"]["timestamp"],
                namespace=list(source["params"]["namespace"]),
                data=data,
            )
        )
