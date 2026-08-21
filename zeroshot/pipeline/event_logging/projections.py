"""Projections of a run's protocol event stream into the channels we consume.

LangGraph emits one raw stream of protocol events; a transformer turns it into
a named channel a sink can read.  Two of them live here because they are the
same kind of thing, and because what separates them is only legible side by
side: one is the durable record of what happened, complete but only after the
fact; the other is what the model is saying right now, and survives nowhere.

    RunEventTransformer      -> "run_events"      -> events.jsonl, console progress
    AgentMessageTransformer  -> "agent_messages"  -> console model output

Neither is scoped to a graph level: both see subgraphs, and neither publishes
an item whose consumption waits on an item that may never arrive.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from hashlib import sha256
from typing import Any, TypedDict, cast
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
    """Project graph activity into compact, stable research events.

    A `sink` receives each event where it is produced, rather than where the
    channel is drained.  The two are not the same moment: iterating a
    `ChatModelStream` drives the shared graph pump, so a consumer that is part
    way through one model's output can hold its turn for the rest of the run
    while the graph keeps going.  The durable record cannot depend on that
    consumer coming back.  The channel is still published, for consumers that
    want the events in arrival order alongside other projections.
    """

    CHANNEL = "run_events"

    required_stream_modes = ("values", "tools", "updates", "tasks", "custom")

    def __init__(
        self,
        scope: tuple[str, ...] = (),
        sink: Callable[[RunEvent], None] | None = None,
    ) -> None:
        super().__init__(scope)
        self.events = StreamChannel[RunEvent]()
        # The mux clones this transformer once per subgraph scope, and a root
        # instance already sees every level, so only the root one feeds the
        # sink. A clone would restate the whole run one nesting level at a time.
        self._sink = sink if not scope else None
        self._input_written = False
        self._agent_roles: dict[tuple[tuple[str, ...], str], str] = {}
        self._tool_context: dict[
            tuple[tuple[str, ...], str], tuple[str | None, str]
        ] = {}

    def init(self) -> dict[str, StreamChannel[RunEvent]]:
        return {self.CHANNEL: self.events}

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

        elif method == "custom" and isinstance(data, Mapping) and "model_retry" in data:
            self._push(
                event,
                "model_retry",
                cast(dict[str, Any], _safe_value(data["model_retry"])),
            )

        elif method == "custom" and isinstance(data, Mapping) and "prompt" in data:
            # Custom events reach every level's transformer and the built-in one
            # drops what is not its own scope; this projection keeps the whole
            # run, so an agent nested in a stage subgraph reports here too.
            self._push(
                event, "prompt", cast(dict[str, Any], _safe_value(data["prompt"]))
            )

        elif method == "tasks" and isinstance(data, Mapping):
            namespace = tuple(event["params"]["namespace"])
            node = str(data.get("name", ""))
            started = "input" in data
            if started:
                metadata = data.get("metadata")
                if isinstance(metadata, Mapping) and isinstance(
                    role := metadata.get("lc_agent_name"), str
                ):
                    self._agent_roles[(namespace, node)] = role
                if node == "tools" and isinstance(tool_calls := data["input"], list):
                    for call in tool_calls:
                        if not isinstance(call, Mapping):
                            continue
                        call_id = str(call.get("id", ""))
                        self._tool_context[(namespace, call_id)] = (
                            str(call.get("name", "")),
                            "model",
                        )
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
            if role := self._agent_roles.get((namespace, node)):
                details["role"] = role
            self._push(
                event,
                "node_started" if started else "node_finished",
                details,
            )
            if not started:
                self._agent_roles.pop((namespace, node), None)

        elif method == "tools" and isinstance(data, Mapping):
            namespace = tuple(event["params"]["namespace"])
            event_name = str(data.get("event", "tool-event")).replace("-", "_")
            call_id = str(data.get("tool_call_id", ""))
            context_key = (namespace, call_id)
            payload = {
                str(key): _safe_value(value)
                for key, value in data.items()
                if key != "event"
            }
            if event_name == "tool_started":
                context = self._tool_context.get(
                    context_key,
                    (str(data.get("tool_name", "")), "workflow"),
                )
                self._tool_context[context_key] = context
            else:
                context = self._tool_context.get(context_key, (None, "unknown"))
            payload.setdefault("tool_name", context[0])
            payload["caller"] = context[1]
            self._push(event, event_name, payload)
            if event_name in {"tool_finished", "tool_error"}:
                self._tool_context.pop(context_key, None)

        elif method == "updates" and isinstance(data, Mapping):
            namespace = tuple(event["params"]["namespace"])
            for node, update in data.items():
                if node == "tools" or not isinstance(update, Mapping):
                    continue
                role = self._agent_roles.get((namespace, str(node)))
                if "messages" in update:
                    details = {
                        "node": str(node),
                        "messages": [
                            _safe_value(message) for message in update["messages"]
                        ],
                    }
                    if role is not None:
                        details["role"] = role
                    self._push(
                        event,
                        "message",
                        details,
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
                # `lc_agent_name` arrives on the task-start event, while the
                # reason arrives in the following state update.  Joining them
                # here preserves role attribution without copying agent
                # progress into the outer workflow state.
                if update.get("stop_reason") is not None:
                    details = {
                        "node": str(node),
                        "reason": _safe_value(update["stop_reason"]),
                    }
                    if role is not None:
                        details["role"] = role
                    self._push(
                        event,
                        "stop_reason",
                        details,
                    )

        return True

    def _push(
        self,
        source: ProtocolEvent,
        event: str,
        data: dict[str, Any],
    ) -> None:
        run_event = RunEvent(
            event=event,
            timestamp_ms=source["params"]["timestamp"],
            namespace=list(source["params"]["namespace"]),
            data=data,
        )
        if self._sink is not None:
            self._sink(run_event)
        self.events.push(run_event)


class ModelStreamItem(TypedDict):
    """One piece of live model output, in whichever shape it arrived.

    A model that streams reports itself in parts, as protocol events; one that
    does not reports a whole message at the end.  Exactly one of the two shapes
    arrives per call, so `streamed` says which `payload` is, and no consumer
    has to reconcile the two against each other.

    `role` is the agent the call belongs to and `node` the node inside it, so a
    run with six agents can say which one is speaking rather than reporting
    every one of them as `model`.  `namespace` is where in the graph the call
    happened, which is what separates two calls a role cannot: the fan-out's
    branches share a role and run at the same time, and a run whose stages
    share one agent gives every stage the same role as well.
    """

    role: str | None
    node: str | None
    namespace: list[str]
    run_id: str
    streamed: bool
    payload: dict[str, Any]


class AgentMessageTransformer(StreamTransformer):
    """Live model output from every graph level, as the pieces it arrives in.

    LangGraph's own `messages` projection keeps to the run's own scope, so once
    the agent loop became a subgraph its tokens stopped reaching the console --
    the very output a long run is watched for.  Its docstring points here:
    "consumers that need subgraph tokens should ... register a custom
    transformer".

    That projection also hands out one `ChatModelStream` per call, and reading
    one means blocking until that call completes.  An attempt abandoned in
    flight -- which is what a retry does -- leaves a stream that never
    completes, and a consumer waiting on it reads nothing else for the rest of
    the run.  Publishing the pieces instead keeps every item consumable the
    moment it arrives, so an attempt nobody will ever finish costs its own
    output and nothing after it.
    """

    CHANNEL = "agent_messages"

    required_stream_modes = ("messages",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self.events = StreamChannel[ModelStreamItem]()

    def init(self) -> dict[str, StreamChannel[ModelStreamItem]]:
        return {self.CHANNEL: self.events}

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] != "messages":
            return True
        payload, metadata = event["params"]["data"]
        streamed = isinstance(payload, Mapping) and "event" in payload
        self.events.push(
            ModelStreamItem(
                role=metadata.get("lc_agent_name"),
                node=metadata.get("langgraph_node"),
                namespace=list(event["params"].get("namespace") or ()),
                run_id=str(metadata.get("run_id", "")),
                streamed=streamed,
                payload=cast(dict[str, Any], _safe_value(payload)),
            )
        )
        return True
