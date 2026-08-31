from __future__ import annotations

import json
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from rich.console import Console
from rich.pretty import Pretty
from rich.text import Text

from zeroshot.pipeline.event_logging.projections import (
    ModelStreamItem,
    RunEvent,
    _safe_value,
)


def _graph_path(namespace: Sequence[str]) -> str:
    """Where in the graph a call happened, without the per-run ids.

    A namespace entry is `node:uuid`; the uuid changes every run and says
    nothing to a reader watching one.
    """
    return "/".join(entry.split(":", 1)[0] for entry in namespace if entry)


def _speaker(role: str | None, node: str | None, namespace: Sequence[str] = ()) -> str:
    """Who this output belongs to, and where in the run they are speaking.

    The role first: a run drives several agents through the same `model` node,
    and which of them is speaking is the thing worth reading. The graph path
    follows, because a role is not always enough to tell two callers apart: the
    fan-out's proposers share a role and stream at the same time, and a run
    whose stages share one agent gives every stage that role too, so without
    the path the whole run reads as one speaker. The node is added last when it
    is not the call itself, so a message some middleware injected is attributed
    to what injected it rather than to the agent it landed in.
    """
    speaker = role or node or "unknown"
    if path := _graph_path(namespace):
        speaker = f"{speaker} · {path}"
    if node and node != "model":
        return f"{speaker} · {node}"
    return speaker


@dataclass
class _ModelSection:
    """What rendering one model's output carries between its pieces."""

    speaker: str
    opened: bool = False
    active_block: str | None = None
    emitted_delta_indexes: set[int] = field(default_factory=set)
    tool_args_by_index: dict[int, str] = field(default_factory=dict)


class ConsoleReporter:
    """Render live pipeline activity without changing the canonical event log."""

    def __init__(
        self,
        console: Console | None = None,
        muted_graph_nodes: Sequence[str] = (),
    ) -> None:
        self.console = console or Console(stderr=True, highlight=False)
        self._muted_graph_nodes = frozenset(muted_graph_nodes)
        # One per model call in flight, keyed by the run the pieces belong to.
        # An abandoned attempt leaves its entry behind unopened, which costs a
        # dict slot and nothing else.
        self._model_sections: dict[str, _ModelSection] = {}

    def _is_muted(self, namespace: Sequence[str]) -> bool:
        """Whether a configured graph node occurs in this nested call path."""
        return bool(self._muted_graph_nodes) and any(
            entry.split(":", 1)[0] in self._muted_graph_nodes for entry in namespace
        )

    @contextmanager
    def run_context(
        self,
        *,
        run_id: str,
        sample_id: str,
    ) -> Generator[None, None, None]:
        started_at = perf_counter()
        self.console.rule(Text(f"run {sample_id}"), style="bold blue")
        self.console.print(f"run_id: {run_id}", style="dim", markup=False)
        try:
            yield
        except BaseException as error:
            self.console.print(
                f"run failed: {type(error).__name__}: {error}",
                style="bold red",
                markup=False,
            )
            raise
        else:
            elapsed = perf_counter() - started_at
            self.console.print(
                f"run completed in {elapsed:.1f}s",
                style="bold green",
                markup=False,
            )

    def render_event(self, event: RunEvent) -> None:
        if self._is_muted(event.get("namespace", ())):
            return
        name = event["event"]
        data = event["data"]
        if name == "input":
            self.console.print("\n[input]", style="bold cyan", markup=False)
            self._render_messages(data.get("messages", []))
        elif name == "node_started":
            node = data.get("node", "unknown")
            suffix = " — waiting" if node == "model" else ""
            self.console.print(
                f"\n[node] {node} started{suffix}",
                style="bold blue",
                markup=False,
            )
        elif name == "node_finished":
            node = data.get("node", "unknown")
            error = data.get("error")
            style = "bold red" if error else "blue"
            self.console.print(
                f"[node] {node} finished",
                style=style,
                markup=False,
            )
            if error:
                self._render_value(error)
        elif name == "tool_started":
            tool = data.get("tool_name", "unknown")
            caller = data.get("caller", "unknown")
            self.console.print(
                f"\n[tool] {tool} started ({caller})",
                style="bold magenta",
                markup=False,
            )
        elif name == "tool_finished":
            tool = data.get("tool_name", "unknown")
            self.console.print(
                f"[tool] {tool} finished",
                style="magenta",
                markup=False,
            )
            self._render_tool_output(data.get("output"))
        elif name == "tool_error":
            tool = data.get("tool_name", "unknown")
            self.console.print(
                f"[tool] {tool} failed",
                style="bold red",
                markup=False,
            )
            self._render_value(
                {key: value for key, value in data.items() if key != "tool_name"}
            )
        elif name == "prompt":
            self.console.print(
                f"\n[prompt] {data.get('role', 'unknown')}",
                style="bold cyan",
                markup=False,
            )
            # Empty on a re-entry: the role's prompt was stated on the first ask
            # and has not changed since.
            if system := data.get("system"):
                self.console.print("system:", style="cyan", markup=False)
                self._render_value(system)
            self._render_messages(data.get("messages", []))
        elif name == "model_retry":
            outcome = "retrying" if data.get("retrying") else "giving up"
            self.console.print(
                f"\n[retry] {data.get('role', 'unknown')} "
                f"attempt {data.get('attempt')}/{data.get('max_attempts')} "
                f"failed with {data.get('error_type')} — {outcome}",
                style="bold red",
                markup=False,
            )
            self._render_value(data.get("error"))
        elif name == "verification":
            self.console.print(
                "\n[verification]",
                style="bold yellow",
                markup=False,
            )
            self._render_value(data.get("report"))
        elif name == "stop_reason":
            self.console.print(
                f"\n[stop] {' '.join(filter(None, (data.get('role'), data['reason'])))}",
                style="bold yellow",
                markup=False,
            )
        elif name == "stage_submission":
            # Most carry null: every node that touches the channel reports it,
            # and clearing it says nothing the model output did not.
            if (submission := data.get("submission")) is not None:
                self.console.print(
                    f"\n[submission] {data.get('node', 'unknown')}",
                    style="bold cyan",
                    markup=False,
                )
                self._render_value(submission)
        elif name == "stage_validation":
            if error := data.get("error"):
                self.console.print(
                    f"\n[invalid] {data.get('node', 'unknown')} — "
                    f"failure {data.get('failure_count')}",
                    style="bold red",
                    markup=False,
                )
                self._render_value(error)
        elif name == "audit":
            if (report := data.get("report")) is not None:
                verdict = "accepted" if report.get("accepted") else "rejected"
                self.console.print(
                    f"\n[audit] {verdict}", style="bold yellow", markup=False
                )
                self._render_value(report)
        elif name == "message":
            # The corresponding ChatModelStream is rendered incrementally instead.
            return
        else:
            self._render_unhandled("event", name, data)

    def render_model_item(self, item: ModelStreamItem) -> None:
        """Render one piece of model output as it arrives.

        Each piece is rendered on its own and the section it belongs to keeps
        what carries between them, so nothing here waits on a piece that may
        never come.  A call whose attempt was abandoned mid-flight simply stops
        contributing, and the next call renders normally.
        """
        if self._is_muted(item.get("namespace", ())):
            return
        if not item["streamed"]:
            self._render_whole_message(item)
            return

        run_id = item["run_id"]
        event = item["payload"]
        event_type = event.get("event")
        speaker = _speaker(item["role"], item["node"], item["namespace"])
        if event_type == "message-start":
            self._model_sections[run_id] = _ModelSection(speaker=speaker)
            return

        section = self._model_sections.setdefault(
            run_id, _ModelSection(speaker=speaker)
        )
        index = event.get("index")

        if event_type == "content-block-delta":
            delta = event.get("delta")
            if not isinstance(delta, Mapping):
                self._render_unhandled("model delta", "invalid payload", delta)
                return
            delta_type = delta.get("type")
            if delta_type == "text-delta":
                self._open(section)
                section.active_block = self._render_delta(
                    "text", str(delta.get("text", "")), section.active_block
                )
                if isinstance(index, int):
                    section.emitted_delta_indexes.add(index)
            elif delta_type == "reasoning-delta":
                self._open(section)
                section.active_block = self._render_delta(
                    "reasoning", str(delta.get("reasoning", "")), section.active_block
                )
                if isinstance(index, int):
                    section.emitted_delta_indexes.add(index)
            elif delta_type == "block-delta" and isinstance(index, int):
                fields = delta.get("fields")
                if (
                    isinstance(fields, Mapping)
                    and fields.get("type") == "tool_call_chunk"
                ):
                    self._open(section)
                    section.active_block = self._render_tool_call_delta(
                        fields, index, section.tool_args_by_index, section.active_block
                    )
                else:
                    self._render_unhandled(
                        "model delta", str(delta_type or "unknown"), delta
                    )
            else:
                self._render_unhandled(
                    "model delta", str(delta_type or "unknown"), delta
                )

        elif event_type == "content-block-finish":
            content = event.get("content")
            if not isinstance(content, Mapping):
                self._render_unhandled("model content", "invalid payload", content)
                return
            content_type = content.get("type")
            if content_type == "tool_call":
                self._open(section)
                if isinstance(index, int) and index in section.tool_args_by_index:
                    self.console.print()
                    section.active_block = None
                else:
                    if section.active_block is not None:
                        self.console.print()
                    section.active_block = None
                    self.console.print(
                        f"tool call: {content.get('name', 'unknown')}",
                        style="bold magenta",
                        markup=False,
                    )
                    self._render_value(content.get("args"))
            elif content_type in {"text", "reasoning"}:
                if (
                    not isinstance(index, int)
                    or index not in section.emitted_delta_indexes
                ):
                    self._open(section)
                    if content_type == "text":
                        section.active_block = self._render_delta(
                            "text", str(content.get("text", "")), section.active_block
                        )
                    else:
                        section.active_block = self._render_delta(
                            "reasoning",
                            str(content.get("reasoning", "")),
                            section.active_block,
                        )
            else:
                self._render_unhandled(
                    "model content", str(content_type or "unknown"), content
                )

        elif event_type == "message-finish":
            section = self._model_sections.pop(run_id, section)
            if section.active_block is not None:
                self.console.print()
            usage = event.get("usage")
            if usage and section.opened:
                self.console.print("usage:", style="dim", markup=False)
                self._render_value(usage)

        elif event_type != "content-block-start":
            self._render_unhandled("model event", str(event_type or "unknown"), event)

    def _render_whole_message(self, item: ModelStreamItem) -> None:
        """Render a message that arrived complete rather than in parts."""
        message = item["payload"]
        self.console.print(
            f"\n[model] {_speaker(item['role'], item['node'], item['namespace'])}",
            style="bold green",
            markup=False,
        )
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, Mapping) and block.get("type") == "text":
                    self._render_delta("text", str(block.get("text", "")), None)
                    self.console.print()
                else:
                    self._render_value(block)
        else:
            self._render_value(content)
        for call in message.get("tool_calls") or []:
            if not isinstance(call, Mapping):
                continue
            self.console.print(
                f"tool call: {call.get('name', 'unknown')}",
                style="bold magenta",
                markup=False,
            )
            self._render_value(call.get("args"))

    def _open(self, section: _ModelSection) -> None:
        """Announce the section on its first piece of content.

        Deferred so an attempt that never produces any leaves no trace at all.
        """
        if section.opened:
            return
        section.opened = True
        self.console.print(
            f"\n[model] {section.speaker}", style="bold green", markup=False
        )

    def _render_delta(
        self,
        block_type: str,
        value: str,
        active_block: str | None,
    ) -> str:
        if not value:
            return active_block or block_type
        if active_block != block_type:
            if active_block is not None:
                self.console.print()
            self.console.print(
                f"{block_type}:",
                style="green",
                markup=False,
            )
        self.console.print(
            value,
            end="",
            markup=False,
            highlight=False,
            soft_wrap=True,
        )
        return block_type

    def _render_tool_call_delta(
        self,
        fields: Mapping[str, Any],
        index: int,
        args_by_index: dict[int, str],
        active_block: str | None,
    ) -> str:
        current_args = str(fields.get("args") or "")
        previous_args = args_by_index.get(index, "")
        new_args = current_args.removeprefix(previous_args)
        if active_block != "tool_call":
            if active_block is not None:
                self.console.print()
            self.console.print(
                f"tool call: {fields.get('name') or 'unknown'}",
                style="bold magenta",
                markup=False,
            )
        if new_args:
            self.console.print(
                new_args,
                end="",
                markup=False,
                highlight=False,
                soft_wrap=True,
            )
        args_by_index[index] = current_args
        return "tool_call"

    def _render_tool_output(self, output: Any) -> None:
        if isinstance(output, Mapping):
            content = output.get("content")
            if isinstance(content, str):
                try:
                    self._render_value(json.loads(content))
                    return
                except ValueError:
                    pass
        self._render_value(output)

    def _render_unhandled(self, category: str, name: str, value: Any) -> None:
        self.console.print(
            f"\n[unhandled {category}] {name}",
            style="bold yellow",
            markup=False,
        )
        try:
            safe_value = _safe_value(value)
        except Exception as error:  # noqa: BLE001 - reporting must not break the run
            safe_value = {
                "value_type": type(value).__name__,
                "render_error": f"{type(error).__name__}: {error}",
            }
        self._render_value(safe_value)

    def _render_messages(self, messages: Any) -> None:
        """Render a transcript the way an input and a prompt both want it.

        A block that is not text is named rather than printed: reasoning is
        already on the console from the live stream, and encrypted or encoded
        content says nothing at all when spelled out.
        """
        for message in messages if isinstance(messages, list) else []:
            if not isinstance(message, Mapping):
                self._render_value(message)
                continue
            role = message.get("type", "message")
            self.console.print(f"{role}:", style="cyan", markup=False)
            content = message.get("content")
            if not isinstance(content, list):
                self._render_value(content)
                continue
            for block in content:
                if not isinstance(block, Mapping):
                    self._render_value(block)
                elif block.get("type") == "text":
                    self._render_value(block.get("text", ""))
                else:
                    self.console.print(
                        f"[{block.get('type', 'block')}]", style="dim", markup=False
                    )

    def _render_value(self, value: Any) -> None:
        if isinstance(value, str):
            self.console.print(value, markup=False, highlight=False, soft_wrap=True)
            return
        if isinstance(value, list):
            for item in value:
                if isinstance(item, Mapping) and item.get("type") == "text":
                    self._render_value(item.get("text", ""))
                else:
                    self._render_value(item)
            return
        self.console.print(
            Pretty(value, overflow="fold", max_length=None, max_string=None),
            soft_wrap=True,
        )
