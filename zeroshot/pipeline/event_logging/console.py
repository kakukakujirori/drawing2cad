from __future__ import annotations

import json
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from time import perf_counter
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.pretty import Pretty
from rich.text import Text

from zeroshot.pipeline.event_logging.projections import RunEvent, _safe_value

if TYPE_CHECKING:
    from langchain_core.language_models.chat_model_stream import ChatModelStream


class ConsoleReporter:
    """Render live pipeline activity without changing the canonical event log."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console(stderr=True, highlight=False)

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
        name = event["event"]
        data = event["data"]

        if name == "input":
            self.console.print("\n[input]", style="bold cyan", markup=False)
            for message in data.get("messages", []):
                if not isinstance(message, Mapping):
                    self._render_value(message)
                    continue
                role = message.get("type", "message")
                self.console.print(f"{role}:", style="cyan", markup=False)
                self._render_value(message.get("content"))
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
        elif name == "verification":
            self.console.print(
                "\n[verification]",
                style="bold yellow",
                markup=False,
            )
            self._render_value(data.get("report"))
        elif name == "stop_reason":
            self.console.print(
                f"\n[stop] {data.get('reason')}",
                style="bold yellow",
                markup=False,
            )
        elif name == "message":
            # The corresponding ChatModelStream is rendered incrementally instead.
            return
        else:
            self._render_unhandled("event", name, data)

    def render_message(self, message: ChatModelStream) -> None:
        self.console.print(
            f"\n[model] {message.node or 'unknown'}",
            style="bold green",
            markup=False,
        )
        active_block: str | None = None
        emitted_delta_indexes: set[int] = set()
        tool_args_by_index: dict[int, str] = {}

        for event in message:
            event_type = event.get("event")
            index = event.get("index")
            if event_type == "content-block-delta":
                delta = event.get("delta")
                if not isinstance(delta, Mapping):
                    self._render_unhandled("model delta", "invalid payload", delta)
                    continue
                delta_type = delta.get("type")
                if delta_type == "text-delta":
                    active_block = self._render_delta(
                        "text",
                        str(delta.get("text", "")),
                        active_block,
                    )
                    if isinstance(index, int):
                        emitted_delta_indexes.add(index)
                elif delta_type == "reasoning-delta":
                    active_block = self._render_delta(
                        "reasoning",
                        str(delta.get("reasoning", "")),
                        active_block,
                    )
                    if isinstance(index, int):
                        emitted_delta_indexes.add(index)
                elif delta_type == "block-delta" and isinstance(index, int):
                    fields = delta.get("fields")
                    if (
                        isinstance(fields, Mapping)
                        and fields.get("type") == "tool_call_chunk"
                    ):
                        active_block = self._render_tool_call_delta(
                            fields,
                            index,
                            tool_args_by_index,
                            active_block,
                        )
                    else:
                        self._render_unhandled(
                            "model delta",
                            str(delta_type or "unknown"),
                            delta,
                        )
                else:
                    self._render_unhandled(
                        "model delta",
                        str(delta_type or "unknown"),
                        delta,
                    )
            elif event_type == "content-block-finish":
                content = event.get("content")
                if not isinstance(content, Mapping):
                    self._render_unhandled(
                        "model content",
                        "invalid payload",
                        content,
                    )
                    continue
                content_type = content.get("type")
                if content_type == "tool_call":
                    if isinstance(index, int) and index in tool_args_by_index:
                        self.console.print()
                        active_block = None
                    else:
                        if active_block is not None:
                            self.console.print()
                        active_block = None
                        self.console.print(
                            f"tool call: {content.get('name', 'unknown')}",
                            style="bold magenta",
                            markup=False,
                        )
                        self._render_value(content.get("args"))
                elif content_type in {"text", "reasoning"}:
                    if not isinstance(index, int) or index not in emitted_delta_indexes:
                        if content_type == "text":
                            active_block = self._render_delta(
                                "text",
                                str(content.get("text", "")),
                                active_block,
                            )
                        else:
                            active_block = self._render_delta(
                                "reasoning",
                                str(content.get("reasoning", "")),
                                active_block,
                            )
                else:
                    self._render_unhandled(
                        "model content",
                        str(content_type or "unknown"),
                        content,
                    )
            elif event_type not in {
                "message-start",
                "content-block-start",
                "message-finish",
            }:
                self._render_unhandled(
                    "model event",
                    str(event_type or "unknown"),
                    event,
                )

        if active_block is not None:
            self.console.print()
        usage = message.output.usage_metadata
        if usage:
            self.console.print("usage:", style="dim", markup=False)
            self._render_value(usage)

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
