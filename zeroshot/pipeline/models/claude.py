"""ChatModel wrapper for Anthropic Claude using subscription OAuth credentials.

Targets Anthropic Messages API (`https://api.anthropic.com/v1/messages`) using
an OAuth access token from an active Claude subscription (minted via Claude Code CLI
`claude auth login`). This enables subscription-quota usage without per-token API charges.

Handles:
- Base64 PNG images formatted by LangGraph (converts `image_url` data URLs to Anthropic `image` source)
- Image history trimmed to the input drawings and the newest others (`image_history_limit`)
- Native Anthropic tool calling (`bind_tools`, `tool_use`, `tool_result`)
- SSE streaming with `AIMessageChunk` and `ToolCallChunk` merging
- Thinking blocks sent back with their signatures, so Claude keeps its reasoning,
  the oldest withheld past `thinking_replay_budget`
- Top-level automatic prompt caching (`prompt_cache_ttl`)
- Anthropic usage reported as LangChain `usage_metadata`
- OAuth token discovery from `~/.claude/.credentials.json` or `CLAUDE_CODE_OAUTH_TOKEN`
- Friendly 429 usage limit and 401 token expiration error messaging, raised as
  `ClaudeAPIError` so that transport retry can tell transient failures apart
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, override

import httpx
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.messages.tool import ToolCallChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableBinding
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field, PrivateAttr

from .image_history import trim_image_history

logger = logging.getLogger(__name__)

ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages?beta=true"
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_ANTHROPIC_BETA = (
    "claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14"
)

# The HTTP status each Anthropic error type stands for, so that an error event
# inside a 200 stream is judged as the same error returned as a status would be.
_ERROR_TYPE_STATUS = {
    "invalid_request_error": 400,
    "authentication_error": 401,
    "permission_error": 403,
    "not_found_error": 404,
    "request_too_large": 413,
    "rate_limit_error": 429,
    "api_error": 500,
    "timeout_error": 504,
    "overloaded_error": 529,
}


class ClaudeAPIError(RuntimeError):
    """A failure the Messages API reported, as an HTTP status or mid-stream.

    `status_code` is None for a stream that ended before `message_stop`.
    """

    def __init__(
        self, message: str, *, status_code: int | None, error_type: str | None = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type

    @property
    def retryable(self) -> bool:
        """Rate limits, overload, server faults and cut streams can clear unchanged."""
        return (
            self.status_code is None
            or self.status_code == 429
            or self.status_code >= 500
        )


def usage_metadata_from_anthropic(
    usage: Mapping[str, Any] | None,
) -> UsageMetadata | None:
    """Anthropic's `usage` in LangChain's terms.

    Anthropic counts cache reads and writes apart from `input_tokens`; LangChain's
    `input_tokens` includes both, as the other backends here report it.
    """
    if not usage:
        return None
    cache_read = usage.get("cache_read_input_tokens") or 0
    cache_creation = usage.get("cache_creation_input_tokens") or 0
    input_tokens = (usage.get("input_tokens") or 0) + cache_read + cache_creation
    output_tokens = usage.get("output_tokens") or 0
    metadata = UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        input_token_details={
            "cache_read": cache_read,
            "cache_creation": cache_creation,
        },
    )
    thinking = (usage.get("output_tokens_details") or {}).get("thinking_tokens")
    if thinking is not None:
        metadata["output_token_details"] = {"reasoning": thinking}
    return metadata


def _resolve_oauth_token(
    explicit_token: str | None = None,
    credentials_path: Path | None = None,
) -> str:
    """Resolve Anthropic OAuth access token from arguments, environment, or CLI credentials."""
    if explicit_token:
        return explicit_token

    env_token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    if env_token:
        return env_token

    cred_path = credentials_path or (Path.home() / ".claude" / ".credentials.json")
    if cred_path.is_file():
        try:
            data = json.loads(cred_path.read_text(encoding="utf-8"))
            oauth = data.get("claudeAiOauth") or {}
            access_token = oauth.get("accessToken")
            if access_token:
                return access_token
        except Exception as e:
            logger.warning("Failed to read credentials from %s: %s", cred_path, e)

    raise ValueError(
        "No Claude OAuth token found. Please ensure you are logged in via "
        "'claude auth login' or set the CLAUDE_CODE_OAUTH_TOKEN environment variable."
    )


def _convert_tool_to_anthropic(tool: Any) -> dict[str, Any]:
    """Convert a LangChain tool / function to Anthropic tool schema."""
    if isinstance(tool, dict) and "name" in tool and "input_schema" in tool:
        return tool
    openai_tool = convert_to_openai_tool(tool)
    fn = openai_tool.get("function", {})
    return {
        "name": fn.get("name"),
        "description": fn.get("description", ""),
        "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
    }


def _anthropic_image(block: dict[str, Any]) -> dict[str, Any]:
    """A LangChain v1 image block in Anthropic's `source` form.

    The prompt attaches input drawings as `base64` + `mime_type` (staged) or
    as a data `url` (native); Anthropic takes neither key directly.
    """
    if "source" in block:  # already Anthropic's
        return {"type": "image", "source": block["source"]}
    if block.get("base64") is not None:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": block.get("mime_type") or "image/png",
                "data": block["base64"],
            },
        }
    url = block.get("url")
    if isinstance(url, str) and url.startswith("data:"):
        header, b64_data = url.split(",", 1)
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": header.split(";")[0].split(":")[1],
                "data": b64_data,
            },
        }
    if isinstance(url, str) and url:
        return {"type": "image", "source": {"type": "url", "url": url}}
    raise ValueError(f"Unsupported image block: {sorted(block)}")


def _anthropic_thinking(block: dict[str, Any]) -> dict[str, Any] | None:
    """A stored reasoning block as Claude wrote it, or None if it cannot go back.

    The API accepts a thinking block only unchanged and signed; an unsigned one
    (never from this model) could not be verified.
    """
    if "redacted" in block:
        return {"type": "redacted_thinking", "data": block["redacted"]}
    if not block.get("signature"):
        return None
    return {
        "type": "thinking",
        "thinking": block.get("reasoning", ""),
        "signature": block["signature"],
    }


def _anthropic_tool_use(call: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "tool_use",
        "id": call["id"],
        "name": call["name"],
        "input": call["args"],
    }


def _is_reasoning(block: Any) -> bool:
    return isinstance(block, dict) and block.get("type") == "reasoning"


def _thinking_tokens(message: AIMessage) -> int:
    """What the turn's thinking costs as input when replayed: its full count.

    The summary text is far shorter than the thinking a signature stands for.
    """
    usage = message.usage_metadata or usage_metadata_from_anthropic(
        message.response_metadata.get("usage")
    )
    return ((usage or {}).get("output_token_details") or {}).get("reasoning") or 0


def trim_thinking_history(
    messages: Sequence[BaseMessage], budget: int | None
) -> list[BaseMessage]:
    """Withhold the oldest turns' thinking once replayed thinking passes `budget`.

    Every replayed block counts as input at its full thinking count, so a long
    run outgrows the context window. Only a leading run of thinking can go
    without invalidating the blocks after it, so turns lose theirs oldest
    first. They go half a budget at a time: the cut moves only when the total
    crosses another half budget, so the prompt cache restarts at the cut a few
    times a run rather than on every request. ``None`` replays everything.
    """
    outgoing = list(messages)
    if budget is None:
        return outgoing
    costs = [
        (index, _thinking_tokens(message))
        for index, message in enumerate(messages)
        if isinstance(message, AIMessage)
        and isinstance(message.content, list)
        and any(_is_reasoning(block) for block in message.content)
    ]
    total = sum(cost for _, cost in costs)
    if total <= budget:
        return outgoing

    step = max(1, budget // 2)
    target = -(-(total - budget) // step) * step  # rounded up to whole steps
    dropped = turns = 0
    for index, cost in costs:
        if dropped >= target:
            break
        message = messages[index]
        outgoing[index] = message.model_copy(
            update={"content": [b for b in message.content if not _is_reasoning(b)]}
        )
        dropped += cost
        turns += 1
    logger.info(
        "Thinking history: withheld %s of %s turns (%s of %s tokens)",
        turns,
        len(costs),
        dropped,
        total,
    )
    return outgoing


def _format_messages_for_anthropic(
    messages: Sequence[BaseMessage],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert LangChain messages into Anthropic's top-level system prompt and messages list."""
    system_parts: list[str] = []
    formatted_messages: list[dict[str, Any]] = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            system_parts.append(str(msg.content))
        elif isinstance(msg, HumanMessage):
            parts: list[dict[str, Any]] = []
            if isinstance(msg.content, str):
                parts.append({"type": "text", "text": msg.content})
            elif isinstance(msg.content, list):
                for item in msg.content:
                    if isinstance(item, str):
                        parts.append({"type": "text", "text": item})
                    elif isinstance(item, dict):
                        item_type = item.get("type")
                        if item_type == "text":
                            parts.append({"type": "text", "text": item.get("text", "")})
                        elif item_type == "image_url":
                            url = item.get("image_url", {}).get("url", "")
                            if url.startswith("data:"):
                                header, b64_data = url.split(",", 1)
                                media_type = header.split(";")[0].split(":")[1]
                                parts.append(
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": media_type,
                                            "data": b64_data,
                                        },
                                    }
                                )
                            else:
                                raise ValueError(
                                    f"Unsupported image_url format (expected data URL): {url[:40]}..."
                                )
                        elif item_type == "image":
                            parts.append(_anthropic_image(item))
                        else:
                            parts.append(item)
            formatted_messages.append({"role": "user", "content": parts})
        elif isinstance(msg, AIMessage):
            # Blocks go back in the order Claude wrote them, thinking included:
            # its earlier reasoning reaches it only through these blocks.
            parts: list[dict[str, Any]] = []
            calls = {tc["id"]: tc for tc in msg.tool_calls}
            if isinstance(msg.content, str) and msg.content:
                parts.append({"type": "text", "text": msg.content})
            elif isinstance(msg.content, list):
                for item in msg.content:
                    if isinstance(item, str):
                        parts.append({"type": "text", "text": item})
                    elif not isinstance(item, dict):
                        continue
                    elif item.get("type") == "text":
                        parts.append({"type": "text", "text": item.get("text", "")})
                    elif item.get("type") == "reasoning":
                        if (thinking := _anthropic_thinking(item)) is not None:
                            parts.append(thinking)
                    elif item.get("type") == "tool_call" and item.get("id") in calls:
                        parts.append(_anthropic_tool_use(calls.pop(item["id"])))
            # Calls the content does not place go last, as tool_use blocks do.
            parts += [_anthropic_tool_use(tc) for tc in calls.values()]
            formatted_messages.append({"role": "assistant", "content": parts})
        elif isinstance(msg, ToolMessage):
            tool_content: Any
            if isinstance(msg.content, str):
                tool_content = msg.content
            elif isinstance(msg.content, list):
                tool_parts: list[dict[str, Any]] = []
                for item in msg.content:
                    if isinstance(item, str):
                        tool_parts.append({"type": "text", "text": item})
                    elif isinstance(item, dict):
                        itype = item.get("type")
                        if itype == "text":
                            tool_parts.append(
                                {"type": "text", "text": item.get("text", "")}
                            )
                        elif itype == "image_url":
                            url = item.get("image_url", {}).get("url", "")
                            if isinstance(url, str) and url.startswith("data:"):
                                header, b64_data = url.split(",", 1)
                                media_type = header.split(";")[0].split(":")[1]
                                tool_parts.append(
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": media_type,
                                            "data": b64_data,
                                        },
                                    }
                                )
                        elif itype == "image":
                            tool_parts.append(_anthropic_image(item))
                        else:
                            tool_parts.append(item)
                tool_content = tool_parts if tool_parts else str(msg.content)
            else:
                tool_content = str(msg.content)

            formatted_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": tool_content,
                            "is_error": msg.status == "error",
                        }
                    ],
                }
            )
        else:
            formatted_messages.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": str(msg.content)}],
                }
            )

    # Anthropic requires strict role alternation (user -> assistant -> user).
    # Merge consecutive messages sharing the same role.
    merged_messages: list[dict[str, Any]] = []
    for m in formatted_messages:
        if merged_messages and merged_messages[-1]["role"] == m["role"]:
            merged_messages[-1]["content"].extend(m["content"])
        else:
            merged_messages.append(m)

    system_prompt = "\n\n".join(system_parts) if system_parts else None
    return system_prompt, merged_messages


class ChatClaude(BaseChatModel):
    """ChatModel targeting Anthropic Claude using Claude subscription OAuth credentials."""

    model: str = Field(
        default="claude-sonnet-5", description="Anthropic model identifier"
    )
    max_tokens: int = Field(default=128000, description="Max tokens to generate")
    temperature: float | None = Field(
        default=None, description="Sampling temperature (0.0 to 1.0)"
    )
    thinking: dict[str, Any] | None = Field(
        default=None,
        description="Extended thinking configuration, e.g. {'type': 'adaptive'}",
    )
    reasoning: dict[str, Any] | None = Field(
        default=None,
        description="Alias for reasoning configuration from configs",
    )
    output_config: dict[str, Any] | None = Field(
        default=None,
        description="Anthropic output configuration, e.g. {'effort': 'max'}",
    )
    timeout: float = Field(default=600.0, description="Request timeout in seconds")
    credentials_path: Path | None = Field(
        default=None,
        description="Path to Claude CLI credentials file (defaults to ~/.claude/.credentials.json)",
    )
    token: str | None = Field(
        default=None,
        description="Explicit OAuth access token override",
    )
    max_retries: int = Field(
        default=0,
        description="Max internal HTTP retries. Kept at 0 so agent middleware handles retry policy.",
    )
    # None sends every image; anchors remain even if they exceed this budget.
    image_history_limit: int | None = Field(default=None, gt=0, strict=True)
    # Top-level automatic caching: its breakpoint follows the end of the
    # conversation, so each request reads the prefix the one before it wrote.
    # A cache entry lives this long from the start of the request that last
    # used it. None sends no cache_control.
    prompt_cache_ttl: Literal["5m", "1h"] | None = "5m"
    # Replayed thinking tokens past which the oldest turns' thinking is
    # withheld (`trim_thinking_history`). None replays all, as the API would.
    thinking_replay_budget: int | None = Field(default=None, gt=0, strict=True)

    # Fixed for the life of this instance, so a run's requests differ only in
    # what was appended. The API leaves the billing header out of both the
    # cache key and thinking-block binding (checked live on 2026-10-16), so a
    # fresh id per request cost nothing; it only made requests harder to diff.
    _prompt_id: str = PrivateAttr(default_factory=lambda: str(uuid.uuid4()))

    @property
    @override
    def _llm_type(self) -> str:
        return "claude"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "thinking": self.thinking,
            "output_config": self.output_config,
        }

    def _resolved_model_name(self) -> str:
        """Strip provider prefix if present (e.g. 'anthropic/claude-sonnet-5' -> 'claude-sonnet-5')."""
        if "/" in self.model:
            return self.model.split("/", 1)[1]
        return self.model

    def _resolved_timeout(self) -> float:
        """Normalize timeout in milliseconds (e.g. 600000) to seconds."""
        if self.timeout > 1000:
            return self.timeout / 1000.0
        return self.timeout

    def _get_headers(self) -> dict[str, str]:
        token = _resolve_oauth_token(
            explicit_token=self.token,
            credentials_path=self.credentials_path,
        )
        return {
            "Authorization": f"Bearer {token}",
            "anthropic-version": DEFAULT_ANTHROPIC_VERSION,
            "anthropic-beta": DEFAULT_ANTHROPIC_BETA,
            "anthropic-dangerous-direct-browser-access": "true",
            "User-Agent": "claude-cli/2.1.284 (external, sdk-cli)",
            "x-app": "cli",
            "content-type": "application/json",
        }

    def _build_payload(
        self,
        messages: Sequence[BaseMessage],
        stop: list[str] | None = None,
        tools: Sequence[Any] | None = None,
        tool_choice: Any = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        # Prune copies of the same LangChain messages the other backends see;
        # stored conversation messages stay intact.
        if self.image_history_limit is not None:
            messages = trim_image_history(messages, self.image_history_limit)
        messages = trim_thinking_history(messages, self.thinking_replay_budget)
        system_prompt, formatted_messages = _format_messages_for_anthropic(messages)
        billing_header = (
            f"x-anthropic-billing-header: cc_version=2.1.284.12a; cc_entrypoint=sdk-cli; "
            f"cch=00000; cc_prompt_id={self._prompt_id}; cc_turn_origin=sdk; cc_prompt_index=0; cc_turn_index=1;"
        )
        system_blocks: list[dict[str, Any]] = [{"type": "text", "text": billing_header}]
        if system_prompt:
            system_blocks.append({"type": "text", "text": system_prompt})

        payload: dict[str, Any] = {
            "model": self._resolved_model_name(),
            "max_tokens": self.max_tokens,
            "messages": formatted_messages,
            "system": system_blocks,
            "stream": stream,
        }
        if self.prompt_cache_ttl is not None:
            payload["cache_control"] = {
                "type": "ephemeral",
                "ttl": self.prompt_cache_ttl,
            }
        if self.temperature is not None:
            payload["temperature"] = self.temperature

        # Resolve thinking / reasoning / output_config configuration
        thinking_conf = self.thinking
        if thinking_conf is None and self.reasoning is not None:
            thinking_conf = {"type": "adaptive"}

        effort = None
        if self.output_config and "effort" in self.output_config:
            effort = self.output_config["effort"]
        elif thinking_conf and "effort" in thinking_conf:
            effort = thinking_conf["effort"]
        elif self.reasoning and "effort" in self.reasoning:
            effort = self.reasoning["effort"]

        if thinking_conf is not None:
            thinking_payload = {k: v for k, v in thinking_conf.items() if k != "effort"}
            if "display" not in thinking_payload:
                thinking_payload["display"] = "summarized"
            payload["thinking"] = thinking_payload

        if effort is not None:
            payload["output_config"] = {"effort": effort}

        if stop:
            payload["stop_sequences"] = stop
        if tools:
            payload["tools"] = [_convert_tool_to_anthropic(t) for t in tools]
            if tool_choice:
                # Claude Opus 5.5 and Sonnet 5.5 answer a forced choice ("any" or
                # a named tool, as ToolStrategy sends) with a 400, so ask for
                # "auto". ModelCallRetryMiddleware re-asks a turn that answers in
                # text instead of calling the answer tool.
                options = tool_choice if isinstance(tool_choice, dict) else {}
                kind = options.get("type", tool_choice)
                choice: dict[str, Any] = {"type": "none" if kind == "none" else "auto"}
                if kind != "none" and "disable_parallel_tool_use" in options:
                    choice["disable_parallel_tool_use"] = options[
                        "disable_parallel_tool_use"
                    ]
                payload["tool_choice"] = choice
        return payload

    @override
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> RunnableBinding:
        """Bind tool definitions to the chat model for LangGraph compatibility."""
        if "response_format" in kwargs:
            # ProviderStrategy binds its schema here. Nothing sends it, so every
            # answer would come back unstructured; fail on the first call instead.
            raise ValueError(
                "ChatClaude sends no response format; use response_format_strategy: tool"
            )
        formatted_tools = [_convert_tool_to_anthropic(t) for t in tools]
        return self.bind(tools=formatted_tools, tool_choice=tool_choice, **kwargs)

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")
        payload = self._build_payload(
            messages, stop=stop, tools=tools, tool_choice=tool_choice, stream=False
        )

        with httpx.Client(timeout=self._resolved_timeout()) as client:
            resp = client.post(
                ANTHROPIC_MESSAGES_URL,
                headers=self._get_headers(),
                json=payload,
            )
            if resp.is_error:
                self._handle_error(resp)
            data = resp.json()

        content_parts: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []

        for block in data.get("content", []):
            btype = block.get("type")
            if btype == "thinking":
                content_parts.append(
                    {
                        "type": "reasoning",
                        "reasoning": block.get("thinking", ""),
                        "signature": block.get("signature", ""),
                    }
                )
            elif btype == "redacted_thinking":
                content_parts.append(
                    {"type": "reasoning", "redacted": block.get("data", "")}
                )
            elif btype == "text":
                content_parts.append(
                    {
                        "type": "text",
                        "text": block.get("text", ""),
                    }
                )
            elif btype == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id"),
                        "name": block.get("name"),
                        "args": block.get("input", {}),
                        "type": "tool_call",
                    }
                )

        ai_msg = AIMessage(
            content=content_parts if content_parts else "",
            tool_calls=tool_calls,
            response_metadata={
                "id": data.get("id"),
                "model": data.get("model"),
                "stop_reason": data.get("stop_reason"),
                "usage": data.get("usage"),
            },
            usage_metadata=usage_metadata_from_anthropic(data.get("usage")),
        )
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    @override
    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")
        payload = self._build_payload(
            messages, stop=stop, tools=tools, tool_choice=tool_choice, stream=True
        )

        with httpx.Client(timeout=self._resolved_timeout()) as client:
            with client.stream(
                "POST",
                ANTHROPIC_MESSAGES_URL,
                headers=self._get_headers(),
                json=payload,
            ) as resp:
                if resp.is_error:
                    resp.read()
                    self._handle_error(resp)

                decoder = _StreamDecoder()
                for line in resp.iter_lines():
                    for chunk, token in decoder.feed(line):
                        generation = ChatGenerationChunk(message=chunk)
                        if run_manager and token is not None:
                            run_manager.on_llm_new_token(token, chunk=generation)
                        yield generation
                decoder.finish()

    @override
    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")
        payload = self._build_payload(
            messages, stop=stop, tools=tools, tool_choice=tool_choice, stream=True
        )

        async with httpx.AsyncClient(timeout=self._resolved_timeout()) as client:
            async with client.stream(
                "POST",
                ANTHROPIC_MESSAGES_URL,
                headers=self._get_headers(),
                json=payload,
            ) as resp:
                if resp.is_error:
                    await resp.aread()
                    self._handle_error(resp)

                decoder = _StreamDecoder()
                async for line in resp.aiter_lines():
                    for chunk, token in decoder.feed(line):
                        generation = ChatGenerationChunk(message=chunk)
                        if run_manager and token is not None:
                            await run_manager.on_llm_new_token(token, chunk=generation)
                        yield generation
                decoder.finish()

    def _handle_error(self, resp: httpx.Response) -> None:
        try:
            err = resp.json().get("error", {})
            err_msg = err.get("message", resp.text)
            err_type = err.get("type")
        except Exception:
            err_msg = resp.text
            err_type = None

        if resp.status_code == 429:
            message = f"Claude subscription usage limit / rate limit exceeded (HTTP 429): {err_msg}"
        elif resp.status_code == 401:
            message = (
                f"Claude OAuth token expired or unauthorized (HTTP 401): {err_msg}. "
                f"Please run 'claude auth login' in terminal to refresh credentials."
            )
        else:
            message = (
                f"Anthropic API request failed (HTTP {resp.status_code}): {err_msg}"
            )
        raise ClaudeAPIError(message, status_code=resp.status_code, error_type=err_type)


class _StreamDecoder:
    """Turn the Messages API's server-sent events into LangChain chunks.

    Shared by the sync and async streams, which differ only in how lines
    arrive. A content block's chunks carry the index the API gave it, so they
    merge into one block. `finish` raises unless the stream reached
    `message_stop`: one cut short would otherwise pass for a whole answer.
    """

    def __init__(self) -> None:
        self._index = 0
        self._message: dict[str, Any] = {}
        self._usage: dict[str, Any] = {}
        self._stopped = False

    def feed(self, line: str) -> list[tuple[AIMessageChunk, str | None]]:
        """The chunks one SSE line yields, each with the token to report, if any."""
        if not line or not line.startswith("data: "):
            return []
        data = line[len("data: ") :].strip()
        if data == "[DONE]":
            self._stopped = True
            return []
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            return []

        etype = event.get("type")
        if etype == "message_start":
            message = event.get("message") or {}
            self._message = {"id": message.get("id"), "model": message.get("model")}
            self._usage = dict(message.get("usage") or {})
            return []
        if etype == "content_block_start":
            self._index = event.get("index", 0)
            return self._block_start(event.get("content_block") or {})
        if etype == "content_block_delta":
            return self._block_delta(event.get("delta") or {})
        if etype == "message_delta":
            # Counts here are cumulative; message_start's fill in any left out.
            usage = {**self._usage, **(event.get("usage") or {})}
            chunk = AIMessageChunk(
                content="",
                response_metadata={
                    **self._message,
                    "stop_reason": (event.get("delta") or {}).get("stop_reason"),
                    "usage": usage,
                },
                usage_metadata=usage_metadata_from_anthropic(usage),
            )
            return [(chunk, None)]
        if etype == "message_stop":
            self._stopped = True
        elif etype == "error":
            error = event.get("error") or {}
            error_type = error.get("type")
            raise ClaudeAPIError(
                f"Claude stream failed ({error_type}): {error.get('message')}",
                # An error that arrives mid-stream is the server's, unless typed.
                status_code=_ERROR_TYPE_STATUS.get(error_type, 500),
                error_type=error_type,
            )
        return []

    def finish(self) -> None:
        if not self._stopped:
            raise ClaudeAPIError(
                "Claude stream ended before message_stop",
                status_code=None,
                error_type="incomplete_stream",
            )

    def _block_start(
        self, block: Mapping[str, Any]
    ) -> list[tuple[AIMessageChunk, str | None]]:
        btype = block.get("type")
        if btype == "tool_use":
            chunk = AIMessageChunk(
                content="",
                tool_call_chunks=[
                    ToolCallChunk(
                        name=block.get("name"),
                        args="",
                        id=block.get("id"),
                        index=self._index,
                    )
                ],
            )
            return [(chunk, "")]
        if btype == "thinking" and block.get("thinking"):
            return [(self._reasoning(reasoning=block["thinking"]), None)]
        if btype == "redacted_thinking":
            return [(self._reasoning(redacted=block.get("data", "")), None)]
        return []

    def _block_delta(
        self, delta: Mapping[str, Any]
    ) -> list[tuple[AIMessageChunk, str | None]]:
        dtype = delta.get("type")
        if dtype == "thinking_delta":
            text = delta.get("thinking", "")
            return [(self._reasoning(reasoning=text), text)]
        if dtype == "signature_delta":
            return [(self._reasoning(signature=delta.get("signature", "")), None)]
        if dtype == "text_delta":
            text = delta.get("text", "")
            chunk = AIMessageChunk(
                content=[{"type": "text", "text": text, "index": self._index}]
            )
            return [(chunk, text)]
        if dtype == "input_json_delta":
            chunk = AIMessageChunk(
                content="",
                tool_call_chunks=[
                    ToolCallChunk(
                        name=None,
                        args=delta.get("partial_json", ""),
                        id=None,
                        index=self._index,
                    )
                ],
            )
            return [(chunk, None)]
        return []

    def _reasoning(self, **fields: str) -> AIMessageChunk:
        return AIMessageChunk(
            content=[{"type": "reasoning", **fields, "index": self._index}]
        )
