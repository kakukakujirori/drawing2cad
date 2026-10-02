"""ChatModel wrapper for Anthropic Claude using subscription OAuth credentials.

Targets Anthropic Messages API (`https://api.anthropic.com/v1/messages`) using
an OAuth access token from an active Claude subscription (minted via Claude Code CLI
`claude auth login`). This enables subscription-quota usage without per-token API charges.

Handles:
- Base64 PNG images formatted by LangGraph (converts `image_url` data URLs to Anthropic `image` source)
- Native Anthropic tool calling (`bind_tools`, `tool_use`, `tool_result`)
- SSE streaming with `AIMessageChunk` and `ToolCallChunk` merging
- OAuth token discovery from `~/.claude/.credentials.json` or `CLAUDE_CODE_OAUTH_TOKEN`
- Friendly 429 usage limit and 401 token expiration error messaging
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, override

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
from langchain_core.messages.tool import ToolCallChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableBinding
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

logger = logging.getLogger(__name__)

ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages?beta=true"
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_ANTHROPIC_BETA = (
    "claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14"
)


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


def _image_parts(content: Any) -> list[dict[str, Any]]:
    if not isinstance(content, list):
        return []
    parts = []
    for part in content:
        if isinstance(part, dict):
            if part.get("type") in {"image_url", "image"}:
                parts.append(part)
            elif part.get("type") == "tool_result" and isinstance(part.get("content"), list):
                for subpart in part["content"]:
                    if isinstance(subpart, dict) and subpart.get("type") in {"image_url", "image"}:
                        parts.append(subpart)
    return parts


def _cap_tool_images(messages: list[dict[str, Any]], limit: int) -> None:
    """Keep inputs and newest tool images on the wire, trimming older tool images when capped."""
    originals = sum(
        len(_image_parts(msg.get("content")))
        for msg in messages
        if msg.get("role") != "user" or not any(
            isinstance(p, dict) and p.get("type") == "tool_result"
            for p in (msg.get("content") if isinstance(msg.get("content"), list) else [])
        )
    )
    if originals > limit:
        raise ValueError(
            f"Original image attachments ({originals}) exceed limit ({limit}); none dropped."
        )
    remaining = limit - originals
    for msg in reversed(messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "tool_result":
                subcontent = part.get("content")
                if isinstance(subcontent, list):
                    new_subcontent = []
                    for subpart in subcontent:
                        if isinstance(subpart, dict) and subpart.get("type") == "image":
                            if remaining > 0:
                                remaining -= 1
                                new_subcontent.append(subpart)
                            else:
                                new_subcontent.append({
                                    "type": "text",
                                    "text": "[Previous tool image trimmed to stay within context limit]",
                                })
                        else:
                            new_subcontent.append(subpart)
                    part["content"] = new_subcontent


def _format_messages_for_anthropic(
    messages: Sequence[BaseMessage],
    max_images_per_request: int | None = None,
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
                                parts.append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": b64_data,
                                    },
                                })
                            else:
                                raise ValueError(
                                    f"Unsupported image_url format (expected data URL): {url[:40]}..."
                                )
                        elif item_type == "image":
                            parts.append(item)
                        else:
                            parts.append(item)
            formatted_messages.append({"role": "user", "content": parts})
        elif isinstance(msg, AIMessage):
            parts: list[dict[str, Any]] = []
            if isinstance(msg.content, str) and msg.content:
                parts.append({"type": "text", "text": msg.content})
            elif isinstance(msg.content, list):
                for item in msg.content:
                    if isinstance(item, str):
                        parts.append({"type": "text", "text": item})
                    elif isinstance(item, dict) and item.get("type") == "text":
                        parts.append({"type": "text", "text": item.get("text", "")})
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    parts.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["args"],
                    })
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
                            tool_parts.append({"type": "text", "text": item.get("text", "")})
                        elif itype == "image_url":
                            url = item.get("image_url", {}).get("url", "")
                            if isinstance(url, str) and url.startswith("data:"):
                                header, b64_data = url.split(",", 1)
                                media_type = header.split(";")[0].split(":")[1]
                                tool_parts.append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": b64_data,
                                    },
                                })
                        elif itype == "image":
                            tool_parts.append(item)
                        else:
                            tool_parts.append(item)
                tool_content = tool_parts if tool_parts else str(msg.content)
            else:
                tool_content = str(msg.content)

            formatted_messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": tool_content,
                        "is_error": msg.status == "error",
                    }
                ],
            })
        else:
            formatted_messages.append({
                "role": "user",
                "content": [{"type": "text", "text": str(msg.content)}],
            })

    # Anthropic requires strict role alternation (user -> assistant -> user).
    # Merge consecutive messages sharing the same role.
    merged_messages: list[dict[str, Any]] = []
    for m in formatted_messages:
        if merged_messages and merged_messages[-1]["role"] == m["role"]:
            merged_messages[-1]["content"].extend(m["content"])
        else:
            merged_messages.append(m)

    if max_images_per_request is not None:
        _cap_tool_images(merged_messages, max_images_per_request)

    system_prompt = "\n\n".join(system_parts) if system_parts else None
    return system_prompt, merged_messages


class ChatClaude(BaseChatModel):
    """ChatModel targeting Anthropic Claude using Claude subscription OAuth credentials."""

    model: str = Field(default="claude-sonnet-5", description="Anthropic model identifier")
    max_tokens: int = Field(default=128000, description="Max tokens to generate")
    temperature: float | None = Field(default=None, description="Sampling temperature (0.0 to 1.0)")
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
    max_images_per_request: int | None = Field(
        default=4,
        gt=0,
        description="Optional limit on total image blocks sent per request.",
    )

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
        system_prompt, formatted_messages = _format_messages_for_anthropic(
            messages, max_images_per_request=self.max_images_per_request
        )
        prompt_id = str(uuid.uuid4())
        billing_header = (
            f"x-anthropic-billing-header: cc_version=2.1.284.12a; cc_entrypoint=sdk-cli; "
            f"cch=00000; cc_prompt_id={prompt_id}; cc_turn_origin=sdk; cc_prompt_index=0; cc_turn_index=1;"
        )
        system_blocks: list[dict[str, Any]] = [
            {"type": "text", "text": billing_header}
        ]
        if system_prompt:
            system_blocks.append({"type": "text", "text": system_prompt})

        payload: dict[str, Any] = {
            "model": self._resolved_model_name(),
            "max_tokens": self.max_tokens,
            "messages": formatted_messages,
            "system": system_blocks,
            "stream": stream,
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
                if tool_choice == "auto":
                    payload["tool_choice"] = {"type": "auto"}
                elif tool_choice == "any":
                    payload["tool_choice"] = {"type": "any"}
                elif isinstance(tool_choice, str):
                    payload["tool_choice"] = {"type": "tool", "name": tool_choice}
                elif isinstance(tool_choice, dict):
                    payload["tool_choice"] = tool_choice
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
                content_parts.append({
                    "type": "reasoning",
                    "reasoning": block.get("thinking", ""),
                    "signature": block.get("signature", ""),
                })
            elif btype == "text":
                content_parts.append({
                    "type": "text",
                    "text": block.get("text", ""),
                })
            elif btype == "tool_use":
                tool_calls.append({
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "args": block.get("input", {}),
                    "type": "tool_call",
                })

        ai_msg = AIMessage(
            content=content_parts if content_parts else "",
            tool_calls=tool_calls,
            response_metadata={
                "id": data.get("id"),
                "model": data.get("model"),
                "stop_reason": data.get("stop_reason"),
                "usage": data.get("usage"),
            },
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

                current_block_index: int = 0
                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[len("data: "):].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    etype = event.get("type")
                    if etype == "content_block_start":
                        current_block_index = event.get("index", 0)
                        block = event.get("content_block", {})
                        btype = block.get("type")
                        if btype == "tool_use":
                            chunk = AIMessageChunk(
                                content="",
                                tool_call_chunks=[
                                    ToolCallChunk(
                                        name=block.get("name"),
                                        args="",
                                        id=block.get("id"),
                                        index=current_block_index,
                                    )
                                ],
                            )
                            if run_manager:
                                run_manager.on_llm_new_token(
                                    "",
                                    chunk=ChatGenerationChunk(message=chunk),
                                )
                            yield ChatGenerationChunk(message=chunk)
                        elif btype == "thinking" and block.get("thinking"):
                            chunk = AIMessageChunk(
                                content=[{
                                    "type": "reasoning",
                                    "reasoning": block.get("thinking", ""),
                                    "index": current_block_index,
                                }]
                            )
                            yield ChatGenerationChunk(message=chunk)
                    elif etype == "content_block_delta":
                        delta = event.get("delta", {})
                        dtype = delta.get("type")
                        if dtype == "thinking_delta":
                            text = delta.get("thinking", "")
                            chunk = AIMessageChunk(
                                content=[{
                                    "type": "reasoning",
                                    "reasoning": text,
                                    "index": current_block_index,
                                }]
                            )
                            if run_manager:
                                run_manager.on_llm_new_token(
                                    text,
                                    chunk=ChatGenerationChunk(message=chunk),
                                )
                            yield ChatGenerationChunk(message=chunk)
                        elif dtype == "signature_delta":
                            sig = delta.get("signature", "")
                            chunk = AIMessageChunk(
                                content=[{
                                    "type": "reasoning",
                                    "signature": sig,
                                    "index": current_block_index,
                                }]
                            )
                            yield ChatGenerationChunk(message=chunk)
                        elif dtype == "text_delta":
                            text = delta.get("text", "")
                            chunk = AIMessageChunk(
                                content=[{
                                    "type": "text",
                                    "text": text,
                                    "index": current_block_index,
                                }]
                            )
                            if run_manager:
                                run_manager.on_llm_new_token(
                                    text,
                                    chunk=ChatGenerationChunk(message=chunk),
                                )
                            yield ChatGenerationChunk(message=chunk)
                        elif dtype == "input_json_delta":
                            partial_json = delta.get("partial_json", "")
                            chunk = AIMessageChunk(
                                content="",
                                tool_call_chunks=[
                                    ToolCallChunk(
                                        name=None,
                                        args=partial_json,
                                        id=None,
                                        index=current_block_index,
                                    )
                                ],
                            )
                            yield ChatGenerationChunk(message=chunk)
                    elif etype == "message_delta":
                        usage = event.get("usage", {})
                        stop_reason = event.get("delta", {}).get("stop_reason")
                        chunk = AIMessageChunk(
                            content="",
                            response_metadata={"stop_reason": stop_reason, "usage": usage},
                        )
                        yield ChatGenerationChunk(message=chunk)

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

                current_block_index: int = 0
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[len("data: "):].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    etype = event.get("type")
                    if etype == "content_block_start":
                        current_block_index = event.get("index", 0)
                        block = event.get("content_block", {})
                        btype = block.get("type")
                        if btype == "tool_use":
                            chunk = AIMessageChunk(
                                content="",
                                tool_call_chunks=[
                                    ToolCallChunk(
                                        name=block.get("name"),
                                        args="",
                                        id=block.get("id"),
                                        index=current_block_index,
                                    )
                                ],
                            )
                            if run_manager:
                                await run_manager.on_llm_new_token(
                                    "",
                                    chunk=ChatGenerationChunk(message=chunk),
                                )
                            yield ChatGenerationChunk(message=chunk)
                        elif btype == "thinking" and block.get("thinking"):
                            chunk = AIMessageChunk(
                                content=[{
                                    "type": "reasoning",
                                    "reasoning": block.get("thinking", ""),
                                    "index": current_block_index,
                                }]
                            )
                            yield ChatGenerationChunk(message=chunk)
                    elif etype == "content_block_delta":
                        delta = event.get("delta", {})
                        dtype = delta.get("type")
                        if dtype == "thinking_delta":
                            text = delta.get("thinking", "")
                            chunk = AIMessageChunk(
                                content=[{
                                    "type": "reasoning",
                                    "reasoning": text,
                                    "index": current_block_index,
                                }]
                            )
                            if run_manager:
                                await run_manager.on_llm_new_token(
                                    text,
                                    chunk=ChatGenerationChunk(message=chunk),
                                )
                            yield ChatGenerationChunk(message=chunk)
                        elif dtype == "signature_delta":
                            sig = delta.get("signature", "")
                            chunk = AIMessageChunk(
                                content=[{
                                    "type": "reasoning",
                                    "signature": sig,
                                    "index": current_block_index,
                                }]
                            )
                            yield ChatGenerationChunk(message=chunk)
                        elif dtype == "text_delta":
                            text = delta.get("text", "")
                            chunk = AIMessageChunk(
                                content=[{
                                    "type": "text",
                                    "text": text,
                                    "index": current_block_index,
                                }]
                            )
                            if run_manager:
                                await run_manager.on_llm_new_token(
                                    text,
                                    chunk=ChatGenerationChunk(message=chunk),
                                )
                            yield ChatGenerationChunk(message=chunk)
                        elif dtype == "input_json_delta":
                            partial_json = delta.get("partial_json", "")
                            chunk = AIMessageChunk(
                                content="",
                                tool_call_chunks=[
                                    ToolCallChunk(
                                        name=None,
                                        args=partial_json,
                                        id=None,
                                        index=current_block_index,
                                    )
                                ],
                            )
                            yield ChatGenerationChunk(message=chunk)
                    elif etype == "message_delta":
                        usage = event.get("usage", {})
                        stop_reason = event.get("delta", {}).get("stop_reason")
                        chunk = AIMessageChunk(
                            content="",
                            response_metadata={"stop_reason": stop_reason, "usage": usage},
                        )
                        yield ChatGenerationChunk(message=chunk)

    def _handle_error(self, resp: httpx.Response) -> None:
        try:
            err = resp.json()
            err_msg = err.get("error", {}).get("message", resp.text)
        except Exception:
            err_msg = resp.text

        if resp.status_code == 429:
            raise RuntimeError(
                f"Claude subscription usage limit / rate limit exceeded (HTTP 429): {err_msg}"
            )
        elif resp.status_code == 401:
            raise RuntimeError(
                f"Claude OAuth token expired or unauthorized (HTTP 401): {err_msg}. "
                f"Please run 'claude auth login' in terminal to refresh credentials."
            )
        else:
            raise RuntimeError(
                f"Anthropic API request failed (HTTP {resp.status_code}): {err_msg}"
            )
