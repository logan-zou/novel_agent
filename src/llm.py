"""OpenAI-compatible Chat Completions 模型接入。"""

from __future__ import annotations

from openai import OpenAI

from src.config import (
    ConfigurationError,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_ENABLE_THINKING,
    LLM_INCLUDE_REASONING_EFFORT,
)
from src.harness.runtime import AssistantMessage, ChatProvider, ModelRequest, ModelResponse, ToolCall


class OpenAICompatibleProvider:
    """通用 OpenAI Chat Completions 适配器。

    ``reasoning_effort`` 和 ``thinking`` 都是兼容服务的可选扩展，只有在配置中
    显式开启才发送。返回中的 ``reasoning_content`` 若存在会继续透传。
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        client: OpenAI | None = None,
        include_reasoning_effort: bool = False,
        enable_thinking: bool = False,
        extra_body: dict | None = None,
    ) -> None:
        self.client = client or OpenAI(api_key=api_key, base_url=base_url)
        self.include_reasoning_effort = include_reasoning_effort
        self.enable_thinking = enable_thinking
        self.extra_body = dict(extra_body or {})

    def complete(self, request: ModelRequest) -> ModelResponse:
        kwargs = {
            "model": request.model,
            "messages": request.messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        if self.include_reasoning_effort and request.reasoning_effort:
            kwargs["reasoning_effort"] = request.reasoning_effort

        extra_body = dict(self.extra_body)
        if self.enable_thinking:
            chat_template = dict(extra_body.get("chat_template_kwargs") or {})
            chat_template.update({"thinking": True, "enable_thinking": True})
            extra_body["chat_template_kwargs"] = chat_template
        if extra_body:
            kwargs["extra_body"] = extra_body
        if request.tools:
            kwargs["tools"] = request.tools
        stream = self.client.chat.completions.create(**kwargs)
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        calls_by_index: dict[int, dict[str, str]] = {}
        finish_reason = "stop"
        usage = None

        for chunk in stream:
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = chunk_usage
            for choice in getattr(chunk, "choices", ()) or ():
                if getattr(choice, "finish_reason", None):
                    finish_reason = choice.finish_reason
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue

                reasoning = _reasoning_content(delta)
                if reasoning:
                    reasoning_parts.append(reasoning)
                    if request.on_reasoning_delta:
                        request.on_reasoning_delta(reasoning)

                content = _text_content(getattr(delta, "content", None))
                if content:
                    content_parts.append(content)
                    if request.on_content_delta:
                        request.on_content_delta(content)

                for fragment in getattr(delta, "tool_calls", None) or ():
                    index = int(getattr(fragment, "index", 0) or 0)
                    call = calls_by_index.setdefault(
                        index,
                        {"id": "", "name": "", "arguments": "", "type": "function"},
                    )
                    if getattr(fragment, "id", None):
                        call["id"] = fragment.id
                    if getattr(fragment, "type", None):
                        call["type"] = fragment.type
                    function = getattr(fragment, "function", None)
                    if function is not None:
                        call["name"] += getattr(function, "name", None) or ""
                        call["arguments"] += getattr(function, "arguments", None) or ""

        calls = tuple(
            ToolCall(
                value["id"] or f"call_{index}",
                value["name"],
                value["arguments"],
                value["type"],
            )
            for index, value in sorted(calls_by_index.items())
        )
        return ModelResponse(
            AssistantMessage("".join(content_parts), calls, "".join(reasoning_parts)),
            finish_reason,
            usage,
        )


_provider: ChatProvider | None = None


def get_provider() -> ChatProvider:
    """按公开环境配置创建并缓存默认 Provider。"""
    global _provider
    if _provider is None:
        if not LLM_API_KEY:
            raise ConfigurationError("缺少 LLM_API_KEY：请复制 .env.example 为 .env 后填写")
        if not LLM_BASE_URL:
            raise ConfigurationError("缺少 LLM_BASE_URL：请填写 OpenAI-compatible API 地址")
        _provider = OpenAICompatibleProvider(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            include_reasoning_effort=LLM_INCLUDE_REASONING_EFFORT,
            enable_thinking=LLM_ENABLE_THINKING,
        )
    return _provider


def _reasoning_content(message) -> str:
    value = getattr(message, "reasoning_content", None)
    if value is None:
        extra = getattr(message, "model_extra", None)
        if isinstance(extra, dict):
            value = extra.get("reasoning_content")
    return str(value or "")


def _text_content(value) -> str:
    """兼容文本 delta，以及少数兼容接口返回的 content parts。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or ""))
            else:
                parts.append(str(getattr(item, "text", "") or ""))
        return "".join(parts)
    return str(value)
