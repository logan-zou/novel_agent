"""OpenAI-compatible Chat Completions 模型接入。"""

from __future__ import annotations

import logging
import time

import httpx
from openai import APIConnectionError, APIStatusError, OpenAI

from src.config import (
    ConfigurationError,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_ENABLE_THINKING,
    LLM_INCLUDE_REASONING_EFFORT,
    LLM_STREAM_MAX_RETRIES,
)
from src.harness.runtime import AssistantMessage, ChatProvider, ModelRequest, ModelResponse, ToolCall


logger = logging.getLogger(__name__)


class _StreamTransportError(Exception):
    """只标识读取流时的传输错误，避免重试回调中的业务异常。"""


def _stream_chunks(stream):
    iterator = iter(stream)
    while True:
        try:
            chunk = next(iterator)
        except StopIteration:
            return
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError,
                APIConnectionError, TimeoutError, ConnectionError) as error:
            raise _StreamTransportError() from error
        yield chunk


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
        stream_max_retries: int = 5,
    ) -> None:
        if stream_max_retries < 0:
            raise ConfigurationError("LLM_STREAM_MAX_RETRIES 不能小于 0")
        self.stream_max_retries = stream_max_retries
        self.client = (client.with_options(max_retries=0) if isinstance(client, OpenAI) else client) or OpenAI(
            api_key=api_key, base_url=base_url, max_retries=0
        )
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
        for attempt in range(self.stream_max_retries + 1):
            stream = None
            try:
                stream = self._create_stream(kwargs)
                return self._consume_stream(stream, request)
            except _StreamTransportError as error:
                failure_type = type(error.__cause__).__name__
                if attempt == self.stream_max_retries:
                    logger.error("[模型流传输失败] %s：已用完 %s 次重试，当前调用失败", failure_type, self.stream_max_retries)
                    raise error.__cause__ from None
            finally:
                close = getattr(stream, "close", None)
                if close is not None:
                    try:
                        close()
                    except Exception as error:
                        logger.warning("[模型流关闭失败] %s", type(error).__name__)
            delay = min(2 ** (attempt + 1), 30)
            logger.warning(
                "[模型流传输失败] %s：%s 秒后重试当前调用（%s/%s）；丢弃本次未完成的文本和工具参数",
                failure_type, delay, attempt + 1, self.stream_max_retries,
            )
            time.sleep(delay)

        raise AssertionError("unreachable")

    def _create_stream(self, kwargs):
        try:
            return self.client.chat.completions.create(**kwargs)
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError,
                APIConnectionError, TimeoutError, ConnectionError) as error:
            raise _StreamTransportError() from error
        except APIStatusError as error:
            if error.status_code in {408, 409, 429} or error.status_code >= 500:
                raise _StreamTransportError() from error
            raise

    def _consume_stream(self, stream, request: ModelRequest) -> ModelResponse:
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        calls_by_index: dict[int, dict[str, str]] = {}
        finish_reason = "stop"
        usage = None

        for chunk in _stream_chunks(stream):
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
            stream_max_retries=LLM_STREAM_MAX_RETRIES,
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
