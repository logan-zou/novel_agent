import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from src.llm import OpenAICompatibleProvider
from src.harness.runtime import ModelRequest
from src.harness.usage import extract_usage


class _FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        fragments = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None,
                            reasoning_content="先检索",
                            tool_calls=None,
                        ),
                        finish_reason=None,
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content="完",
                            reasoning_content="实体。",
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id="call_1",
                                    type="function",
                                    function=SimpleNamespace(
                                        name="read_", arguments='{"ref":'
                                    ),
                                )
                            ],
                        ),
                        finish_reason=None,
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content="成",
                            reasoning_content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id=None,
                                    type=None,
                                    function=SimpleNamespace(
                                        name="entity", arguments='"角色/a"}'
                                    ),
                                )
                            ],
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                choices=[],
                usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            ),
        ]
        return iter(fragments)


class _FakeClient:
    def __init__(self):
        self.completions = _FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


class LlmConfigurationTests(unittest.TestCase):
    def test_sdk_retries_are_disabled_for_default_and_injected_clients(self):
        client = OpenAI(api_key="test", base_url="https://example.com", max_retries=9)
        try:
            for injected in (None, client):
                provider = OpenAICompatibleProvider(api_key="test", base_url="https://example.com", client=injected)
                self.assertEqual(provider.client.max_retries, 0)
                provider.client.close()
        finally:
            client.close()

    def test_http_transient_errors_retry_but_bad_requests_do_not(self):
        for status in (400, 401, 429, 503):
            with self.subTest(status=status):
                response = httpx.Response(status, request=httpx.Request("POST", "https://example.com"))
                error = APIStatusError("error", response=response, body=None)
                client = _FakeClient()
                client.completions.create = Mock(side_effect=error)
                provider = OpenAICompatibleProvider(api_key="test", base_url="https://example.com", client=client)
                with patch("src.llm.time.sleep"), patch("src.llm.logger"), self.assertRaises(APIStatusError):
                    provider.complete(self.request())
                self.assertEqual(client.completions.create.call_count, 6 if status in (429, 503) else 1)

    def test_mixed_connection_and_stream_failures_then_success(self):
        def broken():
            yield from _FakeCompletions().create()
            raise httpx.RemoteProtocolError("incomplete chunked read")
        client = _FakeClient()
        client.completions.create = Mock(side_effect=[
            APIConnectionError(request=httpx.Request("POST", "https://example.com")),
            broken(), _FakeCompletions().create(),
        ])
        provider = OpenAICompatibleProvider(api_key="test", base_url="https://example.com", client=client)
        with patch("src.llm.time.sleep") as sleep, self.assertLogs("src.llm"):
            response = provider.complete(self.request())
        self.assertEqual(response.message.content, "完成")
        self.assertEqual(client.completions.create.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [2, 4])

    def request(self, **kwargs):
        return ModelRequest(model="test-model", messages=[{"role": "user", "content": "测试"}],
                            tools=[], max_tokens=1000, temperature=0.2, **kwargs)

    def test_stream_timeout_discards_partial_response_and_retries_same_request(self):
        closed = []

        def broken_stream():
            try:
                # 包含部分文本、推理和工具参数；这些不能混入成功响应。
                fragments = _FakeCompletions().create()
                yield next(fragments)
                yield next(fragments)
                raise httpx.RemoteProtocolError("peer closed connection without sending complete message body (incomplete chunked read)")
            finally:
                closed.append(True)

        client = _FakeClient()
        client.completions.create = Mock(side_effect=[broken_stream(), _FakeCompletions().create()])
        provider = OpenAICompatibleProvider(api_key="test", base_url="https://example.com", client=client)
        with patch("src.llm.time.sleep") as sleep, self.assertLogs("src.llm", level="WARNING") as logs:
            response = provider.complete(self.request())
        self.assertEqual(closed, [True])
        self.assertEqual(client.completions.create.call_count, 2)
        self.assertEqual(client.completions.create.call_args_list[0], client.completions.create.call_args_list[1])
        sleep.assert_called_once_with(2)
        self.assertIn("1/5", logs.output[0])
        self.assertEqual(response.message.content, "完成")
        self.assertEqual(response.message.reasoning_content, "先检索实体。")
        self.assertEqual(len(response.message.tool_calls), 1)
        self.assertEqual(response.message.tool_calls[0].name, "read_entity")
        self.assertEqual(response.message.tool_calls[0].arguments, '{"ref":"角色/a"}')
        self.assertEqual(response.usage["total_tokens"], 120)

    def test_stream_retry_limit_and_timeout_types(self):
        for error in [httpx.ReadTimeout("timed out"), TimeoutError("timed out"),
                      httpx.RemoteProtocolError("incomplete chunked read"),
                      httpx.ReadError("connection reset"), ConnectionResetError("reset"),
                      APIConnectionError(request=httpx.Request("POST", "https://example.com")),
                      APITimeoutError(request=httpx.Request("POST", "https://example.com"))]:
            with self.subTest(error=type(error).__name__):
                streams = []

                def create(**kwargs):
                    stream = Mock()
                    stream.__iter__ = Mock(return_value=stream)
                    stream.__next__ = Mock(side_effect=error)
                    streams.append(stream)
                    return stream

                client = _FakeClient()
                client.completions.create = Mock(side_effect=create)
                provider = OpenAICompatibleProvider(api_key="test", base_url="https://example.com", client=client)
                with patch("src.llm.time.sleep") as sleep, self.assertLogs("src.llm"), self.assertRaises(type(error)) as caught:
                    provider.complete(self.request())
                self.assertIs(caught.exception, error)
                self.assertEqual(len(streams), 6)
                self.assertEqual([call.args[0] for call in sleep.call_args_list], [2, 4, 8, 16, 30])
                for stream in streams:
                    stream.close.assert_called_once()

    def test_callback_timeout_is_not_retried_and_stream_is_closed(self):
        closed = []

        def stream():
            try:
                yield from _FakeCompletions().create()
            finally:
                closed.append(True)

        client = _FakeClient()
        client.completions.create = Mock(return_value=stream())
        provider = OpenAICompatibleProvider(api_key="test", base_url="https://example.com", client=client)
        with patch("src.llm.time.sleep") as sleep, self.assertRaises(TimeoutError):
            provider.complete(self.request(on_content_delta=Mock(side_effect=TimeoutError("callback failed"))))
        client.completions.create.assert_called_once()
        sleep.assert_not_called()
        self.assertEqual(closed, [True])

    def test_request_creation_errors_share_five_retry_budget(self):
        client = _FakeClient()
        client.completions.create = Mock(side_effect=APITimeoutError(request=httpx.Request("POST", "https://example.com")))
        provider = OpenAICompatibleProvider(api_key="test", base_url="https://example.com", client=client)
        with patch("src.llm.time.sleep") as sleep, self.assertLogs("src.llm"), self.assertRaises(APITimeoutError):
            provider.complete(self.request())
        self.assertEqual(client.completions.create.call_count, 6)
        self.assertEqual(sleep.call_count, 5)

    def test_zero_stream_retries_and_non_timeout_errors(self):
        for error in [httpx.ReadTimeout("timed out"), ValueError("invalid data")]:
            with self.subTest(error=type(error).__name__):
                def broken():
                    yield from ()
                    raise error

                client = _FakeClient()
                client.completions.create = Mock(return_value=broken())
                provider = OpenAICompatibleProvider(api_key="test", base_url="https://example.com", client=client,
                                                    stream_max_retries=0)
                with patch("src.llm.time.sleep") as sleep, patch("src.llm.logger"), self.assertRaises(type(error)):
                    provider.complete(self.request())
                client.completions.create.assert_called_once()
                sleep.assert_not_called()

    def test_optional_reasoning_parameters_and_streaming_response(self):
        client = _FakeClient()
        provider = OpenAICompatibleProvider(
            api_key="test-key",
            base_url="https://example.com/v1",
            client=client,
            include_reasoning_effort=True,
            enable_thinking=True,
        )
        reasoning_deltas = []
        content_deltas = []

        response = provider.complete(
            ModelRequest(
                model="test-model",
                messages=[{"role": "user", "content": "测试"}],
                tools=[{"type": "function", "function": {"name": "read_entity"}}],
                max_tokens=16_384,
                temperature=0.5,
                reasoning_effort="high",
                on_reasoning_delta=reasoning_deltas.append,
                on_content_delta=content_deltas.append,
            )
        )

        kwargs = client.completions.kwargs
        self.assertEqual(kwargs["model"], "test-model")
        self.assertTrue(kwargs["stream"])
        self.assertTrue(kwargs["stream_options"]["include_usage"])
        self.assertEqual(kwargs["max_tokens"], 16_384)
        self.assertEqual(kwargs["reasoning_effort"], "high")
        self.assertTrue(kwargs["extra_body"]["chat_template_kwargs"]["thinking"])
        self.assertTrue(kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(response.message.reasoning_content, "先检索实体。")
        self.assertEqual(response.message.content, "完成")
        self.assertEqual(reasoning_deltas, ["先检索", "实体。"])
        self.assertEqual(content_deltas, ["完", "成"])
        self.assertEqual(response.finish_reason, "tool_calls")
        self.assertEqual(response.message.tool_calls[0].id, "call_1")
        self.assertEqual(response.message.tool_calls[0].name, "read_entity")
        self.assertEqual(response.message.tool_calls[0].arguments, '{"ref":"角色/a"}')
        normalized = extract_usage(response.usage)
        self.assertEqual(normalized["cached_prompt_tokens"], 0)
        self.assertEqual(normalized["uncached_prompt_tokens"], 0)
        self.assertEqual(normalized["unknown_cache_prompt_tokens"], 100)

    def test_generic_openai_compatible_provider_omits_maas_only_options(self):
        client = _FakeClient()
        provider = OpenAICompatibleProvider(
            api_key="user-key",
            base_url="https://example.com/v1",
            client=client,
        )
        provider.complete(
            ModelRequest(
                model="user-model",
                messages=[{"role": "user", "content": "测试"}],
                tools=[],
                max_tokens=1000,
                temperature=0.2,
                reasoning_effort="high",
            )
        )

        kwargs = client.completions.kwargs
        self.assertEqual(kwargs["model"], "user-model")
        self.assertNotIn("reasoning_effort", kwargs)
        self.assertNotIn("extra_body", kwargs)


if __name__ == "__main__":
    unittest.main()
