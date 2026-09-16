import unittest
from types import SimpleNamespace

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
