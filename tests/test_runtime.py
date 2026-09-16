import unittest
from copy import deepcopy
from dataclasses import replace

from src.harness.runtime import (
    AgentConfig,
    AgentLimitError,
    AgentLoopStalled,
    AgentRuntime,
    AssistantMessage,
    MalformedToolArguments,
    MaxIterationsExceeded,
    ModelResponse,
    Tool,
    ToolCall,
    trim_old_tool_results,
)


class FakeProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.message_snapshots = []

    def complete(self, request):
        self.requests.append(request)
        self.message_snapshots.append(deepcopy(request.messages))
        return self.responses.pop(0)


def config():
    return AgentConfig("test", "model", "system", 5, 1000, 0, 10_000)


class RuntimeTests(unittest.TestCase):
    def test_routes_tools_and_accumulates_cached_usage(self):
        provider = FakeProvider(
            [
                ModelResponse(
                    AssistantMessage(
                        tool_calls=(ToolCall("1", "echo", '{"text":"hi"}'),),
                        reasoning_content="先调用工具。",
                    ),
                    "tool_calls",
                    {
                        "prompt_tokens": 100,
                        "completion_tokens": 10,
                        "total_tokens": 110,
                        "prompt_tokens_details": {"cached_tokens": 70},
                    },
                ),
                ModelResponse(
                    AssistantMessage("done"),
                    "stop",
                    {
                        "prompt_tokens": 120,
                        "completion_tokens": 5,
                        "total_tokens": 125,
                        "prompt_tokens_details": {"cached_tokens": 100},
                    },
                ),
            ]
        )
        events = []
        recorded_usage = []
        result = AgentRuntime(provider).run(
            config(),
            [{"role": "user", "content": "go"}],
            [Tool("echo", "echo", {"type": "object"}, lambda text: text.upper())],
            on_event=lambda kind, data: events.append((kind, data)),
            usage_recorder=recorded_usage.append,
        )

        self.assertEqual(result.output, "done")
        # 忽略 API 声称的 70/100 缓存命中：首轮 100 全部非缓存，
        # 第二轮以上一轮 100 为缓存前缀，新增 20 为非缓存。
        self.assertEqual(result.usage["cached_prompt_tokens"], 100)
        self.assertEqual(result.usage["uncached_prompt_tokens"], 120)
        self.assertEqual(result.usage["unknown_cache_prompt_tokens"], 0)
        self.assertEqual(recorded_usage[0]["uncached_prompt_tokens"], 100)
        self.assertEqual(recorded_usage[1]["cached_prompt_tokens"], 100)
        self.assertEqual(recorded_usage[1]["uncached_prompt_tokens"], 20)
        self.assertEqual(provider.requests[1].messages[-1]["content"], "HI")
        self.assertIn("tool_started", [kind for kind, _ in events])
        self.assertIn("tool_completed", [kind for kind, _ in events])
        started = [data for kind, data in events if kind == "tool_started"][0]
        finished = [data for kind, data in events if kind == "tool_completed"][0]
        self.assertEqual(started["call_id"], "1")
        self.assertEqual(started["arguments"], {"text": "hi"})
        self.assertEqual(finished["result"], "HI")
        self.assertLess(
            [kind for kind, _ in events].index("tool_started"),
            [kind for kind, _ in events].index("tool_completed"),
        )
        completed = [data for kind, data in events if kind == "model_completed"]
        self.assertEqual(completed[0]["reasoning_content"], "先调用工具。")
        self.assertEqual(
            provider.requests[1].messages[-2]["reasoning_content"], "先调用工具。"
        )

    def test_forwards_reasoning_and_answer_deltas_with_step(self):
        class StreamingProvider:
            def complete(self, request):
                request.on_reasoning_delta("正在检索")
                request.on_content_delta("最终")
                request.on_content_delta("答案")
                return ModelResponse(
                    AssistantMessage("最终答案", reasoning_content="正在检索"),
                    "stop",
                )

        events = []
        result = AgentRuntime(StreamingProvider()).run(
            config(),
            [{"role": "user", "content": "go"}],
            on_event=lambda kind, data: events.append((kind, data)),
        )

        self.assertEqual(result.output, "最终答案")
        reasoning = [data for kind, data in events if kind == "model_reasoning_delta"]
        content = [data for kind, data in events if kind == "model_content_delta"]
        self.assertEqual(reasoning, [{"run_id": result.run_id, "step": 1, "delta": "正在检索"}])
        self.assertEqual([item["delta"] for item in content], ["最终", "答案"])
        self.assertTrue(all(item["step"] == 1 for item in content))

    def test_recovers_truncation_and_retries_invalid_output(self):
        provider = FakeProvider(
            [
                ModelResponse(AssistantMessage("part"), "length"),
                ModelResponse(AssistantMessage("bad"), "stop"),
                ModelResponse(AssistantMessage("good [EV001]"), "stop"),
            ]
        )
        result = AgentRuntime(provider).run(
            config(),
            [{"role": "user", "content": "go"}],
            output_validator=lambda value: None if "EV001" in value else "missing evidence",
        )
        self.assertEqual(result.output, "good [EV001]")
        self.assertIn("missing evidence", provider.requests[2].messages[-1]["content"])

    def test_context_trim_keeps_recent_results_without_artifacts(self):
        messages = [
            {"role": "tool", "tool_call_id": str(index), "content": "x" * 300}
            for index in range(9)
        ]
        count = trim_old_tool_results(messages, 7000, 10_000, keep_recent=3)
        self.assertEqual(count, 6)
        self.assertTrue(messages[0]["content"].startswith("[旧工具结果已省略"))
        self.assertEqual(messages[-1]["content"], "x" * 300)

    def test_soft_limit_extends_while_write_progress_continues(self):
        responses = [
            ModelResponse(
                AssistantMessage(tool_calls=(ToolCall(str(index), "write", f'{{"value":{index}}}'),)),
                "tool_calls",
            )
            for index in range(1, 5)
        ]
        responses.append(ModelResponse(AssistantMessage("done"), "stop"))
        provider = FakeProvider(responses)
        result = AgentRuntime(provider).run(
            replace(
                config(),
                max_iterations=5,
                soft_iteration_limit=2,
                progress_grace_iterations=2,
            ),
            [{"role": "user", "content": "go"}],
            [
                Tool(
                    "write",
                    "write",
                    {"type": "object"},
                    lambda value: f"saved {value}",
                    progress_detector=lambda result: result.startswith("saved"),
                )
            ],
        )

        self.assertEqual(result.output, "done")
        self.assertEqual(result.steps, 5)

    def test_soft_limit_stops_when_no_write_progress_exists(self):
        provider = FakeProvider(
            [
                ModelResponse(
                    AssistantMessage(tool_calls=(ToolCall(str(index), "read", f'{{"value":{index}}}'),)),
                    "tool_calls",
                )
                for index in range(1, 3)
            ]
        )
        with self.assertRaises(MaxIterationsExceeded) as raised:
            AgentRuntime(provider).run(
                replace(
                    config(),
                    max_iterations=5,
                    soft_iteration_limit=2,
                    progress_grace_iterations=1,
                ),
                [{"role": "user", "content": "go"}],
                [Tool("read", "read", {"type": "object"}, lambda value: str(value))],
            )

        self.assertEqual(raised.exception.steps, 2)
        self.assertEqual(raised.exception.tool_counts, {"read": 2})
        self.assertIn("最近 1 轮没有写入进展", str(raised.exception))

    def test_repeated_identical_tool_call_is_stopped(self):
        response = ModelResponse(
            AssistantMessage(tool_calls=(ToolCall("same", "read", '{"value":1}'),)),
            "tool_calls",
        )
        provider = FakeProvider([response, response])
        with self.assertRaises(AgentLoopStalled) as raised:
            AgentRuntime(provider).run(
                replace(config(), max_iterations=5, max_identical_tool_calls=2),
                [{"role": "user", "content": "go"}],
                [Tool("read", "read", {"type": "object"}, lambda value: str(value))],
            )

        self.assertEqual(raised.exception.repeated_tool_calls, 1)
        self.assertIn("read 2次", str(raised.exception))

    def test_retries_malformed_tool_arguments_without_polluting_history(self):
        provider = FakeProvider(
            [
                ModelResponse(
                    AssistantMessage(
                        tool_calls=(
                            ToolCall("broken", "echo", '{"text":"缺少结尾"'),
                            ToolCall("sibling", "echo", '{"text":"本身合法"}'),
                        )
                    ),
                    "tool_calls",
                ),
                ModelResponse(
                    AssistantMessage(
                        tool_calls=(ToolCall("fixed", "echo", '{"text":"已修复"}'),)
                    ),
                    "tool_calls",
                ),
                ModelResponse(AssistantMessage("done"), "stop"),
            ]
        )
        executed = []
        events = []

        result = AgentRuntime(provider).run(
            config(),
            [{"role": "user", "content": "go"}],
            [Tool("echo", "echo", {"type": "object"}, lambda text: executed.append(text) or text)],
            on_event=lambda kind, data: events.append((kind, data)),
        )

        self.assertEqual(result.output, "done")
        self.assertEqual(executed, ["已修复"])
        self.assertEqual(provider.message_snapshots[1][-1]["role"], "user")
        self.assertIn("本轮没有执行任何工具", provider.message_snapshots[1][-1]["content"])
        polluted_ids = {
            call["id"]
            for message in provider.message_snapshots[-1]
            for call in message.get("tool_calls", [])
        }
        self.assertNotIn("broken", polluted_ids)
        self.assertNotIn("sibling", polluted_ids)
        self.assertIn("fixed", polluted_ids)
        rejected = [data for kind, data in events if kind == "tool_arguments_rejected"]
        self.assertEqual(len(rejected), 1)
        self.assertTrue(rejected[0]["will_retry"])
        started_ids = [data["call_id"] for kind, data in events if kind == "tool_started"]
        self.assertEqual(started_ids, ["fixed"])

    def test_persistent_malformed_tool_arguments_raise_splittable_limit_error(self):
        malformed = ModelResponse(
            AssistantMessage(tool_calls=(ToolCall("broken", "write", '{"value":'),)),
            "tool_calls",
        )
        provider = FakeProvider([malformed, malformed, malformed])

        with self.assertRaises(MalformedToolArguments) as raised:
            AgentRuntime(provider).run(
                replace(config(), max_tool_argument_retries=2),
                [{"role": "user", "content": "go"}],
                [Tool("write", "write", {"type": "object"}, lambda value: str(value))],
            )

        self.assertIsInstance(raised.exception, AgentLimitError)
        self.assertEqual(raised.exception.steps, 3)
        self.assertEqual(raised.exception.limit, 2)
        self.assertEqual(len(provider.requests), 3)
        self.assertIn("已用完 2 次修复机会", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
