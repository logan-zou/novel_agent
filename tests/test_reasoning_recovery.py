"""用严格离线 Provider 校验恢复请求保留真实的 assistant 思考。"""

import unittest
from copy import deepcopy

from src.harness.runtime import (
    AgentConfig,
    AgentRuntime,
    AssistantMessage,
    ModelResponse,
    Tool,
    ToolCall,
)


class StrictReasoningProvider:
    def __init__(self, responses, *, require_reasoning=True):
        self.responses = iter(responses)
        self.require_reasoning = require_reasoning
        self.returned = []
        self.histories = []

    def complete(self, request):
        history = deepcopy(request.messages)
        self.histories.append(history)
        for message in history:
            if message["role"] != "assistant":
                continue
            if self.require_reasoning and not message.get("reasoning_content"):
                raise AssertionError("Provider 拒绝缺少 reasoning_content 的 assistant")
            actual_calls = [call["id"] for call in message.get("tool_calls", [])]
            if not any(
                message["content"] == original.content
                and message.get("reasoning_content", "") == original.reasoning_content
                and actual_calls == [call.id for call in original.tool_calls]
                for original in self.returned
            ):
                raise AssertionError("历史 assistant 内容或思考与真实模型响应不一致")
        response = next(self.responses)
        self.returned.append(response.message)
        return response


def config():
    return AgentConfig("test", "model", "system", 5, 1000, 0, 10_000)


def response(text, reasoning, finish_reason="stop"):
    return ModelResponse(AssistantMessage(text, reasoning_content=reasoning), finish_reason)


class ReasoningRecoveryTests(unittest.TestCase):
    def test_validation_repair_preserves_reasoning_before_tool_lookup(self):
        provider = StrictReasoningProvider([
            response("无引用回答", "先尝试已有信息"),
            ModelResponse(AssistantMessage(
                tool_calls=(ToolCall("lookup", "read", "{}"),),
                reasoning_content="需要检索缺失的原文证据",
            ), "tool_calls"),
            response("有证据的回答 [EV001]", "根据新证据修正结论"),
        ])
        result = AgentRuntime(provider).run(
            config(), [{"role": "user", "content": "请回答"}],
            [Tool("read", "读取证据", {}, lambda: "EV001 原文")],
            output_validator=lambda output: None if "EV001" in output else "缺少证据",
        )
        self.assertEqual(result.output, "有证据的回答 [EV001]")
        self.assertEqual(result.steps, 3)
        assistants = [item for item in provider.histories[-1] if item["role"] == "assistant"]
        self.assertEqual([item["reasoning_content"] for item in assistants], [
            "先尝试已有信息", "需要检索缺失的原文证据",
        ])
        self.assertEqual(provider.histories[-1][-1]["role"], "tool")

    def test_truncation_then_validation_repair_keeps_each_original_fragment(self):
        provider = StrictReasoningProvider([
            response("前半段", "前半段的思考", "length"),
            response("后半段", "续写时的思考"),
            response("修正后的完整回答", "检查证据后重新回答"),
        ])
        validated = []

        def validate(output):
            validated.append(output)
            return None if output == "修正后的完整回答" else "引用无效"

        result = AgentRuntime(provider).run(
            config(), [{"role": "user", "content": "请回答"}],
            [Tool("read", "读取证据", {}, lambda: "证据")],
            output_validator=validate,
        )
        self.assertEqual(result.output, "修正后的完整回答")
        self.assertEqual(validated, ["前半段后半段", "修正后的完整回答"])
        assistants = [item for item in provider.histories[-1] if item["role"] == "assistant"]
        self.assertEqual(assistants, [
            {"role": "assistant", "content": "前半段", "reasoning_content": "前半段的思考"},
            {"role": "assistant", "content": "后半段", "reasoning_content": "续写时的思考"},
        ])

    def test_tool_completion_repair_does_not_invent_an_assistant_summary(self):
        pending = {}

        def finish(summary):
            pending["output"] = summary
            return "完成请求已收到"

        provider = StrictReasoningProvider([
            ModelResponse(AssistantMessage(
                tool_calls=(ToolCall("finish", "finish", '{"summary":"缺证据摘要"}'),),
                reasoning_content="通过工具声明任务完成",
            ), "tool_calls"),
            response("修正后的回答", "根据校验反馈补齐证据"),
        ])
        result = AgentRuntime(provider).run(
            config(), [{"role": "user", "content": "请回答"}],
            [Tool("finish", "声明完成", {}, finish)],
            completion_requested=lambda: pending.pop("output", None),
            output_validator=lambda output: None if output == "修正后的回答" else "缺少证据",
        )
        self.assertEqual(result.output, "修正后的回答")
        assistants = [item for item in provider.histories[-1] if item["role"] == "assistant"]
        self.assertEqual(len(assistants), 1)
        self.assertEqual(assistants[0]["tool_calls"][0]["id"], "finish")
        self.assertEqual(assistants[0]["reasoning_content"], "通过工具声明任务完成")
        self.assertIn("缺少证据", provider.histories[-1][-1]["content"])

    def test_non_reasoning_provider_is_not_given_fabricated_reasoning(self):
        provider = StrictReasoningProvider([
            response("前半段", "", "length"),
            response("后半段", ""),
            response("修正后的回答", ""),
        ], require_reasoning=False)
        result = AgentRuntime(provider).run(
            config(), [{"role": "user", "content": "请回答"}],
            output_validator=lambda output: None if output == "修正后的回答" else "引用无效",
        )
        self.assertEqual(result.output, "修正后的回答")
        for history in provider.histories:
            for message in history:
                self.assertNotIn("reasoning_content", message)


if __name__ == "__main__":
    unittest.main()
