import unittest
import hashlib
import json
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
    OutputValidationError,
    Tool,
    ToolCall,
    _tool_argument_errors,
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


def completion_tool():
    pending = {}

    def request(summary="done"):
        pending["output"] = summary
        return "completion requested"

    return Tool("finish", "request completion", {}, request), lambda: pending.pop("output", None)


def continuation_tool():
    pending = {}

    def request(summary="remaining work"):
        pending["summary"] = summary
        return "continuation requested"

    return Tool("continue", "request continuation", {}, request), lambda: pending.pop("summary", None)


class RuntimeTests(unittest.TestCase):
    def test_default_config_keeps_old_iteration_limit_without_phase_callbacks(self):
        self.assertEqual((config().finalization_iterations, config().continuation_iterations,
                          config().max_continuations), (0, 0, 0))
        read = ModelResponse(AssistantMessage(tool_calls=(ToolCall("r", "read", '{}'),)), "tool_calls")
        provider = FakeProvider([read, read])
        phases, events = [], []
        with self.assertRaises(MaxIterationsExceeded) as caught:
            AgentRuntime(provider).run(
                replace(config(), max_iterations=2), [], [Tool("read", "read", {}, lambda: "current")],
                phase_feedback=lambda *args: phases.append(args) or "unexpected",
                on_event=lambda kind, data: events.append((kind, data)),
            )
        self.assertEqual(caught.exception.steps, 2)
        self.assertEqual(phases, [])
        self.assertFalse(any(kind == "phase_started" for kind, _ in events))

    def test_phase_feedback_failure_does_not_start_the_next_model_request(self):
        read = ModelResponse(AssistantMessage(tool_calls=(ToolCall("r", "read", '{}'),)), "tool_calls")
        provider, events = FakeProvider([read]), []

        def broken_feedback(*_):
            raise ValueError("phase feedback unavailable")

        with self.assertRaisesRegex(ValueError, "phase feedback unavailable"):
            AgentRuntime(provider).run(
                replace(config(), max_iterations=1, finalization_iterations=2), [],
                [Tool("read", "read", {}, lambda: "current")], phase_feedback=broken_feedback,
                on_event=lambda kind, data: events.append((kind, data)),
            )
        self.assertEqual(len(provider.requests), 1)
        self.assertEqual(events[-1], ("run_failed", {
            "run_id": events[0][1]["run_id"], "error": "phase feedback unavailable",
        }))

    def test_dedicated_finalization_preserves_full_window_and_requires_explicit_finish(self):
        finish, consume = completion_tool()
        phases, events, validations, writes = [], [], [], []
        provider = FakeProvider([
            ModelResponse(AssistantMessage(tool_calls=(ToolCall("w1", "write", '{"value":1}'),)), "tool_calls"),
            ModelResponse(AssistantMessage(tool_calls=(ToolCall("w2", "write", '{"value":2}'),)), "tool_calls"),
            ModelResponse(AssistantMessage(tool_calls=(ToolCall("r", "read", '{}'),)), "tool_calls"),
            ModelResponse(AssistantMessage(tool_calls=(ToolCall("fix", "write", '{"value":3}'),)), "tool_calls"),
            ModelResponse(AssistantMessage(tool_calls=(ToolCall("done", "finish", '{}'),)), "tool_calls"),
        ])
        result = AgentRuntime(provider).run(
            replace(config(), max_iterations=2, finalization_iterations=3,
                    finalization_reserve_iterations=1, soft_iteration_limit=2), [],
            [finish, Tool("write", "write", {}, lambda value: writes.append(value) or "saved",
                          progress_detector=lambda result: True), Tool("read", "read", {}, lambda: "current")],
            phase_feedback=lambda phase, step, remaining: phases.append((phase, step, remaining)) or "专用收尾",
            completion_requested=consume,
            output_validator=lambda output: validations.append((output, list(writes))) or None,
            on_event=lambda kind, data: events.append((kind, data)),
        )
        self.assertEqual(result.steps, 5)
        self.assertEqual(phases, [("finalization", 3, 3)])
        self.assertEqual(validations, [("done", [1, 2, 3])])
        self.assertEqual(provider.message_snapshots[2][-1]["content"], "专用收尾")
        self.assertEqual([data["step"] for kind, data in events if kind == "finalization_budget"], [2])
        self.assertLess(next(index for index, item in enumerate(events) if item[0] == "phase_started"),
                        next(index for index, item in enumerate(events)
                             if item[0] == "model_started" and item[1]["step"] == 3))

    def test_no_completion_uses_exactly_eight_dedicated_rounds_without_validation(self):
        read = ModelResponse(AssistantMessage(tool_calls=(ToolCall("read", "read", '{}'),)), "tool_calls")
        provider = FakeProvider([read] * 10)
        phases, validations = [], []
        with self.assertRaises(MaxIterationsExceeded) as caught:
            AgentRuntime(provider).run(
                replace(config(), max_iterations=2, finalization_iterations=8, soft_iteration_limit=2), [],
                [Tool("read", "read", {}, lambda: "unchanged")],
                phase_feedback=lambda *args: phases.append(args) or "finish remaining work",
                output_validator=lambda output: validations.append(output) or None,
            )
        self.assertEqual(caught.exception.steps, 10)
        self.assertEqual(len(provider.requests), 10)
        self.assertEqual(phases, [("finalization", 3, 8)])
        self.assertEqual(validations, [])

    def test_soft_stagnation_enters_dedicated_finalization_without_callback_progress(self):
        read = ModelResponse(AssistantMessage(tool_calls=(ToolCall("read", "read", '{}'),)), "tool_calls")
        provider = FakeProvider([read] * 5)
        phases = []
        with self.assertRaises(MaxIterationsExceeded) as caught:
            AgentRuntime(provider).run(
                replace(config(), max_iterations=10, finalization_iterations=3,
                        soft_iteration_limit=2, progress_grace_iterations=1), [],
                [Tool("read", "read", {}, lambda: "unchanged")],
                phase_feedback=lambda *args: phases.append(args) or "phase context",
            )
        self.assertEqual(caught.exception.steps, 5)
        self.assertEqual(phases, [("finalization", 3, 3)])

    def test_early_finalization_repair_never_emits_late_regular_budget_warning(self):
        finish, consume = completion_tool()
        phases, warnings, events, validations = [], [], [], []
        names = ["read", "finish", "read", "read", "read", "finish"]
        provider = FakeProvider([
            ModelResponse(AssistantMessage(tool_calls=(ToolCall(str(index), name, '{}'),)), "tool_calls")
            for index, name in enumerate(names)
        ])

        def validate(output):
            validations.append(output)
            return "remaining correction" if len(validations) == 1 else None

        result = AgentRuntime(provider).run(
            replace(config(), max_iterations=6, finalization_iterations=2,
                    finalization_reserve_iterations=2, soft_iteration_limit=1,
                    progress_grace_iterations=1, validation_repair_iterations=4), [],
            [finish, Tool("read", "read", {}, lambda: "current")],
            completion_requested=consume, output_validator=validate,
            phase_feedback=lambda *args: phases.append(args) or "dedicated finalization",
            budget_feedback=lambda *args: warnings.append(args) or "obsolete regular warning",
            on_event=lambda kind, data: events.append((kind, data)),
        )
        self.assertEqual(result.steps, 6)
        self.assertEqual(phases, [("finalization", 2, 2)])
        self.assertEqual(warnings, [])
        self.assertEqual(validations, ["done", "done"])
        self.assertFalse(any(kind == "finalization_budget" for kind, _ in events))
        self.assertFalse(any(message.get("content") == "obsolete regular warning"
                             for history in provider.message_snapshots for message in history))
        self.assertEqual([data["iteration_deadline"] for kind, data in events
                          if kind == "validation_repair_budget"], [6])

    def test_continuation_preserves_context_usage_and_gets_a_second_finalization(self):
        finish, consume_finish = completion_tool()
        continued, consume_continue = continuation_tool()
        phases, writes = [], []
        calls = [
            ToolCall("initial", "write", '{"value":1}'), ToolCall("read", "read", '{}'),
            ToolCall("more", "continue", '{}'), ToolCall("next", "write", '{"value":2}'),
            ToolCall("read2", "read", '{}'), ToolCall("read3", "read", '{}'),
            ToolCall("done", "finish", '{}'),
        ]
        provider = FakeProvider([
            ModelResponse(AssistantMessage(tool_calls=(call,)), "tool_calls",
                          {"prompt_tokens": 100 + index, "completion_tokens": 1, "total_tokens": 101 + index})
            for index, call in enumerate(calls)
        ])
        result = AgentRuntime(provider).run(
            replace(config(), max_iterations=2, finalization_iterations=2, continuation_iterations=2,
                    max_continuations=1, soft_iteration_limit=1, progress_grace_iterations=1), [],
            [finish, continued, Tool("read", "read", {}, lambda: "current"),
             Tool("write", "write", {}, lambda value: writes.append(value) or "saved",
                  progress_detector=lambda result: True)],
            completion_requested=consume_finish, continuation_requested=consume_continue,
            phase_feedback=lambda *args: phases.append(args) or str(args),
        )
        self.assertEqual(result.steps, 7)
        self.assertEqual(result.usage["completion_tokens"], 7)
        self.assertEqual(writes, [1, 2])
        self.assertEqual(phases, [("finalization", 3, 2), ("continuation", 4, 2), ("finalization", 6, 2)])
        history = provider.message_snapshots[3]
        self.assertIn("initial", [call["id"] for message in history for call in message.get("tool_calls", [])])
        self.assertIn("more", [message["tool_call_id"] for message in history if message.get("role") == "tool"])

    def test_continuation_limit_and_progress_are_not_reset_by_new_phases(self):
        for limit, second_write, expected_error in ((1, True, MaxIterationsExceeded),
                                                    (2, False, AgentLoopStalled)):
            with self.subTest(limit=limit):
                continued, consume = continuation_tool()
                calls = [ToolCall("w", "write", '{}'), ToolCall("c1", "continue", '{}'),
                         ToolCall("middle", "write" if second_write else "read", '{}'),
                         ToolCall("c2", "continue", '{}')]
                provider = FakeProvider([ModelResponse(AssistantMessage(tool_calls=(call,)), "tool_calls")
                                         for call in calls])
                events = []
                with self.assertRaises(expected_error) as caught:
                    AgentRuntime(provider).run(
                        replace(config(), max_iterations=1, finalization_iterations=1,
                                continuation_iterations=1, max_continuations=limit), [],
                        [continued, Tool("write", "write", {}, lambda: "saved", progress_detector=lambda _: True),
                         Tool("read", "read", {}, lambda: "unchanged")], continuation_requested=consume,
                        on_event=lambda kind, data: events.append((kind, data)),
                    )
                self.assertEqual(caught.exception.steps, 4)
                self.assertEqual(caught.exception.tool_counts["continue"], 2)
                self.assertEqual(sum(kind == "continuation_granted" for kind, _ in events), 1)
                self.assertIsNone(consume())

    def test_read_and_phase_feedback_cannot_earn_first_continuation(self):
        continued, consume = continuation_tool()
        provider = FakeProvider([
            ModelResponse(AssistantMessage(tool_calls=(ToolCall("r", "read", '{}'),)), "tool_calls"),
            ModelResponse(AssistantMessage(tool_calls=(ToolCall("c", "continue", '{}'),)), "tool_calls"),
        ])
        with self.assertRaises(AgentLoopStalled):
            AgentRuntime(provider).run(
                replace(config(), max_iterations=1, finalization_iterations=1,
                        continuation_iterations=2, max_continuations=1), [],
                [continued, Tool("read", "read", {}, lambda: "current")],
                continuation_requested=consume, phase_feedback=lambda *_: "new context",
            )
        self.assertIsNone(consume())

    def test_continuation_does_not_reset_repetition_protection(self):
        continued, consume = continuation_tool()
        read = ToolCall("r", "read", '{}')
        provider = FakeProvider([
            ModelResponse(AssistantMessage(tool_calls=(ToolCall("w", "write", '{}'),)), "tool_calls"),
            ModelResponse(AssistantMessage(tool_calls=(ToolCall("c", "continue", '{}'), read)), "tool_calls"),
            ModelResponse(AssistantMessage(tool_calls=(read,)), "tool_calls"),
            ModelResponse(AssistantMessage(tool_calls=(read,)), "tool_calls"),
        ])
        events = []
        with self.assertRaises(AgentLoopStalled) as caught:
            AgentRuntime(provider).run(
                replace(config(), max_iterations=1, finalization_iterations=1, continuation_iterations=2,
                        max_continuations=1, max_identical_tool_calls=3), [],
                [continued, Tool("read", "read", {}, lambda: "same"),
                 Tool("write", "write", {}, lambda: "saved", progress_detector=lambda _: True)],
                continuation_requested=consume, on_event=lambda kind, data: events.append((kind, data)),
            )
        self.assertEqual(caught.exception.steps, 4)
        self.assertEqual(caught.exception.repeated_tool_calls, 2)
        self.assertEqual([data["step"] for kind, data in events if kind == "tool_repetition_warning"], [3])

    def test_failed_truncated_and_conflicting_continuations_are_consumed_once(self):
        for failure in ("business", "field", "unknown", "json", "length", "conflict", "regular"):
            with self.subTest(failure=failure):
                finish, consume_finish = completion_tool()
                pending, consumed, events, validations = {}, [], [], []

                def request_more():
                    pending["summary"] = "known remaining work"
                    return "requested"

                def consume_more():
                    value = pending.pop("summary", None)
                    consumed.append(value)
                    return value

                sibling = (ToolCall("other", "finish", '{}') if failure == "conflict" else
                           ToolCall("other", "missing", '{}') if failure == "unknown" else
                           ToolCall("other", "bad", '{broken' if failure == "json" else '{}'))
                rejected = ModelResponse(AssistantMessage(tool_calls=(ToolCall("more", "continue", '{}'), sibling)),
                                         "length" if failure == "length" else "tool_calls")

                class PendingProvider(FakeProvider):
                    def complete(self, request):
                        if len(self.requests) == 1 and failure == "json":
                            pending["summary"] = "preexisting request must be consumed"
                        return super().complete(request)

                provider = PendingProvider([
                    ModelResponse(AssistantMessage(tool_calls=(ToolCall("w", "write", '{}'),)), "tool_calls"),
                    rejected,
                    ModelResponse(AssistantMessage(tool_calls=(ToolCall("read", "read", '{}'),)), "tool_calls"),
                    ModelResponse(AssistantMessage(tool_calls=(ToolCall("done", "finish", '{}'),)), "tool_calls"),
                ])
                result = AgentRuntime(provider).run(
                    replace(config(), max_iterations=2 if failure == "regular" else 1,
                            finalization_iterations=3, continuation_iterations=2, max_continuations=1), [],
                    [finish, Tool("continue", "continue", {}, request_more),
                     Tool("write", "write", {}, lambda: "saved", progress_detector=lambda _: True),
                     Tool("read", "read", {}, lambda: "current"),
                     Tool("bad", "bad", {}, lambda: "[错误] failed" if failure in {"business", "field"} else "ok",
                          argument_error_detector=(lambda _: True) if failure == "field" else None)],
                    completion_requested=consume_finish, continuation_requested=consume_more,
                    output_validator=lambda output: validations.append(output) or None,
                    on_event=lambda kind, data: events.append((kind, data)),
                )
                self.assertEqual(result.steps, 4)
                self.assertEqual(validations, ["done"])
                self.assertEqual(sum(value is not None for value in consumed), 1)
                self.assertEqual(sum(kind == "continuation_request_rejected" for kind, _ in events), 1)
                self.assertFalse(any(kind == "continuation_granted" for kind, _ in events))
                self.assertEqual(pending, {})
                self.assertIsNone(consume_finish())
                if failure == "conflict":
                    self.assertEqual(sum(kind == "completion_request_rejected" for kind, _ in events), 1)

    def test_continuation_revokes_old_completion_but_keeps_global_validation_limit(self):
        for final_valid in (True, False):
            with self.subTest(final_valid=final_valid):
                finish, consume_finish = completion_tool()
                continued, consume_continue = continuation_tool()
                state, events = {}, []

                def write():
                    state["writes"] = state.get("writes", 0) + 1
                    return "saved"

                calls = [ToolCall("w1", "write", '{}'), ToolCall("done1", "finish", '{}'),
                         ToolCall("more", "continue", '{}'), ToolCall("w2", "write", '{}'),
                         ToolCall("done2", "finish", '{}')]
                provider = FakeProvider([ModelResponse(AssistantMessage(tool_calls=(call,)), "tool_calls")
                                         for call in calls])

                def run():
                    return AgentRuntime(provider).run(
                        replace(config(), max_iterations=1, finalization_iterations=2, continuation_iterations=1,
                                max_continuations=1, validation_repair_iterations=2, max_validation_retries=1,
                                finish_after_validation_repair=True), [],
                        [finish, continued, Tool("write", "write", {}, write, progress_detector=lambda _: True)],
                        completion_requested=consume_finish, continuation_requested=consume_continue,
                        output_validator=lambda _: None if final_valid and state.get("writes") == 2 else "unfinished",
                        on_event=lambda kind, data: events.append((kind, data)),
                    )

                if final_valid:
                    self.assertEqual(run().steps, 5)
                else:
                    with self.assertRaises(OutputValidationError):
                        run()
                    self.assertEqual([data["attempt"] for kind, data in events if kind == "output_rejected"], [1, 2])
                self.assertEqual(len(provider.requests), 5)
                self.assertFalse(any(kind == "validation_repair_checked" for kind, _ in events))
                self.assertEqual(sum(kind == "validation_repair_budget" for kind, _ in events), 1)

    def test_combined_phase_and_validation_windows_have_an_absolute_bound(self):
        finish, consume_finish = completion_tool()
        continued, consume_continue = continuation_tool()
        names = ["write", "finish", "continue", "write", "finish", "read"]
        provider = FakeProvider([ModelResponse(AssistantMessage(tool_calls=(ToolCall(str(i), name, '{}'),)), "tool_calls")
                                 for i, name in enumerate(names)])
        events = []
        with self.assertRaises(MaxIterationsExceeded) as caught:
            AgentRuntime(provider).run(
                replace(config(), max_iterations=1, finalization_iterations=1, continuation_iterations=1,
                        max_continuations=1, max_validation_retries=2, validation_repair_iterations=1), [],
                [finish, continued, Tool("write", "write", {}, lambda: "saved", progress_detector=lambda _: True),
                 Tool("read", "read", {}, lambda: "current")],
                completion_requested=consume_finish, continuation_requested=consume_continue,
                output_validator=lambda _: "unfinished", on_event=lambda kind, data: events.append((kind, data)),
            )
        self.assertEqual(caught.exception.steps, 6)  # 1 + 1 + (1 + 1) + 2 * 1
        self.assertEqual(caught.exception.limit, 6)
        self.assertEqual([data["phase"] for kind, data in events if kind == "phase_started"],
                         ["finalization", "continuation", "finalization"])
        self.assertEqual([data["attempt"] for kind, data in events if kind == "output_rejected"], [1, 2])

    def test_json_error_excerpt_guides_correct_retry_without_executing_bad_round(self):
        entity = {"name": "姜述", "content": "长正文" * 1_000, "source_chapters": [2800, 2803]}
        valid = json.dumps({"entities": [entity]}, ensure_ascii=False)
        malformed = valid.replace("2803]}", "2803}", 1)
        sibling = json.dumps({"entities": [{"name": "旁支", "source_chapters": [2803]}]}, ensure_ascii=False)
        bad_response = ModelResponse(AssistantMessage(
            "BAD_ASSISTANT_MUST_NOT_ENTER_HISTORY",
            tool_calls=(ToolCall("bad", "write", malformed), ToolCall("skipped", "write", sibling)),
        ), "tool_calls")
        executed, events = [], []
        case = self

        class CheckingProvider(FakeProvider):
            def complete(self, request):
                if len(self.requests) == 1:
                    case.assertEqual(executed, [])
                    feedback = request.messages[-1]["content"]
                    detail = next(json.loads(line.removeprefix("错误：")) for line in feedback.splitlines()
                                  if line.removeprefix("错误：").startswith('{"name":'))
                    diagnostic = detail["diagnostic"]
                    case.assertEqual(diagnostic["argument_length"], len(malformed))
                    case.assertEqual(diagnostic["argument_sha256"], hashlib.sha256(malformed.encode()).hexdigest())
                    case.assertIn('"source_chapters": [2800, 2803}]}', diagnostic["excerpt"])
                    case.assertEqual(
                        diagnostic["pos"], diagnostic["excerpt_start"] + diagnostic["cursor_offset"],
                    )
                    case.assertIn("括号、引号和逗号", feedback)
                    case.assertNotIn(malformed, feedback)
                    case.assertFalse(any(message.get("tool_calls") for message in request.messages))
                    case.assertFalse(any("BAD_ASSISTANT" in message.get("content", "") for message in request.messages))
                    # 模拟模型读到尾部诊断后，重发修正后的整轮调用。
                    self.responses.insert(0, ModelResponse(AssistantMessage(tool_calls=(
                        ToolCall("fixed", "write", json.dumps({"entities": [entity]}, ensure_ascii=False)),
                        ToolCall("retried-sibling", "write", sibling),
                    )), "tool_calls"))
                return super().complete(request)

        provider = CheckingProvider([bad_response, ModelResponse(AssistantMessage("done"), "stop")])
        result = AgentRuntime(provider).run(
            config(), [],
            [Tool("write", "write", {}, lambda entities: executed.extend(entities) or "saved")],
            on_event=lambda kind, data: events.append((kind, data)),
        )
        self.assertEqual(result.output, "done")
        self.assertEqual([item["name"] for item in executed], ["姜述", "旁支"])
        started = [data["call_id"] for kind, data in events if kind == "tool_started"]
        self.assertEqual(started, ["fixed", "retried-sibling"])
        history = provider.message_snapshots[-1]
        requested_ids = [call["id"] for message in history for call in message.get("tool_calls", [])]
        returned_ids = [message["tool_call_id"] for message in history if message.get("role") == "tool"]
        self.assertEqual(requested_ids, returned_ids)

    def test_json_error_diagnostic_is_stable_even_on_exhausted_attempt(self):
        raw = '{"value":"' + "汉字" * 2_000 + '","source_chapters":[2800,2803}]}'
        bad = ModelResponse(AssistantMessage(tool_calls=(ToolCall("bad", "write", raw),)), "tool_calls")
        events, executed = [], []
        with self.assertRaises(MalformedToolArguments):
            AgentRuntime(FakeProvider([bad, bad])).run(
                replace(config(), max_tool_argument_retries=1), [],
                [Tool("write", "write", {}, lambda **arguments: executed.append(arguments) or "saved")],
                on_event=lambda kind, data: events.append((kind, data)),
            )
        rejected = [data for kind, data in events if kind == "tool_arguments_rejected"]
        self.assertEqual([data["will_retry"] for data in rejected], [True, False])
        self.assertEqual(rejected[0]["errors"][0]["diagnostic"], rejected[1]["errors"][0]["diagnostic"])
        self.assertLessEqual(len(rejected[-1]["errors"][0]["diagnostic"]["excerpt"]), 320)
        self.assertNotIn(raw, json.dumps(events, ensure_ascii=False))
        self.assertEqual(executed, [])

    def test_json_error_excerpt_offsets_cover_unicode_and_string_boundaries(self):
        cases = [
            "?" + "汉🙂" * 4_000,
            '{"正文":"' + "汉🙂" * 4_000 + '","source_chapters":[2803}',
            '{\n  "正文": "汉🙂",\n  "source_chapters": [2803}\n',
            '{"unfinished":"汉🙂',
        ]
        for raw in cases:
            with self.subTest(raw_length=len(raw)):
                diagnostic = _tool_argument_errors([ToolCall("bad", "write", raw)])[0]["diagnostic"]
                try:
                    json.loads(raw)
                except json.JSONDecodeError as error:
                    self.assertEqual((diagnostic["pos"], diagnostic["lineno"], diagnostic["colno"]),
                                     (error.pos, error.lineno, error.colno))
                self.assertEqual(diagnostic["argument_length"], len(raw))
                self.assertEqual(diagnostic["argument_sha256"], hashlib.sha256(raw.encode()).hexdigest())
                self.assertGreaterEqual(diagnostic["excerpt_start"], 0)
                self.assertLessEqual(diagnostic["excerpt_end"], len(raw))
                self.assertLessEqual(len(diagnostic["excerpt"]), 320)
                self.assertEqual(diagnostic["excerpt"], raw[diagnostic["excerpt_start"]:diagnostic["excerpt_end"]])
                self.assertGreaterEqual(diagnostic["cursor_offset"], 0)
                self.assertLessEqual(diagnostic["cursor_offset"], len(diagnostic["excerpt"]))

    def test_non_object_and_non_text_arguments_have_diagnostics_without_fake_positions(self):
        for raw, kind, parsed_type in (("[]", "non_object", "list"),
                                       ('"正文"', "non_object", "str"),
                                       ("null", "non_object", "NoneType"),
                                       (42, "invalid_type", "int")):
            with self.subTest(raw=raw):
                diagnostic = _tool_argument_errors([ToolCall("bad", "write", raw)])[0]["diagnostic"]
                self.assertEqual(diagnostic["kind"], kind)
                self.assertEqual(diagnostic["parsed_type"], parsed_type)
                self.assertIsNone(diagnostic["pos"])
                self.assertIsNone(diagnostic["cursor_offset"])
                if isinstance(raw, str):
                    self.assertEqual(diagnostic["argument_length"], len(raw))
                    self.assertEqual(diagnostic["argument_sha256"], hashlib.sha256(raw.encode()).hexdigest())

    def test_many_large_bad_arguments_keep_feedback_bounded_and_control_characters_escaped(self):
        raw = '{"value":"' + "\0" * 6_000 + '"}'
        calls = tuple(ToolCall(f"bad-{index}", "write", raw) for index in range(8))
        provider = FakeProvider([
            ModelResponse(AssistantMessage(tool_calls=calls), "tool_calls"),
            ModelResponse(AssistantMessage("done"), "stop"),
        ])
        events, executed = [], []
        AgentRuntime(provider).run(
            config(), [], [Tool("write", "write", {}, lambda **arguments: executed.append(arguments))],
            on_event=lambda kind, data: events.append((kind, data)),
        )
        feedback = provider.message_snapshots[1][-1]["content"]
        self.assertLessEqual(len(feedback), 6_000)
        self.assertIn("\\u0000", feedback)
        self.assertNotIn("\0", feedback)
        self.assertIn("个失败项未展开", feedback)
        rejected = next(data for kind, data in events if kind == "tool_arguments_rejected")
        self.assertEqual(len(rejected["errors"]), 8)
        self.assertTrue(all(len(item["diagnostic"]["excerpt"]) <= 320 for item in rejected["errors"]))
        self.assertEqual(executed, [])

    def test_finalization_budget_is_notified_once_without_automatic_completion(self):
        read = ModelResponse(AssistantMessage(tool_calls=(ToolCall("r", "read", "{}"),)), "tool_calls")
        for maximum, reserve, expected in ((60, 8, (53, 8)), (5, 2, (4, 2)), (2, 8, (1, 2))):
            with self.subTest(maximum=maximum, reserve=reserve):
                provider = FakeProvider([read] * maximum)
                feedback_calls, validated = [], []
                with self.assertRaises(MaxIterationsExceeded) as caught:
                    AgentRuntime(provider).run(
                        replace(config(), max_iterations=maximum,
                                finalization_reserve_iterations=reserve,
                                finish_after_validation_repair=True), [],
                        [Tool("read", "read", {}, lambda: "current state")],
                        budget_feedback=lambda step, remaining: feedback_calls.append((step, remaining)) or "请收尾",
                        output_validator=lambda output: validated.append(output),
                        completion_requested=lambda: None,
                    )
                self.assertEqual(feedback_calls, [expected])
                self.assertEqual(caught.exception.steps, maximum)
                self.assertEqual(len(provider.requests), maximum)
                self.assertEqual(validated, [])
                self.assertEqual(provider.message_snapshots[expected[0] - 1][-1]["content"], "请收尾")

    def test_finalization_feedback_does_not_count_as_write_progress(self):
        read = ModelResponse(AssistantMessage(tool_calls=(ToolCall("r", "read", "{}"),)), "tool_calls")
        provider = FakeProvider([read] * 5)
        feedback_calls = []
        with self.assertRaises(MaxIterationsExceeded) as caught:
            AgentRuntime(provider).run(
                replace(config(), soft_iteration_limit=4, progress_grace_iterations=1,
                        finalization_reserve_iterations=2), [],
                [Tool("read", "read", {}, lambda: "current state")],
                budget_feedback=lambda step, remaining: feedback_calls.append((step, remaining)) or "请收尾",
            )
        self.assertEqual(feedback_calls, [(4, 2)])
        self.assertEqual(caught.exception.steps, 4)
        self.assertEqual(len(provider.requests), 4)

    def test_finalization_callback_failure_is_reported_before_calling_model(self):
        provider, events = FakeProvider([]), []

        def feedback(step, remaining):
            raise ValueError("cannot inspect remaining work")

        with self.assertRaisesRegex(ValueError, "cannot inspect remaining work"):
            AgentRuntime(provider).run(
                replace(config(), finalization_reserve_iterations=8), [],
                budget_feedback=feedback,
                on_event=lambda kind, data: events.append((kind, data)),
            )
        self.assertEqual(provider.requests, [])
        self.assertEqual(events[-1][0], "run_failed")
        self.assertIn("cannot inspect remaining work", events[-1][1]["error"])

    def test_tool_completion_validates_after_all_successful_sibling_tools(self):
        finish, consume = completion_tool()
        written, validated = [], []
        provider = FakeProvider([ModelResponse(AssistantMessage(tool_calls=(
            ToolCall("f", "finish", '{"summary":"both complete"}'),
            ToolCall("w", "write", "{}"),
        )), "tool_calls")])
        result = AgentRuntime(provider).run(
            config(), [],
            [finish, Tool("write", "write", {}, lambda: written.append(True) or "saved")],
            completion_requested=consume,
            output_validator=lambda output: validated.append((output, list(written))),
        )
        self.assertEqual(result.output, "both complete")
        self.assertEqual(result.steps, 1)
        self.assertEqual(validated, [("both complete", [True])])
        self.assertIsNone(consume())

    def test_failed_or_invalid_sibling_tool_requires_another_completion_request(self):
        cases = (
            (ToolCall("bad", "bad", "{}"), False),
            (ToolCall("bad", "bad", "{}"), True),
            (ToolCall("unknown", "unknown", "{}"), False),
            (ToolCall("json", "bad", "{"), False),
        )
        for bad_call, argument_error in cases:
            with self.subTest(name=bad_call.name, raw=bad_call.arguments, argument_error=argument_error):
                finish, consume = completion_tool()
                validated = []
                provider = FakeProvider([
                    ModelResponse(AssistantMessage(tool_calls=(ToolCall("f1", "finish", "{}"), bad_call)), "tool_calls"),
                    ModelResponse(AssistantMessage(tool_calls=(ToolCall("w", "write", "{}"),)), "tool_calls"),
                    ModelResponse(AssistantMessage(tool_calls=(ToolCall("f2", "finish", '{"summary":"confirmed"}'),)), "tool_calls"),
                ])
                result = AgentRuntime(provider).run(
                    replace(config(), finish_after_validation_repair=True), [],
                    [finish, Tool("bad", "fail", {}, lambda: "[错误] failed",
                                  argument_error_detector=lambda result: argument_error),
                     Tool("write", "write", {}, lambda: "saved", progress_detector=lambda result: True)],
                    completion_requested=consume,
                    output_validator=lambda output: validated.append(output),
                    can_finish_after_validation_repair=lambda: True,
                )
                self.assertEqual(result.steps, 3)
                self.assertEqual(result.output, "confirmed")
                self.assertEqual(validated, ["confirmed"])
                self.assertIsNone(consume())
                if bad_call.arguments != "{":
                    self.assertIn("再次明确声明完成", provider.message_snapshots[1][-1]["content"])

    def test_truncated_tool_completion_is_consumed_without_finishing(self):
        finish, consume = completion_tool()
        validated = []
        provider = FakeProvider([
            ModelResponse(AssistantMessage(tool_calls=(ToolCall("f1", "finish", "{}"),)), "length"),
            ModelResponse(AssistantMessage(tool_calls=(ToolCall("w", "write", "{}"),)), "tool_calls"),
            ModelResponse(AssistantMessage(tool_calls=(ToolCall("f2", "finish", '{"summary":"confirmed"}'),)), "tool_calls"),
        ])
        result = AgentRuntime(provider).run(
            replace(config(), finish_after_validation_repair=True), [],
            [finish, Tool("write", "write", {}, lambda: "saved", progress_detector=lambda result: True)],
            completion_requested=consume,
            output_validator=lambda output: validated.append(output),
        )
        self.assertEqual(result.steps, 3)
        self.assertEqual(validated, ["confirmed"])
        self.assertIn("响应被截断", provider.message_snapshots[1][-1]["content"])
        self.assertIsNone(consume())

    def test_rejected_completion_during_repair_blocks_automatic_finish_until_redeclared(self):
        finish, consume = completion_tool()
        written = []
        provider = FakeProvider([
            ModelResponse(AssistantMessage("first attempt"), "stop"),
            ModelResponse(AssistantMessage(tool_calls=(
                ToolCall("f1", "finish", "{}"), ToolCall("bad", "fail", "{}"),
            )), "tool_calls"),
            ModelResponse(AssistantMessage(tool_calls=(ToolCall("w", "write", "{}"),)), "tool_calls"),
            ModelResponse(AssistantMessage(tool_calls=(ToolCall("f2", "finish", '{"summary":"confirmed"}'),)), "tool_calls"),
        ])
        result = AgentRuntime(provider).run(
            replace(config(), finish_after_validation_repair=True), [],
            [finish, Tool("fail", "fail", {}, lambda: "[错误] failed"),
             Tool("write", "write", {}, lambda: written.append(True) or "saved",
                  progress_detector=lambda result: True)],
            completion_requested=consume,
            output_validator=lambda output: None if written else "missing content",
            can_finish_after_validation_repair=lambda: True,
        )
        self.assertEqual(result.steps, 4)
        self.assertEqual(result.output, "confirmed")

    def test_tool_completion_reuses_existing_validation_repair_budget(self):
        finish, consume = completion_tool()
        written, events = [], []
        provider = FakeProvider([
            ModelResponse(AssistantMessage(tool_calls=(ToolCall("f", "finish", "{}"),)), "tool_calls"),
            ModelResponse(AssistantMessage(tool_calls=(ToolCall("w", "write", "{}"),)), "tool_calls"),
        ])
        result = AgentRuntime(provider).run(
            replace(config(), max_iterations=1, finish_after_validation_repair=True,
                    validation_repair_iterations=8, max_validation_retries=5), [],
            [finish, Tool("write", "write", {}, lambda: written.append(True) or "saved",
                          progress_detector=lambda result: True)],
            completion_requested=consume,
            output_validator=lambda output: None if written else "missing content",
            can_finish_after_validation_repair=lambda: True,
            on_event=lambda kind, data: events.append((kind, data)),
        )
        self.assertEqual(result.steps, 2)
        self.assertEqual(result.output, "done")
        budget = next(data for kind, data in events if kind == "validation_repair_budget")
        self.assertEqual(budget["iteration_deadline"], 9)
        self.assertEqual(budget["repair_iterations"], 8)
        self.assertEqual(events[-1][1]["completion_reason"], "validation_repaired")

    def test_json_and_field_errors_share_five_repair_attempts(self):
        malformed = ModelResponse(
            AssistantMessage(tool_calls=(ToolCall("bad-json", "write", "{"),)), "tool_calls",
        )
        incomplete = ModelResponse(
            AssistantMessage(tool_calls=(ToolCall("bad-fields", "write", "{}"),)), "tool_calls",
        )
        fixed = ModelResponse(
            AssistantMessage(tool_calls=(ToolCall("fixed", "write", '{"value":"fixed"}'),)), "tool_calls",
        )
        for succeeds in (True, False):
            with self.subTest(succeeds=succeeds):
                provider = FakeProvider(
                    [malformed, incomplete, malformed, incomplete, incomplete]
                    + ([fixed, ModelResponse(AssistantMessage("done"), "stop")] if succeeds else [incomplete])
                )
                events, saved = [], []
                def write(value=None):
                    if value is None:
                        return '[错误] {"error_type":"invalid_arguments","missing_fields":["value"]}'
                    saved.append(value)
                    return "saved"
                def run():
                    return AgentRuntime(provider).run(
                        replace(config(), max_iterations=10, max_tool_argument_retries=5,
                                max_consecutive_tool_errors=3, max_identical_tool_calls=2,
                                soft_iteration_limit=2, progress_grace_iterations=2), [],
                        [Tool("write", "write", {}, write,
                              progress_detector=lambda result: result == "saved",
                              argument_error_detector=lambda result: result.startswith("[错误]"))],
                        on_event=lambda kind, data: events.append((kind, data)),
                    )
                if succeeds:
                    self.assertEqual(run().output, "done")
                    self.assertEqual(saved, ["fixed"])
                else:
                    with self.assertRaises(MalformedToolArguments) as caught:
                        run()
                    self.assertEqual(caught.exception.limit, 5)
                    self.assertEqual(caught.exception.steps, 6)
                    self.assertEqual(saved, [])
                rejected = [data for kind, data in events if kind == "tool_arguments_rejected"]
                self.assertEqual([data["attempt"] for data in rejected], list(range(1, len(rejected) + 1)))
                self.assertEqual(rejected[0]["executed"], False)
                self.assertEqual(rejected[1]["executed"], True)
                self.assertEqual(rejected[-1]["will_retry"], succeeds)
                self.assertIn("只重试失败项", provider.message_snapshots[2][-1]["content"])

    def test_multiple_field_errors_in_one_response_use_one_repair_attempt(self):
        bad_calls = tuple(ToolCall(str(index), "write", "{}") for index in range(6))
        provider = FakeProvider([
            ModelResponse(AssistantMessage(tool_calls=bad_calls), "tool_calls"),
            ModelResponse(AssistantMessage(tool_calls=(ToolCall("fixed", "write", '{"value":"ok"}'),)), "tool_calls"),
            ModelResponse(AssistantMessage("done"), "stop"),
        ])
        events = []
        result = AgentRuntime(provider).run(
            replace(config(), max_tool_argument_retries=1, max_consecutive_tool_errors=3,
                    max_identical_tool_calls=2), [],
            [Tool("write", "write", {}, lambda value=None: "saved" if value else "[错误] missing",
                  argument_error_detector=lambda result: True)],
            on_event=lambda kind, data: events.append((kind, data)),
        )
        self.assertEqual(result.steps, 3)
        rejected = [data for kind, data in events if kind == "tool_arguments_rejected"]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["attempt"], 1)
        self.assertEqual(len(rejected[0]["errors"]), 6)

    def test_successful_round_resets_field_repair_budget(self):
        bad = ModelResponse(AssistantMessage(tool_calls=(ToolCall("bad", "write", "{}"),)), "tool_calls")
        good = ModelResponse(AssistantMessage(tool_calls=(ToolCall("good", "write", '{"value":"ok"}'),)), "tool_calls")
        events = []
        result = AgentRuntime(FakeProvider([
            bad, good, bad, good, ModelResponse(AssistantMessage("done"), "stop"),
        ])).run(
            replace(config(), max_tool_argument_retries=1), [],
            [Tool("write", "write", {}, lambda value=None: "saved" if value else "[错误] missing",
                  argument_error_detector=lambda result: True)],
            on_event=lambda kind, data: events.append((kind, data)),
        )
        self.assertEqual(result.steps, 5)
        self.assertEqual([data["attempt"] for kind, data in events if kind == "tool_arguments_rejected"], [1, 1])

    def test_parameter_retries_do_not_clear_real_tool_failures(self):
        responses = [
            ModelResponse(AssistantMessage(tool_calls=(ToolCall(str(step), name, "{}"),)), "tool_calls")
            for step, name in enumerate(["broken", "fields", "broken", "broken"])
        ]
        with self.assertRaises(AgentLoopStalled) as caught:
            AgentRuntime(FakeProvider(responses)).run(
                replace(config(), max_consecutive_tool_errors=3, max_tool_argument_retries=5), [],
                [Tool("broken", "broken", {}, lambda: "[错误] write failed"),
                 Tool("fields", "fields", {}, lambda: "[错误] missing",
                      argument_error_detector=lambda result: True)],
            )
        self.assertEqual(caught.exception.steps, 4)
        self.assertEqual(caught.exception.failed_tool_calls, 4)

    def test_state_repair_finishes_on_last_tool_round_without_another_model_call(self):
        for repair_step in (2, 3):
            with self.subTest(repair_step=repair_step):
                responses = [ModelResponse(AssistantMessage("done"), "stop")]
                if repair_step == 3:
                    responses.append(ModelResponse(
                        AssistantMessage(tool_calls=(ToolCall("read", "read", "{}"),)),
                        "tool_calls",
                    ))
                responses.append(ModelResponse(
                    AssistantMessage(tool_calls=(ToolCall("fix", "write", "{}"),)),
                    "tool_calls",
                    {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
                ))
                provider = FakeProvider(responses)
                written, events, usage = [], [], []
                result = AgentRuntime(provider).run(
                    replace(config(), max_iterations=3, finish_after_validation_repair=True),
                    [],
                    [Tool("write", "write", {}, lambda: written.append(True) or "saved",
                          progress_detector=lambda result: True),
                     Tool("read", "read", {}, lambda: "data")],
                    output_validator=lambda output: None if written else "missing link",
                    on_event=lambda kind, data: events.append((kind, data)),
                    usage_recorder=usage.append,
                )
                self.assertEqual(result.steps, repair_step)
                self.assertEqual(result.output, "done")
                self.assertEqual(len(provider.requests), repair_step)
                self.assertEqual(result.usage["total_tokens"], 110)
                self.assertEqual(sum(item["total_tokens"] for item in usage), 110)
                self.assertEqual(events[-1][0], "run_completed")
                self.assertEqual(events[-1][1]["completion_reason"], "validation_repaired")

    def test_valid_intermediate_state_does_not_end_normal_work(self):
        provider = FakeProvider([
            ModelResponse(AssistantMessage(tool_calls=(ToolCall("1", "write", "{}"),)), "tool_calls"),
            ModelResponse(AssistantMessage(tool_calls=(ToolCall("2", "write", "{}"),)), "tool_calls"),
            ModelResponse(AssistantMessage("finished both"), "stop"),
        ])
        written, validated = [], []
        result = AgentRuntime(provider).run(
            replace(config(), finish_after_validation_repair=True), [],
            [Tool("write", "write", {}, lambda: written.append(True) or "saved")],
            output_validator=lambda output: validated.append(output),
        )
        self.assertEqual(written, [True, True])
        self.assertEqual(validated, ["finished both"])
        self.assertEqual(result.steps, 3)

    def test_state_repair_checks_whole_tool_round_and_reports_remaining_errors(self):
        provider = FakeProvider([
            ModelResponse(AssistantMessage("done"), "stop"),
            ModelResponse(AssistantMessage(tool_calls=(
                ToolCall("fix", "set_state", '{"value":"valid"}'),
                ToolCall("break", "set_state", '{"value":"new error"}'),
            )), "tool_calls"),
            ModelResponse(AssistantMessage(tool_calls=(
                ToolCall("fix2", "set_state", '{"value":"valid"}'),
            )), "tool_calls"),
        ])
        state = {"value": "old error"}
        def set_state(value):
            state["value"] = value
            return "saved"
        result = AgentRuntime(provider).run(
            replace(config(), finish_after_validation_repair=True), [],
            [Tool("set_state", "set state", {}, set_state, progress_detector=lambda result: True)],
            output_validator=lambda output: None if state["value"] == "valid" else state["value"],
        )
        self.assertEqual(result.steps, 3)
        feedback = provider.message_snapshots[2][-1]["content"]
        self.assertIn("new error", feedback)
        self.assertNotIn("old error", feedback)

    def test_tool_errors_prevent_automatic_repair_completion(self):
        for result_text in ("[错误] partially saved", "[错误] failed"):
            with self.subTest(result_text=result_text):
                written = []
                provider = FakeProvider([
                    ModelResponse(AssistantMessage("done"), "stop"),
                    ModelResponse(AssistantMessage(tool_calls=(
                        ToolCall("fix", "write", "{}"),
                    )), "tool_calls"),
                    ModelResponse(AssistantMessage("checked again"), "stop"),
                ])
                result = AgentRuntime(provider).run(
                    replace(config(), finish_after_validation_repair=True), [],
                    [Tool("write", "write", {}, lambda: written.append(True) or result_text,
                          progress_detector=lambda result: True)],
                    output_validator=lambda output: None if written else "missing link",
                )
                self.assertEqual(result.steps, 3)
                self.assertEqual(result.output, "checked again")

    def test_invalid_repair_still_stops_at_iteration_limit(self):
        provider = FakeProvider([
            ModelResponse(AssistantMessage("done"), "stop"),
            ModelResponse(AssistantMessage(tool_calls=(ToolCall("fix", "write", "{}"),)), "tool_calls"),
        ])
        with self.assertRaises(MaxIterationsExceeded):
            AgentRuntime(provider).run(
                replace(config(), max_iterations=2, finish_after_validation_repair=True), [],
                [Tool("write", "write", {}, lambda: "saved", progress_detector=lambda result: True)],
                output_validator=lambda output: "still missing link",
            )

    def test_failed_repair_is_not_hidden_by_later_read_or_unrelated_write(self):
        written = []
        provider = FakeProvider([
            ModelResponse(AssistantMessage("done"), "stop"),
            ModelResponse(AssistantMessage(tool_calls=(
                ToolCall("fix", "write", "{}"), ToolCall("fail", "fail", "{}"),
            )), "tool_calls"),
            ModelResponse(AssistantMessage(tool_calls=(ToolCall("read", "read", "{}"),)), "tool_calls"),
            ModelResponse(AssistantMessage(tool_calls=(ToolCall("other", "write", "{}"),)), "tool_calls"),
            ModelResponse(AssistantMessage("checked remaining failure"), "stop"),
        ])
        result = AgentRuntime(provider).run(
            replace(config(), finish_after_validation_repair=True), [],
            [Tool("write", "write", {}, lambda: written.append(True) or "saved",
                  progress_detector=lambda result: True),
             Tool("fail", "fail", {}, lambda: "[错误] failed"),
             Tool("read", "read", {}, lambda: "data")],
            output_validator=lambda output: None if written else "missing link",
        )
        self.assertEqual(result.steps, 5)
        self.assertEqual(result.output, "checked remaining failure")

    def test_incomplete_repair_response_requires_explicit_completion_even_with_guard(self):
        for interrupted in (
            ModelResponse(AssistantMessage(tool_calls=(ToolCall("bad", "write", "{"),)), "tool_calls"),
            ModelResponse(AssistantMessage(tool_calls=(ToolCall("unknown", "unknown", "{}"),)), "tool_calls"),
            ModelResponse(AssistantMessage("partial"), "length"),
        ):
            with self.subTest(interrupted=interrupted):
                written = []
                provider = FakeProvider([
                    ModelResponse(AssistantMessage("done"), "stop"), interrupted,
                    ModelResponse(AssistantMessage(tool_calls=(ToolCall("fix", "write", "{}"),)), "tool_calls"),
                    ModelResponse(AssistantMessage("confirmed"), "stop"),
                ])
                result = AgentRuntime(provider).run(
                    replace(config(), finish_after_validation_repair=True), [],
                    [Tool("write", "write", {}, lambda: written.append(True) or "saved",
                          progress_detector=lambda result: True)],
                    output_validator=lambda output: None if written else "missing link",
                    can_finish_after_validation_repair=lambda: True,
                )
                self.assertEqual(result.steps, 4)
                self.assertEqual(result.output, "confirmed")

    def test_truncated_repair_does_not_automatically_complete(self):
        written = []
        provider = FakeProvider([
            ModelResponse(AssistantMessage("done"), "stop"),
            ModelResponse(AssistantMessage(tool_calls=(ToolCall("fix", "write", "{}"),)), "length"),
            ModelResponse(AssistantMessage("confirmed"), "stop"),
        ])
        result = AgentRuntime(provider).run(
            replace(config(), finish_after_validation_repair=True), [],
            [Tool("write", "write", {}, lambda: written.append(True) or "saved")],
            output_validator=lambda output: None if written else "missing link",
        )
        self.assertEqual(result.steps, 3)
        self.assertEqual(result.output, "confirmed")

    def test_validation_at_last_iteration_has_room_for_tool_repair(self):
        provider = FakeProvider([
            ModelResponse(AssistantMessage("bad"), "stop"),
            ModelResponse(AssistantMessage(tool_calls=(ToolCall("fix", "write", "{}"),)), "tool_calls"),
            ModelResponse(AssistantMessage("fixed"), "stop"),
        ])
        written = []
        result = AgentRuntime(provider).run(
            replace(config(), max_iterations=1, soft_iteration_limit=1,
                    max_validation_retries=5, validation_repair_iterations=2), [],
            [Tool("write", "write", {}, lambda: written.append(True) or "saved")],
            output_validator=lambda output: None if written else "summary 不能为空",
        )
        self.assertEqual(result.steps, 3)
        self.assertEqual(written, [True])

    def test_validation_repair_budget_is_bounded(self):
        read = ModelResponse(AssistantMessage(tool_calls=(ToolCall("r", "read", "{}"),)), "tool_calls")
        provider = FakeProvider([ModelResponse(AssistantMessage("bad"), "stop"), read, read])
        with self.assertRaises(MaxIterationsExceeded) as caught:
            AgentRuntime(provider).run(
                replace(config(), max_iterations=1, max_validation_retries=5, validation_repair_iterations=2), [],
                [Tool("read", "read", {}, lambda: "data")], output_validator=lambda output: "invalid",
            )
        self.assertEqual(len(provider.requests), 3)
        self.assertEqual(caught.exception.limit, 3)

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
        self.assertIn("请勿继续提交完全相同的参数", provider.message_snapshots[1][-1]["content"])

    def test_model_can_finish_after_repetition_warning(self):
        response = ModelResponse(AssistantMessage(tool_calls=(ToolCall("r", "read", "{}"),)), "tool_calls")
        provider = FakeProvider([response, response, ModelResponse(AssistantMessage("done"), "stop")])
        events = []
        result = AgentRuntime(provider).run(
            replace(config(), max_identical_tool_calls=3), [],
            [Tool("read", "read", {}, lambda: "same")],
            on_event=lambda kind, data: events.append(kind),
        )
        self.assertEqual(result.output, "done")
        self.assertIn("tool_repetition_warning", events)
        self.assertIn("请勿继续提交完全相同的参数", provider.message_snapshots[2][-1]["content"])

    def test_repetition_recovery_context_reaches_next_request_once(self):
        response = ModelResponse(
            AssistantMessage(tool_calls=(ToolCall("w", "write", '{"value":1}'),)), "tool_calls",
        )
        provider = FakeProvider([response, response, ModelResponse(AssistantMessage("done"), "stop")])
        recovery_calls, events = [], []

        def recover(arguments):
            recovery_calls.append(arguments)
            return "当前正文与候选编号：E17"

        result = AgentRuntime(provider).run(
            replace(config(), max_identical_tool_calls=3), [],
            [Tool("write", "write", {}, lambda value: "[错误] unchanged",
                  recovery_handler=recover)],
            on_event=lambda kind, data: events.append((kind, data)),
        )

        self.assertEqual(recovery_calls, [{"value": 1}])
        feedback = provider.message_snapshots[2][-1]["content"]
        self.assertIn("请勿继续提交完全相同的参数", feedback)
        self.assertIn("当前正文与候选编号：E17", feedback)
        self.assertEqual(result.steps, 3)
        self.assertEqual(sum(kind == "tool_recovery_provided" for kind, _ in events), 1)

    def test_recovery_context_and_callback_failure_do_not_reset_repetition(self):
        response = ModelResponse(
            AssistantMessage(tool_calls=(ToolCall("w", "write", '{"value":1}'),)), "tool_calls",
        )
        for callback_fails in (False, True):
            with self.subTest(callback_fails=callback_fails):
                provider = FakeProvider([response] * 3)
                recovery_calls, events = [], []

                def recover(arguments):
                    recovery_calls.append(arguments)
                    if callback_fails:
                        raise ValueError("context unavailable")
                    return "只读恢复上下文"

                with self.assertRaises(AgentLoopStalled) as caught:
                    AgentRuntime(provider).run(
                        replace(config(), max_identical_tool_calls=3), [],
                        [Tool("write", "write", {}, lambda value: "[错误] unchanged",
                              recovery_handler=recover)],
                        on_event=lambda kind, data: events.append((kind, data)),
                    )

                self.assertEqual(caught.exception.steps, 3)
                self.assertEqual(caught.exception.failed_tool_calls, 3)
                self.assertEqual(caught.exception.tool_counts, {"write": 3})
                self.assertEqual(recovery_calls, [{"value": 1}])
                self.assertIn("请勿继续提交完全相同的参数", provider.message_snapshots[2][-1]["content"])
                self.assertIn(
                    "tool_recovery_failed" if callback_fails else "tool_recovery_provided",
                    [kind for kind, _ in events],
                )

    def test_recovery_context_does_not_count_as_write_progress(self):
        response = ModelResponse(AssistantMessage(tool_calls=(ToolCall("w", "write", "{}"),)), "tool_calls")
        recovery_calls = []
        provider = FakeProvider([response] * 3)
        with self.assertRaises(MaxIterationsExceeded) as caught:
            AgentRuntime(provider).run(
                replace(config(), soft_iteration_limit=2, progress_grace_iterations=1,
                        max_identical_tool_calls=3), [],
                [Tool("write", "write", {}, lambda: "[错误] unchanged",
                      recovery_handler=lambda arguments: recovery_calls.append(arguments) or "只读上下文")],
            )

        self.assertEqual(caught.exception.steps, 2)
        self.assertEqual(recovery_calls, [{}])
        self.assertEqual(len(provider.requests), 2)

    def test_successful_repeated_calls_do_not_invoke_failure_recovery(self):
        response = ModelResponse(AssistantMessage(tool_calls=(ToolCall("r", "read", "{}"),)), "tool_calls")
        provider = FakeProvider([response, response, ModelResponse(AssistantMessage("done"), "stop")])
        recovery_calls = []
        result = AgentRuntime(provider).run(
            replace(config(), max_identical_tool_calls=3), [],
            [Tool("read", "read", {}, lambda: "unchanged",
                  recovery_handler=lambda arguments: recovery_calls.append(arguments) or "unused")],
        )

        self.assertEqual(result.output, "done")
        self.assertEqual(recovery_calls, [])

    def test_successful_repeated_calls_opt_in_to_recovery_before_the_next_request(self):
        response = ModelResponse(
            AssistantMessage(tool_calls=(ToolCall("s", "search", '{"keywords":["张三"]}'),)),
            "tool_calls", {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
        )
        provider = FakeProvider([response, response, ModelResponse(AssistantMessage("done"), "stop")])
        recovery_calls, events = [], []

        def recover(arguments):
            recovery_calls.append(arguments.copy())
            arguments["keywords"] = ["修改回调内副本"]
            return "已找到实体 E1，请读取；未覆盖词请单独补查。"

        result = AgentRuntime(provider).run(
            replace(config(), max_identical_tool_calls=3), [],
            [Tool("search", "search", {}, lambda keywords: "same matches",
                  recovery_handler=recover, recover_successful_repetition=True)],
            on_event=lambda kind, data: events.append((kind, data)),
        )

        self.assertEqual(result.output, "done")
        self.assertEqual(result.steps, 3)
        self.assertEqual(result.usage["prompt_tokens"], 200)
        self.assertEqual(result.usage["completion_tokens"], 20)
        self.assertEqual(recovery_calls, [{"keywords": ["张三"]}])
        feedback = provider.message_snapshots[2][-1]["content"]
        self.assertIn("请勿继续提交完全相同的参数", feedback)
        self.assertIn("已找到实体 E1，请读取", feedback)
        self.assertEqual(sum(kind == "tool_recovery_provided" for kind, _ in events), 1)
        assistant = next(message for message in provider.message_snapshots[2]
                         if message.get("tool_calls"))
        self.assertEqual(json.loads(assistant["tool_calls"][0]["function"]["arguments"]),
                         {"keywords": ["张三"]})

    def test_successful_recovery_and_callback_failure_preserve_the_repetition_limit(self):
        response = ModelResponse(
            AssistantMessage(tool_calls=(ToolCall("s", "search", '{"keywords":["张三"]}'),)), "tool_calls",
        )
        for callback_fails in (False, True):
            with self.subTest(callback_fails=callback_fails):
                provider = FakeProvider([response] * 3)
                recovery_calls, events = [], []

                def recover(arguments):
                    recovery_calls.append(arguments)
                    if callback_fails:
                        raise ValueError("context unavailable")
                    return "已有命中，请读取实体。"

                with self.assertRaises(AgentLoopStalled) as caught:
                    AgentRuntime(provider).run(
                        replace(config(), max_identical_tool_calls=3), [],
                        [Tool("search", "search", {}, lambda keywords: "same matches",
                              recovery_handler=recover, recover_successful_repetition=True)],
                        on_event=lambda kind, data: events.append((kind, data)),
                    )

                self.assertEqual(caught.exception.steps, 3)
                self.assertEqual(caught.exception.successful_tool_calls, 3)
                self.assertEqual(caught.exception.failed_tool_calls, 0)
                self.assertEqual(caught.exception.repeated_tool_calls, 2)
                self.assertEqual(caught.exception.tool_counts, {"search": 3})
                self.assertEqual(recovery_calls, [{"keywords": ["张三"]}])
                self.assertEqual(sum(kind == "tool_repetition_warning" for kind, _ in events), 1)
                self.assertIn("tool_recovery_failed" if callback_fails else "tool_recovery_provided",
                              [kind for kind, _ in events])

    def test_successful_repetition_recovery_does_not_extend_the_soft_budget(self):
        response = ModelResponse(
            AssistantMessage(tool_calls=(ToolCall("s", "search", "{}"),)), "tool_calls",
        )
        for callback_fails in (False, True):
            with self.subTest(callback_fails=callback_fails):
                provider, recovery_calls = FakeProvider([response] * 3), []

                def recover(arguments):
                    recovery_calls.append(arguments)
                    if callback_fails:
                        raise ValueError("context unavailable")
                    return "只读上下文不会成为写入进展。"

                with self.assertRaises(MaxIterationsExceeded) as caught:
                    AgentRuntime(provider).run(
                        replace(config(), soft_iteration_limit=2, progress_grace_iterations=1,
                                max_identical_tool_calls=3), [],
                        [Tool("search", "search", {}, lambda: "same matches",
                              recovery_handler=recover, recover_successful_repetition=True)],
                    )

                self.assertEqual(caught.exception.steps, 2)
                self.assertEqual(recovery_calls, [{}])
                self.assertEqual(len(provider.requests), 2)

    def test_duplicate_calls_in_one_response_allow_model_to_receive_warning(self):
        calls = tuple(ToolCall(str(n), "read", "{}") for n in range(3))
        provider = FakeProvider([ModelResponse(AssistantMessage(tool_calls=calls), "tool_calls"),
                                 ModelResponse(AssistantMessage("done"), "stop")])
        result = AgentRuntime(provider).run(
            replace(config(), max_identical_tool_calls=3), [],
            [Tool("read", "read", {}, lambda: "same")],
        )
        self.assertEqual(result.output, "done")
        self.assertIn("请勿继续提交完全相同的参数", provider.message_snapshots[1][-1]["content"])

    def test_changed_read_results_reset_repetition_count(self):
        response = ModelResponse(AssistantMessage(tool_calls=(ToolCall("r", "read", "{}"),)), "tool_calls")
        values = iter(["old", "old", "updated", "updated"])
        provider = FakeProvider([response] * 4 + [ModelResponse(AssistantMessage("done"), "stop")])
        result = AgentRuntime(provider).run(
            replace(config(), max_identical_tool_calls=3), [],
            [Tool("read", "read", {}, lambda: next(values))],
        )
        self.assertEqual(result.output, "done")

    def test_write_progress_resets_read_repetition_even_within_same_step(self):
        read = ToolCall("r", "read", "{}")
        write = ToolCall("w", "write", "{}")
        responses = [ModelResponse(AssistantMessage(tool_calls=calls), "tool_calls")
                     for calls in [(read,), (write, read), (read,)]]
        provider = FakeProvider(responses + [ModelResponse(AssistantMessage("done"), "stop")])
        result = AgentRuntime(provider).run(
            replace(config(), max_identical_tool_calls=3), [],
            [Tool("read", "read", {}, lambda: "same"),
             Tool("write", "write", {}, lambda: "changed", progress_detector=lambda r: True)],
        )
        self.assertEqual(result.output, "done")

    def test_interleaved_no_progress_calls_still_stop(self):
        responses = [ModelResponse(AssistantMessage(tool_calls=(ToolCall("r", name, "{}"),)), "tool_calls")
                     for name in ["read", "other", "read", "other", "read"]]
        with self.assertRaises(AgentLoopStalled):
            AgentRuntime(FakeProvider(responses)).run(
                replace(config(), max_identical_tool_calls=3), [],
                [Tool(name, name, {}, lambda: "same") for name in ["read", "other"]],
            )

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

    def test_five_validation_repairs_can_succeed_or_exhaust(self):
        for succeeds in (True, False):
            with self.subTest(succeeds=succeeds):
                outputs = ["invalid"] * 5 + ["done" if succeeds else "invalid"]
                provider = FakeProvider([ModelResponse(AssistantMessage(value), "stop") for value in outputs])
                events = []
                def run():
                    return AgentRuntime(provider).run(
                        replace(config(), max_iterations=10, max_validation_retries=5), [],
                        output_validator=lambda value: None if value == "done" else "链接目标不存在",
                        on_event=lambda kind, data: events.append((kind, data)),
                    )
                if succeeds:
                    self.assertEqual(run().output, "done")
                else:
                    with self.assertRaises(OutputValidationError):
                        run()
                self.assertEqual(len(provider.requests), 6)
                for messages in provider.message_snapshots[1:]:
                    self.assertIn("链接目标不存在", messages[-1]["content"])
                rejected = [data for kind, data in events if kind == "output_rejected"]
                self.assertEqual(rejected[4]["attempt"], 5)
                self.assertTrue(rejected[4]["will_retry"])
                if not succeeds:
                    self.assertFalse(rejected[5]["will_retry"])

    def test_five_json_repairs_can_succeed_or_exhaust(self):
        malformed = ModelResponse(AssistantMessage(tool_calls=(ToolCall("bad", "write", '{"value":'),)), "tool_calls")
        valid = ModelResponse(AssistantMessage(tool_calls=(ToolCall("ok", "write", '{"value":1}'),)), "tool_calls")
        for succeeds in (True, False):
            with self.subTest(succeeds=succeeds):
                provider = FakeProvider([malformed] * 5 + ([valid, ModelResponse(AssistantMessage("done"), "stop")]
                                                          if succeeds else [malformed]))
                executed = []
                def run():
                    return AgentRuntime(provider).run(
                        replace(config(), max_iterations=10, max_tool_argument_retries=5), [],
                        [Tool("write", "write", {}, lambda value: executed.append(value) or "saved")],
                    )
                if succeeds:
                    self.assertEqual(run().output, "done")
                    self.assertEqual(executed, [1])
                else:
                    with self.assertRaises(MalformedToolArguments) as caught:
                        run()
                    self.assertEqual(caught.exception.limit, 5)
                    self.assertEqual(len(provider.requests), 6)
                    self.assertEqual(executed, [])

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
