import contextlib
import io
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from src.harness.runtime import (
    AgentLoopStalled,
    AssistantMessage,
    MalformedToolArguments,
    ModelResponse,
    OutputValidationError,
    ToolCall,
)
from src.novel.builder import run_extraction_agent
from src.novel.corpus import CorpusStore
from src.novel.knowledge import BuildSession, KnowledgeStore


def _entity(name, content):
    return {
        "kind": "character",
        "name": name,
        "summary": f"{name}的经历",
        "content": content,
        "source_chapters": [1],
    }


def _call(call_id, name, **arguments):
    return ModelResponse(
        AssistantMessage(
            tool_calls=(
                ToolCall(
                    call_id,
                    name,
                    json.dumps(arguments, ensure_ascii=False),
                ),
            ),
        ),
        "tool_calls",
        {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
    )


def _write(call_id, *entities):
    return _call(call_id, "upsert_entities", entities=entities)


def _replacement(ref, old_text, new_text):
    return {
        "ref": ref,
        "replacements": [{"old_text": old_text, "new_text": new_text}],
    }


def _finish():
    return ModelResponse(
        AssistantMessage("本章实体已整理完成。"),
        "stop",
        {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
    )


class _ScriptedProvider:
    def __init__(self, responses, before_response):
        self.responses = iter(responses)
        self.before_response = before_response
        self.calls = 0
        self.message_snapshots = []

    def complete(self, request):
        self.before_response()
        self.calls += 1
        self.message_snapshots.append(deepcopy(request.messages))
        return next(self.responses)


class BuilderRepairTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.batch = [{"chapter_no": 1, "title": "初遇", "text": "张三结识李四。"}]
        CorpusStore(self.root).save_chapters(self.batch)
        session = BuildSession(KnowledgeStore(self.root))
        session.upsert_entity(**_entity("张三", "此前尚未结识李四。"))
        session.commit()
        self.original_files = self._snapshot()
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(patch("src.novel.builder.create_trace", return_value=None))
        self.stack.enter_context(patch("src.novel.builder.finish_trace"))
        self.stack.enter_context(
            patch("src.novel.builder.get_provider", side_effect=AssertionError("禁止真实模型调用"))
        )
        self.record_usage = self.stack.enter_context(patch("src.novel.builder.record_usage"))

    def _snapshot(self):
        return {
            str(path.relative_to(self.root)): path.read_bytes()
            for path in self.root.rglob("*")
            if path.is_file()
        }

    def _run(self, responses):
        provider = _ScriptedProvider(
            responses,
            lambda: self.assertEqual(self._snapshot(), self.original_files),
        )
        usage = run_extraction_agent(
            self.batch,
            "repair-integration-test",
            str(self.root),
            provider=provider,
            model="test-model",
        )
        return provider, usage

    def test_patch_feedback_shows_actual_noop_parameters_before_corrected_retry(self):
        self.batch = [{
            "chapter_no": 1,
            "title": "驱山",
            "text": "张三称，驱山徙岳不用多说，驱山徙岳他自认不比袁首差。",
        }]
        CorpusStore(self.root).save_chapters(self.batch)
        self.original_files = self._snapshot()
        unchanged_text = "驱山徒岳他自认不比袁首差"
        provider, usage = self._run([
            _write("draft", _entity("张三", f"驱山驱山不用多说；{unchanged_text}。")),
            _call("invalid-patch", "patch_entities", entities=[{
                "ref": "characters/张三",
                "replacements": [
                    {"old_text": "驱山驱山不用多说", "new_text": "驱山徒岳不用多说"},
                    {"old_text": unchanged_text, "new_text": unchanged_text},
                ],
            }]),
            _call("corrected-patch", "patch_entities", entities=[{
                "ref": "characters/张三",
                "replacements": [
                    {"old_text": "驱山驱山不用多说", "new_text": "驱山徙岳不用多说"},
                    {"old_text": unchanged_text, "new_text": "驱山徙岳他自认不比袁首差"},
                ],
            }]),
            _finish(),
        ])

        rejected = next(
            message["content"]
            for message in provider.message_snapshots[2]
            if message.get("tool_call_id") == "invalid-patch"
        )
        payload = json.loads(rejected.removeprefix("[错误]").strip())
        feedback = payload["results"][0]["result"]
        self.assertEqual(payload["changed_count"], 0)
        self.assertEqual(payload["failed_count"], 1)
        self.assertIn("replacements[1]", feedback)
        self.assertIn("old_text", feedback)
        self.assertIn("new_text", feedback)
        self.assertGreaterEqual(feedback.count(unchanged_text), 2)
        for instruction in ("原文", "原样重试", "所有替换均未应用", "重新提交"):
            self.assertIn(instruction, feedback)
        self.assertEqual(provider.calls, 4)
        self.assertEqual(usage["total_tokens"], 440)
        self.assertEqual(self.record_usage.call_count, 4)
        document = KnowledgeStore(self.root).read_entity("characters/张三")
        self.assertIn("驱山徙岳不用多说；驱山徙岳他自认不比袁首差。", document)
        self.assertNotIn("驱山驱山", document)
        self.assertNotIn("驱山徒岳", document)

    def test_unchanged_patch_retries_still_stop_without_committing(self):
        unchanged_text = "驱山徒岳他自认不比袁首差"
        failed_patch = {
            "ref": "characters/张三",
            "replacements": [
                {"old_text": "驱山驱山不用多说", "new_text": "驱山徒岳不用多说"},
                {"old_text": unchanged_text, "new_text": unchanged_text},
            ],
        }
        provider = _ScriptedProvider(
            [
                _write("draft", _entity("张三", f"驱山驱山不用多说；{unchanged_text}。")),
                *[
                    _call(f"unchanged-patch-{index}", "patch_entities", entities=[failed_patch])
                    for index in range(3)
                ],
            ],
            lambda: self.assertEqual(self._snapshot(), self.original_files),
        )

        with self.assertRaises(AgentLoopStalled) as caught:
            run_extraction_agent(
                self.batch,
                "repair-integration-test",
                str(self.root),
                provider=provider,
                model="test-model",
            )

        self.assertEqual(caught.exception.steps, 4)
        self.assertEqual(caught.exception.limit, 3)
        self.assertEqual(caught.exception.tool_counts["patch_entities"], 3)
        self.assertIn("相同工具调用达到 3 次", str(caught.exception))
        self.assertEqual(provider.calls, 4)
        self.assertEqual(self.record_usage.call_count, 4)
        self.assertEqual(provider.message_snapshots[-1][-1]["role"], "user")
        self.assertIn("请勿继续提交完全相同的参数", provider.message_snapshots[-1][-1]["content"])
        self.assertEqual(self._snapshot(), self.original_files)

    def test_repair_tool_round_commits_without_another_model_response(self):
        provider, usage = self._run(
            [
                _write("draft", _entity("张三", "结识[[characters/李四]]。")),
                _finish(),
                _write("repair", _entity("李四", "结识[[characters/张三]]。")),
            ]
        )

        self.assertEqual(provider.calls, 3)
        self.assertEqual(usage["total_tokens"], 330)
        self.assertEqual(self.record_usage.call_count, 3)
        store = KnowledgeStore(self.root)
        self.assertIn("结识[[characters/李四]]", store.read_entity("characters/张三"))
        self.assertIn("结识[[characters/张三]]", store.read_entity("characters/李四"))
        self.assertEqual({record.name for record in store.catalog_records()}, {"张三", "李四"})

    def test_incomplete_repair_preserves_existing_knowledge(self):
        with self.assertRaises(OutputValidationError) as caught:
            self._run(
                [
                    _write("draft", _entity("张三", "结识[[characters/李四]]。")),
                    _finish(),
                    _write("partial-repair", _entity("李四", "结识[[characters/王五]]。")),
                    *[_finish() for _ in range(5)],
                ]
            )

        self.assertIn("链接目标不存在 characters/王五", str(caught.exception))
        self.assertEqual(self._snapshot(), self.original_files)
        self.assertEqual(self.record_usage.call_count, 8)

    def test_failed_entity_repair_must_be_retried_before_automatic_completion(self):
        provider, _ = self._run(
            [
                _write(
                    "draft",
                    _entity("张三", "结识[[characters/赵六]]。"),
                    _entity("李四", "经历待修正。"),
                ),
                _finish(),
                _call(
                    "partial-repair",
                    "patch_entities",
                    entities=[
                        _replacement("characters/张三", "赵六", "李四"),
                        _replacement("characters/李四", "不存在的旧正文", "曾到访青山。"),
                    ],
                ),
                _call(
                    "read",
                    "read_entities",
                    requests=[{"ref": "characters/李四"}],
                ),
                _write("unrelated", _entity("王五", "独自修行。")),
                _call(
                    "retry-failed-entity",
                    "patch_entities",
                    entities=[_replacement("characters/李四", "经历待修正。", "曾到访青山。")],
                ),
            ]
        )

        self.assertEqual(provider.calls, 6)
        self.assertEqual(self.record_usage.call_count, 6)
        store = KnowledgeStore(self.root)
        self.assertIn("结识[[characters/李四]]", store.read_entity("characters/张三"))
        self.assertIn("曾到访青山", store.read_entity("characters/李四"))
        self.assertEqual({record.name for record in store.catalog_records()}, {"张三", "李四", "王五"})

    def test_unidentified_failed_write_requires_explicit_completion(self):
        provider, _ = self._run(
            [
                _write("draft", _entity("张三", "结识[[characters/李四]]。")),
                _finish(),
                _call(
                    "unidentified-failure",
                    "patch_entities",
                    entities=[
                        _replacement("characters/张三", "[[characters/李四]]", "李四"),
                        {},
                    ],
                ),
                _call(
                    "read",
                    "read_entities",
                    requests=[{"ref": "characters/张三"}],
                ),
                _write("unrelated", _entity("王五", "独自修行。")),
                _finish(),
            ]
        )

        self.assertEqual(provider.calls, 6)
        self.assertEqual(self.record_usage.call_count, 6)
        store = KnowledgeStore(self.root)
        self.assertIn("结识李四", store.read_entity("characters/张三"))
        self.assertEqual({record.name for record in store.catalog_records()}, {"张三", "王五"})

    def test_missing_fields_can_be_repaired_after_three_consecutive_failures(self):
        incomplete = {"content": "群山中的修行道场。", "source_chapters": [1]}
        identified = {**incomplete, "kind": "setting", "name": "灵山"}
        with_domain = {**identified, "domain": "世界构成"}
        complete = {**with_domain, "summary": "修行道场"}
        provider, _ = self._run(
            [
                *[
                    _call(f"search-{index}", "search_entities", keywords=[f"人物{index}"])
                    for index in range(8)
                ],
                _write("missing-identity", incomplete),
                _write("missing-domain", identified),
                _write("missing-summary", with_domain),
                _write("complete-arguments", complete),
                _finish(),
            ]
        )

        self.assertEqual(provider.calls, 13)
        self.assertEqual(self.record_usage.call_count, 13)
        first_error = next(
            message["content"]
            for message in provider.message_snapshots[9]
            if message.get("tool_call_id") == "missing-identity"
        )
        for missing_field in ("kind", "name", "summary"):
            self.assertIn(missing_field, first_error)
        second_error = next(
            message["content"]
            for message in provider.message_snapshots[10]
            if message.get("tool_call_id") == "missing-domain"
        )
        self.assertIn("domain", second_error)
        self.assertIn("summary", second_error)
        self.assertIn(
            "群山中的修行道场",
            KnowledgeStore(self.root).read_entity("settings/世界构成/灵山"),
        )

    def test_exhausted_field_repairs_preserve_existing_knowledge(self):
        incomplete = _entity("李四", "结识张三。")
        incomplete.pop("summary")
        with self.assertRaises(MalformedToolArguments) as caught:
            self._run(
                [
                    _write("valid-draft", _entity("张三", "结识李四。")),
                    *[_write(f"invalid-{index}", incomplete) for index in range(6)],
                ]
            )

        self.assertEqual(caught.exception.steps, 7)
        self.assertEqual(caught.exception.limit, 5)
        self.assertEqual(self.record_usage.call_count, 7)
        self.assertEqual(self._snapshot(), self.original_files)

    def test_partial_argument_failure_preserves_success_and_retries_only_failed_entity(self):
        incomplete = _entity("李四", "结识[[characters/张三]]。")
        incomplete.pop("summary")
        original_upsert = BuildSession.upsert_entity
        successful_writes = []

        def track_upsert(session, **arguments):
            result = original_upsert(session, **arguments)
            if result.startswith("[已暂存]"):
                successful_writes.append(arguments["name"])
            return result

        with patch.object(BuildSession, "upsert_entity", new=track_upsert):
            provider, _ = self._run(
                [
                    _write(
                        "partial",
                        _entity("张三", "结识[[characters/李四]]。"),
                        incomplete,
                    ),
                    _finish(),
                    _write("retry-failed-only", {**incomplete, "summary": "张三的朋友"}),
                ]
            )

        self.assertEqual(provider.calls, 3)
        self.assertEqual(successful_writes, ["张三", "李四"])
        partial_result = next(
            message["content"]
            for message in provider.message_snapshots[1]
            if message.get("tool_call_id") == "partial"
        )
        payload = json.loads(partial_result.removeprefix("[错误]").strip())
        self.assertEqual(payload["staged_count"], 1)
        self.assertEqual(payload["failed_count"], 1)
        store = KnowledgeStore(self.root)
        self.assertIn("结识[[characters/李四]]", store.read_entity("characters/张三"))
        self.assertIn("结识[[characters/张三]]", store.read_entity("characters/李四"))


if __name__ == "__main__":
    unittest.main()
