import contextlib
import io
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from src.harness.runtime import AgentLoopStalled, AssistantMessage, ModelResponse, ToolCall
from src.novel.builder import run_extraction_agent
from src.novel.corpus import CorpusStore
from src.novel.knowledge import BuildSession, KnowledgeStore


def _call(call_id, name, **arguments):
    return ModelResponse(AssistantMessage(tool_calls=(ToolCall(
        call_id, name, json.dumps(arguments, ensure_ascii=False),
    ),)), "tool_calls")


def _result(request, call_id):
    return json.loads(next(
        message["content"] for message in reversed(request.messages)
        if message.get("role") == "tool" and message.get("tool_call_id") == call_id
    ))


class _Provider:
    def __init__(self, responses, before_response):
        self.responses = iter(responses)
        self.before_response = before_response
        self.messages = []

    def complete(self, request):
        self.before_response()
        self.messages.append(deepcopy(request.messages))
        response = next(self.responses)
        return response(request) if callable(response) else response


class EntitySearchRecoveryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.batch = [{"chapter_no": 2857, "title": "征途", "text": "姬凤洲率景国军出征，吴询率秦国军参战。"}]
        CorpusStore(self.root).save_chapters(self.batch)
        session = BuildSession(KnowledgeStore(self.root))
        for name in ("姬凤洲", "吴询"):
            session.upsert_entity("character", name, "原有身份。", [2857], summary=f"{name}的身份")
        for index in range(40):
            session.upsert_entity(
                "event", f"关联事件{index:02}", "此前的相关事件。", [2857],
                summary="姬凤洲、吴询、景国、秦国相关事件。",
            )
        session.commit()
        self.original_files = self._snapshot()
        self.keywords = ["姬凤洲", "吴询", "景国", "秦国", "目录缺席词"]
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        stack.enter_context(patch("src.novel.builder.create_trace", return_value=None))
        stack.enter_context(patch("src.novel.builder.finish_trace"))
        stack.enter_context(patch("src.novel.builder.record_usage"))
        stack.enter_context(patch("src.novel.builder.get_provider", side_effect=AssertionError("禁止真实模型调用")))

    def _snapshot(self):
        return {str(path.relative_to(self.root)): path.read_bytes()
                for path in self.root.rglob("*") if path.is_file()}

    def _run(self, responses):
        self.provider = _Provider(responses, lambda: self.assertEqual(self._snapshot(), self.original_files))
        run_extraction_agent(
            self.batch, "entity-search-recovery-test", str(self.root), provider=self.provider, model="test-model",
        )
        return self.provider

    def test_successful_search_recovery_guides_read_write_and_explicit_commit(self):
        def read_after_warning(request):
            first, second = _result(request, "search-1"), _result(request, "search-2")
            self.assertEqual(first, second)
            self.assertEqual({item["name"] for item in second["matches"][:2]}, {"姬凤洲", "吴询"})
            self.assertEqual(second["total_matches"], 42)
            self.assertTrue(second["truncated"])
            self.assertEqual(second["unmatched_keywords"], ["目录缺席词"])
            self.assertEqual(second["uncovered_keywords"], [])
            feedback = request.messages[-1]["content"]
            self.assertIn("请勿继续提交完全相同的参数", feedback)
            recovery = json.loads(feedback.partition("当前修复上下文：\n")[2])
            self.assertIn("read_entities", recovery["instruction"])
            self.assertLessEqual(len(recovery["ready_to_read"]), 8)
            self.assertEqual(recovery["unmatched_keywords"], ["目录缺席词"])
            self.assertEqual(recovery["keywords_to_narrow"], [])
            self.assertEqual({item["name"] for item in recovery["ready_to_read"][:2]}, {"姬凤洲", "吴询"})
            return _call("read", "read_entities", requests=[
                {"ref": "characters/姬凤洲"}, {"ref": "characters/吴询"},
            ])

        def update_after_read(request):
            read = _result(request, "read")
            self.assertEqual(read["failed_count"], 0)
            self.assertTrue(all("原有身份。" in item["content"] for item in read["entities"]))
            return _call("write", "upsert_entities", entities=[
                {"kind": "character", "name": name, "content": "原有身份。\n" + detail,
                 "summary": detail, "source_chapters": [2857]}
                for name, detail in (("姬凤洲", "率景国军出征。"), ("吴询", "率秦国军参战。"))
            ])

        def finish_after_write(request):
            self.assertEqual(_result(request, "write")["changed_count"], 2)
            return _call("done", "finish_batch", summary="已读取并更新两名参战人物")

        provider = self._run([
            _call("search-1", "search_entities", keywords=self.keywords),
            _call("search-2", "search_entities", keywords=self.keywords),
            read_after_warning, update_after_read, finish_after_write,
        ])
        self.assertEqual(len(provider.messages), 5)
        self.assertIn("率景国军出征。", KnowledgeStore(self.root).read_entity("characters/姬凤洲"))
        self.assertIn("率秦国军参战。", KnowledgeStore(self.root).read_entity("characters/吴询"))

    def test_recovery_feedback_cannot_make_unchanged_search_loop_count_as_progress(self):
        with self.assertRaises(AgentLoopStalled) as caught:
            self._run([_call(f"search-{step}", "search_entities", keywords=self.keywords) for step in range(3)])
        self.assertEqual(caught.exception.steps, 3)
        self.assertEqual(caught.exception.tool_counts, {"search_entities": 3})
        self.assertEqual(caught.exception.successful_tool_calls, 3)
        self.assertEqual(caught.exception.failed_tool_calls, 0)
        self.assertEqual(self._snapshot(), self.original_files)
        self.assertIn("当前修复上下文", self.provider.messages[-1][-1]["content"])


if __name__ == "__main__":
    unittest.main()
