import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.harness.runtime import AssistantMessage, ModelResponse, ToolCall
from src.novel.corpus import CorpusStore
from src.novel.qa import run_qa_agent
from src.qa_agent import run_qa_agent as run_compatible_qa


class ScriptedProvider:
    def __init__(self):
        self.responses = iter([
            ModelResponse(
                AssistantMessage(tool_calls=(ToolCall(
                    "source-1", "search_source", json.dumps({"keywords": ["张三"]}),
                ),)),
                "tool_calls",
                {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            ),
            ModelResponse(
                AssistantMessage("张三是守灯人。[[cite:EV001]]"),
                "stop",
                {"prompt_tokens": 150, "completion_tokens": 10, "total_tokens": 160},
            ),
        ])

    def complete(self, request):
        return next(self.responses)


class QaUsageRecorderTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.kb = self.root / "book"
        CorpusStore(self.kb).save_chapters([
            {"chapter_no": 1, "title": "雾港", "text": "张三是守灯人。"},
        ])
        self.before = self.snapshot()
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch("src.novel.qa.create_trace", return_value=None))
        stack.enter_context(patch("src.novel.qa.finish_trace"))
        stack.enter_context(patch("src.novel.qa.get_provider", side_effect=AssertionError("no network")))
        self.default_recorder = stack.enter_context(patch("src.novel.qa.record_usage"))

    def snapshot(self):
        return {str(p.relative_to(self.kb)): p.read_bytes() for p in self.kb.rglob("*") if p.is_file()}

    def readonly(self):
        paths = list(self.kb.rglob("*")) + [self.kb]
        modes = {p: p.stat().st_mode & 0o777 for p in paths}
        self.addCleanup(lambda: [p.chmod(mode) for p, mode in modes.items()])
        for path in paths:
            path.chmod(0o555 if path.is_dir() else 0o444)

    def test_injected_recorder_receives_each_round_without_writing_readonly_knowledge(self):
        self.readonly()
        usages = []
        result = run_qa_agent(
            "谁是守灯人？", "book", str(self.kb),
            provider=ScriptedProvider(), model="scripted-test", usage_recorder=usages.append,
        )
        self.assertEqual([item["total_tokens"] for item in usages], [120, 160])
        self.assertEqual([item["cached_prompt_tokens"] for item in usages], [0, 100])
        self.assertEqual(result["usage"]["total_tokens"], 280)
        self.assertTrue(result["citation_validation"]["valid"])
        self.assertEqual(self.snapshot(), self.before)
        self.default_recorder.assert_not_called()

    def test_omitting_hook_keeps_existing_usage_recording(self):
        result = run_qa_agent(
            "谁是守灯人？", "book", str(self.kb), provider=ScriptedProvider(), model="scripted-test",
        )
        self.assertEqual(self.default_recorder.call_count, 2)
        self.assertEqual(self.default_recorder.call_args.args[:2], ("book", "qa"))
        self.assertEqual(result["usage"]["total_tokens"], 280)

    def test_compatible_entrypoint_supports_injected_recorder(self):
        usages = []
        with patch("src.qa_agent.KNOWLEDGE_BASE_DIR", str(self.root)):
            result = run_compatible_qa(
                "谁是守灯人？", "book", provider=ScriptedProvider(),
                model="scripted-test", usage_recorder=usages.append,
            )
        self.assertEqual(len(usages), 2)
        self.assertEqual(result["sources"], [1])
        self.default_recorder.assert_not_called()

    def test_recorder_error_is_reported_without_falling_back_to_knowledge_writes(self):
        def fail(usage):
            raise OSError("state volume is full")

        with self.assertRaisesRegex(OSError, "state volume is full"):
            run_qa_agent(
                "谁是守灯人？", "book", str(self.kb), provider=ScriptedProvider(),
                model="scripted-test", usage_recorder=fail,
            )
        self.default_recorder.assert_not_called()
        self.assertEqual(self.snapshot(), self.before)


if __name__ == "__main__":
    unittest.main()
