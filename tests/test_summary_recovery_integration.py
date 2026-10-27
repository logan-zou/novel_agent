import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from src.harness.runtime import AssistantMessage, ModelResponse, ToolCall
from src.novel.builder import run_extraction_agent
from src.novel.corpus import CorpusStore
from src.novel.knowledge import BuildSession, KnowledgeStore


_REF = "characters/姜述.md"
_TOKEN = "<｜begin▁of▁sentence｜>"
_BODY = "## 经历\n姜述执掌齐国，与姜望议事。\n\n## 治理\n任用群臣。"
_SUMMARY = "齐国皇帝，执掌齐国政事。"
_CORRECTED_SUMMARY = "齐国皇帝，执掌齐国政事并与姜望议事。"


def _entity(summary):
    return {
        "kind": "character", "name": "姜述", "content": _BODY,
        "summary": summary, "aliases": ["齐帝"], "source_chapters": [1],
    }


def _call(call_id, name, **arguments):
    return ModelResponse(
        AssistantMessage(tool_calls=(ToolCall(
            call_id, name, json.dumps(arguments, ensure_ascii=False),
        ),)),
        "tool_calls",
    )


def _result(request, call_id):
    message = next(
        message for message in reversed(request.messages)
        if message.get("role") == "tool" and message.get("tool_call_id") == call_id
    )
    return json.loads(message["content"].removeprefix("[错误]").strip())


def _metadata(document):
    return json.loads(document.splitlines()[0].removeprefix("<!-- novel-agent: ").removesuffix(" -->"))


def _body(document):
    return document.partition("\n\n")[2].rpartition("\n\n来源章节：")[0]


class _DynamicProvider:
    def __init__(self, responses, before_response):
        self.responses = iter(responses)
        self.before_response = before_response
        self.message_snapshots = []

    def complete(self, request):
        self.before_response()
        self.message_snapshots.append(deepcopy(request.messages))
        response = next(self.responses)
        return response(request) if callable(response) else response


class SummaryRecoveryIntegrationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.batch = [{
            "chapter_no": 1, "title": "齐国朝议",
            "text": "姜述执掌齐国，与姜望议事，任用群臣。",
        }]
        CorpusStore(self.root).save_chapters(self.batch)
        baseline = BuildSession(KnowledgeStore(self.root))
        baseline.upsert_entity(**_entity(_SUMMARY))
        baseline.commit()
        self.original_document = KnowledgeStore(self.root).read_entity(_REF)
        self.original_files = self._snapshot()
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        stack.enter_context(patch("src.novel.builder.create_trace", return_value=None))
        stack.enter_context(patch("src.novel.builder.finish_trace"))
        stack.enter_context(patch("src.novel.builder.record_usage"))
        stack.enter_context(patch(
            "src.novel.builder.get_provider", side_effect=AssertionError("禁止真实模型调用"),
        ))

    def _snapshot(self):
        return {
            str(path.relative_to(self.root)): path.read_bytes()
            for path in self.root.rglob("*") if path.is_file()
        }

    def _run(self, responses):
        provider = _DynamicProvider(
            responses, lambda: self.assertEqual(self._snapshot(), self.original_files),
        )
        run_extraction_agent(
            self.batch, "summary-recovery-integration-test", str(self.root),
            provider=provider, model="test-model",
        )
        return provider

    def _assert_summary_read(self, item, expected_summary, expected_document):
        self.assertEqual(item["field"], "summary")
        self.assertEqual(item["summary"], expected_summary)
        self.assertEqual(item["content"], expected_summary)
        self.assertEqual(
            item["source_version"], hashlib.sha256(expected_document.encode("utf-8")).hexdigest(),
        )

    def _assert_only_summary_changed(self, expected_summary):
        document = KnowledgeStore(self.root).read_entity(_REF)
        self.assertEqual(_body(document), _body(self.original_document))
        before = _metadata(self.original_document)
        after = _metadata(document)
        self.assertEqual(after.pop("summary"), expected_summary)
        before.pop("summary")
        self.assertEqual(after, before)
        self.assertEqual(
            document.rpartition("\n\n来源章节：")[2],
            self.original_document.rpartition("\n\n来源章节：")[2],
        )
        record = next(item for item in KnowledgeStore(self.root).catalog_records() if item.ref == _REF)
        self.assertEqual(record.summary, expected_summary)

    def test_protocol_token_rejection_preserves_original_entity_until_clean_retry(self):
        polluted = {**_entity(f"齐国皇帝{_TOKEN}，执掌齐国政事。"), "content": "错误的新正文。"}

        def read_after_rejection(request):
            rejected = _result(request, "polluted-write")
            self.assertEqual(rejected["failed_count"], 1)
            self.assertEqual(rejected["changed_count"], 0)
            self.assertIn(_TOKEN, rejected["results"][0]["result"])
            return _call("read-original", "read_entities", requests=[
                {"ref": _REF, "field": "summary"},
                {"ref": _REF, "field": "content"},
                {"ref": _REF},
            ])

        def retry_after_read(request):
            items = _result(request, "read-original")["entities"]
            self._assert_summary_read(items[0], _SUMMARY, self.original_document)
            self.assertEqual(items[1]["content"], _BODY)
            self.assertEqual(items[2]["content"], self.original_document)
            return _call("clean-write", "upsert_entities", entities=[_entity(_CORRECTED_SUMMARY)])

        def finish_after_clean_retry(request):
            written = _result(request, "clean-write")
            self.assertEqual(written["failed_count"], 0)
            return _call("finish", "finish_batch", summary="已按真实摘要修正并完成本批")

        provider = self._run([
            _call("polluted-write", "upsert_entities", entities=[polluted]),
            read_after_rejection,
            retry_after_read,
            finish_after_clean_retry,
        ])

        self.assertEqual(len(provider.message_snapshots), 4)
        self._assert_only_summary_changed(_CORRECTED_SUMMARY)
        self.assertNotIn(_TOKEN, KnowledgeStore(self.root).read_entity(_REF))

    def test_summary_match_failure_exposes_current_value_and_recovers_with_complete_old_text(self):
        def read_after_bad_patch(request):
            rejected = _result(request, "bad-summary-patch")
            self.assertEqual(rejected["failed_count"], 1)
            self.assertEqual(rejected["changed_count"], 0)
            feedback = rejected["results"][0]["result"]
            self.assertIn(_SUMMARY, feedback)
            self.assertIn("summary", feedback)
            return _call("read-summary", "read_entities", requests=[{"ref": _REF, "field": "summary"}])

        def patch_after_summary_read(request):
            item = _result(request, "read-summary")["entities"][0]
            self._assert_summary_read(item, _SUMMARY, self.original_document)
            return _call("correct-summary-patch", "patch_entities", entities=[{
                "ref": _REF, "field": "summary", "replacements": [{
                    "old_text": item["summary"], "new_text": _CORRECTED_SUMMARY,
                }],
            }])

        def inspect_after_correct_patch(request):
            written = _result(request, "correct-summary-patch")
            self.assertEqual(written["failed_count"], 0)
            self.assertEqual(written["changed_count"], 1)
            return _call("inspect", "inspect_repairs")

        def finish_after_failure_resolution(request):
            report = _result(request, "inspect")
            self.assertEqual(report["patch_failures"], [])
            self.assertEqual(report["validation_errors"], [])
            return _call("finish", "finish_batch", summary="已修正摘要并完成本批")

        provider = self._run([
            _call("draft", "upsert_entities", entities=[_entity(_SUMMARY)]),
            _call("bad-summary-patch", "patch_entities", entities=[{
                "ref": _REF, "field": "summary", "replacements": [{
                    "old_text": "姜述是齐国皇帝", "new_text": _CORRECTED_SUMMARY,
                }],
            }]),
            read_after_bad_patch,
            patch_after_summary_read,
            inspect_after_correct_patch,
            finish_after_failure_resolution,
        ])

        self.assertEqual(len(provider.message_snapshots), 6)
        self._assert_only_summary_changed(_CORRECTED_SUMMARY)

    def test_historical_polluted_summary_is_readable_and_clean_full_write_preserves_body(self):
        polluted_summary = f"齐国皇帝{_TOKEN}，执掌齐国政事。"
        metadata = _metadata(self.original_document)
        metadata["summary"] = polluted_summary
        _, _, remainder = self.original_document.partition("\n")
        polluted_document = (
            f"<!-- novel-agent: {json.dumps(metadata, ensure_ascii=False, separators=(',', ':'))} -->"
            f"\n{remainder}"
        )
        (self.root / _REF).write_text(polluted_document, encoding="utf-8")
        KnowledgeStore(self.root).rebuild_catalog()
        self.original_document = polluted_document
        self.original_files = self._snapshot()

        def clean_after_reading_actual_metadata(request):
            items = _result(request, "read-history")["entities"]
            self._assert_summary_read(items[0], polluted_summary, polluted_document)
            self.assertEqual(items[1]["content"], polluted_summary)
            self.assertEqual(items[2]["content"], polluted_document)
            return _call("clean-history", "upsert_entities", entities=[_entity(_SUMMARY)])

        def finish_after_cleaning_history(request):
            self.assertEqual(_result(request, "clean-history")["failed_count"], 0)
            return _call("finish", "finish_batch", summary="历史摘要已修复并完成本批")

        provider = self._run([
            _call("read-history", "read_entities", requests=[
                {"ref": _REF, "field": "summary"},
                {"ref": _REF, "section": "summary"},
                {"ref": _REF},
            ]),
            clean_after_reading_actual_metadata,
            finish_after_cleaning_history,
        ])

        self.assertEqual(len(provider.message_snapshots), 3)
        self._assert_only_summary_changed(_SUMMARY)
        self.assertNotIn(_TOKEN, KnowledgeStore(self.root).read_entity(_REF))


if __name__ == "__main__":
    unittest.main()
