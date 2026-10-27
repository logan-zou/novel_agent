import contextlib
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


def _tool(call_id, name, **arguments):
    return ToolCall(call_id, name, json.dumps(arguments, ensure_ascii=False))


def _calls(*calls):
    return ModelResponse(AssistantMessage(tool_calls=calls), "tool_calls")


def _call(call_id, name, **arguments):
    return _calls(_tool(call_id, name, **arguments))


def _entity(name, content, summary="蒋去的经历"):
    return {
        "kind": "character", "name": name, "content": content,
        "summary": summary, "source_chapters": [1], "aliases": ["学塾先生"],
    }


def _result(request, call_id):
    message = next(
        message for message in reversed(request.messages)
        if message.get("role") == "tool" and message.get("tool_call_id") == call_id
    )
    return json.loads(message["content"].removeprefix("[错误]").strip())


def _body(document):
    return document.partition("\n\n")[2].rpartition("\n\n来源章节：")[0]


def _metadata(document):
    return json.loads(document.splitlines()[0].removeprefix("<!-- novel-agent: ").removesuffix(" -->"))


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


class BatchFinalizationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.batch = [{
            "chapter_no": 1, "title": "回信",
            "text": "后来蒋去收到崔瀺的亲笔回信，读后有所领悟。蒋去正在青山读书。",
        }]
        CorpusStore(self.root).save_chapters(self.batch)
        session = BuildSession(KnowledgeStore(self.root))
        session.upsert_entity(
            "character", "崔瀺", "## 事迹\n曾给蒋去写亲笔回信。", [1], summary="给蒋去写信的人",
        )
        session.commit()
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
            self.batch, "batch-finalization-test", str(self.root),
            provider=provider, model="test-model",
        )
        return provider

    def _satisfied_failure(self, request, call_id="inspect"):
        report = _result(request, call_id)
        failure = next(
            item for item in report["patch_failures"]
            if item["ref"].removesuffix(".md") == "characters/蒋去"
        )
        self.assertTrue(failure["can_confirm"], failure)
        self.assertEqual(failure["field"], "content")
        self.assertTrue(failure["source_version"])
        return failure

    def test_confirm_already_satisfied_patch_then_finish_without_rewriting_document(self):
        content = "## 经历\n后来蒋去收到崔瀺的亲笔回信，读后有所领悟。"
        state = {}

        def read_after_inspection(request):
            state["failure_id"] = self._satisfied_failure(request)["failure_id"]
            return _call("read", "read_entities", requests=[{"ref": "characters/蒋去"}])

        def confirm_after_read(request):
            state["document"] = _result(request, "read")["entities"][0]["content"]
            return _call("confirm", "confirm_patch_satisfied", failure_ids=[state["failure_id"]])

        def finish_after_confirmation(request):
            confirmed = _result(request, "confirm")
            self.assertEqual(confirmed["resolved_count"], 1)
            self.assertEqual(confirmed["changed_count"], 0)
            return _call("finish", "finish_batch", summary="已完成本批抽取")

        provider = self._run([
            _call("draft", "upsert_entities", entities=[_entity("蒋去", content)]),
            _call("already-done", "patch_entities", entities=[{
                "ref": "characters/蒋去", "replacements": [{
                    "old_text": "后来蒋去收到崔巉的亲笔回信",
                    "new_text": "后来蒋去收到崔瀺的亲笔回信",
                }],
            }]),
            _call("inspect", "inspect_repairs"),
            read_after_inspection,
            confirm_after_read,
            finish_after_confirmation,
        ])

        self.assertEqual(len(provider.message_snapshots), 6)
        self.assertEqual(KnowledgeStore(self.root).read_entity("characters/蒋去"), state["document"])
        self.assertEqual(_body(state["document"]), content)

    def test_summary_patch_uses_entity_name_id_without_rewriting_body_or_other_metadata(self):
        content = "## 经历\n后来蒋去收到崔瀺的亲笔回信，读后有所领悟。"
        wrong_summary = "蒋去收到崔巉亲笔回信后返回学塾。"
        state = {}

        def search_after_read(request):
            state["document"] = _result(request, "read")["entities"][0]["content"]
            return _call("search", "search_entities", keywords=["崔瀺"])

        def patch_summary_from_search(request):
            target = next(
                item for item in _result(request, "search")["matches"]
                if item["ref"] == "characters/崔瀺.md"
            )
            return _call("summary-patch", "patch_entities", entities=[{
                "ref": "characters/蒋去", "field": "summary", "replacements": [{
                    "old_text": "崔巉", "new_entity_id": target["entity_id"],
                }],
            }])

        def finish_after_patch(request):
            patched = _result(request, "summary-patch")
            self.assertEqual(patched["failed_count"], 0)
            self.assertEqual(patched["changed_count"], 1)
            return _call("finish", "finish_batch", summary="已完成本批抽取")

        provider = self._run([
            _call("draft", "upsert_entities", entities=[_entity("蒋去", content, wrong_summary)]),
            _call("read", "read_entities", requests=[{"ref": "characters/蒋去"}]),
            search_after_read,
            patch_summary_from_search,
            finish_after_patch,
        ])

        self.assertEqual(len(provider.message_snapshots), 5)
        store = KnowledgeStore(self.root)
        document = store.read_entity("characters/蒋去")
        before = _metadata(state["document"])
        after = _metadata(document)
        self.assertEqual(after.pop("summary"), wrong_summary.replace("崔巉", "崔瀺"))
        before.pop("summary")
        self.assertEqual(after, before)
        self.assertEqual(_body(document), _body(state["document"]))
        record = next(item for item in store.catalog_records() if item.name == "蒋去")
        self.assertEqual(record.summary, wrong_summary.replace("崔巉", "崔瀺"))

    def test_finish_with_failed_tool_in_same_round_waits_for_repair_and_new_finish(self):
        def repair_after_failed_finish(request):
            failed = _result(request, "bad-write")
            self.assertEqual(failed["failed_count"], 1)
            return _calls(
                _tool("fixed-write", "upsert_entities", entities=[_entity("蒋去", "蒋去正在白山读书。")]),
                _tool("finish-again", "finish_batch", summary="已修复并完成本批抽取"),
            )

        provider = self._run([
            _call("draft", "upsert_entities", entities=[_entity("蒋去", "蒋去正在青山读书。")]),
            _calls(
                _tool("premature-finish", "finish_batch", summary="已完成本批抽取"),
                _tool("bad-write", "upsert_entities", entities=[{
                    key: value for key, value in _entity("蒋去", "蒋去正在白山读书。").items()
                    if key != "summary"
                }]),
            ),
            repair_after_failed_finish,
        ])

        self.assertEqual(len(provider.message_snapshots), 3)
        self.assertIn("蒋去正在白山读书。", KnowledgeStore(self.root).read_entity("characters/蒋去"))

    def test_confirmation_after_final_validation_failure_can_finish_without_body_progress(self):
        def confirm_after_rejected_finish(request):
            self.assertIn("尚有未解决的写入失败", request.messages[-1]["content"])
            failure = self._satisfied_failure(request)
            return _call("confirm", "confirm_patch_satisfied", failure_ids=[failure["failure_id"]])

        content = "## 经历\n后来蒋去收到崔瀺的亲笔回信，读后有所领悟。"
        provider = self._run([
            _call("draft", "upsert_entities", entities=[_entity("蒋去", content)]),
            _call("already-done", "patch_entities", entities=[{
                "ref": "characters/蒋去", "replacements": [{
                    "old_text": "后来蒋去收到崔巉的亲笔回信",
                    "new_text": "后来蒋去收到崔瀺的亲笔回信",
                }],
            }]),
            _call("inspect", "inspect_repairs"),
            ModelResponse(AssistantMessage("本批抽取已完成。"), "stop"),
            confirm_after_rejected_finish,
        ])

        self.assertEqual(len(provider.message_snapshots), 5)
        self.assertEqual(_body(KnowledgeStore(self.root).read_entity("characters/蒋去")), content)


if __name__ == "__main__":
    unittest.main()
