import contextlib
import io
import json
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from src.harness.runtime import AgentRuntime, AssistantMessage, MaxIterationsExceeded, ModelResponse, ToolCall
from src.novel.builder import run_extraction_agent
from src.novel.corpus import CorpusStore
from src.novel.knowledge import BuildSession, KnowledgeStore


def _tool(call_id, name, **arguments):
    return ToolCall(call_id, name, json.dumps(arguments, ensure_ascii=False))


def _calls(*calls):
    return ModelResponse(AssistantMessage(tool_calls=calls), "tool_calls")


def _call(call_id, name, **arguments):
    return _calls(_tool(call_id, name, **arguments))


def _entity(name, content):
    return {
        "kind": "character", "name": name, "content": content,
        "summary": f"{name}在第1122章的经历", "aliases": [], "source_chapters": [1122],
    }


def _tool_content(request, call_id):
    return next(
        message["content"] for message in reversed(request.messages)
        if message.get("role") == "tool" and message.get("tool_call_id") == call_id
    )


def _result(request, call_id):
    return json.loads(_tool_content(request, call_id).removeprefix("[错误]").strip())


def _latest_phase(request):
    for message in reversed(request.messages):
        if message.get("role") != "user":
            continue
        try:
            value = json.loads(message.get("content", ""))
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and value.get("stage") in ("finalization", "continuation"):
            return value
    raise AssertionError("模型请求中缺少阶段说明")


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


class ExtractionContinuationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.batch = [{
            "chapter_no": 1122, "title": "登山",
            "text": "陈平安拜访黄粱派，随后回山。李槐同游。",
        }]
        CorpusStore(self.root).save_chapters(self.batch)
        baseline = BuildSession(KnowledgeStore(self.root))
        baseline.upsert_entity(
            "setting", "黄粱派", "山上门派。", [1122], summary="山上门派", domain="势力组织",
        )
        baseline.commit()
        self.original_files = self._snapshot()
        self.provider = None
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        stack.enter_context(patch("src.novel.builder.create_trace", return_value=None))
        stack.enter_context(patch("src.novel.builder.finish_trace"))
        stack.enter_context(patch("src.novel.builder.record_usage"))
        stack.enter_context(patch(
            "src.novel.builder.get_provider", side_effect=AssertionError("禁止真实模型调用"),
        ))
        original_run = AgentRuntime.run

        def run_with_short_budgets(runtime, config, *args, **kwargs):
            shortened = replace(
                config, max_iterations=2, soft_iteration_limit=None,
                finalization_iterations=3, continuation_iterations=3, max_continuations=1,
                finalization_reserve_iterations=0,
            )
            return original_run(runtime, shortened, *args, **kwargs)

        stack.enter_context(patch.object(AgentRuntime, "run", new=run_with_short_budgets))

    def _snapshot(self):
        return {
            str(path.relative_to(self.root)): path.read_bytes()
            for path in self.root.rglob("*") if path.is_file()
        }

    def _run(self, responses):
        self.provider = _DynamicProvider(
            responses, lambda: self.assertEqual(self._snapshot(), self.original_files),
        )
        run_extraction_agent(
            self.batch, "extraction-continuation-test", str(self.root),
            provider=self.provider, model="test-model",
        )
        return self.provider

    def test_dedicated_finalization_repairs_staged_work_after_regular_budget_is_exhausted(self):
        content = "陈平安拜访[[settings/势力组织/黄梁派]]，随后回山山。"

        def repair_in_dedicated_window(request):
            phase = _latest_phase(request)
            self.assertEqual(phase["stage"], "finalization")
            self.assertEqual(phase["remaining_phase_iterations"], 3)
            report = _result(request, "inspect")
            issue = next(
                item for item in report["issues"]
                if item["missing_ref"] == "settings/势力组织/黄梁派.md"
            )
            candidate = next(
                item for item in issue["candidates"]
                if item["ref"] == "settings/势力组织/黄粱派.md"
            )
            return _calls(
                _tool("repair-link", "repair_links", repairs=[{
                    "issue_id": issue["issue_id"], "target_id": candidate["entity_id"],
                }]),
                _tool("patch-typo", "patch_entities", entities=[{
                    "ref": "characters/陈平安", "replacements": [{
                        "old_text": "随后回山山", "new_text": "随后回山",
                    }],
                }]),
            )

        def finish_after_repairs(request):
            for call_id in ("repair-link", "patch-typo"):
                repaired = _result(request, call_id)
                self.assertEqual(repaired["failed_count"], 0)
                self.assertEqual(repaired["changed_count"], 1)
            return _call("finish", "finish_batch", summary="已完成长章抽取并修正链接和错字")

        provider = self._run([
            _call("draft", "upsert_entities", entities=[_entity("陈平安", content)]),
            _call("inspect", "inspect_repairs"),
            repair_in_dedicated_window,
            finish_after_repairs,
        ])

        self.assertEqual(len(provider.message_snapshots), 4)
        document = KnowledgeStore(self.root).read_entity("characters/陈平安")
        self.assertIn("陈平安拜访[[settings/势力组织/黄粱派]]，随后回山。", document)
        self.assertNotIn("黄梁派", document)
        self.assertNotIn("回山山", document)

    def test_explicit_continuation_preserves_staged_state_and_allows_unfinished_extraction(self):
        content = "陈平安拜访黄粱派，随后回山。"
        remaining = _entity("李槐", "李槐同陈平安游历。")
        state = {}

        def try_new_entity_in_protected_window(request):
            self.assertEqual(_latest_phase(request)["stage"], "finalization")
            state["source_document"] = _result(request, "read-staged")["entities"][0]["content"]
            return _calls(
                _tool("scope-rejected", "upsert_entities", entities=[remaining]),
                _tool("inspect-scope-failure", "inspect_repairs"),
            )

        def request_continuation_after_scope_rejection(request):
            rejected = _result(request, "scope-rejected")
            self.assertEqual(rejected["failed_count"], 1)
            self.assertEqual(rejected["changed_count"], 0)
            self.assertEqual(rejected["results"][0]["error_type"], "finalization_scope")
            self.assertIn("continue_batch", rejected["results"][0]["result"])
            errors = _result(request, "inspect-scope-failure")["validation_errors"]
            self.assertTrue(any("characters/李槐.md" in error and "未解决的写入失败" in error for error in errors))
            return _call("continue", "continue_batch", remaining_tasks=["抽取李槐在本章中的同行经历"])

        def read_staged_state_in_continuation(request):
            self.assertTrue(_result(request, "continue")["continuation_requested"])
            self.assertEqual(_latest_phase(request)["stage"], "continuation")
            return _call("read-in-continuation", "read_entities", requests=[{"ref": "characters/陈平安"}])

        def write_remaining_entity(request):
            self.assertEqual(
                _result(request, "read-in-continuation")["entities"][0]["content"],
                state["source_document"],
            )
            self.assertTrue(any(message.get("tool_call_id") == "scope-rejected" for message in request.messages))
            return _calls(
                _tool("allowed-new-entity", "upsert_entities", entities=[remaining]),
                _tool("inspect-after-continuation-write", "inspect_repairs"),
            )

        def read_after_new_entity(request):
            written = _result(request, "allowed-new-entity")
            self.assertEqual(written["failed_count"], 0)
            self.assertEqual(written["changed_count"], 1)
            self.assertEqual(_result(request, "inspect-after-continuation-write")["validation_errors"], [])
            return _call("read-new", "read_entities", requests=[{"ref": "characters/李槐"}])

        def inspect_in_second_finalization_window(request):
            self.assertEqual(_latest_phase(request)["stage"], "finalization")
            self.assertIn("李槐同陈平安游历。", _result(request, "read-new")["entities"][0]["content"])
            return _call("inspect-final", "inspect_repairs")

        def finish_after_clean_inspection(request):
            report = _result(request, "inspect-final")
            self.assertEqual(report["validation_errors"], [])
            self.assertEqual(report["patch_failures"], [])
            return _call("finish", "finish_batch", summary="续跑完成剩余抽取，所有修复已通过")

        provider = self._run([
            _call("draft", "upsert_entities", entities=[_entity("陈平安", content)]),
            _call("read-staged", "read_entities", requests=[{"ref": "characters/陈平安"}]),
            try_new_entity_in_protected_window,
            request_continuation_after_scope_rejection,
            read_staged_state_in_continuation,
            write_remaining_entity,
            read_after_new_entity,
            inspect_in_second_finalization_window,
            finish_after_clean_inspection,
        ])

        self.assertEqual(len(provider.message_snapshots), 9)
        store = KnowledgeStore(self.root)
        self.assertEqual(store.read_entity("characters/陈平安"), state["source_document"])
        self.assertIn("李槐同陈平安游历。", store.read_entity("characters/李槐"))
        self.assertEqual({item.name for item in store.catalog_records()}, {"黄粱派", "陈平安", "李槐"})

    def test_scope_rejection_blocks_finish_until_continuation_write_and_new_completion(self):
        remaining = _entity("李槐", "李槐同陈平安游历。")
        state = {}

        def reject_new_entity_in_finalization(request):
            state["source_document"] = _result(request, "read-staged")["entities"][0]["content"]
            self.assertEqual(_latest_phase(request)["stage"], "finalization")
            return _calls(
                _tool("scope-rejected", "upsert_entities", entities=[remaining]),
                _tool("inspect-scope-failure", "inspect_repairs"),
            )

        def try_finish_with_unresolved_scope_failure(request):
            self.assertEqual(_result(request, "scope-rejected")["failed_count"], 1)
            errors = _result(request, "inspect-scope-failure")["validation_errors"]
            self.assertTrue(any("characters/李槐.md" in error and "未解决的写入失败" in error for error in errors))
            return _call("premature-finish", "finish_batch", summary="本批已完成")

        def continue_after_rejected_finish(request):
            self.assertEqual(request.messages[-1]["role"], "user")
            self.assertIn("characters/李槐.md", request.messages[-1]["content"])
            self.assertIn("未解决的写入失败", request.messages[-1]["content"])
            return _call("continue", "continue_batch", remaining_tasks=["完成李槐在本章中的同行经历抽取"])

        def write_pending_work_after_continuation(request):
            self.assertTrue(_result(request, "continue")["continuation_requested"])
            self.assertEqual(_latest_phase(request)["stage"], "continuation")
            self.assertTrue(any(message.get("tool_call_id") == "draft" for message in request.messages))
            self.assertTrue(any("第1122章" in message.get("content", "") for message in request.messages))
            return _calls(
                _tool("continued-write", "upsert_entities", entities=[remaining]),
                _tool("read-preserved-source", "read_entities", requests=[{"ref": "characters/陈平安"}]),
                _tool("inspect-repaired", "inspect_repairs"),
            )

        def read_new_entity_before_new_completion(request):
            self.assertEqual(_result(request, "continued-write")["failed_count"], 0)
            self.assertEqual(_result(request, "inspect-repaired")["validation_errors"], [])
            self.assertEqual(
                _result(request, "read-preserved-source")["entities"][0]["content"],
                state["source_document"],
            )
            return _call("read-new", "read_entities", requests=[{"ref": "characters/李槐"}])

        def finish_after_new_work_is_done(request):
            self.assertIn("李槐同陈平安游历。", _result(request, "read-new")["entities"][0]["content"])
            return _call("new-finish", "finish_batch", summary="已完成续跑抽取，重新确认本批完成")

        provider = self._run([
            _call("draft", "upsert_entities", entities=[_entity("陈平安", "陈平安拜访黄粱派，随后回山。")]),
            _call("read-staged", "read_entities", requests=[{"ref": "characters/陈平安"}]),
            reject_new_entity_in_finalization,
            try_finish_with_unresolved_scope_failure,
            continue_after_rejected_finish,
            write_pending_work_after_continuation,
            read_new_entity_before_new_completion,
            finish_after_new_work_is_done,
        ])

        self.assertEqual(len(provider.message_snapshots), 8)
        store = KnowledgeStore(self.root)
        self.assertEqual(store.read_entity("characters/陈平安"), state["source_document"])
        self.assertIn("李槐同陈平安游历。", store.read_entity("characters/李槐"))

    def test_valid_staged_state_without_completion_exhausts_bounded_window_without_commit(self):
        with self.assertRaises(MaxIterationsExceeded) as caught:
            self._run([
                _call("draft", "upsert_entities", entities=[_entity("陈平安", "陈平安拜访黄粱派。")]),
                _call("read-summary", "read_entities", requests=[{"ref": "characters/陈平安", "field": "summary"}]),
                _call("read-content", "read_entities", requests=[{"ref": "characters/陈平安", "field": "content"}]),
                _call("search", "search_entities", keywords=["陈平安"]),
                _call("inspect", "inspect_repairs"),
            ])

        self.assertEqual(caught.exception.steps, 5)
        self.assertEqual(len(self.provider.message_snapshots), 5)
        self.assertEqual(self._snapshot(), self.original_files)
        self.assertFalse((self.root / "characters/陈平安.md").exists())


if __name__ == "__main__":
    unittest.main()
