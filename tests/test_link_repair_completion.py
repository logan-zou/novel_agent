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


_SOURCE_REF = "characters/田常.md"
_TARGET_REF = "settings/力量体系/封魔印.md"
_WRONG_REF = "settings/力量体系/封魔咒.md"
_CONTENT = "## 功法\n田常修习[[settings/力量体系/封魔印#封印|封魔印]]。"


def _tool(call_id, name, **arguments):
    return ToolCall(call_id, name, json.dumps(arguments, ensure_ascii=False))


def _calls(*calls):
    return ModelResponse(AssistantMessage(tool_calls=calls), "tool_calls")


def _call(call_id, name, **arguments):
    return _calls(_tool(call_id, name, **arguments))


def _result(request, call_id):
    message = next(
        item for item in reversed(request.messages)
        if item.get("role") == "tool" and item.get("tool_call_id") == call_id
    )
    return json.loads(message["content"].removeprefix("[错误]").strip())


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


class LinkRepairCompletionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.batch = [{
            "chapter_no": 1, "title": "封印",
            "text": "田常持有封魔印，封印传承与封魔咒不同。",
        }]
        CorpusStore(self.root).save_chapters(self.batch)
        baseline = BuildSession(KnowledgeStore(self.root))
        baseline.upsert_entity(
            "character", "田常", "田氏修士。", [1], summary="田氏修士", aliases=["田大夫"],
        )
        baseline.commit()
        self.source = {
            "kind": "character", "name": "田常", "content": _CONTENT,
            "summary": "田常持有封魔印。", "aliases": ["田大夫"], "source_chapters": [1],
        }
        preview = BuildSession(KnowledgeStore(self.root))
        preview.upsert_entity(**self.source)
        self.expected_document = preview.read_entity(_SOURCE_REF)
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
            self.batch, "link-repair-completion-test", str(self.root),
            provider=provider, model="test-model",
        )
        self.assertEqual(KnowledgeStore(self.root).read_entity(_SOURCE_REF), self.expected_document)
        return provider

    def _prefix(self, state):
        def create_original_target_after_inspection(request):
            report = _result(request, "inspect-missing")
            issue = next(item for item in report["issues"] if item["missing_ref"] == _TARGET_REF)
            self.assertEqual(issue["source_ref"], _SOURCE_REF)
            state["issue_id"] = issue["issue_id"]
            return _call("create-targets", "upsert_entities", entities=[
                {
                    "kind": "setting", "name": name, "domain": "力量体系",
                    "summary": f"{name}的传承", "content": f"## 封印\n{name}的传承。",
                    "source_chapters": [1],
                }
                for name in ("封魔印", "封魔咒")
            ])

        return [
            _call("draft", "upsert_entities", entities=[self.source]),
            _call("inspect-missing", "inspect_repairs"),
            create_original_target_after_inspection,
            _call("find-targets", "search_entities", keywords=["封魔印", "封魔咒"]),
        ]

    def _targets_from_search(self, request, state):
        matches = {item["ref"]: item for item in _result(request, "find-targets")["matches"]}
        state["target_id"] = matches[_TARGET_REF]["entity_id"]
        state["wrong_id"] = matches[_WRONG_REF]["entity_id"]

    def _repair(self, state, call_id, target="target_id"):
        return _call(call_id, "repair_links", repairs=[{
            "issue_id": state["issue_id"], "target_id": state[target],
        }])

    def _wrong_repair(self, request, state):
        self._targets_from_search(request, state)
        return self._repair(state, "wrong-target", "wrong_id")

    def _assert_wrong_target_rejected(self, request):
        result = _result(request, "wrong-target")
        self.assertEqual(result["failed_count"], 1)
        self.assertEqual(result["changed_count"], 0)

    def _assert_actionable_link_failure(self, request, state):
        report = _result(request, "inspect-failure")
        self.assertEqual(report["issues"], [])
        failure = next(item for item in report["link_failures"] if item["source_ref"] == _SOURCE_REF)
        self.assertEqual(failure["missing_ref"], _TARGET_REF)
        self.assertEqual(failure["issue_id"], state["issue_id"])
        self.assertEqual(failure["target_id"], state["target_id"])
        self.assertTrue(failure["can_confirm"], failure)
        self.assertTrue(failure["reason"])
        self.assertTrue(any("尚有未解决的链接修复失败" in error for error in report["validation_errors"]))

    def test_existing_target_confirmation_resolves_failure_without_rewriting_source(self):
        state = {}

        def inspect_after_wrong_target(request):
            self._assert_wrong_target_rejected(request)
            return _call("inspect-failure", "inspect_repairs")

        def read_after_actionable_failure(request):
            self._assert_actionable_link_failure(request, state)
            return _call("read-source", "read_entities", requests=[{"ref": _SOURCE_REF}])

        def confirm_after_read(request):
            state["document"] = _result(request, "read-source")["entities"][0]["content"]
            return self._repair(state, "confirm-target")

        def repeat_after_confirmation(request):
            result = _result(request, "confirm-target")
            self.assertEqual(result["failed_count"], 0)
            self.assertEqual(result["changed_count"], 0)
            self.assertEqual(result["resolved_count"], 1)
            return self._repair(state, "confirm-again")

        def finish_after_repeat(request):
            result = _result(request, "confirm-again")
            self.assertEqual(result["failed_count"], 0)
            self.assertEqual(result["changed_count"], 0)
            self.assertEqual(result["resolved_count"], 0)
            return _call("finish", "finish_batch", summary="链接已确认，本批抽取完成")

        provider = self._run([
            *self._prefix(state),
            lambda request: self._wrong_repair(request, state),
            inspect_after_wrong_target,
            read_after_actionable_failure,
            confirm_after_read,
            repeat_after_confirmation,
            finish_after_repeat,
        ])

        self.assertEqual(len(provider.message_snapshots), 10)
        self.assertEqual(state["document"], self.expected_document)
        self.assertEqual(KnowledgeStore(self.root).read_entity(_SOURCE_REF), state["document"])

    def test_confirmation_after_rejected_finish_counts_resolution_as_progress(self):
        state = {}

        def finish_with_unresolved_link_failure(request):
            self._assert_wrong_target_rejected(request)
            return _call("rejected-finish", "finish_batch", summary="本批抽取完成")

        def confirm_after_validation_failure(request):
            self.assertEqual(request.messages[-1]["role"], "user")
            self.assertIn("尚有未解决的链接修复失败", request.messages[-1]["content"])
            return self._repair(state, "confirm-target")

        provider = self._run([
            *self._prefix(state),
            lambda request: self._wrong_repair(request, state),
            finish_with_unresolved_link_failure,
            confirm_after_validation_failure,
        ])

        self.assertEqual(len(provider.message_snapshots), 7)

    def test_finish_and_wrong_repair_same_round_require_new_completion_declaration(self):
        state = {}

        def finish_and_wrong_repair(request):
            self._targets_from_search(request, state)
            return _calls(
                _tool("premature-finish", "finish_batch", summary="本批抽取完成"),
                _tool("wrong-target", "repair_links", repairs=[{
                    "issue_id": state["issue_id"], "target_id": state["wrong_id"],
                }]),
            )

        def inspect_after_rejected_completion(request):
            self._assert_wrong_target_rejected(request)
            self.assertEqual(request.messages[-1]["role"], "user")
            self.assertIn("同轮工具", request.messages[-1]["content"])
            return _call("inspect-failure", "inspect_repairs")

        def confirm_after_inspection(request):
            self._assert_actionable_link_failure(request, state)
            return self._repair(state, "confirm-target")

        def read_before_new_completion(request):
            result = _result(request, "confirm-target")
            self.assertEqual(result["resolved_count"], 1)
            self.assertEqual(result["changed_count"], 0)
            return _call("read-source", "read_entities", requests=[{"ref": _SOURCE_REF}])

        def declare_completion_after_read(request):
            self.assertEqual(
                _result(request, "read-source")["entities"][0]["content"], self.expected_document,
            )
            return _call("finish-again", "finish_batch", summary="已核对正确目标，本批抽取完成")

        provider = self._run([
            *self._prefix(state),
            finish_and_wrong_repair,
            inspect_after_rejected_completion,
            confirm_after_inspection,
            read_before_new_completion,
            declare_completion_after_read,
        ])

        self.assertEqual(len(provider.message_snapshots), 9)


if __name__ == "__main__":
    unittest.main()
