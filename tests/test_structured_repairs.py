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


def _call(call_id, name, **arguments):
    return ModelResponse(
        AssistantMessage(tool_calls=(ToolCall(
            call_id, name, json.dumps(arguments, ensure_ascii=False),
        ),)),
        "tool_calls",
    )


def _finish():
    return ModelResponse(AssistantMessage("本章实体已整理完成。"), "stop")


def _entity(kind, name, content):
    return {
        "kind": kind, "name": name, "content": content,
        "summary": f"{name}的记录", "source_chapters": [1],
    }


def _result(request, call_id):
    message = next(
        message for message in reversed(request.messages)
        if message.get("role") == "tool" and message.get("tool_call_id") == call_id
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


class StructuredRepairIntegrationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.batch = [{
            "chapter_no": 1, "title": "登山",
            "text": "陈平安登山拜访黄粱派，与高枕论道。",
        }]
        CorpusStore(self.root).save_chapters(self.batch)
        session = BuildSession(KnowledgeStore(self.root))
        session.upsert_entity(
            "setting", "黄粱派", "高枕掌管的门派。", [1],
            summary="仙家门派", domain="势力组织",
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
            self.batch, "structured-repair-test", str(self.root),
            provider=provider, model="test-model",
        )
        return provider

    def _link_repair(self, request, call_id, state):
        report = _result(request, call_id)
        issue = next(
            item for item in report["issues"]
            if item["missing_ref"].removesuffix(".md") == "settings/势力组织/黄梁派"
        )
        target = next(
            candidate for candidate in issue["candidates"]
            if candidate["ref"] == "settings/势力组织/黄粱派.md"
        )
        state["source_id"] = issue["source_id"]
        return _call("repair-link", "repair_links", repairs=[{
            "issue_id": issue["issue_id"], "target_id": target["entity_id"],
        }])

    def test_link_ids_and_discard_remove_misspelled_duplicate_before_commit(self):
        wrong_name = "陈平安登山访黄梁派"
        right_name = "陈平安登山访黄粱派"
        state = {}

        def discard_redundant_event(request):
            repaired = _result(request, "repair-link")
            self.assertEqual(repaired["failed_count"], 0)
            self.assertEqual(repaired["changed_count"], 1)
            return _call("discard-event", "manage_entities", actions=[{
                "action": "discard", "entity_id": state["source_id"],
            }])

        def finish_after_discard(request):
            discarded = _result(request, "discard-event")
            self.assertEqual(discarded["failed_count"], 0)
            self.assertGreater(discarded["changed_count"], 0)
            return _finish()

        provider = self._run([
            _call("draft", "upsert_entities", entities=[
                _entity("event", wrong_name, "拜访[[settings/势力组织/黄梁派]]。"),
                _entity("event", right_name, "拜访[[settings/势力组织/黄粱派]]。"),
            ]),
            _call("inspect", "inspect_repairs"),
            lambda request: self._link_repair(request, "inspect", state),
            discard_redundant_event,
            finish_after_discard,
        ])

        self.assertEqual(len(provider.message_snapshots), 5)
        store = KnowledgeStore(self.root)
        self.assertEqual({record.name for record in store.catalog_records()}, {"黄粱派", right_name})
        self.assertFalse((self.root / "plot/events" / f"{wrong_name}.md").exists())
        self.assertIn("[[settings/势力组织/黄粱派]]", store.read_entity(f"plot/events/{right_name}"))

    def test_repeated_patch_failure_recovers_through_link_ids(self):
        name = "陈平安登山"
        broken_link = "[[settings/势力组织/黄梁派]]"
        noop = {"ref": f"plot/events/{name}", "replacements": [{
            "old_text": broken_link, "new_text": broken_link,
        }]}

        def inspect_after_recovery_context(request):
            feedback = request.messages[-1]["content"]
            for expected in ("issue_id", "repair_links", "黄粱派", "黄梁派"):
                self.assertIn(expected, feedback)
            self.assertIn("请勿继续提交完全相同的参数", feedback)
            recovery = json.loads(feedback.split("当前修复上下文：\n", 1)[1])
            current = next(
                item for item in recovery["current_entities"]
                if item["ref"].removesuffix(".md") == f"plot/events/{name}"
            )
            self.assertIn(f"拜访{broken_link}。", current["content"])
            self.assertTrue(recovery["repair_options"]["issues"])
            return _call("inspect", "inspect_repairs")

        provider = self._run([
            _call("draft", "upsert_entities", entities=[
                _entity("event", name, f"拜访{broken_link}。"),
            ]),
            _call("noop-1", "patch_entities", entities=[noop]),
            _call("noop-2", "patch_entities", entities=[noop]),
            inspect_after_recovery_context,
            lambda request: self._link_repair(request, "inspect", {}),
            _finish(),
        ])

        self.assertEqual(len(provider.message_snapshots), 6)
        document = KnowledgeStore(self.root).read_entity(f"plot/events/{name}")
        self.assertIn("[[settings/势力组织/黄粱派]]", document)
        self.assertNotIn(broken_link, document)

    def test_link_id_repair_finishes_after_final_validation_failure(self):
        name = "陈平安登山"

        def inspect_rejected_output(request):
            self.assertEqual(request.messages[-1]["role"], "user")
            self.assertIn("链接目标不存在", request.messages[-1]["content"])
            return _call("inspect", "inspect_repairs")

        provider = self._run([
            _call("draft", "upsert_entities", entities=[
                _entity("event", name, "拜访[[settings/势力组织/黄梁派]]。"),
            ]),
            _finish(),
            inspect_rejected_output,
            lambda request: self._link_repair(request, "inspect", {}),
        ])

        self.assertEqual(len(provider.message_snapshots), 4)
        self.assertIn(
            "[[settings/势力组织/黄粱派]]",
            KnowledgeStore(self.root).read_entity(f"plot/events/{name}"),
        )

    def test_rename_new_entity_updates_staged_incoming_links(self):
        wrong_name = "登山访黄梁派"
        right_name = "登山访黄粱派"

        def rename_from_inspection(request):
            report = _result(request, "inspect")
            event = next(item for item in report["entities"] if item["name"] == wrong_name)
            self.assertTrue(event["is_new"])
            return _call("rename-event", "manage_entities", actions=[{
                "action": "rename", "entity_id": event["entity_id"], "new_name": right_name,
            }])

        provider = self._run([
            _call("draft", "upsert_entities", entities=[
                _entity("event", wrong_name, "## 经过\n陈平安拜访黄粱派。"),
                _entity("character", "陈平安", f"见[[plot/events/{wrong_name}.md#经过|登山]]。"),
            ]),
            _call("inspect", "inspect_repairs"),
            rename_from_inspection,
            _finish(),
        ])

        self.assertEqual(len(provider.message_snapshots), 4)
        store = KnowledgeStore(self.root)
        self.assertFalse((self.root / "plot/events" / f"{wrong_name}.md").exists())
        self.assertIn(f"[[plot/events/{right_name}.md#经过|登山]]", store.read_entity("characters/陈平安"))
        self.assertEqual({record.name for record in store.catalog_records()}, {"黄粱派", "陈平安", right_name})


if __name__ == "__main__":
    unittest.main()
