import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.novel.builder import _builder_tools, _log_tool
from src.novel.corpus import CorpusStore
from src.novel.knowledge import BuildSession, KnowledgeStore


class BuilderToolTests(unittest.TestCase):
    def test_continuation_requires_finalization_and_valid_bounded_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {"phase": "regular"}
            session = BuildSession(KnowledgeStore(tmp))
            tool = next(tool for tool in _builder_tools(session, CorpusStore(tmp), state)
                        if tool.name == "continue_batch")
            early = tool.handler(remaining_tasks=["补充角色关系"])
            self.assertIn("invalid_state", early)
            self.assertNotIn("continuation_request", state)
            state["phase"] = "finalization"
            for tasks in (None, [], [" "], [1], ["过长" * 121], ["任务"] * 17):
                with self.subTest(tasks=tasks):
                    result = tool.handler(remaining_tasks=tasks)
                    self.assertTrue(tool.argument_error_detector(result))
                    self.assertNotIn("continuation_request", state)
            result = tool.handler(remaining_tasks=["  补充角色关系  "])
            self.assertTrue(json.loads(result)["continuation_requested"])
            self.assertEqual(state["remaining_tasks"], ["补充角色关系"])
            self.assertFalse(session.pending)
            self.assertEqual(session.change_count, 0)

    def test_finalization_scope_does_not_expand_with_links_added_during_same_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = BuildSession(KnowledgeStore(tmp))
            session.upsert_entity("character", "张三", "认识[[characters/李四]]。", [1], summary="张三")
            state = {"phase": "finalization", "write_scope": {"characters/张三.md", "characters/李四.md"}}
            tools = {tool.name: tool for tool in _builder_tools(session, CorpusStore(tmp), state)}
            result = tools["upsert_entities"].handler(entities=[
                {"kind": "character", "name": "李四", "summary": "李四", "content": "认识[[characters/王五]]。", "source_chapters": [1]},
                {"kind": "character", "name": "王五", "summary": "王五", "content": "另一人物。", "source_chapters": [1]},
            ])
            payload = json.loads(result.removeprefix("[错误]"))
            self.assertEqual(payload["staged_count"], 1)
            self.assertEqual(payload["failed_count"], 1)
            self.assertEqual(payload["results"][1]["error_type"], "finalization_scope")
            self.assertIn("continue_batch", payload["results"][1]["result"])
            self.assertIn("characters/李四.md", session.pending)
            self.assertNotIn("characters/王五.md", session.pending)
            self.assertIn("characters/王五.md", session.failed_writes)

    def test_repeated_single_entity_writes_receive_batch_hint_without_fake_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = BuildSession(KnowledgeStore(tmp))
            tools = {tool.name: tool for tool in _builder_tools(session, CorpusStore(tmp))}

            def entity(name):
                return {"kind": "character", "name": name, "summary": name, "content": "人物经历。", "source_chapters": [1]}

            for name in ("张三", "李四", "王五"):
                result = tools["upsert_entities"].handler(entities=[entity(name)])
            payload = json.loads(result)
            self.assertEqual(payload["batching_hint"]["single_entity_write_streak"], 3)
            unchanged = tools["upsert_entities"].handler(entities=[entity("王五")])
            self.assertFalse(tools["upsert_entities"].progress_detector(unchanged))
            tools["upsert_entities"].handler(entities=[entity("赵六"), entity("钱七")])
            last = tools["upsert_entities"].handler(entities=[entity("孙八")])
            self.assertNotIn("batching_hint", json.loads(last))

    def test_disk_summary_version_hashes_full_document_even_when_display_is_truncated(self):
        with tempfile.TemporaryDirectory() as tmp:
            seed = BuildSession(KnowledgeStore(tmp))
            seed.upsert_entity("character", "姜述", "短正文", [1], summary="御极七十九年")
            document = seed.read_entity("characters/姜述").replace("短正文", "长" * 33_000)
            target = Path(tmp) / "characters/姜述.md"
            target.parent.mkdir(parents=True)
            target.write_text(document, encoding="utf-8")
            session = BuildSession(KnowledgeStore(tmp))
            self.assertNotEqual(session.read_entity("characters/姜述"), document)
            tools = {tool.name: tool for tool in _builder_tools(session, CorpusStore(tmp))}
            item = json.loads(tools["read_entities"].handler(requests=[{
                "ref": "characters/姜述", "field": "summary",
            }]))["entities"][0]
            self.assertEqual(item["source_version"], hashlib.sha256(document.encode()).hexdigest())
            self.assertEqual(item["summary"], "御极七十九年")
            self.assertEqual(session.pending, {})

    def test_summary_read_returns_exact_metadata_and_document_version_without_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = BuildSession(KnowledgeStore(tmp))
            session.upsert_entity("character", "姜述", "## 身份\n齐天子。", [1], summary="御极七十九年")
            tools = {tool.name: tool for tool in _builder_tools(session, CorpusStore(tmp))}
            before = dict(session.pending)
            changes = session.change_count
            explicit = json.loads(tools["read_entities"].handler(requests=[{
                "ref": "characters/姜述", "field": "summary",
            }]))["entities"][0]
            alias = json.loads(tools["read_entities"].handler(requests=[{
                "ref": "characters/姜述", "section": "summary",
            }]))["entities"][0]
            self.assertEqual(explicit, alias)
            self.assertEqual(explicit["content"], "御极七十九年")
            self.assertEqual(explicit["summary"], "御极七十九年")
            self.assertEqual(explicit["field"], "summary")
            self.assertTrue(explicit["source_version"])
            self.assertEqual(session.pending, before)
            self.assertEqual(session.change_count, changes)

    def test_read_errors_are_reported_as_failures_instead_of_entity_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = BuildSession(KnowledgeStore(tmp))
            session.upsert_entity("character", "姜述", "## 身份\n齐天子。", [1], summary="御极七十九年")
            tools = {tool.name: tool for tool in _builder_tools(session, CorpusStore(tmp))}
            result = tools["read_entities"].handler(requests=[
                {"ref": "characters/姜述", "field": "summary"},
                {"ref": "characters/姜述", "field": "summary", "section": "身份"},
                {"ref": "characters/姜述", "field": "unknown"},
            ])
            self.assertTrue(result.startswith("[错误]"))
            payload = json.loads(result.removeprefix("[错误]"))
            self.assertEqual(payload["failed_count"], 2)
            self.assertEqual(payload["entities"][0]["summary"], "御极七十九年")
            for item in payload["entities"][1:]:
                self.assertIn("error", item)
                self.assertNotIn("content", item)

    def test_patch_schema_rejects_ambiguous_replacement_target_and_invalid_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = BuildSession(KnowledgeStore(tmp))
            session.upsert_entity("character", "张三", "旧正文", [1], summary="旧摘要")
            tools = {tool.name: tool for tool in _builder_tools(session, CorpusStore(tmp))}
            before = dict(session.pending)
            cases = [
                ({"ref": "characters/张三", "replacements": [
                    {"old_text": "旧正文", "new_text": "新正文", "new_entity_id": "E1"},
                ]}, "replacements[0]"),
                ({"ref": "characters/张三", "field": "metadata", "replacements": [
                    {"old_text": "旧摘要", "new_text": "新摘要"},
                ]}, "field"),
            ]
            for request, invalid_field in cases:
                with self.subTest(request=request):
                    result = tools["patch_entities"].handler(entities=[request])
                    payload = json.loads(result.removeprefix("[错误]"))
                    self.assertIn(invalid_field, payload["results"][0]["invalid_fields"])
                    self.assertTrue(tools["patch_entities"].argument_error_detector(result))
                    self.assertEqual(session.pending, before)

    def test_finish_batch_invalid_summary_does_not_leave_a_completion_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {}
            session = BuildSession(KnowledgeStore(tmp))
            tools = {tool.name: tool for tool in _builder_tools(session, CorpusStore(tmp), state)}
            result = tools["finish_batch"].handler(summary=" ")
            self.assertTrue(tools["finish_batch"].argument_error_detector(result))
            self.assertEqual(state, {})
            self.assertEqual(session.pending, {})

    def test_structured_repair_tools_reject_invalid_batches_without_progress(self):
        for name, field in (("repair_links", "repairs"), ("manage_entities", "actions")):
            for invalid in (None, [], "invalid", [{}] * 9):
                with self.subTest(tool=name, value=invalid), tempfile.TemporaryDirectory() as tmp:
                    session = BuildSession(KnowledgeStore(tmp))
                    tools = {tool.name: tool for tool in _builder_tools(session, CorpusStore(tmp))}

                    result = tools[name].handler(**{field: invalid})

                    self.assertTrue(result.startswith("[错误]"))
                    self.assertTrue(tools[name].argument_error_detector(result))
                    self.assertFalse(tools[name].progress_detector(result))
                    self.assertTrue(session.repair_completion_blocked)
                    self.assertEqual(session.pending, {})
                    self.assertEqual(session.change_count, 0)

    def test_upsert_reports_all_missing_fields_and_conditional_domain_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = BuildSession(KnowledgeStore(tmp))
            upsert = next(t for t in _builder_tools(session, CorpusStore(tmp)) if t.name == "upsert_entities")

            result = upsert.handler(entities=[{"name": "张三"}])

            payload = json.loads(result.removeprefix("[错误]"))
            detail = payload["results"][0]
            self.assertTrue(upsert.argument_error_detector(result))
            self.assertEqual(payload["error_type"], "invalid_arguments")
            self.assertEqual(detail["error_type"], "invalid_arguments")
            self.assertEqual(detail["missing_fields"], ["kind", "summary", "content", "source_chapters"])
            self.assertIn("domain", detail["conditional_required_fields"])
            self.assertIn("setting", detail["result"])
            self.assertEqual(session.pending, {})

    def test_upsert_reports_empty_required_fields_and_setting_domain_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = BuildSession(KnowledgeStore(tmp))
            upsert = next(t for t in _builder_tools(session, CorpusStore(tmp)) if t.name == "upsert_entities")

            result = upsert.handler(entities=[{
                "kind": "setting", "name": " ", "summary": None, "content": "",
                "source_chapters": [], "domain": "",
            }])

            payload = json.loads(result.removeprefix("[错误]"))
            detail = payload["results"][0]
            self.assertEqual(detail["missing_fields"], ["name", "summary", "content", "source_chapters", "domain"])
            self.assertIn("domain", detail["required_fields"])
            self.assertEqual(session.pending, {})

    def test_upsert_reports_all_invalid_field_types_without_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = BuildSession(KnowledgeStore(tmp))
            upsert = next(t for t in _builder_tools(session, CorpusStore(tmp)) if t.name == "upsert_entities")

            result = upsert.handler(entities=[{
                "kind": "unknown", "name": ["张三"], "summary": 3, "content": {},
                "source_chapters": [True, "1"], "domain": 4, "aliases": "小三", "extra": 1,
            }])

            payload = json.loads(result.removeprefix("[错误]"))
            detail = payload["results"][0]
            self.assertEqual(set(detail["invalid_fields"]), {
                "kind", "name", "summary", "content", "source_chapters", "domain", "aliases", "extra",
            })
            self.assertTrue(upsert.argument_error_detector(result))
            self.assertEqual(session.pending, {})

    def test_mixed_business_and_argument_failures_are_not_all_classified_as_argument_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = BuildSession(KnowledgeStore(tmp))
            upsert = next(t for t in _builder_tools(session, CorpusStore(tmp)) if t.name == "upsert_entities")
            entity = {"kind": "character", "name": "张三", "summary": "修士",
                      "content": "正文", "source_chapters": [1]}

            result = upsert.handler(entities=[
                entity,
                {**entity, "name": "李四", "content": "[[省略完整路径]]"},
                {"kind": "character", "name": "王五"},
            ])

            payload = json.loads(result.removeprefix("[错误]"))
            self.assertEqual(payload["staged_count"], 1)
            self.assertEqual(payload["changed_count"], 1)
            self.assertEqual(payload["failed_count"], 2)
            self.assertNotIn("error_type", payload)
            self.assertNotIn("error_type", payload["results"][1])
            self.assertEqual(payload["results"][2]["error_type"], "invalid_arguments")
            self.assertFalse(upsert.argument_error_detector(result))
            self.assertIn("characters/李四.md", session.failed_writes)
            self.assertIn("characters/王五.md", session.failed_writes)

    def test_patch_reports_all_nested_argument_errors_and_keeps_pending_document(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = BuildSession(KnowledgeStore(tmp))
            session.upsert_entity("character", "张三", "原内容", [1], summary="摘要")
            before = dict(session.pending)
            edit = next(t for t in _builder_tools(session, CorpusStore(tmp)) if t.name == "patch_entities")

            result = edit.handler(entities=[{"ref": "characters/张三", "replacements": [
                {}, {"old_text": " ", "new_text": 12, "extra": True}, None,
            ]}])

            payload = json.loads(result.removeprefix("[错误]"))
            detail = payload["results"][0]
            self.assertEqual(detail["missing_fields"], [
                "replacements[0].old_text", "replacements[0].new_text", "replacements[1].old_text",
            ])
            self.assertEqual(set(detail["invalid_fields"]), {
                "replacements[1].new_text", "replacements[1].extra", "replacements[2]",
            })
            self.assertTrue(edit.argument_error_detector(result))
            self.assertEqual(session.pending, before)
            self.assertIn("characters/张三.md", session.failed_writes)

    def test_batch_argument_errors_use_structured_classification(self):
        for name in ("upsert_entities", "patch_entities"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                session = BuildSession(KnowledgeStore(tmp))
                tool = next(t for t in _builder_tools(session, CorpusStore(tmp)) if t.name == name)

                result = tool.handler(entities=123, extra=True)

                payload = json.loads(result.removeprefix("[错误]"))
                self.assertEqual(set(payload["invalid_fields"]), {"entities", "extra"})
                self.assertTrue(tool.argument_error_detector(result))
                self.assertTrue(session.repair_completion_blocked)

    @patch("builtins.print")
    def test_logging_invalid_batch_arguments_does_not_interrupt_error_feedback(self, _print):
        for name, field in (("upsert_entities", "entities"), ("patch_entities", "entities"),
                            ("read_entities", "requests")):
            with self.subTest(name=name):
                _log_tool(name, {field: 123}, "[错误] 参数必须是数组")

    def test_write_argument_errors_track_entity_when_identifiable(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = BuildSession(KnowledgeStore(tmp))
            tools = {tool.name: tool for tool in _builder_tools(session, CorpusStore(tmp))}

            result = tools["upsert_entities"].handler(entities=[{"kind": "character", "name": "张三"}])
            self.assertTrue(result.startswith("[错误]"))
            self.assertIn("characters/张三.md", session.failed_writes)
            self.assertFalse(session.repair_completion_blocked)
            session.upsert_entity("character", "李四", "原正文", [1], summary="摘要")
            result = tools["patch_entities"].handler(entities=[{"ref": "characters/李四"}])
            self.assertTrue(result.startswith("[错误]"))
            self.assertIn("characters/李四.md", session.failed_writes)
            self.assertFalse(session.repair_completion_blocked)

    def test_unidentified_write_argument_errors_block_automatic_completion(self):
        for name in ("upsert_entities", "patch_entities"):
            for arguments in ({}, {"entities": []}, {"entities": 123}, {"entities": [None]},
                              {"entities": [{}]}, {"unexpected": True}):
                with self.subTest(name=name, arguments=arguments), tempfile.TemporaryDirectory() as tmp:
                    session = BuildSession(KnowledgeStore(tmp))
                    tools = {tool.name: tool for tool in _builder_tools(session, CorpusStore(tmp))}

                    result = tools[name].handler(**arguments)

                    self.assertTrue(result.startswith("[错误]"))
                    self.assertTrue(session.repair_completion_blocked)
                    self.assertEqual(session.failed_writes, {})
                    self.assertIsNone(session.validator("显式结束"))

    def test_batch_patch_exposes_targeted_edits_and_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = BuildSession(KnowledgeStore(tmp))
            tools = {tool.name: tool for tool in _builder_tools(session, CorpusStore(tmp))}
            session.upsert_entity("character", "张三", "认识[[characters/朱琙]]。", [1], summary="修士")
            session.upsert_entity("character", "李四", "遇到朱琙。", [1], summary="修士")
            patch = tools["patch_entities"]

            result = patch.handler(entities=[
                {"ref": "characters/张三", "replacements": [
                    {"old_text": "[[characters/朱琙]]", "new_text": "[[characters/朱敛]]"},
                ]},
                {"ref": "characters/李四", "replacements": [
                    {"old_text": "朱琙", "new_text": "朱敛"},
                ]},
            ])

            payload = json.loads(result)
            self.assertEqual(payload["staged_count"], 2)
            self.assertEqual(payload["changed_count"], 2)
            self.assertEqual(payload["failed_count"], 0)
            self.assertTrue(patch.progress_detector(result))
            self.assertIn("[[characters/朱敛]]", session.read_entity("characters/张三"))
            repeated = patch.handler(entities=[
                {"ref": "characters/李四", "replacements": [
                    {"old_text": "朱琙", "new_text": "朱敛"},
                ]},
            ])
            self.assertTrue(repeated.startswith("[错误]"))
            self.assertFalse(patch.progress_detector(repeated))

    def test_partial_batch_patch_reports_error_and_preserves_successful_edits(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = BuildSession(KnowledgeStore(tmp))
            session.upsert_entity("character", "张三", "旧内容", [1], summary="修士")
            session.upsert_entity("character", "李四", "旧内容", [1], summary="修士")
            patch = next(
                tool for tool in _builder_tools(session, CorpusStore(tmp))
                if tool.name == "patch_entities"
            )

            result = patch.handler(entities=[
                {"ref": "characters/张三", "replacements": [
                    {"old_text": "旧内容", "new_text": "新内容"},
                ]},
                {"ref": "characters/李四", "replacements": [
                    {"old_text": "不匹配", "new_text": "新内容"},
                ]},
            ])

            self.assertTrue(result.startswith("[错误]"))
            payload = json.loads(result[len("[错误]"):])
            self.assertEqual(payload["staged_count"], 1)
            self.assertEqual(payload["changed_count"], 1)
            self.assertEqual(payload["failed_count"], 1)
            self.assertIn("新内容", session.read_entity("characters/张三"))
            self.assertIn("旧内容", session.read_entity("characters/李四"))
            self.assertIn("characters/李四.md", session.failed_writes)

    def test_partial_batch_upsert_reports_error_and_preserves_successful_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = BuildSession(KnowledgeStore(tmp))
            upsert = next(
                tool for tool in _builder_tools(session, CorpusStore(tmp))
                if tool.name == "upsert_entities"
            )

            result = upsert.handler(entities=[
                {"kind": "character", "name": "张三", "content": "正文", "source_chapters": [1],
                 "summary": "修士"},
                {"kind": "character", "name": "李四", "content": "正文", "source_chapters": [1],
                 "summary": ""},
            ])

            self.assertTrue(result.startswith("[错误]"))
            payload = json.loads(result[len("[错误]"):])
            self.assertEqual(payload["staged_count"], 1)
            self.assertEqual(payload["changed_count"], 1)
            self.assertEqual(payload["failed_count"], 1)
            self.assertIn("characters/张三.md", session.pending)
            self.assertNotIn("characters/李四.md", session.pending)
            self.assertIn("characters/李四.md", session.failed_writes)
            self.assertTrue(upsert.argument_error_detector(result))
            self.assertEqual(payload["error_type"], "invalid_arguments")

    def test_identical_writes_are_not_progress_even_in_new_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = CorpusStore(tmp)
            corpus.save_chapters([{"chapter_no": 1, "title": "开始", "text": "正文"}])
            store = KnowledgeStore(tmp)
            session = BuildSession(store)
            tool = next(t for t in _builder_tools(session, corpus) if t.name == "upsert_entities")
            entity = {"kind": "character", "name": "张三", "summary": "修士",
                      "content": "开始修行", "source_chapters": [1]}
            self.assertTrue(tool.progress_detector(tool.handler(entities=[entity])))
            unchanged = tool.handler(entities=[entity])
            self.assertFalse(tool.progress_detector(unchanged))
            self.assertIn("请不要重复提交相同内容", json.loads(unchanged)["feedback"])
            session.commit()
            tool = next(t for t in _builder_tools(BuildSession(store), corpus) if t.name == "upsert_entities")
            self.assertFalse(tool.progress_detector(tool.handler(entities=[entity])))
            self.assertTrue(tool.progress_detector(tool.handler(entities=[{**entity, "content": "修行突破"}])))

    def test_extraction_exposes_batch_entity_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = CorpusStore(tmp)
            corpus.save_chapters(
                [{"chapter_no": 1, "title": "开始", "text": "张三得到筑基传承。"}]
            )
            session = BuildSession(KnowledgeStore(tmp))
            tools = {tool.name: tool for tool in _builder_tools(session, corpus)}

            self.assertIn("read_entities", tools)
            self.assertIn("upsert_entities", tools)
            self.assertNotIn("read_entity", tools)
            self.assertNotIn("upsert_entity", tools)

            written = tools["upsert_entities"].handler(
                entities=[
                    {
                        "kind": "setting",
                        "name": "筑基",
                        "aliases": [],
                        "summary": "修行境界",
                        "content": "修行的第二境。",
                        "source_chapters": [1],
                        "domain": "力量体系",
                    },
                    {
                        "kind": "character",
                        "name": "张三",
                        "aliases": [],
                        "summary": "获得传承的修士",
                        "content": "得到[[settings/力量体系/筑基]]传承。",
                        "source_chapters": [1],
                    },
                ]
            )
            written_payload = json.loads(written)
            self.assertEqual(written_payload["staged_count"], 2)
            self.assertTrue(tools["upsert_entities"].progress_detector(written))

            read = tools["read_entities"].handler(
                requests=[
                    {"ref": "settings/力量体系/筑基"},
                    {"ref": "characters/张三"},
                ]
            )
            read_payload = json.loads(read)
            self.assertEqual(len(read_payload["entities"]), 2)
            self.assertIn("修行的第二境", read_payload["entities"][0]["content"])


if __name__ == "__main__":
    unittest.main()
