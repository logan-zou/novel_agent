import json
import tempfile
import unittest
from pathlib import Path

from src.novel.corpus import CorpusStore
from src.novel.knowledge import BuildSession, KnowledgeStore


def parsed(result):
    return json.loads(result.removeprefix("[错误]").strip())


class PatchConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        CorpusStore(self.root).save_chapters([{"chapter_no": 1, "title": "信件", "text": "原文"}])
        self.session = BuildSession(KnowledgeStore(self.root))
        self.ref = "characters/蒋去.md"
        self.old = "后来蒋去收到崔巉的亲笔回信"
        self.new = "后来蒋去收到崔瀺的亲笔回信"
        self.other_old = "信封上还留有崔巉亲手题写的落款"
        self.other_new = "信封上还留有崔瀺亲手题写的落款"
        self.stage(f"{self.new}。{self.other_new}。其他旧句待修正。")

    def stage(self, body, summary="收到崔巉来信", name="蒋去"):
        return self.session.upsert_entity(
            "character", name, body, [1], aliases=["小蒋"], summary=summary,
        )

    def failures(self):
        return parsed(self.session.inspect_repairs())["patch_failures"]

    def reject(self, replacements, field="content"):
        result = self.session.patch_entity(self.ref, replacements, field=field)
        self.assertTrue(result.startswith("[错误]"), result)
        return self.failures()[-1]

    def test_confirm_actual_jiangqu_two_replacements_only_resolves_failure_without_writing(self):
        before = dict(self.session.pending)
        changes = self.session.change_count
        failure = self.reject([
            {"old_text": self.old, "new_text": self.new},
            {"old_text": self.other_old, "new_text": self.other_new},
        ])
        self.assertTrue(failure["can_confirm"])
        self.assertEqual(failure["field"], "content")
        self.assertTrue(failure["source_version"])
        result = parsed(self.session.confirm_patch_satisfied([failure["failure_id"]]))
        self.assertEqual(result["resolved_count"], 1)
        self.assertEqual(result["changed_count"], 0)
        self.assertEqual(result["failed_count"], 0)
        self.assertEqual(self.session.change_count, changes)
        self.assertEqual(self.session.pending, before)
        self.assertEqual(self.session.validate(), [])
        repeated = parsed(self.session.confirm_patch_satisfied([failure["failure_id"]]))
        self.assertEqual(repeated["resolved_count"], 0)
        self.assertEqual(repeated["failed_count"], 1)

    def test_same_failed_request_reuses_id_without_growing_failure_state(self):
        changes = self.session.change_count
        replacements = [{"old_text": self.old, "new_text": self.new}]
        first = self.reject(replacements)
        second = self.reject(replacements)
        self.assertEqual(first["failure_id"], second["failure_id"])
        self.assertEqual(len(self.failures()), 1)
        self.assertEqual(self.session.change_count, changes)

    def test_confirmed_same_version_request_is_idempotent_without_new_progress(self):
        replacements = [
            {"old_text": self.old, "new_text": self.new},
            {"old_text": self.other_old, "new_text": self.other_new},
        ]
        failure = self.reject(replacements)
        self.assertEqual(parsed(self.session.confirm_patch_satisfied([failure["failure_id"]]))["resolved_count"], 1)
        before = dict(self.session.pending)
        changes = self.session.change_count
        for request in (replacements, list(reversed(replacements)), replacements):
            with self.subTest(request=request):
                result = self.session.patch_entity(self.ref, request)
                self.assertTrue(result.startswith("[已暂存]"), result)
                self.assertIn("此前已确认满足，本次无变化", result)
                self.assertEqual(self.failures(), [])
                self.assertEqual(len(self.session._patch_failures), 1)
                self.assertEqual(self.session.pending, before)
                self.assertEqual(self.session.change_count, changes)
                repeated = parsed(self.session.confirm_patch_satisfied([failure["failure_id"]]))
                self.assertEqual(repeated["resolved_count"], 0)
        self.assertTrue(self.session._patch_failures[failure["failure_id"]]["confirmed"])

    def test_confirmed_receipt_does_not_apply_after_summary_changes_source_version(self):
        replacements = [{"old_text": self.old, "new_text": self.new}]
        first = self.reject(replacements)
        self.session.confirm_patch_satisfied([first["failure_id"]])
        self.session.patch_entity(self.ref, [{"old_text": "崔巉", "new_text": "崔瀺"}], field="summary")
        second = self.reject(replacements)
        self.assertNotEqual(first["failure_id"], second["failure_id"])
        self.assertFalse(self.session._patch_failures[second["failure_id"]]["confirmed"])
        self.assertTrue(second["can_confirm"])

    def test_summary_change_expires_content_confirmation_even_if_text_still_satisfied(self):
        failure = self.reject([{"old_text": self.old, "new_text": self.new}])
        result = self.session.patch_entity(self.ref, [{"old_text": "崔巉", "new_text": "崔瀺"}], field="summary")
        self.assertTrue(result.startswith("[已暂存]"))
        result = parsed(self.session.confirm_patch_satisfied([failure["failure_id"]]))
        self.assertEqual(result["resolved_count"], 0)
        self.assertIn("版本已变化", result["results"][0]["result"])
        self.assertIn(self.ref, self.session.failed_writes)

    def test_fresh_same_request_supersedes_stale_failure_but_still_requires_confirmation(self):
        replacements = [{"old_text": self.old, "new_text": self.new}]
        first = self.reject(replacements)
        self.session.patch_entity(self.ref, [{"old_text": "崔巉", "new_text": "崔瀺"}], field="summary")
        self.assertEqual([item["failure_id"] for item in self.failures()], [first["failure_id"]])
        self.assertFalse(self.failures()[0]["can_confirm"])
        second = self.reject(replacements)
        self.assertNotEqual(first["failure_id"], second["failure_id"])
        self.assertNotEqual(first["source_version"], second["source_version"])
        self.assertEqual(second["supersedes"], [first["failure_id"]])
        self.assertTrue(second["can_confirm"])
        self.assertEqual([item["failure_id"] for item in self.failures()], [second["failure_id"]])
        self.assertIn(self.ref, self.session.failed_writes)
        self.assertEqual(self.session._patch_failures[first["failure_id"]]["superseded_by"], second["failure_id"])
        old_result = parsed(self.session.confirm_patch_satisfied([first["failure_id"]]))
        self.assertEqual(old_result["resolved_count"], 0)
        before = dict(self.session.pending)
        changes = self.session.change_count
        result = parsed(self.session.confirm_patch_satisfied([second["failure_id"]]))
        self.assertEqual(result["resolved_count"], 1)
        self.assertEqual(result["changed_count"], 0)
        self.assertEqual(self.session.pending, before)
        self.assertEqual(self.session.change_count, changes)
        self.assertEqual(self.session.validate(), [])

    def test_superseding_same_request_does_not_clear_other_content_or_summary_failures(self):
        replacements = [{"old_text": self.old, "new_text": self.new}]
        first = self.reject(replacements)
        other = self.reject([{"old_text": self.other_old, "new_text": self.other_new}])
        self.session.patch_entity(self.ref, [{"old_text": "崔巉", "new_text": "崔瀺"}], field="summary")
        summary = self.reject([{"old_text": "未定义的摘要内容", "new_text": "预期摘要内容"}], field="summary")
        second = self.reject(replacements)
        self.assertEqual(second["supersedes"], [first["failure_id"]])
        active = {item["failure_id"] for item in self.failures()}
        self.assertEqual(active, {other["failure_id"], summary["failure_id"], second["failure_id"]})
        result = parsed(self.session.confirm_patch_satisfied([second["failure_id"]]))
        self.assertEqual(result["resolved_count"], 1)
        self.assertEqual({item["failure_id"] for item in self.failures()}, {other["failure_id"], summary["failure_id"]})
        self.assertIn(self.ref, self.session.failed_writes)

    def test_only_selected_failure_is_confirmed_and_unknown_id_clears_nothing(self):
        first = self.reject([{"old_text": self.old, "new_text": self.new}])
        second = self.reject([{"old_text": self.other_old, "new_text": self.other_new}])
        before = dict(self.session.failed_writes)
        self.assertEqual(parsed(self.session.confirm_patch_satisfied(["P99999"]))["resolved_count"], 0)
        self.assertEqual(self.session.failed_writes, before)
        result = parsed(self.session.confirm_patch_satisfied([first["failure_id"]]))
        self.assertEqual(result["resolved_count"], 1)
        self.assertEqual([failure["failure_id"] for failure in self.failures()], [second["failure_id"]])
        self.assertIn(self.ref, self.session.failed_writes)

    def test_short_ambiguous_empty_and_noop_requests_cannot_be_confirmed(self):
        cases = [
            ("后来蒋去收到崔瀺的亲笔回信。", "崔巉", "崔瀺"),
            (f"{self.new}。{self.new}。", self.old, self.new),
            ("正文已不存在原句。", self.old, ""),
            (f"{self.new}。", self.new, self.new),
        ]
        for body, old, new in cases:
            with self.subTest(old=old, new=new, body=body):
                self.stage(body)
                failure = self.reject([{"old_text": old, "new_text": new}])
                self.assertFalse(failure["can_confirm"])
                before = dict(self.session.pending)
                result = parsed(self.session.confirm_patch_satisfied([failure["failure_id"]]))
                self.assertEqual(result["resolved_count"], 0)
                self.assertEqual(self.session.pending, before)
                self.assertIn(self.ref, self.session.failed_writes)

    def test_confirm_does_not_clear_generic_link_or_management_failures(self):
        self.stage(f"{self.new}。[[characters/不存在的人物]]")
        failure = self.reject([{"old_text": self.old, "new_text": self.new}])
        self.session.upsert_entity("character", "蒋去", "未提交正文", [1], summary="")
        self.stage("正确记录。", name="正确名称")
        inspection = parsed(self.session.inspect_repairs())
        entity_id = next(entity["entity_id"] for entity in inspection["entities"] if entity["ref"] == self.ref)
        self.session.manage_entities([{"action": "rename", "entity_id": entity_id, "new_name": "正确名称"}])
        self.session.repair_links([{"issue_id": inspection["issues"][0]["issue_id"], "target_id": "E99999"}])
        result = parsed(self.session.confirm_patch_satisfied([failure["failure_id"]]))
        self.assertEqual(result["resolved_count"], 1)
        errors = self.session.validate()
        self.assertTrue(any("summary 不能为空" in error for error in errors))
        self.assertTrue(any("链接修复失败" in error for error in errors))
        self.assertTrue(any("实体管理失败" in error for error in errors))

    def test_summary_patch_changes_only_metadata_and_preserves_body_sources_aliases(self):
        record_before, document_before = self.session.pending[self.ref]
        body_before = document_before.split("\n\n", 1)[1]
        result = self.session.patch_entity(self.ref, [{"old_text": "崔巉", "new_text": "崔瀺"}], field="summary")
        self.assertTrue(result.startswith("[已暂存]"))
        record_after, document_after = self.session.pending[self.ref]
        self.assertEqual(record_after.summary, "收到崔瀺来信")
        self.assertEqual(record_after.source_chapters, record_before.source_chapters)
        self.assertEqual(record_after.aliases, record_before.aliases)
        self.assertEqual(document_after.split("\n\n", 1)[1], body_before)
        self.assertIn('"summary":"收到崔瀺来信"', document_after)

    def test_summary_cannot_be_empty_and_content_patch_does_not_clear_its_failure(self):
        record, document = self.session.pending[self.ref]
        self.reject([{"old_text": record.summary, "new_text": ""}], field="summary")
        self.assertEqual(self.session.pending[self.ref], (record, document))
        result = self.session.patch_entity(self.ref, [{"old_text": "其他旧句待修正", "new_text": "其他句子已修正"}])
        self.assertTrue(result.startswith("[已暂存]"))
        self.assertEqual([failure["field"] for failure in self.failures()], ["summary"])
        self.assertIn(self.ref, self.session.failed_writes)

    def test_unrelated_successful_patch_does_not_clear_unfinished_same_field_failure(self):
        self.stage("旧句必须改正。其他旧句待修正。")
        failure = self.reject([
            {"old_text": "旧句必须改正", "new_text": "目标修改尚未完成"},
            {"old_text": "不存在的另一个原句", "new_text": "另一个目标也没写入"},
        ])
        result = self.session.patch_entity(self.ref, [{"old_text": "其他旧句待修正", "new_text": "其他句子已修正"}])
        self.assertTrue(result.startswith("[已暂存]"))
        self.assertIn(failure["failure_id"], [item["failure_id"] for item in self.failures()])
        self.assertIn(self.ref, self.session.failed_writes)

    def test_corrected_old_context_can_resolve_same_target_by_successful_patch_receipt(self):
        self.stage("经历待修正。")
        self.reject([{"old_text": "不存在的旧正文", "new_text": "曾到访青山。"}])
        result = self.session.patch_entity(self.ref, [{"old_text": "经历待修正。", "new_text": "曾到访青山。"}])
        self.assertTrue(result.startswith("[已暂存]"))
        self.assertEqual(self.failures(), [])
        self.assertEqual(self.session.validate(), [])

    def test_unrelated_success_does_not_implicitly_confirm_already_satisfied_request(self):
        failure = self.reject([{"old_text": self.old, "new_text": self.new}])
        self.assertTrue(failure["can_confirm"])
        result = self.session.patch_entity(self.ref, [{"old_text": "其他旧句待修正", "new_text": "其他句子已修正"}])
        self.assertTrue(result.startswith("[已暂存]"))
        self.assertEqual([item["failure_id"] for item in self.failures()], [failure["failure_id"]])
        result = parsed(self.session.confirm_patch_satisfied([failure["failure_id"]]))
        self.assertEqual(result["resolved_count"], 0)
        self.assertIn("版本已变化", result["results"][0]["result"])

    def test_conflicting_duplicate_old_request_is_never_confirmable(self):
        failure = self.reject([
            {"old_text": self.old, "new_text": self.new},
            {"old_text": self.old, "new_text": self.other_new},
        ])
        self.assertFalse(failure["can_confirm"])
        result = parsed(self.session.confirm_patch_satisfied([failure["failure_id"]]))
        self.assertEqual(result["resolved_count"], 0)
        self.assertIn(self.ref, self.session.failed_writes)

    def test_one_existing_new_occurrence_cannot_confirm_two_distinct_requests(self):
        failure = self.reject([
            {"old_text": self.old, "new_text": self.new},
            {"old_text": "后来蒋去收到崔嵬的亲笔回信", "new_text": self.new},
        ])
        self.assertFalse(failure["can_confirm"])
        self.assertIn("无法唯一对应", failure["reason"])
        result = parsed(self.session.confirm_patch_satisfied([failure["failure_id"]]))
        self.assertEqual(result["resolved_count"], 0)

    def test_patch_can_copy_exact_canonical_name_by_registered_entity_id(self):
        self.stage("已有正确人物记录。", name="崔瀺", summary="正确名字")
        target = next(match for match in parsed(self.session.search_entities(["崔瀺"]))["matches"] if match["name"] == "崔瀺")
        source_id = next(match for match in parsed(self.session.search_entities(["蒋去"]))["matches"] if match["name"] == "蒋去")["entity_id"]
        before = self.session.pending[self.ref][1].split("\n\n", 1)[1]
        result = self.session.patch_entity(source_id, [{"old_text": "崔巉", "new_entity_id": target["entity_id"]}], field="summary")
        self.assertTrue(result.startswith("[已暂存]"))
        self.assertEqual(self.session.pending[self.ref][0].summary, "收到崔瀺来信")
        self.assertEqual(self.session.pending[self.ref][1].split("\n\n", 1)[1], before)
        self.assertEqual(self.session.read_entity(source_id), self.session.read_entity(self.ref))
        self.assertTrue(self.session.read_entity("E99999").startswith("[错误]"))

    def test_invalid_or_mixed_name_replacement_targets_do_not_mutate_document(self):
        for replacement in (
            {"old_text": "崔巉", "new_entity_id": "E99999"},
            {"old_text": "崔巉", "new_entity_id": "E99999", "new_text": "崔瀺"},
        ):
            with self.subTest(replacement=replacement):
                before = dict(self.session.pending)
                result = self.session.patch_entity(self.ref, [replacement], field="summary")
                self.assertTrue(result.startswith("[错误]"))
                self.assertEqual(self.session.pending, before)
                self.assertFalse(self.failures()[-1]["can_confirm"])

    def test_full_upsert_clears_all_patch_fields_and_ordinary_write_failures(self):
        self.reject([{"old_text": self.old, "new_text": self.new}])
        self.reject([{"old_text": "不存在的摘要内容", "new_text": "目标摘要"}], field="summary")
        self.session.upsert_entity("character", "蒋去", "新正文", [1], summary="")
        self.stage("已重新确认的完整正文。", summary="重新确认的完整摘要")
        self.assertEqual(self.failures(), [])
        self.assertEqual(self.session.failed_writes, {})
        self.assertEqual(self.session.validate(), [])


if __name__ == "__main__":
    unittest.main()
