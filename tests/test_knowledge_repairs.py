import json
import tempfile
import unittest
from pathlib import Path

from src.novel.corpus import CorpusStore
from src.novel.knowledge import BuildSession, KnowledgeStore


def payload(result):
    return json.loads(result.removeprefix("[错误]").strip())


class KnowledgeRepairTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        CorpusStore(self.root).save_chapters([{"chapter_no": 1, "title": "开始", "text": "原文"}])
        self.store = KnowledgeStore(self.root)
        self.session = BuildSession(self.store)

    def stage(self, name, body, *, kind="character", domain="", summary="摘要"):
        result = self.session.upsert_entity(kind, name, body, [1], summary=summary, domain=domain)
        self.assertTrue(result.startswith("[已暂存]"), result)
        return self.store.entity_ref(kind, name, domain)

    def inspect(self):
        return payload(self.session.inspect_repairs())

    def entity_id(self, name):
        matches = payload(self.session.search_entities([name]))["matches"]
        return next(item["entity_id"] for item in matches if item["name"] == name)

    def seed_links(self):
        self.stage("黄粱派", "## 沿革\n正确门派。", kind="setting", domain="势力组织")
        self.stage("高枕", "黄梁派掌门。", summary="掌门")
        return self.stage(
            "拜访", "黄梁派文本保持不变，拜访[[settings/势力组织/黄梁派.md#沿革|黄梁派]]。"
        )

    def test_inspection_is_readonly_stable_and_repairs_exact_links_by_id(self):
        source = self.seed_links()
        before = dict(self.session.pending)
        changes = self.session.change_count
        first = self.inspect()
        second = self.inspect()
        self.assertEqual(first, second)
        self.assertEqual(self.session.pending, before)
        self.assertEqual(self.session.change_count, changes)
        self.assertFalse(self.store.catalog_path.exists())
        issue = first["issues"][0]
        target = next(item for item in issue["candidates"] if item["name"] == "黄粱派")
        result = self.session.repair_links([{"issue_id": issue["issue_id"], "target_id": target["entity_id"]}])
        self.assertEqual(payload(result)["changed_count"], 1)
        updated = self.session.read_entity(source)
        self.assertIn("[[settings/势力组织/黄粱派.md#沿革|黄梁派]]", updated)
        self.assertIn("黄梁派文本保持不变", updated)
        self.assertEqual(self.session.validate(), [])
        self.session.commit()
        self.assertEqual(self.store.read_entity(source), updated)

    def test_candidate_similarity_does_not_select_or_merge_automatically(self):
        self.stage("朱敛", "修士。")
        self.stage("朱琼", "另一位修士。")
        source = self.stage("张三", "认识[[characters/朱琙]]。")
        before = dict(self.session.pending)
        issue = self.inspect()["issues"][0]
        self.assertGreaterEqual(len(issue["candidates"]), 2)
        self.assertEqual(self.session.pending, before)
        self.assertIn("链接目标不存在", self.session.validator("完成"))
        target_id = self.entity_id("朱敛")
        result = self.session.repair_links([{"issue_id": issue["issue_id"], "target_id": target_id}])
        self.assertEqual(payload(result)["failed_count"], 0)
        self.assertIn("[[characters/朱敛]]", self.session.read_entity(source))

    def test_stale_issue_rejects_changed_source_then_fresh_issue_can_repair(self):
        source = self.seed_links()
        issue = self.inspect()["issues"][0]
        target_id = self.entity_id("黄粱派")
        self.stage("拜访", "改写后的[[settings/势力组织/黄梁派]]。")
        before = dict(self.session.pending)
        changes = self.session.change_count
        result = self.session.repair_links([{"issue_id": issue["issue_id"], "target_id": target_id}])
        self.assertIn("已过期", result)
        self.assertEqual(self.session.pending, before)
        self.assertEqual(self.session.change_count, changes)
        fresh = self.inspect()["issues"][0]
        self.assertNotEqual(fresh["issue_id"], issue["issue_id"])
        result = self.session.repair_links([{"issue_id": fresh["issue_id"], "target_id": target_id}])
        self.assertEqual(payload(result)["failed_count"], 0)
        self.assertIn("改写后的[[settings/势力组织/黄粱派]]", self.session.read_entity(source))
        self.assertEqual(self.session.validate(), [])

    def test_discarded_target_id_cannot_be_reused_as_repair_target(self):
        self.seed_links()
        issue = self.inspect()["issues"][0]
        target_id = self.entity_id("黄粱派")
        self.session.manage_entities([{"action": "discard", "entity_id": target_id}])
        self.stage("别的门派", "另一门派。", kind="setting", domain="势力组织")
        self.assertNotEqual(self.entity_id("别的门派"), target_id)
        before = dict(self.session.pending)
        result = self.session.repair_links([{"issue_id": issue["issue_id"], "target_id": target_id}])
        self.assertIn("目标实体编号无效", result)
        self.assertEqual(self.session.pending, before)

    def test_multiple_repairs_for_one_source_are_atomic_and_share_original_snapshot(self):
        self.stage("朱敛", "修士。")
        self.stage("黄粱派", "门派。", kind="setting", domain="势力组织")
        source = self.stage("张三", "[[characters/朱琙]]曾访[[settings/势力组织/黄梁派]]。")
        issues = self.inspect()["issues"]
        operations = [{
            "issue_id": issue["issue_id"],
            "target_id": self.entity_id("朱敛" if issue["missing_ref"].startswith("characters/") else "黄粱派"),
        } for issue in issues]
        before = dict(self.session.pending)
        changes = self.session.change_count
        invalid = [operations[0], {**operations[1], "target_id": "E99999"}]
        result = payload(self.session.repair_links(invalid))
        self.assertEqual(result["failed_count"], 2)
        self.assertEqual(result["changed_count"], 0)
        self.assertEqual(self.session.pending, before)
        self.assertEqual(self.session.change_count, changes)
        result = payload(self.session.repair_links(operations))
        self.assertEqual(result["failed_count"], 0)
        self.assertEqual(result["changed_count"], 1)
        self.assertIn("[[characters/朱敛]]曾访[[settings/势力组织/黄粱派]]", self.session.read_entity(source))
        self.assertEqual(self.session.validate(), [])

    def test_successful_link_repair_can_replace_only_pure_noop_patch_failure(self):
        source = self.seed_links()
        old = "[[settings/势力组织/黄梁派.md#沿革|黄梁派]]"
        self.assertIn("[错误]", self.session.patch_entity(source, [{"old_text": old, "new_text": old}]))
        issue = self.inspect()["issues"][0]
        result = self.session.repair_links([{"issue_id": issue["issue_id"], "target_id": self.entity_id("黄粱派")}])
        self.assertEqual(payload(result)["failed_count"], 0)
        self.assertNotIn(source, self.session.failed_writes)
        self.assertEqual(self.session.validate(), [])

    def test_link_repair_preserves_mixed_patch_and_unrelated_field_failures(self):
        for failure in ("mixed_patch", "field_after_noop"):
            with self.subTest(failure=failure):
                self.session = BuildSession(self.store)
                source = self.seed_links()
                old = "[[settings/势力组织/黄梁派.md#沿革|黄梁派]]"
                replacements = [{"old_text": old, "new_text": old}]
                if failure == "mixed_patch":
                    replacements.insert(0, {"old_text": "文本保持不变", "new_text": "应保留的另一个修复"})
                self.session.patch_entity(source, replacements)
                if failure == "field_after_noop":
                    self.session.upsert_entity("character", "拜访", "新正文", [1], summary="")
                issue = self.inspect()["issues"][0]
                result = self.session.repair_links([{"issue_id": issue["issue_id"], "target_id": self.entity_id("黄粱派")}])
                self.assertEqual(payload(result)["failed_count"], 0)
                self.assertIn(source, self.session.failed_writes)
                self.assertTrue(any("尚有未解决的写入失败" in error for error in self.session.validate()))

    def test_rename_new_entity_updates_metadata_and_pending_links_without_changing_alias_or_anchor(self):
        old = self.stage("误名", "## 身份\n新实体。")
        incoming = self.stage("张三", "认识[[characters/误名.md#身份|旧称]]，原文写误名。")
        entity_id = self.entity_id("误名")
        result = self.session.manage_entities([{"action": "rename", "entity_id": entity_id, "new_name": "正名"}])
        self.assertEqual(payload(result)["changed_count"], 2)
        self.assertNotIn(old, self.session.pending)
        record, document = self.session.pending["characters/正名.md"]
        self.assertEqual(record.name, "正名")
        self.assertEqual(record.ref, "characters/正名.md")
        self.assertIn("\n# 正名\n", document)
        self.assertIn("[[characters/正名.md#身份|旧称]]，原文写误名", self.session.read_entity(incoming))
        self.assertEqual(self.entity_id("正名"), entity_id)
        self.assertEqual(self.session.validate(), [])
        self.session.commit()
        self.assertFalse((self.root / old).exists())
        self.assertTrue((self.root / "characters/正名.md").exists())

    def test_rename_conflict_does_not_overwrite_either_entity_or_count_progress(self):
        self.stage("误名", "误建内容。")
        self.stage("正名", "需保留的正确内容。")
        entity_id = self.entity_id("误名")
        before = dict(self.session.pending)
        changes = self.session.change_count
        result = self.session.manage_entities([{"action": "rename", "entity_id": entity_id, "new_name": "正名"}])
        self.assertIn("禁止覆盖", result)
        self.assertEqual(self.session.pending, before)
        self.assertEqual(self.session.change_count, changes)
        self.stage("误名", "误建内容。")
        self.assertTrue(any("实体管理失败" in error for error in self.session.validate()))
        with self.assertRaisesRegex(ValueError, "实体管理失败"):
            self.session.commit()
        result = self.session.manage_entities([{"action": "discard", "entity_id": entity_id}])
        self.assertEqual(payload(result)["failed_count"], 0)
        self.assertEqual(self.session.validate(), [])
        self.session.commit()
        self.assertNotIn("characters/误名.md", {record.ref for record in self.store.catalog_records()})
        self.assertIn("需保留的正确内容", self.store.read_entity("characters/正名"))

    def test_existing_persisted_entity_cannot_be_renamed_or_discarded(self):
        self.stage("张三", "已有内容。")
        self.session.commit()
        for action in ("rename", "discard"):
            with self.subTest(action=action):
                self.session = BuildSession(self.store)
                persisted_before = self.store.read_entity("characters/张三")
                self.stage("张三", "本批新内容。")
                entity_id = self.entity_id("张三")
                before = dict(self.session.pending)
                changes = self.session.change_count
                operation = {"action": action, "entity_id": entity_id}
                if action == "rename":
                    operation["new_name"] = "李四"
                result = self.session.manage_entities([operation])
                self.assertIn("只能撤销或更名本批新增", result)
                self.assertEqual(self.session.pending, before)
                self.assertEqual(self.session.change_count, changes)
                self.assertEqual(persisted_before, self.store.read_entity("characters/张三"))
                self.assertFalse((self.root / "characters/李四.md").exists())
                self.assertTrue(self.session.repair_completion_blocked)
                self.assertIsNone(self.session.validator("已确认不更改已有实体名称，显式结束"))
                self.session.commit()
                self.assertIn("本批新内容", self.store.read_entity("characters/张三"))

    def test_discard_rejects_incoming_link_then_clears_failures_when_explicitly_removed(self):
        old = self.stage("误名", "新建内容。")
        incoming = self.stage("张三", "认识[[characters/误名#身份|旧称]]。")
        entity_id = self.entity_id("误名")
        before = dict(self.session.pending)
        result = self.session.manage_entities([{"action": "discard", "entity_id": entity_id}])
        self.assertIn("仍被其他实体引用", result)
        self.assertEqual(self.session.pending, before)
        self.session.patch_entity(incoming, [{"old_text": "[[characters/误名#身份|旧称]]", "new_text": "旧称"}])
        result = self.session.manage_entities([{"action": "discard", "entity_id": entity_id}])
        self.assertEqual(payload(result)["failed_count"], 0)
        self.assertNotIn(old, self.session.pending)
        self.assertEqual(self.session.validate(), [])

    def test_persisted_incoming_link_requires_staging_before_rename(self):
        self.stage("张三", "认识旧称。")
        self.session.commit()
        persisted = self.root / "characters/张三.md"
        original = persisted.read_text()
        persisted.write_text(original.replace("认识旧称。", "认识[[characters/误名]]。"))
        self.session = BuildSession(self.store)
        self.stage("误名", "新实体。")
        entity_id = self.entity_id("误名")
        result = self.session.manage_entities([{"action": "rename", "entity_id": entity_id, "new_name": "正名"}])
        self.assertIn("未暂存的已落盘实体引用原名", result)
        self.assertIn("[[characters/误名]]", persisted.read_text())
        self.stage("张三", "认识[[characters/误名]]。")
        result = self.session.manage_entities([{"action": "rename", "entity_id": entity_id, "new_name": "正名"}])
        self.assertEqual(payload(result)["failed_count"], 0)
        self.assertIn("[[characters/正名]]", self.session.read_entity("characters/张三"))

    def test_invalid_batches_and_unknown_ids_are_bounded_and_block_automatic_completion(self):
        for method, operations in (
            (self.session.repair_links, []), (self.session.manage_entities, []),
            (self.session.repair_links, [{"issue_id": "L404", "target_id": "E404"}]),
            (self.session.manage_entities, [{"action": "discard", "entity_id": "E404"}]),
        ):
            with self.subTest(method=method.__name__, operations=operations):
                before = dict(self.session.pending)
                result = method(operations)
                self.assertTrue(result.startswith("[错误]"))
                self.assertEqual(self.session.pending, before)
                self.assertTrue(self.session.repair_completion_blocked)

    def late_target_case(self):
        source = self.stage("田常", "身上留有[[settings/力量体系/封魔印#口诀|封魔印]]。")
        self.stage("麒观应", "另一位人物。")
        issue = self.inspect()["issues"][0]
        self.stage("封魔印", "## 口诀\n封印之术。", kind="setting", domain="力量体系")
        return source, issue, self.entity_id("封魔印"), self.entity_id("麒观应")

    def test_original_target_created_after_inspection_is_confirmed_without_writing(self):
        _, issue, target_id, _ = self.late_target_case()
        before = dict(self.session.pending)
        changes = self.session.change_count
        result = payload(self.session.repair_links([{"issue_id": issue["issue_id"], "target_id": target_id}]))
        self.assertEqual(result["failed_count"], 0)
        self.assertEqual(result["changed_count"], 0)
        self.assertEqual(result["resolved_count"], 0)
        self.assertIn("链接已有效，未修改", result["results"][0]["result"])
        self.assertEqual(self.session.pending, before)
        self.assertEqual(self.session.change_count, changes)
        self.assertEqual(self.session.validate(), [])

    def test_wrong_target_error_has_explicit_confirmation_entry_and_correct_id_only_resolves_once(self):
        source, issue, target_id, wrong_id = self.late_target_case()
        before = dict(self.session.pending)
        changes = self.session.change_count
        result = payload(self.session.repair_links([{"issue_id": issue["issue_id"], "target_id": wrong_id}]))
        self.assertEqual(result["failed_count"], 1)
        self.assertEqual(result["changed_count"], 0)
        self.assertEqual(result["resolved_count"], 0)
        self.assertIn("指向其他实体", result["results"][0]["result"])
        self.assertIn(target_id, result["results"][0]["result"])
        self.assertEqual(self.session.pending, before)
        inspection = self.inspect()
        self.assertEqual(inspection["issues"], [])
        self.assertEqual(len(inspection["link_failures"]), 1)
        failure = inspection["link_failures"][0]
        self.assertEqual(failure["issue_id"], issue["issue_id"])
        self.assertEqual(failure["source_ref"], source)
        self.assertEqual(failure["missing_ref"], "settings/力量体系/封魔印.md")
        self.assertEqual(failure["target_id"], target_id)
        self.assertTrue(failure["can_confirm"])
        result = payload(self.session.repair_links([{"issue_id": failure["issue_id"], "target_id": failure["target_id"]}]))
        self.assertEqual(result["resolved_count"], 1)
        self.assertEqual(result["changed_count"], 0)
        self.assertEqual(self.session.pending, before)
        self.assertEqual(self.session.change_count, changes)
        self.assertEqual(self.session.validate(), [])
        repeated = payload(self.session.repair_links([{"issue_id": issue["issue_id"], "target_id": target_id}]))
        self.assertEqual(repeated["resolved_count"], 0)
        self.assertEqual(repeated["changed_count"], 0)
        self.assertEqual(self.inspect()["link_failures"], [])

    def test_created_target_does_not_make_stale_source_issue_confirmable(self):
        source, issue, target_id, _ = self.late_target_case()
        self.session.patch_entity(source, [{"old_text": "身上留有", "new_text": "体内保留"}])
        before = dict(self.session.pending)
        changes = self.session.change_count
        result = payload(self.session.repair_links([{"issue_id": issue["issue_id"], "target_id": target_id}]))
        self.assertEqual(result["failed_count"], 1)
        self.assertEqual(result["resolved_count"], 0)
        self.assertIn("已过期", result["results"][0]["result"])
        self.assertEqual(self.session.pending, before)
        self.assertEqual(self.session.change_count, changes)
        inspection = self.inspect()
        self.assertEqual(inspection["issues"], [])
        failure = inspection["link_failures"][0]
        self.assertFalse(failure["can_confirm"])
        self.assertIn("完整重新提交", failure["reason"])
        record, _ = self.session.pending[source]
        self.stage("田常", "体内保留[[settings/力量体系/封魔印#口诀|封魔印]]。", summary=record.summary)
        self.assertEqual(self.session.validate(), [])

    def test_wrong_correct_cycles_do_not_manufacture_repeated_resolution_progress(self):
        _, issue, target_id, wrong_id = self.late_target_case()
        before = dict(self.session.pending)
        changes = self.session.change_count
        for expected in (1, 0, 0):
            with self.subTest(expected_resolved=expected):
                rejected = payload(self.session.repair_links([{"issue_id": issue["issue_id"], "target_id": wrong_id}]))
                self.assertEqual(rejected["failed_count"], 1)
                self.assertEqual(rejected["resolved_count"], 0)
                self.assertTrue(self.inspect()["link_failures"])
                confirmed = payload(self.session.repair_links([{"issue_id": issue["issue_id"], "target_id": target_id}]))
                self.assertEqual(confirmed["failed_count"], 0)
                self.assertEqual(confirmed["resolved_count"], expected)
                self.assertEqual(confirmed["changed_count"], 0)
                self.assertEqual(self.session.pending, before)
                self.assertEqual(self.session.change_count, changes)
                self.assertEqual(self.session.validate(), [])

    def test_previously_valid_confirmation_cannot_gain_progress_by_creating_error_afterwards(self):
        _, issue, target_id, wrong_id = self.late_target_case()
        first = payload(self.session.repair_links([{"issue_id": issue["issue_id"], "target_id": target_id}]))
        self.assertEqual(first["resolved_count"], 0)
        rejected = payload(self.session.repair_links([{"issue_id": issue["issue_id"], "target_id": wrong_id}]))
        self.assertEqual(rejected["failed_count"], 1)
        confirmed = payload(self.session.repair_links([{"issue_id": issue["issue_id"], "target_id": target_id}]))
        self.assertEqual(confirmed["resolved_count"], 0)
        self.assertEqual(confirmed["changed_count"], 0)
        self.assertEqual(self.session.validate(), [])

    def test_confirmation_and_replacement_for_same_source_are_atomic(self):
        source = self.stage("田常", "[[settings/力量体系/封魔印]]与[[settings/力量体系/失传封印]]。")
        self.stage("封印术", "已有封印术。", kind="setting", domain="力量体系")
        self.stage("麒观应", "另一人物。")
        issues = {issue["missing_ref"]: issue for issue in self.inspect()["issues"]}
        original = issues["settings/力量体系/封魔印.md"]
        missing = issues["settings/力量体系/失传封印.md"]
        self.stage("封魔印", "封印之术。", kind="setting", domain="力量体系")
        target_id = self.entity_id("封魔印")
        self.session.repair_links([{"issue_id": original["issue_id"], "target_id": self.entity_id("麒观应")}])
        before = dict(self.session.pending)
        changes = self.session.change_count
        failed = payload(self.session.repair_links([
            {"issue_id": original["issue_id"], "target_id": target_id},
            {"issue_id": missing["issue_id"], "target_id": "E99999"},
        ]))
        self.assertEqual(failed["failed_count"], 2)
        self.assertEqual(failed["resolved_count"], 0)
        self.assertEqual(failed["changed_count"], 0)
        self.assertEqual(self.session.pending, before)
        self.assertEqual(self.session.change_count, changes)
        self.assertIn((source, "settings/力量体系/封魔印.md"), self.session._link_failures)
        succeeded = payload(self.session.repair_links([
            {"issue_id": original["issue_id"], "target_id": target_id},
            {"issue_id": missing["issue_id"], "target_id": self.entity_id("封印术")},
        ]))
        self.assertEqual(succeeded["failed_count"], 0)
        self.assertEqual(succeeded["changed_count"], 1)
        self.assertEqual(succeeded["resolved_count"], 2)
        self.assertIn("[[settings/力量体系/封魔印]]与[[settings/力量体系/封印术]]", self.session.read_entity(source))
        self.assertEqual(self.session.validate(), [])

    def test_link_only_confirmation_does_not_clear_patch_write_or_management_failures(self):
        source, issue, target_id, wrong_id = self.late_target_case()
        self.session.repair_links([{"issue_id": issue["issue_id"], "target_id": wrong_id}])
        link = "[[settings/力量体系/封魔印#口诀|封魔印]]"
        self.session.patch_entity(source, [{"old_text": link, "new_text": link}])
        self.session.upsert_entity("character", "田常", "未提交正文", [1], summary="")
        entity_id = self.entity_id("田常")
        self.session.manage_entities([{"action": "rename", "entity_id": entity_id, "new_name": "麒观应"}])
        before = dict(self.session.pending)
        result = payload(self.session.repair_links([{"issue_id": issue["issue_id"], "target_id": target_id}]))
        self.assertEqual(result["resolved_count"], 1)
        self.assertEqual(result["changed_count"], 0)
        self.assertEqual(self.session.pending, before)
        self.assertTrue(self.inspect()["patch_failures"])
        errors = self.session.validate()
        self.assertTrue(any("old_text 与 new_text 逐字相同" in error for error in errors))
        self.assertTrue(any("summary 不能为空" in error for error in errors))
        self.assertTrue(any("实体管理失败" in error for error in errors))
        self.assertFalse(any("链接修复失败" in error for error in errors))

    def test_locatable_invalid_arguments_prevent_same_source_confirmation_atomically(self):
        source, issue, target_id, wrong_id = self.late_target_case()
        self.session.repair_links([{"issue_id": issue["issue_id"], "target_id": wrong_id}])
        before = dict(self.session.pending)
        changes = self.session.change_count
        for invalid in (
            {"issue_id": issue["issue_id"], "target_id": target_id, "extra": True},
            {"issue_id": issue["issue_id"], "target_id": 42},
        ):
            with self.subTest(invalid=invalid):
                result = payload(self.session.repair_links([
                    {"issue_id": issue["issue_id"], "target_id": target_id}, invalid,
                ]))
                self.assertEqual(result["failed_count"], 2)
                self.assertEqual(result["resolved_count"], 0)
                self.assertEqual(result["changed_count"], 0)
                self.assertEqual(self.session.pending, before)
                self.assertEqual(self.session.change_count, changes)
                self.assertIn((source, issue["missing_ref"]), self.session._link_failures)
        final = payload(self.session.repair_links([{"issue_id": issue["issue_id"], "target_id": target_id}]))
        self.assertEqual(final["resolved_count"], 1)
        self.assertEqual(self.session.validate(), [])


if __name__ == "__main__":
    unittest.main()
