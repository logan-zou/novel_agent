import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from src.novel.corpus import CorpusStore
from src.novel.knowledge import (
    BuildSession, KnowledgeStore, _document_body, _document_fingerprint, _render_document,
)


_BOS = "<｜begin▁of▁sentence｜>"
_EOS = "<｜end▁of▁sentence｜>"


def parsed(result):
    return json.loads(result.removeprefix("[错误]").strip())


class EntityFieldsAndTokensTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        CorpusStore(self.root).save_chapters([{"chapter_no": 1, "title": "朝会", "text": "原文"}])
        self.store = KnowledgeStore(self.root)
        self.session = BuildSession(self.store)
        self.ref = "characters/姜述.md"
        self.summary = "齐天子，御极七十九年。"
        self.body = "## 经历\n召见鲍玄镜。\n\n## 关系\n与天妃相识。"
        self.stage()

    def stage(self, **updates):
        values = dict(kind="character", name="姜述", content=self.body, source_chapters=[1],
                      aliases=["齐天子"], summary=self.summary, domain="")
        values.update(updates)
        return self.session.upsert_entity(**values)

    def test_upsert_rejects_protocol_in_each_field_without_changing_pending(self):
        fields = {"name": "姜" + _BOS, "summary": "御" + _BOS + "七十九年", "content": "朝会" + _EOS,
                  "aliases": ["齐天子" + _BOS], "domain": "齐国" + _EOS}
        for field, value in fields.items():
            with self.subTest(field=field):
                before = dict(self.session.pending)
                changes = self.session.change_count
                result = self.stage(**{field: value})
                self.assertTrue(result.startswith("[错误]"), result)
                self.assertIn(field, result)
                self.assertIn("模型协议特殊标记", result)
                self.assertIn("重新读取相关原文", result)
                self.assertEqual(self.session.pending, before)
                self.assertEqual(self.session.change_count, changes)

    def test_dsml_control_family_is_rejected_but_ordinary_tags_are_accepted(self):
        controls = [
            '<｜DSML｜tool_calls>', '</｜DSML｜tool_calls>',
            '<｜DSML｜invoke name="upsert_entities">', '</｜DSML｜invoke>',
            '<｜DSML｜parameter name="entities" string="false">', '</｜DSML｜parameter>',
            '<｜DSML｜function_calls>', '<|DSML|invoke name="legacy">',
        ]
        for token in controls:
            with self.subTest(token=token):
                result = self.stage(content="正文" + token)
                self.assertTrue(result.startswith("[错误]"), result)
                self.assertIn(token, result)
        text = '他说：“DSML 是书名。”<标签>正文</标签><div class="note">旁注</div><think>思考</think>'
        self.assertTrue(self.stage(content=text).startswith("[已暂存]"))
        self.assertEqual(self.session.read_entity(self.ref, field="content"), text)
        self.assertEqual(self.session.validate(), [])

    def test_patch_outputs_are_guarded_atomically_for_summary_and_content(self):
        for field, old in (("summary", "御极"), ("content", "召见")):
            with self.subTest(field=field):
                before = dict(self.session.pending)
                changes = self.session.change_count
                result = self.session.patch_entity(self.ref, [{"old_text": old, "new_text": old + _BOS}], field)
                self.assertTrue(result.startswith("[错误]"), result)
                self.assertIn(field, result)
                self.assertIn(_BOS, result)
                self.assertEqual(self.session.pending, before)
                self.assertEqual(self.session.change_count, changes)
        self.assertEqual({item["field"] for item in parsed(self.session.inspect_repairs())["patch_failures"]},
                         {"summary", "content"})

    def test_rename_rejects_protocol_without_redirecting_other_entities(self):
        self.session.upsert_entity("character", "天妃", "丈夫是[[characters/姜述]]。", [1], summary="天妃")
        entity_id = next(item["entity_id"] for item in parsed(self.session.inspect_repairs())["entities"]
                         if item["ref"] == self.ref)
        before = dict(self.session.pending)
        changes = self.session.change_count
        result = parsed(self.session.manage_entities([{"action": "rename", "entity_id": entity_id,
                                                     "new_name": "姜" + _BOS}]))
        self.assertEqual(result["failed_count"], 1)
        self.assertIn("name", result["results"][0]["result"])
        self.assertEqual(self.session.pending, before)
        self.assertEqual(self.session.change_count, changes)

    def test_summary_content_and_document_reads_match_before_and_after_commit(self):
        document = self.session.pending[self.ref][1]
        entity_id = next(item["entity_id"] for item in parsed(self.session.inspect_repairs())["entities"]
                         if item["ref"] == self.ref)
        for ref in (self.ref, "characters/姜述", entity_id):
            self.assertEqual(self.session.read_entity(ref), document)
            self.assertEqual(self.session.read_entity(ref, field="summary"), self.summary)
            self.assertEqual(self.session.read_entity(ref, field="content"), self.body)
        self.session.commit()
        fresh = BuildSession(self.store)
        disk_id = parsed(fresh.search_entities(["姜述"]))["matches"][0]["entity_id"]
        for reader, ref in ((self.store, self.ref), (fresh, disk_id)):
            self.assertEqual(reader.read_entity(ref), document)
            self.assertEqual(reader.read_entity(ref, field="summary"), self.summary)
            self.assertEqual(reader.read_entity(ref, field="content"), self.body)

    def test_section_summary_compatibility_preserves_real_heading_priority(self):
        self.assertEqual(self.session.read_entity(self.ref, section="summary"), self.summary)
        self.session.commit()
        self.assertEqual(self.store.read_entity(self.ref, section="summary"), self.summary)
        self.stage(content="## summary\n实际同名小节。\n\n## 其他\n其他内容。")
        expected = "## summary\n实际同名小节。"
        self.assertEqual(self.session.read_entity(self.ref, section="summary"), expected)
        self.assertEqual(self.session.read_entity(self.ref, section="summary", field="content"), expected)
        self.assertEqual(self.session.read_entity(self.ref, field="summary"), self.summary)
        self.session.commit()
        self.assertEqual(self.store.read_entity(self.ref, section="summary"), expected)

    def test_summary_field_conflicts_with_section_or_anchor_and_unknown_field_is_error(self):
        self.session.commit()
        for reader in (self.session, self.store):
            for ref, section in ((self.ref, "经历"), (self.ref + "#经历", ""),
                                 ("[[characters/姜述#经历|姜述]]", "")):
                with self.subTest(reader=type(reader).__name__, ref=ref):
                    result = reader.read_entity(ref, section=section, field="summary")
                    self.assertTrue(result.startswith("[错误]"), result)
                    self.assertIn("不能同时", result)
            self.assertTrue(reader.read_entity(self.ref, field="summary_note").startswith("[错误]"))
        self.assertTrue(self.session.read_entity("E99999", field="summary").startswith("[错误]"))

    def test_legacy_metadata_less_summary_returns_error_and_document_stays_readable(self):
        path = self.root / "characters" / "旧档案.md"
        path.parent.mkdir(parents=True)
        text = "# 旧档案\n\n## 经历\n旧正文。"
        path.write_text(text, encoding="utf-8")
        for reader in (self.session, self.store):
            self.assertEqual(reader.read_entity("characters/旧档案"), text)
            self.assertEqual(reader.read_entity("characters/旧档案", field="content"), text)
            for kwargs in ({"field": "summary"}, {"section": "summary"}):
                result = reader.read_entity("characters/旧档案", **kwargs)
                self.assertTrue(result.startswith("[错误]"), result)
                self.assertIn("元数据摘要", result)

    def test_summary_match_error_reports_current_exact_summary_and_document_version(self):
        document = self.session.pending[self.ref][1]
        result = self.session.patch_entity(self.ref, [{"old_text": "御#七十九年", "new_text": "御极七十九年"}],
                                           field="summary")
        self.assertIn("匹配 0 处", result)
        self.assertIn(self.summary, result)
        self.assertIn("field=summary", result)
        self.assertIn(_document_fingerprint(document), result)
        self.assertEqual(self.session.pending[self.ref][1], document)

    def test_legacy_pollution_is_readable_rejected_by_validation_and_exact_patch_repairs_only_summary(self):
        record, document = self.session.pending[self.ref]
        bad_summary = self.summary.replace("御极", "御" + _BOS + "#")
        polluted = replace(record, summary=bad_summary)
        # 模拟旧版本运行已暂存污染；写入 API 不提供绕过新 guard 的路径。
        bad_document = _render_document(polluted, self.body)
        self.session.pending[self.ref] = (polluted, bad_document)
        self.assertEqual(self.session.read_entity(self.ref, field="summary"), bad_summary)
        self.assertEqual(self.session.read_entity(self.ref, section="summary"), bad_summary)
        self.assertTrue(any("summary 含模型协议特殊标记" in error for error in self.session.validate()))
        with self.assertRaisesRegex(ValueError, "模型协议特殊标记"):
            self.session.commit()
        self.session.patch_entity(self.ref, [{"old_text": "不相干的旧正文", "new_text": "另一个目标"}])
        result = self.session.patch_entity(self.ref, [{"old_text": "御" + _BOS + "#七十九年",
                                                      "new_text": "御极七十九年"}], field="summary")
        self.assertTrue(result.startswith("[已暂存]"), result)
        current, repaired = self.session.pending[self.ref]
        self.assertEqual(current, record)
        self.assertEqual(repaired, document)
        self.assertEqual(_document_body(repaired), _document_body(bad_document))
        self.assertEqual(current.aliases, polluted.aliases)
        self.assertEqual(current.source_chapters, polluted.source_chapters)
        self.assertIn(self.ref, self.session.failed_writes)
        remaining = parsed(self.session.inspect_repairs())["patch_failures"]
        self.assertEqual([item["field"] for item in remaining], ["content"])

    def test_field_patch_never_imports_or_auto_strips_remaining_protocol_pollution(self):
        record, document = self.session.pending[self.ref]
        polluted = replace(record, summary=self.summary + _EOS)
        self.session.pending[self.ref] = (polluted, _render_document(polluted, self.body))
        before = dict(self.session.pending)
        result = self.session.patch_entity(self.ref, [{"old_text": "召见", "new_text": "召回"}])
        self.assertTrue(result.startswith("[错误]"), result)
        self.assertIn("summary", result)
        self.assertEqual(self.session.pending, before)

    def test_field_patch_permission_remains_pending_only(self):
        self.session.commit()
        fresh = BuildSession(self.store)
        result = fresh.patch_entity(self.ref, [{"old_text": "御极", "new_text": "在位"}], field="summary")
        self.assertTrue(result.startswith("[错误]"), result)
        self.assertIn("只能局部修改本批已暂存实体", result)
        self.assertEqual(fresh.pending, {})

    def test_version_fingerprints_full_long_document_instead_of_truncated_display(self):
        record, _ = self.session.pending[self.ref]
        long_body = "## summary\n" + "正文" * 16_000
        document = _render_document(record, long_body)
        path = self.root / self.ref
        path.parent.mkdir(parents=True)
        path.write_text(document, encoding="utf-8")
        expected = _document_fingerprint(document)
        self.assertNotEqual(self.store.read_entity(self.ref), document)
        self.assertEqual(self.store.entity_version(self.ref), expected)
        self.assertNotEqual(_document_fingerprint(self.store.read_entity(self.ref)), expected)
        self.assertEqual(self.store.read_entity(self.ref, field="summary"), self.summary)
        # 旧版 section 读取不受整文档展示上限影响。
        self.assertIn(long_body, self.store.read_entity(self.ref, section="summary"))
        fresh = BuildSession(self.store)
        entity_id = parsed(fresh.search_entities(["姜述"]))["matches"][0]["entity_id"]
        self.assertEqual(fresh.entity_version(entity_id), expected)
        self.assertEqual(fresh.entity_version("[[characters/姜述#summary|齐天子]]"), expected)
        self.assertEqual(fresh.entity_version("E99999"), "")
        self.assertEqual(fresh.entity_version("../越界"), "")
        self.assertEqual(fresh.entity_version("characters/不存在"), "")
        pending_id = parsed(self.session.inspect_repairs())["entities"][0]["entity_id"]
        pending_document = self.session.pending[self.ref][1]
        self.assertEqual(self.session.entity_version(pending_id), _document_fingerprint(pending_document))
        self.assertNotEqual(self.session.entity_version(self.ref), expected)


if __name__ == "__main__":
    unittest.main()
