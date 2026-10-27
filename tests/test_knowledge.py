import json
import tempfile
import unittest
from pathlib import Path

from src.novel.corpus import CorpusStore
from src.novel.knowledge import BuildSession, KnowledgeStore


class KnowledgeStoreTests(unittest.TestCase):
    def test_unknown_patch_target_blocks_automatic_completion_without_poisoning_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            CorpusStore(tmp).save_chapters([{"chapter_no": 1, "title": "开始", "text": "正文"}])
            session = BuildSession(KnowledgeStore(tmp))
            session.upsert_entity("character", "张三", "旧正文", [1], summary="摘要")
            replacement = [{"old_text": "旧正文", "new_text": "新正文"}]

            self.assertTrue(session.patch_entity("characters/张叁", replacement).startswith("[错误]"))
            self.assertTrue(session.repair_completion_blocked)
            self.assertFalse(session.failed_writes)
            self.assertTrue(session.patch_entity("characters/张三", replacement).startswith("[已暂存]"))
            self.assertIsNone(session.validator("已确认完成"))
            session.commit()
            self.assertFalse((Path(tmp) / "characters" / "张叁.md").exists())

    def test_failed_write_remains_unresolved_until_same_entity_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            CorpusStore(tmp).save_chapters(
                [{"chapter_no": 1, "title": "开始", "text": "正文"}]
            )
            session = BuildSession(KnowledgeStore(tmp))
            session.upsert_entity("character", "张三", "原内容", [1], summary="摘要")

            session.patch_entity("[[characters/张三|张三]]", [
                {"old_text": "不匹配", "new_text": "新内容"},
            ])
            session.upsert_entity("character", "李四", "其他内容", [1], summary="摘要")

            self.assertEqual(list(session.failed_writes), ["characters/张三.md"])
            self.assertFalse(session.repair_completion_blocked)
            self.assertTrue(any("尚有未解决的写入失败" in error for error in session.validate()))
            with self.assertRaisesRegex(ValueError, "尚有未解决的写入失败"):
                session.commit()
            session.upsert_entity("character", "张三", "原内容", [1], summary="摘要")
            self.assertEqual(session.failed_writes, {})
            self.assertEqual(session.validate(), [])

    def test_unidentified_write_failure_only_blocks_automatic_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = BuildSession(KnowledgeStore(tmp))

            result = session.upsert_entity("character", "../非法名称", "正文", [1], summary="摘要")

            self.assertTrue(result.startswith("[错误]"))
            self.assertTrue(session.repair_completion_blocked)
            self.assertEqual(session.failed_writes, {})
            self.assertIsNone(session.validator("显式结束"))

    def test_patch_changes_only_body_and_allows_pending_link_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            CorpusStore(tmp).save_chapters(
                [{"chapter_no": 1, "title": "开始", "text": "正文"}]
            )
            session = BuildSession(KnowledgeStore(tmp))
            session.upsert_entity(
                "character", "张三", "## 关系\n师父是[[characters/朱琙|朱琙]]。", [1],
                aliases=["小三"], summary="师父是朱琙",
            )
            before_record, before_document = session.pending["characters/张三.md"]
            changes_before = session.change_count

            result = session.patch_entity("characters/张三", [
                {"old_text": "[[characters/朱琙|朱琙]]", "new_text": "[[characters/朱敛|朱敛]]"},
            ])

            record, document = session.pending["characters/张三.md"]
            self.assertTrue(result.startswith("[已暂存]"))
            self.assertEqual(record, before_record)
            self.assertEqual(document, before_document.replace(
                "[[characters/朱琙|朱琙]]", "[[characters/朱敛|朱敛]]"
            ))
            self.assertEqual(session.change_count, changes_before + 1)
            self.assertTrue(any("链接目标不存在" in error for error in session.validate()))
            session.upsert_entity("character", "朱敛", "张三的师父。", [1], summary="修士")
            self.assertEqual(session.validate(), [])
            session.commit()
            self.assertEqual(session.store.read_entity("characters/张三"), document)

    def test_patch_matches_original_body_without_cascading_replacements(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = BuildSession(KnowledgeStore(tmp))
            session.upsert_entity("character", "张三", "甲乙", [1], summary="甲乙")
            before = session.read_entity("characters/张三")

            result = session.patch_entity("characters/张三.md", [
                {"old_text": "甲", "new_text": "乙"},
                {"old_text": "乙", "new_text": "丙"},
            ])

            self.assertTrue(result.startswith("[已暂存]"))
            self.assertEqual(
                session.read_entity("characters/张三"),
                before.replace("\n\n甲乙\n\n", "\n\n乙丙\n\n"),
            )

    def test_patch_rejects_invalid_matches_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = BuildSession(KnowledgeStore(tmp))
            session.upsert_entity("character", "张三", "甲乙乙", [1], summary="只有摘要")
            before = dict(session.pending)
            changes_before = session.change_count
            cases = [
                ([], "非空数组"),
                ([{"old_text": "", "new_text": "丁"}], "非空文本"),
                ([{"old_text": "甲", "new_text": None}], "必须是文本"),
                ([{"old_text": "甲", "new_text": "甲"}], "相同"),
                ([{"old_text": "只有摘要", "new_text": "修改摘要"}], "匹配 0 处"),
                ([{"old_text": "乙", "new_text": "丁"}], "匹配 2 处"),
                ([{"old_text": "甲", "new_text": "丁"}, {"old_text": "丙", "new_text": "戊"}],
                 "replacements[1]"),
                ([{"old_text": "甲", "new_text": "丁"}, {"old_text": "甲", "new_text": "戊"}],
                 "重复指定"),
                ([{"old_text": "甲乙", "new_text": "丁"}, {"old_text": "乙乙", "new_text": "戊"}],
                 "匹配重叠"),
                ([{"old_text": "甲乙乙", "new_text": ""}], "不能为空"),
                ([{"old_text": "甲", "new_text": "[[李四]]"}], "完整知识库路径"),
                ([{"old_text": "甲", "new_text": "[[characters/../../outside]]"}], "路径越界"),
            ]
            for replacements, error_text in cases:
                with self.subTest(replacements=replacements):
                    result = session.patch_entity("characters/张三", replacements)
                    self.assertTrue(result.startswith("[错误]"), result)
                    self.assertIn(error_text, result)
                    self.assertIn("本实体这次所有替换均未应用", result)
                    self.assertIn("不要原样重试", result)
                    self.assertEqual(session.pending, before)
                    self.assertEqual(session.change_count, changes_before)

    def test_patch_counts_overlapping_matches_as_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = BuildSession(KnowledgeStore(tmp))
            session.upsert_entity("character", "张三", "甲甲甲", [1], summary="摘要")

            result = session.patch_entity("characters/张三", [
                {"old_text": "甲甲", "new_text": "乙"},
            ])

            self.assertIn("匹配 2 处", result)
            self.assertIn("read_entities", result)
            self.assertIn("从中复制 old_text", result)
            self.assertIn("\n\n甲甲甲\n\n", session.read_entity("characters/张三"))

    def test_patch_cannot_change_unstaged_entities_or_use_unsafe_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            CorpusStore(tmp).save_chapters(
                [{"chapter_no": 1, "title": "开始", "text": "正文"}]
            )
            store = KnowledgeStore(tmp)
            previous = BuildSession(store)
            previous.upsert_entity("character", "张三", "原内容", [1], summary="摘要")
            previous.commit()
            session = BuildSession(store)
            replacements = [{"old_text": "原内容", "new_text": "新内容"}]
            result = session.patch_entity("characters/张三", replacements)
            self.assertIn("只能局部修改本批已暂存实体", result)
            self.assertEqual(session.pending, {})
            self.assertIn("原内容", store.read_entity("characters/张三"))

            session.upsert_entity("character", "张三", "原内容", [1], summary="摘要")
            for ref, error in [
                ("../../outside", "路径越界"),
                ("characters/张三#身份", "不能包含小节锚点"),
            ]:
                with self.subTest(ref=ref):
                    result = session.patch_entity(ref, replacements)
                    self.assertIn(error, result)
                    self.assertIn("原内容", session.read_entity("characters/张三"))

    def test_invalid_update_is_rejected_without_replacing_good_entity(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = BuildSession(KnowledgeStore(tmp))
            session.upsert_entity("character", "张三", "原内容", [1], summary="摘要")
            before = dict(session.pending)
            for content, summary in [("新内容", ""), ("[[李四]]", "摘要"), ("[[characters/../../outside]]", "摘要")]:
                result = session.upsert_entity("character", "张三", content, [1], summary=summary)
                self.assertTrue(result.startswith("[错误]"))
                self.assertEqual(session.pending, before)
            result = session.upsert_entity("character", "张三", "[[characters/李四|李四]]", [1], summary="摘要")
            self.assertTrue(result.startswith("[已暂存]"))

    def test_session_commits_entity_and_generates_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            CorpusStore(tmp).save_chapters(
                [{"chapter_no": 1, "title": "开始", "text": "张三拜师。"}]
            )
            store = KnowledgeStore(tmp)
            session = BuildSession(store)
            result = session.upsert_entity(
                "character",
                "张三",
                "## 身份\n剑客。",
                [1],
                aliases=["小三"],
                summary="年轻剑客",
            )
            self.assertIn("已暂存", result)
            self.assertFalse((Path(tmp) / "characters" / "张三.md").exists())
            session.commit()
            matches = json.loads(store.search_entities(["小三"]))["matches"]

        self.assertEqual(matches[0]["ref"], "characters/张三.md")
        self.assertEqual(matches[0]["source_chapters"], [1])

    def test_validation_sees_pending_links_and_rejects_missing_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            CorpusStore(tmp).save_chapters(
                [{"chapter_no": 1, "title": "开始", "text": "正文"}]
            )
            session = BuildSession(KnowledgeStore(tmp))
            session.upsert_entity("setting", "筑基", "第二境。", [1], summary="第二境", domain="力量体系")
            session.upsert_entity(
                "character",
                "张三",
                "境界：[[settings/力量体系/筑基]]",
                [1],
                summary="修士",
            )
            self.assertEqual(session.validate(), [])
            session.upsert_entity(
                "character",
                "李四",
                "师父：[[characters/不存在]]",
                [1],
                summary="修士",
            )
            errors = session.validate()

        self.assertTrue(any("链接目标不存在" in error for error in errors))

    def test_pending_entity_supports_wikilink_display_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            CorpusStore(tmp).save_chapters(
                [{"chapter_no": 1, "title": "开始", "text": "正文"}]
            )
            session = BuildSession(KnowledgeStore(tmp))
            session.upsert_entity(
                "character", "张三", "## 身份\n剑客。", [1], summary="年轻剑客"
            )

            result = session.read_entity("[[characters/张三|张三]]", "身份")

        self.assertEqual(result, "## 身份\n剑客。\n\n---\n来源章节：1")

    def test_rejects_entity_names_that_break_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = KnowledgeStore(tmp)

            for name in ("甲#身份", "甲|别名", "[[甲]]", "甲/乙", "甲\n乙"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    store.entity_ref("character", name)

    def test_legacy_anchor_reads_exact_section_even_in_large_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings" / "力量体系" / "神话生物.md"
            path.parent.mkdir(parents=True)
            path.write_text(
                "# 神话生物\n\n## 前部\n" + "x" * 35_000 + "\n## 海拉\n死亡女神。\n",
                encoding="utf-8",
            )
            result = KnowledgeStore(tmp).read_entity("settings/力量体系/神话生物#海拉")

        self.assertEqual(result, "## 海拉\n死亡女神。")


if __name__ == "__main__":
    unittest.main()
