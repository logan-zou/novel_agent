import json
import tempfile
import unittest
from pathlib import Path

from src.novel.corpus import CorpusStore
from src.novel.knowledge import BuildSession, KnowledgeStore


class KnowledgeStoreTests(unittest.TestCase):
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
