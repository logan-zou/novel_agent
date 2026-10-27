import json
import tempfile
import unittest
from pathlib import Path

from src.novel.knowledge import BuildSession, EntityRecord, KnowledgeStore


def record(name, summary="人物经历。", *, aliases=(), kind="character", ref=None):
    return EntityRecord(kind, name, ref or f"characters/{name}.md", tuple(aliases), summary, (1,))


class EntitySearchTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.store = KnowledgeStore(self.root)

    def search(self, keywords, records=(), **options):
        return json.loads(self.store.search_entities(keywords, extra_records=list(records), **options))

    def write_catalog(self, records):
        self.store.catalog_path.write_text(
            "".join(json.dumps(item.to_dict(), ensure_ascii=False) + "\n" for item in records),
            encoding="utf-8",
        )

    def write_legacy(self, text):
        target = self.root / "characters/_index.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    def test_exact_name_and_alias_survive_many_multi_keyword_summary_hits(self):
        records = [record(f"关联事件{i:02}", "景国秦国之间的大事。") for i in range(40)]
        records += [record("吴询"), record("甲真人", aliases=["兵仙"]), record("吴询传闻")]

        result = self.search(["吴询", "兵仙", "景国", "秦国"], records)

        names = [item["name"] for item in result["matches"]]
        self.assertEqual(set(names[:2]), {"吴询", "甲真人"})
        self.assertEqual(names[2], "吴询传闻")
        self.assertEqual(result["total_matches"], 43)
        self.assertEqual(result["returned_count"], 30)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["uncovered_keywords"], [])

    def test_casefold_and_duplicate_keywords_do_not_inflate_ranking_or_counts(self):
        records = [record("Straße"), record("张三", "Strasse")]
        unique = self.search(["STRASSE", "张三"], records)
        repeated = self.search(["  STRASSE ", "strasse", "Straße", "张三", "", " "], records)

        self.assertEqual(repeated, unique)
        self.assertEqual(len(repeated["keyword_matches"]), 2)
        first = repeated["keyword_matches"][0]
        self.assertEqual(first["keyword"], "STRASSE")
        self.assertEqual(first["exact_count"], 1)
        self.assertEqual(first["name_count"], 1)
        self.assertEqual(first["summary_count"], 1)
        self.assertEqual(first["total_count"], 2)
        self.assertEqual(first["returned_count"], 2)

    def test_exact_matches_beyond_limit_are_reported_for_targeted_followup(self):
        records = [record(f"人物{i:02}") for i in range(35)]
        keywords = [item.name for item in records]

        result = self.search(keywords, records)

        self.assertEqual(result["uncovered_keywords"], keywords[30:])
        self.assertEqual(result["unmatched_keywords"], [])
        omitted = result["keyword_matches"][30]
        self.assertEqual(omitted["status"], "omitted")
        self.assertEqual(omitted["exact_count"], 1)
        self.assertEqual(omitted["returned_exact_count"], 0)
        self.assertEqual(result["limit"], 30)
        followup = self.search(result["uncovered_keywords"], records)
        self.assertEqual({item["name"] for item in followup["matches"]}, set(keywords[30:]))
        self.assertEqual(followup["uncovered_keywords"], [])

    def test_missing_keyword_is_distinct_from_existing_but_omitted_candidate(self):
        result = self.search(["张三", "藏书", "未出现"], [record("张三"), record("李四", "收藏藏书。")], limit=1)

        status = {item["keyword"]: item for item in result["keyword_matches"]}
        self.assertEqual(status["张三"]["status"], "complete")
        self.assertEqual(status["藏书"]["status"], "omitted")
        self.assertEqual(status["未出现"]["status"], "not_found")
        self.assertEqual(status["未出现"]["total_count"], 0)
        self.assertEqual(result["uncovered_keywords"], ["藏书"])
        self.assertEqual(result["unmatched_keywords"], ["未出现"])
        self.assertIn("当前知识目录", result["instruction"])
        self.assertIn("不要原样重复", result["instruction"])

    def test_exact_candidate_omission_is_reported_even_when_summary_result_is_visible(self):
        result = self.search(
            ["甲", "乙"], [record("甲", "认识乙。"), record("乙")], limit=1,
        )

        self.assertEqual(result["matches"][0]["name"], "甲")
        item = result["keyword_matches"][1]
        self.assertEqual(item["status"], "partial")
        self.assertEqual(item["returned_count"], 1)
        self.assertEqual(item["exact_count"], 1)
        self.assertEqual(item["returned_exact_count"], 0)
        self.assertEqual(result["uncovered_keywords"], ["乙"])

    def test_kind_filter_and_pending_override_apply_to_counts_and_results(self):
        self.write_catalog([record("张三", "旧摘要"), record("门派", kind="setting", ref="settings/势力组织/门派.md")])
        before = self.store.catalog_path.read_bytes()
        session = BuildSession(self.store)
        session.upsert_entity("character", "张三", "更新经历。", [1], summary="新摘要")
        session.upsert_entity("character", "李四", "另一人物。", [1], summary="新摘要")

        result = json.loads(session.search_entities(["新摘要", "旧摘要", "门派"], entity_types=["character"]))

        self.assertEqual(result["total_matches"], 2)
        self.assertEqual(result["unmatched_keywords"], ["旧摘要", "门派"])
        self.assertTrue(all(item["entity_id"].startswith("E") for item in result["matches"]))
        self.assertEqual({item["name"] for item in result["matches"]}, {"张三", "李四"})
        self.assertEqual(self.store.catalog_path.read_bytes(), before)
        self.assertFalse((self.root / "characters/李四.md").exists())

    def test_legacy_ref_deduplication_preserves_distinct_anchors_and_display_alias(self):
        self.write_catalog([record("张三")])
        self.write_legacy(
            "- [[characters/张三]]：张三。\n"
            "- [[characters/张三.md]]：张三重复。\n"
            "- [[characters/主要角色#李四|小四]]：剑客。\n"
            "- [[characters/主要角色.md#李四|小四]]：剑客重复。\n"
            "- [[characters/主要角色#王五]]：另一人物。\n"
        )

        result = self.search(["张三", "李四", "王五", "小四"])

        self.assertEqual(result["total_matches"], 3)
        names = {item["name"]: item for item in result["matches"]}
        self.assertNotIn("legacy", names["张三"])
        self.assertEqual(names["李四"]["ref"], "characters/主要角色#李四")
        self.assertEqual(names["李四"]["aliases"], ["小四"])
        self.assertEqual(names["王五"]["ref"], "characters/主要角色#王五")
        self.assertTrue(all(item["exact_count"] == 1 for item in result["keyword_matches"]))

    def test_legacy_plain_file_name_is_exact_and_duplicate_line_coverage_is_preserved(self):
        self.write_legacy(
            "- [[characters/张三.md]]：收藏剑谱。\n"
            "- [[characters/张三]]：收藏古画。\n"
        )

        result = self.search(["张三", "剑谱", "古画"])

        self.assertEqual(result["total_matches"], 1)
        self.assertEqual(result["matches"][0]["name"], "张三")
        self.assertTrue(all(item["returned_count"] == 1 for item in result["keyword_matches"]))
        self.assertEqual(result["keyword_matches"][0]["exact_count"], 1)
        self.assertNotIn("古画", result["matches"][0]["summary"])

    def test_limit_clamping_and_empty_results_keep_truthful_metadata(self):
        records = [record(f"人物{i:03}", "共同词") for i in range(110)]
        large = self.search(["共同词"], records, limit=1000)
        small = self.search(["共同词"], records, limit=0)
        empty = self.search(["不存在"], records)

        self.assertEqual((large["limit"], large["returned_count"], large["total_matches"]), (100, 100, 110))
        self.assertEqual((small["limit"], small["returned_count"]), (1, 1))
        self.assertTrue(small["truncated"])
        self.assertEqual(empty["matches"], [])
        self.assertFalse(empty["truncated"])
        self.assertEqual(empty["total_matches"], 0)
        self.assertEqual(empty["unmatched_keywords"], ["不存在"])


if __name__ == "__main__":
    unittest.main()
