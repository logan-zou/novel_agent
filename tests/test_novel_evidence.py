import json
import unittest

from src.novel.evidence import EvidenceCollector, parse_citation_ids


class NovelEvidenceTests(unittest.TestCase):
    def test_only_search_source_evidence_can_be_used_as_final_citation(self):
        collector = EvidenceCollector()
        enriched = collector.capture(
            "read_entity",
            {"ref": "settings/力量体系/筑基.md"},
            "## 筑基期\n来源章节：3, 12\n筑基是第二境。",
        )

        self.assertIn("证据 EV001", enriched)
        report = collector.validate("筑基是第二境。[[cite:EV001]]")
        self.assertFalse(report.valid)
        self.assertIn("只能引用 search_source", report.message)

    def test_search_source_builds_clickable_context_around_match(self):
        source = "前" * 600 + "烛九阴" + "后" * 600
        collector = EvidenceCollector(source_reader=lambda chapter: source)
        result = collector.capture(
            "search_source",
            {"keywords": ["烛九阴"]},
            json.dumps(
                {
                    "keywords": ["烛九阴"],
                    "matches": [{"chapter": 58, "snippets": ["烛九阴是旧日集会成员。"]}],
                },
                ensure_ascii=False,
            ),
        )
        payload = json.loads(result)

        self.assertEqual(payload["matches"][0]["evidence_id"], "EV001")
        self.assertIn("[[cite:EVxxx]]", payload["citation_hint"])
        report = collector.validate("烛九阴是旧日集会成员。[[cite:EV001]]")
        citation = collector.cited_items("烛九阴是旧日集会成员。[[cite:EV001]]")[0]

        self.assertTrue(report.valid)
        self.assertEqual(report.cited_chapters, (58,))
        self.assertEqual(citation["number"], 1)
        self.assertEqual(citation["excerpt"], "前" * 500 + "烛九阴" + "后" * 500)

    def test_citations_are_ordered_by_first_appearance_and_reject_unknown_ids(self):
        chapters = {
            58: "甲" * 600 + "烛九阴" + "乙" * 600,
            64: "丙" * 600 + "时空之眼" + "丁" * 600,
        }
        collector = EvidenceCollector(source_reader=lambda chapter: chapters[chapter])
        collector.capture(
            "search_source",
            {"keywords": ["烛九阴"]},
            json.dumps({"matches": [{"chapter": 58, "snippets": ["烛九阴"]}]}, ensure_ascii=False),
        )
        collector.capture(
            "search_source",
            {"keywords": ["时空之眼"]},
            json.dumps({"matches": [{"chapter": 64, "snippets": ["时空之眼"]}]}, ensure_ascii=False),
        )

        answer = "先说第二条。[[cite:EV002]]再说第一条。[[cite:EV001]][[cite:EV002]]"
        self.assertEqual(parse_citation_ids(answer), ("EV002", "EV001"))
        citations = collector.cited_items(answer)
        self.assertEqual([item["evidence_id"] for item in citations], ["EV002", "EV001"])
        self.assertEqual([item["number"] for item in citations], [1, 2])
        self.assertEqual(collector.validate(answer).cited_chapters, (58, 64))

        invalid = collector.validate("错误引用。[[cite:EV001]][[cite:EV999]]")
        self.assertFalse(invalid.valid)
        self.assertEqual(invalid.invalid_evidence_ids, ("EV999",))

    def test_legacy_footer_is_no_longer_a_valid_citation(self):
        collector = EvidenceCollector()
        report = collector.validate("结论。\n【证据：EV001】\n【引用章节：第58章】")
        self.assertFalse(report.valid)
        self.assertIn("[[cite:EVxxx]]", report.message)

    def test_allows_explicit_insufficient_answer_when_no_source_evidence_exists(self):
        collector = EvidenceCollector()
        collector.capture("read_entity", {"ref": "characters/张三.md"}, "来源章节：8\n张三是剑客。")
        self.assertTrue(collector.validate("现有信息不足以确认。").valid)


if __name__ == "__main__":
    unittest.main()
