"""The public sample must be reproducible without API keys or model calls."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.scoring import evaluate_results, load_cases, score_case
from src.novel.corpus import CorpusStore
from src.novel.evidence import EvidenceCollector
from src.preprocess import split_chapters


EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "mist_harbor"


class PublicExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with contextlib.redirect_stdout(io.StringIO()):
            cls.chapters = split_chapters(str(EXAMPLE_DIR / "mist_harbor.txt"))
        cls.cases = load_cases(EXAMPLE_DIR / "qa_cases.jsonl")
        cls.responses = [
            json.loads(line)
            for line in (EXAMPLE_DIR / "reference_responses.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip() and not line.startswith("#")
        ]

    def test_story_keeps_eight_real_chapters_without_line_fallback(self):
        self.assertEqual([chapter["chapter_no"] for chapter in self.chapters], list(range(1, 9)))
        self.assertEqual(self.chapters[0]["title"], "铜钥归港")
        self.assertEqual(self.chapters[-1]["title"], "新的保管人")
        for chapter in self.chapters:
            with self.subTest(chapter=chapter["chapter_no"]):
                self.assertTrue(chapter["text"].strip())
                self.assertNotRegex(chapter["text"], r"(?m)^第[一二三四五六七八]章")

    def test_every_question_has_chapter_requirements_and_coverage(self):
        self.assertEqual(len(self.cases), 14)
        available = {chapter["chapter_no"] for chapter in self.chapters}
        for case in self.cases:
            with self.subTest(case=case.case_id):
                self.assertEqual(case.novel_id, "mist_harbor")
                self.assertTrue(case.required_any)
                self.assertTrue(case.expected_chapters)
                self.assertTrue(set(case.expected_chapters) <= available)
                self.assertTrue(case.require_valid_citations)
                self.assertGreaterEqual(case.min_evidence, 1)
        categories = {case.notes.split("：", 1)[0] for case in self.cases}
        self.assertEqual(categories, {"事实", "别名", "部件", "反转", "枚举", "跨章", "时间变化", "无法确认"})

    def test_reference_answers_are_explicitly_authored_and_pass_offline_scoring(self):
        ids = [response["case_id"] for response in self.responses]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {case.case_id for case in self.cases})
        for response in self.responses:
            with self.subTest(case=response["case_id"]):
                self.assertEqual(
                    response["provenance"],
                    {"kind": "hand_authored_reference", "live_model_call": False},
                )
                for field in ("model", "usage", "run_id", "latency", "elapsed_seconds", "cost"):
                    self.assertNotIn(field, response)
        report = evaluate_results(self.cases, {response["case_id"]: response for response in self.responses})
        self.assertEqual(report["summary"]["passed"], len(self.cases))
        self.assertEqual(report["summary"]["pass_rate"], 1.0)

    def test_reference_evidence_rebuilds_from_original_chapters(self):
        with tempfile.TemporaryDirectory() as temporary:
            corpus = CorpusStore(temporary)
            corpus.save_chapters(self.chapters)
            for response in self.responses:
                with self.subTest(case=response["case_id"]):
                    collector = EvidenceCollector(source_reader=corpus.read_chapter)
                    for item in response["evidence"]:
                        self.assertEqual(item["tool"], "search_source")
                        self.assertEqual(item["kind"], "primary_source")
                        arguments = item["arguments"]
                        result = corpus.search(**arguments)
                        matches = json.loads(result)["matches"]
                        self.assertEqual(len(matches), 1)
                        self.assertEqual(item["chapters"], [matches[0]["chapter"]])
                        collector.capture("search_source", arguments, result)
                    actual_report = collector.validate(response["reply"])
                    self.assertTrue(actual_report.valid, actual_report.message)
                    self.assertEqual(actual_report.to_dict(), response["citation_validation"])
                    self.assertEqual(list(actual_report.cited_chapters), response["sources"])
                    actual_evidence = collector.cited_items(response["reply"])
                    self.assertEqual(actual_evidence, response["evidence"])
                    self.assertEqual(actual_evidence, response["citations"])

    def test_questions_reject_missing_facts_chapters_and_invalid_citations(self):
        by_id = {response["case_id"]: response for response in self.responses}
        for case in self.cases:
            with self.subTest(case=case.case_id):
                reference = by_id[case.case_id]
                missing_fact = dict(reference, reply="无可核验答案。")
                self.assertFalse(score_case(case, missing_fact)["checks"]["required_facts"])
                missing_chapter = dict(reference, sources=[])
                self.assertFalse(score_case(case, missing_chapter)["checks"]["expected_chapters"])
                invalid_citation = dict(reference, citation_validation={"valid": False})
                self.assertFalse(score_case(case, invalid_citation)["checks"]["citation_valid"])


if __name__ == "__main__":
    unittest.main()
