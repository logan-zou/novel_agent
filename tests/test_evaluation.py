import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.scoring import EvalCase, evaluate_results, load_cases, score_case


class EvaluationTests(unittest.TestCase):
    def test_scores_required_forbidden_chapters_and_evidence(self):
        case = EvalCase(
            case_id="case-1",
            novel_id="novel",
            question="question",
            required_any=(("烛九阴",), ("左眼", "眼睛")),
            forbidden_terms=("独立个体",),
            expected_chapters=(58,),
        )
        result = {
            "reply": "时空之眼是烛九阴的左眼。",
            "sources": [58],
            "evidence": [{"evidence_id": "EV001"}],
            "citation_validation": {"valid": True},
        }

        self.assertTrue(score_case(case, result)["passed"])

        bad = dict(result)
        bad["reply"] = "两者是独立个体。"
        self.assertFalse(score_case(case, bad)["passed"])

    def test_loads_jsonl_and_reports_missing_results(self):
        payload = {
            "case_id": "case-1",
            "novel_id": "novel",
            "question": "question",
            "required_any": [["答案"]],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
            cases = load_cases(path)

        report = evaluate_results(cases, {})
        self.assertEqual(report["summary"]["pass_rate"], 0.0)
        self.assertEqual(report["cases"][0]["error"], "缺少评测结果")


if __name__ == "__main__":
    unittest.main()
