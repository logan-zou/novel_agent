import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src import evaluate
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


class EvaluationCliTests(unittest.TestCase):
    def _write_jsonl(self, path, values):
        path.write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values), encoding="utf-8")

    def _case(self, case_id="case-1", novel_id="novel", **extra):
        return {"case_id": case_id, "novel_id": novel_id, "question": "谁守灯？", "required_any": [["林舟"]], **extra}

    def _result(self, case_id="case-1", reply="林舟守灯。"):
        return {"case_id": case_id, "reply": reply, "sources": [1], "evidence": [{"evidence_id": "EV001"}], "citation_validation": {"valid": True}}

    def _run_cli(self, *args):
        env = dict(os.environ, LLM_API_KEY="", LLM_MODEL="", LANGFUSE_PUBLIC_KEY="", LANGFUSE_SECRET_KEY="")
        return subprocess.run([sys.executable, "-m", "src.cli", "evaluate", *map(str, args)], cwd=Path(__file__).resolve().parent.parent, env=env, capture_output=True, text=True)

    def _write_knowledge(self, root, novel_id="novel", chapter="林舟在雾港守灯。"):
        base = root / novel_id
        (base / "characters").mkdir(parents=True)
        (base / "characters/林舟.md").write_text("# 林舟\n\n雾港守灯人。\n", encoding="utf-8")
        self._write_jsonl(base / "catalog.jsonl", [{"kind": "character", "name": "林舟", "ref": "characters/林舟.md", "aliases": [], "summary": "雾港守灯人", "source_chapters": [1]}])
        (base / "chapters").mkdir()
        (base / "chapters/1.txt").write_text(chapter, encoding="utf-8")
        return base

    def test_failing_cli_returns_one_after_printing_and_saving_all_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, responses, output = root / "cases.jsonl", root / "responses.jsonl", root / "report.json"
            self._write_jsonl(dataset, [self._case(), self._case("case-2")])
            self._write_jsonl(responses, [self._result(reply="王五守灯。")])

            result = self._run_cli("--dataset", dataset, "--responses", responses, "--output", output)

            self.assertEqual(result.returncode, 1, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)
            self.assertEqual(report["summary"]["total"], 2)
            self.assertEqual(report["summary"]["passed"], 0)
            self.assertFalse(report["cases"][0]["checks"]["required_facts"])
            self.assertEqual(report["cases"][1]["error"], "缺少评测结果")

    def test_passing_cli_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, responses = root / "cases.jsonl", root / "responses.jsonl"
            self._write_jsonl(dataset, [self._case()])
            self._write_jsonl(responses, [self._result()])
            result = self._run_cli("--dataset", dataset, "--responses", responses)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["summary"]["pass_rate"], 1.0)

    def test_default_validation_does_not_run_or_load_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "cases.jsonl"
            self._write_jsonl(dataset, [self._case()])
            with patch.object(evaluate, "_run_live") as live, patch.object(evaluate, "_load_results") as load, contextlib.redirect_stdout(io.StringIO()) as stdout:
                evaluate.main(["--dataset", str(dataset)])
            self.assertTrue(json.loads(stdout.getvalue())["valid"])
            live.assert_not_called()
            load.assert_not_called()

    def test_live_preflights_every_selected_novel_before_first_qa_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_knowledge(root, "ready")
            dataset = root / "cases.jsonl"
            self._write_jsonl(dataset, [self._case(novel_id="ready"), self._case("case-2", "missing"), self._case("case-3", "also-missing")])
            qa = Mock()
            with patch("src.config.KNOWLEDGE_BASE_DIR", str(root)), patch.dict(sys.modules, {"src.qa_agent": SimpleNamespace(run_qa_agent=qa)}), contextlib.redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as error:
                    evaluate.main(["--dataset", str(dataset), "--live"])
            self.assertEqual(error.exception.code, 2)
            self.assertIn("missing", stderr.getvalue())
            self.assertIn("also-missing", stderr.getvalue())
            self.assertIn("尚未发起任何模型请求", stderr.getvalue())
            qa.assert_not_called()

    def test_live_rejects_missing_empty_and_unreadable_resources(self):
        mutations = {
            "empty-knowledge": lambda base: (base / "catalog.jsonl").write_text("", encoding="utf-8"),
            "missing-entity": lambda base: (base / "characters/林舟.md").unlink(),
            "empty-entity": lambda base: (base / "characters/林舟.md").write_text(" \n", encoding="utf-8"),
            "no-chapters": lambda base: (base / "chapters/1.txt").unlink(),
            "empty-chapter": lambda base: (base / "chapters/1.txt").write_text(" \n", encoding="utf-8"),
            "unreadable-chapter": lambda base: (base / "chapters/1.txt").write_bytes(b"\xff"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                base = self._write_knowledge(root)
                mutate(base)
                qa = Mock()
                with patch("src.config.KNOWLEDGE_BASE_DIR", str(root)), patch.dict(sys.modules, {"src.qa_agent": SimpleNamespace(run_qa_agent=qa)}), contextlib.redirect_stderr(io.StringIO()) as stderr:
                    with self.assertRaises(SystemExit) as error:
                        evaluate._run_live([EvalCase.from_dict(self._case())])
                self.assertEqual(error.exception.code, 2)
                self.assertIn("novel", stderr.getvalue())
                qa.assert_not_called()

    def test_live_rejects_missing_expected_chapter_before_qa(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_knowledge(root)
            qa = Mock()
            with patch("src.config.KNOWLEDGE_BASE_DIR", str(root)), patch.dict(sys.modules, {"src.qa_agent": SimpleNamespace(run_qa_agent=qa)}), contextlib.redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as error:
                    evaluate._run_live([EvalCase.from_dict(self._case(expected_chapters=[2]))])
            self.assertEqual(error.exception.code, 2)
            self.assertIn("章节原文缺失或为空", stderr.getvalue())
            self.assertIn("2", stderr.getvalue())
            qa.assert_not_called()

    def test_live_case_filter_preflights_only_the_selected_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_knowledge(root)
            dataset = root / "cases.jsonl"
            self._write_jsonl(dataset, [self._case(), self._case("case-2", "missing")])
            qa = Mock(return_value=self._result())
            with patch("src.config.KNOWLEDGE_BASE_DIR", str(root)), patch.dict(sys.modules, {"src.qa_agent": SimpleNamespace(run_qa_agent=qa)}), contextlib.redirect_stdout(io.StringIO()) as stdout, contextlib.redirect_stderr(io.StringIO()):
                evaluate.main(["--dataset", str(dataset), "--live", "--case-id", "case-1"])
            self.assertEqual(json.loads(stdout.getvalue())["summary"]["passed"], 1)
            qa.assert_called_once_with("谁守灯？", "novel")

    def test_live_accepts_legacy_markdown_with_readable_link_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = self._write_knowledge(root)
            (base / "catalog.jsonl").unlink()
            (base / "characters/_index.md").write_text("- [[characters/林舟|守灯人]]\n", encoding="utf-8")
            qa = Mock(return_value=self._result())
            with patch("src.config.KNOWLEDGE_BASE_DIR", str(root)), patch.dict(sys.modules, {"src.qa_agent": SimpleNamespace(run_qa_agent=qa)}), contextlib.redirect_stderr(io.StringIO()):
                evaluate._run_live([EvalCase.from_dict(self._case())])
            qa.assert_called_once()

    def test_empty_dataset_returns_two_with_actionable_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "cases.jsonl"
            dataset.write_text("", encoding="utf-8")
            result = self._run_cli("--dataset", dataset)
            self.assertEqual(result.returncode, 2)
            self.assertIn("没有可运行的 case", result.stderr)


if __name__ == "__main__":
    unittest.main()
