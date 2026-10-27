import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.build import _source_fingerprint, main
from src.harness.usage import empty_usage
from src.novel.corpus import CorpusStore
from src.preprocess import split_chapters


class SourceIdentityTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source = self.root / "source.txt"
        self.chapters = [
            {"chapter_no": number, "title": f"相遇{number}", "text": f"张三拜访李四，记录第{number}次见闻。"}
            for number in range(1, 6)
        ]
        self.write_source(self.chapters)
        self.kb_root = self.root / "knowledge_base"
        self.kb = self.kb_root / "demo"
        self.progress_path = self.kb / "progress.json"
        self.output = io.StringIO()
        self.errors = io.StringIO()
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch("src.build.KNOWLEDGE_BASE_DIR", self.kb_root))
        self.extract = stack.enter_context(patch("src.build.run_extraction_agent", return_value=empty_usage()))
        self.status = stack.enter_context(patch("src.build.report_status"))
        stack.enter_context(patch("src.build.query", return_value=[]))
        stack.enter_context(contextlib.redirect_stdout(self.output))
        stack.enter_context(contextlib.redirect_stderr(self.errors))

    def write_source(self, chapters, *, path=None, encoding="utf-8", newline="\n"):
        contents = "\n".join(
            f"第{chapter['chapter_no']}章 {chapter['title']}\n{chapter['text']}"
            for chapter in chapters
        )
        (path or self.source).write_bytes(contents.replace("\n", newline).encode(encoding))

    def run_build(self, *extra, source=None):
        main([str(source or self.source), "demo", "--window", "25", "--min-window", "25", *extra])

    def progress(self):
        return json.loads(self.progress_path.read_text(encoding="utf-8"))

    def snapshot(self):
        return {
            path.relative_to(self.kb).as_posix(): path.read_bytes() if path.is_file() else None
            for path in self.kb.rglob("*")
        }

    def legacy(self):
        progress = self.progress()
        progress.pop("source_fingerprint")
        self.progress_path.write_text(json.dumps(progress, ensure_ascii=False), encoding="utf-8")
        return progress

    def assert_rejected_without_writes(self, *, source=None, source_error=True):
        before = self.snapshot()
        self.extract.reset_mock()
        self.status.reset_mock()
        with self.assertRaises(SystemExit) as raised:
            self.run_build(source=source)
        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(self.snapshot(), before)
        self.extract.assert_not_called()
        self.status.assert_not_called()
        if source_error:
            self.assertIn("新的 novel_id", self.errors.getvalue())

    def test_new_build_records_fingerprint_before_extraction_and_resumes(self):
        def inspect_source(batch, novel_id, debug=False):
            progress = self.progress()
            self.assertEqual(progress["source_fingerprint"], _source_fingerprint(self.chapters))
            CorpusStore(self.kb).validate_chapters(self.chapters)
            return empty_usage()

        self.extract.side_effect = inspect_source
        self.run_build("--max-batches", "1")
        first = self.progress()
        self.assertEqual(first["completed_batches"], [0])
        self.assertEqual(first["status"], "in_progress")
        self.run_build("--max-batches", "1")
        self.assertEqual(self.progress()["completed_batches"], [0, 1])
        self.assertEqual(self.progress()["source_fingerprint"], first["source_fingerprint"])
        self.assertEqual(self.extract.call_count, 2)

    def test_same_length_body_change_is_rejected_for_partial_and_completed_builds(self):
        for complete in (False, True):
            with self.subTest(complete=complete):
                self.write_source(self.chapters)
                self.run_build(*([] if complete else ["--max-batches", "1"]))
                changed = copy.deepcopy(self.chapters)
                changed[-1]["text"] = changed[-1]["text"].replace("李四", "王五")
                self.write_source(changed)
                self.assert_rejected_without_writes()

    def test_title_change_is_rejected_before_completed_skip(self):
        self.run_build()
        changed = copy.deepcopy(self.chapters)
        changed[0]["title"] = "别离1"
        self.write_source(changed)
        self.assert_rejected_without_writes()

    def test_chapter_number_change_is_rejected(self):
        self.run_build("--max-batches", "1")
        changed = copy.deepcopy(self.chapters)
        changed[-1]["chapter_no"] = 6
        self.write_source(changed)
        self.assert_rejected_without_writes()
        self.assertFalse((self.kb / "chapters" / "6.txt").exists())

    def test_moved_and_reencoded_identical_source_keeps_completed_build(self):
        self.run_build()
        before = self.snapshot()
        moved = self.root / "renamed.txt"
        self.write_source(self.chapters, path=moved, encoding="gb18030", newline="\r\n")
        self.extract.reset_mock()
        self.run_build(source=moved)
        self.extract.assert_not_called()
        self.assertEqual(self.snapshot(), before)
        self.assertIn("已构建完成", self.output.getvalue())

    def test_legacy_partial_progress_migrates_only_after_exact_comparison(self):
        self.run_build("--max-batches", "1")
        legacy = self.legacy()
        legacy.pop("batching")
        legacy.pop("source_chars")
        self.progress_path.write_text(json.dumps(legacy), encoding="utf-8")
        originals = {
            path.name: path.read_bytes() for path in (self.kb / "chapters").glob("*.txt")
        }
        self.run_build("--max-batches", "1")
        self.assertEqual(self.progress()["source_fingerprint"], _source_fingerprint(self.chapters))
        self.assertEqual(self.progress()["completed_batches"], [0, 1])
        self.assertEqual(
            {path.name: path.read_bytes() for path in (self.kb / "chapters").glob("*.txt")},
            originals,
        )

    def test_legacy_completed_build_migrates_without_model_calls(self):
        self.run_build()
        legacy = self.legacy()
        before = self.snapshot()
        self.extract.reset_mock()
        self.run_build()
        self.extract.assert_not_called()
        updated = self.progress()
        self.assertEqual(updated.pop("source_fingerprint"), _source_fingerprint(self.chapters))
        self.assertEqual(updated, legacy)
        after = self.snapshot()
        before.pop("progress.json")
        after.pop("progress.json")
        self.assertEqual(before, after)

    def test_legacy_same_length_body_and_title_changes_are_rejected(self):
        self.run_build("--max-batches", "1")
        self.legacy()
        for field, replacement in (("text", "王五拜访李四，记录第1次见闻。"), ("title", "别离1")):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.chapters)
                changed[0][field] = replacement
                self.assertEqual(len(changed[0][field]), len(self.chapters[0][field]))
                self.write_source(changed)
                self.assert_rejected_without_writes()

    def test_legacy_completed_progress_cannot_skip_changed_source(self):
        self.run_build()
        self.legacy()
        changed = copy.deepcopy(self.chapters)
        changed[-1]["text"] = changed[-1]["text"].replace("李四", "王五")
        self.write_source(changed)
        self.assert_rejected_without_writes()

    def test_legacy_chapter_number_change_cannot_create_new_evidence_files(self):
        self.run_build("--max-batches", "1")
        self.legacy()
        changed = copy.deepcopy(self.chapters)
        changed[-1]["chapter_no"] = 6
        self.write_source(changed)
        self.assert_rejected_without_writes()
        self.assertFalse((self.kb / "chapters" / "6.txt").exists())

    def test_missing_saved_chapter_rejects_both_legacy_and_fingerprinted_progress(self):
        self.run_build("--max-batches", "1")
        (self.kb / "chapters" / "5.txt").unlink()
        self.assert_rejected_without_writes()
        self.legacy()
        self.assert_rejected_without_writes()

    def test_modified_saved_evidence_is_rejected_even_when_fingerprint_matches(self):
        self.run_build()
        chapter = self.kb / "chapters" / "1.txt"
        chapter.write_text(chapter.read_text().replace("李四", "王五"), encoding="utf-8")
        self.assert_rejected_without_writes()

    def test_extra_saved_chapter_is_rejected(self):
        self.run_build("--max-batches", "1")
        (self.kb / "chapters" / "99.txt").write_text("其他版本的原文", encoding="utf-8")
        self.assert_rejected_without_writes()

    def test_legacy_knowledge_without_source_cannot_be_adopted(self):
        self.kb.mkdir(parents=True)
        (self.kb / "catalog.jsonl").write_text('{"name":"旧角色"}\n', encoding="utf-8")
        self.assert_rejected_without_writes()
        self.assertFalse(self.progress_path.exists())
        self.assertFalse((self.kb / "chapters").exists())

    def test_legacy_progress_without_source_is_not_trusted_even_when_completed(self):
        self.kb.mkdir(parents=True)
        self.progress_path.write_text(json.dumps({
            "status": "completed", "total_batches": 5, "completed_batches": list(range(5)),
            "total_chapters": 5, "source_chars": sum(len(c["text"]) for c in self.chapters),
        }), encoding="utf-8")
        self.assert_rejected_without_writes()

    def test_invalid_batch_plan_is_rejected_before_source_or_progress_writes(self):
        self.run_build("--max-batches", "1")
        self.legacy()
        progress = self.progress()
        progress["batching"]["strategy"] = "unknown"
        self.progress_path.write_text(json.dumps(progress), encoding="utf-8")
        self.assert_rejected_without_writes(source_error=False)

    def test_new_empty_source_does_not_create_knowledge_directory(self):
        self.source.write_text("", encoding="utf-8")
        with self.assertRaises(SystemExit) as raised:
            self.run_build()
        self.assertEqual(raised.exception.code, 1)
        self.assertFalse(self.kb.exists())
        self.extract.assert_not_called()

    def test_fingerprint_depends_only_on_parsed_identity(self):
        current = split_chapters(str(self.source))
        self.assertEqual(_source_fingerprint(current), _source_fingerprint(self.chapters))
        with_extras = [dict(chapter, unrelated="ignored") for chapter in self.chapters]
        self.assertEqual(_source_fingerprint(with_extras), _source_fingerprint(self.chapters))


if __name__ == "__main__":
    unittest.main()
