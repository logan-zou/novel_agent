import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.build import (
    _clear_split_progress,
    _is_progress_complete,
    _resolve_batches,
    _run_batch_with_recovery,
    main,
)
from src.harness.runtime import MaxIterationsExceeded
from src.harness.usage import empty_usage


class BuildProgressTests(unittest.TestCase):
    def test_requires_every_batch_before_completed(self):
        self.assertFalse(_is_progress_complete({"status": "completed", "total_batches": 3}, {0, 1}))
        self.assertFalse(_is_progress_complete({"status": "completed", "total_batches": 3}, {0, 1, 4}))
        self.assertTrue(_is_progress_complete({"status": "completed", "total_batches": 3}, {0, 1, 2}))

    def test_max_batches_counts_new_work_when_resuming(self):
        chapters = [
            {"chapter_no": number, "title": str(number), "text": f"正文 {number}"}
            for number in (1, 2, 3)
        ]
        calls = []

        def fake_extract(batch, novel_id, debug=False):
            calls.append(batch[0]["chapter_no"])
            return empty_usage()

        with tempfile.TemporaryDirectory() as tmp:
            novel = Path(tmp) / "novel.txt"
            novel.write_text("测试小说", encoding="utf-8")
            kb_root = Path(tmp) / "knowledge_base"
            argv = ["build.py", str(novel), "demo", "--max-batches", "1"]
            with (
                patch("src.build.KNOWLEDGE_BASE_DIR", kb_root),
                patch("src.build.split_chapters", return_value=chapters),
                patch("src.build.make_batches", return_value=[[item] for item in chapters]),
                patch("src.build.run_extraction_agent", side_effect=fake_extract),
                patch.object(sys, "argv", argv),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                main()
                main()

            progress = json.loads(
                (kb_root / "demo" / "progress.json").read_text(encoding="utf-8")
            )

        self.assertEqual(calls, [1, 2])
        self.assertEqual(progress["completed_batches"], [0, 1])
        self.assertEqual(progress["status"], "in_progress")

    def test_batch_limit_reports_partial_progress_to_launcher(self):
        chapters = [{"chapter_no": n, "title": str(n), "text": "正文"} for n in (1, 2)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            novel = root / "book.txt"
            novel.write_text("正文", encoding="utf-8")
            status = root / "status.json"
            with (
                patch.dict(os.environ, {"NOVEL_BUILD_STATUS_FILE": str(status)}),
                patch("src.build.KNOWLEDGE_BASE_DIR", root / "kb"),
                patch("src.build.split_chapters", return_value=chapters),
                patch("src.build.make_batches", return_value=[[c] for c in chapters]),
                patch("src.build.run_extraction_agent", return_value=empty_usage()),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                main([str(novel), "--max-batches", "1"])
            snapshot = json.loads(status.read_text())
        self.assertEqual(snapshot, {"stage": "达到本次上限，可续跑", "completed": 1, "total": 2})

    def test_old_progress_keeps_completed_prefix_and_replans_only_tail(self):
        chapters = [
            {"chapter_no": number, "title": str(number), "text": "x" * 40}
            for number in range(1, 11)
        ]
        progress = {
            "total_chapters": 10,
            "total_batches": 5,
            "window_chars": 100,
            "completed_batches": [0, 1],
            "status": "in_progress",
        }

        batches, batching = _resolve_batches(
            chapters,
            progress,
            {0, 1},
            requested_start=100,
            requested_min=50,
        )

        self.assertEqual([[c["chapter_no"] for c in batch] for batch in batches[:2]], [[1, 2], [3, 4]])
        self.assertEqual(batches[2][0]["chapter_no"], 5)
        self.assertEqual(batching["strategy"], "progressive")
        self.assertEqual(batching["fixed_prefix_batches"], 2)
        self.assertEqual(
            [chapter["chapter_no"] for batch in batches for chapter in batch],
            list(range(1, 11)),
        )

        resumed_progress = {
            **progress,
            "source_chars": 400,
            "batching": batching,
            "total_batches": len(batches),
        }
        resumed, resumed_batching = _resolve_batches(
            chapters,
            resumed_progress,
            {0, 1},
            requested_start=100,
            requested_min=80,
        )
        self.assertEqual(
            [[c["chapter_no"] for c in batch] for batch in resumed],
            [[c["chapter_no"] for c in batch] for batch in batches],
        )
        self.assertEqual(resumed_batching, batching)

    def test_limit_failure_splits_batch_and_persists_completed_parts(self):
        chapters = [
            {"chapter_no": number, "title": str(number), "text": "x" * 40}
            for number in range(1, 4)
        ]
        calls = []

        def usage(value):
            result = empty_usage()
            result["prompt_tokens"] = value
            result["uncached_prompt_tokens"] = value
            result["total_tokens"] = value
            return result

        def fake_extract(batch, novel_id, debug=False):
            calls.append([chapter["chapter_no"] for chapter in batch])
            if len(batch) > 1:
                raise MaxIterationsExceeded(
                    "达到硬迭代上限 60",
                    steps=60,
                    limit=60,
                    usage=usage(10),
                    tool_counts={"upsert_entities": 20},
                    successful_tool_calls=20,
                    failed_tool_calls=0,
                    repeated_tool_calls=0,
                )
            return usage(1)

        with tempfile.TemporaryDirectory() as tmp:
            progress_path = Path(tmp) / "progress.json"
            progress = {"completed_batches": [], "status": "in_progress"}
            with (
                patch("src.build.run_extraction_agent", side_effect=fake_extract),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = _run_batch_with_recovery(
                    chapters,
                    "demo",
                    debug=False,
                    batch_no=4,
                    progress=progress,
                    progress_path=progress_path,
                )

            self.assertEqual(calls, [[1, 2, 3], [1], [2, 3], [2], [3]])
            self.assertEqual(result["total_tokens"], 23)
            self.assertEqual(progress["batch_splits"]["4"], [[1], [2, 3]])
            self.assertIn("4.1.1", progress["completed_batch_parts"])

            _clear_split_progress(progress, 4)
            self.assertNotIn("batch_splits", progress)
            self.assertNotIn("completed_batch_parts", progress)


if __name__ == "__main__":
    unittest.main()
