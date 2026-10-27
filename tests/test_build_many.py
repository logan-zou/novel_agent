import asyncio
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.build_many import collect_inputs, format_progress, main, run_jobs


class BuildManyTests(unittest.TestCase):
    def test_progress_with_real_child_keeps_details_out_of_terminal(self):
        script = (
            "import json,os,time; from pathlib import Path; "
            "p=Path(os.environ['NOVEL_BUILD_STATUS_FILE']); "
            "p.write_text(json.dumps({'stage':'处理中','completed':2,'total':5,'batch':3,'chapters':'11-15'})); "
            "print('详细模型内容，仅写日志',flush=True); time.sleep(0.15); "
            "p.write_text(json.dumps({'stage':'达到本次上限，可续跑','completed':3,'total':5}))"
        )
        job = {"novel_id": "测试", "source_file": "book.txt", "command": [sys.executable, "-c", script]}
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(output):
            results = asyncio.run(run_jobs([job], 1, Path(tmp), 0.02))
            log = Path(results[0]["log"]).read_text()
        self.assertIn("详细模型内容，仅写日志", log)
        self.assertNotIn("详细模型内容", output.getvalue())
        self.assertIn("已完成 2/5 批（40.0%）", output.getvalue())
        self.assertIn("第 11-15 章", output.getvalue())
        self.assertIn("达到本次上限，可续跑", output.getvalue())
        self.assertIn("已结束 1/1 本", output.getvalue())

    def test_completed_progress_does_not_require_current_batch(self):
        self.assertEqual(format_progress({"stage": "已完成，跳过", "completed": 5, "total": 5}),
                         "已完成，跳过 | 已完成 5/5 批（100.0%）")

    def test_collect_deduplicates_and_rejects_conflicting_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            book = root / "测试.txt"
            book.touch()
            (root / "ignore.md").touch()
            self.assertEqual(collect_inputs([tmp, str(book)]), [book.resolve()])
            other = root / "nested"
            other.mkdir()
            duplicate = other / book.name
            duplicate.touch()
            with self.assertRaisesRegex(ValueError, "文件名重复"):
                collect_inputs([tmp, str(duplicate)])

    def test_dry_run_needs_no_model_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            book = root / "book.txt"
            book.touch()
            with patch("src.cli._require_runtime_config") as require, contextlib.redirect_stdout(io.StringIO()):
                main([str(book), "--dry-run", "--log-dir", str(root / "logs")])
            require.assert_not_called()
            self.assertFalse((root / "logs").exists())

    def test_real_processes_overlap_with_limit_and_failure_isolation(self):
        script = (
            "import json,sys,time; "
            "print(json.dumps({'start':time.monotonic()}),flush=True); "
            "time.sleep(0.2); "
            "print(json.dumps({'end':time.monotonic()}),flush=True); "
            "sys.exit(int(sys.argv[1]))"
        )
        jobs = [
            {"novel_id": str(i), "source_file": str(i),
             "command": [sys.executable, "-c", script, "1" if i == 0 else "0"]}
            for i in range(4)
        ]
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            results = asyncio.run(run_jobs(jobs, 2, Path(tmp)))
            events = []
            for result in results:
                lines = Path(result["log"]).read_text().splitlines()
                events.extend([(json.loads(lines[0])["start"], 1), (json.loads(lines[1])["end"], -1)])
        active = peak = 0
        for _, change in sorted(events):
            active += change
            peak = max(peak, active)
        self.assertEqual(peak, 2)
        self.assertEqual([r["status"] for r in results], ["failed", "succeeded", "succeeded", "succeeded"])

    def test_cancellation_reaps_running_child_and_does_not_start_queue(self):
        async def scenario(log_dir):
            real_create = asyncio.create_subprocess_exec
            processes = []
            started = asyncio.Event()

            async def create(*args, **kwargs):
                process = await real_create(*args, **kwargs)
                processes.append(process)
                started.set()
                return process

            jobs = [{"novel_id": str(i), "source_file": str(i),
                     "command": [sys.executable, "-c", "import time; time.sleep(30)"]}
                    for i in range(2)]
            with patch("src.build_many.asyncio.create_subprocess_exec", side_effect=create):
                task = asyncio.create_task(run_jobs(jobs, 1, log_dir))
                await asyncio.wait_for(started.wait(), 5)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            self.assertEqual(len(processes), 1)
            self.assertIsNotNone(processes[0].returncode)

        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            asyncio.run(scenario(Path(tmp)))


if __name__ == "__main__":
    unittest.main()
