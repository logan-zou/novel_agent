"""并发启动独立的单本构建进程，保留原有断点续跑语义。"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from src.config import DEFAULT_MIN_WINDOW_CHARS, DEFAULT_WINDOW_CHARS, KNOWLEDGE_BASE_DIR


def collect_inputs(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    ids: dict[str, Path] = {}
    for value in inputs:
        path = Path(value).expanduser().resolve()
        candidates = sorted(path.iterdir()) if path.is_dir() else [path]
        for candidate in candidates:
            if path.is_dir() and (not candidate.is_file() or candidate.suffix.lower() != ".txt"):
                continue
            candidate = candidate.resolve()
            if not candidate.is_file() or candidate.suffix.lower() != ".txt":
                raise ValueError(f"不是有效的 TXT 文件：{candidate}")
            if candidate in seen:
                continue
            novel_id = candidate.stem
            if novel_id in {"", ".", ".."}:
                raise ValueError(f"无法作为知识库标识符的文件名：{candidate.name}")
            # 大小写不敏感的文件系统上也不能写入同一知识库。
            key = novel_id.casefold()
            if key in ids:
                raise ValueError(f"文件名重复，会写入同一知识库：{ids[key]} 和 {candidate}")
            ids[key] = candidate
            seen.add(candidate)
            files.append(candidate)
    if not files:
        raise ValueError("没有找到 TXT 小说文件")
    return files


def format_progress(status: dict) -> str:
    parts = [status.get("stage", "启动中")]
    total = status.get("total", 0)
    if total:
        completed = status.get("completed", 0)
        parts.append(f"已完成 {completed}/{total} 批（{completed / total:.1%}）")
    if status.get("batch"):
        parts.append(f"当前第 {status['batch']} 批 · 第 {status['chapters']} 章")
    return " | ".join(parts)


async def watch_progress(process, status_path: Path, novel_id: str, started: float,
                         interval: float) -> None:
    previous = None
    last_print = started
    while True:
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            status = {"stage": "启动中 / 等待构建状态"}
        now = time.monotonic()
        if status != previous or now - last_print >= interval:
            elapsed = int(now - started)
            print(f"[进度][{novel_id}] {format_progress(status)} | 已运行 {elapsed // 60}m{elapsed % 60:02d}s", flush=True)
            previous, last_print = status, now
        if process.returncode is not None:
            return
        await asyncio.sleep(min(1, interval))


async def run_jobs(jobs: list[dict], concurrency: int, log_dir: Path,
                   progress_interval: float = 30) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)
    finished = 0
    failures = 0

    async def run_one(job: dict) -> dict:
        nonlocal finished, failures
        async with semaphore:
            novel_id = job["novel_id"]
            log_path = log_dir / f"{novel_id}.log"
            started = time.monotonic()
            result = {"novel_id": novel_id, "source_file": job["source_file"], "log": str(log_path)}
            print(f"[开始] {novel_id} → {log_path}", flush=True)
            process = None
            watcher = None
            try:
                with log_path.open("w", encoding="utf-8") as log:
                    process = await asyncio.create_subprocess_exec(
                        *job["command"], stdout=log, stderr=asyncio.subprocess.STDOUT,
                        env={**os.environ, "NOVEL_BUILD_STATUS_FILE": str(log_dir / f"{novel_id}.status.json")},
                    )
                    watcher = asyncio.create_task(watch_progress(
                        process, log_dir / f"{novel_id}.status.json", novel_id, started, progress_interval,
                    ))
                    code = await process.wait()
                    await watcher
                result.update(returncode=code, status="succeeded" if code == 0 else "failed")
            except asyncio.CancelledError:
                if process is not None and process.returncode is None:
                    try:
                        process.terminate()
                    except ProcessLookupError:
                        pass
                    try:
                        await asyncio.wait_for(process.wait(), timeout=5)
                    except asyncio.TimeoutError:
                        try:
                            process.kill()
                        except ProcessLookupError:
                            pass
                        await process.wait()
                raise
            except OSError as error:
                result.update(returncode=None, status="failed", error=str(error))
            finally:
                if watcher is not None and not watcher.done():
                    watcher.cancel()
                    await asyncio.gather(watcher, return_exceptions=True)
            result["elapsed_seconds"] = round(time.monotonic() - started, 2)
            label = "本次运行成功" if result["status"] == "succeeded" else "失败"
            print(f"[{label}] {novel_id}（{result['elapsed_seconds']} 秒）", flush=True)
            finished += 1
            failures += result["status"] == "failed"
            if result["status"] == "failed":
                print(f"[失败详情] {novel_id}：{result.get('error', '请查看日志')} → {log_path}", flush=True)
            print(f"[总进度] 已结束 {finished}/{len(jobs)} 本，成功 {finished - failures}，失败 {failures}", flush=True)
            return result

    tasks = [asyncio.create_task(run_one(job)) for job in jobs]
    try:
        return await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="novel-agent build-many", description="并发构建多本小说知识库")
    parser.add_argument("inputs", nargs="+", help="多个 TXT 文件或目录（仅扫描目录第一层）")
    parser.add_argument("-j", "--workers", type=int, default=2, help="同时构建的小说数（默认 2）")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW_CHARS)
    parser.add_argument("--min-window", type=int, default=DEFAULT_MIN_WINDOW_CHARS)
    parser.add_argument("--max-batches", type=int, default=0, help="每本本次最多处理的批次数；0 不限制")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=30, help="长批次进度提醒间隔秒数（默认 30）；状态变化约 1 秒内显示")
    parser.add_argument("--log-dir", type=Path, help="日志根目录，默认 knowledge_base/_build_logs")
    parser.add_argument("--dry-run", action="store_true", help="只展示任务，不调用模型或写入文件")
    args = parser.parse_args(argv)
    if args.workers <= 0:
        parser.error("--workers 必须大于 0")
    if args.progress_interval <= 0:
        parser.error("--progress-interval 必须大于 0")
    if args.window <= 0 or not 0 < args.min_window <= args.window:
        parser.error("--window 和 --min-window 必须大于 0，且 min-window 不能超过 window")
    if args.max_batches < 0:
        parser.error("--max-batches 不能小于 0")
    try:
        files = collect_inputs(args.inputs)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    jobs = []
    for path in files:
        command = [
            sys.executable, "-u", "-m", "src.cli", "build", str(path), path.stem,
            "--window", str(args.window), "--min-window", str(args.min_window),
            "--max-batches", str(args.max_batches),
        ]
        if args.debug:
            command.append("--debug")
        jobs.append({"novel_id": path.stem, "source_file": str(path), "command": command})
    print(f"共 {len(jobs)} 本小说，最多 {args.workers} 本同时处理。", flush=True)
    if args.dry_run:
        for job in jobs:
            print(f"{job['source_file']} → {Path(KNOWLEDGE_BASE_DIR) / job['novel_id']}")
        return

    from src.cli import _require_runtime_config
    from src.config import EXTRACTION_MODEL

    _require_runtime_config(EXTRACTION_MODEL, "EXTRACTION_MODEL")
    root = (args.log_dir or Path(KNOWLEDGE_BASE_DIR) / "_build_logs").expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    log_dir = Path(tempfile.mkdtemp(prefix=time.strftime("%Y%m%d-%H%M%S-"), dir=root))
    try:
        results = asyncio.run(run_jobs(jobs, args.workers, log_dir, args.progress_interval))
    except KeyboardInterrupt:
        print(f"\n已停止构建。重新运行相同命令可续跑；日志：{log_dir}", file=sys.stderr)
        raise SystemExit(130)
    summary = log_dir / "summary.json"
    summary.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = sum(result["status"] == "failed" for result in results)
    print(f"本次运行结束：成功 {len(results) - failed}，失败 {failed}。汇总：{summary}")
    if args.max_batches:
        print("已设置每本批次上限；运行成功不代表整本构建完成，重新运行可继续。")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
