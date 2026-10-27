"""
构建流程编排：预处理 → 逐批抽取 → 进度管理 → 终端展示。
由 build.sh 调用。
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from src.config import DEFAULT_MIN_WINDOW_CHARS, DEFAULT_WINDOW_CHARS, KNOWLEDGE_BASE_DIR
from src.build_status import report_status
from src.extraction_agent import run_extraction_agent
from src.novel.corpus import CorpusStore
from src.preprocess import make_batches, split_chapters
from src.harness.runtime import AgentLimitError
from src.usage_tracker import accumulate_usage, empty_usage, query, summarize


def _format_usage(usage: dict) -> str:
    """把 token 用量格式化为包含缓存拆分的终端摘要。"""
    parts = [
        f"非缓存读（估算） {usage.get('uncached_prompt_tokens', 0):,}",
        f"缓存读（估算） {usage.get('cached_prompt_tokens', 0):,}",
    ]
    unknown = usage.get("unknown_cache_prompt_tokens", 0)
    if unknown:
        parts.append(f"缓存归属未知 {unknown:,}")
    parts.append(f"输出 {usage.get('completion_tokens', 0):,}")
    return f"{usage.get('total_tokens', 0):,} tokens（{' / '.join(parts)}）"


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    txt_path = Path(args.txt_path)

    if not txt_path.exists():
        print(f"[错误] 文件不存在：{txt_path}", file=sys.stderr)
        sys.exit(1)

    novel_id = args.novel_id or txt_path.stem
    kb_dir = Path(KNOWLEDGE_BASE_DIR) / novel_id
    progress_path = kb_dir / "progress.json"

    # ── 读取或初始化进度 ─────────────────────────────────────────────────────
    progress, completed = _load_progress(progress_path)

    # ── 预处理 ───────────────────────────────────────────────────────────────
    print(f"\n📚 《{txt_path.stem}》—— 开始构建知识库")
    print(f"   小说文件：{txt_path}")
    print(f"   知识库路径：{kb_dir}\n")

    print("📖 正在划分章节...", end="", flush=True)
    chapters = split_chapters(str(txt_path))
    print(f"\r📖 章节划分完成，共 {len(chapters)} 章")

    try:
        source_fingerprint = _validate_source_identity(kb_dir, chapters, progress)
        batches, batching = _resolve_batches(
            chapters,
            progress,
            completed,
            requested_start=args.window,
            requested_min=args.min_window,
        )
    except ValueError as error:
        print(f"[错误] 无法安全恢复批次：{error}", file=sys.stderr)
        sys.exit(1)

    is_really_complete = _is_progress_complete(progress, completed)
    if is_really_complete:
        if "source_fingerprint" not in progress:
            progress["source_fingerprint"] = source_fingerprint
            _save_progress(progress_path, progress)
        report_status("已完成，跳过", completed=len(completed), total=progress["total_batches"])
        print(f"✅ 《{novel_id}》的知识库已构建完成。")
        print("   如需重新构建，请使用新的 novel_id")
        return
    if progress.get("status") == "completed":
        print("⚠️  进度曾被标记为 completed，但批次数不完整，将继续构建。")

    # 原文身份与批次计划全部通过后，才允许写入章节或恢复进度。
    report_status("正在预处理")
    kb_dir.mkdir(parents=True, exist_ok=True)
    CorpusStore(kb_dir).save_chapters(chapters)
    total_batches = len(batches)
    total_chapters = len(chapters)
    if batching["strategy"] == "progressive":
        start_window = batching["start_window_chars"]
        min_window = batching["min_window_chars"]
        fixed_prefix = batching.get("fixed_prefix_batches", 0)
        suffix = f"；保留前 {fixed_prefix} 个旧批次，只重排未处理部分" if fixed_prefix else ""
        print(
            f"📦 渐进批次：{start_window:,} → {min_window:,} 字，"
            f"共 {total_batches} 个批次{suffix}\n"
        )
    else:
        print(f"📦 固定批次：{batching['start_window_chars']:,} 字，共 {total_batches} 个批次\n")

    # ── 保存初始进度 ─────────────────────────────────────────────────────────
    progress.update({
        "novel_id": novel_id,
        "source_file": str(txt_path),
        "total_chapters": total_chapters,
        "total_batches": total_batches,
        "window_chars": batching["start_window_chars"],
        "source_chars": sum(len(chapter["text"]) for chapter in chapters),
        "source_fingerprint": source_fingerprint,
        "batching": batching,
        "completed_batches": list(completed),
        "status": "in_progress",
    })
    _save_progress(progress_path, progress)

    # ── 逐批处理 ─────────────────────────────────────────────────────────────
    max_batches = args.max_batches
    if max_batches:
        print(f"🧪 测试模式：最多处理 {max_batches} 个批次\n")

    total_start = time.time()
    total_usage_used = empty_usage()  # 本次运行累计，构建结束时汇报
    processed_this_run = 0

    for batch_no, batch in enumerate(batches):
        ch_start = batch[0]["chapter_no"]
        ch_end = batch[-1]["chapter_no"]
        label = f"[批次 {batch_no + 1:02d}/{total_batches:02d}] 第 {ch_start}-{ch_end} 章"

        if batch_no in completed:
            print(f"⏭  {label}  已完成，跳过")
            continue
        if max_batches and processed_this_run >= max_batches:
            print(f"\n🧪 已达测试上限（本次处理 {max_batches} 批），提前结束。")
            break

        print(f"⏳ {label}  处理中...", end="", flush=True)
        batch_start = time.time()
        report_status("处理中", completed=len(completed), total=total_batches,
                      batch=batch_no + 1, chapters=f"{ch_start}-{ch_end}")

        try:
            batch_usage = _run_batch_with_recovery(
                batch,
                novel_id,
                debug=args.debug,
                batch_no=batch_no,
                progress=progress,
                progress_path=progress_path,
            )
        except Exception as e:
            report_status("批次失败", completed=len(completed), total=total_batches,
                          batch=batch_no + 1, chapters=f"{ch_start}-{ch_end}")
            print(f"\n[错误] {label} 处理失败：{e}", file=sys.stderr)
            print("进度已保存，修复问题后重新运行可从此处续跑。", file=sys.stderr)
            sys.exit(1)

        elapsed = time.time() - batch_start
        mins, secs = divmod(int(elapsed), 60)
        accumulate_usage(total_usage_used, batch_usage)
        print(f"\r✅ {label}  完成 (耗时 {mins}m{secs:02d}s, {_format_usage(batch_usage)})")

        completed.add(batch_no)
        processed_this_run += 1
        _clear_split_progress(progress, batch_no)
        progress["completed_batches"] = sorted(completed)
        _save_progress(progress_path, progress)

        report_status("批次完成", completed=len(completed), total=total_batches)

    # 测试上限或中途停止不等于整本完成。只有每一个批次都有完成记录，
    # 才允许把状态置为 completed。
    if completed != set(range(total_batches)):
        progress["status"] = "in_progress"
        progress["completed_batches"] = sorted(completed)
        _save_progress(progress_path, progress)
        print(f"\n⏸️  本次处理结束：已完成 {len(completed)}/{total_batches} 批，可继续断点续跑。")
        report_status("达到本次上限，可续跑", completed=len(completed), total=total_batches)
        return

    # ── 完成 ─────────────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    mins, secs = divmod(int(total_elapsed), 60)

    progress["status"] = "completed"
    _save_progress(progress_path, progress)
    report_status("整本完成", completed=len(completed), total=total_batches)

    # token 用量从持久化记录里查全量汇总，不只是本次运行——断点续跑时
    # 之前几次运行也应该算进这本小说知识库的总花费
    usage_summary = summarize(query(novel_id, agent="extraction"))
    total_usage_all_time = usage_summary["total"]

    print(f"\n🎉 知识库构建完成！")
    print(f"   共处理 {total_chapters} 章 / {total_batches} 批次，总耗时 {mins}m{secs:02d}s")
    print(f"   本次运行消耗：{_format_usage(total_usage_used)}")
    print(f"   累计（含历次运行）：{_format_usage(total_usage_all_time)}")
    print(f"   知识库路径：{kb_dir}")
    print(f"   查看用量明细：python -m src.usage_tracker {novel_id} --agent extraction --detail")


# ── 进度文件读写 ─────────────────────────────────────────────────────────────

def _source_fingerprint(chapters: list[dict]) -> dict:
    """只对解析后的章号、标题和正文做指纹，不依赖文件路径或编码。"""
    payload = [
        [int(chapter["chapter_no"]), chapter["title"], chapter["text"]]
        for chapter in chapters
    ]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return {"version": 1, "sha256": hashlib.sha256(encoded).hexdigest()}


def _validate_source_identity(kb_dir: Path, chapters: list[dict], progress: dict) -> dict:
    """恢复前只读核验输入及证据原文；旧库必须能逐章证明同一来源。"""
    if not chapters:
        raise ValueError("原文没有可构建的章节")
    fingerprint = _source_fingerprint(chapters)
    if "source_fingerprint" in progress and progress["source_fingerprint"] != fingerprint:
        raise ValueError("原文章号、标题或正文已变化；请恢复原文件或使用新的 novel_id")

    if progress or (kb_dir.exists() and any(kb_dir.iterdir())):
        try:
            CorpusStore(kb_dir).validate_chapters(chapters)
        except (OSError, UnicodeError, ValueError) as error:
            raise ValueError(
                f"无法确认当前原文与已保存章节完全一致：{error}；"
                "请恢复原文件及章节副本，或使用新的 novel_id"
            ) from error
    return fingerprint


def _load_progress(progress_path: Path) -> tuple[dict, set]:
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        completed = set(progress.get("completed_batches", []))
        if completed:
            print(f"🔄 检测到未完成的构建进度，将从断点续跑（已完成 {len(completed)} 批）")
    else:
        progress = {}
        completed = set()
    return progress, completed


def _save_progress(progress_path: Path, progress: dict) -> None:
    progress_path.write_text(
        json.dumps(progress, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _is_progress_complete(progress: dict, completed: set[int]) -> bool:
    recorded_total = progress.get("total_batches")
    return (
        progress.get("status") == "completed"
        and isinstance(recorded_total, int)
        and recorded_total > 0
        and len(completed) == recorded_total
        and completed == set(range(recorded_total))
    )


def _run_batch_with_recovery(
    batch: list[dict],
    novel_id: str,
    *,
    debug: bool,
    batch_no: int,
    progress: dict,
    progress_path: Path,
) -> dict:
    """达到 Agent 限制时按章节递归拆分，并持久化已完成的子批次。"""
    splits = progress.get("batch_splits")
    if not isinstance(splits, dict):
        splits = {}
    completed_parts = set(progress.get("completed_batch_parts") or [])

    def save_split_state() -> None:
        progress["batch_splits"] = splits
        progress["completed_batch_parts"] = sorted(completed_parts)
        _save_progress(progress_path, progress)

    def process(part: list[dict], part_key: str) -> dict:
        total = empty_usage()
        saved_parts = splits.get(part_key)
        if saved_parts is None:
            try:
                return run_extraction_agent(part, novel_id, debug=debug)
            except AgentLimitError as error:
                accumulate_usage(total, error.usage)
                child_parts = _split_batch(part)
                if child_parts is None:
                    raise
                saved_parts = [
                    [chapter["chapter_no"] for chapter in child]
                    for child in child_parts
                ]
                splits[part_key] = saved_parts
                save_split_state()
                left, right = child_parts
                print(f"\n    ↘️  {error}")
                print(
                    "    自动拆分为 "
                    f"第 {left[0]['chapter_no']}-{left[-1]['chapter_no']} 章 / "
                    f"第 {right[0]['chapter_no']}-{right[-1]['chapter_no']} 章后继续"
                )
        else:
            child_parts = _restore_split_parts(part, saved_parts)

        for index, child in enumerate(child_parts):
            child_key = f"{part_key}.{index}"
            if child_key in completed_parts:
                continue
            child_usage = process(child, child_key)
            accumulate_usage(total, child_usage)
            completed_parts.add(child_key)
            save_split_state()
        return total

    return process(batch, str(batch_no))


def _split_batch(batch: list[dict]) -> tuple[list[dict], list[dict]] | None:
    if len(batch) < 2:
        return None
    lengths = [len(chapter["text"]) for chapter in batch]
    total = sum(lengths)
    prefix = 0
    best_index = 1
    best_distance = float("inf")
    for index in range(1, len(batch)):
        prefix += lengths[index - 1]
        distance = abs(total - prefix * 2)
        if distance < best_distance:
            best_index = index
            best_distance = distance
    return batch[:best_index], batch[best_index:]


def _restore_split_parts(batch: list[dict], saved_parts) -> list[list[dict]]:
    if not isinstance(saved_parts, list) or len(saved_parts) < 2:
        raise ValueError("已保存的子批次计划损坏")
    by_number = {chapter["chapter_no"]: chapter for chapter in batch}
    restored = []
    flattened = []
    for numbers in saved_parts:
        if not isinstance(numbers, list) or not numbers:
            raise ValueError("已保存的子批次计划损坏")
        try:
            child = [by_number[int(number)] for number in numbers]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("子批次计划与当前原文不一致") from error
        restored.append(child)
        flattened.extend(int(number) for number in numbers)
    expected = [chapter["chapter_no"] for chapter in batch]
    if flattened != expected:
        raise ValueError("子批次计划未完整、按序覆盖当前批次")
    return restored


def _clear_split_progress(progress: dict, batch_no: int) -> None:
    prefix = str(batch_no)
    splits = progress.get("batch_splits")
    if isinstance(splits, dict):
        progress["batch_splits"] = {
            key: value
            for key, value in splits.items()
            if key != prefix and not key.startswith(prefix + ".")
        }
        if not progress["batch_splits"]:
            progress.pop("batch_splits")
    parts = progress.get("completed_batch_parts")
    if isinstance(parts, list):
        progress["completed_batch_parts"] = [
            key for key in parts if key != prefix and not key.startswith(prefix + ".")
        ]
        if not progress["completed_batch_parts"]:
            progress.pop("completed_batch_parts")


def _resolve_batches(
    chapters: list[dict],
    progress: dict,
    completed: set[int],
    *,
    requested_start: int,
    requested_min: int,
) -> tuple[list[list[dict]], dict]:
    """生成可重复的批次，并把旧固定窗口进度安全迁移到渐进窗口。"""
    total_chars = sum(len(chapter["text"]) for chapter in chapters)
    recorded_chapters = progress.get("total_chapters")
    if isinstance(recorded_chapters, int) and recorded_chapters != len(chapters):
        raise ValueError(
            f"原进度记录 {recorded_chapters} 章，当前原文解析为 {len(chapters)} 章；"
            "为避免跳章，请恢复原文件或使用新的 novel_id"
        )
    recorded_chars = progress.get("source_chars")
    if isinstance(recorded_chars, int) and recorded_chars != total_chars:
        raise ValueError("原文内容长度已变化；为避免批次错位，请恢复原文件或使用新的 novel_id")

    saved = progress.get("batching")
    if isinstance(saved, dict):
        batching = dict(saved)
        return _batches_from_config(chapters, batching), batching

    # 旧进度只有固定 window_chars。已完成部分按旧规则原样保留，仅将尚未处理
    # 的连续尾部改为渐进窗口，因此当前失败批次可以直接缩小后重试。
    old_start = int(progress.get("window_chars", requested_start))
    legacy_batches = make_batches(chapters, old_start)
    recorded_total = progress.get("total_batches")
    if progress and isinstance(recorded_total, int) and recorded_total != len(legacy_batches):
        raise ValueError(
            f"旧进度记录 {recorded_total} 批，按当前原文只能重建出 {len(legacy_batches)} 批"
        )

    prefix_count = len(completed)
    is_prefix = completed == set(range(prefix_count))
    if completed and not is_prefix:
        batching = {
            "strategy": "fixed",
            "start_window_chars": old_start,
            "min_window_chars": old_start,
            "fixed_prefix_batches": 0,
        }
        return legacy_batches, batching

    batching = {
        "strategy": "progressive",
        "start_window_chars": old_start,
        "min_window_chars": min(requested_min, old_start),
        "fixed_prefix_batches": prefix_count,
    }
    return _batches_from_config(chapters, batching), batching


def _batches_from_config(chapters: list[dict], batching: dict) -> list[list[dict]]:
    start = int(batching["start_window_chars"])
    minimum = int(batching.get("min_window_chars", start))
    if batching.get("strategy") == "fixed":
        return make_batches(chapters, start)
    if batching.get("strategy") != "progressive":
        raise ValueError(f"未知批次策略：{batching.get('strategy')}")

    prefix_count = int(batching.get("fixed_prefix_batches", 0))
    prefix: list[list[dict]] = []
    prefix_chapter_count = 0
    prefix_chars = 0
    if prefix_count:
        legacy_batches = make_batches(chapters, start)
        if prefix_count > len(legacy_batches):
            raise ValueError("固定前缀批次数超过按旧规则生成的总批次数")
        prefix = legacy_batches[:prefix_count]
        prefix_chapter_count = sum(len(batch) for batch in prefix)
        prefix_chars = sum(len(chapter["text"]) for batch in prefix for chapter in batch)

    tail = make_batches(
        chapters[prefix_chapter_count:],
        start,
        minimum,
        processed_chars=prefix_chars,
        total_chars=sum(len(chapter["text"]) for chapter in chapters),
    )
    return prefix + tail


# ── CLI 参数 ─────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="novel-agent build",
        description="小说知识库离线构建工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python src/build.py ./novels/斗破苍穹.txt
  python src/build.py ./novels/斗破苍穹.txt doupo --window 30000 --min-window 20000
        """,
    )
    parser.add_argument("txt_path", help="小说 txt 文件路径")
    parser.add_argument("novel_id", nargs="?", help="知识库标识符（默认取文件名去后缀）")
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW_CHARS,
        help=f"首批字符数上限（默认 {DEFAULT_WINDOW_CHARS}）",
    )
    parser.add_argument(
        "--min-window",
        type=int,
        default=DEFAULT_MIN_WINDOW_CHARS,
        dest="min_window",
        help=(
            f"随进度下降后的最小字符数（默认 {DEFAULT_MIN_WINDOW_CHARS}；"
            "设为与 --window 相同可使用固定批次）"
        ),
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        dest="max_batches",
        help="最多处理的批次数（0 表示不限制，用于测试）",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="调试模式：打印 Agent 每轮的完整输出",
    )
    args = parser.parse_args(argv)
    if args.window <= 0:
        parser.error("--window 必须大于 0")
    if args.min_window <= 0 or args.min_window > args.window:
        parser.error("--min-window 必须大于 0 且不能超过 --window")
    return args


if __name__ == "__main__":
    main()
