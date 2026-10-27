"""
Token 用量记录：所有 LLM 调用统一记录到 knowledge_base/{novel_id}/token_usage.jsonl，
一行一次真实的 API 调用，供事后查询。新运行的缓存归属由 Runtime 按 Agent loop
稳定前缀估算，不代表服务商实际命中或计费；历史记录仍兼容缓存归属未知字段。抽取 Agent 以"批次"为 label 分组，
问答 Agent 以"单次问答"为 label 分组，同一 label 下累加即为该批次/该次问答的
总消耗——这是问答页面和构建终端展示的数字的来源。

对外接口：
    record(novel_id, agent, label, model, usage) -> dict | None
    query(novel_id=None, agent=None) -> list[dict]
    summarize(records) -> dict
"""

import argparse
import json
import time
from pathlib import Path

from src.config import KNOWLEDGE_BASE_DIR
from src.harness.usage import (
    TOKEN_FIELDS,
    accumulate_usage,
    empty_usage,
    extract_usage,
)


def _usage_path(novel_id: str) -> Path:
    return Path(KNOWLEDGE_BASE_DIR) / novel_id / "token_usage.jsonl"


def record(novel_id: str, agent: str, label: str, model: str, usage) -> dict | None:
    """
    追加一条用量记录到 knowledge_base/{novel_id}/token_usage.jsonl。
    usage 取不到有效 token 数时静默跳过，不中断调用方的主流程。
    """
    tokens = extract_usage(usage)
    if tokens is None:
        return None

    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "novel_id": novel_id,
        "agent": agent,
        "label": label,
        "model": model,
        **tokens,
    }

    path = _usage_path(novel_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def query(novel_id: str | None = None, agent: str | None = None) -> list[dict]:
    """读取用量记录，按时间顺序返回。novel_id 为 None 时聚合 knowledge_base 下所有小说。"""
    kb_dir = Path(KNOWLEDGE_BASE_DIR)
    paths = [_usage_path(novel_id)] if novel_id else sorted(kb_dir.glob("*/token_usage.jsonl"))

    records = []
    for p in paths:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if agent and rec.get("agent") != agent:
                continue
            records.append(rec)
    records.sort(key=lambda r: r.get("ts", ""))
    return records


def summarize(records: list[dict]) -> dict:
    """按 agent 和 label 两个维度汇总 token 数（label 维度即"每批/每次问答"的总消耗）。"""

    def _empty():
        return {**empty_usage(), "calls": 0}

    total = _empty()
    by_agent: dict[str, dict] = {}
    by_label: dict[str, dict] = {}

    for r in records:
        tokens = extract_usage(r)
        if tokens is None:
            continue

        for key, bucket in ((r.get("agent", "unknown"), by_agent), (r.get("label", "unknown"), by_label)):
            b = bucket.setdefault(key, _empty())
            accumulate_usage(b, tokens)
            b["calls"] += 1

        accumulate_usage(total, tokens)
        total["calls"] += 1

    return {"total": total, "by_agent": by_agent, "by_label": by_label}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Token 用量查询工具",
        epilog="示例：python -m src.usage_tracker shitu --agent extraction",
    )
    parser.add_argument("novel_id", nargs="?", default=None, help="只看某本小说；不传则聚合所有小说")
    parser.add_argument("--agent", default=None, help="只看某个 agent（extraction/qa）")
    parser.add_argument("--detail", action="store_true", help="额外打印按 label（每批/每次问答）的明细")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    records = query(args.novel_id, args.agent)
    if not records:
        print("没有找到用量记录")
    else:
        s = summarize(records)
        scope = args.novel_id or "全部小说"
        print(f"用量汇总（{scope}）")
        print("  缓存拆分为 Agent 前缀估算，不代表服务商命中或账单；历史记录可能使用旧口径。")
        print(f"  调用次数：{s['total']['calls']}")
        print(f"  输入 tokens：{s['total']['prompt_tokens']:,}")
        print(f"    估算非缓存读：{s['total']['uncached_prompt_tokens']:,}")
        print(f"    估算缓存读：{s['total']['cached_prompt_tokens']:,}")
        if s["total"]["unknown_cache_prompt_tokens"]:
            print(f"    缓存归属未知（历史记录）：{s['total']['unknown_cache_prompt_tokens']:,}")
        print(f"  输出 tokens：{s['total']['completion_tokens']:,}")
        print(f"  合计 tokens：{s['total']['total_tokens']:,}")
        print()
        print("按 Agent 分类：")
        for agent, b in sorted(s["by_agent"].items()):
            print(
                f"  {agent:12s}  {b['calls']:5d} 次调用  "
                f"估算非缓存 {b['uncached_prompt_tokens']:>10,}  "
                f"估算缓存 {b['cached_prompt_tokens']:>10,}  "
                f"未知 {b['unknown_cache_prompt_tokens']:>10,}  "
                f"输出 {b['completion_tokens']:>10,}  "
                f"合计 {b['total_tokens']:>12,}"
            )

        if args.detail:
            print()
            print("按 label 明细（每批次/每次问答的总消耗）：")
            for label, b in sorted(s["by_label"].items(), key=lambda kv: kv[1]["total_tokens"], reverse=True):
                print(
                    f"  {label:30s}  {b['calls']:3d} 次调用  "
                    f"估算非缓存 {b['uncached_prompt_tokens']:>10,}  "
                    f"估算缓存 {b['cached_prompt_tokens']:>10,}  "
                    f"未知 {b['unknown_cache_prompt_tokens']:>10,}  "
                    f"输出 {b['completion_tokens']:>10,}  "
                    f"合计 {b['total_tokens']:>12,}"
                )
