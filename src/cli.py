"""Novel Agent 统一命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import (
    ConfigurationError,
    EXTRACTION_MODEL,
    KNOWLEDGE_BASE_DIR,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_ENABLE_THINKING,
    LLM_INCLUDE_REASONING_EFFORT,
    QA_MODEL,
)


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        _print_help()
        return

    command, rest = arguments[0], arguments[1:]
    try:
        if command == "build":
            from src.build import main as build_main

            if not any(option in rest for option in {"-h", "--help"}):
                _require_runtime_config(EXTRACTION_MODEL, "EXTRACTION_MODEL")
            build_main(rest)
        elif command == "ask":
            _ask(rest)
        elif command == "evaluate":
            from src.evaluate import main as evaluate_main

            evaluate_main(rest)
        elif command == "config":
            _show_config(rest)
        else:
            raise SystemExit(f"未知命令：{command}\n\n{_help_text()}")
    except ConfigurationError as error:
        raise SystemExit(f"[配置错误] {error}") from error


def _ask(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="novel-agent ask", description="对已构建的小说知识库提问"
    )
    parser.add_argument("novel_id", help="知识库标识符")
    parser.add_argument("question", help="要询问的问题")
    parser.add_argument("--model", help="本次问答使用的模型，默认读取 QA_MODEL/LLM_MODEL")
    parser.add_argument(
        "--show-reasoning",
        action="store_true",
        help="在标准错误流中实时展示 Provider 返回的 reasoning_content",
    )
    parser.add_argument("--json", action="store_true", help="输出完整 JSON 结果")
    args = parser.parse_args(argv)
    _require_runtime_config(args.model or QA_MODEL, "QA_MODEL")

    novel_dir = Path(KNOWLEDGE_BASE_DIR) / args.novel_id
    if not novel_dir.is_dir():
        parser.error(f"知识库不存在：{novel_dir}；请先运行 build")

    reasoning_started = False

    def on_event(event: dict) -> None:
        nonlocal reasoning_started
        if not args.show_reasoning or event.get("type") != "reasoning_delta":
            return
        if not reasoning_started:
            print("\n[reasoning_content]", file=sys.stderr)
            reasoning_started = True
        print(event.get("delta", ""), end="", flush=True, file=sys.stderr)

    from src.qa_agent import run_qa_agent

    result = run_qa_agent(
        args.question,
        args.novel_id,
        on_event=on_event,
        model=args.model,
    )
    if reasoning_started:
        print("\n", file=sys.stderr)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result["reply"])
        usage = result.get("usage") or {}
        if usage.get("total_tokens"):
            print(f"\n本次使用 {usage['total_tokens']:,} tokens", file=sys.stderr)


def _show_config(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="novel-agent config", description="检查当前公开配置（不会输出 API Key）"
    )
    parser.parse_args(argv)
    values = {
        "ready": bool(LLM_API_KEY and LLM_BASE_URL and EXTRACTION_MODEL and QA_MODEL),
        "LLM_BASE_URL": LLM_BASE_URL,
        "LLM_API_KEY": "已配置" if LLM_API_KEY else "未配置",
        "EXTRACTION_MODEL": EXTRACTION_MODEL or "未配置",
        "QA_MODEL": QA_MODEL or "未配置",
        "KNOWLEDGE_BASE_DIR": KNOWLEDGE_BASE_DIR,
        "LLM_INCLUDE_REASONING_EFFORT": LLM_INCLUDE_REASONING_EFFORT,
        "LLM_ENABLE_THINKING": LLM_ENABLE_THINKING,
    }
    print(json.dumps(values, ensure_ascii=False, indent=2))


def _require_runtime_config(model: str, setting_name: str) -> None:
    from src.config import require_llm_model
    from src.llm import get_provider

    require_llm_model(model, setting_name)
    get_provider()


def _help_text() -> str:
    return """Novel Agent — 长篇小说知识构建与证据化问答

用法：novel-agent <command> [options]

命令：
  config     检查 LLM 和数据目录配置
  build      从小说 TXT 构建知识库
  ask        对已构建的知识库提问
  evaluate   运行固定问答评测

示例：
  novel-agent config
  novel-agent build ./novels/example.txt example
  novel-agent ask example "主角为什么离开故乡？" --show-reasoning
"""


def _print_help() -> None:
    print(_help_text())


if __name__ == "__main__":
    main()
