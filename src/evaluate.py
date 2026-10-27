"""运行小说 QA 固定评测集；默认只校验数据集，--live 才会产生模型调用。"""

import argparse
import json
import re
import sys
from pathlib import Path

from src.evaluation.scoring import case_to_dict, evaluate_results, load_cases


_SOURCE_DATASET = Path(__file__).parent.parent / "evals" / "qa_cases.jsonl"
_INSTALLED_DATASET = Path(sys.prefix) / "share" / "novel-agent" / "evals" / "qa_cases.jsonl"
DEFAULT_DATASET = _SOURCE_DATASET if _SOURCE_DATASET.is_file() else _INSTALLED_DATASET


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        cases = load_cases(args.dataset)
    except (OSError, ValueError, KeyError, TypeError) as error:
        _input_error(f"无法加载评测集 {args.dataset}：{error}")
    if args.case_id:
        cases = [case for case in cases if case.case_id == args.case_id]
        if not cases:
            _input_error(f"找不到 case：{args.case_id}")
    if not cases:
        _input_error("评测集没有可运行的 case")

    if not args.live and not args.responses:
        _write_report({"valid": True, "cases": [case_to_dict(case) for case in cases]}, args.output)
        return

    if args.live:
        results = _run_live(cases)
    else:
        try:
            results = _load_results(args.responses)
        except (OSError, ValueError, KeyError, TypeError) as error:
            _input_error(f"无法加载离线回答 {args.responses}：{error}")
    report = evaluate_results(cases, results)
    _write_report(report, args.output)
    if report["summary"]["passed"] < report["summary"]["total"]:
        raise SystemExit(1)


def _write_report(report: dict, output: str | None) -> None:
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if output:
        try:
            Path(output).write_text(rendered + "\n", encoding="utf-8")
        except OSError as error:
            _input_error(f"无法保存评测报告 {output}：{error}")


def _input_error(message: str) -> None:
    print(f"[评测错误] {message}", file=sys.stderr)
    raise SystemExit(2)


def _run_live(cases) -> dict[str, dict]:
    _preflight_live(cases)
    from src.qa_agent import run_qa_agent

    results = {}
    for case in cases:
        print(f"[live] {case.case_id}: {case.question}", file=sys.stderr)
        results[case.case_id] = run_qa_agent(case.question, case.novel_id)
    return results


def _preflight_live(cases) -> None:
    from src.config import KNOWLEDGE_BASE_DIR
    from src.novel.corpus import CorpusStore
    from src.novel.knowledge import KnowledgeStore

    novels = {}
    for case in cases:
        novels.setdefault(case.novel_id, set()).update(case.expected_chapters)

    errors = []
    for novel_id, expected_chapters in novels.items():
        base_dir = Path(KNOWLEDGE_BASE_DIR) / novel_id
        if not base_dir.is_dir():
            errors.append(f"{novel_id}：知识库目录不存在：{base_dir}")
            continue
        try:
            knowledge = KnowledgeStore(base_dir)
            records = knowledge.catalog_records()
            refs = [record.ref for record in records]
            if not refs:
                # QA 仍支持旧 Markdown 索引，预检同样接受可回源的旧条目。
                for relative in ("characters/_index.md", "settings/_index.md", "plot/arcs/_index.md"):
                    index = base_dir / relative
                    if index.is_file():
                        refs.extend(re.findall(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", index.read_text(encoding="utf-8")))
            if not refs:
                errors.append(f"{novel_id}：知识库没有可检索的实体，请先运行 build")
            else:
                invalid_refs = []
                for ref in dict.fromkeys(refs):
                    text = knowledge.read_entity(ref)
                    if not text.strip() or text.startswith("[错误]"):
                        invalid_refs.append(ref)
                if invalid_refs:
                    errors.append(f"{novel_id}：实体文件缺失或为空（共 {len(invalid_refs)} 项）：{', '.join(invalid_refs[:5])}")

            corpus = CorpusStore(base_dir)
            chapters = corpus.chapter_numbers()
            if not chapters:
                errors.append(f"{novel_id}：没有章节原文，请先运行 build：{corpus.chapters_dir}")
            invalid_chapters = sorted(set(expected_chapters) - set(chapters))
            invalid_chapters.extend(chapter for chapter in chapters if not corpus.read_chapter(chapter).strip())
            if invalid_chapters:
                errors.append(f"{novel_id}：章节原文缺失或为空（共 {len(invalid_chapters)} 项）：{', '.join(map(str, invalid_chapters[:10]))}")
        except (OSError, UnicodeError, ValueError) as error:
            errors.append(f"{novel_id}：无法读取知识库或章节原文：{error}")

    if errors:
        _input_error("live 预检未通过，尚未发起任何模型请求：\n- " + "\n- ".join(errors))


def _load_results(path: str) -> dict[str, dict]:
    results = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        value = json.loads(line)
        case_id = value.pop("case_id")
        results[case_id] = value
    return results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="novel-agent evaluate", description="小说问答固定评测")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="JSONL 评测集")
    parser.add_argument("--responses", help="离线回答 JSONL；每行包含 case_id 和 QA 返回字段")
    parser.add_argument("--live", action="store_true", help="真实调用 QA Agent（会产生模型费用）")
    parser.add_argument("--case-id", help="只运行一个 case")
    parser.add_argument("--output", help="把 JSON 报告写入指定路径")
    args = parser.parse_args(argv)
    if args.live and args.responses:
        parser.error("--live 与 --responses 不能同时使用")
    return args


if __name__ == "__main__":
    main()
