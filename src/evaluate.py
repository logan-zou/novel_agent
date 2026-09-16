"""运行小说 QA 固定评测集；默认只校验数据集，--live 才会产生模型调用。"""

import argparse
import json
import sys
from pathlib import Path

from src.evaluation.scoring import case_to_dict, evaluate_results, load_cases


_SOURCE_DATASET = Path(__file__).parent.parent / "evals" / "qa_cases.jsonl"
_INSTALLED_DATASET = Path(sys.prefix) / "share" / "novel-agent" / "evals" / "qa_cases.jsonl"
DEFAULT_DATASET = _SOURCE_DATASET if _SOURCE_DATASET.is_file() else _INSTALLED_DATASET


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    cases = load_cases(args.dataset)
    if args.case_id:
        cases = [case for case in cases if case.case_id == args.case_id]
        if not cases:
            raise SystemExit(f"找不到 case：{args.case_id}")

    if not args.live and not args.responses:
        print(json.dumps({"valid": True, "cases": [case_to_dict(case) for case in cases]}, ensure_ascii=False, indent=2))
        return

    results = _run_live(cases) if args.live else _load_results(args.responses)
    report = evaluate_results(cases, results)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


def _run_live(cases) -> dict[str, dict]:
    from src.qa_agent import run_qa_agent

    results = {}
    for case in cases:
        print(f"[live] {case.case_id}: {case.question}")
        results[case.case_id] = run_qa_agent(case.question, case.novel_id)
    return results


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
