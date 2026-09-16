"""无需 LLM judge 的基础评测：关键事实、反例、章节与证据有效性。"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    novel_id: str
    question: str
    required_any: tuple[tuple[str, ...], ...] = ()
    forbidden_terms: tuple[str, ...] = ()
    expected_chapters: tuple[int, ...] = ()
    require_valid_citations: bool = True
    min_evidence: int = 1
    notes: str = ""

    @classmethod
    def from_dict(cls, value: dict) -> "EvalCase":
        return cls(
            case_id=value["case_id"],
            novel_id=value["novel_id"],
            question=value["question"],
            required_any=tuple(tuple(group) for group in value.get("required_any", [])),
            forbidden_terms=tuple(value.get("forbidden_terms", [])),
            expected_chapters=tuple(value.get("expected_chapters", [])),
            require_valid_citations=value.get("require_valid_citations", True),
            min_evidence=value.get("min_evidence", 1),
            notes=value.get("notes", ""),
        )


def load_cases(path: str | Path) -> list[EvalCase]:
    cases = []
    seen = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        case = EvalCase.from_dict(json.loads(line))
        if case.case_id in seen:
            raise ValueError(f"评测 case_id 重复（第 {line_number} 行）：{case.case_id}")
        seen.add(case.case_id)
        cases.append(case)
    return cases


def score_case(case: EvalCase, result: dict) -> dict:
    reply = str(result.get("reply", ""))
    lowered_reply = reply.lower()
    required_hits = [
        any(candidate.lower() in lowered_reply for candidate in group)
        for group in case.required_any
    ]
    forbidden_hits = [
        term for term in case.forbidden_terms if term.lower() in lowered_reply
    ]
    sources = {int(chapter) for chapter in result.get("sources", [])}
    missing_chapters = [
        chapter for chapter in case.expected_chapters if chapter not in sources
    ]
    evidence_count = len(result.get("evidence") or [])
    citation_valid = bool(
        (result.get("citation_validation") or {}).get("valid", False)
    )

    checks = {
        "required_facts": all(required_hits),
        "forbidden_absent": not forbidden_hits,
        "expected_chapters": not missing_chapters,
        "citation_valid": citation_valid if case.require_valid_citations else True,
        "evidence_count": evidence_count >= case.min_evidence,
    }
    return {
        "case_id": case.case_id,
        "passed": all(checks.values()),
        "checks": checks,
        "required_group_hits": required_hits,
        "forbidden_hits": forbidden_hits,
        "missing_chapters": missing_chapters,
        "evidence_count": evidence_count,
        "citation_valid": citation_valid,
        "reply": reply,
    }


def evaluate_results(cases: list[EvalCase], results_by_id: dict[str, dict]) -> dict:
    scored = []
    for case in cases:
        result = results_by_id.get(case.case_id)
        if result is None:
            scored.append(
                {
                    "case_id": case.case_id,
                    "passed": False,
                    "checks": {"result_present": False},
                    "error": "缺少评测结果",
                }
            )
        else:
            scored.append(score_case(case, result))

    total = len(scored)
    passed = sum(1 for item in scored if item["passed"])
    check_names = sorted(
        {name for item in scored for name in item.get("checks", {})}
    )
    check_rates = {
        name: (
            sum(1 for item in scored if item.get("checks", {}).get(name) is True) / total
            if total
            else 0.0
        )
        for name in check_names
    }
    return {
        "summary": {
            "total": total,
            "passed": passed,
            "pass_rate": passed / total if total else 0.0,
            "check_rates": check_rates,
        },
        "cases": scored,
    }


def case_to_dict(case: EvalCase) -> dict:
    value = asdict(case)
    value["required_any"] = [list(group) for group in case.required_any]
    return value
