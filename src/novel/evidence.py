"""小说问答证据：生成原文证据，并校验答案中的内联引用。"""

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Callable, Literal


EvidenceKind = Literal["knowledge_base", "primary_source"]


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    kind: EvidenceKind
    source: str
    chapters: tuple[int, ...]
    excerpt: str
    tool: str
    arguments: dict

    def to_dict(self) -> dict:
        value = asdict(self)
        value["chapters"] = list(self.chapters)
        return value


@dataclass(frozen=True)
class CitationReport:
    valid: bool
    cited_evidence_ids: tuple[str, ...]
    invalid_evidence_ids: tuple[str, ...]
    cited_chapters: tuple[int, ...]
    unsupported_chapters: tuple[int, ...]
    message: str = ""

    def to_dict(self) -> dict:
        value = asdict(self)
        for key in (
            "cited_evidence_ids",
            "invalid_evidence_ids",
            "cited_chapters",
            "unsupported_chapters",
        ):
            value[key] = list(value[key])
        return value


_CITATION_RE = re.compile(r"\[\[cite:(EV\d{3,})\]\]", re.IGNORECASE)
_CITATION_MARKER_RE = re.compile(r"\[\[cite:([^\]]+)\]\]", re.IGNORECASE)
_SOURCE_RE = re.compile(r"来源章节[：:]\s*([\d、，,\s\-~～]+)")
_META_CHAPTERS_RE = re.compile(r'"source_chapters":\[([^\]]*)\]')
_ANSWER_CHAPTERS_RE = re.compile(r"【引用章节[：:]([\d、，,\s第章\-~～]+)】")
_INSUFFICIENT = ("信息不足", "无法确认", "未找到", "没有足够", "不足以确认")


def parse_sources(answer: str) -> list[int]:
    """兼容旧答案 footer；新答案的章节由 EvidenceCollector 从脚标推导。"""
    match = _ANSWER_CHAPTERS_RE.search(answer)
    if not match:
        return []
    return sorted({int(value) for value in re.findall(r"\d+", match.group(1))})


def parse_citation_ids(answer: str) -> tuple[str, ...]:
    """按首次出现顺序解析 ``[[cite:EV001]]``，重复引用只保留一次。"""
    return tuple(dict.fromkeys(value.upper() for value in _CITATION_RE.findall(answer)))


class EvidenceCollector:
    def __init__(
        self,
        excerpt_chars: int = 1_200,
        source_reader: Callable[[int], str] | None = None,
        context_chars: int = 500,
    ) -> None:
        self.excerpt_chars = excerpt_chars
        self.source_reader = source_reader
        self.context_chars = context_chars
        self._items: list[EvidenceItem] = []
        self._by_id: dict[str, EvidenceItem] = {}
        self._fingerprints: dict[str, str] = {}

    @property
    def items(self) -> tuple[EvidenceItem, ...]:
        return tuple(self._items)

    def wrap(self, tool_name: str, handler: Callable[..., str]) -> Callable[..., str]:
        def wrapped(**arguments) -> str:
            return self.capture(tool_name, arguments, str(handler(**arguments)))

        return wrapped

    def capture(self, tool_name: str, arguments: dict, result: str) -> str:
        if result.startswith("[错误]"):
            return result
        if tool_name == "read_entity":
            item = self._add(
                "knowledge_base",
                str(arguments.get("ref", "")),
                _extract_chapters(result),
                result,
                tool_name,
                arguments,
            )
            return _with_header(item, result)
        if tool_name == "read_chapter":
            chapter = int(arguments.get("chapter_no"))
            item = self._add(
                "primary_source",
                f"chapters/{chapter}.txt",
                {chapter},
                result,
                tool_name,
                arguments,
            )
            return _with_header(item, result)
        if tool_name in {"search_source", "search_chapters"}:
            return self._capture_search(arguments, result)
        # 兼容旧工具名，方便已有调用逐步迁移。
        if tool_name == "read_file":
            source = str(arguments.get("path", ""))
            match = re.search(r"(?:^|/)chapters/(\d+)\.txt$", source)
            chapters = {int(match.group(1))} if match else _extract_chapters(result)
            kind: EvidenceKind = "primary_source" if match else "knowledge_base"
            return _with_header(
                self._add(kind, source, chapters, result, tool_name, arguments), result
            )
        return result

    def validate(self, answer: str) -> CitationReport:
        cited_ids = parse_citation_ids(answer)
        invalid = tuple(value for value in cited_ids if value not in self._by_id)
        items = [self._by_id[value] for value in cited_ids if value in self._by_id]
        cited_chapters = tuple(sorted({chapter for item in items for chapter in item.chapters}))
        non_source = tuple(item.evidence_id for item in items if item.tool != "search_source")
        malformed = tuple(
            value
            for value in _CITATION_MARKER_RE.findall(answer)
            if not re.fullmatch(r"EV\d{3,}", value, re.IGNORECASE)
        )

        errors = []
        if not cited_ids:
            has_source_evidence = any(item.tool == "search_source" for item in self._items)
            if not has_source_evidence and any(term in answer for term in _INSUFFICIENT):
                return CitationReport(True, (), (), (), (), "")
            errors.append("答案缺少内联引用 [[cite:EVxxx]]")
        if malformed:
            errors.append("引用格式错误，应使用 [[cite:EVxxx]]：" + ", ".join(malformed))
        if invalid:
            errors.append("引用了不存在的证据 ID：" + ", ".join(invalid))
        if non_source:
            errors.append(
                "最终脚标只能引用 search_source 返回的原文证据：" + ", ".join(non_source)
            )
        return CitationReport(
            not errors,
            cited_ids,
            invalid,
            cited_chapters,
            (),
            "；".join(errors),
        )

    def validator(self, answer: str) -> str | None:
        report = self.validate(answer)
        return None if report.valid else report.message

    def cited_items(self, answer: str) -> list[dict]:
        report = self.validate(answer)
        values = []
        for number, evidence_id in enumerate(report.cited_evidence_ids, 1):
            if evidence_id not in self._by_id:
                continue
            item = self._by_id[evidence_id].to_dict()
            item["number"] = number
            values.append(item)
        return values

    def _capture_search(self, arguments: dict, result: str) -> str:
        try:
            payload = json.loads(result)
        except json.JSONDecodeError:
            return result
        matches = payload.get("matches")
        if not isinstance(matches, list):
            return result
        enriched = []
        for match in matches:
            if not isinstance(match, dict) or not isinstance(match.get("chapter"), int):
                continue
            chapter = match["chapter"]
            snippets = [str(value) for value in match.get("snippets") or []]
            content = "\n\n".join(snippets)
            if self.source_reader:
                source_text = self.source_reader(chapter)
                if not source_text.startswith("[错误]"):
                    content = _context_around_match(
                        source_text,
                        arguments.get("keywords") or [],
                        self.context_chars,
                        snippets,
                    )
            item = self._add(
                "primary_source",
                f"chapters/{chapter}.txt",
                {chapter},
                content,
                "search_source",
                arguments,
            )
            value = dict(match)
            value["evidence_id"] = item.evidence_id
            enriched.append(value)
        payload["matches"] = enriched
        payload["citation_hint"] = "在对应结论后用 [[cite:EVxxx]] 内联引用 evidence_id"
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _add(self, kind, source, chapters, content, tool, arguments) -> EvidenceItem:
        fingerprint = hashlib.sha256(f"{kind}\0{source}\0{content}".encode()).hexdigest()
        if fingerprint in self._fingerprints:
            return self._by_id[self._fingerprints[fingerprint]]
        evidence_id = f"EV{len(self._items) + 1:03d}"
        normalized = content.strip()
        excerpt = normalized[: self.excerpt_chars]
        if len(normalized) > self.excerpt_chars:
            excerpt += "…"
        item = EvidenceItem(
            evidence_id,
            kind,
            source,
            tuple(sorted(chapters)),
            excerpt,
            tool,
            dict(arguments),
        )
        self._items.append(item)
        self._by_id[evidence_id] = item
        self._fingerprints[fingerprint] = evidence_id
        return item


def _extract_chapters(content: str) -> set[int]:
    chapters = set()
    for pattern in (_SOURCE_RE, _META_CHAPTERS_RE):
        for match in pattern.finditer(content):
            chapters.update(int(value) for value in re.findall(r"\d+", match.group(1)))
    return chapters


def _with_header(item: EvidenceItem, content: str) -> str:
    chapter_label = "、".join(f"第{value}章" for value in item.chapters) or "章节未标注"
    return f"【证据 {item.evidence_id}｜{item.kind}｜{item.source}｜{chapter_label}】\n{content}"


def _context_around_match(
    text: str,
    keywords: list[str],
    radius: int,
    snippets: list[str] | None = None,
) -> str:
    """截取首个关键词命中位置前后各 radius 字；找不到时取原文开头。"""
    lowered = text.lower()
    terms = [str(keyword).strip() for keyword in keywords if str(keyword).strip()]

    # 优先围绕 search_source 实际返回的首个片段定位，避免同一关键词在章节
    # 更前面出现过时，浮窗跳到与模型看到的检索片段无关的位置。
    for snippet in snippets or []:
        needle = snippet.strip()
        snippet_start = lowered.find(needle.lower())
        if snippet_start < 0:
            continue
        local_hits = [
            (needle.lower().find(term.lower()), len(term))
            for term in terms
            if needle.lower().find(term.lower()) >= 0
        ]
        local_start, length = min(local_hits) if local_hits else (0, len(needle))
        start = snippet_start + local_start
        return text[
            max(0, start - radius) : min(len(text), start + length + radius)
        ].strip()

    hits = []
    for term in terms:
        index = lowered.find(term.lower())
        if index >= 0:
            hits.append((index, len(term)))
    start, length = min(hits) if hits else (0, 0)
    return text[max(0, start - radius) : min(len(text), start + length + radius)].strip()
