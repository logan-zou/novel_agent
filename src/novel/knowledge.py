"""一实体一文件的小说知识库，以及一次抽取批次的轻量暂存会话。"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path


ENTITY_KINDS = {"character", "setting", "event", "arc"}
_META_RE = re.compile(r"^<!-- novel-agent: (\{.*\}) -->$", re.MULTILINE)
_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_SOURCE_FOOTER_RE = re.compile(r"\n+(来源章节[：:].*)\Z")
_LEGACY_INDEXES = (
    ("character", "characters/_index.md"),
    ("setting", "settings/_index.md"),
    ("arc", "plot/arcs/_index.md"),
)
_MAX_ENTITY_CHARS = 30_000


@dataclass(frozen=True)
class EntityRecord:
    kind: str
    name: str
    ref: str
    aliases: tuple[str, ...]
    summary: str
    source_chapters: tuple[int, ...]
    domain: str = ""

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "name": self.name,
            "ref": self.ref,
            "aliases": list(self.aliases),
            "summary": self.summary,
            "source_chapters": list(self.source_chapters),
            "domain": self.domain,
        }


class KnowledgeStore:
    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir).resolve()
        self.catalog_path = self.base_dir / "catalog.jsonl"

    def entity_ref(self, kind: str, name: str, domain: str = "") -> str:
        _validate_name(name, "实体名")
        if kind not in ENTITY_KINDS:
            raise ValueError(f"不支持的实体类型：{kind}")
        if kind == "character":
            return f"characters/{name}.md"
        if kind == "event":
            return f"plot/events/{name}.md"
        if kind == "arc":
            return f"plot/arcs/{name}.md"
        _validate_name(domain, "设定领域")
        return f"settings/{domain}/{name}.md"

    def read_entity(self, ref: str, section: str = "") -> str:
        try:
            path, anchor = self._resolve_ref(ref)
        except (PermissionError, ValueError) as error:
            return f"[错误] {error}"
        if not path.exists():
            return f"[错误] 实体不存在：{ref}"
        text = path.read_text(encoding="utf-8")
        wanted = section.strip() or anchor
        if wanted:
            extracted = _extract_section(text, wanted)
            if extracted is None:
                return f"[错误] 实体 {ref} 中不存在章节：{wanted}"
            return extracted
        if len(text) <= _MAX_ENTITY_CHARS:
            return text
        return text[:_MAX_ENTITY_CHARS] + f"\n\n……（实体过长，后续 {len(text) - _MAX_ENTITY_CHARS} 字省略）"

    def search_entities(
        self,
        keywords: list[str],
        entity_types: list[str] | None = None,
        limit: int = 30,
        extra_records: list[EntityRecord] | None = None,
    ) -> str:
        terms = [str(value).strip().lower() for value in keywords if str(value).strip()]
        if not terms:
            return json.dumps({"error": "keywords 不能为空"}, ensure_ascii=False)
        allowed = set(entity_types or ENTITY_KINDS)
        records = {record.ref: record for record in self.catalog_records()}
        for record in extra_records or []:
            records[record.ref] = record

        scored = []
        for record in records.values():
            if record.kind not in allowed:
                continue
            name_text = " ".join((record.name, *record.aliases)).lower()
            full_text = f"{name_text} {record.summary}".lower()
            hits = sum(term in full_text for term in terms)
            if not hits:
                continue
            exact_bonus = sum(term == record.name.lower() or term in [a.lower() for a in record.aliases] for term in terms)
            scored.append((hits * 10 + exact_bonus * 5, record))

        # 旧版多实体 Markdown 知识库保持只读兼容。
        legacy = self._search_legacy_indexes(terms, allowed)
        scored.sort(key=lambda item: (-item[0], item[1].name))
        safe_limit = max(1, min(int(limit), 100))
        result = [record.to_dict() for _, record in scored[:safe_limit]]
        known_refs = {item["ref"] for item in result}
        for item in legacy:
            if item["ref"] not in known_refs and len(result) < safe_limit:
                result.append(item)
                known_refs.add(item["ref"])
        return json.dumps({"matches": result}, ensure_ascii=False, indent=2)

    def catalog_records(self) -> list[EntityRecord]:
        if self.catalog_path.exists():
            records = []
            for line in self.catalog_path.read_text(encoding="utf-8").splitlines():
                try:
                    records.append(_record_from_dict(json.loads(line)))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
            return records
        return self._scan_records()

    def rebuild_catalog(self) -> list[EntityRecord]:
        records = sorted(self._scan_records(), key=lambda item: (item.kind, item.name, item.ref))
        self.base_dir.mkdir(parents=True, exist_ok=True)
        rendered = "".join(
            json.dumps(record.to_dict(), ensure_ascii=False) + "\n" for record in records
        )
        temp = self.catalog_path.with_suffix(".jsonl.tmp")
        temp.write_text(rendered, encoding="utf-8")
        os.replace(temp, self.catalog_path)
        return records

    def _scan_records(self) -> list[EntityRecord]:
        records = []
        for root in ("characters", "settings", "plot/events", "plot/arcs"):
            directory = self.base_dir / root
            if not directory.exists():
                continue
            for path in directory.rglob("*.md"):
                record = _parse_record(path, self.base_dir)
                if record:
                    records.append(record)
        return records

    def _search_legacy_indexes(self, terms: list[str], allowed: set[str]) -> list[dict]:
        matches = []
        for kind, relative in _LEGACY_INDEXES:
            if kind not in allowed:
                continue
            path = self.base_dir / relative
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not any(term in line.lower() for term in terms):
                    continue
                link = _WIKILINK_RE.search(line)
                if not link:
                    continue
                ref = link.group(1).split("|", 1)[0].strip()
                matches.append(
                    {
                        "kind": kind,
                        "name": ref.rsplit("#", 1)[-1].rsplit("/", 1)[-1],
                        "ref": ref,
                        "aliases": [],
                        "summary": line.lstrip("- ").strip(),
                        "source_chapters": [],
                        "domain": "",
                        "legacy": True,
                    }
                )
        return matches

    def _resolve_ref(self, ref: str) -> tuple[Path, str]:
        value = str(ref).strip()
        if value.startswith("[[") and value.endswith("]]"):
            value = value[2:-2]
        value = value.split("|", 1)[0].strip()
        relative, _, anchor = value.partition("#")
        if not relative.endswith(".md"):
            relative += ".md"
        target = (self.base_dir / relative).resolve()
        try:
            target.relative_to(self.base_dir)
        except ValueError as error:
            raise PermissionError(f"路径越界：{ref}") from error
        return target, anchor.strip()


class BuildSession:
    """一次抽取批次的内存 overlay；最终校验后再写入真实知识库。"""

    def __init__(self, store: KnowledgeStore) -> None:
        self.store = store
        self.pending: dict[str, tuple[EntityRecord, str]] = {}

    def search_entities(self, keywords, entity_types=None, limit=30) -> str:
        return self.store.search_entities(
            keywords,
            entity_types,
            limit,
            extra_records=[record for record, _ in self.pending.values()],
        )

    def read_entity(self, ref: str, section: str = "") -> str:
        value = str(ref).strip()
        if value.startswith("[[") and value.endswith("]]"):
            value = value[2:-2]
        value = value.split("|", 1)[0].strip()
        normalized, _, anchor = value.partition("#")
        if not normalized.endswith(".md"):
            normalized += ".md"
        if normalized in self.pending:
            text = self.pending[normalized][1]
            wanted = section.strip() or anchor.strip()
            if wanted:
                return _extract_section(text, wanted) or f"[错误] 不存在章节：{wanted}"
            return text
        return self.store.read_entity(ref, section)

    def upsert_entity(
        self,
        kind: str,
        name: str,
        content: str,
        source_chapters: list[int],
        aliases: list[str] | None = None,
        summary: str = "",
        domain: str = "",
    ) -> str:
        try:
            ref = self.store.entity_ref(kind, name.strip(), domain.strip())
            chapters = tuple(sorted({int(value) for value in source_chapters}))
            clean_aliases = tuple(dict.fromkeys(str(alias).strip() for alias in aliases or [] if str(alias).strip()))
            record = EntityRecord(
                kind=kind,
                name=name.strip(),
                ref=ref,
                aliases=clean_aliases,
                summary=" ".join(summary.split()),
                source_chapters=chapters,
                domain=domain.strip(),
            )
            document = _render_document(record, content)
        except (TypeError, ValueError) as error:
            return f"[错误] {error}"
        self.pending[ref] = (record, document)
        return f"[已暂存] {ref}"

    def validate(self) -> list[str]:
        errors = []
        chapter_files = (self.store.base_dir / "chapters").glob("*.txt")
        available_chapters = {int(path.stem) for path in chapter_files if path.stem.isdigit()}
        pending_refs = set(self.pending)
        for ref, (record, document) in self.pending.items():
            if not record.summary:
                errors.append(f"{ref}: summary 不能为空")
            if not record.source_chapters:
                errors.append(f"{ref}: source_chapters 不能为空")
            missing_chapters = [
                number for number in record.source_chapters if number not in available_chapters
            ]
            if missing_chapters:
                errors.append(f"{ref}: 来源章节不存在 {missing_chapters}")
            if len(document) > _MAX_ENTITY_CHARS:
                errors.append(f"{ref}: 实体超过 {_MAX_ENTITY_CHARS} 字，请只保留关键事实和里程碑")
            for raw_link in _WIKILINK_RE.findall(document):
                linked = raw_link.split("|", 1)[0].split("#", 1)[0].strip()
                linked_ref = linked if linked.endswith(".md") else linked + ".md"
                if linked_ref in pending_refs:
                    continue
                try:
                    target, _ = self.store._resolve_ref(linked_ref)
                except (PermissionError, ValueError):
                    errors.append(f"{ref}: 非法链接 {raw_link}")
                    continue
                if not target.exists():
                    errors.append(f"{ref}: 链接目标不存在 {raw_link}")
        return errors

    def validator(self, _: str) -> str | None:
        errors = self.validate()
        return "\n".join(f"- {error}" for error in errors) if errors else None

    def commit(self) -> dict:
        errors = self.validate()
        if errors:
            raise ValueError("知识变更校验失败：\n" + "\n".join(errors))
        originals: dict[str, str | None] = {}
        applied: list[str] = []
        temporary: list[Path] = []
        try:
            for ref, (_, document) in self.pending.items():
                target = self.store.base_dir / ref
                target.parent.mkdir(parents=True, exist_ok=True)
                originals[ref] = target.read_text(encoding="utf-8") if target.exists() else None
                temp = target.with_name(f".{target.name}.tmp")
                temp.write_text(document, encoding="utf-8")
                temporary.append(temp)
            for ref in sorted(self.pending):
                target = self.store.base_dir / ref
                temp = target.with_name(f".{target.name}.tmp")
                os.replace(temp, target)
                applied.append(ref)
            self.store.rebuild_catalog()
        except Exception:
            for ref in reversed(applied):
                target = self.store.base_dir / ref
                original = originals[ref]
                if original is None:
                    if target.exists():
                        target.unlink()
                else:
                    target.write_text(original, encoding="utf-8")
            raise
        finally:
            for path in temporary:
                if path.exists():
                    path.unlink()
        return {"modified_entities": sorted(self.pending), "count": len(self.pending)}


def _validate_name(value: str, label: str) -> None:
    reserved = ("/", "\\", "\0", "#", "|", "[", "]", "\r", "\n")
    if not value or value in {".", ".."} or any(char in value for char in reserved):
        raise ValueError(f"非法{label}：{value!r}")


def _render_document(record: EntityRecord, content: str) -> str:
    body = str(content).strip()
    if not body:
        raise ValueError("content 不能为空")
    metadata = json.dumps(record.to_dict(), ensure_ascii=False, separators=(",", ":"))
    chapters = ", ".join(str(number) for number in record.source_chapters)
    return f"<!-- novel-agent: {metadata} -->\n# {record.name}\n\n{body}\n\n来源章节：{chapters}\n"


def _parse_record(path: Path, base_dir: Path) -> EntityRecord | None:
    text = path.read_text(encoding="utf-8")
    match = _META_RE.search(text)
    if not match:
        return None
    try:
        value = json.loads(match.group(1))
        value["ref"] = str(path.relative_to(base_dir))
        return _record_from_dict(value)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _record_from_dict(value: dict) -> EntityRecord:
    return EntityRecord(
        kind=str(value["kind"]),
        name=str(value["name"]),
        ref=str(value["ref"]),
        aliases=tuple(str(item) for item in value.get("aliases", [])),
        summary=str(value.get("summary", "")),
        source_chapters=tuple(int(item) for item in value.get("source_chapters", [])),
        domain=str(value.get("domain", "")),
    )


def _extract_section(text: str, heading: str) -> str | None:
    matches = list(_HEADING_RE.finditer(text))
    for index, match in enumerate(matches):
        if match.group(2).strip() != heading.strip():
            continue
        level = len(match.group(1))
        end = len(text)
        for later in matches[index + 1 :]:
            if len(later.group(1)) <= level:
                end = later.start()
                break
        section = text[match.start() : end].strip()
        if _META_RE.search(text):
            # 新实体把来源作为文档级元数据；局部读取时明确与小节正文分隔。
            section = _SOURCE_FOOTER_RE.sub(r"\n\n---\n\1", section)
        return section
    return None
