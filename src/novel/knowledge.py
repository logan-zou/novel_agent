"""一实体一文件的小说知识库，以及一次抽取批次的轻量暂存会话。"""

import json
import hashlib
import os
import re
from collections import Counter
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
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
_MODEL_PROTOCOL_RE = re.compile(
    r"<｜(?:begin|end)▁of▁sentence｜>"
    r"|</?(?:｜DSML｜|\|DSML\|)(?:tool_calls|function_calls|invoke|parameter)(?=[\s>])[^>]*>"
)


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

    def read_entity(self, ref: str, section: str = "", field: str = "document") -> str:
        try:
            path, anchor = self._resolve_ref(ref)
            _validate_read_field(field, section, anchor)
        except (PermissionError, ValueError) as error:
            return f"[错误] {error}"
        if not path.exists():
            return f"[错误] 实体不存在：{ref}"
        text = path.read_text(encoding="utf-8")
        result = _read_document_field(text, ref, section, anchor, field)
        if field == "summary" or section.strip() or anchor:
            return result
        if len(result) <= _MAX_ENTITY_CHARS:
            return result
        return result[:_MAX_ENTITY_CHARS] + f"\n\n……（实体过长，后续 {len(result) - _MAX_ENTITY_CHARS} 字省略）"

    def entity_version(self, ref: str) -> str:
        """完整文档的只读版本指纹，不受读取展示截断影响；无法读取时返回空串。"""
        try:
            path, _ = self._resolve_ref(ref)
            if not path.is_file():
                return ""
            return _document_fingerprint(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError, PermissionError):
            return ""

    def search_entities(
        self,
        keywords: list[str],
        entity_types: list[str] | None = None,
        limit: int = 30,
        extra_records: list[EntityRecord] | None = None,
    ) -> str:
        query_terms = {}
        for value in keywords:
            keyword = str(value).strip()
            if keyword:
                query_terms.setdefault(keyword.casefold(), keyword)
        if not query_terms:
            return json.dumps({"error": "keywords 不能为空"}, ensure_ascii=False)
        terms = list(query_terms)
        allowed = set(entity_types or ENTITY_KINDS)
        records = {record.ref: record for record in self.catalog_records()}
        for record in extra_records or []:
            records[record.ref] = record

        def ref_key(ref: str) -> str:
            try:
                path, anchor = self._resolve_ref(ref)
                relative = path.relative_to(self.base_dir).as_posix()
                return relative + (f"#{anchor}" if anchor else "")
            except (ValueError, PermissionError):
                return ref

        candidates = {
            ref_key(record.ref): record.to_dict()
            for record in records.values() if record.kind in allowed
        }
        legacy, legacy_summary_terms = {}, {}
        for item in self._search_legacy_indexes(terms, allowed):
            key = ref_key(item["ref"])
            if key in candidates:
                continue
            legacy_summary_terms.setdefault(key, set()).update(
                term for term in terms if term in item["summary"].casefold()
            )
            if key not in legacy:
                legacy[key] = item
            else:
                current = legacy[key]
                current["aliases"] = list(dict.fromkeys((*current["aliases"], *item["aliases"])))
        candidates.update(legacy)

        scored = []
        for key, item in candidates.items():
            names = [str(value).casefold() for value in (item["name"], *item["aliases"])]
            summary = item["summary"].casefold()
            exact = [term in names for term in terms]
            name_hits = [any(term in name for name in names) for term in terms]
            summary_hits = [term in legacy_summary_terms[key] if key in legacy_summary_terms else term in summary
                            for term in terms]
            matches = [name_hit or summary_hit for name_hit, summary_hit in zip(name_hits, summary_hits)]
            hits = sum(matches)
            if not hits:
                continue
            # 精确名称或别名不能被摘要中碰巧包含多个查询词的实体挤掉。
            tier = 2 if any(exact) else 1 if any(name_hits) else 0
            scored.append((tier, hits * 10 + sum(exact) * 5, item, exact, name_hits, summary_hits, matches))

        scored.sort(key=lambda row: (-row[0], -row[1], row[2]["name"], row[2]["ref"]))
        safe_limit = max(1, min(int(limit), 100))
        returned = scored[:safe_limit]
        keyword_matches = []
        uncovered, unmatched = [], []
        for index, keyword in enumerate(query_terms.values()):
            total = sum(row[6][index] for row in scored)
            exact_count = sum(row[3][index] for row in scored)
            returned_count = sum(row[6][index] for row in returned)
            returned_exact_count = sum(row[3][index] for row in returned)
            status = (
                "not_found" if not total else "omitted" if not returned_count else
                "partial" if returned_count < total else "complete"
            )
            keyword_matches.append({
                "keyword": keyword, "total_count": total, "exact_count": exact_count,
                "name_count": sum(row[4][index] for row in scored),
                "summary_count": sum(row[5][index] for row in scored),
                "returned_count": returned_count, "returned_exact_count": returned_exact_count,
                "status": status,
            })
            if not total:
                unmatched.append(keyword)
            elif returned_exact_count < exact_count or not returned_count:
                uncovered.append(keyword)
        return json.dumps({
            "matches": [row[2] for row in returned],
            "total_matches": len(scored), "returned_count": len(returned),
            "limit": safe_limit, "truncated": len(returned) < len(scored),
            "keyword_matches": keyword_matches,
            "uncovered_keywords": uncovered, "unmatched_keywords": unmatched,
            "instruction": (
                "先读取 matches 中已命中的实体，再处理未完成事项。"
                "uncovered_keywords 表示已有候选或精确名称被截断，请只对这些词少量或单独补查；"
                "不要原样重复整个查询。unmatched_keywords 只表示当前知识目录中未命中，"
                "不代表小说中不存在该对象，可根据本批原文核实别名或建立有依据的新实体。"
            ),
        }, ensure_ascii=False, indent=2)

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
                if not any(term in line.casefold() for term in terms):
                    continue
                link = _WIKILINK_RE.search(line)
                if not link:
                    continue
                target, _, label = link.group(1).partition("|")
                ref = target.strip()
                relative, _, anchor = ref.partition("#")
                name = anchor.strip() or Path(relative).stem
                matches.append(
                    {
                        "kind": kind,
                        "name": name,
                        "ref": ref,
                        "aliases": [label.strip()] if label.strip() and label.strip() != name else [],
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
        self.change_count = 0
        self.failed_writes: dict[str, str] = {}
        self.repair_completion_blocked = False
        self._new_refs: set[str] = set()
        self._entity_ids: dict[str, str] = {}
        self._entity_refs: dict[str, str] = {}
        self._issue_keys: dict[tuple[str, str, str], str] = {}
        self._issues: dict[str, dict] = {}
        self._noop_failures: dict[str, str] = {}
        self._link_failures: dict[tuple[str, str], str] = {}
        self._confirmed_link_versions: set[tuple[str, str, str]] = set()
        self._management_failures: dict[str, str] = {}
        self._write_failures: dict[str, str] = {}
        self._patch_failures: dict[str, dict] = {}
        self._patch_failure_keys: dict[str, str] = {}

    def search_entities(self, keywords, entity_types=None, limit=30) -> str:
        payload = json.loads(self.store.search_entities(
            keywords,
            entity_types,
            limit,
            extra_records=[record for record, _ in self.pending.values()],
        ))
        for record in payload.get("matches", []):
            if not record.get("legacy"):
                record["entity_id"] = self._entity_id(record["ref"])
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _entity_id(self, ref: str) -> str:
        if ref not in self._entity_ids:
            entity_id = f"E{len(self._entity_refs) + 1}"
            self._entity_ids[ref] = entity_id
            self._entity_refs[entity_id] = ref
        return self._entity_ids[ref]

    def _known_records(self) -> dict[str, EntityRecord]:
        records = {record.ref: record for record in self.store.catalog_records()}
        records.update({ref: record for ref, (record, _) in self.pending.items()})
        valid = {}
        for ref, record in records.items():
            try:
                path, anchor = self.store._resolve_ref(ref)
            except (ValueError, PermissionError):
                continue
            if not anchor and (ref in self.pending or path.is_file()):
                valid[ref] = record
        return valid

    def _repair_entity(self, record: EntityRecord) -> dict:
        return {
            "entity_id": self._entity_id(record.ref), "ref": record.ref,
            "name": record.name, "kind": record.kind,
            "is_new": record.ref in self._new_refs,
        }

    def inspect_repairs(self) -> str:
        """把缺失链接和可选目标编号交给模型；只检查、不改动暂存内容。"""
        records = self._known_records()
        entities = {
            ref: self._repair_entity(record) for ref, (record, _) in self.pending.items()
        }
        issues = []
        for source_ref, (_, document) in sorted(self.pending.items()):
            missing = set()
            for raw_link in _WIKILINK_RE.findall(_document_body(document)):
                try:
                    target, _ = self.store._resolve_ref(raw_link)
                    linked_ref = target.relative_to(self.store.base_dir).as_posix()
                except (ValueError, PermissionError):
                    continue
                if linked_ref not in self.pending and not target.is_file():
                    missing.add(linked_ref)
            for missing_ref in sorted(missing):
                fingerprint = _document_fingerprint(document)
                key = (source_ref, missing_ref, fingerprint)
                if key not in self._issue_keys:
                    issue_id = f"L{len(self._issues) + 1}"
                    self._issue_keys[key] = issue_id
                    self._issues[issue_id] = {
                        "source_ref": source_ref, "missing_ref": missing_ref,
                        "fingerprint": fingerprint,
                    }
                name = Path(missing_ref).stem
                kind = _kind_from_ref(missing_ref)
                ranked = []
                for record in records.values():
                    if record.kind != kind:
                        continue
                    score = max(
                        SequenceMatcher(None, name, candidate).ratio()
                        for candidate in (record.name, *record.aliases)
                    )
                    if score >= 0.4:
                        ranked.append((score, record.ref, record))
                ranked.sort(key=lambda item: (-item[0], item[1]))
                candidates = []
                for _, ref, record in ranked[:5]:
                    candidate = self._repair_entity(record)
                    candidates.append(candidate)
                    entities[ref] = candidate
                issues.append({
                    "issue_id": self._issue_keys[key],
                    "source_id": self._entity_id(source_ref),
                    "source_ref": source_ref, "missing_ref": missing_ref,
                    "candidates": candidates,
                })
        link_failures = []
        for (source_ref, missing_ref), _ in sorted(self._link_failures.items()):
            document = self.pending[source_ref][1] if source_ref in self.pending else None
            fingerprint = _document_fingerprint(document) if document is not None else None
            previous = [(issue_id, issue) for issue_id, issue in self._issues.items()
                        if issue["source_ref"] == source_ref and issue["missing_ref"] == missing_ref]
            matching = [(issue_id, issue) for issue_id, issue in previous
                        if issue["fingerprint"] == fingerprint]
            selected = (matching or previous)[-1] if previous else None
            target = records.get(missing_ref)
            target_id = self._entity_id(missing_ref) if target is not None else ""
            if target is not None:
                entities[missing_ref] = self._repair_entity(target)
            can_confirm = bool(matching and target is not None and document is not None
                               and _contains_ref(_document_body(document), self.store, missing_ref))
            if can_confirm:
                reason = "原链接目标现已存在，来源版本一致；请用 issue_id 和 target_id 调用 repair_links 显式核验解除记录"
            elif document is None:
                reason = "来源实体已不存在，请核对本批实体后明确结束"
            elif not matching:
                reason = "来源版本已变化，旧问题不可确认；请读取当前来源实体并完整重新提交，保留已核实内容"
            elif target is None:
                reason = "原目标仍缺失，请处理当前 issues；原目标尚无可用于确认的实体编号"
            else:
                reason = "原引用状态已变化，请读取来源实体并完整重新提交，不要重试旧编号"
            link_failures.append({
                "issue_id": selected[0] if selected else "", "source_ref": source_ref,
                "missing_ref": missing_ref, "target_id": target_id,
                "can_confirm": can_confirm, "reason": reason,
            })
        return json.dumps({
            "entities": list(entities.values()), "issues": issues,
            "link_failures": link_failures,
            "patch_failures": [self._describe_patch_failure(failure_id, failure)
                               for failure_id, failure in self._patch_failures.items()
                               if not failure["resolved"]],
            "validation_errors": self.validate(),
            "feedback": "候选仅供核对，不会自动选择。确认后用 repair_links 的 issue_id 和 target_id 修复；"
                        "本批误建实体可用 manage_entities 撤销或更名。",
        }, ensure_ascii=False)

    def repair_links(self, repairs: list[dict]) -> str:
        """按问题和目标编号精确修复链接；同实体各项均通过后一起应用。"""
        if not isinstance(repairs, list) or not repairs or len(repairs) > 8:
            self.repair_completion_blocked = True
            return _repair_summary([{"result": "[错误] repairs 必须是最多 8 项的非空数组"}], 0, resolved_count=0)
        changes_before = self.change_count
        records = self._known_records()
        original = dict(self.pending)
        results = []
        prepared: dict[str, list[tuple[int, dict, str, bool]]] = {}
        rejected_sources = set()
        seen_issues = set()
        resolved = 0
        for index, item in enumerate(repairs):
            issue = (
                self._issues.get(item.get("issue_id"))
                if isinstance(item, dict) and isinstance(item.get("issue_id"), str) else None
            )
            try:
                if not isinstance(item, dict) or set(item) != {"issue_id", "target_id"}:
                    raise ValueError("每项必须且只能包含 issue_id 和 target_id")
                if not isinstance(item["issue_id"], str) or not isinstance(item["target_id"], str):
                    raise ValueError("issue_id 和 target_id 必须是文本编号")
                issue = self._issues.get(item["issue_id"])
                if issue is None:
                    raise ValueError("问题编号不存在，请先调用 inspect_repairs")
                source = issue["source_ref"]
                if source not in original or _document_fingerprint(original[source][1]) != issue["fingerprint"]:
                    raise ValueError("问题已过期，请重新调用 inspect_repairs，不要重试旧编号")
                if item["issue_id"] in seen_issues:
                    raise ValueError("同一问题不能在一次调用中重复指定")
                seen_issues.add(item["issue_id"])
                target_ref = self._entity_refs.get(item["target_id"])
                if target_ref not in records:
                    raise ValueError("目标实体编号无效或目标已不存在，请重新查询")
                missing_path, _ = self.store._resolve_ref(issue["missing_ref"])
                already_valid = issue["missing_ref"] in self.pending or missing_path.is_file()
                if already_valid and target_ref != issue["missing_ref"]:
                    expected = self._entity_id(issue["missing_ref"]) if issue["missing_ref"] in records else ""
                    hint = f"请用 target_id={expected} 显式核验原链接" if expected else "请重新查询原链接目标"
                    raise ValueError(f"原链接目标现已存在，但 target_id 指向其他实体，未执行重定向；{hint}")
                prepared.setdefault(source, []).append((index, issue, target_ref, already_valid))
                results.append({"index": index, "result": ""})
            except (KeyError, TypeError, ValueError, PermissionError) as error:
                result = f"[错误] {error}"
                results.append({"index": index, "result": result})
                if issue and issue["source_ref"] in self.pending:
                    source = issue["source_ref"]
                    rejected_sources.add(source)
                    if _contains_ref(self.pending[source][1], self.store, issue["missing_ref"]):
                        self._link_failures[(source, issue["missing_ref"])] = result
                else:
                    self.repair_completion_blocked = True
        for source, operations in prepared.items():
            if source in rejected_sources:
                for index, _, _, _ in operations:
                    results[index]["result"] = "[错误] 同一实体存在其他失败项，本次链接替换与确认均未应用"
                continue
            record, document = original[source]
            try:
                body = _document_body(document)
                for _, issue, target_ref, already_valid in operations:
                    if not already_valid:
                        body = _redirect_links(body, self.store, issue["missing_ref"], target_ref)
                if any(not item[3] for item in operations):
                    _validate_entity_protocol(record, body)
                    updated = _render_document(record, body)
                else:
                    updated = document
            except (ValueError, PermissionError) as error:
                for index, issue, _, _ in operations:
                    result = f"[错误] {error}；同一实体本次链接替换与确认均未应用"
                    results[index]["result"] = result
                    self._link_failures[(source, issue["missing_ref"])] = result
                continue
            if updated != document:
                self.pending[source] = (record, updated)
                self.change_count += 1
            if updated != document and self._noop_failures.get(source) == _document_fingerprint(document):
                for failure in self._patch_failures.values():
                    if (not failure["resolved"] and failure["ref"] == source
                            and failure["field"] == "content" and failure["pure_noop"]
                            and failure["source_version"] == _document_fingerprint(document)):
                        failure["resolved"] = True
                self._sync_write_failure(source)
                self._noop_failures.pop(source, None)
            for index, issue, _, already_valid in operations:
                removed_failure = self._link_failures.pop((source, issue["missing_ref"]), None) is not None
                if already_valid:
                    receipt = (source, issue["missing_ref"], issue["fingerprint"])
                    if removed_failure and receipt not in self._confirmed_link_versions:
                        resolved += 1
                    self._confirmed_link_versions.add(receipt)
                elif removed_failure:
                    resolved += 1
                results[index]["result"] = (
                    f"[已暂存] {source} 链接已有效，未修改"
                    if already_valid else f"[已暂存] {source}"
                )
        return _repair_summary(results, self.change_count - changes_before, resolved_count=resolved)

    def manage_entities(self, actions: list[dict]) -> str:
        """只撤销或更名本批新增实体；已有落盘实体不可删除或更名。"""
        if not isinstance(actions, list) or not actions or len(actions) > 8:
            self.repair_completion_blocked = True
            return _repair_summary([{"result": "[错误] actions 必须是最多 8 项的非空数组"}], 0)
        changes_before = self.change_count
        results = []
        for index, item in enumerate(actions):
            entity_id = None
            try:
                if not isinstance(item, dict):
                    raise ValueError("每项必须是 object")
                entity_id = item.get("entity_id")
                if not isinstance(entity_id, str):
                    raise ValueError("entity_id 必须是文本编号")
                source = self._entity_refs.get(entity_id)
                if source not in self.pending or source not in self._new_refs or (self.store.base_dir / source).exists():
                    raise ValueError("只能撤销或更名本批新增、尚未落盘的实体")
                action = item.get("action")
                expected = {"action", "entity_id", "new_name"} if action == "rename" else {"action", "entity_id"}
                if action not in {"rename", "discard"} or set(item) != expected:
                    raise ValueError("action 必须是 rename 或 discard；rename 还必须提供 new_name")
                record, document = self.pending[source]
                if action == "discard":
                    incoming = self._incoming_refs(source)
                    if incoming:
                        raise ValueError("仍被其他实体引用，请先修复引用后撤销：" + ", ".join(incoming[:5]))
                    del self.pending[source]
                    self._new_refs.remove(source)
                    self._entity_ids.pop(source, None)
                    # 保留旧编号槽位，但不再解析为任何目标，避免编号被复用。
                    self._entity_refs[entity_id] = ""
                    self._clear_source_failures(source)
                    self.change_count += 1
                    result = f"[已暂存] 已撤销 {source}"
                else:
                    name = item["new_name"]
                    if not isinstance(name, str):
                        raise ValueError("new_name 必须是文本")
                    _validate_protocol_text(name, "name")
                    target = self.store.entity_ref(record.kind, name.strip(), record.domain)
                    if target == source:
                        raise ValueError("new_name 与原名相同，无需更名")
                    if target in self.pending or (self.store.base_dir / target).exists():
                        raise ValueError("更名目标已存在，禁止覆盖；若已另建正确实体，请核对后撤销误建实体")
                    incoming = self._incoming_refs(source, persisted_only=True)
                    if incoming:
                        raise ValueError("仍有未暂存的已落盘实体引用原名，请先读取并暂存这些实体：" + ", ".join(incoming[:5]))
                    staged = {}
                    changed = 0
                    for ref, (current, text) in self.pending.items():
                        body = _redirect_links(_document_body(text), self.store, source, target)
                        summary = _redirect_links(current.summary, self.store, source, target)
                        updated_record = replace(current, summary=summary)
                        if ref == source:
                            updated_record = replace(updated_record, name=name.strip(), ref=target)
                        if updated_record != current or body != _document_body(text):
                            _validate_entity_protocol(updated_record, body)
                        updated = _render_document(updated_record, body)
                        if updated != text:
                            changed += 1
                        staged[target if ref == source else ref] = (updated_record, updated)
                    self.pending = staged
                    self.change_count += changed
                    self._new_refs.remove(source)
                    self._new_refs.add(target)
                    self._entity_ids.pop(source, None)
                    self._entity_ids[target] = entity_id
                    self._entity_refs[entity_id] = target
                    if source in self._write_failures:
                        self._write_failures[target] = self._write_failures.pop(source)
                    for failure in self._patch_failures.values():
                        if failure["ref"] == source and not failure["resolved"]:
                            failure["ref"] = target
                    self._sync_write_failure(source)
                    self._sync_write_failure(target)
                    self._noop_failures.pop(source, None)
                    for key in list(self._link_failures):
                        if key[0] == source:
                            self._link_failures[(target, key[1])] = self._link_failures.pop(key)
                    result = f"[已暂存] 已更名 {source} -> {target}"
                self._management_failures.pop(entity_id, None)
            except (KeyError, TypeError, ValueError, PermissionError) as error:
                result = f"[错误] {error}"
                source = self._entity_refs.get(entity_id) if isinstance(entity_id, str) else None
                if (
                    source in self.pending and source in self._new_refs
                    and not (self.store.base_dir / source).exists()
                ):
                    self._management_failures[entity_id] = result
                else:
                    self.repair_completion_blocked = True
            results.append({"index": index, "result": result})
        return _repair_summary(results, self.change_count - changes_before)

    def _incoming_refs(self, target: str, *, persisted_only: bool = False) -> list[str]:
        incoming = []
        documents = {} if persisted_only else {ref: text for ref, (_, text) in self.pending.items()}
        for record in self.store.catalog_records():
            if record.ref in self.pending:
                continue
            try:
                path, _ = self.store._resolve_ref(record.ref)
            except (ValueError, PermissionError):
                continue
            if path.is_file():
                documents[record.ref] = path.read_text(encoding="utf-8")
        for ref, text in documents.items():
            if ref == target:
                continue
            for raw_link in _WIKILINK_RE.findall(text):
                try:
                    path, _ = self.store._resolve_ref(raw_link)
                except (ValueError, PermissionError):
                    continue
                if path.relative_to(self.store.base_dir).as_posix() == target:
                    incoming.append(ref)
                    break
        return incoming

    def _clear_source_failures(self, ref: str) -> None:
        self._write_failures.pop(ref, None)
        for failure in self._patch_failures.values():
            if failure["ref"] == ref:
                failure["resolved"] = True
        self.failed_writes.pop(ref, None)
        self._noop_failures.pop(ref, None)
        for key in list(self._link_failures):
            if key[0] == ref:
                self._link_failures.pop(key)

    def read_entity(self, ref: str, section: str = "", field: str = "document") -> str:
        value = str(ref).strip()
        if re.fullmatch(r"E\d+", value):
            value = self._entity_refs.get(value, "")
            if not value:
                return f"[错误] 实体编号无效或已撤销：{ref}；请重新搜索实体"
        if value.startswith("[[") and value.endswith("]]"):
            value = value[2:-2]
        value = value.split("|", 1)[0].strip()
        normalized, _, anchor = value.partition("#")
        try:
            _validate_read_field(field, section, anchor)
        except ValueError as error:
            return f"[错误] {error}"
        if not normalized.endswith(".md"):
            normalized += ".md"
        if normalized in self.pending:
            text = self.pending[normalized][1]
            return _read_document_field(text, value, section, anchor, field)
        return self.store.read_entity(value, section, field)

    def entity_version(self, ref: str) -> str:
        """取得会话可见完整文档的版本；未知或已撤销编号返回空串。"""
        value = str(ref).strip()
        if re.fullmatch(r"E\d+", value):
            value = self._entity_refs.get(value, "")
            if not value:
                return ""
        try:
            path, _ = self.store._resolve_ref(value)
            normalized = path.relative_to(self.store.base_dir).as_posix()
        except (ValueError, PermissionError):
            return ""
        if normalized in self.pending:
            return _document_fingerprint(self.pending[normalized][1])
        return self.store.entity_version(value)

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
        ref = None
        try:
            # 先检查原值，使控制标签中的路径禁用字符不会遮蔽实际污染原因。
            _validate_protocol_text(name, "name")
            _validate_protocol_text(domain, "domain")
            ref = self.store.entity_ref(kind, name.strip(), domain.strip())
            _validate_protocol_text(summary, "summary")
            _validate_protocol_text(content, "content")
            for index, alias in enumerate(aliases or []):
                _validate_protocol_text(str(alias), f"aliases[{index}]")
            if not summary.strip():
                raise ValueError("summary 不能为空，请提供实体摘要后重新提交")
            self._validate_content_links(content)
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
        except (AttributeError, TypeError, ValueError, PermissionError) as error:
            result = f"[错误] {error}"
            self.record_write_failure(ref, result)
            return result
        if self.read_entity(ref) != document:
            self.change_count += 1
        if not (self.store.base_dir / ref).exists():
            self._new_refs.add(ref)
        self.pending[ref] = (record, document)
        self._clear_source_failures(ref)
        return f"[已暂存] {ref}"

    def patch_entity(self, ref: str, replacements: list[dict], field: str = "content") -> str:
        """原子替换正文或摘要；目标姓名可由已查询的实体编号精确取得。"""
        normalized = None
        parsed = []
        parsed_complete = False
        confirmable_error = False
        document = None
        try:
            if not isinstance(ref, str) or not ref.strip():
                raise ValueError("ref 必须是非空实体路径")
            if re.fullmatch(r"E\d+", ref.strip()):
                ref = self._entity_refs.get(ref.strip(), "")
                if not ref:
                    raise ValueError("实体编号无效或已撤销，请重新搜索实体")
            path, anchor = self.store._resolve_ref(ref)
            normalized = path.relative_to(self.store.base_dir).as_posix()
            if anchor:
                raise ValueError("ref 不能包含小节锚点，请提供完整实体路径")
            if normalized not in self.pending:
                raise ValueError(f"只能局部修改本批已暂存实体：{ref}；请先读取并通过 upsert_entities 暂存")
            record, document = self.pending[normalized]
            if field not in ("content", "summary"):
                raise ValueError("field 必须是 content 或 summary")
            if not isinstance(replacements, list) or not replacements:
                raise ValueError("replacements 必须是非空数组")
            known_records = None
            for index, item in enumerate(replacements):
                label = f"replacements[{index}]"
                if not isinstance(item, dict) or set(item) not in (
                    {"old_text", "new_text"}, {"old_text", "new_entity_id"},
                ):
                    raise ValueError(f"{label} 必须包含 old_text，并且只提供 new_text 或 new_entity_id 之一")
                old = item["old_text"]
                if not isinstance(old, str) or not old.strip():
                    raise ValueError(f"{label} 的 old_text 必须是非空文本")
                if "new_entity_id" in item:
                    entity_id = item["new_entity_id"]
                    if not isinstance(entity_id, str):
                        raise ValueError(f"{label} 的 new_entity_id 必须是文本编号")
                    target_ref = self._entity_refs.get(entity_id)
                    if known_records is None:
                        known_records = self._known_records()
                    if target_ref not in known_records:
                        raise ValueError(f"{label} 的目标实体编号无效或目标已不存在，请重新查询")
                    new = known_records[target_ref].name
                else:
                    new = item["new_text"]
                if not isinstance(new, str):
                    raise ValueError(f"{label} 的 new_text 必须是文本")
                parsed.append({"old_text": old, "new_text": new})
            parsed_complete = True
            seen = set()
            for index, item in enumerate(parsed):
                if item["old_text"] in seen:
                    raise ValueError(f"replacements[{index}] 重复指定 old_text：{item['old_text']!r}，请合并重复替换")
                seen.add(item["old_text"])
            original = _document_body(document) if field == "content" else record.summary
            edits = []
            seen = set()
            for index, item in enumerate(parsed):
                label = f"replacements[{index}]"
                old, new = item["old_text"], item["new_text"]
                if old in seen:
                    raise ValueError(f"{label} 重复指定 old_text：{old!r}，请合并重复替换")
                seen.add(old)
                if old == new:
                    actual = json.dumps(item, ensure_ascii=False)
                    raise ValueError(
                        f"{label} 的 old_text 与 new_text 逐字相同，无需替换；实际收到参数：{actual}。"
                        "若无需改动，请删除此项；若确需修字，请逐字核对已读取正文和相关原文，"
                        "将实际 new_text 改为核实后的文本，不能只在说明中声称修改"
                    )
                matches = [match.start() for match in re.finditer(f"(?={re.escape(old)})", original)]
                if len(matches) != 1:
                    confirmable_error = len(matches) == 0
                    location = "正文" if field == "content" else "摘要"
                    read_hint = (
                        "请用 read_entities 指定 field=summary 重新读取当前准确元数据摘要；"
                        f"当前 summary={json.dumps(record.summary, ensure_ascii=False)}；"
                        f"source_version={_document_fingerprint(document)}；"
                        if field == "summary" else "请用 read_entities 重新读取当前已暂存正文，"
                    )
                    raise ValueError(
                        f"{label} 的 old_text 在{location}匹配 {len(matches)} 处，必须恰好匹配 1 处："
                        f"{old!r}；{read_hint}"
                        "从中复制 old_text 并补充足够上下文，不要沿用未匹配的旧片段；"
                        "若完整修改后的文本已存在，可检查 inspect_repairs 的 patch_failures 后显式确认"
                    )
                start = matches[0]
                edits.append((start, start + len(old), new, index))
            edits.sort()
            for previous, current in zip(edits, edits[1:]):
                if current[0] < previous[1]:
                    raise ValueError(
                        f"replacements[{previous[3]}] 与 replacements[{current[3]}] 的匹配重叠，"
                        "请合并成一次替换"
                    )
            result_text = original
            for start, end, new, _ in reversed(edits):
                result_text = result_text[:start] + new + result_text[end:]
            if not result_text.strip():
                raise ValueError(f"替换后 {field} 不能为空")
            self._validate_content_links(result_text)
            if field == "summary":
                record = replace(record, summary=" ".join(result_text.split()))
                body = _document_body(document)
            else:
                body = result_text
            _validate_entity_protocol(record, body)
            updated = _render_document(record, body)
        except (TypeError, ValueError, PermissionError) as error:
            result = (
                f"[错误] {error}\n"
                "本实体这次所有替换均未应用，先前已暂存内容保持原样。"
                "修正后保留本实体其余仍需执行的替换，一起重新提交；不要原样重试，"
                "也不要重复提交同批其他实体已成功的修改。"
            )
            if normalized in self.pending and document is not None:
                failure_id = self._record_patch_failure(
                    normalized, field if isinstance(field, str) else str(field),
                    parsed, document, result,
                    confirmable=parsed_complete and confirmable_error,
                    pure_noop=parsed_complete and _pure_noop_replacements(parsed),
                    parsed_complete=parsed_complete,
                )
                if self._patch_failures[failure_id].get("confirmed"):
                    return f"[已暂存] {normalized} ({field}，请求 {failure_id} 此前已确认满足，本次无变化)"
                if field == "content" and parsed_complete and _pure_noop_replacements(parsed):
                    self._noop_failures[normalized] = _document_fingerprint(document)
                return result + f"\n失败编号：{failure_id}；可用 inspect_repairs 检查该项。"
            self.record_write_failure(normalized, result, staged_only=True)
            return result
        if updated != document:
            self.change_count += 1
        self.pending[normalized] = (record, updated)
        for failure in self._patch_failures.values():
            if (failure["ref"] == normalized and failure["field"] == field
                    and self._successful_patch_resolves(failure, parsed)):
                failure["resolved"] = True
        self._sync_write_failure(normalized)
        if field == "content":
            self._noop_failures.pop(normalized, None)
            for key in list(self._link_failures):
                if key[0] == normalized and not _contains_ref(updated, self.store, key[1]):
                    self._link_failures.pop(key)
        return f"[已暂存] {normalized} ({field})"

    def _sync_write_failure(self, ref: str) -> None:
        errors = []
        if ref in self._write_failures:
            errors.append(self._write_failures[ref])
        errors.extend(
            f"[{failure_id} {failure['field']}] {failure['error']}"
            for failure_id, failure in self._patch_failures.items()
            if failure["ref"] == ref and not failure["resolved"]
        )
        if errors:
            self.failed_writes[ref] = "\n".join(errors)
        else:
            self.failed_writes.pop(ref, None)

    def _record_patch_failure(self, ref, field, replacements, document, error, *, confirmable, pure_noop, parsed_complete):
        version = _document_fingerprint(document)
        key = json.dumps([ref, field, replacements, version, error], ensure_ascii=False, sort_keys=True)
        if parsed_complete:
            normalized = _normalized_patch_request(replacements)
            for previous_id, previous in self._patch_failures.items():
                if (previous.get("confirmed") and previous["ref"] == ref
                        and previous["field"] == field and previous["source_version"] == version
                        and _normalized_patch_request(previous["replacements"]) == normalized):
                    self._patch_failure_keys[key] = previous_id
                    return previous_id
        failure_id = self._patch_failure_keys.get(key)
        if failure_id is None or self._patch_failures[failure_id]["resolved"]:
            failure_id = f"P{len(self._patch_failures) + 1}"
            superseded = []
            if parsed_complete:
                for previous_id, previous in self._patch_failures.items():
                    if (not previous["resolved"] and previous["parsed_complete"]
                            and previous["ref"] == ref and previous["field"] == field
                            and previous["replacements"] == replacements
                            and previous["source_version"] != version):
                        previous["resolved"] = True
                        previous["superseded_by"] = failure_id
                        superseded.append(previous_id)
            self._patch_failure_keys[key] = failure_id
            self._patch_failures[failure_id] = {
                "ref": ref, "field": field, "replacements": replacements,
                "source_version": version, "error": error, "confirmable": confirmable,
                "pure_noop": pure_noop, "parsed_complete": parsed_complete, "resolved": False,
                "supersedes": superseded, "confirmed": False,
            }
        self._sync_write_failure(ref)
        return failure_id

    def _successful_patch_resolves(self, failure: dict, replacements: list[dict]) -> bool:
        failed = failure["replacements"]
        if failure["parsed_complete"] and failed:
            failed_old = {item["old_text"] for item in failed}
            submitted_old = {item["old_text"] for item in replacements}
            if len(failed_old) == len(failed) and failed_old <= submitted_old:
                return True
            if len(failed_old) == len(failed) and all(item["new_text"].strip() for item in failed):
                required = Counter(item["new_text"] for item in failed)
                applied = Counter(item["new_text"] for item in replacements)
                if required <= applied:
                    return True
        return False

    def _patch_postcondition_reason(self, failure: dict, value: str) -> str:
        if not failure["confirmable"] or failure["field"] not in ("content", "summary"):
            return "此失败类型不支持确认，请实际修正参数或完整重新提交实体"
        if not failure["replacements"]:
            return "缺少完整的有效替换参数"
        if len({item["old_text"] for item in failure["replacements"]}) != len(failure["replacements"]):
            return "原请求重复指定 old_text，不支持确认冲突请求"
        if len({item["new_text"] for item in failure["replacements"]}) != len(failure["replacements"]):
            return "多项请求的修改后文本相同，无法唯一对应，不能确认"
        for item in failure["replacements"]:
            old, new = item["old_text"], item["new_text"]
            if not new.strip() or old == new:
                return "删除或无变化替换不支持确认"
            if _shared_patch_context(old, new) < 8:
                return "共同上下文不足 8 个非空白字符，请提供完整上下文或实际修复"
            if old in value:
                return "仍有旧文本，原请求尚未全部满足"
            if sum(1 for _ in re.finditer(f"(?={re.escape(new)})", value)) != 1:
                return "修改后的文本须在目标字段恰好出现一次，当前不存在或存在歧义"
        return ""

    def _patch_confirmation_status(self, failure: dict) -> tuple[bool, str]:
        if failure["resolved"]:
            if failure.get("superseded_by"):
                return False, f"该失败已被新版本请求 {failure['superseded_by']} 替代，请处理新编号"
            return False, "该失败已解除，不能重复确认"
        ref = failure["ref"]
        if ref not in self.pending:
            return False, "来源实体已不存在"
        record, document = self.pending[ref]
        if _document_fingerprint(document) != failure["source_version"]:
            return False, "来源版本已变化，确认请求已过期"
        value = _document_body(document) if failure["field"] == "content" else record.summary
        reason = self._patch_postcondition_reason(failure, value)
        if reason:
            return False, reason
        return True, "所有旧文本均不存在，完整修改后的文本均唯一存在；可显式确认解除此项失败"

    def _describe_patch_failure(self, failure_id: str, failure: dict) -> dict:
        can_confirm, reason = self._patch_confirmation_status(failure)
        return {
            "failure_id": failure_id, "entity_id": self._entity_id(failure["ref"]),
            "ref": failure["ref"], "field": failure["field"],
            "source_version": failure["source_version"],
            "replacements": failure["replacements"],
            "can_confirm": can_confirm, "reason": reason,
            "supersedes": failure.get("supersedes", []),
        }

    def confirm_patch_satisfied(self, failure_ids: list[str]) -> str:
        """显式确认完整修改已存在，仅解除指定失败，不写入实体。"""
        if not isinstance(failure_ids, list) or not failure_ids or len(failure_ids) > 8:
            self.repair_completion_blocked = True
            return "[错误] " + json.dumps({
                "staged_count": 0, "changed_count": 0, "resolved_count": 0,
                "failed_count": 1, "results": [{"result": "[错误] failure_ids 必须是最多 8 项的非空数组"}],
            }, ensure_ascii=False)
        resolved = 0
        results = []
        for index, failure_id in enumerate(failure_ids):
            failure = self._patch_failures.get(failure_id) if isinstance(failure_id, str) else None
            if failure is None:
                self.repair_completion_blocked = True
                results.append({"index": index, "failure_id": failure_id, "result": "[错误] 失败编号不存在，请调用 inspect_repairs"})
                continue
            allowed, reason = self._patch_confirmation_status(failure)
            if not allowed:
                results.append({"index": index, "failure_id": failure_id, "result": "[错误] " + reason})
                continue
            failure["resolved"] = True
            failure["confirmed"] = True
            self._sync_write_failure(failure["ref"])
            resolved += 1
            results.append({"index": index, "failure_id": failure_id,
                            "result": f"[已确认] {failure_id} 的补丁后置条件已满足，未修改实体"})
        failed = len(results) - resolved
        result = json.dumps({"staged_count": 0, "changed_count": 0,
                             "resolved_count": resolved, "failed_count": failed, "results": results}, ensure_ascii=False)
        return "[错误] " + result if failed else result

    def record_write_failure(self, ref: str | None, error: str, *, staged_only: bool = False) -> None:
        """可定位的失败须由同一实体成功写入解除，其余失败禁止自动收尾。"""
        try:
            if not isinstance(ref, str) or not ref.strip():
                raise ValueError("未提供实体路径")
            path, _ = self.store._resolve_ref(ref)
            normalized = path.relative_to(self.store.base_dir).as_posix()
            if staged_only and normalized not in self.pending:
                raise ValueError("无法定位待修复的暂存实体")
        except (TypeError, ValueError, PermissionError):
            self.repair_completion_blocked = True
        else:
            self._write_failures[normalized] = error
            self._sync_write_failure(normalized)
            self._noop_failures.pop(normalized, None)

    def _validate_content_links(self, content: str) -> None:
        for raw_link in _WIKILINK_RE.findall(content):
            linked = raw_link.split("|", 1)[0].split("#", 1)[0].strip()
            if not linked.startswith(("characters/", "settings/", "plot/events/", "plot/arcs/")):
                raise ValueError(f"链接必须使用完整知识库路径：{raw_link}；例如 [[characters/人物名]]")
            self.store._resolve_ref(linked)

    def validate(self) -> list[str]:
        errors = [
            f"{ref}: 尚有未解决的写入失败，请修正参数并重新提交该实体：{error}"
            for ref, error in sorted(self.failed_writes.items())
        ]
        errors.extend(
            f"{ref}: 尚有未解决的链接修复失败（{missing_ref}）：{error}"
            for (ref, missing_ref), error in sorted(self._link_failures.items())
        )
        errors.extend(
            f"{entity_id}: 尚有未解决的实体管理失败：{error}"
            for entity_id, error in sorted(self._management_failures.items())
        )
        chapter_files = (self.store.base_dir / "chapters").glob("*.txt")
        available_chapters = {int(path.stem) for path in chapter_files if path.stem.isdigit()}
        pending_refs = set(self.pending)
        for ref, (record, document) in self.pending.items():
            try:
                _validate_entity_protocol(record, _document_body(document))
            except ValueError as error:
                errors.append(f"{ref}: {error}")
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
        if not errors:
            return None
        return "\n".join(f"- {error}" for error in errors) + "\n可用的结构化修复信息：\n" + self.inspect_repairs()

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


def _validate_protocol_text(value: str, field: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{field} 必须是文本")
    match = _MODEL_PROTOCOL_RE.search(value)
    if match:
        raise ValueError(
            f"{field} 含模型协议特殊标记 {match.group(0)!r}；"
            "请重新读取相关原文，按原文重新生成该字段，不要原样重试或直接剥除标记"
        )


def _validate_entity_protocol(record: EntityRecord, content: str) -> None:
    for field, value in (("name", record.name), ("summary", record.summary),
                         ("content", content), ("domain", record.domain)):
        _validate_protocol_text(value, field)
    for index, alias in enumerate(record.aliases):
        _validate_protocol_text(alias, f"aliases[{index}]")


def _validate_read_field(field: str, section: str, anchor: str) -> None:
    if field not in ("document", "content", "summary"):
        raise ValueError("field 必须是 document、content 或 summary")
    if not isinstance(section, str):
        raise ValueError("section 必须是文本")
    if field == "summary" and (section.strip() or anchor.strip()):
        raise ValueError("field=summary 不能同时指定 section 或 ref 小节锚点")


def _read_document_field(text: str, ref: str, section: str, anchor: str, field: str) -> str:
    wanted = section.strip() or anchor.strip()
    metadata = _META_RE.search(text)
    if field == "summary" or (field == "document" and section.strip() == "summary"
                              and _extract_section(text, "summary") is None):
        if metadata:
            try:
                summary = json.loads(metadata.group(1)).get("summary")
                if isinstance(summary, str) and summary.strip():
                    return summary
            except (json.JSONDecodeError, AttributeError):
                pass
        return f"[错误] 实体 {ref} 无可读取的元数据摘要（旧版知识库可能未保存 summary）；请读取 document 或 content"
    value = _document_body(text) if field == "content" and metadata else text
    if wanted:
        extracted = _extract_section(value, wanted)
        return extracted if extracted is not None else f"[错误] 实体 {ref} 中不存在章节：{wanted}"
    return value


def _document_body(document: str) -> str:
    _, _, remainder = document.partition("\n\n")
    body, _, _ = remainder.rpartition("\n\n来源章节：")
    return body


def _document_fingerprint(document: str) -> str:
    return hashlib.sha256(document.encode("utf-8")).hexdigest()


def _shared_patch_context(old: str, new: str) -> int:
    prefix = 0
    for left, right in zip(old, new):
        if left != right:
            break
        prefix += 1
    suffix = 0
    for left, right in zip(reversed(old[prefix:]), reversed(new[prefix:])):
        if left != right:
            break
        suffix += 1
    context = old[:prefix] + (old[len(old) - suffix:] if suffix else "")
    return sum(not char.isspace() for char in context)


def _normalized_patch_request(replacements: list[dict]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((item["old_text"], item["new_text"]) for item in replacements))


def _pure_noop_replacements(replacements) -> bool:
    return (
        isinstance(replacements, list) and bool(replacements)
        and all(
            isinstance(item, dict) and set(item) == {"old_text", "new_text"}
            and isinstance(item["old_text"], str) and bool(item["old_text"].strip())
            and item["old_text"] == item["new_text"]
            for item in replacements
        )
        and len({item["old_text"] for item in replacements}) == len(replacements)
    )


def _kind_from_ref(ref: str) -> str:
    for prefix, kind in (("characters/", "character"), ("settings/", "setting"),
                         ("plot/events/", "event"), ("plot/arcs/", "arc")):
        if ref.startswith(prefix):
            return kind
    return ""


def _contains_ref(text: str, store: KnowledgeStore, ref: str) -> bool:
    for raw in _WIKILINK_RE.findall(text):
        try:
            path, _ = store._resolve_ref(raw)
            if path.relative_to(store.base_dir).as_posix() == ref:
                return True
        except (ValueError, PermissionError):
            continue
    return False


def _redirect_links(text: str, store: KnowledgeStore, old_ref: str, new_ref: str) -> str:
    def redirect(match):
        raw = match.group(1)
        target, separator, alias = raw.partition("|")
        reference, anchor_separator, anchor = target.partition("#")
        try:
            path, _ = store._resolve_ref(reference)
        except (ValueError, PermissionError):
            return match.group(0)
        if path.relative_to(store.base_dir).as_posix() != old_ref:
            return match.group(0)
        replacement = new_ref if reference.strip().endswith(".md") else new_ref.removesuffix(".md")
        return "[[" + replacement + anchor_separator + anchor + separator + alias + "]]"
    return _WIKILINK_RE.sub(redirect, text)


def _repair_summary(results: list[dict], changed: int, *, resolved_count: int | None = None) -> str:
    staged = sum(item["result"].startswith("[已暂存]") for item in results)
    failed = len(results) - staged
    summary = {"staged_count": staged, "changed_count": changed,
               "failed_count": failed, "results": results}
    if resolved_count is not None:
        summary["resolved_count"] = resolved_count
    payload = json.dumps(summary, ensure_ascii=False)
    return "[错误] " + payload if failed else payload


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
