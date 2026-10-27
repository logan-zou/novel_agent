"""小说知识构建：一个抽取 Agent + 领域化实体工具 + 批次暂存。"""

import json
import re
from pathlib import Path

from src.config import (
    EXTRACTION_MODEL,
    EXTRACTION_PROMPT_PATH,
    EXTRACTION_REASONING_EFFORT,
    MODEL_CONTEXT_TOKENS,
    MODEL_MAX_OUTPUT_TOKENS,
    MODEL_TEMPERATURE,
    require_llm_model,
)
from src.llm import get_provider
from src.novel.corpus import CorpusStore
from src.novel.knowledge import BuildSession, KnowledgeStore
from src.harness.runtime import AgentConfig, AgentRuntime, ChatProvider, Tool
from src.tracing import create_trace, finish_trace
from src.usage_tracker import record as record_usage


_MAX_BATCH_TOOL_ITEMS = 8
_MAX_BATCH_READ_CHARS = 60_000


def run_extraction_agent(
    batch: list[dict],
    novel_id: str,
    base_dir: str,
    debug: bool = False,
    provider: ChatProvider | None = None,
    model: str | None = None,
) -> dict:
    selected_model = require_llm_model(model or EXTRACTION_MODEL, "EXTRACTION_MODEL")
    knowledge = KnowledgeStore(base_dir)
    corpus = CorpusStore(base_dir)
    session = BuildSession(knowledge)
    completion_state: dict = {"phase": "extraction"}

    def take_completion() -> str | None:
        return completion_state.pop("summary", None)

    def take_continuation() -> str | None:
        return completion_state.pop("continuation_request", None)
    first, last = batch[0]["chapter_no"], batch[-1]["chapter_no"]
    label = f"batch_ch{first}-{last}"
    trace = create_trace(
        "extraction",
        metadata={"novel_id": novel_id, "chapter_range": [first, last], "model": selected_model},
    )

    def on_event(event_type: str, data: dict) -> None:
        if event_type == "model_started":
            completion_state["write_scope"] = (
                _finalization_write_scope(session)
                if completion_state["phase"] == "finalization" else None
            )
        if debug:
            print(f"[Agent:{event_type}] {json.dumps(data, ensure_ascii=False, default=str)}", flush=True)
        if event_type == "tool_completed":
            _log_tool(data["name"], data.get("arguments", {}), data.get("result", ""))
        elif event_type == "tool_arguments_rejected" and data.get("will_retry"):
            detail = (
                "工具参数缺失或无效，已保留本轮成功暂存项，"
                if data.get("executed") else "工具参数 JSON 无效，本轮未执行，"
            )
            print(
                f"    ⚠️  {detail}"
                f"正在自动重试（{data['attempt']}/{data['max_retries']}）"
            )
        elif event_type == "context_compacted":
            print(f"    ✂️  已省略 {data['messages_compacted']} 条较早工具结果")
        elif event_type == "output_rejected" and data.get("will_retry"):
            print(
                f"    ⚠️  知识校验未通过，正在修复（{data['attempt']}/{data['max_retries']}）："
                f"{data['reason']}", flush=True,
            )
        elif event_type == "validation_repair_checked" and data.get("passed"):
            print("    ✅ 修复后知识校验通过，结束修复并提交本批次。", flush=True)
        elif event_type == "phase_started":
            phase = "独立收尾" if data.get("phase") == "finalization" else "同章续跑"
            print(f"    ⏳ 进入{phase}窗口，保留当前暂存内容与上下文。", flush=True)
        elif debug and event_type == "model_completed" and data.get("finish_reason") == "length":
            print("    ⚠️  模型输出被截断，正在恢复")

    try:
        result = AgentRuntime(provider or get_provider()).run(
            AgentConfig(
                name="novel-extraction",
                model=selected_model,
                system_prompt=Path(EXTRACTION_PROMPT_PATH).read_text(encoding="utf-8"),
                max_iterations=60,
                max_validation_retries=5,
                validation_repair_iterations=8,
                finish_after_validation_repair=True,
                max_tool_argument_retries=5,
                max_output_tokens=MODEL_MAX_OUTPUT_TOKENS,
                temperature=MODEL_TEMPERATURE,
                context_window_tokens=MODEL_CONTEXT_TOKENS,
                reasoning_effort=EXTRACTION_REASONING_EFFORT,
                soft_iteration_limit=40,
                progress_grace_iterations=8,
                max_identical_tool_calls=3,
                max_consecutive_tool_errors=3,
                finalization_reserve_iterations=8,
                finalization_iterations=8,
                continuation_iterations=24,
                max_continuations=1,
            ),
            [{"role": "user", "content": format_batch(batch)}],
            _builder_tools(session, corpus, completion_state),
            on_event=on_event,
            usage_recorder=lambda usage: record_usage(
                novel_id, "extraction", label, selected_model, usage
            ),
            output_validator=session.validator,
            can_finish_after_validation_repair=lambda: not session.repair_completion_blocked,
            budget_feedback=lambda step, remaining: _finalization_context(session, step, remaining),
            completion_requested=take_completion,
            continuation_requested=take_continuation,
            phase_feedback=lambda phase, step, remaining: _phase_context(
                session, completion_state, phase, step, remaining,
            ),
        )
        report = session.commit()
        finish_trace(trace, {"message": f"完成第 {first}-{last} 章抽取", **report})
        if debug and result.output:
            print(f"\n🤖 Agent 最终输出\n{result.output}")
        return result.usage
    except Exception as error:
        finish_trace(trace, f"[错误] {error}")
        raise


def format_batch(batch: list[dict]) -> str:
    parts = [f"处理以下 {len(batch)} 章。只记录值得长期检索的知识，并完成所有实体暂存。"]
    for chapter in batch:
        parts.append(f"---\n【第{chapter['chapter_no']}章 {chapter['title']}】\n\n{chapter['text']}")
    return "\n".join(parts)


def _builder_tools(
    session: BuildSession, corpus: CorpusStore, completion_state: dict | None = None,
) -> list[Tool]:
    if completion_state is None:
        completion_state = {}
    return [
        Tool(
            "search_entities",
            "批量搜索已有实体，精确名称和别名优先。默认只返回全查询前 30 项，不是每词 30 项。"
            "查看 keyword_matches、uncovered_keywords 和 unmatched_keywords，区分未命中与被截断。"
            "先读取已命中的主体，仅将未覆盖词拆成少量或单独查询；不要反复提交相同大查询。",
            {
                "type": "object",
                "properties": {
                    "keywords": {"type": "array", "minItems": 1, "items": {"type": "string"}},
                    "entity_types": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer", "default": 30, "minimum": 1, "maximum": 100},
                },
                "required": ["keywords"],
            },
            session.search_entities,
            recovery_handler=lambda arguments: _search_recovery_context(session, arguments),
            recover_successful_repetition=True,
        ),
        Tool(
            "read_entities",
            "批量读取已有或本批已暂存实体。一次最多读取 8 个；更新实体前必须在这里取得旧正文。"
            "field=summary 精确读取当前摘要及版本，field=content 只读正文，默认 document 返回完整文档。"
            "section 仅用于正文小节；摘要不是 Markdown 小节，读取摘要时不要同时提供 section。",
            {
                "type": "object",
                "properties": {
                    "requests": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": _MAX_BATCH_TOOL_ITEMS,
                        "items": {
                            "type": "object",
                            "properties": {
                                "ref": {"type": "string"},
                                "field": {"type": "string", "enum": ["document", "summary", "content"], "default": "document"},
                                "section": {"type": "string", "description": "正文小节标题；不能与 field=summary 同时使用"},
                            },
                            "required": ["ref"],
                        },
                    }
                },
                "required": ["requests"],
            },
            lambda requests: _read_entities(session, requests),
        ),
        Tool(
            "upsert_entities",
            "批量新增或完整更新实体。一次最多提交 8 个；系统负责路径、Markdown 外壳和 Catalog。",
            {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": _MAX_BATCH_TOOL_ITEMS,
                        "items": {
                            "type": "object",
                            "properties": {
                                "kind": {
                                    "type": "string",
                                    "enum": ["character", "setting", "event", "arc"],
                                },
                                "name": {"type": "string"},
                                "aliases": {"type": "array", "items": {"type": "string"}},
                                "summary": {"type": "string"},
                                "content": {"type": "string"},
                                "source_chapters": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                },
                                "domain": {
                                    "type": "string",
                                    "description": "仅 setting 必填，如 力量体系",
                                },
                            },
                            "required": [
                                "kind",
                                "name",
                                "summary",
                                "content",
                                "source_chapters",
                            ],
                        },
                    }
                },
                "required": ["entities"],
            },
            lambda entities=None, **unexpected: _upsert_with_budget_feedback(
                session, entities, unexpected, completion_state,
            ),
            progress_detector=_has_staged_entities,
            argument_error_detector=_has_argument_errors,
        ),
        Tool(
            "patch_entities",
            "局部修复本批已暂存实体，一次最多 8 项。field 默认 content，可设 summary 仅修摘要，避免重写正文。"
            "每个 old_text 必须在指定字段唯一出现；"
            "new_text 必须与 old_text 不同。同一实体所有替换一起校验后应用，不会串联替换。"
            "失败实体本次全部替换未应用，须按反馈修正参数后重试；同批其他成功实体保留。"
            "修正姓名可用 new_entity_id 选择已核对原文的实体，由代码填入正式名称，与 new_text 二选一。"
            "指定字段以外的正文、元数据和来源保持原样。",
            {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": _MAX_BATCH_TOOL_ITEMS,
                        "items": {
                            "type": "object",
                            "properties": {
                                "ref": {"type": "string"},
                                "field": {"type": "string", "enum": ["content", "summary"], "default": "content"},
                                "replacements": {
                                    "type": "array",
                                    "minItems": 1,
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "old_text": {"type": "string"},
                                            "new_text": {"type": "string"},
                                            "new_entity_id": {"type": "string", "description": "从实体编号复制姓名，与 new_text 二选一"},
                                        },
                                        "required": ["old_text"],
                                        "additionalProperties": False,
                                    },
                                },
                            },
                            "required": ["ref", "replacements"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["entities"],
            },
            lambda entities=None, **unexpected: _patch_entities(session, entities, unexpected),
            progress_detector=_has_staged_entities,
            argument_error_detector=_has_argument_errors,
            recovery_handler=lambda arguments: _patch_recovery_context(session, arguments),
        ),
        Tool(
            "inspect_repairs",
            "查看本批暂存实体编号、缺失链接问题编号及已有目标候选。链接修复优先使用编号，"
            "link_failures 列出历史失败的核验入口；不要反复输入近形字路径。候选仅供选择，不会自动修改。",
            {"type": "object", "properties": {}, "additionalProperties": False},
            session.inspect_repairs,
        ),
        Tool(
            "confirm_patch_satisfied",
            "确认失败补丁的目标状态已存在，解除指定 failure_id，不重写实体。"
            "先重读并从 inspect_repairs 的 patch_failures 取得可确认编号；"
            "程序核验版本、完整上下文和目标文本。短词、删除、歧义、过期均拒绝。每次最多 8 项。",
            {
                "type": "object", "properties": {
                    "failure_ids": {"type": "array", "minItems": 1, "maxItems": _MAX_BATCH_TOOL_ITEMS,
                                    "items": {"type": "string"}},
                },
                "required": ["failure_ids"], "additionalProperties": False,
            },
            lambda failure_ids=None, **unexpected: _structured_write(session, "failure_ids", failure_ids, unexpected),
            progress_detector=_has_repair_progress,
            argument_error_detector=_has_argument_errors,
        ),
        Tool(
            "finish_batch",
            "明确声明本批所有值得记录的知识已抽取完成，请求校验并提交。"
            "summary 简短说明处理结果。程序会在整轮工具完成后检查全部失败项和知识合法性；"
            "未通过则继续定点修复，不能用结构合法代替抽取完成。",
            {"type": "object", "properties": {"summary": {"type": "string"}},
             "required": ["summary"], "additionalProperties": False},
            lambda summary=None, **unexpected: _request_completion(completion_state, summary, unexpected),
            argument_error_detector=_has_argument_errors,
        ),
        Tool(
            "continue_batch",
            "在独立收尾窗口明确声明本批仍有未抽取的知识，请求保留当前暂存内容和上下文继续处理。"
            "remaining_tasks 列出具体剩余任务；程序在整轮成功后决定是否批准，最多续跑一次。"
            "续跑不是完成声明，不能和 finish_batch 同轮使用；普通抽取阶段无需申请。",
            {"type": "object", "properties": {
                "remaining_tasks": {"type": "array", "minItems": 1, "maxItems": 16,
                                    "items": {"type": "string", "minLength": 1, "maxLength": 240}},
            }, "required": ["remaining_tasks"], "additionalProperties": False},
            lambda remaining_tasks=None, **unexpected: _request_continuation(
                completion_state, remaining_tasks, unexpected,
            ),
            argument_error_detector=_has_argument_errors,
        ),
        Tool(
            "repair_links",
            "按问题编号和目标实体编号精确修复本批缺失链接，代码复制真实路径并保留链接别名和锚点。"
            "先从 inspect_repairs 或校验反馈取得编号；原目标已创建时，使用 link_failures 中精确对应的"
            " target_id 核验并解除该项历史失败，不重写实体。来源版本过期时重新检查。每次最多 8 项。",
            {
                "type": "object",
                "properties": {
                    "repairs": {
                        "type": "array", "minItems": 1, "maxItems": _MAX_BATCH_TOOL_ITEMS,
                        "items": {
                            "type": "object",
                            "properties": {
                                "issue_id": {"type": "string"},
                                "target_id": {"type": "string"},
                            },
                            "required": ["issue_id", "target_id"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["repairs"],
                "additionalProperties": False,
            },
            lambda repairs=None, **unexpected: _structured_write(session, "repairs", repairs, unexpected),
            progress_detector=_has_repair_progress,
            argument_error_detector=_has_argument_errors,
        ),
        Tool(
            "manage_entities",
            "撤销或更名本批新建的暂存实体，使用 inspect_repairs 返回的 entity_id。"
            "已有落盘实体不能操作；rename 提供 new_name 并更新本批精确引用，名称冲突拒绝覆盖；"
            "discard 仅能撤销无入链的新实体。改名请用此工具，不要另建一份。每次最多 8 项。",
            {
                "type": "object",
                "properties": {
                    "actions": {
                        "type": "array", "minItems": 1, "maxItems": _MAX_BATCH_TOOL_ITEMS,
                        "items": {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string", "enum": ["rename", "discard"]},
                                "entity_id": {"type": "string"},
                                "new_name": {"type": "string", "description": "rename 时必填"},
                            },
                            "required": ["action", "entity_id"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["actions"],
                "additionalProperties": False,
            },
            lambda actions=None, **unexpected: _structured_write(session, "actions", actions, unexpected),
            progress_detector=_has_staged_entities,
            argument_error_detector=_has_argument_errors,
        ),
        Tool(
            "search_source",
            "搜索全书原文，仅用于核实静态设定；不得据此提前更新未来角色状态或情节。",
            {
                "type": "object",
                "properties": {
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "mode": {"type": "string", "enum": ["any", "all"]},
                    "chapter_start": {"type": "integer"},
                    "chapter_end": {"type": "integer"},
                    "order": {"type": "string", "enum": ["asc", "desc"]},
                    "cursor": {"type": "integer"},
                    "limit": {"type": "integer"},
                },
                "required": ["keywords"],
            },
            corpus.search,
        ),
        Tool(
            "read_chapter",
            "读取一个完整章节，用于核实搜索片段。",
            {
                "type": "object",
                "properties": {"chapter_no": {"type": "integer"}},
                "required": ["chapter_no"],
            },
            corpus.read_chapter,
        ),
    ]


def _log_tool(name: str, arguments: dict, result: str) -> None:
    if name in {"upsert_entities", "patch_entities"}:
        entities = arguments.get("entities")
        if not isinstance(entities, list):
            entities = []
        names = [str(item.get("name", item.get("ref", "?"))) for item in entities if isinstance(item, dict)]
        summary = _json_object(result)
        staged = summary.get("staged_count", 0)
        failed = summary.get("failed_count", 0)
        action = "局部修复" if name == "patch_entities" else "批量暂存"
        print(f"    ✏️  {action} {staged} 个实体（失败 {failed}）：{', '.join(names)}")
    elif name in {"repair_links", "manage_entities", "confirm_patch_satisfied"}:
        summary = _json_object(result)
        action = {"repair_links": "编号修复链接", "manage_entities": "整理本批实体",
                  "confirm_patch_satisfied": "核验已满足的补丁"}[name]
        succeeded = summary.get("resolved_count", 0) if name == "confirm_patch_satisfied" else summary.get("staged_count", 0)
        resolved = f"，新增修复进展 {summary.get('resolved_count', 0)} 项" if name == "repair_links" else ""
        print(f"    ✏️  {action}：成功 {succeeded} 项{resolved}，失败 {summary.get('failed_count', 0)} 项")
    elif name == "finish_batch":
        print("    🏁 已请求本批收尾校验" if not result.startswith("[错误]") else "    ⚠️  收尾声明参数无效")
    elif name == "continue_batch":
        print("    ⏳ 已请求处理明确剩余任务，等待整轮结果。" if not result.startswith("[错误]") else "    ⚠️  续跑请求未接受")
    elif name == "inspect_repairs":
        summary = _json_object(result)
        print(f"    🔍 待修复链接 {len(summary.get('issues', []))} 项，"
              f"链接修复失败 {len(summary.get('link_failures', []))} 项，补丁失败 {len(summary.get('patch_failures', []))} 项")
    elif name == "read_entities":
        requests = arguments.get("requests")
        if not isinstance(requests, list):
            requests = []
        print(f"    📖 批量读取 {len(requests)} 个实体（返回 {len(result)} 字）")
    elif name in {"search_entities", "search_source"}:
        print(f"    🔍 {name} {arguments.get('keywords')} ({len(result)} 字)")
    elif name == "read_chapter":
        print(f"    📖 第{arguments.get('chapter_no')}章 ({len(result)} 字)")


def _read_entities(session: BuildSession, requests) -> str:
    if not isinstance(requests, list) or not requests:
        return "[错误] requests 必须是非空数组"
    results = []
    used_chars = 0
    deferred = 0
    failed = 0
    for item in requests:
        if not isinstance(item, dict) or not isinstance(item.get("ref"), str) or not item["ref"].strip():
            results.append({"ref": "", "error": "每一项都必须包含 ref"})
            failed += 1
            continue
        ref = item["ref"]
        field = item.get("field", "document")
        section = item.get("section", "")
        if field not in ("document", "summary", "content") or not isinstance(section, str):
            results.append({"ref": ref, "error": "field 必须是 document、summary 或 content，section 必须是文本"})
            failed += 1
            continue
        content = session.read_entity(ref, section, field=field)
        if content.startswith("[错误]"):
            results.append({"ref": ref, "field": field, "error": content})
            failed += 1
            continue
        summary_read = field == "summary" or (
            field == "document" and section.strip() == "summary"
            and content == session.read_entity(ref, field="summary")
        )
        size = len(content) * (2 if summary_read else 1)
        if used_chars + size > _MAX_BATCH_READ_CHARS:
            results.append(
                {
                    "ref": ref,
                    "error": "本次批量读取已达到 60000 字上限，请在下一轮再次读取该实体",
                }
            )
            deferred += 1
            continue
        entry = {"ref": ref, "field": "summary" if summary_read else field, "content": content}
        if summary_read:
            version = session.entity_version(ref)
            if not version:
                results.append({"ref": ref, "field": "summary", "error": "来源文档已不可读取，请重新读取摘要和版本"})
                failed += 1
                continue
            entry.update({"summary": content, "source_version": version})
        results.append(entry)
        used_chars += size
    payload = json.dumps(
        {"entities": results, "returned_chars": used_chars, "deferred_count": deferred, "failed_count": failed},
        ensure_ascii=False,
    )
    return "[错误] " + payload if failed else payload


def _search_recovery_context(session: BuildSession, arguments: dict) -> str:
    payload = _json_object(session.search_entities(
        arguments.get("keywords", []),
        entity_types=arguments.get("entity_types"),
        limit=arguments.get("limit", 30),
    ))
    matches = payload.get("matches", [])
    uncovered = payload.get("uncovered_keywords", [])
    unmatched = payload.get("unmatched_keywords", [])
    return json.dumps({
        "instruction": (
            "这是成功但未带来新信息的重复检索，不要再次原样提交或只调整关键词顺序。"
            "先根据本批原文确认下方主体，用 read_entities 批量读取已命中的 ref，每次最多 8 个，随后完成抽取。"
            "若仍缺已有主体，仅对 keywords_to_narrow 中的词少量或单独补查，保留原 entity_types 范围；"
            "不要把已查到的名字和宽泛国家名重新混回大查询。"
            "unmatched_keywords 只表示当前目录名称、别名和摘要没有命中，不证明小说不存在该对象；"
            "原文依据明确且确认没有已有主体的新知识可正常抽取。"
            "本反馈不算写入进展，不延长预算，也不解除重复调用保护。"
        ),
        "total_matches": payload.get("total_matches", len(matches)),
        "returned_count": len(matches), "truncated": payload.get("truncated", False),
        "ready_to_read": [
            {key: item[key] for key in ("entity_id", "ref", "name") if key in item}
            for item in matches[:_MAX_BATCH_TOOL_ITEMS]
        ],
        "other_returned_count": max(0, len(matches) - _MAX_BATCH_TOOL_ITEMS),
        "keywords_to_narrow": uncovered[:_MAX_BATCH_TOOL_ITEMS],
        "other_uncovered_count": max(0, len(uncovered) - _MAX_BATCH_TOOL_ITEMS),
        "unmatched_keywords": unmatched[:16],
        "other_unmatched_count": max(0, len(unmatched) - 16),
    }, ensure_ascii=False)


def _patch_recovery_context(session: BuildSession, arguments: dict) -> str:
    current = []
    remaining = 24_000
    entities = arguments.get("entities")
    seen = set()
    for item in entities[:_MAX_BATCH_TOOL_ITEMS] if isinstance(entities, list) else []:
        if not isinstance(item, dict) or not isinstance(item.get("ref"), str):
            continue
        ref = item["ref"]
        if ref in seen or remaining <= 0:
            continue
        seen.add(ref)
        text = session.read_entity(ref)
        excerpt = text[:min(remaining, 12_000)]
        current.append({"ref": ref, "content": excerpt, "truncated": len(excerpt) < len(text)})
        remaining -= len(excerpt)
    return json.dumps({
        "instruction": (
            "相同补丁已连续失败，请切换修复路径。下方是当前已暂存正文及可用编号。"
            "缺失链接请用 repair_links 选择 issue_id 和 target_id；"
            "本批误建实体请用 manage_entities 更名或撤销；"
            "目标文本本就存在时，请重读并用 confirm_patch_satisfied 核验可确认的失败编号；"
            "只改摘要时请用 patch_entities 的 field=summary；"
            "其他文字修改须根据当前正文和原文重新核实参数。不要再次原样调用。"
            "正文被截断时可用 read_entities 指定小节读取；读取不算写入进展。"
        ),
        "current_entities": current,
        "repair_options": _json_object(session.inspect_repairs()),
    }, ensure_ascii=False)


def _structured_write(session: BuildSession, field: str, items, unexpected: dict) -> str:
    invalid = {key: "不支持的工具参数" for key in unexpected}
    if not isinstance(items, list) or not items or len(items) > _MAX_BATCH_TOOL_ITEMS:
        invalid[field] = f"必须是非空数组，每次最多 {_MAX_BATCH_TOOL_ITEMS} 项"
    if invalid:
        details = _argument_error_details([], invalid, [field])
        result = _argument_error_message(details)
        session.record_write_failure(None, result)
        return "[错误] " + json.dumps({**details, "message": result}, ensure_ascii=False)
    handler = {"repairs": session.repair_links, "actions": session.manage_entities,
               "failure_ids": session.confirm_patch_satisfied}[field]
    return handler(items)


def _request_completion(state: dict, summary, unexpected: dict) -> str:
    invalid = {key: "不支持的工具参数" for key in unexpected}
    if not isinstance(summary, str) or not summary.strip():
        invalid["summary"] = "必须是非空文本"
    if invalid:
        details = _argument_error_details([], invalid, ["summary"])
        return "[错误] " + json.dumps({**details, "message": _argument_error_message(details)}, ensure_ascii=False)
    state["summary"] = summary.strip()
    return json.dumps({"completion_requested": True, "feedback": "已声明抽取完成，等待整轮工具结束后统一校验。"}, ensure_ascii=False)


def _request_continuation(state: dict, remaining_tasks, unexpected: dict) -> str:
    invalid = {key: "不支持的工具参数" for key in unexpected}
    if (not isinstance(remaining_tasks, list) or not 1 <= len(remaining_tasks) <= 16
            or any(not isinstance(task, str) or not task.strip() or len(task) > 240 for task in remaining_tasks)):
        invalid["remaining_tasks"] = "必须是 1 至 16 项非空任务文本，每项最多 240 字"
    if invalid:
        details = _argument_error_details([], invalid, ["remaining_tasks"])
        return "[错误] " + json.dumps({**details, "message": _argument_error_message(details)}, ensure_ascii=False)
    if state.get("phase") != "finalization":
        return "[错误] " + json.dumps({
            "error_type": "invalid_state", "message": "当前仍是普通抽取阶段，请继续完成任务；进入独立收尾窗口后才能申请续跑。",
        }, ensure_ascii=False)
    tasks = [task.strip() for task in remaining_tasks]
    state["remaining_tasks"] = tasks
    state["continuation_request"] = json.dumps({"remaining_tasks": tasks}, ensure_ascii=False)
    return json.dumps({
        "continuation_requested": True,
        "feedback": "已列明剩余任务，等待整轮工具成功后批准续跑；尚未完成本批。",
    }, ensure_ascii=False)


def _finalization_write_scope(session: BuildSession) -> set[str]:
    """只在模型轮次开始时生成快照，整轮不会因新增引用而扩大权限。"""
    allowed = set(session.pending)
    for _, document in session.pending.values():
        for raw_link in re.findall(r"\[\[([^\]]+)\]\]", document):
            try:
                path, _ = session.store._resolve_ref(raw_link)
                target_ref = path.relative_to(session.store.base_dir).as_posix()
            except (ValueError, PermissionError):
                continue
            if target_ref not in session.pending and not path.is_file():
                allowed.add(target_ref)
    return allowed


def _phase_context(session: BuildSession, state: dict, phase: str, step: int, remaining: int) -> str:
    state["phase"] = phase
    instruction = (
        "已进入独立收尾窗口，常规预算不会挤占此窗口。先修复下方明确问题，"
        "批量补齐缺失链接目标、更新已暂存实体；本轮新建目标以轮次开始前的缺失清单为准。"
        "把短小且独立的目标合并到一次 upsert_entities 的 entities 数组，最多 8 项。"
        "若仍有其他值得记录的知识未抽取，必须用 continue_batch 列出具体 remaining_tasks 请求续跑，"
        "不要漏掉这些知识直接声明完成。全部抽取和修复确已完成才调用 finish_batch。"
        if phase == "finalization" else
        "已批准一次同章续跑，当前暂存实体、工具历史和错误保护均保留。"
        "请批量完成列明的剩余任务，避免重复改写已成功内容；本窗口用完后还有独立收尾窗口。"
        "全部任务确已完成后仍须 finish_batch 声明并通过校验，不得沿用先前的完成声明。"
    )
    return json.dumps({
        "stage": phase, "step": step, "remaining_phase_iterations": remaining,
        "remaining_tasks": state.get("remaining_tasks", []),
        "staged_entities": len(session.pending), "instruction": instruction,
        "repair_options": _json_object(session.inspect_repairs()),
    }, ensure_ascii=False)


def _finalization_context(session: BuildSession, step: int, remaining: int) -> str:
    return json.dumps({
        "stage": "regular_budget_warning", "step": step, "remaining_regular_iterations": remaining,
        "instruction": (
            "已进入本批常规预算的最后阶段。请集中完成明确的剩余事项，优先处理下方失败清单，"
            "不要继续扩展无界检查。将剩余独立实体合并到一次 upsert_entities 的 entities 数组，每次最多 8 项。"
            "常规预算结束后还有独立收尾窗口；若仍有未抽取知识，可在该窗口列明任务请求 continue_batch。"
            "摘要修改用 field=summary；姓名可按实体编号复制；"
            "误修目标已满足时重读后确认指定失败。抽取确已完成时调用 finish_batch 明确声明。"
            "当前结构合法不表示抽取已经完成，不能遗漏已知未完成事项。"
        ),
        "repair_options": _json_object(session.inspect_repairs()),
    }, ensure_ascii=False)


def _upsert_with_budget_feedback(session: BuildSession, entities, unexpected: dict, state: dict) -> str:
    result = _upsert_entities(session, entities, unexpected, write_scope=state.get("write_scope"))
    payload = _json_object(result)
    if isinstance(entities, list) and int(payload.get("changed_count", 0)) > 0:
        state["single_entity_writes"] = state.get("single_entity_writes", 0) + 1 if len(entities) == 1 else 0
        if state["single_entity_writes"] >= 3:
            payload["batching_hint"] = {
                "single_entity_write_streak": state["single_entity_writes"], "max_items": _MAX_BATCH_TOOL_ITEMS,
                "instruction": "请把剩余独立实体合并进一次 entities 数组，短实体可一次提交 2 至 8 个；不要逐轮单个写入耗尽预算。",
            }
            return ("[错误] " if result.startswith("[错误]") else "") + json.dumps(payload, ensure_ascii=False)
    return result


def _upsert_entities(
    session: BuildSession, entities, unexpected: dict | None = None, *, write_scope: set[str] | None = None,
) -> str:
    batch_error = _write_batch_argument_error(session, entities, unexpected)
    if batch_error:
        return batch_error
    results = []
    changes_before = session.change_count
    for index, item in enumerate(entities):
        details = _upsert_argument_errors(item)
        if not isinstance(item, dict):
            result = _argument_error_message(details)
            session.record_write_failure(None, result)
            name = ""
        else:
            name = str(item.get("name", ""))
            if details:
                result = _argument_error_message(details)
                _record_write_argument_failure(session, item, result)
            else:
                try:
                    target_ref = session.store.entity_ref(item["kind"], item["name"].strip(), item.get("domain", "").strip())
                except (ValueError, PermissionError):
                    target_ref = None
                if write_scope is not None and target_ref is not None and target_ref not in write_scope:
                    details = {"error_type": "finalization_scope"}
                    result = (
                        "[错误] 独立收尾窗口仅能更新已暂存实体或补齐本轮开始前明确缺失的链接目标。"
                        "此实体尚未暂存，若属于未完成的抽取任务，请用 continue_batch 的 remaining_tasks 列明后申请续跑。"
                        "本项未写入，不影响已有暂存内容。"
                    )
                    session.record_write_failure(target_ref, result)
                else:
                    result = session.upsert_entity(**item)
        results.append({"index": index, "name": name, "result": result, **details})
    return _entity_write_summary(results, session.change_count - changes_before)


def _patch_entities(session: BuildSession, entities, unexpected: dict | None = None) -> str:
    batch_error = _write_batch_argument_error(session, entities, unexpected)
    if batch_error:
        return batch_error
    results = []
    changes_before = session.change_count
    for index, item in enumerate(entities):
        details = _patch_argument_errors(item)
        if not isinstance(item, dict):
            result = _argument_error_message(details)
            session.record_write_failure(None, result)
            ref = ""
        else:
            ref = str(item.get("ref", ""))
            if details:
                result = _argument_error_message(details)
                session.record_write_failure(item.get("ref"), result, staged_only=True)
            else:
                result = session.patch_entity(**item)
        results.append({"index": index, "ref": ref, "result": result, **details})
    return _entity_write_summary(results, session.change_count - changes_before)


def _write_batch_argument_error(session: BuildSession, entities, unexpected: dict | None) -> str | None:
    invalid = {key: "不支持的工具参数" for key in unexpected or {}}
    if not isinstance(entities, list) or not entities:
        invalid["entities"] = "必须是非空数组"
    elif len(entities) > _MAX_BATCH_TOOL_ITEMS:
        invalid["entities"] = f"每次最多 {_MAX_BATCH_TOOL_ITEMS} 个实体"
    if not invalid:
        return None
    details = _argument_error_details([], invalid, ["entities"])
    result = _argument_error_message(details)
    session.record_write_failure(None, result)
    return "[错误] " + json.dumps({**details, "message": result}, ensure_ascii=False)


def _upsert_argument_errors(item) -> dict:
    required = ["kind", "name", "summary", "content", "source_chapters"]
    if not isinstance(item, dict):
        return _argument_error_details([], {"entity": "必须是 object"}, required)
    if item.get("kind") == "setting":
        required.append("domain")
    missing = []
    invalid = {}
    for key in required:
        value = item.get(key)
        if value is None or (isinstance(value, str) and not value.strip()) or (
            key == "source_chapters" and isinstance(value, list) and not value
        ):
            missing.append(key)
        elif key == "source_chapters":
            if not isinstance(value, list) or any(type(chapter) is not int for chapter in value):
                invalid[key] = "必须是非空整数数组"
        elif not isinstance(value, str):
            invalid[key] = "必须是非空文本"
    if "kind" not in missing and "kind" not in invalid and item["kind"] not in {
        "character", "setting", "event", "arc",
    }:
        invalid["kind"] = "必须是 character、setting、event 或 arc"
    if "domain" in item and "domain" not in required and not isinstance(item["domain"], str):
        invalid["domain"] = "必须是文本"
    if "aliases" in item and (
        not isinstance(item["aliases"], list)
        or any(not isinstance(alias, str) for alias in item["aliases"])
    ):
        invalid["aliases"] = "必须是文本数组"
    allowed = {*required, "aliases", "domain"}
    invalid.update({key: "不支持的实体字段" for key in item if key not in allowed})
    details = _argument_error_details(missing, invalid, required)
    if details and "kind" in missing:
        details["conditional_required_fields"] = {"domain": "kind 为 setting 时还必须提供非空 domain"}
    return details


def _patch_argument_errors(item) -> dict:
    required = ["ref", "replacements"]
    if not isinstance(item, dict):
        return _argument_error_details([], {"entity": "必须是 object"}, required)
    missing = []
    invalid = {key: "不支持的实体字段" for key in item if key not in {*required, "field"}}
    if "field" in item and item["field"] not in ("content", "summary"):
        invalid["field"] = "必须是 content 或 summary"
    ref = item.get("ref")
    if ref is None or (isinstance(ref, str) and not ref.strip()):
        missing.append("ref")
    elif not isinstance(ref, str):
        invalid["ref"] = "必须是非空文本"
    replacements = item.get("replacements")
    if replacements is None or replacements == []:
        missing.append("replacements")
    elif not isinstance(replacements, list):
        invalid["replacements"] = "必须是非空数组"
    else:
        for index, replacement in enumerate(replacements):
            prefix = f"replacements[{index}]"
            if not isinstance(replacement, dict):
                invalid[prefix] = "必须是 object"
                continue
            new_fields = [key for key in ("new_text", "new_entity_id") if key in replacement]
            if len(new_fields) > 1:
                invalid[prefix] = "new_text 和 new_entity_id 必须二选一"
            for key in ("old_text", *(new_fields or ["new_text"])):
                value = replacement.get(key)
                if key not in replacement or (
                    key in {"old_text", "new_entity_id"} and (value is None or isinstance(value, str) and not value.strip())
                ):
                    missing.append(f"{prefix}.{key}")
                elif not isinstance(value, str):
                    invalid[f"{prefix}.{key}"] = "必须是文本"
            invalid.update({
                f"{prefix}.{key}": "不支持的替换字段"
                for key in replacement if key not in {"old_text", "new_text", "new_entity_id"}
            })
    return _argument_error_details(missing, invalid, required)


def _argument_error_details(missing: list[str], invalid: dict[str, str], required: list[str]) -> dict:
    if not missing and not invalid:
        return {}
    return {
        "error_type": "invalid_arguments",
        "missing_fields": missing,
        "invalid_fields": invalid,
        "required_fields": required,
    }


def _argument_error_message(details: dict) -> str:
    problems = []
    if details["missing_fields"]:
        problems.append(f"缺少或为空的必填字段：{', '.join(details['missing_fields'])}")
    problems.extend(f"{key}：{reason}" for key, reason in details["invalid_fields"].items())
    problems.extend(details.get("conditional_required_fields", {}).values())
    return "[错误] 参数不合法：" + "；".join(problems)


def _record_write_argument_failure(session: BuildSession, item: dict, result: str) -> None:
    try:
        ref = session.store.entity_ref(
            item.get("kind"), item.get("name", "").strip(), item.get("domain", "").strip()
        )
    except (AttributeError, TypeError, ValueError, PermissionError):
        ref = None
    session.record_write_failure(ref, result)


def _entity_write_summary(results: list[dict], changed: int) -> str:
    staged = sum(item["result"].startswith("[已暂存]") for item in results)
    failed = len(results) - staged
    summary = {"staged_count": staged, "changed_count": changed,
               "failed_count": failed, "results": results}
    failures = [item for item in results if not item["result"].startswith("[已暂存]")]
    if failures and all(item.get("error_type") == "invalid_arguments" for item in failures):
        summary["error_type"] = "invalid_arguments"
    if staged and not changed:
        summary["feedback"] = (
            "本次成功暂存的实体内容均未变化，请不要重复提交相同内容。"
            "请处理其他未完成实体；如需修正，请根据原文证据更新；若任务已完成，请输出最终结果。"
        )
    payload = json.dumps(summary, ensure_ascii=False)
    return f"[错误] {payload}" if failed else payload


def _has_staged_entities(result: str) -> bool:
    return int(_json_object(result).get("changed_count", 0)) > 0


def _has_repair_progress(result: str) -> bool:
    payload = _json_object(result)
    return int(payload.get("changed_count", 0)) > 0 or int(payload.get("resolved_count", 0)) > 0


def _has_argument_errors(result: str) -> bool:
    return _json_object(result).get("error_type") == "invalid_arguments"


def _json_object(value: str) -> dict:
    raw = value.strip()
    if raw.startswith("[错误]"):
        raw = raw[len("[错误]") :].strip()
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}
