"""小说知识构建：一个抽取 Agent + 领域化实体工具 + 批次暂存。"""

import json
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
    first, last = batch[0]["chapter_no"], batch[-1]["chapter_no"]
    label = f"batch_ch{first}-{last}"
    trace = create_trace(
        "extraction",
        metadata={"novel_id": novel_id, "chapter_range": [first, last], "model": selected_model},
    )

    def on_event(event_type: str, data: dict) -> None:
        if event_type == "tool_completed":
            _log_tool(data["name"], data.get("arguments", {}), data.get("result", ""))
        elif event_type == "tool_arguments_rejected" and data.get("will_retry"):
            print(
                "    ⚠️  工具参数 JSON 无效，本轮未执行，"
                f"正在自动重试（{data['attempt']}/{data['max_retries']}）"
            )
        elif event_type == "context_compacted":
            print(f"    ✂️  已省略 {data['messages_compacted']} 条较早工具结果")
        elif debug and event_type == "model_completed" and data.get("finish_reason") == "length":
            print("    ⚠️  模型输出被截断，正在恢复")

    try:
        result = AgentRuntime(provider or get_provider()).run(
            AgentConfig(
                name="novel-extraction",
                model=selected_model,
                system_prompt=Path(EXTRACTION_PROMPT_PATH).read_text(encoding="utf-8"),
                max_iterations=60,
                max_output_tokens=MODEL_MAX_OUTPUT_TOKENS,
                temperature=MODEL_TEMPERATURE,
                context_window_tokens=MODEL_CONTEXT_TOKENS,
                reasoning_effort=EXTRACTION_REASONING_EFFORT,
                soft_iteration_limit=40,
                progress_grace_iterations=8,
                max_identical_tool_calls=3,
                max_consecutive_tool_errors=3,
            ),
            [{"role": "user", "content": format_batch(batch)}],
            _builder_tools(session, corpus),
            on_event=on_event,
            usage_recorder=lambda usage: record_usage(
                novel_id, "extraction", label, selected_model, usage
            ),
            output_validator=session.validator,
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


def _builder_tools(session: BuildSession, corpus: CorpusStore) -> list[Tool]:
    return [
        Tool(
            "search_entities",
            "批量搜索已有实体。把本批候选名称和别名尽量放在一次 keywords 中，写入前确认已有记录。",
            {
                "type": "object",
                "properties": {
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "entity_types": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer", "default": 30},
                },
                "required": ["keywords"],
            },
            session.search_entities,
        ),
        Tool(
            "read_entities",
            "批量读取已有或本批已暂存实体。一次最多读取 8 个；更新实体前必须在这里取得旧正文。",
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
                                "section": {"type": "string"},
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
            lambda entities: _upsert_entities(session, entities),
            progress_detector=_has_staged_entities,
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
    if name == "upsert_entities":
        entities = arguments.get("entities") or []
        names = [str(item.get("name", "?")) for item in entities if isinstance(item, dict)]
        summary = _json_object(result)
        staged = summary.get("staged_count", 0)
        failed = summary.get("failed_count", 0)
        print(f"    ✏️  批量暂存 {staged} 个实体（失败 {failed}）：{', '.join(names)}")
    elif name == "read_entities":
        requests = arguments.get("requests") or []
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
    for item in requests:
        if not isinstance(item, dict) or not str(item.get("ref", "")).strip():
            results.append({"ref": "", "error": "每一项都必须包含 ref"})
            continue
        ref = str(item["ref"])
        content = session.read_entity(ref, str(item.get("section", "")))
        if used_chars + len(content) > _MAX_BATCH_READ_CHARS:
            results.append(
                {
                    "ref": ref,
                    "error": "本次批量读取已达到 60000 字上限，请在下一轮再次读取该实体",
                }
            )
            deferred += 1
            continue
        results.append({"ref": ref, "content": content})
        used_chars += len(content)
    return json.dumps(
        {"entities": results, "returned_chars": used_chars, "deferred_count": deferred},
        ensure_ascii=False,
    )


def _upsert_entities(session: BuildSession, entities) -> str:
    if not isinstance(entities, list) or not entities:
        return "[错误] entities 必须是非空数组"
    results = []
    staged = 0
    failed = 0
    for index, item in enumerate(entities):
        if not isinstance(item, dict):
            result = "[错误] 实体必须是 object"
            name = ""
        else:
            name = str(item.get("name", ""))
            try:
                result = session.upsert_entity(**item)
            except TypeError as error:
                result = f"[错误] 参数不合法：{error}"
        if result.startswith("[已暂存]"):
            staged += 1
        else:
            failed += 1
        results.append({"index": index, "name": name, "result": result})
    payload = json.dumps(
        {"staged_count": staged, "failed_count": failed, "results": results},
        ensure_ascii=False,
    )
    return payload if staged else f"[错误] {payload}"


def _has_staged_entities(result: str) -> bool:
    return int(_json_object(result).get("staged_count", 0)) > 0


def _json_object(value: str) -> dict:
    raw = value.strip()
    if raw.startswith("[错误]"):
        raw = raw[len("[错误]") :].strip()
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}
