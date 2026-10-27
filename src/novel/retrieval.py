"""小说问答的最小只读工具集。"""

from src.novel.corpus import CorpusStore
from src.novel.evidence import EvidenceCollector
from src.novel.knowledge import KnowledgeStore
from src.harness.runtime import Tool


def qa_tools(
    knowledge: KnowledgeStore,
    corpus: CorpusStore,
    evidence: EvidenceCollector,
) -> list[Tool]:
    return [
        Tool(
            "search_entities",
            "按名称、别名和摘要搜索，精确名称和别名优先；这是定位工具，不是回答证据。"
            "结果为整次查询的限量候选，查看截断和逐词覆盖反馈；仅对未覆盖词缩小查询，先读取已有主体。",
            {
                "type": "object",
                "properties": {
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "entity_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["character", "setting", "event", "arc"]},
                    },
                    "limit": {"type": "integer", "default": 30, "minimum": 1, "maximum": 100},
                },
                "required": ["keywords"],
            },
            knowledge.search_entities,
        ),
        Tool(
            "read_entity",
            "读取一个知识实体。ref 来自 search_entities；旧知识库的 #锚点也可精确读取对应段落。",
            {
                "type": "object",
                "properties": {
                    "ref": {"type": "string"},
                    "section": {"type": "string", "description": "可选 Markdown 小节标题"},
                },
                "required": ["ref"],
            },
            evidence.wrap("read_entity", knowledge.read_entity),
        ),
        Tool(
            "search_source",
            "检索章节原文。支持 any/all、多章节范围、正序/倒序和分页；关键结论应以原文为准。",
            {
                "type": "object",
                "properties": {
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "mode": {"type": "string", "enum": ["any", "all"], "default": "any"},
                    "chapter_start": {"type": "integer"},
                    "chapter_end": {"type": "integer"},
                    "order": {"type": "string", "enum": ["asc", "desc"], "default": "asc"},
                    "cursor": {"type": "integer", "default": 0},
                    "limit": {"type": "integer", "default": 20},
                    "context_lines": {"type": "integer", "default": 1},
                },
                "required": ["keywords"],
            },
            evidence.wrap("search_source", corpus.search),
        ),
        Tool(
            "read_chapter",
            "读取指定章节完整原文；用于补足搜索片段的上下文。",
            {
                "type": "object",
                "properties": {"chapter_no": {"type": "integer"}},
                "required": ["chapter_no"],
            },
            evidence.wrap("read_chapter", corpus.read_chapter),
        ),
    ]
