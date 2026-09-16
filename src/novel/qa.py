"""小说问答：知识库定位、原文核验和证据化回答。"""

import time
from pathlib import Path
from typing import Callable

from src.config import (
    MODEL_CONTEXT_TOKENS,
    MODEL_MAX_OUTPUT_TOKENS,
    MODEL_TEMPERATURE,
    QA_MODEL,
    QA_PROMPT_PATH,
    QA_REASONING_EFFORT,
    require_llm_model,
)
from src.llm import get_provider
from src.novel.corpus import CorpusStore
from src.novel.evidence import EvidenceCollector
from src.novel.knowledge import KnowledgeStore
from src.novel.retrieval import qa_tools
from src.harness.runtime import AgentConfig, AgentRuntime, ChatProvider
from src.tracing import create_trace, finish_trace
from src.usage_tracker import record as record_usage


def run_qa_agent(
    message: str,
    novel_id: str,
    base_dir: str,
    history: list[dict] | None = None,
    on_event: Callable[[dict], None] | None = None,
    provider: ChatProvider | None = None,
    model: str | None = None,
) -> dict:
    selected_model = require_llm_model(model or QA_MODEL, "QA_MODEL")
    corpus = CorpusStore(base_dir)
    knowledge = KnowledgeStore(base_dir)
    evidence = EvidenceCollector(source_reader=corpus.read_chapter)
    trace = create_trace(
        "qa",
        input=message,
        metadata={"novel_id": novel_id, "model": selected_model, "history_turns": len(history or [])},
    )
    label = f"qa_{int(time.time() * 1000)}"
    input_messages = [dict(item) for item in history or []]
    input_messages.append({"role": "user", "content": message})

    def forward(event_type: str, data: dict) -> None:
        if not on_event:
            return
        if event_type == "model_started":
            on_event({"type": "reasoning_start", "step": data["step"]})
        elif event_type == "model_reasoning_delta":
            on_event(
                {"type": "reasoning_delta", "step": data["step"], "delta": data["delta"]}
            )
        elif event_type == "model_content_delta":
            on_event({"type": "answer_delta", "step": data["step"], "delta": data["delta"]})
        elif event_type == "model_completed":
            on_event(
                {
                    "type": "reasoning_end",
                    "step": data["step"],
                    "content": data.get("reasoning_content", ""),
                    "finish_reason": data.get("finish_reason", ""),
                }
            )
            if data.get("finish_reason") == "tool_calls":
                on_event({"type": "answer_reset", "step": data["step"]})
        elif event_type == "output_rejected":
            on_event({"type": "answer_reset", "step": data["step"]})
        elif event_type == "tool_started":
            on_event(
                {
                    "type": "tool_start",
                    "step": data["step"],
                    "call_id": data["call_id"],
                    "name": data["name"],
                    "arguments": data.get("arguments", {}),
                    "arguments_raw": data.get("arguments_raw", ""),
                    "arguments_error": data.get("arguments_error"),
                }
            )
        elif event_type == "tool_completed":
            on_event(
                {
                    "type": "tool_end",
                    "step": data["step"],
                    "call_id": data["call_id"],
                    "name": data["name"],
                    "arguments": data.get("arguments", {}),
                    "arguments_raw": data.get("arguments_raw", ""),
                    "result": data.get("result", ""),
                    "result_len": data.get("result_len", 0),
                }
            )

    try:
        result = AgentRuntime(provider or get_provider()).run(
            AgentConfig(
                name="novel-qa",
                model=selected_model,
                system_prompt=Path(QA_PROMPT_PATH).read_text(encoding="utf-8"),
                max_iterations=24,
                max_output_tokens=MODEL_MAX_OUTPUT_TOKENS,
                temperature=MODEL_TEMPERATURE,
                context_window_tokens=MODEL_CONTEXT_TOKENS,
                reasoning_effort=QA_REASONING_EFFORT,
            ),
            input_messages,
            qa_tools(knowledge, corpus, evidence),
            on_event=forward,
            usage_recorder=lambda usage: record_usage(novel_id, "qa", label, selected_model, usage),
            output_validator=evidence.validator,
        )
        reply = result.output
        citations = evidence.cited_items(reply)
        report = evidence.validate(reply)
        finish_trace(trace, reply)
        return {
            "reply": reply,
            "sources": list(report.cited_chapters),
            "citations": citations,
            "evidence": citations,
            "citation_validation": report.to_dict(),
            "usage": result.usage,
            "run_id": result.run_id,
        }
    except Exception as error:
        finish_trace(trace, f"[错误] {error}")
        raise
