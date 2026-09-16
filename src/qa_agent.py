"""兼容入口：具体业务实现在 ``src.novel.qa``。"""

from pathlib import Path
from typing import Callable

from src.config import KNOWLEDGE_BASE_DIR
from src.harness.runtime import ChatProvider
from src.novel.evidence import parse_sources
from src.novel.qa import run_qa_agent as _run


def run_qa_agent(
    message: str,
    novel_id: str,
    history: list[dict] | None = None,
    on_event: Callable[[dict], None] | None = None,
    provider: ChatProvider | None = None,
    model: str | None = None,
) -> dict:
    base_dir = str(Path(KNOWLEDGE_BASE_DIR) / novel_id)
    return _run(
        message,
        novel_id,
        base_dir,
        history=history,
        on_event=on_event,
        provider=provider,
        model=model,
    )


_parse_sources = parse_sources
