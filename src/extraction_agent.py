"""兼容入口：具体业务实现在 ``src.novel.builder``。"""

from pathlib import Path

from src.config import KNOWLEDGE_BASE_DIR
from src.harness.runtime import ChatProvider
from src.novel.builder import format_batch, run_extraction_agent as _run
from src.harness.runtime import trim_old_tool_results


def run_extraction_agent(
    batch: list[dict],
    novel_id: str,
    debug: bool = False,
    provider: ChatProvider | None = None,
    model: str | None = None,
) -> dict:
    base_dir = str(Path(KNOWLEDGE_BASE_DIR) / novel_id)
    return _run(batch, novel_id, base_dir, debug=debug, provider=provider, model=model)


_format_batch = format_batch


def _trim_messages_if_needed(messages: list[dict], total_tokens: int, model_limit: int) -> bool:
    return bool(trim_old_tool_results(messages, total_tokens, model_limit))
