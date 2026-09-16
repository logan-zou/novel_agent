"""可选 Langfuse 接入；未安装或未配置时不影响主流程。"""

from __future__ import annotations

from typing import Any

try:
    from langfuse import Langfuse
except ImportError:  # observability 是可选依赖
    Langfuse = None  # type: ignore[assignment]

from src.config import LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY


_client: Any | None = None


def get_langfuse_client() -> Langfuse | None:
    global _client
    if Langfuse is None or not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
        return None
    if _client is None:
        _client = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )
    return _client


def create_trace(name: str, **kwargs):
    client = get_langfuse_client()
    return client.trace(name=name, **kwargs) if client else None


def finish_trace(trace, output) -> None:
    if trace:
        trace.update(output=output)
    client = get_langfuse_client()
    if client:
        client.flush()
