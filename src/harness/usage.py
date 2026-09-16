"""模型用量标准化：明确区分缓存读、非缓存读和缓存归属未知。"""

from typing import Any


TOKEN_FIELDS = (
    "prompt_tokens",
    "cached_prompt_tokens",
    "uncached_prompt_tokens",
    "unknown_cache_prompt_tokens",
    "completion_tokens",
    "total_tokens",
)


def empty_usage() -> dict[str, int]:
    return {field: 0 for field in TOKEN_FIELDS}


def _value(obj: Any, key: str):
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def extract_usage(usage: Any) -> dict[str, int] | None:
    """兼容 OpenAI usage、普通 dict 和已经标准化的 usage。"""
    if usage is None:
        return None

    prompt = _value(usage, "prompt_tokens")
    completion = _value(usage, "completion_tokens")
    total = _value(usage, "total_tokens")
    if total is None:
        return None

    prompt = prompt or 0
    completion = completion or 0
    total = total or 0

    if isinstance(usage, dict) and any(field in usage for field in TOKEN_FIELDS[1:4]):
        return {
            "prompt_tokens": prompt,
            "cached_prompt_tokens": usage.get("cached_prompt_tokens") or 0,
            "uncached_prompt_tokens": usage.get("uncached_prompt_tokens") or 0,
            "unknown_cache_prompt_tokens": usage.get("unknown_cache_prompt_tokens") or 0,
            "completion_tokens": completion,
            "total_tokens": total,
        }

    details = _value(usage, "prompt_tokens_details")
    cached_raw = _value(details, "cached_tokens") if details is not None else None
    cache_known = (
        isinstance(cached_raw, int)
        and not isinstance(cached_raw, bool)
        and 0 <= cached_raw <= prompt
    )
    cached = cached_raw if cache_known else 0
    return {
        "prompt_tokens": prompt,
        "cached_prompt_tokens": cached,
        "uncached_prompt_tokens": prompt - cached if cache_known else 0,
        "unknown_cache_prompt_tokens": 0 if cache_known else prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def accumulate_usage(total: dict[str, int], usage: Any) -> dict[str, int]:
    tokens = extract_usage(usage)
    if tokens is None:
        return total
    for field in TOKEN_FIELDS:
        total[field] = (total.get(field) or 0) + tokens[field]
    return total


def attribute_agent_loop_usage(
    usage: Any,
    previous_prompt_tokens: int | None,
) -> dict[str, int] | None:
    """按 Agent loop 的稳定前缀估算缓存，不采用 API 返回的缓存明细。

    首次请求没有可复用前缀，输入全部归为非缓存读。后续请求把上一轮
    prompt 视为稳定缓存前缀，当前 prompt 相比上一轮增加的部分视为本轮
    最新输入。上下文缩短时用当前 prompt 封顶，始终保持输入 token 守恒。
    """
    tokens = extract_usage(usage)
    if tokens is None:
        return None

    prompt = tokens["prompt_tokens"]
    cached = 0 if previous_prompt_tokens is None else min(previous_prompt_tokens, prompt)
    return {
        "prompt_tokens": prompt,
        "cached_prompt_tokens": cached,
        "uncached_prompt_tokens": prompt - cached,
        "unknown_cache_prompt_tokens": 0,
        "completion_tokens": tokens["completion_tokens"],
        "total_tokens": tokens["total_tokens"],
    }
