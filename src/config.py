"""运行配置。

公开配置全部来自环境变量；仓库根目录或当前工作目录下的 ``.env`` 会被
自动加载。环境变量优先于文件中的值，便于容器和 CI 覆盖本地配置。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_env_file(path: Path) -> None:
    """加载简单 KEY=VALUE 格式的 .env，不覆盖进程已有环境变量。"""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


_load_env_file(PROJECT_ROOT / ".env")
if Path.cwd().resolve() != PROJECT_ROOT:
    _load_env_file(Path.cwd() / ".env")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError as error:
        raise ValueError(f"{name} 必须是整数") from error


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError as error:
        raise ValueError(f"{name} 必须是数字") from error


# ── 可选观测 ─────────────────────────────────────────────
LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com").strip()
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()

# ── 数据目录 ─────────────────────────────────────────────
KNOWLEDGE_BASE_DIR = os.environ.get(
    "KNOWLEDGE_BASE_DIR",
    str(PROJECT_ROOT / "knowledge_base"),
)

# ── 预处理 ───────────────────────────────────────────────
DEFAULT_WINDOW_CHARS = _env_int("DEFAULT_WINDOW_CHARS", 30_000)
DEFAULT_MIN_WINDOW_CHARS = _env_int("DEFAULT_MIN_WINDOW_CHARS", 20_000)

# ── OpenAI-compatible LLM ────────────────────────────────
LLM_API_KEY = os.environ.get("LLM_API_KEY", "").strip()
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
LLM_MODEL = os.environ.get("LLM_MODEL", "").strip()

EXTRACTION_MODEL = os.environ.get("EXTRACTION_MODEL", LLM_MODEL).strip()
QA_MODEL = os.environ.get("QA_MODEL", LLM_MODEL).strip()

EXTRACTION_REASONING_EFFORT = os.environ.get(
    "EXTRACTION_REASONING_EFFORT", "low"
).strip()
QA_REASONING_EFFORT = os.environ.get("QA_REASONING_EFFORT", "high").strip()
MODEL_TEMPERATURE = _env_float("MODEL_TEMPERATURE", 0.5)
MODEL_MAX_OUTPUT_TOKENS = _env_int("MODEL_MAX_OUTPUT_TOKENS", 16_384)
MODEL_CONTEXT_TOKENS = _env_int("MODEL_CONTEXT_TOKENS", 200_000)

# 不同兼容服务对扩展推理参数的支持不同，因此默认不发送。
LLM_INCLUDE_REASONING_EFFORT = _env_bool("LLM_INCLUDE_REASONING_EFFORT")
LLM_ENABLE_THINKING = _env_bool("LLM_ENABLE_THINKING")

# ── 提示词文件路径 ───────────────────────────────────────
_source_prompts_dir = PROJECT_ROOT / "prompts"
_installed_prompts_dir = Path(sys.prefix) / "share" / "novel-agent" / "prompts"
PROMPTS_DIR = (
    _source_prompts_dir if _source_prompts_dir.is_dir() else _installed_prompts_dir
)
EXTRACTION_PROMPT_PATH = str(PROMPTS_DIR / "extraction.md")
QA_PROMPT_PATH = str(PROMPTS_DIR / "qa.md")


class ConfigurationError(RuntimeError):
    """公开运行配置缺失或不合法。"""


def require_llm_model(model: str, setting_name: str) -> str:
    if model:
        return model
    raise ConfigurationError(
        f"缺少模型配置：请在 .env 中设置 LLM_MODEL，或单独设置 {setting_name}"
    )
