# Novel Agent

Novel Agent 是面向长篇小说的知识构建与证据化事实问答核心。它先把章节原文压缩成可检索的角色、设定、事件与故事弧，再在回答时用知识库定位线索、回到章节原文核验，最终给出带原文证据的回答。

核心仓库只包含小说业务逻辑、最小 Agent Harness、命令行工具和评测；多小说体验、登录、配额与 Web 部署位于独立的 `novel_agent_example` 仓库。

## 为什么不是普通 RAG

长篇小说会同时遇到三个问题：事件跨越多个文本块、虚构世界的词义不适合直接依赖通用向量相似度、后期揭示可能推翻前期人物猜测。Novel Agent 的核心设计是：

- **知识库负责定位，原文负责裁决**：结构化知识用于快速找到人物和事件，关键结论必须回章节原文验证。
- **一实体一文件**：角色、设定、事件和故事弧以 Markdown 独立存储，并由 `catalog.jsonl` 提供名称、别名和摘要索引。
- **渐进式知识构建**：按章节批次持续抽取；随着全书进度缩小批次，控制知识增长后的检索与合并复杂度。
- **有依据的回答**：原文搜索产生运行内 Evidence ID，答案通过 `[[cite:EVxxx]]` 内联引用，输出前校验证据和章节归属。
- **面向长任务的轻量 Harness**：统一处理工具循环、非法参数重试、上下文裁剪、截断恢复、迭代边界和 token 统计。
- **安全提交与断点续跑**：抽取结果先在批次会话中暂存并校验，通过后统一提交；失败批次可以自动拆分并继续。

## 架构

```text
CLI / 调用方
      │
      ▼
小说业务层 src/novel
  ├─ Builder：章节 → 实体知识
  ├─ QA：知识定位 → 原文核验 → 证据回答
  ├─ CorpusStore：章节原文与全文搜索
  └─ KnowledgeStore：实体、Catalog 与安全提交
      │
      ▼
Agent Harness src/harness
  ├─ 模型与工具循环
  ├─ 重试、裁剪与输出校验
  └─ Token 用量归一化
      │
      ▼
OpenAI-compatible Chat Completions API
```

Harness 不依赖小说领域、具体模型或观测平台，这一边界由 `tests/test_architecture_boundaries.py` 自动检查。更详细的业务设计见 [`docs/DESIGN.md`](docs/DESIGN.md)，Harness 说明见 [`docs/HARNESS_ARCHITECTURE.md`](docs/HARNESS_ARCHITECTURE.md)。

## 快速开始

运行环境要求 Python 3.11 或更高版本。小说原文和构建后的知识库不会进入 Git，`novels/` 与 `knowledge_base/` 已加入 `.gitignore`。

### 1. 安装

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

### 2. 配置模型

```bash
cp .env.example .env
```

至少填写三个配置：

```dotenv
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your-api-key
LLM_MODEL=your-model
```

任何实现 OpenAI Chat Completions 流式接口和 tool calling 的服务都可以接入。配置完成后可以检查生效结果；命令不会输出 API Key：

```bash
./novel-agent config
```

### 3. 构建知识库

```bash
./novel-agent build /path/to/novel.txt example_novel
```

输入为小说 TXT。程序会自动探测 UTF-8、GB18030 等常见编码并划分章节，结果默认写入 `knowledge_base/example_novel/`。构建会产生模型费用；可以先只处理一个批次验证配置：

```bash
./novel-agent build /path/to/novel.txt example_novel --max-batches 1
```

重新执行同一命令会从已保存进度继续。

### 4. 提问

```bash
./novel-agent ask example_novel "主角为什么离开故乡？"
```

如果模型服务返回 `reasoning_content`，可以选择实时展示：

```bash
./novel-agent ask example_novel "主角为什么离开故乡？" --show-reasoning
```

需要程序化消费完整回答、证据、引用校验和 token 用量时，增加 `--json`。

## 模型配置

| 配置 | 必填 | 说明 |
|---|---:|---|
| `LLM_BASE_URL` | 是 | OpenAI-compatible `/v1` 地址；默认是 OpenAI 官方地址 |
| `LLM_API_KEY` | 是 | API Key，只保存在本地 `.env` |
| `LLM_MODEL` | 是* | 构建和问答共用的模型名 |
| `EXTRACTION_MODEL` | 否 | 单独指定知识抽取模型，会覆盖 `LLM_MODEL` |
| `QA_MODEL` | 否 | 单独指定问答模型，会覆盖 `LLM_MODEL` |
| `LLM_INCLUDE_REASONING_EFFORT` | 否 | 向兼容服务发送 `reasoning_effort`，默认关闭 |
| `LLM_ENABLE_THINKING` | 否 | 发送兼容服务的 thinking 扩展参数，默认关闭 |
| `MODEL_CONTEXT_TOKENS` | 否 | 模型上下文上限，用于 Harness 主动裁剪 |
| `MODEL_MAX_OUTPUT_TOKENS` | 否 | 单次模型调用最大输出 |
| `KNOWLEDGE_BASE_DIR` | 否 | 知识库根目录 |

\* 如果分别配置了 `EXTRACTION_MODEL` 和 `QA_MODEL`，可以不设置 `LLM_MODEL`。扩展推理参数并非 OpenAI-compatible 协议的统一能力，只有服务商明确支持时才应开启；`reasoning_content` 不参与业务正确性判断。

## 命令

```text
novel-agent config     检查公开配置
novel-agent build      构建或续跑知识库
novel-agent ask        对已构建的小说提问
novel-agent evaluate   校验或运行固定评测集
```

安装项目后可以直接使用 `novel-agent`；在仓库中也可以使用零额外配置的 `./novel-agent` 启动脚本。原有 `./build.sh` 继续兼容。

## 依赖

核心依赖有且只有：

| 依赖 | 版本 | 用途 |
|---|---|---|
| Python | `>=3.11` | 运行环境 |
| `openai` | `>=1.0,<3.0` | 调用 OpenAI-compatible Chat Completions API |

项目使用 `setuptools>=68` 作为构建后端，由 `pip` 根据 `pyproject.toml` 自动安装，不属于运行时依赖。

Langfuse 只用于可选观测，不是业务运行条件：

```bash
.venv/bin/pip install -e '.[observability]'
```

依赖和命令行入口的权威声明位于 [`pyproject.toml`](pyproject.toml)；[`requirements.txt`](requirements.txt) 保留给传统安装流程，可选观测依赖位于 [`requirements-observability.txt`](requirements-observability.txt)。测试使用 Python 标准库 `unittest`，不引入额外测试框架。

## 测试与评测

完整单元测试不会调用真实模型：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

评测集默认只做离线格式校验。增加 `--live` 才会调用模型并产生费用：

```bash
./novel-agent evaluate
./novel-agent evaluate --live --case-id your_case_id
```

## 数据与示例服务边界

- `novels/`：小说原文，仅本地使用，不提交 Git。
- `knowledge_base/`：章节副本、实体知识、构建进度和用量记录，不提交 Git。
- `../novel_agent_example`：独立 Web 示例服务，负责多小说选择、GitHub 登录、体验配额和部署。

本仓库不附带小说原文或预构建知识库。使用者需要对输入内容拥有合法使用权。

## License

本项目采用 [Apache License 2.0](LICENSE)。小说原文、预构建知识库以及独立示例服务中的数据不属于本仓库许可范围。
