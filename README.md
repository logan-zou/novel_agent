# Novel Agent

Novel Agent 面向长篇小说的知识构建与事实问答。它将章节内容整理为可检索的角色、设定、事件与故事弧，回答时用知识库定位线索，再回原文核验并给出引用。

本仓库提供小说业务核心、最小 Agent Runtime、命令行、测试和原创样例，通过克隆源码安装。
Web 服务、登录、配额和部署由调用方独立实现；当前公开入口以本仓库 CLI 为准。

## 核心设计

- **知识库定位，原文核验**：关键结论回到章节原文，处理后期揭秘、身份变化和主体/部件关系。
- **一实体一文件**：Markdown 保存知识，`catalog.jsonl` 索引名称、别名和摘要，准确名称优先检索。
- **渐进构建**：按章节批次持续抽取，随全书进度缩小窗口，支持批次断点续跑和失败拆批。
- **校验后提交**：实体先在批内暂存，检查元数据、链接和失败请求后统一提交；原文指纹防止续跑混用不同版本。
- **原文引用**：搜索结果生成 Evidence ID，答案使用 `[[cite:EVxxx]]`，校验引用来源与章节归属。
- **轻量 Runtime**：统一处理模型与工具循环、参数重试、上下文裁剪、截断恢复和运行预算。

```text
CLI / 调用方
    ↓
src/novel：构建 → 实体知识；问答 → 原文证据
    ↓
src/harness：工具循环、预算、恢复与用量
    ↓
OpenAI-compatible Chat Completions API
```

[业务设计](docs/DESIGN.md) · [Runtime 边界](docs/HARNESS_ARCHITECTURE.md) · [构建与故障恢复](docs/BUILDING.md)

## 从源码开始

需要 Python 3.11+。下列命令适用于 macOS/Linux，在仓库目录运行：

```bash
git clone https://github.com/logan-zou/novel_agent.git
cd novel_agent
python3 -m venv .venv
.venv/bin/pip install -e .
```

### 1. 先做无模型检查

仓库附带原创短篇《雾港守灯簿》及人工参考答案，无需下载商业小说或配置密钥：

```bash
./novel-agent --help
./novel-agent evaluate --dataset examples/mist_harbor/qa_cases.jsonl
./novel-agent evaluate --dataset examples/mist_harbor/qa_cases.jsonl --responses examples/mist_harbor/reference_responses.jsonl
bash build_many.sh
```

这些命令不调用模型。参考答案用于检查评测流程，不能作为模型准确率成绩。
`build_many.sh` 默认只预览任务；[样例说明](examples/mist_harbor/README.md) 提供原文、逐题依据和完整实验步骤。

### 2. 配置模型

```bash
cp .env.example .env
```

填写实际服务地址、密钥和模型名：

```dotenv
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your-api-key
LLM_MODEL=your-model
```

模型服务需支持流式 Chat Completions、tool calling 和 usage 返回。
协议兼容程度和抽取效果取决于服务及模型；请按实际模型设置上下文、输出容量，先跑小样例验证。
推理扩展参数默认关闭，仅在服务商明确支持时开启。

```bash
./novel-agent config
```

配置检查只显示密钥是否存在，不输出密钥。`.env` 与数据目录已被 Git 忽略。

### 3. 构建并提问

以下命令会调用模型并产生服务商费用：

```bash
./novel-agent build examples/mist_harbor/mist_harbor.txt mist_harbor --max-batches 1
./novel-agent ask mist_harbor "碎星镜和鸣潮灯是什么关系？" --json
```

输出默认保存在 `knowledge_base/mist_harbor/`。示例原文很短，默认窗口可在一个批次内处理完。
自己的小说可以用同样方式先试一个批次；去掉 `--max-batches 1` 后执行完整构建。

重新运行相同命令会从已提交批次续跑。章号、标题、正文指纹和已保存章节必须一致；
修订原文请使用新的 `novel_id`。旧进度仅在已存章节完整匹配时迁移，原文不匹配时停止且保留原库。
TXT 输入支持 UTF-8、GB18030 等常见编码。

`ask --json` 返回回答、证据、引用校验和用量。`--show-reasoning` 可显示服务返回的 `reasoning_content`，该字段是可选能力。

### 4. 跑真实问答评测

完成样例知识库构建后，可对同一组问题运行模型评测并保存报告：

```bash
./novel-agent evaluate --dataset examples/mist_harbor/qa_cases.jsonl --live --output knowledge_base/mist_harbor/evaluation.json
```

真实评测会产生模型费用；全部选中案例先检查知识库和章节是否可用。
默认 `evaluate` 只校验数据格式；传 `--responses` 评分已有回答，任何案例不通过时退出码非零。

## 批量构建

```bash
# 只预览输入和输出，不调用模型或创建日志
./novel-agent build-many ./novels --dry-run

# 实际试跑：同时一本，每本最多一个批次
./novel-agent build-many ./novels --workers 1 --max-batches 1

# 使用脚本顶部的输入列表；默认只预览，显式 --run 才实际构建
bash build_many.sh --run
```

脚本默认使用原创样例、1 路并发、1 批上限，关闭调试。完整运行可修改输入列表并用 `--run --max-batches 0`。
实际运行日志和汇总位于 `knowledge_base/_build_logs/`。同一本知识库应仅由一个进程写入。
批次上限下的成功仅表示本次运行成功，整本完成以 `progress.json` 为准。
更多并发、日志和修复规则见 [构建与故障恢复](docs/BUILDING.md)。

## 模型配置

| 配置 | 说明 |
|---|---|
| `LLM_BASE_URL` | Chat Completions `/v1` 地址，默认 `https://api.openai.com/v1` |
| `LLM_API_KEY` | 模型服务密钥，保存在本地 `.env` 或环境变量 |
| `LLM_MODEL` | 构建和问答共用的模型名；也可分别指定下面两个模型 |
| `EXTRACTION_MODEL` / `QA_MODEL` | 覆盖对应任务的模型名 |
| `MODEL_CONTEXT_TOKENS` | 模型上下文容量，用于主动裁剪 |
| `MODEL_MAX_OUTPUT_TOKENS` | 单次调用最大输出，需按模型配置 |
| `MODEL_TEMPERATURE` | 请求温度，默认 0.5，需确认模型支持 |
| `LLM_INCLUDE_REASONING_EFFORT` | 是否发送 `reasoning_effort`，默认关闭 |
| `LLM_ENABLE_THINKING` | 是否发送 thinking 扩展，默认关闭 |
| `LLM_STREAM_MAX_RETRIES` | 网络/流式中断后的额外重试次数，默认 5，0 关闭 |
| `KNOWLEDGE_BASE_DIR` | 知识库根目录，默认源码仓库内 `knowledge_base/` |

环境变量优先于 `.env`。配置模板见 [.env.example](.env.example)。
核心运行依赖只有 `openai`，完整声明在 [pyproject.toml](pyproject.toml)；安装使用本地源码，不需要 PyPI 发布包。
可选观测使用 `.venv/bin/pip install -e '.[observability]'`。

## 效果与费用的边界

公开样例覆盖事实、别名、部件关系、后期反转、枚举和信息不足；另保留两条历史小说 badcase。
评分检查关键词、禁词、预期章节和引用状态。引用有效只证明证据确实读取过，不自动证明每个结论都被证据支持。
人工参考答案的通过率不能代表模型质量；真实效果需记录所用模型、配置、版本、输入输出 token 和时长。

输入/输出 token 来自服务返回的 usage。显示的缓存拆分是按 Agent 请求前缀长度的**估算**，
不代表服务商实际缓存命中或计费；失败流缺失的 usage 也无法计入本地记录。费用以服务商记录为准。
详细规则和真实评测步骤见 [评测说明](evals/README.md)。

## 数据与隐私

`novels/` 和 `knowledge_base/` 默认仅在本地保存，不提交 Git。`examples/mist_harbor/` 是可随源码分发的原创示例，采用本仓库 Apache-2.0 许可。
使用者需有权处理自己的输入文本，商业小说原文和生产知识库不随仓库提供。

构建和问答会把相关原文、知识及问题发送到配置的模型服务。
显式配置 Langfuse 后，问题、回答和运行元数据也会发送到指定观测服务。
开启 `--debug` 会在本地日志保存模型输出及工具内容，分享日志前请删去私有内容和凭证。

## 开发与许可

```bash
.venv/bin/python -m unittest discover -s tests -v
```

离线测试和 CI 不需要模型密钥。开发流程见 [CONTRIBUTING.md](CONTRIBUTING.md)，
安全报告与集成边界见 [SECURITY.md](SECURITY.md)，变更见 [CHANGELOG.md](CHANGELOG.md)。

本项目采用 [Apache License 2.0](LICENSE)。用户自备小说、生成的知识库和外部服务数据的权利由其来源决定。
