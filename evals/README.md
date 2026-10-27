# 小说 QA 评测集

`qa_cases.jsonl` 保存从真实 badcase 固化下来的确定性检查。每个 case 支持：

- `required_any`：每个候选组至少命中一个表达；
- `forbidden_terms`：历史错误表达必须消失；
- `expected_chapters`：需要覆盖的章节；
- `require_valid_citations`：EvidenceBundle 引用必须通过；
- `min_evidence`：最少引用证据数。

仅校验数据集，不调用模型：

```bash
python -m src.evaluate
```

对已有回答做离线评分：

```bash
python -m src.evaluate --responses path/to/responses.jsonl
```

需要保存完整 JSON 报告时增加 `--output report.json`。只要有一个 case 失败或缺少回答，
命令就会在输出并保存整份报告后以退出码 `1` 结束；全部通过时退出码为 `0`。
数据集为空、输入文件无效或 live 资源预检失败时退出码为 `2`，错误原因写入标准错误流。

仓库中的原创短篇提供了无需模型和密钥的公开评分样例：

```bash
novel-agent evaluate \
  --dataset examples/mist_harbor/qa_cases.jsonl \
  --responses examples/mist_harbor/reference_responses.jsonl \
  --output reference-scoring.json
```

`reference_responses.jsonl` 是人工编写的参考回答夹具，用于验证评分和引用字段的处理；
它不是模型运行结果，不能代表 live 准确率。CI 在 Python 3.11 和 3.13 中执行源码安装、
离线单元测试、CLI 检查和这份人工回答的评分，不使用真实密钥或调用模型。

真实运行（会产生模型调用费用）：

```bash
python -m src.evaluate --live
```

`--live` 前应先完整构建对应 `novel_id` 的知识库，配置好模型，并让 `KNOWLEDGE_BASE_DIR`
指向知识库根目录。使用 `--case-id` 可以只选一个 case。

程序会在任何模型请求之前预检整组选中的 case：知识库目录必须存在，有可检索且非空的实体文件，
并有可读取、非空的数字编号章节 TXT；case 指定的 `expected_chapters` 也必须存在。
当前 Catalog 和具有有效链接的旧 Markdown 索引均可使用；旧 JSON 知识库或只有 progress 文件的目录不能通过预检。
任一本小说资源缺失、为空或无法读取时，程序统一列出问题并停止，不会先运行前面的 case。
预检只检查本地资源，不验证模型服务是否可用，也不判断知识内容的语义正确性。

离线回答 JSONL 每行包含 `case_id`，其余字段与 `run_qa_agent()` 返回值一致。
