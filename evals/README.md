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

真实运行（会产生模型调用费用）：

```bash
python -m src.evaluate --live
```

`--live` 要求对应 `novel_id` 已经用当前 Markdown 知识库格式完整构建；旧 JSON 知识库或只有 progress 文件的目录不能用于评测。

离线回答 JSONL 每行包含 `case_id`，其余字段与 `run_qa_agent()` 返回值一致。
