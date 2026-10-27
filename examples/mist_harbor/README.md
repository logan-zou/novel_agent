# 雾港守灯簿：公开小样例

这是为本仓库原创的八章中文短篇，随仓库采用 Apache License 2.0 分发，不含第三方小说原文。固定知识库标识符为 `mist_harbor`。它用于验证安装、章节解析、构建和证据问答的最小闭环，不代表长篇小说的准确率或性能。

- `mist_harbor.txt`：完整原创语料；文件名也与并发构建推导出的知识库标识符一致。
- `qa_cases.jsonl`：14 道固定问题，覆盖直接事实、别名、部件关系、传闻被正式记录纠正、完整枚举、跨章原因与无法确认的问题；每题指定所需章节。
- `reference_responses.jsonl`：**手工编写的参考回答和离线评分夹具，不是真实模型运行结果**。证据通过本项目的原文搜索与引用协议生成，供测试重建核验。文件不包含模型调用、token 用量或耗时记录，也不能据此宣称 live 评测全通过。

本样例不附预建知识库。下面的构建和问答会实际使用你配置的模型；模型抽取的实体、回答措辞和评分可能与参考夹具不同。

## 从源码安装

在仓库根目录执行：

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

以下命令也都在仓库根目录执行。

## 不调用模型的验证

无需 API Key，可以先检查数据集、对手工参考回答评分，以及验证章节和证据：

```bash
.venv/bin/novel-agent evaluate --dataset examples/mist_harbor/qa_cases.jsonl
.venv/bin/novel-agent evaluate --dataset examples/mist_harbor/qa_cases.jsonl --responses examples/mist_harbor/reference_responses.jsonl
.venv/bin/python -m unittest discover -s tests -p test_public_example.py -v
```

参考回答的预期评分为 14/14；这只证明夹具与确定性检查一致。评分器检查关键词、禁词、指定章节和引用状态，不自动判断每个自然语言结论是否由证据支持。测试会重新解析八章原文，用真实 `CorpusStore` 和 `EvidenceCollector` 重建每条参考证据，检查其片段、章节和引用报告，并验证缺失关键事实或章节会导致评分失败。

## 实际构建和提问

复制根目录的配置样例，填写自己的模型地址、API Key 和支持流式 tool calling 的模型名：

```bash
cp .env.example .env
.venv/bin/novel-agent config
.venv/bin/novel-agent build examples/mist_harbor/mist_harbor.txt mist_harbor
.venv/bin/novel-agent ask mist_harbor "碎星镜和鸣潮灯是什么关系？" --json
```

**`build`、`ask` 和下面的 `--live` 评测都会调用模型并可能产生费用。** 默认知识库写入 `knowledge_base/mist_harbor/`；设置了 `KNOWLEDGE_BASE_DIR` 时，以该配置为准。这个短篇通常可放入一个默认批次，但实际调用次数和花费取决于模型行为，仓库没有预先承诺。

构建保存已完成批次的进度。中断或失败后，重新执行同一条 `build` 命令即可续跑；未提交批次需要重新处理，运行内的暂存不会跨进程保留。不要同时构建同一个知识库。为了先检查配置，可以在 `build` 命令后加 `--max-batches 1`；达到本次批次上限不一定表示整本完成，运行完整评测前应确认构建已完成。

## 用自己的模型运行评测

先完成上述构建。只试一道题：

```bash
.venv/bin/novel-agent evaluate --dataset examples/mist_harbor/qa_cases.jsonl --live --case-id mist-harbor-mirror-part
```

运行全部 14 题并保存报告到临时文件：

```bash
.venv/bin/novel-agent evaluate --dataset examples/mist_harbor/qa_cases.jsonl --live --output /tmp/mist_harbor_live_report.json
```

检查失败项时，应同时读回答和它引用的章节。固定词语检查可能漏过错误表达，也可能拒绝意思正确但措辞不同的回答；手工参考答案是核对起点。尤其要区分第二章的传闻与第四、五章的正式记录，完整列出第六章的三名参与者，并在蓝船去向问题上承认未知。

如果分享自己的实测，请注明代码版本、语料是否改动、模型及配置、运行日期、构建/问答总输入输出 token 和实测耗时。仓库中的参考夹具没有这些实测数据。费用应另行注明账单来源或估算方法，不能把参考夹具的离线通过率当作模型效果或成本数据。
