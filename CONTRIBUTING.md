# 参与贡献

项目聚焦小说知识构建与有原文依据的事实问答。欢迎提交可复现的错误、评测案例和小范围修复。

## 本地开发

需要 Python 3.11+。从仓库源码安装：

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python -m unittest discover -s tests -v
./novel-agent evaluate --dataset examples/mist_harbor/qa_cases.jsonl --responses examples/mist_harbor/reference_responses.jsonl
bash build_many.sh
```

这些检查不需要模型密钥或小说数据。CI 使用 Python 3.11 和 3.13 运行源码安装及离线检查。
真实模型实验须显式配置 `.env`，并使用 `--live` 或实际构建命令，调用会产生费用。

## 提交问题或改动

- 错误报告附命令、Python 版本、复现步骤和期望/实际行为。模型相关问题注明服务和模型，删除密钥及私有文本。
- 优先使用原创小片段复现。请勿上传未经授权的小说、知识库、完整生产日志或 `.env`。
- 修复行为缺陷时添加能重现问题的回归测试；文档或展示文案无需增加重复实现细节的测试。
- 保持 `src/harness/` 与小说领域、模型 SDK 和观测服务解耦；领域规则放在 `src/novel/`。
- Pull Request 说明问题、最终行为和实际验证。对真实模型未做过的验证应明确注明。

## 发布检查

本项目通过源码仓库分发，不发布 PyPI 包。首次发布前整理本地修改和新增文件，保证新检出目录能完成上述检查。
核对原文、知识库和 `.env` 没有进入提交；补充变更记录，在完整验证后的提交上创建版本标签。
公开样例的人工参考答案只用于检查评分流程，发布模型成绩时应同时提供模型、配置、原始回答和实验用量/时长。
