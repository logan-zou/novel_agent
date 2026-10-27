# 小说问答系统设计

## 1. 目标

项目当前只聚焦一件事：**先构建一份可回源的小说知识库，再基于知识库和原文回答事实问题。**

效果用五个维度衡量：

1. 事实正确。
2. 枚举完整。
3. 时间状态正确。
4. 别名、部件和主体边界正确。
5. 结论有实际读取的实体或章节支持。

概括写作、开放推理、多 Agent 协作和通用 workflow 暂不进入核心架构。

## 2. 总体架构

```text
build.py / evaluate.py / 外部示例服务       应用入口
                  │
                  ▼
┌───────────────────────────────────────┐
│ novel/ 小说业务                       │
│ corpus → knowledge → builder          │
│                 retrieval → qa        │
│                          evidence     │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│ harness/ + llm.py                     │
│ 模型/工具循环、重试、上下文、token       │
└───────────────────────────────────────┘
```

业务层决定“什么是小说知识、怎样找证据、何时回原文”；Runtime 只保证这些动作能稳定执行。

## 3. 两层记忆

### 原文层

`chapters/{chapter_no}.txt` 永久保存完整章节，是最终事实来源。

`CorpusStore` 提供：

- 保存和读取章节。
- `any/all` 关键词模式。
- 章节范围过滤。
- 正序/倒序检索。
- cursor 分页。
- 搜索片段不足时读取完整章节。

### 知识层

知识库是原文的语义索引，按角色、设定、事件、故事弧组织：

```text
knowledge_base/{novel_id}/
├── chapters/{chapter_no}.txt
├── characters/{角色名}.md
├── settings/{领域}/{概念名}.md
├── plot/events/{事件名}.md
├── plot/arcs/{故事弧名}.md
├── catalog.jsonl
├── progress.json
└── token_usage.jsonl
```

新知识库采用“一实体一文件”。每个文件由代码生成元数据和一级标题，LLM 只提交正文内容。`catalog.jsonl` 从实体文件自动生成，Agent 不直接编辑索引。

实体元数据包括：

- `kind`
- `name`
- `aliases`
- `summary`
- `source_chapters`
- `domain`（设定使用）
- `ref`

实体正文仍使用自然的 Markdown 和 wikilink。新链接只指向整份实体文件，不再依赖锚点。

## 4. 离线知识构建

```text
小说 txt
  ↓ 章节识别与内容指纹核验
CorpusStore 保存原文
  ↓ 按全书进度使用渐进字符窗口组成批次（默认 30K → 20K）
Extraction Agent
  ├─ search_entities：写前查重
  ├─ read_entities：一次读取最多 8 个旧实体
  ├─ upsert_entities：一次暂存最多 8 个完整实体
  ├─ search_source：只核实静态设定
  └─ read_chapter：补足原文语境
  ↓
BuildSession 校验并提交
  ↓
自动重建 catalog.jsonl
  ↓
标记批次完成
```

### 抽取 Agent 的职责

- 识别值得长期检索的角色、设定、事件和故事弧。
- 把互不依赖的搜索、读取和写入放进同一模型轮次批量调用，按实体依赖分阶段推进。
- 合并已有实体，而不是重复创建。
- 记录角色当前状态和真正重要的变化。
- 处理别名、部件、化身和主体关系。
- 按信息权威判断补充、纠正、冲突和推测。
- 为实体维护准确的来源章节并集。

### 代码的职责

- 决定文件路径和 Markdown 外壳。
- 暂存本批实体，使批内读写一致。
- 校验实体摘要、来源章节、长度和链接目标。
- 统一提交文件并自动生成 Catalog。
- 按全书已处理文本比例逐渐缩小后续批次；旧任务只重排未处理尾部，并固化策略以保证断点稳定。
- 构建先核对章号、标题、正文的内容指纹及已保存原文，防止续跑混用修订版文本；已完成任务跳过前也要核验。
- 抽取常规最多 60 轮，40 轮后检查停滞；另有 8 轮独立收尾、最多一次 24 轮同章续跑及后续 8 轮收尾，连同最终校验修复的绝对上限为 140 轮。详见 [构建与恢复](BUILDING.md)。
- 达到运行边界时按章节长度自动二分当前批次；每个子批次完成后立即记录断点，全部完成后再标记原批次完成。
- 只有全部批次 `0..N-1` 完成时才写入 `status=completed`。

去重 Agent、周期 compact、聚合条目晋升、手工索引更新均不再进入构建流程。

## 5. 在线问答

QA Agent 只有四个只读工具：

```text
search_entities(keywords, entity_types, limit)
read_entity(ref, section?)
search_source(keywords, mode, chapter range, order, cursor, limit)
read_chapter(chapter_no)
```

每一轮模型调用使用用户配置的 OpenAI-compatible 流式接口。Provider 返回
`reasoning_content` 时可以逐块转发；它是可选展示能力，不参与回答正确性判断。
独立示例服务可将推理步骤展开显示，并在生成完成后自动折叠。

工具调用使用 `tool_start/tool_end` 两阶段事件：执行时展开并展示完整调用参数，
结束后补入完整返回结果并自动折叠。工具卡片始终可以再次点开，不把小说业务
工具的参数或结果塞进通用 Runtime 的展示逻辑。

### 快速路径

具体角色、设定或事件问题：搜索实体 → 读取目标 → 按需展开少量关系 → 原文核验 → 回答。

### 深度核查路径

枚举、数量、等级、真实身份、主体/部件关系、结局和特定时间点问题：

1. 从知识库收集候选和别名。
2. 回原文核实关键结论。
3. 最终口径优先倒序搜索，时间点问题限制章节范围。
4. 主动搜索反例和同义名称，避免漏项或重复计数。
5. 片段语境不足时读取完整章节。

原文中的信息权威通常为：终局揭秘/全知叙述/正式记录 > 知情角色明确说明 > 普通转述 > 猜测和感叹。时间更晚只是辅助信号，不自动代表更可靠。

## 6. 证据协议

`read_entity` 和 `search_entities` 用于定位，`read_chapter` 用于补足语境；最终答案的
可展示脚注只允许引用 `search_source` 返回的原文证据。

最终答案格式：

```text
时空之眼是烛九阴的左眼。[[cite:EV001]]
另一处原文也印证了这一点。[[cite:EV002]][[cite:EV003]]
```

系统校验：

- 证据 ID 确实来自本次问答。
- 只能引用 `search_source` 取得的原文证据。
- 引用必须使用 `[[cite:EVxxx]]` 并紧跟对应结论，不再使用证据/章节 footer。
- 完全没有证据时，只允许明确回答信息不足。

工程侧按引用首次出现顺序生成脚标，并从原文命中位置截取前后各 500 字放入
`citations[].excerpt`；前端点击脚标后在浮窗中展示。返回值保留 `sources` 和
`evidence` 兼容字段，同时增加语义更明确的 `citations`。

当前校验不判断每个自然语言结论是否被证据语义蕴含，这部分通过评测集和人工 badcase 继续完善。

## 7. 评测

`evals/qa_cases.jsonl` 固化真实问题，评分覆盖：

- 必须出现的关键事实。
- 禁止出现的错误事实。
- 预期章节。
- 引用有效性。
- 最少证据数。

```bash
python -m src.evaluate
python -m src.evaluate --responses responses.jsonl
python -m src.evaluate --live
```

默认只校验数据集；`--live` 才产生真实模型费用。

## 8. 旧知识库兼容

已有知识库仍包含 `_index.md`、聚合文件和 `#锚点`。新 `KnowledgeStore` 提供只读兼容：

- 搜索不到新 Catalog 时继续搜索旧索引。
- `read_entity(path#anchor)` 精确抽取对应 Markdown 段落，不再因整个文件超长而丢失后半部分实体。

旧数据不会在后台自动迁移。下一次重新构建时才生成一实体一文件和 `catalog.jsonl`。

## 9. 当前代码

```text
src/
├── build.py
├── evaluate.py
├── extraction_agent.py       # 兼容入口
├── qa_agent.py               # 兼容入口
├── harness/
│   ├── runtime.py
│   └── usage.py
├── llm.py
├── tracing.py                # 可选 Langfuse
├── usage_tracker.py
├── preprocess.py
└── novel/
    ├── corpus.py
    ├── knowledge.py
    ├── builder.py
    ├── retrieval.py
    ├── qa.py
    └── evidence.py
```
