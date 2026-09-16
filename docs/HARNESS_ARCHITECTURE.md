# 最小 Agent Runtime 边界

## 为什么只保留最小 Runtime

项目目标是小说知识构建和有依据的问答，不是建设通用 Agent 平台。Runtime 只保留两个业务 Agent 都确实需要、且重复实现容易出错的机制。

```text
小说 Builder / QA
        │ AgentConfig + Tool[] + input
        ▼
    AgentRuntime
        │ ModelRequest
        ▼
 OpenAI-compatible LLM
```

## Runtime 负责

- 构造模型请求。
- 执行工具调用并把结果写回对话。
- 处理未知工具、参数 JSON 错误和工具异常。工具参数会在写入对话历史前按整轮校验；任一参数非法时整轮工具均不执行，避免半写入和把非法消息回传给模型接口。Runtime 会要求模型缩短参数后有限重试，仍失败则抛出可由构建层拆批处理的 `AgentLimitError`。
- 恢复被截断的文本或工具调用。
- 使用进度感知的迭代边界：40 轮软上限、60 轮硬上限，并识别重复调用和连续工具错误。
- 上下文接近上限时省略较早工具结果。
- 按 Agent loop 稳定前缀自行拆分缓存读与非缓存读，并汇总输出 token；不采用模型 API 的缓存明细。
- 调用业务提供的最终输出校验函数。
- 发出少量字符串事件供终端和 SSE 展示。

Runtime 的代码集中在 `src/harness/runtime.py`，主要契约只有：

- `AgentConfig`
- `Tool`
- `ChatProvider`
- `ModelRequest` / `ModelResponse`
- `RunResult`
- `AgentRuntime`

## Runtime 不负责

- 章节、角色、设定和情节定义。
- 知识库路径和存储结构。
- 什么问题需要回原文。
- 信息权威和时间判断。
- 原文 Evidence ID、内联引用协议和脚标上下文。
- 构建批次暂存和实体校验。
- Langfuse 或其他具体观测产品。

这些规则全部留在 `src/novel/`。

## 一次运行

```text
system prompt + input
        ↓
     LLM 调用 ───────────────► token 记录
        ↓
  ┌─────┴─────┐
  │ tool_calls│ stop/length
  ▼           ▼
执行领域工具   截断恢复或输出校验
  │           │
  └─────► 下一轮 / RunResult
```

上下文裁剪后不保存 artifact。小说数据本来就在 Corpus 和 Knowledge Store 中，Agent 仍需要旧信息时重新调用对应业务工具即可。

## 明确移除的机制

本次轻量化删除：

- 旧版 `src/harness/` 中的多文件平台化抽象（当前目录只保留 Runtime 与 usage 两个核心模块）。
- 通用 `RunContext` / `StepContext` / `RunEvent` 类型系统。
- Observer 协议和 Provider Adapter 层。
- append-only run ledger。
- artifact 内容寻址存储和 `read_artifact`。
- 任意步骤自动恢复。
- 通用工具权限/副作用模型。

保留的可靠性边界是：

- 构建按批次断点续跑。
- 非法工具参数不会进入下一轮上下文；同轮调用原子拒绝并有限重试，持续失败的抽取批次进入自动拆分。
- 达到 Agent 运行边界的批次会按章节自动拆分，子批次也有独立的轻量断点。
- `BuildSession` 在批次结束前暂存实体并校验。
- 单文件使用临时文件替换；提交失败时回滚已写实体。
- QA 最终答案必须通过原文 Evidence ID 和 `[[cite:EVxxx]]` 内联格式校验。

这几项直接保护知识库一致性和回答可信度，因此属于当前产品真正需要的稳定性能力。
