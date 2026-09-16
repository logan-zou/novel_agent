"""项目所需的最小 Agent 运行时。

只负责模型/工具循环、截断恢复、上下文裁剪、输出校验和 token 汇总；
不知道小说、章节、知识库或证据等领域概念。
"""

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Protocol
from uuid import uuid4

from src.harness.usage import accumulate_usage, attribute_agent_loop_usage, empty_usage


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str
    type: str = "function"


@dataclass(frozen=True)
class AssistantMessage:
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    reasoning_content: str = ""


@dataclass(frozen=True)
class ModelRequest:
    model: str
    messages: list[dict]
    tools: list[dict]
    max_tokens: int
    temperature: float
    reasoning_effort: str = "low"
    on_reasoning_delta: Callable[[str], None] | None = None
    on_content_delta: Callable[[str], None] | None = None


@dataclass(frozen=True)
class ModelResponse:
    message: AssistantMessage
    finish_reason: str
    usage: Any = None


class ChatProvider(Protocol):
    def complete(self, request: ModelRequest) -> ModelResponse: ...


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable[..., str]
    progress_detector: Callable[[str], bool] | None = None

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True)
class AgentConfig:
    name: str
    model: str
    system_prompt: str
    max_iterations: int
    max_output_tokens: int
    temperature: float
    context_window_tokens: int
    max_truncation_retries: int = 3
    max_validation_retries: int = 2
    reasoning_effort: str = "low"
    soft_iteration_limit: int | None = None
    progress_grace_iterations: int = 0
    max_identical_tool_calls: int = 0
    max_consecutive_tool_errors: int = 0
    max_tool_argument_retries: int = 2


@dataclass(frozen=True)
class RunResult:
    output: str
    usage: dict[str, int]
    steps: int
    run_id: str


EventHandler = Callable[[str, dict], None]
UsageRecorder = Callable[[Any], None]
OutputValidator = Callable[[str], str | None]


class OutputValidationError(RuntimeError):
    pass


class AgentLimitError(RuntimeError):
    """Agent 达到迭代/停滞边界；携带轻量诊断与已消耗用量。"""

    def __init__(
        self,
        reason: str,
        *,
        steps: int,
        limit: int,
        usage: dict[str, int],
        tool_counts: dict[str, int],
        successful_tool_calls: int,
        failed_tool_calls: int,
        repeated_tool_calls: int,
    ) -> None:
        self.reason = reason
        self.steps = steps
        self.limit = limit
        self.usage = dict(usage)
        self.tool_counts = dict(tool_counts)
        self.successful_tool_calls = successful_tool_calls
        self.failed_tool_calls = failed_tool_calls
        self.repeated_tool_calls = repeated_tool_calls
        counts = ", ".join(
            f"{name} {count}次" for name, count in sorted(tool_counts.items())
        ) or "无工具调用"
        super().__init__(
            f"Agent {reason}（已执行 {steps} 轮；工具统计：{counts}；"
            f"成功 {successful_tool_calls} / 失败 {failed_tool_calls} / "
            f"重复 {repeated_tool_calls}）"
        )


class MaxIterationsExceeded(AgentLimitError):
    pass


class AgentLoopStalled(AgentLimitError):
    pass


class MalformedToolArguments(AgentLimitError):
    """模型连续生成非法工具参数，Harness 无法在当前输入上安全恢复。"""

    pass


class AgentRuntime:
    def __init__(self, provider: ChatProvider) -> None:
        self.provider = provider

    def run(
        self,
        config: AgentConfig,
        input_messages: list[dict],
        tools: list[Tool] | tuple[Tool, ...] = (),
        *,
        on_event: EventHandler | None = None,
        usage_recorder: UsageRecorder | None = None,
        output_validator: OutputValidator | None = None,
        run_id: str | None = None,
    ) -> RunResult:
        run_id = run_id or uuid4().hex
        tool_map = {tool.name: tool for tool in tools}
        if len(tool_map) != len(tools):
            raise ValueError("工具名称重复")

        messages = [{"role": "system", "content": config.system_prompt}]
        messages.extend(dict(message) for message in input_messages)
        usage_total = empty_usage()
        accumulated_text = ""
        truncation_retries = 0
        validation_retries = 0
        previous_prompt_tokens: int | None = None
        tool_counts: Counter[str] = Counter()
        seen_tool_signatures: Counter[str] = Counter()
        successful_tool_calls = 0
        failed_tool_calls = 0
        repeated_tool_calls = 0
        consecutive_tool_errors = 0
        last_progress_step: int | None = None
        tool_argument_retries = 0

        def limit_error(error_type, reason: str, steps: int, limit: int):
            return error_type(
                reason,
                steps=steps,
                limit=limit,
                usage=usage_total,
                tool_counts=dict(tool_counts),
                successful_tool_calls=successful_tool_calls,
                failed_tool_calls=failed_tool_calls,
                repeated_tool_calls=repeated_tool_calls,
            )

        def record_executions(executions: list[dict], step: int) -> None:
            nonlocal successful_tool_calls, failed_tool_calls
            nonlocal repeated_tool_calls, consecutive_tool_errors, last_progress_step
            for execution in executions:
                name = execution["name"]
                signature = execution["signature"]
                tool_counts[name] += 1
                seen_tool_signatures[signature] += 1
                if seen_tool_signatures[signature] > 1:
                    repeated_tool_calls += 1
                if execution["error"]:
                    failed_tool_calls += 1
                    consecutive_tool_errors += 1
                else:
                    successful_tool_calls += 1
                    consecutive_tool_errors = 0
                if execution["made_progress"]:
                    last_progress_step = step

                repeat_limit = config.max_identical_tool_calls
                if repeat_limit and seen_tool_signatures[signature] >= repeat_limit:
                    raise limit_error(
                        AgentLoopStalled,
                        f"重复相同工具调用达到 {repeat_limit} 次",
                        step,
                        repeat_limit,
                    )
                error_limit = config.max_consecutive_tool_errors
                if error_limit and consecutive_tool_errors >= error_limit:
                    raise limit_error(
                        AgentLoopStalled,
                        f"连续工具错误达到 {error_limit} 次",
                        step,
                        error_limit,
                    )

        _emit(on_event, "run_started", {"run_id": run_id, "agent": config.name})
        try:
            for step in range(1, config.max_iterations + 1):
                soft_limit = config.soft_iteration_limit
                if soft_limit and step > soft_limit:
                    grace = max(0, config.progress_grace_iterations)
                    no_recent_progress = (
                        last_progress_step is None
                        or step - 1 - last_progress_step >= grace
                    )
                    if no_recent_progress:
                        raise limit_error(
                            MaxIterationsExceeded,
                            f"达到软迭代上限 {soft_limit}，且最近 {grace} 轮没有写入进展",
                            step - 1,
                            soft_limit,
                        )
                _emit(on_event, "model_started", {"run_id": run_id, "step": step})
                response = self.provider.complete(
                    ModelRequest(
                        model=config.model,
                        messages=messages,
                        tools=[tool.schema for tool in tools],
                        max_tokens=config.max_output_tokens,
                        temperature=config.temperature,
                        reasoning_effort=config.reasoning_effort,
                        on_reasoning_delta=lambda delta, current_step=step: _emit(
                            on_event,
                            "model_reasoning_delta",
                            {"run_id": run_id, "step": current_step, "delta": delta},
                        ),
                        on_content_delta=lambda delta, current_step=step: _emit(
                            on_event,
                            "model_content_delta",
                            {"run_id": run_id, "step": current_step, "delta": delta},
                        ),
                    )
                )
                normalized = attribute_agent_loop_usage(response.usage, previous_prompt_tokens)
                message = response.message
                if normalized:
                    previous_prompt_tokens = normalized["prompt_tokens"]
                    accumulate_usage(usage_total, normalized)
                    if usage_recorder:
                        usage_recorder(normalized)
                _emit(
                    on_event,
                    "model_completed",
                    {
                        "run_id": run_id,
                        "step": step,
                        "finish_reason": response.finish_reason,
                        "reasoning_content": message.reasoning_content,
                        "usage": normalized,
                    },
                )

                if normalized and normalized["total_tokens"]:
                    compacted = trim_old_tool_results(
                        messages,
                        normalized["total_tokens"],
                        config.context_window_tokens,
                    )
                    if compacted:
                        _emit(
                            on_event,
                            "context_compacted",
                            {"run_id": run_id, "step": step, "messages_compacted": compacted},
                        )

                if response.finish_reason == "tool_calls":
                    accumulated_text = ""
                    argument_errors = _tool_argument_errors(message.tool_calls)
                    if argument_errors:
                        tool_argument_retries += 1
                        will_retry = (
                            tool_argument_retries <= config.max_tool_argument_retries
                        )
                        _emit(
                            on_event,
                            "tool_arguments_rejected",
                            {
                                "run_id": run_id,
                                "step": step,
                                "attempt": tool_argument_retries,
                                "max_retries": config.max_tool_argument_retries,
                                "will_retry": will_retry,
                                "errors": argument_errors,
                            },
                        )
                        details = "; ".join(
                            f"{item['name'] or '<unknown>'}: {item['error']}"
                            for item in argument_errors
                        )
                        if not will_retry:
                            raise limit_error(
                                MalformedToolArguments,
                                (
                                    "工具参数无法恢复：连续生成非法 JSON，"
                                    f"已用完 {config.max_tool_argument_retries} 次修复机会"
                                    f"（{details}）"
                                ),
                                step,
                                config.max_tool_argument_retries,
                            )
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "上一轮工具调用参数不是合法 JSON，本轮没有执行任何工具。"
                                    f"错误：{details}。请重新生成这一轮工具调用，参数必须是"
                                    "完整的 JSON object；如果内容较长，请减少单次工具项数量并"
                                    "分多轮调用。"
                                ),
                            }
                        )
                        continue

                    truncation_retries = 0
                    tool_argument_retries = 0
                    messages.append(_assistant_dict(message))
                    executions = _execute_tools(
                        message.tool_calls, tool_map, messages, on_event, run_id, step
                    )
                    record_executions(executions, step)
                    continue

                if response.finish_reason == "length":
                    truncation_retries += 1
                    if truncation_retries > config.max_truncation_retries:
                        raise RuntimeError("模型输出连续截断，超过恢复次数上限")
                    if message.tool_calls:
                        broken = [call for call in message.tool_calls if _parse_json(call.arguments)[1]]
                        if broken:
                            messages.append(
                                {
                                    "role": "user",
                                    "content": "工具参数因输出截断而不完整。请缩短单次内容并重新调用工具。",
                                }
                            )
                        else:
                            messages.append(_assistant_dict(message))
                            executions = _execute_tools(
                                message.tool_calls, tool_map, messages, on_event, run_id, step
                            )
                            record_executions(executions, step)
                    else:
                        partial = message.content or ""
                        accumulated_text += partial
                        if partial:
                            messages.append({"role": "assistant", "content": partial})
                        messages.append({"role": "user", "content": "请从截断处继续输出。"})
                    continue

                output = accumulated_text + (message.content or "")
                validation_error = output_validator(output) if output_validator else None
                if validation_error:
                    validation_retries += 1
                    _emit(
                        on_event,
                        "output_rejected",
                        {"run_id": run_id, "step": step, "reason": validation_error},
                    )
                    if validation_retries > config.max_validation_retries:
                        raise OutputValidationError(validation_error)
                    messages.extend(
                        [
                            {"role": "assistant", "content": output},
                            {
                                "role": "user",
                                "content": (
                                    "上一版最终输出未通过校验：\n"
                                    f"{validation_error}\n请修正后重新输出完整结果。"
                                ),
                            },
                        ]
                    )
                    accumulated_text = ""
                    continue

                result = RunResult(output, usage_total, step, run_id)
                _emit(on_event, "run_completed", {"run_id": run_id, "step": step})
                return result

            raise limit_error(
                MaxIterationsExceeded,
                f"达到硬迭代上限 {config.max_iterations}",
                config.max_iterations,
                config.max_iterations,
            )
        except Exception as error:
            _emit(on_event, "run_failed", {"run_id": run_id, "error": str(error)})
            raise


def trim_old_tool_results(
    messages: list[dict],
    current_tokens: int,
    context_window_tokens: int,
    *,
    trigger_ratio: float = 0.6,
    keep_recent: int = 6,
) -> int:
    """裁剪较早的工具结果；需要时 Agent 可重新调用领域工具。"""
    if current_tokens < context_window_tokens * trigger_ratio:
        return 0
    indices = [i for i, message in enumerate(messages) if message.get("role") == "tool"]
    targets = indices[:-keep_recent] if len(indices) > keep_recent else []
    compacted = 0
    for index in targets:
        content = messages[index].get("content")
        if isinstance(content, str) and len(content) > 200 and not content.startswith("[旧工具结果已省略"):
            messages[index]["content"] = (
                f"[旧工具结果已省略，原长度 {len(content)} 字；如仍需要请重新调用该工具]"
            )
            compacted += 1
    return compacted


def _execute_tools(calls, tool_map, messages, on_event, run_id, step) -> list[dict]:
    executions = []
    for call in calls:
        arguments, error = _parse_json(call.arguments)
        _emit(
            on_event,
            "tool_started",
            {
                "run_id": run_id,
                "step": step,
                "call_id": call.id,
                "name": call.name,
                "arguments": arguments,
                "arguments_raw": call.arguments,
                "arguments_error": error,
            },
        )
        execution_error = bool(error)
        if error:
            result = f"[错误] 工具参数解析失败：{error}"
        elif call.name not in tool_map:
            result = f"[错误] 未知工具：{call.name}"
            execution_error = True
        else:
            try:
                result = str(tool_map[call.name].handler(**arguments))
            except Exception as tool_error:
                result = f"[错误] 工具 {call.name} 执行失败：{tool_error}"
                execution_error = True
        if result.lstrip().startswith("[错误]"):
            execution_error = True
        progress_detector = (
            tool_map[call.name].progress_detector if call.name in tool_map else None
        )
        try:
            made_progress = bool(progress_detector and progress_detector(result))
        except Exception:
            made_progress = False
        messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
        _emit(
            on_event,
            "tool_completed",
            {
                "run_id": run_id,
                "step": step,
                "call_id": call.id,
                "name": call.name,
                "arguments": arguments,
                "arguments_raw": call.arguments,
                "result": result,
                "result_len": len(result),
                "error": execution_error,
                "made_progress": made_progress,
            },
        )
        signature = f"{call.name}:{json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}"
        executions.append(
            {
                "name": call.name,
                "signature": signature,
                "error": execution_error,
                "made_progress": made_progress,
            }
        )
    return executions


def _parse_json(raw: str) -> tuple[dict, str | None]:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError as error:
        return {}, str(error)
    if not isinstance(value, dict):
        return {}, "工具参数必须是 JSON object"
    return value, None


def _tool_argument_errors(calls) -> list[dict[str, str]]:
    """在写入对话历史前校验整轮工具参数，避免把非法 JSON 回传给 API。"""
    errors = []
    for call in calls:
        _, error = _parse_json(call.arguments)
        if error:
            errors.append({"call_id": call.id, "name": call.name, "error": error})
    return errors


def _assistant_dict(message: AssistantMessage) -> dict:
    value: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
    if message.reasoning_content:
        value["reasoning_content"] = message.reasoning_content
    if message.tool_calls:
        value["tool_calls"] = [
            {
                "id": call.id,
                "type": call.type,
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in message.tool_calls
        ]
    return value


def _emit(handler: EventHandler | None, event_type: str, data: dict) -> None:
    if handler:
        handler(event_type, data)
