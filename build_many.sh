#!/bin/bash
# 默认预览任务；修改下面配置后用 bash build_many.sh --run 实际构建。
set -euo pipefail
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# ── 配置区：相对路径以本脚本所在目录为准 ──
# 可以填写多个 TXT 文件，也可以填写目录（扫描第一层所有 TXT）。
NOVELS=(
  "./examples/mist_harbor/mist_harbor.txt"
)

WORKERS=1                 # 同时处理的小说数
WINDOW=30000             # 初始批次字符数上限
MIN_WINDOW=20000         # 后期批次字符数上限
MAX_BATCHES=1            # 每本本次最多处理的批次数；0 为不限
DEBUG=false              # true 会把模型输出、工具参数和原文片段写入日志
PROGRESS_INTERVAL=30     # 长批次每隔多少秒提醒进度；批次变化约 1 秒内显示
DRY_RUN=true             # true：只展示任务；--run 显式启用模型调用
LOG_DIR="./knowledge_base/_build_logs" # 日志根目录，每次运行新建子目录
# 模型地址、密钥和模型名称继续读取项目 .env。
# ── 配置区结束 ──

cd "$SCRIPT_DIR"
EXTRA_ARGS=()
for argument in "$@"; do
  case "$argument" in
    --run) DRY_RUN=false ;;
    --dry-run) DRY_RUN=true ;;
    *) EXTRA_ARGS+=("$argument") ;;
  esac
done
ARGS=(
  "${NOVELS[@]}"
  --workers "$WORKERS"
  --window "$WINDOW"
  --min-window "$MIN_WINDOW"
  --max-batches "$MAX_BATCHES"
  --log-dir "$LOG_DIR"
  --progress-interval "$PROGRESS_INTERVAL"
)
if [[ "$DEBUG" == "true" ]]; then
  ARGS+=(--debug)
fi
if [[ "$DRY_RUN" == "true" ]]; then
  ARGS+=(--dry-run)
fi
# 可追加参数，例如 bash build_many.sh --run --max-batches 0 --workers 2。
exec "$SCRIPT_DIR/novel-agent" build-many "${ARGS[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
