#!/bin/bash
# 兼容旧入口；推荐使用 ./novel-agent build
# 用法：./build.sh <txt文件> [novel_id] [--window N] [--min-window N] [--max-batches N]
# 示例：./build.sh ./novels/斗破苍穹.txt doupo --window 30000 --min-window 20000
# 测试：./build.sh ./novels/斗破苍穹.txt doupo --max-batches 10

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

exec ./novel-agent build "$@"
