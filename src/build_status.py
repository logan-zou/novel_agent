"""单本构建向批量启动器报告状态；不影响知识库断点文件。"""

import json
import os
from pathlib import Path


def report_status(stage: str, **details) -> None:
    target = os.environ.get("NOVEL_BUILD_STATUS_FILE")
    if not target:
        return
    path = Path(target)
    temporary = path.with_suffix(".tmp")
    try:
        temporary.write_text(json.dumps({"stage": stage, **details}, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
    except OSError:
        # 展示状态失败不应中止真实构建。
        pass
