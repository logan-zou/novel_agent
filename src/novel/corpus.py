"""小说原文库：章节保存、精确读取和可分页全文检索。"""

import json
import re
from pathlib import Path


_CHAPTER_RE = re.compile(r"^(\d+)\.txt$")


class CorpusStore:
    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.chapters_dir = self.base_dir / "chapters"

    def save_chapters(self, chapters: list[dict]) -> None:
        self.chapters_dir.mkdir(parents=True, exist_ok=True)
        for chapter in chapters:
            target = self.chapters_dir / f"{int(chapter['chapter_no'])}.txt"
            if not target.exists():
                target.write_text(
                    f"【第{chapter['chapter_no']}章 {chapter['title']}】\n\n{chapter['text']}",
                    encoding="utf-8",
                )

    def chapter_numbers(self) -> list[int]:
        if not self.chapters_dir.exists():
            return []
        values = []
        for path in self.chapters_dir.glob("*.txt"):
            match = _CHAPTER_RE.match(path.name)
            if match:
                values.append(int(match.group(1)))
        return sorted(values)

    def read_chapter(self, chapter_no: int) -> str:
        try:
            chapter_no = int(chapter_no)
        except (TypeError, ValueError):
            return f"[错误] 非法章节号：{chapter_no}"
        target = self.chapters_dir / f"{chapter_no}.txt"
        if not target.exists():
            return f"[错误] 章节不存在：第{chapter_no}章"
        return target.read_text(encoding="utf-8")

    def search(
        self,
        keywords: list[str],
        mode: str = "any",
        chapter_start: int | None = None,
        chapter_end: int | None = None,
        order: str = "asc",
        cursor: int = 0,
        limit: int = 20,
        context_lines: int = 1,
    ) -> str:
        terms = [str(keyword).strip().lower() for keyword in keywords if str(keyword).strip()]
        if not terms:
            return json.dumps({"error": "keywords 不能为空"}, ensure_ascii=False)
        if mode not in {"any", "all"}:
            return json.dumps({"error": "mode 只能是 any 或 all"}, ensure_ascii=False)
        if order not in {"asc", "desc"}:
            return json.dumps({"error": "order 只能是 asc 或 desc"}, ensure_ascii=False)

        numbers = self.chapter_numbers()
        if chapter_start is not None:
            numbers = [number for number in numbers if number >= int(chapter_start)]
        if chapter_end is not None:
            numbers = [number for number in numbers if number <= int(chapter_end)]
        if order == "desc":
            numbers.reverse()

        predicate = all if mode == "all" else any
        found = []
        for chapter_no in numbers:
            text = self.read_chapter(chapter_no)
            lowered = text.lower()
            if not predicate(term in lowered for term in terms):
                continue
            lines = text.splitlines()
            snippets = []
            for index, line in enumerate(lines):
                lowered_line = line.lower()
                if predicate(term in lowered_line for term in terms):
                    start = max(0, index - max(0, min(int(context_lines), 5)))
                    end = min(len(lines), index + max(0, min(int(context_lines), 5)) + 1)
                    snippets.append("\n".join(lines[start:end]).strip())
                    if len(snippets) >= 3:
                        break
            if not snippets:
                # 多个关键词可能分散在不同段落；返回各自首个命中附近的短片段。
                for index, line in enumerate(lines):
                    if any(term in line.lower() for term in terms):
                        snippets.append(line.strip())
                        if len(snippets) >= 3:
                            break
            found.append({"chapter": chapter_no, "snippets": snippets})

        cursor = max(0, int(cursor))
        limit = max(1, min(int(limit), 50))
        page = found[cursor : cursor + limit]
        next_cursor = cursor + len(page)
        payload = {
            "keywords": keywords,
            "mode": mode,
            "order": order,
            "matches": page,
            "has_more": next_cursor < len(found),
            "next_cursor": next_cursor if next_cursor < len(found) else None,
            "total_matches": len(found),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)
