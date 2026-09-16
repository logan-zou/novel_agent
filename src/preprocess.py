"""
预处理层：章节划分 + 批次聚合。
两个纯函数，无副作用，不依赖任何外部服务。
"""

import re
from pathlib import Path

# 中文小说常见编码，按优先级依次尝试
_ENCODINGS = ["utf-8", "gb18030", "gbk", "gb2312", "big5"]

# 解码成功后，用前 2000 字符里中文字符的比例来判断是否真正正确解码
# 中文小说正文比例一般 > 30%
_MIN_CHINESE_RATIO = 0.2


def _read_text(txt_path: str) -> str:
    """
    自动检测编码读取文件。

    先按优先级用 errors='strict' 严格解码逐个尝试——一份字节流能被某个多字节
    编码严格解码成功，本身就是这份文件是该编码的强证据（尤其是 UTF-8，其字节
    序列约束很强，随机字节几乎不可能被误判为合法 UTF-8）。
    只有全部编码都严格解码失败（文件本身就有少量非法字节），才回退到用
    errors='ignore' 宽松解码 + 中文字符占比打分，尽量拿到可用文本而不是报错。

    注意：不能反过来先用 errors='ignore' 宽松解码再比占比——GBK/GB18030 等
    编码宽松解码 UTF-8 字节流时，容易把大段字节巧合映射到中文区，占比反而
    比正确的 UTF-8 解码结果更高，导致误判编码、正则完全匹配不到章节。
    """
    raw = Path(txt_path).read_bytes()

    for enc in _ENCODINGS:
        try:
            text = raw.decode(enc, errors="strict")
        except (LookupError, UnicodeDecodeError):
            continue
        print(f"[预处理] 文件编码：{enc}（严格解码成功）")
        return text.replace('\r\n', '\n').replace('\r', '\n')

    # 严格解码全部失败，回退到宽松解码 + 占比打分
    best_enc, best_text, best_ratio = "utf-8", "", 0.0
    for enc in _ENCODINGS:
        try:
            text = raw.decode(enc, errors="ignore")
            ratio = _chinese_ratio(text[:2000])
            if ratio > best_ratio:
                best_ratio, best_enc, best_text = ratio, enc, text
        except LookupError:
            continue

    if best_ratio < _MIN_CHINESE_RATIO:
        print(f"[预处理] 警告：中文字符占比偏低（{best_ratio:.1%}），可能编码识别有误")
    print(f"[预处理] 文件编码：{best_enc}（严格解码均失败，回退宽松解码，中文占比 {best_ratio:.1%}）")
    # 统一换行符（兼容 \r\n、\r\r\n 等混合格式）
    return best_text.replace('\r\n', '\n').replace('\r', '\n')


def _chinese_ratio(text: str) -> float:
    """计算字符串中中文字符（CJK 统一汉字）的占比。"""
    if not text:
        return 0.0
    chinese = sum(1 for c in text if '一' <= c <= '鿿')
    return chinese / len(text)


# ── 章节划分 ─────────────────────────────────────────────────────────────────

# 匹配常见中文网文章节标题，支持：
#   第123章 标题      第123章
#   第一百二十三章 标题   第一百二十三章
# 行首允许有空格（部分 txt 文件章节标题前有缩进）
_CHAPTER_PATTERN = re.compile(
    r'^ *(第[零一二三四五六七八九十百千万\d]+章[^\n]*)',
    re.MULTILINE
)

# 最少识别到多少章才认为正则有效
_MIN_CHAPTERS_THRESHOLD = 5

# 正则失败时的回退：每隔多少行切一章
_FALLBACK_LINES_PER_CHAPTER = 200


def split_chapters(txt_path: str) -> list[dict]:
    """
    解析 txt 文件，按章节标题切分。

    返回：
        [
            {"chapter_no": 1, "title": "天才少年", "text": "正文内容..."},
            ...
        ]

    若正则匹配到的章节数不足阈值，回退到等长行数切分。
    """
    text = _read_text(txt_path)
    chapters = _split_by_pattern(text)

    if len(chapters) >= _MIN_CHAPTERS_THRESHOLD:
        return chapters

    # 回退：等长行数切分
    print(f"[预处理] 正则仅识别到 {len(chapters)} 章，回退到按行数切分（每 {_FALLBACK_LINES_PER_CHAPTER} 行一章）")
    return _split_by_lines(text)


def _split_by_pattern(text: str) -> list[dict]:
    """按章节标题正则切分。"""
    matches = list(_CHAPTER_PATTERN.finditer(text))
    if not matches:
        return []

    chapters = []
    last_no = 0
    for i, match in enumerate(matches):
        title_line = match.group(1).strip()
        # 章节正文：从标题行结束到下一个标题行开始
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        # 尝试从标题行里分离章节号和标题
        chapter_no, title = _parse_title(title_line, i + 1)

        # 章节号必须严格递增：源文本偶尔会出现重复编号（如同一数字既用阿拉伯
        # 数字又用中文数字标了一次，两者都指向各自"正确"的号码，但撞了车）。
        # 允许正常跳号（章节缺失是常事），只在号码不升反降/重复时按阅读顺序
        # 顺延，避免下游按 chapter_no 存文件、生成引用时互相覆盖。
        if chapter_no <= last_no:
            chapter_no = last_no + 1
        last_no = chapter_no

        chapters.append({
            "chapter_no": chapter_no,
            "title": title,
            "text": body,
        })

    return chapters


_CN_DIGIT = {'零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4,
             '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}
_CN_UNIT = {'十': 10, '百': 100, '千': 1000, '万': 10000}


def _cn_to_int(s: str) -> int | None:
    """
    把"一千零八十一"这样的中文数字转成整数 1081。
    支持到"万"位，出现无法识别的字符时返回 None（交给调用方决定回退策略）。
    """
    if not s:
        return None
    total = 0    # 已确定的高位累积（跨过"万"之后）
    section = 0  # 当前"万"以内的累积段
    num = 0      # 当前数字缓冲（十/百/千前面那个数字）
    for ch in s:
        if ch in _CN_DIGIT:
            num = _CN_DIGIT[ch]
        elif ch in _CN_UNIT:
            unit = _CN_UNIT[ch]
            if unit == 10000:
                total += (section + num) * unit
                section, num = 0, 0
            else:
                section += (num or 1) * unit  # "十二"里的"十"前面没数字，隐含为 1
                num = 0
        else:
            return None
    result = total + section + num
    return result if result > 0 else None


def _parse_title(title_line: str, fallback_no: int) -> tuple[int, str]:
    """
    从 "第123章 天才少年" 解析出 (123, "天才少年")。
    无法解析时使用 fallback_no 作为章节号，完整行作为标题。
    """
    m = re.match(r'^第(\d+)章\s*(.*)', title_line)
    if m:
        return int(m.group(1)), m.group(2).strip() or f"第{m.group(1)}章"

    # 中文数字章节号：优先转换成真实数字，避免退化成正文里"第...章"匹配次数
    # 的position——那个数字和实际章节序号常常对不上，会撞出重复的 chapter_no。
    m2 = re.match(r'^第([零一二三四五六七八九十百千万]+)章\s*(.*)', title_line)
    if m2:
        chapter_no = _cn_to_int(m2.group(1))
        if chapter_no is None:
            chapter_no = fallback_no
        return chapter_no, m2.group(2).strip() or title_line

    return fallback_no, title_line


def _split_by_lines(text: str) -> list[dict]:
    """回退方案：按固定行数等长切分。"""
    lines = text.splitlines()
    chunks = [
        lines[i: i + _FALLBACK_LINES_PER_CHAPTER]
        for i in range(0, len(lines), _FALLBACK_LINES_PER_CHAPTER)
    ]
    return [
        {
            "chapter_no": idx + 1,
            "title": f"段落 {idx + 1}",
            "text": "\n".join(chunk),
        }
        for idx, chunk in enumerate(chunks)
        if chunk
    ]


# ── 批次聚合 ─────────────────────────────────────────────────────────────────

def make_batches(
    chapters: list[dict],
    window_chars: int,
    min_window_chars: int | None = None,
    *,
    processed_chars: int = 0,
    total_chars: int | None = None,
) -> list[list[dict]]:
    """
    将章节列表按动态字符数上限贪心聚合为批次。
    每个章节完整放入一个批次，不在章节内部截断。

    未提供 ``min_window_chars`` 时保持旧行为，使用固定窗口。提供后，窗口会
    根据全书已处理字符比例，从 ``window_chars`` 线性下降到
    ``min_window_chars``。调度只依赖原文位置，因此中断重启不会改变批次边界。

    参数：
        chapters:         split_chapters() 的返回值
        window_chars:     初始字符数上限
        min_window_chars: 处理到全书末尾时的最小上限；省略表示固定窗口
        processed_chars:  本列表之前已经处理的字符数，用于安全重排未处理尾部
        total_chars:      全书总字符数；省略时按当前列表与 processed_chars 计算

    返回：
        [[chapter, chapter, ...], [...], ...]
    """
    if window_chars <= 0:
        raise ValueError("window_chars 必须大于 0")
    minimum = window_chars if min_window_chars is None else min_window_chars
    if minimum <= 0 or minimum > window_chars:
        raise ValueError("min_window_chars 必须大于 0 且不能超过 window_chars")
    if processed_chars < 0:
        raise ValueError("processed_chars 不能小于 0")

    remaining_chars = sum(len(chapter["text"]) for chapter in chapters)
    whole_chars = processed_chars + remaining_chars if total_chars is None else total_chars
    if whole_chars < processed_chars + remaining_chars:
        raise ValueError("total_chars 不能小于已处理与待处理字符数之和")

    batches: list[list[dict]] = []
    chapter_index = 0
    committed_chars = processed_chars

    while chapter_index < len(chapters):
        target = _window_for_progress(
            window_chars,
            minimum,
            committed_chars,
            whole_chars,
        )
        batch: list[dict] = []
        batch_chars = 0

        while chapter_index < len(chapters):
            chapter = chapters[chapter_index]
            chapter_len = len(chapter["text"])
            if batch and batch_chars + chapter_len > target:
                break
            batch.append(chapter)
            batch_chars += chapter_len
            chapter_index += 1
            # 单章超过当前上限时仍完整保留，并立即封存这一批。
            if batch_chars >= target:
                break

        batches.append(batch)
        committed_chars += batch_chars

    return batches


def _window_for_progress(start: int, minimum: int, processed: int, total: int) -> int:
    """按全书文本进度计算当前批次上限。"""
    if total <= 0 or start == minimum:
        return start
    ratio = min(max(processed / total, 0.0), 1.0)
    return max(minimum, round(start - (start - minimum) * ratio))
