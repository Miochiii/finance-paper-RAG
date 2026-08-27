#自适应分块器,根据 structure 和 chunk_size 使用不同切分策略

# chunk_splitter.py
# chunk_splitter.py

# ========== 正经PDF（DISCOURSE）高级分割 ==========
import json
import re
from typing import List, Dict, Any, Optional, Tuple
import tiktoken

# ========== 模块导入与全局缓存 ==========
try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("警告: tiktoken 未安装，部分精确切分功能降级")

_ENC = None

#<editor-fold desc="文本PDF - 工具函数">

def _get_encoder():
    global _ENC
    if _ENC is None and TIKTOKEN_AVAILABLE:
        _ENC = tiktoken.get_encoding("cl100k_base")
    return _ENC


def _count_tokens(text: str) -> int:
    """统一用 tiktoken 精确计数。
    注意：曾有的"短文本快速路径 len//2"对中文系统性低估 2~3 倍（汉字在 cl100k 下约 1~2 token），
    导致按句累积时 800 token 上限形同虚设、块实际超限近 2 倍。已移除，一律走 tiktoken。"""
    if not text:
        return 0
    if not TIKTOKEN_AVAILABLE:
        return len(text) // 3
    enc = _get_encoder()
    return len(enc.encode(text))


# ========== 1. Token 级重叠截取 ==========
def _token_overlap(text: str, overlap_tokens: int) -> str:
    if overlap_tokens <= 0 or not text or not TIKTOKEN_AVAILABLE:
        return ""
    enc = _get_encoder()
    tokens = enc.encode(text)
    if len(tokens) <= overlap_tokens:
        return text
    return enc.decode(tokens[-overlap_tokens:])


# ========== 2. 超长句子智能切割 ==========
def _split_oversized_sentence(text: str, max_tokens: int) -> List[str]:
    """按标点降级切分超长句，避免强行截断单词"""
    # 尝试按逗号/分号拆
    for sep in ['，', '；', '、', ',', ';']:
        if sep in text:
            parts = text.split(sep)
            merged = []
            current = ""
            for p in parts:
                # 加回分隔符（保留语义）
                p_with_sep = p + sep if p != parts[-1] else p
                if _count_tokens(current + p_with_sep) <= max_tokens:
                    current += p_with_sep
                else:
                    if current:
                        merged.append(current)
                    current = p_with_sep
            if current:
                merged.append(current)
            if len(merged) > 1:
                # 逗号片段本身仍可能超限（公式/无标点长串）→ 递归压到上限内，保证块不超 max_tokens
                final = []
                for m in merged:
                    if _count_tokens(m) <= max_tokens:
                        final.append(m)
                    else:
                        final.extend(_split_oversized_sentence(m, max_tokens))
                return final
    # 回退：按字符强制切（精确 token）
    if TIKTOKEN_AVAILABLE:
        enc = _get_encoder()
        tokens = enc.encode(text)
        result = []
        for i in range(0, len(tokens), max_tokens):
            result.append(enc.decode(tokens[i:i + max_tokens]))
        return result
    else:
        # 极端回退：按字符数
        return [text[i:i + max_tokens * 3] for i in range(0, len(text), max_tokens * 3)]


# ========== 3. 递归切分（带 Token 级重叠） ==========
_TABLE_END_MARKS = ("[TABLE_END]", "[/TABLE_END]")


def _table_aware_units(sentences: List[str]) -> List[str]:
    """把连续同表的句子合并为一个单元（表格块内部不再有切块边界）；
    非表句子各成一个单元。返回单元文本列表（单元内部行以换行连接）。"""
    units: List[str] = []
    buf: List[str] = []
    for s in sentences:
        st = s.strip()
        if st == "[TABLE_START]":
            if buf:
                units.append("\n".join(buf))
                buf = []
            buf.append(s)
            continue
        if st in _TABLE_END_MARKS:
            buf.append(s)
            units.append("\n".join(buf))
            buf = []
            continue
        buf.append(s)
    if buf:
        units.append("\n".join(buf))
    return units


def _split_table_by_lines(unit: str, max_tokens: int) -> List[str]:
    """超限表格按行切（行不劈开），每片尽量接近 max_tokens。"""
    lines = unit.split("\n")
    out, cur, cur_t = [], [], 0
    for ln in lines:
        lt = _count_tokens(ln)
        if cur and cur_t + lt > max_tokens:
            out.append("\n".join(cur))
            cur, cur_t = [], 0
        cur.append(ln)
        cur_t += lt
    if cur:
        out.append("\n".join(cur))
    return out


def _recursive_split(text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    if _count_tokens(text) <= max_tokens:
        return [text]

    sentences = re.split(r'(?<=[。！？；\n.!?])\s*', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        sentences = [text.strip()]

    # 表格感知：连续同表句子合成一个单元，切块边界绝不落在表格内部
    units = _table_aware_units(sentences)

    chunks = []
    current_chunk = []
    current_tokens = 0

    for unit in units:
        unit_tokens = _count_tokens(unit)
        if unit_tokens > max_tokens:
            # 超长单元：表格按行切（行不劈）；普通超长句走标点降级切
            sub_units = (
                _split_table_by_lines(unit, max_tokens)
                if unit.startswith("[TABLE_START]")
                else _split_oversized_sentence(unit, max_tokens)
            )
            for sub in sub_units:
                if current_tokens + _count_tokens(sub) > max_tokens and current_chunk:
                    chunk_text = "\n".join(current_chunk)
                    chunks.append(chunk_text)
                    overlap_text = _token_overlap(chunk_text, overlap_tokens)
                    current_chunk = [overlap_text] if overlap_text else []
                    current_tokens = _count_tokens(overlap_text)
                current_chunk.append(sub)
                current_tokens += _count_tokens(sub)
            continue

        if current_tokens + unit_tokens > max_tokens:
            chunk_text = "\n".join(current_chunk)
            chunks.append(chunk_text)
            overlap_text = _token_overlap(chunk_text, overlap_tokens)
            current_chunk = [overlap_text] if overlap_text else []
            current_tokens = _count_tokens(overlap_text)

        current_chunk.append(unit)
        current_tokens += unit_tokens

    if current_chunk:
        chunks.append("\n".join(current_chunk))
    return chunks


# ========== 4. 页眉页脚清洗增强 ==========
def _clean_text(text: str) -> str:
    text = re.sub(r'={5,}\s*Page\s*\d+\s*={5,}', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?<!\w)^\s*\d{1,6}\s*$(?!\w)', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[-=]{5,}\s*.*?\s*[-=]{5,}$', '', text, flags=re.MULTILINE)
    text = re.sub(r'第\s*\d+\s*页', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ========== 4.1 分块页码归属 ==========
_PAGE_MARK_RE = re.compile(r"【第\s*(\d+)\s*页[^】]*】")
_RESIDUE_RE = re.compile(r"【\s*】")


def _strip_marker_residue(t: str) -> str:
    """去掉块开头的页码标记残留（【】空标记或完整【第X页】标记）与首部空白。"""
    for _ in range(4):
        t2 = re.sub(r"^\s*【\s*】\s*", "", t)
        t2 = re.sub(r"^\s*【第\s*\d+\s*页[^】]*】\s*", "", t2)
        if t2 == t:
            break
        t = t2
    return t


def _prefix_candidates(chunk: str, strip_markers: bool):
    """块开头 → 对齐用前缀候选。strip_markers=True 时去掉标记残留
    （用于已清洗文本对齐）；False 时原样取头（用于原始文本对齐）。
    discourse 块额外给出去掉 [标题路径] 包装的变体；另给去掉开头乱码的变体。"""
    t = chunk.strip()
    head = _strip_marker_residue(t) if strip_markers else t
    bodies = [head]
    if head.startswith("[") and "] " in head:
        bodies.append(head.split("] ", 1)[1])
    stripped_fffd = head.lstrip("\ufffd")
    if stripped_fffd and stripped_fffd != head:
        bodies.append(stripped_fffd)
    return bodies


def _find_prefix(text: str, chunk: str, strip_markers: bool, cursor: int) -> int:
    for cand in _prefix_candidates(chunk, strip_markers):
        for n in (80, 50, 30, 16):
            pos = text.find(cand[:n], cursor)
            if pos != -1:
                return pos
    return -1


def attribute_pages(raw_text: str, chunks: List[str]) -> List[Tuple[Optional[int], Optional[int]]]:
    """把每个分块归属到页码，返回 [(起始页, 结束页), ...]，对齐失败为 (None, None)。

    原理（行级对齐）：hmm/discourse 的块是"句子按换行拼接"的产物，页码标记
    会被句切分器拆成独立行，块级前缀必然跨行失配。因此以"行"为单位对齐：
      - 基底 A = 清洗文本（标记残留为【】），基底 B = 原始文本（标记完整）；
        块内含完整【第X页】标记时优先 B（fixed/hybrid），否则优先 A；
      - 镜像 hmm_chunk 的前置处理：字面 "\\n" 先还原为真换行（24 篇文档受影响）；
      - 每行取前缀（去标记残留 / 去 discourse 标题包装）在基底中查找，
        行内游标单调前进；找不到时全局回退一次（自愈重叠行与游标过冲）；
      - 块的页码 = 首个匹配行与末个匹配行之间的标记页码范围。
    """
    # 镜像 hmm_chunk：分句前把字面 "\n" 还原（fixed/hybrid 不做此步，故 raw 基底用原文）
    raw_conv = raw_text
    if "\\n" in raw_text:
        raw_conv = raw_text.replace("\\r\\n", "\n").replace("\\n", "\n")
    cleaned = _clean_text(raw_conv)
    raw_marks = [(m.start(), int(m.group(1))) for m in _PAGE_MARK_RE.finditer(raw_text)]
    conv_marks = [(m.start(), int(m.group(1))) for m in _PAGE_MARK_RE.finditer(raw_conv)]
    residues = [m.start() for m in _RESIDUE_RE.finditer(cleaned)]
    clean_marks = [
        (pos, conv_marks[i][1]) for i, pos in enumerate(residues) if i < len(conv_marks)
    ]
    base_clean = (cleaned, clean_marks, True)
    base_raw = (raw_text, raw_marks, False)
    cursors = {id(base_clean): 0, id(base_raw): 0}
    ranges: List[Tuple[Optional[int], Optional[int]]] = []

    def _align_lines(lines, base):
        text, _marks, strip = base
        cursor = cursors[id(base)]
        first = None
        last_end = None
        for ln in lines:
            pos = -1
            for cand in _prefix_candidates(ln, strip):
                for n in (40, 24, 12):
                    p = text.find(cand[:n], cursor)
                    if p == -1:
                        p = text.find(cand[:n], 0)  # 全局回退（重叠行/游标过冲自愈）
                    if p != -1:
                        pos = p
                        break
                if pos != -1:
                    break
            if pos == -1:
                continue
            cursors[id(base)] = pos + max(len(ln), 1)
            if first is None:
                first = pos
            last_end = pos + len(ln)
        return first, last_end

    for ch in chunks:
        lines = [ln.strip() for ln in ch.strip().split("\n") if ln.strip()]
        order = (base_raw, base_clean) if _PAGE_MARK_RE.search(ch) else (base_clean, base_raw)
        best = None
        for base in order:
            first, last_end = _align_lines(lines, base)
            if first is not None:
                best = (base[1], first, last_end)
                break
        if best is None:
            ranges.append((None, None))
            continue
        marks, first, last_end = best
        start_page = None
        for off, p in marks:
            if off <= first:
                start_page = p
            else:
                break
        end_page = start_page
        for off, p in marks:
            if first < off <= last_end:
                end_page = p
            elif off > last_end:
                break
        ranges.append((start_page, end_page))
    return ranges


# ========== 5. 章节提取（层级路径） ==========
def _extract_chapters(text: str) -> List[Dict[str, Any]]:
    lines = text.split('\n')
    pattern_chapter = r'^(第[一二三四五六七八九十百]+章|第\d+章)\s*(.*?)$'
    pattern_section = r'^(\d+(\.\d+){1,2})\s*(.*?)$'

    hierarchy = []
    buffer_content = []
    chapters = []

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        m_chap = re.match(pattern_chapter, line_stripped)
        if m_chap:
            if buffer_content:
                chapters.append({
                    'title_path': ' > '.join(hierarchy) if hierarchy else m_chap.group(1),
                    'content': '\n'.join(buffer_content).strip(),
                    'level': 1
                })
            hierarchy = [m_chap.group(1)]
            buffer_content = []
            continue

        m_sec = re.match(pattern_section, line_stripped)
        if m_sec and hierarchy:
            # 将节标题作为内容的一部分保留（兼顾元数据）
            buffer_content.append(f"【{m_sec.group(1)}】{m_sec.group(3)}")
        else:
            buffer_content.append(line)

    if buffer_content:
        chapters.append({
            'title_path': ' > '.join(hierarchy) if hierarchy else "全文",
            'content': '\n'.join(buffer_content).strip(),
            'level': len(hierarchy)
        })

    if not chapters:
        chapters = [{'title_path': '全文', 'content': text, 'level': 0}]
    return chapters


# ========== 6. 主入口函数 ==========
def split_discourse_advanced(text: str, chunk_size: int, overlap_tokens: int = None) -> List[str]:
    if not text or not text.strip():
        return []
    if overlap_tokens is None:
        overlap_tokens = max(50, chunk_size // 10)

    text = _clean_text(text)
    chapters = _extract_chapters(text)
    all_chunks = []

    for ch in chapters:
        content = ch['content']
        if not content:
            continue
        # 关键：整个章节传入，跨段落重叠自动生效
        sub_blocks = _recursive_split(content, chunk_size, overlap_tokens)
        for sub in sub_blocks:
            if ch['title_path'] != "全文":
                block_text = f"[{ch['title_path']}] {sub}"
            else:
                block_text = sub
            all_chunks.append(block_text)
    return all_chunks
# </editor-fold>


#<editor-fold desc="docx - 工具函数">
def split_by_hybrid(text: str, chunk_size: int, overlap_tokens: int = 50) -> List[str]:
    """
    针对 docx 混合文档（HYBRID）的智能切分：
    1. 保持表格、段落、列表的原始顺序
    2. 列表项保留缩进层级
    3. 超长表格按行组切分，每块携带表头
    4. Token 级重叠：重叠文本放在新块【开头】（承接上一块结尾），跨元素滑动窗口语义连贯
    """
    if not text or not text.strip():
        return []

    # ---------- 1. 顺序解析元素（保留顺序） ----------
    # 逐行扫描，识别表格块（多行）
    lines = text.split('\n')
    elements = []  # (type, content, indent_level)
    i = 0
    while i < len(lines):
        line = lines[i]
        # 检查是否为表格开始标记
        if line.strip().startswith('[TABLE_START]'):
            # 收集表格内容直到 [TABLE_END]（兼容 [/TABLE_END] 写法：docx 与 MinerU 加载器均用带斜杠形式）
            table_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('[TABLE_END]') \
                    and not lines[i].strip().startswith('[/TABLE_END]'):
                table_lines.append(lines[i])
                i += 1
            # 跳过 [TABLE_END]
            i += 1
            table_content = '\n'.join(table_lines).strip()
            if table_content:
                elements.append(('table', table_content, 0))
            continue

        # 普通行
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        # 检查是否为列表项（[LIST_ITEM]前缀）
        list_match = re.match(r'\[LIST_ITEM\]\s*(.*)', line)
        if list_match:
            # 保留原始缩进（计算前导空格）
            raw_content = list_match.group(1)
            prefix = '[LIST_ITEM]'
            idx = line.find(prefix)
            if idx != -1:
                after_prefix = line[idx + len(prefix):]
                # 计算 after_prefix 的前导空格
                leading_spaces = len(after_prefix) - len(after_prefix.lstrip(' '))
                indent_level = leading_spaces // 2  # 每级2空格
                # 内容去掉前导空格
                content = after_prefix.lstrip(' ')
            else:
                indent_level = 0
                content = raw_content
            elements.append(('list_item', content, indent_level))
        else:
            # 普通段落
            para_match = re.match(r'\[PARA\]\s*(.*)', line)
            if para_match:
                elements.append(('paragraph', para_match.group(1), 0))
            else:
                # 兼容无标记（如纯文本）
                elements.append(('paragraph', line, 0))
        i += 1

    # ---------- 2. 元素 → 展示文本 ----------
    processed_elements = []  # (text, type)
    for etype, content, indent in elements:
        if etype == 'table':
            processed_elements.append((content, 'table'))
        elif etype == 'list_item':
            prefix = "  " * indent
            text_item = f"{prefix}- {content}"
            processed_elements.append((text_item, 'list_item'))
        else:  # paragraph
            processed_elements.append((content, 'paragraph'))

    # ---------- 3. 展开超长表格（按行组切分，每块携带表头） ----------
    expanded = []  # (text, type)，表格切分后的多个子块也保持顺序
    for elem_text, etype in processed_elements:
        if etype == 'table' and _count_tokens(elem_text) > chunk_size:
            table_rows = elem_text.split('\n')
            if len(table_rows) < 2:
                expanded.append((elem_text, 'table'))
                continue
            header = table_rows[0]
            separator = table_rows[1] if len(table_rows) > 1 else ""
            data_rows = table_rows[2:] if len(table_rows) > 2 else []
            if not data_rows:
                # 只有表头无数据行：整表作为一块，避免被切分逻辑丢弃
                expanded.append((elem_text, 'table'))
                continue
            # 估算每块可容纳的数据行数（粗略）
            tokens_per_row = _count_tokens('\n'.join(data_rows[:5])) / max(len(data_rows[:5]), 1)
            header_cost = _count_tokens(header) + _count_tokens(separator)
            rows_per_chunk = max(1, int((chunk_size - header_cost) // (tokens_per_row + 1)))
            for start in range(0, len(data_rows), rows_per_chunk):
                chunk_rows = data_rows[start:start + rows_per_chunk]
                sub_text = header + '\n' + separator + '\n' + '\n'.join(chunk_rows)
                expanded.append((sub_text, 'table'))
        else:
            expanded.append((elem_text, etype))

    # ---------- 4. 构建块（重叠放在新块开头，承接上一块结尾） ----------
    final_chunks = []
    current_chunk_texts = []  # 存储当前块的文本片段
    current_tokens = 0

    def flush_chunk():
        # 收尾当前块，返回其末尾重叠文本（供下一块作开头前缀）
        nonlocal current_chunk_texts, current_tokens
        if not current_chunk_texts:
            return ""
        chunk_text = '\n'.join(current_chunk_texts)
        final_chunks.append(chunk_text)
        current_chunk_texts = []
        current_tokens = 0
        if overlap_tokens > 0:
            return _token_overlap(chunk_text, overlap_tokens)
        return ""

    for idx, (elem_text, etype) in enumerate(expanded):
        elem_tokens = _count_tokens(elem_text)

        # 单个元素本身就超长
        if elem_tokens > chunk_size:
            if etype == 'paragraph':
                # 段落：按句子降级切分
                for sub in _split_oversized_sentence(elem_text, chunk_size):
                    if current_tokens + _count_tokens(sub) > chunk_size:
                        overlap = flush_chunk()
                        if overlap:
                            current_chunk_texts = [overlap]
                            current_tokens = _count_tokens(overlap)
                    current_chunk_texts.append(sub)
                    current_tokens += _count_tokens(sub)
                continue
            else:
                # 表格子块/列表项：独占一块（尽量完整）
                if current_chunk_texts:
                    flush_chunk()
                final_chunks.append(elem_text)
                continue

        # 放不下：先收尾当前块，新块以重叠文本开头
        if current_tokens + elem_tokens > chunk_size:
            overlap = flush_chunk()
            if overlap:
                current_chunk_texts = [overlap]
                current_tokens = _count_tokens(overlap)

        current_chunk_texts.append(elem_text)
        current_tokens += elem_tokens

    # 处理最后的块
    if current_chunk_texts:
        final_chunks.append('\n'.join(current_chunk_texts))

    return final_chunks
# </editor-fold>

_MAX_TOKENS_DEFAULT = 800  # TABULAR/CODE/NESTED 分块的默认 token 上限（与项目统一配置一致）


def _cap_oversize(blocks: List[str], max_tokens: int) -> List[str]:
    """超限块递归切到上限内（复用 _recursive_split），空文本块过滤。"""
    out: List[str] = []
    for b in blocks:
        if not b or not b.strip():
            continue
        if _count_tokens(b) > max_tokens:
            out.extend(_recursive_split(b, max_tokens, 0))
        else:
            out.append(b)
    return out


def split_by_tabular(text: str, group_rule: str, max_tokens: int = _MAX_TOKENS_DEFAULT):
    """表格分块：按行组切分（group_rule 形如 row_group_50），每组携带表头/首行。
    超限行组递归切到 max_tokens 内。"""
    if not text or not text.strip():
        return []
    m = re.match(r"row_group_(\d+)", str(group_rule or ""))
    n_rows = int(m.group(1)) if m else 50
    lines = text.split("\n")
    if len(lines) <= n_rows + 1:
        return _cap_oversize([text], max_tokens)
    header = lines[0]
    groups: List[str] = []
    for start in range(1, len(lines), n_rows):
        groups.append("\n".join([header] + lines[start:start + n_rows]))
    return _cap_oversize(groups, max_tokens)


def split_by_code(text: str, max_tokens: int = _MAX_TOKENS_DEFAULT):
    """代码按函数/类感知分块（func_aware）：ast 顶层节点切分，节点间注释并入上一块；
    超长节点内部按行递归切。ast 解析失败时回退按行递归切。"""
    if not text or not text.strip():
        return []
    try:
        import ast
        tree = ast.parse(text)
    except Exception:
        return _recursive_split(text, max_tokens, 0)
    lines = text.split("\n")
    blocks: List[str] = []
    prev_end = 0
    for node in tree.body:
        lineno = getattr(node, "lineno", None)
        if lineno is None:
            continue
        end = getattr(node, "end_lineno", None) or lineno
        start = lineno - 1
        lead = "\n".join(lines[prev_end:start]).strip()
        if lead:
            if blocks:
                blocks[-1] = blocks[-1] + "\n" + lead
            else:
                blocks.append(lead)
        blocks.append("\n".join(lines[start:end]).rstrip())
        prev_end = end
    tail = "\n".join(lines[prev_end:]).strip()
    if tail:
        blocks.append(tail)
    return _cap_oversize(blocks, max_tokens) or [text]


def split_by_nested(text: str, max_tokens: int = _MAX_TOKENS_DEFAULT):
    """JSON 深度感知分块（depth_aware）：按顶层键/元素拆块，超限块递归切。
    解析失败时回退按行递归切。"""
    if not text or not text.strip():
        return []
    try:
        data = json.loads(text)
    except Exception:
        return _recursive_split(text, max_tokens, 0)
    parts: List[str] = []
    if isinstance(data, dict):
        for k, v in data.items():
            parts.append(json.dumps({k: v}, ensure_ascii=False, indent=2))
    elif isinstance(data, list):
        for v in data:
            parts.append(json.dumps(v, ensure_ascii=False, indent=2))
    else:
        parts = [text]
    return _cap_oversize(parts, max_tokens) or [text]


def split_by_fixed(text: str, chunk_size: int = 800, overlap_tokens: int = 50) -> List[str]:
    """固定长度分块（token 级 + 重叠）。"""
    if not text or not text.strip():
        return []
    if not TIKTOKEN_AVAILABLE:
        step = max(chunk_size * 2, 1)
        return [text[i:i + step] for i in range(0, len(text), step) if text[i:i + step].strip()]
    enc = _get_encoder()
    tokens = enc.encode(text)
    if not tokens:
        return []
    n = len(tokens)
    out, i = [], 0
    while i < n:
        j = min(i + chunk_size, n)
        out.append(enc.decode(tokens[i:j]))
        if j >= n:
            break
        nxt = j - overlap_tokens
        if nxt <= i:
            nxt = i + 1  # 防死循环
        i = nxt
    return out


def dispatch_chunk(text: str, structure: str, chunk_size=800, overlap_tokens=50) -> List[str]:
    """四模式统一分块入口（大小写不敏感）：fixed / discourse / hybrid / hmm。"""
    mode = (structure or "").strip().upper()
    if mode == "FIXED":
        return split_by_fixed(text, chunk_size, overlap_tokens)
    elif mode == "DISCOURSE":
        return split_discourse_advanced(text, chunk_size, overlap_tokens)
    elif mode == "HYBRID":
        return split_by_hybrid(text, chunk_size, overlap_tokens)
    elif mode == "HMM":
        from rag_core.hmm_chunker import hmm_chunk
        return hmm_chunk(text, chunk_size=chunk_size, overlap_tokens=overlap_tokens)
    else:
        raise ValueError(f"未知分块模式 {structure}（可选 fixed / discourse / hybrid / hmm）")
