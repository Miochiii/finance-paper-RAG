# -*- coding: utf-8 -*-
"""md_docx.py —— 综述 Markdown → Word(docx) 转换 + 引用格式转换。

引用格式（导出时由用户选择）：
  - superscript（上标编号）：[1] 保持数字引用，渲染为 Word 上标；
  - author_year（著者-年份）：[1] → （作者，年份）；相邻多引合并为
    （作者A，年份A；作者B，年份B）；引用若已处于中文括号内则不重复加括号。

编号→作者/年份的映射（ref_map: {n: (author, year)}）：
  1. 优先解析正文末尾"## 参考文献"列表（[N] 作者. 标题（年份）），
     编辑稿里编号变化也能对得上；
  2. 兜底用 survey_export 落盘的 exports/<slug>.map.json + refs.json。
  映射不到的编号保持原样 [N]。
"""

import os
import re
from typing import Dict, List, Optional, Tuple

_CITE_RE = re.compile(r"\[(\d+)\]")
_REF_LINE_RE = re.compile(r"^\[(\d+)\]\s*([^\.\s，,]+).*?（(\d{4})）")
_INLINE_RE = re.compile(r"(\*\*.+?\*\*|`[^`]+`)")

Segment = Tuple[str, str]  # (kind, text)：text / cite_sup / cite_ay


# --------------------------------------------------------------------------
# 引用映射
# --------------------------------------------------------------------------
def extract_ref_map(text: str) -> Dict[int, Tuple[str, str]]:
    """从"## 参考文献"列表解析 [N] → (作者, 年份)。无该节返回空。"""
    m = re.search(r"^##\s*参考文献\s*$", text, re.M)
    if not m:
        return {}
    ref_map: Dict[int, Tuple[str, str]] = {}
    for line in text[m.end():].splitlines():
        s = line.strip()
        if s.startswith("#"):
            break
        if not s:
            continue
        mm = _REF_LINE_RE.match(s)
        if mm:
            ref_map[int(mm.group(1))] = (mm.group(2), mm.group(3))
    return ref_map


def _segments(text: str, ref_map: Dict[int, Tuple[str, str]], fmt: str) -> List[Segment]:
    """把文本切成片段：text（普通文本）/ cite_sup（上标引用）/ cite_ay（著者-年份文本）。"""
    singles = list(_CITE_RE.finditer(text))
    if not singles:
        return [("text", text)]
    segs: List[Segment] = []
    pos, i, n = 0, 0, len(singles)
    while i < n:
        m = singles[i]
        if m.start() > pos:
            segs.append(("text", text[pos:m.start()]))
        # 收集相邻引用（[1][2]）
        group = [m]
        j = i + 1
        while j < n and singles[j].start() == group[-1].end():
            group.append(singles[j])
            j += 1
        i = j
        g_end = group[-1].end()
        # 中文括号内引用（如"（见[1]）"）：引用前有未闭合的（ 且紧随其后是 ）
        # → 著者-年份文本不再加括号，直接替换 [N]
        wrapped = False
        if fmt != "superscript" and text[g_end:g_end + 1] == "）":
            prev = text[pos:m.start()]
            if prev.count("（") > prev.count("）"):
                wrapped = True
        pos = g_end
        if fmt == "superscript":
            segs.append(("cite_sup", text[group[0].start():group[-1].end()]))
        else:  # author_year
            parts = []
            for g in group:
                hit = ref_map.get(int(g.group(1)))
                if hit:
                    parts.append(f"{hit[0]}，{hit[1]}")
            if parts:
                joined = "；".join(parts)
                segs.append(("cite_ay", joined if wrapped else f"（{joined}）"))
            else:
                segs.append(("text", text[group[0].start():group[-1].end()]))
    if pos < len(text):
        segs.append(("text", text[pos:]))
    return segs


def convert_citations(text: str, ref_map: Dict[int, Tuple[str, str]], fmt: str) -> str:
    """纯文本转换（测试/预览用）：返回转换后的整段文本。"""
    return "".join(t for _, t in _segments(text, ref_map, fmt))


# --------------------------------------------------------------------------
# Markdown → docx
# --------------------------------------------------------------------------
def _add_text_runs(p, text: str):
    """普通文本段 → docx runs（支持 **粗体** 与 `行内代码`）。"""
    for part in _INLINE_RE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            p.add_run(part[2:-2]).bold = True
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            p.add_run(part[1:-1]).font.name = "Consolas"
        else:
            p.add_run(part)


def _add_rich_paragraph(p, text: str, ref_map: Dict[int, Tuple[str, str]], fmt: str):
    """段落文本（含引用转换 + 行内格式）→ docx runs。"""
    for kind, seg in _segments(text, ref_map, fmt):
        if kind == "text":
            _add_text_runs(p, seg)
        elif kind == "cite_sup":
            r = p.add_run(seg)
            r.font.superscript = True
        else:  # cite_ay
            p.add_run(seg)


def _split_row(line: str) -> List[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def md_to_docx(text: str, out_path: str, title: str = "",
               citation_format: str = "author_year", include_refs: bool = False,
               ref_map: Optional[Dict[int, Tuple[str, str]]] = None) -> None:
    """Markdown → docx。citation_format: superscript / author_year。"""
    from docx import Document

    doc = Document()
    if title:
        doc.add_heading(title, level=0)

    ref_map = ref_map or {}
    lines = text.replace("\r\n", "\n").split("\n")
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        # 参考文献节
        if re.match(r"^##\s*参考文献\s*$", stripped):
            if include_refs:
                doc.add_heading("参考文献", level=1)
                i += 1
                while i < n:
                    s = lines[i].strip()
                    if s.startswith("#"):
                        break
                    if s:
                        doc.add_paragraph(s)
                    i += 1
                continue
            else:
                break  # 不包含参考文献：丢弃其后全部内容

        # 代码块
        if stripped.startswith("```"):
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            if buf:
                p = doc.add_paragraph()
                r = p.add_run("\n".join(buf))
                r.font.name = "Consolas"
            continue

        # 标题
        h = re.match(r"^(#{1,6})\s+(.*)$", line)
        if h:
            level = max(1, len(h.group(1)) - 1)
            doc.add_heading(h.group(2).strip(), level=level)
            i += 1
            continue

        if re.match(r"^(\s*[-*_]){3,}\s*$", line):
            i += 1
            continue

        # 引用块
        if re.match(r"^>\s?", line):
            buf = []
            while i < n and re.match(r"^>\s?", lines[i]):
                buf.append(re.sub(r"^>\s?", "", lines[i]))
                i += 1
            p = doc.add_paragraph()
            p.add_run("　".join(buf)).italic = True
            continue

        # 列表
        if re.match(r"^\s*[-*+]\s+", line):
            while i < n and re.match(r"^\s*[-*+]\s+", lines[i]):
                p = doc.add_paragraph(style="List Bullet")
                _add_rich_paragraph(p, re.sub(r"^\s*[-*+]\s+", "", lines[i]), ref_map, citation_format)
                i += 1
            continue
        if re.match(r"^\s*\d+[.)]\s+", line):
            while i < n and re.match(r"^\s*\d+[.)]\s+", lines[i]):
                p = doc.add_paragraph(style="List Number")
                _add_rich_paragraph(p, re.sub(r"^\s*\d+[.)]\s+", "", lines[i]), ref_map, citation_format)
                i += 1
            continue

        # 表格
        if re.match(r"^\s*\|", line):
            tbl = []
            while i < n and re.match(r"^\s*\|", lines[i]):
                tbl.append(lines[i])
                i += 1
            rows = [_split_row(x) for x in tbl if not re.match(r"^\|[\s:|-]+\|$", x)]
            rows = [r for r in rows if any(c for c in r)]
            if rows:
                ncol = max(len(r) for r in rows)
                table = doc.add_table(rows=len(rows), cols=ncol)
                table.style = "Table Grid"
                for ri, row in enumerate(rows):
                    for ci in range(ncol):
                        cell = table.cell(ri, ci)
                        cell.text = row[ci] if ci < len(row) else ""
            continue

        # 普通段落（合并连续行）
        if stripped:
            buf = [line]
            i += 1
            while i < n and lines[i].strip() and not re.match(
                    r"^(#{1,6}\s|>|\s*[-*+]\s|\s*\d+[.)]\s|\||```|---|\*{3})", lines[i]):
                buf.append(lines[i])
                i += 1
            p = doc.add_paragraph()
            _add_rich_paragraph(p, " ".join(buf), ref_map, citation_format)
            continue
        i += 1

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    doc.save(out_path)
