# -*- coding: utf-8 -*-
r"""
mineru_loader.py — MinerU 输出 → RAG 文档全文（docs_cache_v2 生成器）

设计定位：
  MinerU（整合包）是【外部工具】，本模块只读它的落盘产物，不 import mineru 包，
  主项目 venv 零新增依赖。数据流：
    MinerU 输出目录/output/.../result/<文档名>/vlm/<文档名>_content_list.json
        ↓ 本模块
    {文档名: 全文}，格式与 document_loader 对齐，下游分块/评测零改动：
      - 按 page_idx 插入【第X页】页码标记（保留溯源，兼容现有分块器）；
      - 表格/图表包 [TABLE_START]/[/TABLE_END]，内容转 Markdown（兼容 HYBRID）；
      - 公式保留 LaTeX 文本；图片保留图题 + VLM 内容描述；
      - header/footer/page_number 噪声块直接丢弃（旧管线做不到，MinerU 已识别）。

用法：
    from rag_core.mineru_loader import build_docs_cache_v2
    docs = build_docs_cache_v2(r"你的MinerU输出目录")
命令行（生成缓存 + 新旧对比）：
    python build_docs_cache_v2.py --mineru-out <MinerU输出目录> --save data/docs_cache_v2.json
"""

import json
import os
import re
from html.parser import HTMLParser
from typing import Dict, List, Optional

TABLE_START = "[TABLE_START]"
TABLE_END = "[/TABLE_END]"

# 噪声块类型：MinerU 已把页眉/页脚/页码识别出来，直接丢弃
_NOISE_TYPES = {"header", "footer", "page_number"}

# 部分页眉文字会漏进 text 块（如 "第56页"），同样丢弃——页码溯源由我们自己的【第X页】标记承担
_PAGE_LABEL_RE = re.compile(r"^第\s*\d+\s*页$")


# --------------------------------------------------------------------------
# HTML 表格 → Markdown（MinerU 的 table_body 是 HTML）
# --------------------------------------------------------------------------
class _TableHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows: List[List[str]] = []
        self._row: Optional[List[str]] = None
        self._cell: Optional[List[str]] = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
            self._in_cell = True

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            self._row.append("".join(self._cell).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def _html_table_to_md(html: str) -> str:
    """HTML 表格 → Markdown 行（兼容 HYBRID 分块器的行组切分）。"""
    p = _TableHTMLParser()
    try:
        p.feed(html)
    except Exception:
        return ""
    if not p.rows:
        return ""
    lines = []
    for i, row in enumerate(p.rows):
        cells = [c.replace("\n", " ").replace("|", "\\|") for c in row]
        lines.append("| " + " | ".join(cells) + " |")
        if i == 0:
            lines.append("|" + "|".join([" --- " for _ in row]) + "|")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 单块 → 文本
# --------------------------------------------------------------------------
def _block_to_text(block: Dict, page_no: int, image_idx: List[int]) -> Optional[str]:
    """把 content_list 的一个块转成文本；噪声块返回 None。"""
    btype = block.get("type")
    if btype in _NOISE_TYPES:
        return None

    if btype == "text":
        t = (block.get("text") or "").strip()
        if _PAGE_LABEL_RE.fullmatch(t):
            return None  # 漏进正文的页眉页码文字（如 "第56页"），丢弃
        return t

    if btype == "list":
        items = block.get("list_items") or []
        return "\n".join(str(x).strip() for x in items if str(x).strip())

    if btype == "equation":
        return (block.get("text") or "").strip()

    if btype in ("table", "chart"):
        md = ""
        if btype == "table":
            md = _html_table_to_md(block.get("table_body") or "")
        else:
            # chart：VLM 已把图转成 Markdown 表格/文本
            md = (block.get("content") or "").strip()
        caption = " ".join(str(c) for c in (block.get("table_caption") or block.get("chart_caption") or []))
        footnote = " ".join(str(f) for f in (block.get("table_footnote") or block.get("chart_footnote") or []))
        if not md and not caption:
            return None
        head = caption if caption else "表格"
        if footnote:
            head += f"（{footnote}）"
        return f"{TABLE_START}\n{head}\n{md}\n{TABLE_END}".strip()

    if btype == "image":
        image_idx[0] += 1
        parts = [f"[图片{page_no}-{image_idx[0]}]"]
        caption = " ".join(str(c) for c in (block.get("image_caption") or []))
        content = (block.get("content") or "").strip()
        if caption:
            parts.append(caption)
        if content:
            parts.append(content[:400])
        return "\n".join(parts)

    return None


# --------------------------------------------------------------------------
# 块序列 → 全文（按 page_idx 插页码标记）
# --------------------------------------------------------------------------
def blocks_to_text(blocks: List[Dict]) -> str:
    """content_list 块序列 → 全文。MinerU 已按阅读顺序排序，这里只负责组版。"""
    parts: List[str] = []
    last_page = -1
    image_idx = [0]
    for b in blocks:
        page = int(b.get("page_idx", 0) if b.get("page_idx") is not None else 0)
        if page != last_page:
            parts.append(f"\n【第{page + 1}页】\n")
            last_page = page
        txt = _block_to_text(b, page + 1, image_idx)
        if txt:
            parts.append(txt)
    return "\n".join(parts).strip()


def load_mineru_doc(content_list_path: str) -> str:
    """读取单个 MinerU content_list.json → 全文。"""
    with open(content_list_path, "r", encoding="utf-8") as f:
        blocks = json.load(f)
    if not isinstance(blocks, list):
        raise ValueError(f"content_list 结构异常: {content_list_path}")
    return blocks_to_text(blocks)


# --------------------------------------------------------------------------
# 输出目录扫描 → {文档名: content_list 路径}
# --------------------------------------------------------------------------
def find_mineru_outputs(output_root: str) -> Dict[str, str]:
    """扫描 MinerU 输出目录，返回 {文档名: content_list.json 路径}。
    兼容两种布局（同一文档名取 mtime 最新）：
      - WebUI:  output/gradio/<时间戳>_<hash>_<文档名>/result/<文档名>/vlm/<文档名>_content_list.json
      - CLI:    output/batch/<文档名>/vlm/<文档名>_content_list.json（或 auto/hybrid_auto 目录）"""
    found: Dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(output_root):
        for fn in filenames:
            if not fn.endswith("_content_list.json") or fn.endswith("_content_list_v2.json"):
                continue
            doc = fn[: -len("_content_list.json")]
            path = os.path.join(dirpath, fn)
            if doc not in found or os.path.getmtime(path) > os.path.getmtime(found[doc]):
                found[doc] = path
    return found


def build_docs_cache_v2(output_root: str, target_docs: Optional[List[str]] = None) -> Dict[str, str]:
    """扫描 MinerU 输出 → {文档名.pdf: 全文}（与 document_loader 输出同构，
    键名带 .pdf 后缀以与标注 gold_docs / 旧缓存 / evaluate 的 source 比对一致）。
    target_docs 为 None 时取全部已处理文档。"""
    found = find_mineru_outputs(output_root)
    if target_docs:
        found = {d: p for d, p in found.items() if d in set(target_docs)}
    docs: Dict[str, str] = {}
    for doc, path in found.items():
        try:
            docs[doc + ".pdf"] = load_mineru_doc(path)
        except Exception as e:
            print(f"  [ERR] {doc}: {e}")
    return docs


if __name__ == "__main__":
    import argparse
    from rag_core import config
    ap = argparse.ArgumentParser(description="扫描 MinerU 输出并生成 RAG 文档全文")
    ap.add_argument("--mineru-out", default=None, help="MinerU 输出目录（默认取 config.MINERU_OUT）")
    ap.add_argument("--save", default=None, help="保存为 JSON（如 data/docs_cache_v2.json）")
    args = ap.parse_args()
    docs = build_docs_cache_v2(args.mineru_out or config.MINERU_OUT)
    print(f"共 {len(docs)} 篇:")
    for name, text in docs.items():
        print(f"  {name}: {len(text)} 字")
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(docs, f, ensure_ascii=False, indent=2)
        print(f"已保存 → {args.save}")
