# -*- coding: utf-8 -*-
"""md2docx_report.py —— 把 Markdown 报告转成 Word（复用 rag_core.md_docx 的论文排版）。

用法：
    python md2docx_report.py log/项目进展与模型口径_论文基础_20260921.md
    python md2docx_report.py a.md b.md --header "某某项目报告"
    python md2docx_report.py log/xxx.md --out 桌面/xxx.docx

约定：
  - 文档标题取 Markdown 第一行的一级标题（`# ...`），该行不再重复出现在正文里；
  - 页眉文字默认取标题首行，可用 --header 覆盖；
  - 参考文献按 GB/T 7714 的上标编号渲染（`## 参考文献` 一节会保留）。
"""
import argparse
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_core.md_docx import md_to_docx  # noqa: E402


def convert(md_path: str, out_path: str = "", header: str = "") -> str:
    lines = io.open(md_path, encoding="utf-8").read().splitlines()
    title = os.path.splitext(os.path.basename(md_path))[0]
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip()
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]
    out_path = out_path or os.path.splitext(md_path)[0] + ".docx"
    md_to_docx("\n".join(lines), out_path, title=title, citation_format="superscript",
               include_refs=True,
               style={"page": {"header_text": header or title}})
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Markdown 报告 → Word（论文排版）")
    ap.add_argument("md", nargs="+", help="Markdown 文件（可多个）")
    ap.add_argument("--out", default="", help="输出路径（仅在单个输入时有效）")
    ap.add_argument("--header", default="", help="页眉文字（默认用文档标题）")
    args = ap.parse_args()
    for md in args.md:
        if not os.path.isfile(md):
            print(f"跳过（不存在）：{md}")
            continue
        out = convert(md, args.out if len(args.md) == 1 else "", args.header)
        print(f"OK {out}（{os.path.getsize(out)} 字节）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
