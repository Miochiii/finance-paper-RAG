# -*- coding: utf-8 -*-
"""原始 PDF/docx 定位与按页打开（Windows）。

打开指定页的实现链（优先级从高到低）：
  1. 环境变量 PDF_VIEWER 指定的阅读器 + PDF_VIEWER_ARGS 参数模板（{page} {path} 占位）；
  2. SumatraPDF：`SumatraPDF.exe -page N "文件"`（秒开、体验最好）；
  3. Edge：`file:///...#page=N` 锚点（任何 Win10/11 都有）；
  4. 系统默认程序（只能打开第 1 页，兜底）。
"""

import os
import shutil
import subprocess
import sys
from typing import List, Optional

_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

_EDGE_PATHS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]
_SUMATRA_PATHS = [
    r"C:\Program Files\SumatraPDF\SumatraPDF.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\SumatraPDF\SumatraPDF.exe"),
]


def resolve_pdf_path(doc: str, search_dirs: Optional[List[str]] = None) -> Optional[str]:
    """按文档名找到原始文件（PDF/docx）。

    - doc 已是存在的绝对路径时直接返回；
    - 否则在 search_dirs 各目录（含一层子目录）里按文件名精确匹配。
    """
    if os.path.isabs(doc) and os.path.isfile(doc):
        return doc
    for d in search_dirs or []:
        if not os.path.isdir(d):
            continue
        cand = os.path.join(d, doc)
        if os.path.isfile(cand):
            return cand
        try:
            for sub in os.listdir(d):
                subp = os.path.join(d, sub)
                if os.path.isdir(subp):
                    cand = os.path.join(subp, doc)
                    if os.path.isfile(cand):
                        return cand
        except OSError:
            continue
    return None


def open_pdf_page(pdf_path: str, page: int = 1) -> dict:
    """打开 pdf_path 并跳到第 page 页。返回 {ok, opened_with, page, note/error}。"""
    if not pdf_path or not os.path.isfile(pdf_path):
        return {"ok": False, "error": f"文件不存在: {pdf_path}"}
    page = max(1, int(page or 1))

    # 非 PDF（docx 等）：无页概念，系统默认程序打开
    if not pdf_path.lower().endswith(".pdf"):
        try:
            os.startfile(pdf_path)  # noqa: F821
            return {"ok": True, "opened_with": "default", "note": "非 PDF 文件，已用默认程序打开"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # 1) 自定义查看器（PDF_VIEWER + PDF_VIEWER_ARGS）
    viewer = os.getenv("PDF_VIEWER")
    if viewer and os.path.isfile(viewer):
        tpl = os.getenv("PDF_VIEWER_ARGS", '-page {page} "{path}"')
        cmd = tpl.format(page=page, path=pdf_path)
        try:
            subprocess.Popen(cmd, shell=True, creationflags=_CREATE_NO_WINDOW)
            return {"ok": True, "opened_with": os.path.basename(viewer), "page": page}
        except Exception as e:
            return {"ok": False, "error": f"自定义查看器启动失败: {e}"}

    # 2) SumatraPDF
    sumatra = shutil.which("SumatraPDF") or shutil.which("SumatraPDF.exe")
    for exe in ([sumatra] if sumatra else []) + _SUMATRA_PATHS:
        if exe and os.path.isfile(exe):
            try:
                subprocess.Popen(
                    [exe, "-page", str(page), pdf_path],
                    creationflags=_CREATE_NO_WINDOW,
                )
                return {"ok": True, "opened_with": "SumatraPDF", "page": page}
            except Exception as e:
                return {"ok": False, "error": f"SumatraPDF 启动失败: {e}"}

    # 3) Edge（file:///#page=N 锚点）
    for exe in _EDGE_PATHS:
        if os.path.isfile(exe):
            uri = "file:///" + pdf_path.replace("\\", "/") + f"#page={page}"
            try:
                subprocess.Popen(
                    ["cmd", "/c", "start", "", exe, uri],
                    creationflags=_CREATE_NO_WINDOW,
                )
                return {"ok": True, "opened_with": "Edge", "page": page}
            except Exception as e:
                return {"ok": False, "error": f"Edge 启动失败: {e}"}

    # 4) 系统默认程序（无法跳页，兜底）
    try:
        os.startfile(pdf_path)  # noqa: F821
        return {"ok": True, "opened_with": "default",
                "note": "未找到支持跳页的阅读器，已用默认程序打开（第 1 页）"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
