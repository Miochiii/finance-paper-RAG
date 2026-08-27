# document_loader.py
import os
import pandas as pd
import json
import io
import pdfplumber
import fitz  # PyMuPDF: PDF页面→图片，轻量无外部依赖
from PIL import Image
import numpy as np
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from typing import List

# ========== RapidOCR 引擎 ==========

_OCR_ENGINE = None


def _get_ocr_engine():
    """懒加载 RapidOCR 单例（ONNX Runtime，无 PaddlePaddle 依赖）"""
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR_ENGINE = RapidOCR()
    return _OCR_ENGINE


def _pdf_to_images(file_path: str) -> List:
    """将PDF每一页渲染为PIL Image（200 DPI，OCR够用且更快）"""
    images = []
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        raise RuntimeError(f"无法打开PDF文件（文件可能已损坏）: {e}")

    if doc.is_encrypted:
        doc.close()
        raise ValueError("PDF文件已加密，无法读取")

    if len(doc) == 0:
        doc.close()
        raise ValueError("PDF文件无有效页面")

    try:
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            images.append(img)
    finally:
        doc.close()
    return images


def _pdf_pages_to_images(file_path: str, page_indices: List[int]) -> List:
    """只渲染指定页码（0-based）为PIL Image（200 DPI）。

    用于混合PDF：只渲染被判定为扫描页的页面，避免把整篇几十页都渲染成图，
    极大降低 CPU 开销与耗时（文本页根本不需要渲染）。
    """
    images = []
    if not page_indices:
        return images
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        raise RuntimeError(f"无法打开PDF文件（文件可能已损坏）: {e}")

    if doc.is_encrypted:
        doc.close()
        raise ValueError("PDF文件已加密，无法读取")

    try:
        for idx in sorted(page_indices):
            if 0 <= idx < len(doc):
                page = doc[idx]
                pix = page.get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                images.append(img)
    finally:
        doc.close()
    return images


def _ocr_images(images: List) -> List[str]:
    """RapidOCR 批量识别，返回每页文本列表，单页失败不中断"""
    ocr = _get_ocr_engine()
    page_texts = []
    for idx, img in enumerate(images):
        try:
            img_array = np.array(img)
            result, _ = ocr(img_array)
            if result:
                lines = [line[1] for line in result if line and len(line) > 1]
                page_text = "\n".join(lines)
            else:
                page_text = ""
            marker = (
                f"【第{idx + 1}页】\n{page_text}"
                if page_text
                else f"【第{idx + 1}页-无识别文字】\n"
            )
            page_texts.append(marker)
        except Exception as e:
            page_texts.append(f"【第{idx + 1}页-OCR识别异常: {e}】\n")
    return page_texts

# ========== RapidOCR 引擎 END ==========


#<editor-fold desc="混合PDF加载函数">
def load_pdf_mixed(file_path: str, page_results=None) -> str:
    """
    处理混合PDF（部分文本页 + 部分扫描页）：
    - 文本页：pdfplumber 提取（不渲染）
    - 扫描页：RapidOCR 识别（只渲染扫描页，不整篇渲染）
    返回合并文本，后续送入 DISCOURSE 分块。
    """
    from rag_core.classify_file import PDFPageAnalyzer

    if page_results is None:
        page_results = PDFPageAnalyzer.analyze_pdf(file_path)
    # 只收集扫描页索引（封面/个别扫描页），其余文本页完全不渲染
    scanned_indices = [i for i, r in enumerate(page_results) if r["is_scanned"]]
    scanned_images = _pdf_pages_to_images(file_path, scanned_indices)

    full_text_parts = []
    with pdfplumber.open(file_path) as pdf:
        for idx, page in enumerate(pdf.pages):
            if idx >= len(page_results):
                # 陈旧/不匹配的 page_results（页面数与实际不符）→ 保守按文本页处理，不崩
                raw_text = page.extract_text() or ""
                full_text_parts.append(f"【第{idx + 1}页-文本】\n{raw_text}")
                continue
            result = page_results[idx]
            if result["is_scanned"]:
                pos = scanned_indices.index(idx) if idx in scanned_indices else -1
                if pos >= 0 and pos < len(scanned_images):
                    ocr_results = _ocr_images([scanned_images[pos]])
                    full_text_parts.append(
                        ocr_results[0]
                        if ocr_results
                        else f"【第{idx + 1}页-OCR结果为空】"
                    )
                else:
                    full_text_parts.append(f"【第{idx + 1}页-OCR失败：页面图片缺失】")
            else:
                raw_text = result["raw_text"]
                full_text_parts.append(f"【第{idx + 1}页-文本】\n{raw_text}")
    return "\n".join(full_text_parts)
#</editor-fold>

# ========== 图文描述提取 ==========

_DEEPSEEK_KEY = os.getenv("deepseek_api", "")  # 从环境变量读取，勿硬编码
_IMAGE_MIN_SIZE = 150  # 最小图片边长（像素），过滤图标/logo


def _describe_page_images(fitz_page, page_num: int) -> List[str]:
    """
    提取一页中的嵌入式图片，OCR 扫文字，LLM 生成中文描述。
    返回描述文本列表，空图片/无文字图片返回空列表。
    """
    import io

    image_list = fitz_page.get_images(full=True)
    if not image_list:
        return []

    descriptions = []
    for idx, img_info in enumerate(image_list[:5]):  # 每页最多5张
        try:
            xref = img_info[0]
            base_image = fitz_page.parent.extract_image(xref)
            img_bytes = base_image["image"]
            w, h = base_image.get("width", 0), base_image.get("height", 0)

            # 过滤小图标
            if w < _IMAGE_MIN_SIZE and h < _IMAGE_MIN_SIZE:
                continue

            # OCR 提取图中文字
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            ocr_text = ""
            try:
                ocr = _get_ocr_engine()
                result, _ = ocr(np.array(img))
                if result:
                    lines = [line[1] for line in result if line and len(line) > 1]
                    ocr_text = " ".join(lines)
            except Exception:
                pass

            if not ocr_text.strip():
                continue  # 纯图片无文字标注，跳过

            # LLM 生成描述
            prompt = f"根据以下图表OCR提取的文字，用1-2句话描述图表类型和可能展示的内容。直接输出中文描述：\n{ocr_text[:400]}"
            try:
                from openai import OpenAI
                client = OpenAI(api_key=_DEEPSEEK_KEY, base_url="https://api.deepseek.com")
                resp = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3, max_tokens=200, timeout=60,
                )
                desc = resp.choices[0].message.content.strip()
            except Exception:
                desc = "图表（LLM不可用）"

            descriptions.append(
                f"[图片{page_num}-{idx + 1}] 图表文字: {ocr_text[:200]} | 描述: {desc}"
            )
        except Exception:
            continue

    return descriptions


# ========== 版式感知 PDF 文本提取 ==========

def _table_to_md(cells: List[List[str]]) -> str:
    """pdfplumber 表格单元格 → Markdown 字符串"""
    if not cells or not cells[0]:
        return ""
    rows = []
    for i, row in enumerate(cells):
        cleaned = [str(c).replace("\n", " ") if c else "" for c in row]
        rows.append("| " + " | ".join(cleaned) + " |")
        if i == 0:
            rows.append("|" + "|".join([" --- " for _ in row]) + "|")
    return "\n".join(rows)


def _bbox_overlap(a: tuple, b: tuple) -> bool:
    """两个 bbox (x0, y0, x1, y1) 是否有重叠"""
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def _extract_text_structured(file_path: str) -> str:
    """
    版式感知提取：
    - PyMuPDF(fitz) 获取文本块坐标，按阅读顺序排序（解决双栏乱序）
    - pdfplumber 检测表格，转为 Markdown 格式
    - 表格区域覆盖的零散文本块自动剔除，代以结构化 Markdown 表格
    """
    doc = fitz.open(file_path)
    all_parts = []

    pdf = pdfplumber.open(file_path)
    try:
        for page_idx, page in enumerate(pdf.pages):
            fitz_page = doc[page_idx]

            # 1. 检测表格并转为 Markdown
            table_mds: List[tuple] = []  # [(bbox, md_text), ...]
            for table in page.find_tables():
                cells = table.extract()
                md = _table_to_md(cells)
                if md:
                    table_mds.append((table.bbox, md))

            # 2. 获取 fitz 文本块，按阅读顺序排序
            blocks = fitz_page.get_text("blocks")
            # 过滤纯空白块、图片块
            text_blocks = []
            for b in blocks:
                txt = b[4].strip() if len(b) > 4 else ""
                if txt and len(txt) > 1:  # 过滤单字符碎片
                    text_blocks.append(b)

            if not text_blocks and not table_mds:
                continue

            # 按 y → x 排序（自然阅读顺序）
            text_blocks.sort(key=lambda b: (round(b[1], 1), b[0]))

            # 3. 合并：文本块 + 表格，按坐标位置交错输出
            output_lines = []
            inserted_tables = set()
            table_idx = 0
            # 表格按 y 坐标排序
            table_mds.sort(key=lambda t: t[0][1])

            for block in text_blocks:
                bx0, by0, bx1, by1 = block[0], block[1], block[2], block[3]
                bbox = (bx0, by0, bx1, by1)

                # 在当前文本块之前，插入所有 y 坐标在其上的表格
                while table_idx < len(table_mds):
                    t_bbox, t_md = table_mds[table_idx]
                    if t_bbox[1] <= by1 and table_idx not in inserted_tables:
                        output_lines.append(f"\n[TABLE]\n{t_md}\n[/TABLE]\n")
                        inserted_tables.add(table_idx)
                        table_idx += 1
                    else:
                        break

                # 检查当前文本块是否在表格区域内
                in_table = False
                for ti, (t_bbox, _) in enumerate(table_mds):
                    if _bbox_overlap(bbox, t_bbox):
                        in_table = True
                        break

                if not in_table:
                    output_lines.append(block[4].strip())

            # 剩余未输出的表格
            while table_idx < len(table_mds):
                _, t_md = table_mds[table_idx]
                if table_idx not in inserted_tables:
                    output_lines.append(f"\n[TABLE]\n{t_md}\n[/TABLE]\n")
                    inserted_tables.add(table_idx)
                table_idx += 1

            page_text = "\n".join(output_lines)

            # 提取图片描述，追加到页尾
            img_descs = _describe_page_images(fitz_page, page_idx + 1)
            if img_descs:
                page_text += "\n\n" + "\n".join(img_descs)

            if page_text.strip():
                all_parts.append(f"【第{page_idx + 1}页】\n{page_text}")
    finally:
        # 异常时也要关闭两个 PDF 句柄，避免泄漏
        pdf.close()
        doc.close()
    return "\n\n".join(all_parts)


# ========== 主加载函数 ==========

def load_document(file_path: str, extractor_name: str, page_results=None):
    """根据分类器返回的extractor，加载原始内容"""
    if extractor_name == "pdfplumber":
        return _extract_text_structured(file_path)

    elif extractor_name == "OCR":
        # 扫描件PDF：RapidOCR 识别，结果复用 DISCOURSE 分块逻辑
        images = _pdf_to_images(file_path)
        if not images:
            raise RuntimeError(f"PDF页面渲染失败: {file_path}")
        page_texts = _ocr_images(images)
        full_text = "\n\n".join(page_texts)
        if not full_text.strip():
            raise RuntimeError(f"OCR识别结果为空，请检查PDF文件质量: {file_path}")
        return full_text

    elif extractor_name == "python-docx":
        # 按文档真实顺序提取段落 + 表格（表格转 Markdown 并包裹标记），
        # 修复：原来只读 paragraphs 导致表格内容整体丢失
        return _extract_docx_content(file_path)

    elif extractor_name == "pandas":
        df = pd.read_excel(file_path)
        return df.to_string()

    elif extractor_name == "json.load":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return json.dumps(data, ensure_ascii=False, indent=2)

    elif extractor_name == "ast.parse":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif extractor_name == "MIXED":
        return load_pdf_mixed(file_path, page_results=page_results)

    else:
        raise ValueError(f"不支持的解析器 {extractor_name}")

def _table_to_markdown(table: Table) -> str:
    """将docx表格转为Markdown表格字符串"""
    rows = []
    for i, row in enumerate(table.rows):
        cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
        rows.append("| " + " | ".join(cells) + " |")
        if i == 0:
            # 添加分隔线
            sep = "|" + "|".join([" --- " for _ in cells]) + "|"
            rows.append(sep)
    return "\n".join(rows)


def _extract_docx_content(file_path: str) -> str:
    """按文档真实顺序提取段落与表格（表格转 Markdown 并用 [TABLE_START]/[TABLE_END] 包裹），
    保证 HYBRID 分块器能识别表格并按行组切分。"""
    doc = Document(file_path)
    parts = []
    for child in doc.element.body.iterchildren():
        if child.tag.endswith('}p'):
            para = Paragraph(child, doc)
            text = para.text.strip()
            if text:
                parts.append(text)
        elif child.tag.endswith('}tbl'):
            table = Table(child, doc)
            md = _table_to_markdown(table)
            if md:
                parts.append(f"[TABLE_START]\n{md}\n[/TABLE_END]")
    return "\n\n".join(parts)