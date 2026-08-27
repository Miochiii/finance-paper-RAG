import os
import pdfplumber
from typing import List, Tuple, Dict
import re
#<editor-fold desc="PDF分析函数">
class PDFPageAnalyzer:
    """PDF页面分析器：评估每页是否为可提取文本页"""

    @staticmethod
    def estimate_page_area(page) -> float:
        """估算页面面积（平方点），用于计算文本密度"""
        if page.width and page.height:
            return page.width * page.height
        return 1.0  # 防止除零

    @staticmethod
    def is_garbled(text: str) -> bool:
        """检测文本是否包含大量乱码（非常见字符或控制字符）"""
        # 常见乱码特征：大量特殊符号、非中英文/数字字符
        if not text:
            return True
        # 统计有效字符比例（中文、英文、数字、常见标点）
        valid_chars = re.findall(r'[\u4e00-\u9fff\w\s\.,;!?()（）\u3002\uff01\uff1f]', text)
        ratio = len(valid_chars) / max(len(text), 1)
        return ratio < 0.6  # 低于60%视为乱码

    @staticmethod
    def text_quality_score(text: str) -> float:
        """
        返回文本质量分数（0-1）：
        - 分数越高，说明文本越可能是正常自然语言，越可信任
        """
        if not text:
            return 0.0
        # 1) 长度得分：至少10个字符才有基本分数
        len_score = min(len(text) / 50, 1.0)  # 50字符以上得分1
        # 2) 有效字符比例得分
        valid_ratio = len(re.findall(r'[\u4e00-\u9fff\w\s.,;!?()（）\u3002\uff01\uff1f]', text)) / max(len(text), 1)
        valid_score = min(valid_ratio, 1.0)
        # 3) 是否包含常见标点（表明句子结构）
        punct_score = 1.0 if re.search(r'[。！？\n.!?]', text) else 0.5
        # 4) 无连续乱码块（如一连串特殊符号）
        garbled_block_score = 1.0 if not re.search(r'[^a-zA-Z0-9\u4e00-\u9fff\s]{5,}', text) else 0.3
        # 综合得分（可调权重）
        score = 0.4 * valid_score + 0.3 * len_score + 0.2 * punct_score + 0.1 * garbled_block_score
        return round(score, 2)

    @staticmethod
    def analyze_page(page, min_text_len: int = 20, density_threshold: float = 0.01) -> Dict:
        """
        分析单个页面，返回：
        {
            'page_num': int,
            'raw_text': str,
            'quality_score': float,
            'is_scanned': bool,        # True表示该页应走OCR
            'extracted_text_len': int,
            'density': float
        }
        """
        raw_text = page.extract_text() or ""
        raw_text_len = len(raw_text.strip())
        area = PDFPageAnalyzer.estimate_page_area(page)
        # 文本密度 = 字符数 / 面积（单位面积字符数）
        density = raw_text_len / area if area > 0 else 0

        # 判定逻辑：修正版
        # 只要能提取到足够文本，就通过质量分判断，不卡密度阈值
        if raw_text_len >= min_text_len:
            quality = PDFPageAnalyzer.text_quality_score(raw_text)
            is_scanned = quality < 0.5
        else:
            is_scanned = True
            quality = PDFPageAnalyzer.text_quality_score(raw_text) if raw_text else 0.0

        return {
            'page_num': page.page_number,
            'raw_text': raw_text,
            'quality_score': quality,
            'is_scanned': is_scanned,
            'extracted_text_len': raw_text_len,
            'density': density
        }

    @classmethod
    def analyze_pdf(cls, file_path: str, min_text_len=20, density_threshold=0.01) -> List[Dict]:
        """分析整个PDF，返回每页的分析结果"""
        results = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                result = cls.analyze_page(page, min_text_len, density_threshold)
                results.append(result)
        return results

    @classmethod
    def analyze_pdf_fast(cls, file_path: str, min_text_len=20, density_threshold=0.01) -> List[Dict]:
        """用 PyMuPDF(fitz) 快速分析整个PDF，返回结构与 analyze_pdf 完全一致。

        fitz 的 page.get_text() 比 pdfplumber.extract_text() 快一个数量级，
        适合 evaluate.py 批量评测（每篇只分析一次）。判定逻辑（长度阈值 + 质量分）
        与原 analyze_pdf 保持一致。
        """
        import fitz
        results = []
        doc = fitz.open(file_path)
        try:
            for page in doc:
                raw_text = page.get_text() or ""
                raw_text_len = len(raw_text.strip())
                w, h = page.rect.width, page.rect.height
                area = (w * h) if (w and h) else 1.0
                density = raw_text_len / area if area > 0 else 0
                if raw_text_len >= min_text_len:
                    quality = PDFPageAnalyzer.text_quality_score(raw_text)
                    is_scanned = quality < 0.5
                else:
                    is_scanned = True
                    quality = PDFPageAnalyzer.text_quality_score(raw_text) if raw_text else 0.0
                results.append({
                    'page_num': page.number + 1,
                    'raw_text': raw_text,
                    'quality_score': quality,
                    'is_scanned': is_scanned,
                    'extracted_text_len': raw_text_len,
                    'density': density
                })
        finally:
            doc.close()
        return results
# </editor-fold>

class DocumentClassifier:
    def classify(self, file_path, page_results=None):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            # 调用页面分析器
            if page_results is None:
                page_results = PDFPageAnalyzer.analyze_pdf(file_path)
            # 统计扫描页和文本页比例
            total_pages = len(page_results)
            scanned_pages = sum(1 for r in page_results if r['is_scanned'])

            # 决策：
            if scanned_pages == 0:
                # 全部为高质量文本页 -> 纯文本PDF
                return {'extractor': 'pdfplumber', 'structure': 'DISCOURSE', 'chunk_size': 800, 'pdf_type': 'text'}
            elif scanned_pages == total_pages:
                # 全部为扫描页 -> 纯扫描PDF
                return {'extractor': 'OCR', 'structure': 'DISCOURSE', 'chunk_size': 600, 'pdf_type': 'scan'}
            else:
                # 混合PDF -> 标记为MIXED，后续加载时逐页处理
                return {'extractor': 'MIXED', 'structure': 'DISCOURSE', 'chunk_size': 800, 'pdf_type': 'mixed'}

        # ----- 第二步：结构层分类 -----
        if ext == '.docx':
            return {'extractor': 'python-docx', 'structure': 'HYBRID', 'chunk_size': 1000}

        if ext == '.xlsx':
            return {'extractor': 'pandas', 'structure': 'TABULAR', 'chunk_size': 'row_group_50'}

        if ext == '.json':
            return {'extractor': 'json.load', 'structure': 'NESTED', 'chunk_size': 'depth_aware'}

        if ext == '.py':
            return {'extractor': 'ast.parse', 'structure': 'CODE', 'chunk_size': 'func_aware'}

        # 兜底：未知文件类型
        return {'extractor': None, 'structure': 'UNKNOWN', 'chunk_size': None}

    # 新增：扫描PDF判断方法
    def is_scanned_pdf(self, file_path, min_text_len=10):
        """
        简单判断是否为扫描件PDF
        规则：提取所有页面文字总长度小于阈值 → 判定为扫描件
        """
        try:
            total_text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    total_text += page_text
                    if len(total_text) >= min_text_len:
                        # 提前终止，不用读完所有页
                        return False
            return len(total_text) < min_text_len
        except Exception:
            # PDF损坏/加密等异常，保守判定为扫描件，走OCR流程
            return True
classifier = DocumentClassifier()

if __name__ == "__main__":
    import sys
    file_path = sys.argv[1] if len(sys.argv) > 1 else input("输入文件路径: ").strip()
    cfg = classifier.classify(file_path)
    print(cfg)