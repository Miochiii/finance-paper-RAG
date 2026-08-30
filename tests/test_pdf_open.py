# -*- coding: utf-8 -*-
"""PDF 定位 / 分词词典 测试（不需要 GPU/模型）。"""
import os

from rag_core.pdf_open import resolve_pdf_path
from rag_core.retriever import _tokenize


def test_resolve_pdf_path_exact(work_tmp):
    os.makedirs(os.path.join(work_tmp, "sub"))
    target = os.path.join(work_tmp, "论文甲.pdf")
    with open(target, "w", encoding="utf-8") as f:
        f.write("x")
    # 顶层命中
    assert resolve_pdf_path("论文甲.pdf", [work_tmp]) == target
    # 一层子目录命中
    sub_target = os.path.join(work_tmp, "sub", "论文乙.pdf")
    with open(sub_target, "w", encoding="utf-8") as f:
        f.write("x")
    assert resolve_pdf_path("论文乙.pdf", [work_tmp]) == sub_target


def test_resolve_pdf_path_missing():
    assert resolve_pdf_path("不存在.pdf", [r"E:\file\agent\definitely-not-exist"]) is None


def test_tokenize_finance_terms():
    tokens = list(_tokenize("使用LightGBM和XGBoost构建动量反转策略，分析夏普比率与信息熵"))
    for w in ("LightGBM", "XGBoost", "动量反转", "夏普比率", "信息熵"):
        assert w in tokens, f"{w} 被切碎: {tokens}"
