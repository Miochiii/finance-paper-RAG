# -*- coding: utf-8 -*-
"""表格感知切块 + 页码归属 + 清洗 的回归测试（不需要 GPU/模型）。"""
from rag_core.chunk_splitter import (
    _clean_text,
    _recursive_split,
    _count_tokens,
    attribute_pages,
)


def _table(caption="表 1 评估指标", n_rows=5, wide=False):
    lines = ["[TABLE_START]", caption, "| 指标 | 数值 |", "| --- | --- |"]
    pad = "x" * (300 if wide else 3)
    for i in range(n_rows):
        lines.append(f"| 指标{i} | {pad} |")
    lines.append("[/TABLE_END]")
    return "\n".join(lines)


def _paired(chunk):
    s = chunk.count("[TABLE_START]")
    e = chunk.count("[/TABLE_END]") + chunk.count("[TABLE_END]")
    return s, e


def test_small_table_stays_whole():
    text = ("机器学习模型评估方法多种多样。" * 30) + "\n" + _table(n_rows=5) + "\n尾部说明。"
    chunks = _recursive_split(text, 800, 50)
    for c in chunks:
        s, e = _paired(c)
        if s or e:
            assert (s, e) == (1, 1), f"小表格被劈开: {s}/{e}"


def test_table_at_boundary_stays_whole():
    para = "机器学习模型评估方法多种多样，本文综合比较了多种指标。" * 25  # >800 token
    chunks = _recursive_split(para + "\n" + _table(n_rows=12), 800, 50)
    assert len(chunks) >= 2
    for c in chunks:
        s, e = _paired(c)
        if s or e:
            assert (s, e) == (1, 1), f"边界处表格被劈开: {s}/{e}"


def test_oversized_table_split_by_lines_only():
    text = "前言。" + _table(n_rows=40, wide=True)
    chunks = _recursive_split(text, 800, 50)
    # 超限表允许切分，但每块不严重超限，且表结束标记不丢失
    assert any("[/TABLE_END]" in c for c in chunks)
    assert max(_count_tokens(c) for c in chunks) <= 850


def test_sentence_newlines_preserved():
    chunks = _recursive_split("第一句话。\n第二句话。\n第三句话。", 800, 50)
    assert "\n" in chunks[0]


def test_malformed_table_no_crash():
    chunks = _recursive_split("正文。\n[TABLE_START]\n表 X\n| a | b |", 800, 50)
    assert len(chunks) >= 1


def test_attribute_pages_basic():
    raw = "【第1页】\n第一段内容。\n【第2页】\n第二段内容，带一个不【】同的标记。\n【第3页】\n第三段。"
    chunks = ["第一段内容。", "第二段内容，带一个不【】同的标记。", "第三段。"]
    ranges = attribute_pages(raw, chunks)
    assert ranges == [(1, 1), (2, 2), (3, 3)]


def test_attribute_pages_literal_newline():
    # 镜像 hmm_chunk 的字面 \n 还原
    raw = "【第1页】\\n第一段内容。\\n【第2页】\\n第二段。"
    chunks = ["第一段内容。", "第二段。"]
    ranges = attribute_pages(raw, chunks)
    assert ranges[0][0] == 1 and ranges[1][0] == 2


def test_clean_text_page_marker_residue():
    out = _clean_text("【第12页】\n正文第56页残留\n====Page 3====")
    assert "【】" in out  # 标记残留
    assert "第56页" not in out  # 页眉噪声
    assert "Page 3" not in out
