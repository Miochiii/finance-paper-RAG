# -*- coding: utf-8 -*-
"""文档元数据提取测试（年份/作者/KB 合并，无需 GPU 与 LLM）。"""
import json
import os

from rag_core.doc_metadata import (
    extract_author,
    extract_year,
    patch_kb_with_meta,
    save_meta,
    load_meta,
    DOC_META_FILE,
)


def test_year_arabic():
    assert extract_year("论文提交日期 2022年3月26日 答辩 2022年5月18日") == 2022


def test_year_chinese():
    assert extract_year("苏州科技大学 二〇二二 年 六 月") == 2022
    assert extract_year("二○二一年十二月") == 2021


def test_year_date_formats():
    assert extract_year("提交日期：2022-06-05 论文题目") == 2022
    assert extract_year("收稿日期：2026-03-25 网络首发日期：2026-08-10") == 2026
    assert extract_year("文章编号：1007-4392(2023)07-0039-15") == 2023


def test_year_skips_old_range():
    # 摘要里 "1980年-2020年" 是数据区间：应跳过 1980，取 2020
    assert extract_year("摘要：基于1980年-2020年长沙市日平均气温构建模型") == 2020


def test_year_invalid():
    assert extract_year("没有年份的封面文字") is None


def test_author_cover():
    assert extract_author("x.pdf", "学位申请人姓名 李安哲 培养单位 金融与统计学院") == "李安哲"
    assert extract_author("x.pdf", "学生姓名：＿＿＿＿ 汤孝海 指导教师") == "汤孝海"
    assert extract_author("x.pdf", "硕士研究生：张璐瑶 导师：") == "张璐瑶"


def test_author_filename_fallback():
    assert extract_author("基于机器学习的风险研究_王小明.pdf", "（封面无作者信息）") == "王小明"
    assert extract_author("无下划线文件.pdf", "（无作者信息）") is None


def test_meta_roundtrip_and_kb_patch(work_tmp):
    import rag_core.doc_metadata as dm
    dm.DOC_META_FILE = os.path.join(work_tmp, "doc_metadata.json")
    save_meta({"a.pdf": {"author": "甲", "year": 2023, "methods": ["深度学习"]}})
    assert load_meta()["a.pdf"]["year"] == 2023
    # KB 合并
    kb = os.path.join(work_tmp, "kb.json")
    with open(kb, "w", encoding="utf-8") as f:
        json.dump([
            {"text": "t1", "source": "a.pdf", "source_type": "hmm"},
            {"text": "t2", "source": "b.pdf", "source_type": "hmm"},
        ], f, ensure_ascii=False)
    n = patch_kb_with_meta(kb)
    assert n == 1
    loaded = json.load(open(kb, encoding="utf-8"))
    assert loaded[0]["author"] == "甲" and loaded[0]["year"] == 2023
    assert "author" not in loaded[1]
