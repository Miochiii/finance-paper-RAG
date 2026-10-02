import numpy as np

import generate_change_point_candidates as gen


def test_fresh_excerpts_prefer_unused_evidence_and_distinct_pages():
    chunks = [
        {"text": "旧证据表明模型有效。研究结果显示方法稳定。" * 10, "page_start": 1},
        {"text": "新段落介绍数据来源。实验结果显示效果提升。" * 10, "page_start": 2},
        {"text": "同页内容讨论模型结构。本文分析不同方法。" * 10, "page_start": 2},
    ]
    picked = gen.select_fresh_excerpts(chunks, ["旧证据表明模型有效。研究结果显示方法稳定。" * 3],
                                       np.random.default_rng(0), k=2)
    assert len(picked) == 2
    assert picked[0]["page_start"] == 2
    assert {x["page_start"] for x in picked} == {1, 2}


def test_generated_candidate_stays_pending_and_has_verbatim_evidence():
    excerpt = {"text": "该模型在测试集上的准确率达到百分之九十。" * 5,
               "page_start": 4}
    item = {"q": "该模型的测试集准确率如何？", "a": "百分之九十",
            "type": "指标", "evidence": [1],
            "quote": "该模型在测试集上的准确率达到百分之九十。"}
    row = gen.candidate_from_item(item, "论文.pdf", [excerpt], [excerpt], [],
                                  {"answer_supported": True, "unique": True})
    assert row["status"] == "pending"
    assert row["gold_chunks"] in excerpt["text"]
    assert row["verify"] == "ok"
    assert row["fact_missing"] == 0


def test_quote_with_changed_whitespace_uses_verbatim_fallback():
    excerpt = {"text": "该模型在测试集上的准确率达到百分之九十。",
               "page_start": 4}
    item = {"q": "测试集准确率是多少？", "a": "百分之九十",
            "type": "指标", "evidence": [1],
            "quote": "该模型在测试集上的准确率达到 百分之九十。"}
    row = gen.candidate_from_item(item, "论文.pdf", [excerpt], [excerpt], [],
                                  {"answer_supported": True, "unique": True})
    assert row["gold_chunks"] in excerpt["text"]
    assert row["quote_fallback"] == 1
