# -*- coding: utf-8 -*-
"""研究方向工作台测试（纯函数部分，不依赖 GPU 与 LLM）。"""
from rag_core.research_advisor import (
    DEFAULT_WEIGHTS,
    _features,
    _score_breakdown,
    compare_directions,
)


def test_features_math():
    f = _features({"docs": [
        {"year": 2023, "methods": ["深度学习"]},
        {"year": 2021, "methods": ["深度学习", "集成学习"]},
        {"year": None, "methods": ["时间序列"]},
    ]})
    assert f["docs"] == 3
    assert f["recency"] == 0.5          # 2 篇有年份，1 篇 >=2023
    assert f["distinct_methods"] == 3
    assert f["maturity"] == round(3 / 6, 3)


def test_score_breakdown_and_weight_normalization():
    f = {"recency": 1.0, "maturity": 0.5, "docs": 2}
    llm = {"feasibility": 4, "data_score": 2}
    s = _score_breakdown(f, lit_norm=0.5, llm=llm, weights=DEFAULT_WEIGHTS)
    total_w = sum(DEFAULT_WEIGHTS.values())
    assert abs(s["total"] - (
        0.5 * 0.15 + 1.0 * 0.10 + 0.5 * 0.10 + 0.5 * 0.25
        + (2 / 5) * 0.15 + (4 / 5) * 0.25) / total_w) < 1e-6


def test_compare_directions_ranking(monkeypatch):
    import rag_core.research_advisor as ra

    # 两个方向的热度/方法分布完全相同，仅文献量不同 → gap（创新空间）项决定排序
    common_docs = [{"source": "a.pdf", "author": "甲", "year": 2024, "methods": ["深度学习"]},
                   {"source": "b.pdf", "author": "乙", "year": 2023, "methods": ["集成学习"]}]
    stats_a = {"doc_count": 6, "docs": common_docs}
    stats_b = {"doc_count": 2, "docs": common_docs}
    fake_stats = {"A": stats_a, "B": stats_b}
    monkeypatch.setattr(ra, "_retrieve_stats",
                        lambda direction, top_k=12: fake_stats[direction])
    monkeypatch.setattr(ra, "_llm_json", lambda s, u: {
        "summary": "s", "innovations": [], "feasibility": 5, "data_score": 5,
        "risks": [], "gap": "g", "suggested_plan": "p",
    })

    r = compare_directions(["A", "B"])
    assert r["ok"]
    # 文献少的 B 创新空间大（gap 权重 0.25 > literature 0.15）→ B 排第一
    assert r["ranking"][0]["direction"] == "B"
    assert abs(sum(r["weights"].values()) - 1.0) < 1e-9
    # 字段完整性
    assert set(r["ranking"][0]) == {"rank", "direction", "score", "summary"}
    assert "recommendation" in r and r["recommendation"]["direction"] == "B"


def test_compare_requires_two_directions():
    r = compare_directions(["只有一个"])
    assert r["ok"] is False and "至少" in r["error"]
