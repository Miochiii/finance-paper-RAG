# -*- coding: utf-8 -*-
"""HyDE 与 MMR 的纯函数/管线测试（不碰 GPU）。"""
import numpy as np

import rag_server as core
import rag_core.retriever as rt
import rag_core.query_processor as qp


class TestMmrPick:
    def test_diversity_penalizes_duplicates(self):
        scores = [5.0, 4.9, 4.8]
        sim = np.array([
            [1.0, 1.0, 0.0],   # 0 与 1 完全相同
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        picked = rt._mmr_pick(scores, sim, k=2, lam=0.6)
        assert picked[0] == 0          # 最高分先选
        assert picked[1] == 2          # 4.8-0 > 4.9-0.6，多样性胜出

    def test_lambda_zero_is_score_order(self):
        scores = [3.0, 2.0, 1.0]
        sim = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.1], [0.1, 0.1, 1.0]])
        assert rt._mmr_pick(scores, sim, k=3, lam=0.0) == [0, 1, 2]

    def test_k_geq_n_returns_all(self):
        scores = [1.0, 2.0]
        sim = np.eye(2)
        assert rt._mmr_pick(scores, sim, k=5, lam=0.6) == [0, 1]

    def test_empty(self):
        assert rt._mmr_pick([], np.zeros((0, 0)), k=3, lam=0.6) == []


class TestHydeHypothesis:
    def test_success_trims(self, monkeypatch):
        monkeypatch.setattr(qp, "_call_deepseek",
                            lambda p, q, timeout=45: "假设答案段落。" + "字" * 400)
        out = qp.hyde_hypothesis("问题")
        assert out and out.startswith("假设答案段落")
        assert len(out) <= 300

    def test_failure_returns_none(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("api down")
        monkeypatch.setattr(qp, "_call_deepseek", boom)
        assert qp.hyde_hypothesis("问题") is None


class TestSearchPipelinePassThrough:
    def test_search_kb_passes_hyde_and_mmr(self, monkeypatch):
        captured = {}

        class FakeRetriever:
            last_timing = {}

            def retrieve(self, query, **kw):
                captured.update(kw)
                return [{"index": 0, "text": "t", "metadata": {"source": "a.pdf"}}]

        monkeypatch.setattr(core, "get_retriever", lambda: FakeRetriever())
        monkeypatch.setattr(qp, "hyde_hypothesis", lambda q, timeout=45: "假设文档")
        monkeypatch.setattr(rt, "normalize_filters", lambda f: f)

        core.search_kb("问题", top_k=3, use_hyde=True, mmr=False)
        assert captured["hyde_text"] == "假设文档"
        assert captured["mmr"] is False
        core.search_kb("问题", top_k=3)
        assert captured["hyde_text"] is None
        assert captured["mmr"] is True
