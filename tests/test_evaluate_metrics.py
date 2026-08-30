# -*- coding: utf-8 -*-
"""evaluate.py 指标函数的单元测试。

覆盖 2026-08-28 的 nDCG 标定修复：
- 文档级：同一相关文档的多个块不得重复累积增益（旧实现 nDCG 可达 2.95）；
- 块级：同一条 gold 证据被多个块覆盖时只计一次增益（旧实现可 >1）。
"""
import math

import evaluate
from evaluate import retrieval_metrics, retrieval_metrics_chunk


def _res(source: str, text: str = "") -> dict:
    return {"metadata": {"source": source}, "text": text}


class TestRetrievalMetricsDocLevel:
    def test_single_gold_all_chunks_relevant_no_overflow(self):
        """回归：1 篇 gold + 5 块全部命中，旧实现 ndcg≈2.95，修复后必须恰为 1.0。"""
        results = [_res("A.pdf", "t") for _ in range(5)]
        recall, mrr, ndcg = retrieval_metrics(results, ["A.pdf"], k=5)
        assert recall == 1.0
        assert mrr == 1.0
        assert ndcg == 1.0

    def test_two_gold_expected_values(self):
        """gold 两篇，首次命中位次 1 与 4：dcg=1/log2(2)+1/log2(5)。"""
        results = [
            _res("A.pdf"), _res("X.pdf"), _res("X.pdf"), _res("B.pdf"), _res("X.pdf"),
        ]
        recall, mrr, ndcg = retrieval_metrics(results, ["A.pdf", "B.pdf"], k=5)
        assert recall == 1.0
        assert mrr == 1.0
        dcg = 1.0 / math.log2(2) + 1.0 / math.log2(5)
        idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
        assert abs(ndcg - dcg / idcg) < 1e-9
        assert ndcg <= 1.0

    def test_partial_hit(self):
        results = [_res("X.pdf"), _res("A.pdf")]
        recall, mrr, ndcg = retrieval_metrics(results, ["A.pdf", "B.pdf"], k=5)
        assert recall == 0.5
        assert mrr == 0.5
        assert 0.0 < ndcg < 1.0

    def test_no_hit(self):
        recall, mrr, ndcg = retrieval_metrics([_res("X.pdf")], ["A.pdf"], k=5)
        assert (recall, mrr, ndcg) == (0.0, 0.0, 0.0)

    def test_empty_gold(self):
        assert retrieval_metrics([_res("X.pdf")], [], k=5) == (0.0, 0.0, 0.0)


class TestRetrievalMetricsChunkLevel:
    def test_one_chunk_covers_two_golds(self):
        """一个块覆盖多条证据：该块只计一次增益，nDCG <= 1。"""
        gold = ["证据句甲", "证据句乙"]
        results = [{"text": "前文。证据句甲 与 证据句乙 都在此块。"}]
        recall, mrr, ndcg = retrieval_metrics_chunk(results, gold, k=5)
        assert recall == 1.0
        assert mrr == 1.0
        assert 0.0 < ndcg < 1.0

    def test_same_gold_in_multiple_chunks_counts_once(self):
        """回归：同一条证据被 3 个块覆盖，旧实现 dcg=2.13>idcg，修复后 ndcg 恰为 1.0。"""
        gold = ["证据句甲"]
        results = [{"text": "证据句甲"}, {"text": "证据句甲"}, {"text": "证据句甲"}]
        recall, mrr, ndcg = retrieval_metrics_chunk(results, gold, k=5)
        assert recall == 1.0
        assert mrr == 1.0
        assert ndcg == 1.0

    def test_page_marker_stripped_before_match(self):
        """页码标记（【第2页】）插在证据句中间时仍能逐字命中。"""
        gold = ["进行产品回测"]
        results = [{"text": "前文进行产【第2页】品回测的结论"}]
        recall, mrr, ndcg = retrieval_metrics_chunk(results, gold, k=5)
        assert recall == 1.0
        assert mrr == 1.0
        assert ndcg == 1.0

    def test_empty_gold(self):
        assert retrieval_metrics_chunk([{"text": "x"}], [], k=5) == (0.0, 0.0, 0.0)
