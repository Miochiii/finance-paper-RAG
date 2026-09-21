# -*- coding: utf-8 -*-
"""B 步新增代理（检索一致性 + 散布/嵌入分布类）的纯函数测试。

守三条：
  1. **一致性度量的边界**：两边都空（都没召回）应判为一致；只有一边空则不一致；
     共同命中太少时排名一致性无定义（返回 nan），不能硬算出一个数；
  2. **散布量的正确性**：常数序列的 std/IQR 必须为 0、偏度定义为 0（不能除零）；
  3. **质心距离**：目标等于质心 → 0；正交 → 1；留一法要真的排除自身。
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import probe_extra as px


class TestConsistency:
    def test_jaccard_identical_and_disjoint(self):
        assert px.jaccard(["a", "b"], ["a", "b"]) == pytest.approx(1.0)
        assert px.jaccard(["a"], ["b"]) == pytest.approx(0.0)
        assert px.jaccard(["a", "b"], ["b", "a"]) == pytest.approx(1.0)   # 集合语义，与顺序无关

    def test_jaccard_both_empty_is_consistent(self):
        """两次检索都没命中时，算"一致"（都是没召回）而不是 0。"""
        assert px.jaccard([], []) == pytest.approx(1.0)

    def test_jaccard_partial(self):
        assert px.jaccard(["a", "b", "c"], ["b", "c", "d"]) == pytest.approx(0.5)

    def test_overlap_at_k(self):
        assert px.overlap_at_k(["a", "b", "c"], ["c", "b", "x"], 3) == pytest.approx(2 / 3)
        assert px.overlap_at_k(["a", "b", "c"], ["a", "z"], 3) == pytest.approx(1 / 3)

    def test_rank_corr_common_perfect(self):
        a = ["d1", "d2", "d3", "d4"]
        assert px.rank_corr_common(a, list(a)) == pytest.approx(1.0)
        assert px.rank_corr_common(a, list(reversed(a))) == pytest.approx(-1.0)

    def test_rank_corr_common_needs_enough_overlap(self):
        assert math.isnan(px.rank_corr_common(["a", "b"], ["c", "d"]))


class TestSpread:
    def test_constant_series(self):
        s = px.spread_stats([0.5, 0.5, 0.5, 0.5])
        assert s["ret_sim_std"] == pytest.approx(0.0)
        assert s["ret_sim_iqr"] == pytest.approx(0.0)
        assert s["ret_sim_range"] == pytest.approx(0.0)
        assert s["ret_sim_skew"] == pytest.approx(0.0)      # 除零保护

    def test_iqr_and_range(self):
        s = px.spread_stats([0.1, 0.2, 0.3, 0.4, 0.5])
        assert s["ret_sim_range"] == pytest.approx(0.4, abs=1e-9)
        assert s["ret_sim_iqr"] == pytest.approx(0.2, abs=1e-9)
        assert s["ret_sim_std"] > 0

    def test_insufficient_values(self):
        assert math.isnan(px.spread_stats([0.3])["ret_sim_std"])

    def test_ignores_nan(self):
        s = px.spread_stats([0.2, float("nan"), 0.4])
        assert math.isfinite(s["ret_sim_range"]) or math.isnan(s["ret_sim_range"])


class TestHerfindahl:
    def test_single_document_is_maximally_concentrated(self):
        assert px.herfindahl([5]) == pytest.approx(1.0)

    def test_uniform_across_five(self):
        assert px.herfindahl([1, 1, 1, 1, 1]) == pytest.approx(0.2)

    def test_skewed(self):
        assert px.herfindahl([4, 1]) == pytest.approx((0.8 ** 2 + 0.2 ** 2))

    def test_empty(self):
        assert math.isnan(px.herfindahl([]))


class TestCentroidDistance:
    def test_target_equals_centroid(self):
        v = np.array([[1.0, 0.0], [1.0, 0.0]])
        assert px.centroid_distance(v, np.array([1.0, 0.0])) == pytest.approx(0.0, abs=1e-9)

    def test_orthogonal_gives_one(self):
        v = np.array([[1.0, 0.0], [1.0, 0.0]])
        assert px.centroid_distance(v, np.array([0.0, 1.0])) == pytest.approx(1.0, abs=1e-9)

    def test_leave_one_out_excludes_self(self):
        """留一法：离群点自己不该拉高质心（否则离群度被低估）。"""
        v = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])   # 首行是"自己"
        d_all = px.centroid_distance(v, v[0])
        d_loo = px.centroid_distance(v, v[0], leave_one_out=True)
        assert d_all == pytest.approx(0.0, abs=1e-9) and d_loo == pytest.approx(0.0, abs=1e-9)
        v2 = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
        assert px.centroid_distance(v2, v2[0], leave_one_out=True) == pytest.approx(1.0, abs=1e-9)

    def test_empty_and_degenerate(self):
        assert math.isnan(px.centroid_distance(np.zeros((0, 3)), np.ones(3)))
        assert math.isnan(px.centroid_distance(np.zeros((2, 3)), np.ones(3)))


class TestChunkKey:
    def test_whitespace_insensitive(self):
        assert px.chunk_key("本文 使用\n随机森林") == px.chunk_key("本文使用随机森林")

    def test_truncates(self):
        assert len(px.chunk_key("长" * 500)) == 80

    def test_different_text_different_key(self):
        assert px.chunk_key("甲文") != px.chunk_key("乙文")
