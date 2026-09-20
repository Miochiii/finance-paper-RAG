# -*- coding: utf-8 -*-
"""E0 代理指标有效性验证工具的纯函数测试（不连服务、不加载模型）。

重点覆盖统计口径——这部分错了整个实验的结论就错了：
  1. Spearman 的退化情形（常数序列、样本过少）；
  2. Holm 校正的单调性与 nan 传递；
  3. 最小可检测效应随样本量单调下降（这是"要不要扩标注"的判据）；
  4. Fisher-z 合并；
  5. 回答侧代理的抽取规则（两种引用语法、拒答识别、长度剔除引用标记）。
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import probe_validity as pv


class TestSpearman:
    def test_perfect_monotone(self):
        rho, p = pv.spearman([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
        assert rho == pytest.approx(1.0) and p < 0.05

    def test_reversed(self):
        rho, _ = pv.spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
        assert rho == pytest.approx(-1.0)

    def test_degenerate_returns_nan(self):
        assert math.isnan(pv.spearman([1, 1, 1, 1], [1, 2, 3, 4])[0])
        assert math.isnan(pv.spearman([1, 2], [1, 2])[0])

    def test_bootstrap_ci_is_ordered_and_contains_estimate(self):
        x = list(range(20))
        y = [v * 2 + (1 if i % 3 else -1) for i, v in enumerate(x)]
        rho, _ = pv.spearman(x, y)
        lo, hi = pv.bootstrap_rho_ci(x, y, n_boot=200, seed=1)
        assert lo <= hi
        assert lo - 0.35 <= rho <= hi + 0.35

    def test_bootstrap_ci_nan_for_tiny_sample(self):
        lo, hi = pv.bootstrap_rho_ci([1, 2, 3], [1, 2, 3], n_boot=50)
        assert math.isnan(lo) and math.isnan(hi)


class TestHolm:
    def test_adjustment_is_monotone_and_bounded(self):
        ps = [0.001, 0.02, 0.03, 0.2]
        adj = pv.holm(ps)
        assert all(a >= p - 1e-12 for a, p in zip(adj, ps))     # 只会变大
        assert all(0 <= a <= 1 for a in adj)
        assert adj[0] == pytest.approx(0.004)                    # 最小 p × 4
        assert adj == sorted(adj)                                # 保持单调

    def test_nan_passthrough(self):
        adj = pv.holm([0.01, float("nan"), 0.04])
        assert math.isnan(adj[1])
        assert adj[0] <= adj[2]

    def test_single_value_unchanged(self):
        assert pv.holm([0.03])[0] == pytest.approx(0.03)


class TestMde:
    def test_decreases_with_n(self):
        assert pv.mde_spearman(40) > pv.mde_spearman(120) > pv.mde_spearman(400)

    def test_matches_closed_form_at_n40(self):
        # |ρ| ≈ t/sqrt(t²+df)，n=40 时约 0.31
        assert 0.29 < pv.mde_spearman(40) < 0.33

    def test_tiny_sample(self):
        assert math.isnan(pv.mde_spearman(3))


class TestFisherPool:
    def test_pool_of_identical_rhos(self):
        assert pv.fisher_pool([0.4, 0.4, 0.4], [40, 40, 40]) == pytest.approx(0.4, abs=1e-6)

    def test_empty(self):
        assert math.isnan(pv.fisher_pool([], []))

    def test_skips_invalid(self):
        assert pv.fisher_pool([float("nan"), 0.5], [40, 40]) == pytest.approx(0.5, abs=1e-6)


class TestNorm01:
    def test_range(self):
        assert pv.norm01([0, 5, 10]) == [0.0, 0.5, 1.0]

    def test_constant(self):
        assert pv.norm01([3, 3, 3]) == [0.5, 0.5, 0.5]


class TestAnswerSideProxies:
    def test_counts_both_citation_syntaxes(self):
        p = pv.answer_side_proxies("结论一（来源1）。结论二[来源3][来源4]。")
        assert p["ans_citations"] == 3
        assert p["ans_refusal"] == 0.0

    def test_detects_refusal(self):
        assert pv.answer_side_proxies("参考上下文中未找到相关信息，无法确定。")["ans_refusal"] == 1.0
        assert pv.answer_side_proxies("该论文使用 XGBoost 模型。")["ans_refusal"] == 0.0

    def test_length_excludes_citation_markers(self):
        short = pv.answer_side_proxies("这是结论。")["ans_length"]
        with_cit = pv.answer_side_proxies("这是结论。[来源1][来源2]")["ans_length"]
        assert with_cit == pytest.approx(short)

    def test_claim_count_ignores_short_fragments(self):
        p = pv.answer_side_proxies("好。这是一条足够长的中文结论内容。又一条足够长的中文结论内容。")
        assert p["ans_n_claims"] == 2
