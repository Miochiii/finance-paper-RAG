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


# --------------------------------------------------------------------------
# 批次混淆修正（E0 v2 复核后新增的判定口径）
# --------------------------------------------------------------------------
class TestBatchAwareStats:
    def test_fisher_pool_test_single_batch(self):
        rho, p = pv.fisher_pool_test([0.4], [50])
        assert rho == pytest.approx(0.4, abs=1e-6) and p < 0.01

    def test_fisher_pool_test_skips_tiny_samples(self):
        rho, _ = pv.fisher_pool_test([0.9, 0.4], [3, 50])   # n<5 的批次被丢弃
        assert rho == pytest.approx(0.4, abs=1e-6)

    def test_fisher_pool_test_empty(self):
        rho, p = pv.fisher_pool_test([], [])
        assert math.isnan(rho) and math.isnan(p)

    def test_rho_diff_detects_sign_flip(self):
        assert pv.rho_diff_p(0.5, 100, -0.4, 100) < 0.01     # 方向相反 → 差异显著
        assert pv.rho_diff_p(0.30, 100, 0.32, 100) > 0.5      # 量级相近 → 不显著

    def test_rho_diff_nan_on_bad_input(self):
        assert math.isnan(pv.rho_diff_p(float("nan"), 100, 0.4, 100))

    def test_partial_rank_removes_control_effect(self):
        """控制变量完全解释两者时，残差无变异 → 偏相关无定义（返回 nan 是正确行为）。"""
        rows = [{"batch": "b1", "p": float(i), "t": float(i), "len": float(i)}
                for i in range(60)]              # p、t 完全由 len 决定
        rho, n = pv.partial_rho_rank_from_rows(rows, "p", "t", "len", batch_key="batch")
        assert math.isnan(rho) or abs(rho) < 0.2

    def test_partial_rank_keeps_true_relation(self):
        rows = [{"batch": "b1", "p": float(i % 7), "t": float(i % 7), "len": float(i % 13)}
                for i in range(60)]
        rho, _ = pv.partial_rho_rank_from_rows(rows, "p", "t", "len", batch_key="batch")
        assert rho > 0.8

    def test_partial_rank_handles_missing_control(self):
        rows = [{"batch": "b1", "p": 1.0, "t": 1.0, "len": float("nan")} for _ in range(20)]
        rho, n = pv.partial_rho_rank_from_rows(rows, "p", "t", "len", batch_key="batch")
        assert math.isnan(rho) and n == 0


class TestVarianceLevel:
    def test_degenerate_is_constant(self):
        assert pv.variance_level([5.0] * 20) == "degenerate"

    def test_rare_binary_event_is_usable(self):
        """拒答只有 2 例 → rare（可用但估计不稳），不能判死（它是最敏感的退化信号）。"""
        vals = [0.0] * 40 + [1.0] * 2
        assert pv.variance_level(vals) == "rare"
        assert pv.variance_ok(vals) is True

    def test_ok_binary_with_enough_positives(self):
        assert pv.variance_level([0.0] * 30 + [1.0] * 5) == "ok"

    def test_continuous(self):
        assert pv.variance_level([float(i) for i in range(50)]) == "ok"

    def test_balanced_binary_is_ok(self):
        """20/20 的平衡二元量是健康分布（不是 rare）。"""
        assert pv.variance_level([1.0, 2.0] * 20) == "ok"

    def test_multi_value_discrete_ok(self):
        assert pv.variance_level([1.0] * 10 + [2.0] * 10 + [3.0] * 10) == "ok"


class TestBatchMap:
    def test_loads_batch_column(self, work_tmp):
        p = os.path.join(work_tmp, "ann_batch.csv")
        with open(p, "w", encoding="utf-8-sig", newline="") as f:
            f.write("id,status,question,answer,gold_docs,gold_chunks,notes,batch\n")
            f.write("fin_001,done,q,a,d,c,n,v1_人工出题\n")
            f.write("fin_002,done,q,a,d,c,n,v2_块锚定\n")
            f.write("fin_003,pending,q,a,d,c,n,\n")
        assert pv.load_batch_map(p) == {"fin_001": "v1_人工出题", "fin_002": "v2_块锚定"}

    def test_missing_file_returns_empty(self, work_tmp):
        assert pv.load_batch_map(os.path.join(work_tmp, "nope.csv")) == {}

    def test_missing_column_returns_empty(self, work_tmp):
        p = os.path.join(work_tmp, "ann_nobatch.csv")
        with open(p, "w", encoding="utf-8-sig", newline="") as f:
            f.write("id,status,question,answer\nfin_001,done,q,a\n")
        assert pv.load_batch_map(p) == {}
