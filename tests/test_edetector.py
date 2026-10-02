# -*- coding: utf-8 -*-
"""e-detector 最小实现的纯函数测试。

守四条线（都是踩过坑的地方）：
  1. **有界变量构造的合法性**：X ≤ m 时 L ≤ 1（e-detector 不增长），M 恒 ≥ 1；
  2. **混合封闭性**（命题 2.3）与**取最大不合法**（Remark 3.1）：max ≥ mix ≥ min 逐点成立，
     因此同阈值下 max 必然更早报警——这是"误报膨胀"的确定性机制；
  3. **多流维度不能丢**：build_detectors 必须保留重复流维度（早期版本误取 [0]，
     统计量看着正常其实只算了一条流）；
  4. **无前视**：归一化尺度只能用校准段定，否则会把变点后的漂移值算进尺度。
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import edetector as ed


class TestBoundedConstruction:
    def test_increment_le_one_below_bound(self):
        """X ≤ m ⇒ L ≤ 1（这是 ARL ≥ 1/α 的根基）。"""
        for m in (0.2, 0.5, 0.9):
            for lam in (0.1, 0.5, 0.9):
                x = np.linspace(0, m, 11)
                L = 1.0 + lam * (x / m - 1.0)
                assert (L <= 1.0 + 1e-12).all()

    def test_increment_exceeds_one_above_bound(self):
        assert 1.0 + 0.5 * (0.9 / 0.5 - 1.0) > 1.0

    def test_no_growth_when_below_bound(self):
        """X 恒 ≤ m ⇒ M 恒 ≤ 1 ⇒ 永不可能报警（ARL ≥ 1/α 的直观来源）。

        注意 M 可以被 L≤1 拉到 1 以下（如恒为 0.55）——这是"自重启"语义而非缺陷：
        阈值 1/α > 1，低于 1 的取值不触发报警。所以这里断言的是 M ≤ 1，而不是 M ≥ 1。
        """
        x = np.full(50, 0.05)
        M = ed.cumulative_evalue_series(x, m=0.5, lam=0.5)[0]
        assert (M <= 1.0 + 1e-12).all()
        assert (M >= 0.0).all()
        assert ed.first_alarm(M, 1.0 / 0.05)[0] == 0

    def test_evalue_grows_when_above_bound(self):
        x = np.full(40, 0.9)
        M = ed.cumulative_evalue_series(x, m=0.5, lam=0.5)[0]
        assert M[-1] > M[0] and M[-1] > 1.0

    def test_numerical_cap_prevents_overflow(self):
        x = np.full(200, 1.0)
        M = ed.cumulative_evalue_series(x, m=1e-3, lam=0.9, cap=1e6)[0]
        assert np.isfinite(M).all() and M.max() <= 1e6 + 1e-6

    def test_keeps_repetition_axis(self):
        """回归：多条流的维度不能被压掉。"""
        x = np.random.default_rng(0).random((7, 30))
        M = ed.cumulative_evalue_series(x, m=0.5, lam=0.5)
        assert M.shape == (7, 30)
        assert ed.build_detectors(x, 0.5, np.array([0.2, 0.5]), "mix").shape == (7, 30)


class TestFusion:
    def _series(self):
        rng = np.random.default_rng(1)
        x = rng.random((4, 60))
        lams = np.array([0.2, 0.5, 0.8])
        return [ed.cumulative_evalue_series(x, m=0.5, lam=float(l)) for l in lams]

    def test_pointwise_order_max_mix_min(self):
        """Remark 3.1 的确定性机制：max ≥ mix ≥ min（逐点）。"""
        s = self._series()
        mx = ed.fuse_series(s, "max")
        mix = ed.fuse_series(s, "mix")
        mn = ed.fuse_series(s, "min")
        assert (mx >= mix - 1e-12).all() and (mix >= mn - 1e-12).all()

    def test_max_alarms_no_later_than_mix(self):
        s = self._series()
        t_mx = ed.first_alarm(ed.fuse_series(s, "max"), 20.0)
        t_mix = ed.first_alarm(ed.fuse_series(s, "mix"), 20.0)
        both = (t_mx > 0) & (t_mix > 0)
        assert (t_mx[both] <= t_mix[both]).all()

    def test_weights_are_normalised(self):
        s = self._series()
        mix = ed.fuse_series(s, "mix", weights=[2.0, 3.0, 5.0])   # 会自动归一化
        assert np.isfinite(mix).all()

    def test_single_uses_first_component(self):
        s = self._series()
        assert np.allclose(ed.fuse_series(s, "single"), s[0])


class TestFirstAlarm:
    def test_finds_first_crossing(self):
        series = np.array([[1.0, 1.0, 25.0, 30.0], [1.0, 1.0, 1.0, 1.0]])
        assert list(ed.first_alarm(series, 20.0)) == [3, 0]

    def test_no_alarm_returns_zero(self):
        assert list(ed.first_alarm(np.ones((3, 10)), 20.0)) == [0, 0, 0]


class TestEstimateM:
    def test_ordering_of_strategies(self):
        rng = np.random.default_rng(2)
        calib = rng.random(200) * 0.6
        m_mean = ed.estimate_m(calib, "mean")
        m_p95 = ed.estimate_m(calib, "p95")
        m_hoef = ed.estimate_m(calib, "hoeffding")
        m_max = ed.estimate_m(calib, "max")
        assert m_mean <= m_p95 <= m_max
        assert m_hoef >= m_mean                    # 上界必须不小于均值
        assert m_max <= 1.0 + 1e-9

    def test_binom_bound_is_tighter_than_hoeffding_for_rare_events(self):
        """稀有事件（58 次里 1 次）用精确二项上界应显著紧于 Hoeffding 上界。"""
        calib = np.zeros(58)
        calib[0] = 1.0
        m_binom = ed.estimate_m(calib, "binom")
        m_hoef = ed.estimate_m(calib, "hoeffding")
        assert calib.mean() <= m_binom <= m_hoef <= 1.0
        assert m_binom < 0.15          # Clopper-Pearson 上界约 0.09

    def test_binom_bound_is_at_least_empirical_rate(self):
        rng = np.random.default_rng(11)
        calib = (rng.random(200) < 0.3).astype(float)
        m = ed.estimate_m(calib, "binom")
        assert m >= calib.mean() - 1e-9 and m <= 1.0

    def test_quantile_strategies_collapse_on_binary_data(self):
        """记录一个反直觉事实：二元数据上分位数策略会退化成 1.0（m=1 ⇒ 永不报警）。

        所以 --binarize 的默认 m 策略是 binom/hoeffding，而不是 q60/q85/q100。
        """
        calib = (np.arange(60) % 3 != 0).astype(float)      # 正例率约 2/3
        assert ed.estimate_m(calib, "q60") == pytest.approx(1.0)
        assert ed.estimate_m(calib, "binom") < 1.0

    def test_empty_calibration(self):
        assert ed.estimate_m(np.array([]), "p95") == 1.0

    def test_unknown_strategy(self):
        with pytest.raises(ValueError):
            ed.estimate_m(np.array([0.5]), "nope")


class TestLambdaAndBounds:
    def test_grid_within_open_unit_interval(self):
        lams = ed.lambda_grid(0.5, n_lam=6)
        assert lams.size == 6 and (lams > 0).all() and (lams < 1).all()

    def test_delta_bounds_positive(self):
        dl, du = ed.delta_bounds(0.5)
        assert dl > 0 and du > dl


class TestNoLookahead:
    def test_fit_unit_uses_calibration_only(self):
        calib = np.array([0.0, 0.5, 1.0])
        x = np.array([0.0, 0.5, 1.0, 5.0])          # 5.0 是"变点后"的越界值
        u = ed.fit_unit(calib, x)
        assert u.max() == pytest.approx(1.0)        # 截断到 1，不把 5.0 纳入尺度
        assert u[-1] == pytest.approx(1.0)

    def test_fit_unit_constant_calibration(self):
        assert np.allclose(ed.fit_unit(np.array([0.3, 0.3]), np.array([0.3, 0.9])),
                           [0.5, 1.0])


class TestDriftInjection:
    def test_shift_mode_clips(self):
        s = np.full((1, 10), 0.9)
        out = ed.inject_drift(s, at=5, delta=0.5, direction=+1, mode="shift")
        assert out[0, :5].max() == pytest.approx(0.9)
        assert out[0, 5:].max() <= 1.0

    def test_contaminate_hits_expected_mean_shift(self):
        rng = np.random.default_rng(3)
        s = np.full((2000, 20), 0.4)
        out = ed.inject_drift(s, at=1, delta=0.2, direction=+1, mode="contaminate", rng=rng)
        shift = out[:, 1:].mean() - 0.4
        assert shift == pytest.approx(0.2, abs=0.03)

    def test_contaminate_direction_down(self):
        rng = np.random.default_rng(4)
        s = np.full((1000, 10), 0.6)
        out = ed.inject_drift(s, at=1, delta=0.3, direction=-1, mode="contaminate", rng=rng)
        assert out[:, 1:].mean() < 0.6

    def test_no_injection_at_boundary(self):
        s = np.full((2, 5), 0.5)
        assert np.allclose(ed.inject_drift(s, at=0, delta=0.3, direction=1), s)
        assert np.allclose(ed.inject_drift(s, at=99, delta=0.3, direction=1), s)


class TestE1MultiIndicator:
    """E1：多指标混合。要点是**保持指标间相关结构**与**逐指标用自己的 m**。"""

    def test_is_inf_recognizes_both_representations(self):
        """inf 既可能是浮点也可能是字符串——只判一种会让「未检出」混进有效结果。"""
        assert ed.is_inf(float("inf")) and ed.is_inf("inf") and ed.is_inf(" nan ")
        assert ed.is_inf(float("nan")) and ed.is_inf(None) and ed.is_inf("")
        assert not ed.is_inf(0.0) and not ed.is_inf("12.5") and not ed.is_inf(7)

    def test_edd_horizon_excludes_late_false_alarms(self):
        """变点后很久才响的报警不能算检出：那是零假设侧的偶发误报。"""
        M = np.ones((2, 100))
        M[0, 49] = 2.0        # 第 50 步报警 → 变点(40) 后 10 步
        M[1, 89] = 2.0        # 第 90 步报警 → 变点后 50 步
        s = ed._edd_stats(M, alpha=0.5, at=40, T_post=60, horizon=20)
        assert s["detect_rate"] == pytest.approx(0.5)
        assert s["late_alarm_rate"] == pytest.approx(0.5)
        assert s["edd_mean"] == pytest.approx(10.0)
        assert s["edd_horizon"] == 20
        # 不限视界 = 旧口径：两条都算检出，EDD 被 50 步的那条拉高
        s0 = ed._edd_stats(M, alpha=0.5, at=40, T_post=60, horizon=0)
        assert s0["detect_rate"] == pytest.approx(1.0)
        assert s0["late_alarm_rate"] == 0.0
        assert s0["edd_mean"] == pytest.approx(30.0)

    def test_edd_horizon_pre_alarm_not_counted_as_detection(self):
        """变点前就报警的算变前误报，既不进检出也不进超时报警。"""
        M = np.ones((2, 100))
        M[0, 19] = 2.0        # 第 20 步报警，变点在第 40 步之前
        s = ed._edd_stats(M, alpha=0.5, at=40, T_post=60, horizon=20)
        assert s["pre_alarm_rate"] == pytest.approx(0.5)
        assert s["detect_rate"] == 0.0 and s["late_alarm_rate"] == 0.0
        assert ed.is_inf(s["edd_mean"])

    def test_series_matrix_intersects_rows(self):
        rows = [
            {"qid": "a", "method": "hmm", "x": "0.1", "y": "0.2"},
            {"qid": "b", "method": "hmm", "x": "0.3", "y": ""},      # y 缺失 → 整行丢弃
            {"qid": "c", "method": "hmm", "x": "0.5", "y": "0.6"},
            {"qid": "d", "method": "fixed", "x": "9", "y": "9"},     # 非目标分块 → 丢弃
        ]
        X, qids = ed.series_matrix(rows, ["x", "y"])
        assert qids == ["a", "c"]
        assert X.shape == (2, 2) and X[1].tolist() == [0.5, 0.6]

    def test_bootstrap_rows_keeps_within_row_dependence(self):
        """联合重采样必须保持同一行内指标的相关性（否则混合效果会被高估）。"""
        rng = np.random.default_rng(20)
        base = rng.random((80, 1))
        pool = np.hstack([base, 3.0 * base])          # 第二列恒为第一列的 3 倍
        s = ed.bootstrap_rows(pool, reps=5, T=30, block=10, rng=rng)
        assert s.shape == (5, 30, 2)
        assert np.allclose(s[:, :, 1], 3.0 * s[:, :, 0])   # 相关性被完整保留

    def test_bootstrap_rows_shape_and_range(self):
        rng = np.random.default_rng(21)
        pool = rng.random((50, 3))
        s = ed.bootstrap_rows(pool, reps=4, T=25, block=5, rng=rng)
        assert s.shape == (4, 25, 3) and s.min() >= 0 and s.max() <= 1

    def test_mixture_between_min_and_max(self):
        rng = np.random.default_rng(22)
        streams = rng.random((6, 40, 3))
        ms = [0.5, 0.5, 0.5]
        lams = ed.lambda_grid(0.5, n_lam=3)
        mix = ed.build_multi_detector(streams, ms, lams, "mix")
        mx = ed.build_multi_detector(streams, ms, lams, "max")
        mn = ed.build_multi_detector(streams, ms, lams, "min")
        assert mix.shape == (6, 40)
        assert (mx >= mix - 1e-9).all() and (mix >= mn - 1e-9).all()

    def test_single_indicator_uses_only_that_column(self):
        """how='one' 只应看指定指标：把别的指标灌成极值也不该影响它。"""
        reps, T = 4, 60
        calm = np.full((reps, T), 0.05)
        spike = np.full((reps, T), 0.95)
        streams = np.stack([spike, calm], axis=2)      # 指标0 爆炸、指标1 平静
        lams = ed.lambda_grid(0.5, n_lam=3)
        m_only0 = ed.build_multi_detector(streams, [0.5, 0.5], lams, "one", single_k=0)
        m_only1 = ed.build_multi_detector(streams, [0.5, 0.5], lams, "one", single_k=1)
        assert ed.first_alarm(m_only0, 2.0).min() > 0        # 指标0 很快报警
        assert (ed.first_alarm(m_only1, 2.0) == 0).all()     # 指标1 永不报警

    def test_per_indicator_bounds_are_used(self):
        """每个指标用自己的 m：把某指标的 m 抬高到远超其取值，该指标就应当沉默。"""
        reps, T = 4, 80
        streams = np.stack([np.full((reps, T), 0.8), np.full((reps, T), 0.8)], axis=2)
        lams = ed.lambda_grid(0.8, n_lam=3)
        # m=0.7 < X=0.8：比值 >1，e-value 持续累积 → 报警
        with_tight = ed.build_multi_detector(streams, [0.7, 0.7], lams, "mix")
        # m=0.9 > X=0.8：每个分量的 L 都 <1 ⇒ M≤1 ⇒ 阈值 5 永远够不到
        with_loose = ed.build_multi_detector(streams, [0.9, 0.9], lams, "mix")
        assert ed.first_alarm(with_tight, 5.0).min() > 0
        assert (ed.first_alarm(with_loose, 5.0) == 0).all()   # L≤1 ⇒ 永不报警

    def test_e1_edd_targets_only_one_indicator(self):
        """漂移只打在某个指标上时，只有盯它的检测器该报警（对照矩阵的对角线性质）。"""
        rng = np.random.default_rng(23)
        pool = np.tile(np.array([[0.2, 0.2, 0.2]]), (40, 1))
        pool = np.vstack([pool, np.tile(np.array([[0.25, 0.25, 0.25]]), (40, 1))])
        recs = ed.run_e1_edd(pool, [0.3, 0.3, 0.3], ed.lambda_grid(0.3, n_lam=3),
                             alpha=0.5, at=100, delta=0.5, directions=[+1, +1, +1],
                             reps=3, T_pre=100, T_post=100, rng=rng, targets=[0])
        by_fuse = {(r["fuse"], r["drift_target"]): r for r in recs}
        tgt = "仅指标#0"
        assert by_fuse[("单指标#0", tgt)]["detect_rate"] > 0
        assert by_fuse[("单指标#1", tgt)]["detect_rate"] == 0
        assert by_fuse[("单指标#2", tgt)]["detect_rate"] == 0
        assert by_fuse[("全指标混合", tgt)]["detect_rate"] > 0


class TestE5MScan:
    """E5：m 策略扫描（权衡曲线）的纯函数。"""

    def test_m_grid_sorted_and_named_strategies_included(self):
        calib = np.linspace(0.05, 0.45, 40)
        pts = ed.m_grid_from_calib(calib, ["mean", "max"], n_grid=5)
        assert pts and pts[0][1] <= pts[-1][1]                      # 按 m 升序
        labels = [p[0] for p in pts]
        assert "mean" in labels and "max" in labels
        assert any(lab.startswith("q") for lab in labels)
        assert len({round(m, 9) for _, m in pts}) == len(pts)        # 同值只留一个标签

    def test_m_grid_prefers_named_label_on_tie(self):
        calib = np.array([0.1, 0.2, 0.3, 0.4])                       # max == q100
        pts = dict((round(m, 9), lab) for lab, m in ed.m_grid_from_calib(calib, ["max"], 5))
        assert pts[round(0.4, 9)] == "max"

    def test_e5_curve_rows_detects_and_flags_validity(self):
        null = np.full((20, 50), 0.1)
        drift = np.full((20, 60), 0.1)
        drift[:, 10:] = 0.9                                          # 变点后明显变差
        rows = ed.e5_curve_rows(null, {0.5: drift},
                               [("mean", 0.2), ("max", 0.5)], ["mix"],
                               alpha=0.05, alpha_edd=0.1, at=10, horizon=20, pool_mean=0.1)
        assert len(rows) == 2 and all(r["experiment"] == "e5_curve" for r in rows)
        for r in rows:
            assert r["m_ge_pool_mean"] == 1
            assert r["arl_alarm_rate"] == 0.0        # 变前恒为 0.1 ≤ m → 永不报警
            assert r["detect_rate"] == 1.0
        small = next(r for r in rows if r["m_label"] == "mean")
        big = next(r for r in rows if r["m_label"] == "max")
        assert small["edd_mean"] <= big["edd_mean"]  # m 越保守，报警越慢

    def test_e5_curve_rows_marks_invalid_m(self):
        null = np.full((10, 40), 0.3)
        rows = ed.e5_curve_rows(null, {}, [("mean", 0.2)], ["mix"],
                                alpha=0.05, alpha_edd=0.1, at=10, pool_mean=0.3)
        assert rows[0]["m_ge_pool_mean"] == 0        # m < 变前均值 → 保证不适用

    def test_e5_mix_rows_reports_arl_and_per_target(self):
        null = np.full((20, 50, 2), 0.1)
        drift = np.full((20, 60, 2), 0.1)
        drift[:, 10:, 1] = 0.9                       # 只打第 2 个指标
        rows = ed.e5_mix_rows(null, [("仅指标#1", 0.5, drift)], [("mean", [0.2, 0.2])],
                              alpha=0.05, alpha_edd=0.1, at=10, horizon=20)
        assert len(rows) == 2
        arl = next(r for r in rows if r["drift_target"] == "（变前）")
        edd = next(r for r in rows if r["drift_target"] == "仅指标#1")
        assert arl["arl_alarm_rate"] == 0.0 and arl["ms"] == "0.200、0.200"
        assert edd["detect_rate"] == 1.0 and edd["fuse"] == "全指标混合"


class TestE2Attribution:
    """E2：漂移类型归因的纯函数。"""

    def test_component_series_shape_and_ordering(self):
        rng = np.random.default_rng(31)
        x = rng.random((4, 60, 3)) * 0.4
        x[:, 30:, 1] = 0.95                      # 第 2 个指标变差
        ms = [0.45, 0.45, 0.45]
        M = ed.component_series(x, ms, ed.lambda_grid(0.45, n_lam=3))
        assert M.shape == (4, 60, 3)
        # 报警时刻，被注入的那个指标的自身 e-value 应当最大
        assert (np.argmax(M[:, -1, :], axis=1) == 1).all()

    def test_attribute_rules_pick_drifted_indicator(self):
        rng = np.random.default_rng(32)
        x = rng.random((5, 80, 3)) * 0.3
        x[:, 40:, 2] = 0.9
        ms = [0.4, 0.4, 0.4]
        M = ed.component_series(x, ms, ed.lambda_grid(0.4, n_lam=3))
        t = np.full(5, 70)
        for rule in ed.ATTRIBUTION_RULES:
            pred = ed.attribute(rule, M, x, ms, t, window=30)
            assert (pred == 2).all(), rule

    def test_attribute_recent_ignores_stale_evidence(self):
        """累积量会被历史主导：早期的大波动之后即使真漂移在别的指标上，
        recent 规则（只看最近窗口）仍应指向真正在动的那个。"""
        rng = np.random.default_rng(33)
        x = rng.random((3, 120, 2)) * 0.2
        x[:, 10:30, 0] = 1.0                     # 早期：指标0 曾大幅波动
        x[:, 80:, 1] = 1.0                       # 后期：指标1 持续漂移
        ms = [0.5, 0.5]
        M = ed.component_series(x, ms, ed.lambda_grid(0.5, n_lam=3))
        t = np.full(3, 119)
        assert (ed.attribute("recent", M, x, ms, t, window=30) == 1).all()

    def test_run_e2_attribution_reports_accuracy(self):
        pool = np.full((40, 3), 0.2)
        ms = [0.25, 0.25, 0.25]
        rows = ed.run_e2_attribution(pool, ms, ed.lambda_grid(0.25, n_lam=3),
                                     alpha=0.05, at=30, deltas=[0.5], targets=[1],
                                     directions=[+1, +1, +1], reps=4, T_pre=30, T_post=60,
                                     rng=np.random.default_rng(34), horizon=30, window=10)
        assert len(rows) == len(ed.ATTRIBUTION_RULES)
        ev = next(r for r in rows if r["rule"] == "evalue")
        assert ev["drift_target"] == "仅指标#1" and ev["detect_rate"] > 0
        assert ev["acc"] == 1.0


class TestE3Weights:
    """E3：权重方案（均匀 / 先验 / 自适应）的纯函数。"""

    def test_normalize_weights_sums_to_one(self):
        w = ed.normalize_weights([2.0, 1.0, 1.0])
        assert abs(w.sum() - 1.0) < 1e-12
        assert abs(w[0] - 0.5) < 1e-12

    def test_normalize_weights_handles_degenerate_input(self):
        for bad in ([0.0, 0.0, 0.0], [-1.0, -2.0, -3.0]):
            w = ed.normalize_weights(bad)
            assert abs(w.sum() - 1.0) < 1e-12 and (w > 0).all()   # 退化为均匀，而不是 NaN

    def test_prior_from_values_keeps_floor(self):
        w = ed.prior_from_values([0.42, 0.29, 0.002], floor=0.25)
        assert abs(w.sum() - 1.0) < 1e-12
        assert w[2] >= 0.25 / 3 - 1e-12          # 弱指标不会被先验归零
        assert w[0] > w[1] > w[2]

    def test_weighted_mix_uniform_equals_component_average(self):
        rng = np.random.default_rng(41)
        x = rng.random((3, 50, 2)) * 0.5
        ms = [0.6, 0.6]
        lams = ed.lambda_grid(0.6, n_lam=3)
        M = ed.component_series(x, ms, lams)
        assert np.allclose(ed.weighted_mix(M, None), M.mean(axis=2))
        # 单指标权重 → 退化成那一个指标自己的序列
        assert np.allclose(ed.weighted_mix(M, [1.0, 0.0]), M[:, :, 0])

    def test_adaptive_mix_eta_zero_is_fixed_prior(self):
        rng = np.random.default_rng(42)
        M = rng.random((4, 30, 3)) + 0.5
        a = ed.adaptive_mix(M, [0.5, 0.3, 0.2], eta=0.0)
        assert np.allclose(a, ed.weighted_mix(M, [0.5, 0.3, 0.2]))

    def test_adaptive_mix_recovers_from_wrong_prior(self):
        """自适应应当把权重从错误的先验挪到真正在动的指标上。"""
        T = 60
        M = np.ones((3, T, 2))
        M[:, 20:, 1] = np.cumprod(np.full(T - 20, 1.6))     # 指标1 在 t=20 后持续增长
        wrong = [0.9, 0.1]
        fix = ed.weighted_mix(M, wrong)
        ad = ed.adaptive_mix(M, wrong, eta=1.0)
        c = 20.0
        t_fix = int(ed.first_alarm(fix, c)[0])
        t_ad = int(ed.first_alarm(ad, c)[0])
        assert t_ad > 0 and (t_fix == 0 or t_ad < t_fix)     # 自适应更早越过同一阈值

    def test_matched_threshold_hits_target_far(self):
        rng = np.random.default_rng(43)
        M = rng.random((800, 40))                            # 变前流
        thr = ed.matched_threshold(M, target=0.10)
        got = float((ed.first_alarm(M, thr) > 0).mean())
        assert abs(got - 0.10) < 0.05

    def test_fixed_weight_row_reports_its_actual_threshold(self):
        null = np.random.default_rng(44).uniform(0, 1, size=(20, 30, 2))
        rows = ed.run_e3_weights([("uniform", {"kind": "fixed", "w": None})],
                                 null, [], [0.5, 0.5], ed.lambda_grid(0.5),
                                 alpha=0.05, alpha_edd=0.01, at=10,
                                 match_far=0.1)
        fixed = next(r for r in rows if r["calib"] == "fixed")
        assert fixed["threshold"] == 100.0

    def test_load_proxy_prior_falls_back_to_uniform(self):
        w, src = ed.load_proxy_prior(["a", "b", "c"], path="no_such_file.csv")
        assert np.allclose(w, [1 / 3, 1 / 3, 1 / 3])
        assert "退化" in src or "未找到" in src


class TestE6Compare:
    """E6：与相关工作同口径对比的纯函数（两样本统计量、阈值校准）。"""

    def test_ks_2samp_zero_when_same_sample(self):
        a = np.tile(np.linspace(0, 1, 40), (3, 1))
        assert np.allclose(ed._ks_2samp(a, a), 0.0, atol=1e-12)

    def test_ks_2samp_one_when_disjoint(self):
        a = np.tile(np.linspace(0.0, 0.2, 30), (2, 1))
        b = np.tile(np.linspace(0.5, 0.7, 30), (2, 1))
        assert np.allclose(ed._ks_2samp(a, b), 1.0)

    def test_jsd_2samp_handles_different_lengths(self):
        rng = np.random.default_rng(51)
        win = rng.random((4, 30))
        ref = rng.random((4, 90))                    # 参考窗更长：早期版本在这里索引长度不匹配直接崩
        v = ed._jsd_2samp(win, ref, bins=8)
        assert v.shape == (4,) and np.isfinite(v).all() and (v >= 0).all()

    def test_frechet_1d_detects_mean_shift(self):
        rng = np.random.default_rng(52)
        ref = rng.normal(0.0, 0.1, size=(3, 50))
        same = rng.normal(0.0, 0.1, size=(3, 50))
        shifted = rng.normal(0.5, 0.1, size=(3, 50))
        assert (ed._frechet_1d(shifted, ref) > ed._frechet_1d(same, ref)).all()

    def test_calibrate_threshold_hits_target_rate(self):
        rng = np.random.default_rng(53)
        null = rng.random((600, 30))
        thr = ed.calibrate_threshold(null, 0.10)
        got = float((ed.first_alarm(null, thr) > 0).mean())
        assert abs(got - 0.10) < 0.05

    def test_cusum_k_parameter_is_read_from_name(self):
        """方法名里的 @k 要真的生效：k 越大越迟钝（同样漂移下统计量涨得慢）。"""
        rng = np.random.default_rng(54)
        x = np.concatenate([rng.normal(0.0, 0.1, size=(3, 60)),
                            rng.normal(0.6, 0.1, size=(3, 60))], axis=1)
        calib = rng.normal(0.0, 0.1, size=40)
        s_small = ed.method_score("cusum@0.1", x, calib)
        s_big = ed.method_score("cusum@1.0", x, calib)
        assert s_small[:, -1].mean() > s_big[:, -1].mean()

    def test_two_sample_reference_is_per_stream_head(self):
        """两样本类方法必须用各流自己的前 ref_len 个点当参考，而不是校准段。"""
        rng = np.random.default_rng(55)
        x = rng.normal(0.0, 0.1, size=(3, 300))
        calib = rng.normal(3.0, 1.0, size=40)         # 故意与流完全不同
        s = ed.method_score("ks_window", x, calib, window=40, stride=5, ref_len=100)
        # 用各流自己的头做参考 → 无漂移时统计量不应整段饱和
        assert s[:, 120:].max() < 0.95

    def test_reference_window_cannot_look_ahead(self):
        x = np.tile(np.linspace(0.0, 1.0, 200), (2, 1))
        s = ed.method_score("ks_window", x, x[0, :40],
                            window=40, stride=5, ref_len=100)
        assert np.isneginf(s[:, :100]).all()
        assert np.isfinite(s[:, 100:]).all()

    def test_false_alarm_is_measured_on_heldout_streams(self):
        train = np.vstack([np.zeros(30), np.ones(30)])
        heldout = np.zeros((2, 30))
        rows = ed.run_e6_compare(["cusum"], train, [], np.array([0.0, 1.0]),
                                 [0.5], ed.lambda_grid(0.5), at=10,
                                 target_far=0.5, null_eval_streams=heldout)
        assert rows[0]["alarm_rate"] == 0.0
        assert rows[0]["n_calibration_reps"] == 2


class TestE7Dependence:
    """E7：非 iid 稳健性（相依生成器、诊断量、相依感知上界）。"""

    def test_ar1_streams_preserve_marginal_distribution(self):
        pool = np.linspace(0.0, 1.0, 50)
        s = ed.ar1_streams(pool, reps=20, T=500, rho=0.8, rng=np.random.default_rng(61))
        assert set(np.unique(s)).issubset(set(np.unique(pool)))   # 分位数映射只取池内取值
        assert abs(s.mean() - pool.mean()) < 0.05

    def test_ar1_streams_match_requested_autocorrelation(self):
        pool = np.linspace(0.0, 1.0, 2000)                        # 近似连续 → 映射单调
        for rho in (0.0, 0.5, 0.9):
            s = ed.ar1_streams(pool, reps=50, T=800, rho=rho, rng=np.random.default_rng(62))
            got = ed.dependence_diagnostics(s, m=2.0, window=20)["acf1"]
            assert abs(got - rho) < 0.12, (rho, got)

    def test_dependence_diagnostics_flags_violation(self):
        x = np.full((5, 200), 0.2)
        assert ed.dependence_diagnostics(x, m=0.5, window=20)["local_exceed_rate"] == 0.0
        assert ed.dependence_diagnostics(x, m=0.1, window=20)["local_exceed_rate"] == 1.0

    def test_dependence_aware_m_bounds_local_means(self):
        rng = np.random.default_rng(63)
        calib = rng.random(200) * 0.5
        m_nom = ed.estimate_m(calib, "q85")
        m_aw = ed.dependence_aware_m(calib, window=20, margin=0.5)
        assert m_aw >= m_nom
        csum = np.cumsum(np.insert(calib, 0, 0.0))
        loc = (csum[20:] - csum[:-20]) / 20
        assert loc.max() <= m_aw + 1e-9                            # 上界要罩住滑窗均值

    def test_run_e7_returns_rows_for_both_generators(self):
        pool = np.linspace(0.1, 0.9, 40)
        rows = ed.run_e7_dependence([("block", "块长", 5.0), ("ar1", "ρ", 0.5)], pool,
                                    pool, m_base=0.6, m_aware=0.8,
                                    lams=ed.lambda_grid(0.6), drift_delta=0.3, reps=6,
                                    T=100, at=30, T_post=60, alphas=[0.05],
                                    rng=np.random.default_rng(64), window=10)
        assert len(rows) == 2 and all(r["experiment"] == "e7_dependence" for r in rows)
        assert all("alarm_rate_base_0.05" in r and "edd_aware" in r for r in rows)


class TestExperimentProtocol:
    def test_disjoint_windows_do_not_reuse_queries_or_cross_split(self):
        raw = np.concatenate([np.arange(15, dtype=float),
                              np.arange(100, 115, dtype=float)])
        data, cut = ed.prepare_proxy_split(raw, 0.5, 4,
                                           np.random.default_rng(0), shuffle=False,
                                           window_mode="disjoint")
        assert cut == 3
        assert data.shape == (6,)
        assert np.allclose(data[:cut], [1.5 / 14, 5.5 / 14, 9.5 / 14])
        assert np.all(data[cut:] == 1.0)

    def test_constant_calibration_preserves_new_events(self):
        got = ed.fit_unit(np.zeros(4), np.array([0.0, 1.0, -1.0]))
        assert np.array_equal(got, np.array([0.5, 1.0, 0.0]))

    def test_windows_do_not_cross_calibration_boundary(self):
        raw = np.concatenate([np.arange(15, dtype=float),
                              np.arange(100, 115, dtype=float)])
        data, cut = ed.prepare_proxy_split(raw, 0.5, 4,
                                           np.random.default_rng(0), shuffle=False)
        assert cut == 12
        assert data.shape == (24,)
        assert data[:cut].max() < 1.0
        assert np.all(data[cut:] == 1.0)

    def test_binomial_bound_rejects_continuous_proxy(self):
        calib = np.array([0.1, 0.2, 0.3, 0.4])
        with pytest.raises(ValueError, match="0/1"):
            ed.estimate_m(calib, "binom")
        assert all(label != "binom" for label, _ in
                   ed.m_grid_from_calib(calib, ["binom", "hoeffding"]))

    def test_binomial_all_successes_has_upper_bound_one(self):
        assert ed.estimate_m(np.ones(20), "binom") == 1.0

    def test_right_censored_mean_differs_from_alarm_only_mean(self):
        stats = ed.arl_stats(np.array([1, 0]), horizon=10)
        assert stats["arl_mean"] == 1.0
        assert stats["arl_restricted_mean"] == 6.0
        assert stats["censored"] == 1


class TestSimulation:
    def test_block_bootstrap_shape(self):
        rng = np.random.default_rng(5)
        pool = rng.random(100)
        s = ed.block_bootstrap(pool, reps=8, T=50, block=10, rng=rng)
        assert s.shape == (8, 50) and s.min() >= 0 and s.max() <= 1

    def test_arl_on_stationary_below_bound(self):
        """变前流全部远低于 m 时不应报警（ARL 无界）。"""
        rng = np.random.default_rng(6)
        streams = rng.random((40, 300)) * 0.3        # 均值 0.15，m=0.9
        r = ed.run_arl(streams, m=0.9, lams=ed.lambda_grid(0.9), how="mix", alpha=0.05)
        assert r["alarm_rate"] == 0.0 and math.isinf(r["arl_mean"])

    def test_arl_alarms_when_bound_violated(self):
        rng = np.random.default_rng(7)
        streams = np.full((40, 300), 0.8)             # 远高于 m=0.2
        r = ed.run_arl(streams, m=0.2, lams=ed.lambda_grid(0.2), how="mix", alpha=0.05)
        assert r["alarm_rate"] == 1.0 and r["arl_mean"] < 20

    def test_edd_detects_large_drift(self):
        rng = np.random.default_rng(8)
        pool = rng.random(200) * 0.4
        r = ed.run_edd(pool, m=0.5, lams=ed.lambda_grid(0.5), how="mix", alpha=0.01,
                       at=100, delta=0.5, direction=+1, reps=30, T_pre=100, T_post=200,
                       rng=rng)
        assert r["detect_rate"] > 0.5 and r["edd_mean"] < 200
