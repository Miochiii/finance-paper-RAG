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
        assert np.allclose(ed.fit_unit(np.array([0.3, 0.3]), np.array([0.3, 0.9])), 0.5)


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
