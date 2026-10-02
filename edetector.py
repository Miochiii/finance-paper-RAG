# -*- coding: utf-8 -*-
"""edetector.py —— RAG 质量漂移的序贯检测：最小可行实现（E-detector 框架）。

方法来源
--------
Shin, Ramdas, Rinaldo (2023) *E-detectors: A Nonparametric Framework for Sequential
Change Detection*。本项目只用其**有界变量构造**（原文 §5.2）与**混合封闭性**（命题 2.3）：

    L_n(λ) = 1 + λ( X_n/m − 1 ),   λ ∈ (0,1)          # 基线 e-detector 增量
    M^CU_n(λ) = L_n(λ) · max{ M^CU_{n−1}(λ), 1 }       # 累积型（CUSUM 式）
    M_n = Σ_{k,λ} ω_{k,λ} · M^CU_n(k,λ),  Σω = 1       # 多指标凸混合（命题 2.3）
    报警：M_n ≥ 1/α                                    # 阈值显式，无需校准

三个关键性质（也是本脚本要实验验证的）：
  1. **合法性**：只要变前 μ_n ≤ m，就有 E[L_n|F_{n−1}] ≤ 1，故 X_n ≤ m 时 L_n ≤ 1（不增长）；
     阈值取 1/α 时 ARL ≥ 1/α（定理 2.4，非渐近）。
  2. **混合仍合法**（命题 2.3）——所以"多指标取加权平均"是安全的融合方式。
  3. **取最大不合法**（Remark 3.1）——工程直觉最自然的 max/sup 会破坏保证，
     实测表现为误报膨胀（ARL 远小于 1/α）。这正是本脚本最锋利的一组结果。

用法（手动运行）
----------------
    python edetector.py                          # 默认：ARL + EDD + 融合对比（秒级~分钟级）
    python edetector.py --proxy ret_unique_docs,ans_refusal --alpha 0.05
    python edetector.py --m-strategy mean,p95,hoeffding      # m 估计策略对比（E5）
    python edetector.py --reps 2000 --T 2000                 # 更稳的 ARL 估计
    python edetector.py --no-inject                          # 只做 ARL 验证

输入：`results/proxy_table_hmm_*.csv`（E0 产出的逐查询代理矩阵，含 batch 列）。
输出：`results/edetector_<时间戳>.csv`（逐配置结果）、`results/edetector_series_<时间戳>.csv`
      （一条示例轨迹的 M_n 序列，可直接画图）、`log/E-detector_最小验证_<日期>.md`（报告）。
"""

import argparse
import csv
import glob
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_core.config import PROJECT_DIR  # noqa: E402

RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
LOG_DIR = os.path.join(PROJECT_DIR, "log")

# 代理的"退化方向"：+1 表示数值变大=变差（如检索文档多样性），-1 表示变小=变差（如相似度）
DEGRADE_DIRECTION = {
    "ret_unique_docs": +1,      # 证据越分散越差（E0/E1 实测：集中度 HHI 与之互为镜像）
    "ret_top1_sim": -1, "ret_mean_sim": -1, "ret_margin": -1,
    "ret_sim_entropy": +1, "ret_page_spread": +1, "ret_n_blocks": -1,
    "ans_refusal": +1,          # 拒答率上升=变差（E0 实测最稳健）
    "ans_length": -1, "ans_n_claims": -1, "ans_claim_density": -1,
    "ans_citations": -1, "ans_citation_density": -1,
    # B 步新增代理（退化方向 = 数值往哪边走表示变差）
    "ret_doc_hhi": -1,          # 证据集中度下降（越分散）= 变差
    "cons_doc_jaccard": -1, "cons_chunk_jaccard": -1, "cons_top1_agree": -1,
    "cons_overlap3": -1, "cons_rank_corr": -1, "cons_doc_jaccard_p2": -1,
    "q_evi_cos": +1, "ans_evi_cos": +1, "ans_centroid_dist": 0,
    "ret_sim_std": 0, "ret_sim_iqr": 0, "ret_sim_range": 0, "ret_sim_skew": 0,
}
FUSE_CHOICES = ("mix", "max", "min", "single")


# --------------------------------------------------------------------------
# 纯函数：构造 e-detector
# --------------------------------------------------------------------------
def to_unit(values: Sequence[float]) -> np.ndarray:
    """把代理归一化到 [0,1]（min-max；常数序列返回全 0.5）。越界值截断。"""
    a = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))],
                   dtype=float)
    if a.size == 0:
        return a
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        return np.full(a.shape, 0.5)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def estimate_m(calib: np.ndarray, strategy: str = "p95", delta: float = 0.05) -> float:
    """估计变前均值上界 m（合法性要求 μ_n ≤ m，而不是 m = 样本均值）。

    这是开题报告模块 2 的开放问题（E5：保守性与检测延迟的权衡）：
      - mean      : m = 样本均值 —— 最激进，μ ≈ m 时合法性踩线，ARL 可能不足；
      - p95       : m = 95% 分位数 —— 常用折中；
      - qNN       : m = NN% 分位数（如 q60/q85/q100）—— 用于**扫描 m**，画出 ARL/EDD 权衡曲线；
      - hoeffding : m = 均值 + sqrt(log(1/δ)/(2n)) —— 对独立有界样本的有限样本上界（概率 1−δ 成立），
                    但对**稀有事件太松**（p̂=0.02、n=58 时给到 0.18，约 9 倍均值）；
      - binom     : 仅对原始 0/1 独立样本计算 Clopper-Pearson 上界；
                    连续代理和滑窗均值不能先阈值化再把结果当作原变量均值上界；
      - max       : m = 观测最大值 —— 最保守，检测最慢（常常永远不报警）。
    返回至少比最大值略大的数，避免 m=0 或 m=1 导致 L 退化。
    """
    a = np.asarray(calib, dtype=float)
    if a.size == 0:
        return 1.0
    if strategy == "mean":
        m = float(a.mean())
    elif strategy == "p95":
        m = float(np.quantile(a, 0.95))
    elif strategy.startswith("q") and strategy[1:].isdigit():
        m = float(np.quantile(a, min(100, max(1, int(strategy[1:])))/ 100.0))
    elif strategy == "hoeffding":
        m = float(a.mean()) + math.sqrt(math.log(1.0 / max(delta, 1e-6)) / (2.0 * a.size))
    elif strategy == "binom":
        from scipy import stats
        if not np.all((np.abs(a) < 1e-12) | (np.abs(a - 1.0) < 1e-12)):
            raise ValueError("binom 仅适用于 0/1 代理；连续值或滑窗均值请用其他 m 策略")
        n = int(a.size)
        k = int(np.sum(a >= 0.5))
        m = 1.0 if k == n else float(stats.beta.ppf(1.0 - delta, k + 1, n - k))
    elif strategy == "max":
        m = float(a.max())
    else:
        raise ValueError(f"未知 m 策略: {strategy}")
    return float(min(1.0 + 1e-9, max(m, 1e-6)))


def lambda_grid(m: float, n_lam: int = 7, lo: float = 0.05,
                hi: float = 0.95) -> np.ndarray:
    """λ 网格（等比，越小越敏感于小幅漂移）。

    论文用 (Δ_L, Δ_U) 与 K_max 反推网格（Algorithm 1 / 附录 B.1）；
    最小可行版先用等比网格，并在报告里记录 Δ_L、Δ_U 供后续对齐：
        Δ_L = mδ/(1−m)²,  Δ_U = m(1−m)/δ²
    """
    lo = max(1e-3, min(lo, hi))
    return np.geomspace(lo, min(hi, 0.999), n_lam)


def delta_bounds(m: float, delta: float = 0.05) -> Tuple[float, float]:
    """论文 §5.2 式 75 的 Δ_L、Δ_U（只用于记录/讲解，最小版不据此筛网格）。"""
    m = min(max(m, 1e-6), 1 - 1e-6)
    return (m * delta / (1 - m) ** 2, m * (1 - m) / delta ** 2)


def cumulative_evalue_series(x: np.ndarray, m: float, lam: float,
                             cap: float = 1e12) -> np.ndarray:
    """单分量累积 e-detector：M^CU_n = L_n · max(M^CU_{n−1}, 1)，M_0 = 1。

    向量化在"多条独立流"维度上做（x 形状 (T,) 或 (R,T)），递归沿时间轴。

    注意：M 可以被 L ≤ 1 拉到 1 以下（例如恒为 0.55）——这不是 bug，而是"自重启"语义，
    阈值 1/α > 1，所以低于 1 的取值不会触发报警；关键性质是
    **当 X_n ≤ m 恒成立时 M_n ≤ 1，永不可能报警**（这正是 ARL ≥ 1/α 的直观来源）。
    M 会被截断在 cap（默认 1e12）——阈值只有 1/α（通常 20 左右），
    截断远高于阈值，不影响报警判断，但能避免 m 极小时乘积溢出成 inf。
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    R, T = x.shape
    m = float(max(m, 1e-9))
    out = np.empty((R, T), dtype=float)
    prev = np.ones(R, dtype=float)
    for t in range(T):
        L = 1.0 + lam * (x[:, t] / m - 1.0)
        prev = np.clip(L * np.maximum(prev, 1.0), 0.0, cap)
        out[:, t] = prev
    return out


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """滑动窗口均值：稀有/二值代理（如拒答标志）先做窗口聚合再进检测器。

    这既是工程上的自然做法（生产里监控的是"每 N 条查询里的拒答率"），
    也让 m 有正的下界（否则全零校准窗口会给出 m=0，e-detector 退化）。
    """
    x = np.asarray(x, dtype=float)
    if window <= 1 or x.size < window:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[window:] - c[:-window]) / float(window)


def disjoint_mean(x: np.ndarray, window: int) -> np.ndarray:
    """每 window 条查询只产出一个监控值，不重复使用同一条查询。"""
    a = np.asarray(x, dtype=float)
    if window <= 1:
        return a
    n = a.size // window
    if n == 0:
        raise ValueError("原始查询数不足一个不重叠窗口")
    return a[:n * window].reshape(n, window).mean(axis=1)


def prepare_proxy_split(raw: np.ndarray, calib_frac: float, window: int,
                        rng: np.random.Generator, shuffle: bool = True,
                        invert: bool = False, binarize: float = 0.0,
                        window_mode: str = "overlap") -> Tuple[np.ndarray, int]:
    """先切原始查询，再在各段内做滑窗；返回拼接序列和校准段长度。

    校准段的任何窗口都不能含检验段查询。归一化尺度仍只由原始校准段确定。
    """
    x = np.asarray(raw, dtype=float)
    if x.ndim not in (1, 2) or x.shape[0] < 2:
        raise ValueError("需要至少两条一维或二维查询记录")
    if shuffle:
        x = x[rng.permutation(x.shape[0])]
    cut = max(8, int(x.shape[0] * calib_frac))
    if cut >= x.shape[0]:
        raise ValueError("校准段必须短于完整序列")
    win = max(1, int(window))
    if window_mode not in ("overlap", "disjoint"):
        raise ValueError("window_mode 必须为 overlap 或 disjoint")
    if win > min(cut, x.shape[0] - cut):
        raise ValueError("滑窗长度不能超过校准段或检验段的原始查询数")
    one_dim = x.ndim == 1
    if one_dim:
        x = x[:, None]
    calib, heldout = x[:cut], x[cut:]
    norm_calib = np.column_stack([fit_unit(calib[:, k], calib[:, k])
                                  for k in range(x.shape[1])])
    norm_heldout = np.column_stack([fit_unit(calib[:, k], heldout[:, k])
                                    for k in range(x.shape[1])])
    if invert:
        norm_calib = 1.0 - norm_calib
        norm_heldout = 1.0 - norm_heldout
    if win > 1:
        aggregate = rolling_mean if window_mode == "overlap" else disjoint_mean
        norm_calib = np.column_stack([aggregate(norm_calib[:, k], win)
                                      for k in range(x.shape[1])])
        norm_heldout = np.column_stack([aggregate(norm_heldout[:, k], win)
                                        for k in range(x.shape[1])])
    if binarize > 0:
        norm_calib = (norm_calib >= binarize).astype(float)
        norm_heldout = (norm_heldout >= binarize).astype(float)
    result = np.concatenate([norm_calib, norm_heldout], axis=0)
    return (result[:, 0] if one_dim else result), norm_calib.shape[0]


def fuse_series(series_list: List[np.ndarray], how: str = "mix",
                weights: Optional[Sequence[float]] = None) -> np.ndarray:
    """多分量融合。合法性（命题 2.3）只对 mix/min 成立；max 用于对照实验（Remark 3.1）。"""
    stack = np.stack(series_list, axis=0)          # (K, R, T)
    if how == "max":
        return stack.max(axis=0)
    if how == "min":
        return stack.min(axis=0)
    if how == "single":
        return stack[0]
    w = np.asarray(weights if weights is not None else [1.0 / stack.shape[0]] * stack.shape[0],
                   dtype=float)
    w = w / w.sum()                                # 权重必须和为 1，否则不再合法
    return np.tensordot(w, stack, axes=(0, 0))


def first_alarm(series: np.ndarray, threshold: float) -> np.ndarray:
    """每条流首次越过阈值的时间（1-based）；未报警返回 0。"""
    s = np.atleast_2d(series)
    hit = s >= threshold
    idx = np.argmax(hit, axis=1) + 1
    idx[~hit.any(axis=1)] = 0
    return idx


def arl_stats(times: np.ndarray, horizon: int) -> Dict:
    """固定观察长度下的截尾摘要；不能把已报警流均值称为完整 ARL。"""
    t = np.asarray(times, dtype=int)
    observed = t[t > 0]
    return {
        "alarm_rate": float((t > 0).mean()),
        "arl_mean": float(observed.mean()) if observed.size else float("inf"),
        "arl_median": float(np.median(observed)) if observed.size else float("inf"),
        "arl_restricted_mean": float(np.where(t > 0, t, horizon + 1).mean()),
        "censored": int((t == 0).sum()),
    }


def inject_drift(stream: np.ndarray, at: int, delta: float, direction: int,
                 mode: str = "contaminate",
                 rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """在 at 步之后注入退化漂移（Δ 的含义都是"期望均值漂移量"）。

    - `shift`：加性平移后 clip 到 [0,1]。简单，但当序列贴着 0/1（如离散计数的归一化值）
      会饱和，实际漂移量小于 Δ；
    - `contaminate`（默认）：让一部分查询变成"最差状态"（direction>0 取 1.0，否则取 0.0），
      比例 q 由 q·|target − 当前均值| = Δ 反解 → 期望均值漂移恰为 Δ，且不会饱和。
      这更贴近真实退化（"一部分查询开始答不好"），也便于向评委解释。
    """
    out = np.array(stream, dtype=float, copy=True)
    if at <= 0 or at >= out.shape[-1] or delta <= 0:
        return out
    if mode == "shift":
        out[..., at:] = np.clip(out[..., at:] + direction * delta, 0.0, 1.0)
        return out
    rng = rng or np.random.default_rng(0)
    post = out[..., at:]
    target = 1.0 if direction > 0 else 0.0
    cur = post.mean(axis=-1, keepdims=True)
    q = np.clip(delta / np.maximum(1e-6, np.abs(target - cur)), 0.0, 1.0)
    mask = rng.random(post.shape) < q
    out[..., at:] = np.where(mask, target, post)
    return out


def block_bootstrap(pool: np.ndarray, reps: int, T: int, block: int,
                    rng: np.random.Generator) -> np.ndarray:
    """块自助抽样生成 R×T 的变前流：保留短程自相关（比 iid 重采样更接近真实日志）。"""
    pool = np.asarray(pool, dtype=float)
    n = pool.size
    if n == 0:
        raise ValueError("空数据池")
    if block <= 1:
        return rng.choice(pool, size=(reps, T), replace=True)
    nb = math.ceil(T / block)
    starts = rng.integers(0, max(1, n - block + 1), size=(reps, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(reps, nb * block)
    return pool[idx % n][:, :T]


def bootstrap_rows(pool: np.ndarray, reps: int, T: int, block: int,
                   rng: np.random.Generator) -> np.ndarray:
    """**联合**块自助：对"行"重采样一次，多个指标取同一批行。

    多指标混合必须保持指标之间的相关结构（同一查询上的拒答、集中度、一致性是相关的），
    所以不能各自独立重采样——那样会人为削弱相关性、把混合的效果估得过于乐观。
    返回 (reps, T, K) 的联合流矩阵。
    """
    pool = np.asarray(pool, dtype=float)
    n = pool.shape[0]
    if n == 0:
        raise ValueError("空数据池")
    idx = block_bootstrap(np.arange(n, dtype=float), reps, T, block, rng).astype(int)
    return pool[idx]


def build_multi_detector(x_multi: np.ndarray, ms: Sequence[float], lams: np.ndarray,
                         how: str = "mix", single_k: int = -1) -> np.ndarray:
    """多指标 e-detector：(R,T,K) 联合流 → (R,T) 的 M 序列。

    每个指标 k 用自己的上界 m_k 构造分量（合法性要求逐指标满足 μ_k ≤ m_k）：
        L_n(k,λ) = 1 + λ(X_n(k)/m_k − 1)，M^CU_n(k,λ) = L_n(k,λ)·max(M^CU_{n−1}(k,λ), 1)
    融合：
        mix  = 对全部 (k,λ) 均匀加权平均 —— **命题 2.3 保证仍然合法**（E1 的主角）
        max  = 逐点取最大 —— 不合法（Remark 3.1 对照）
        min  = 逐点取最小 —— 合法但保守
        one  = 只用单个指标（single_k 指定）在其 λ 网格上混合 —— E1 的基线
    """
    x_multi = np.asarray(x_multi, dtype=float)
    if x_multi.ndim != 3:
        raise ValueError("x_multi 应为 (R,T,K) 形状")
    K = x_multi.shape[2]
    comps = []
    for k in range(K):
        if how == "one" and single_k >= 0 and k != single_k:
            continue
        for lam in lams:
            comps.append(cumulative_evalue_series(x_multi[:, :, k], float(ms[k]), float(lam)))
    if not comps:
        raise ValueError("没有可用的检测器分量")
    stack = np.stack(comps, axis=0)                      # (C,R,T)
    if how == "max":
        return stack.max(axis=0)
    if how == "min":
        return stack.min(axis=0)
    w = np.full(stack.shape[0], 1.0 / stack.shape[0])    # 权重和为 1，才保持合法
    return np.tensordot(w, stack, axes=(0, 0))


# --------------------------------------------------------------------------
# 实验
# --------------------------------------------------------------------------
def fit_unit(calib: np.ndarray, x: np.ndarray) -> np.ndarray:
    """用**校准窗口**的极值把序列映射到 [0,1]（避免偷看变点之后的数据）。

    整条序列做 min-max 会把变点后的漂移值也纳入尺度（前视偏差），
    而且会把 m 推到接近 1、检测器再也不报警——实测踩过这个坑。
    这里只用校准段的 lo/hi；漂移导致越界的值截断到 0/1（正是我们想检的信号）。
    """
    c = np.asarray(calib, dtype=float)
    x = np.asarray(x, dtype=float)
    if c.size == 0:
        return np.clip(x, 0.0, 1.0)
    lo, hi = float(c.min()), float(c.max())
    if hi - lo < 1e-12:
        # 校准值恒定时保留检验段越界的方向，不能把新出现的事件抹成常数。
        return np.where(x > hi + 1e-12, 1.0,
                        np.where(x < lo - 1e-12, 0.0, 0.5))
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def build_detectors(x: np.ndarray, m: float, lams: np.ndarray,
                    how: str) -> np.ndarray:
    """按融合方式给出 M 序列。

    x 可以是单条流 (T,) 或一批独立流 (R,T)；返回同形。**注意保留 R 维**
    （早期版本误取 [0]，把 30 条流压成 1 条，统计量看着正常其实只算了一条流）。
    """
    series = [cumulative_evalue_series(x, m, float(lam)) for lam in lams]
    return fuse_series(series, how=how)


def run_arl(x_streams: np.ndarray, m: float, lams: np.ndarray, how: str,
            alpha: float) -> Dict:
    """变前流摘要；arl_restricted_mean 估计 E[min(报警时间,T+1)]。"""
    M = build_detectors(x_streams, m, lams, how)
    times = first_alarm(M, 1.0 / alpha)
    return {
        "n_reps": int(x_streams.shape[0]),
        "T": int(x_streams.shape[1]),
        **arl_stats(times, int(x_streams.shape[1])),
    }


def run_e1_arl(streams: np.ndarray, ms: Sequence[float], lams: np.ndarray,
               alpha: float) -> List[Dict]:
    """E1：单指标 vs 多指标混合 的变前 ARL（同一批联合流，配对比较）。

    返回每个配置一行的列表（配置 = 单独用第 k 个指标 / 全部指标混合 / 取最大 / 取最小）。
    """
    K = streams.shape[2]
    out = []
    for k in range(K):
        M = build_multi_detector(streams, ms, lams, "one", single_k=k)
        times = first_alarm(M, 1.0 / alpha)
        out.append({"fuse": f"单指标#{k}", "k": k,
                    **arl_stats(times, int(streams.shape[1])),
                    "n_comp": int(len(lams))})
    for how, label in (("mix", "全指标混合"), ("max", "全指标取最大"), ("min", "全指标取最小")):
        M = build_multi_detector(streams, ms, lams, how)
        times = first_alarm(M, 1.0 / alpha)
        out.append({"fuse": label, "k": -1,
                    **arl_stats(times, int(streams.shape[1])),
                    "n_comp": int(len(lams) * (K if how != "one" else 1))})
    return out


def run_e1_edd(pool: np.ndarray, ms: Sequence[float], lams: np.ndarray, alpha: float,
               at: int, delta: float, directions: Sequence[int], reps: int,
               T_pre: int, T_post: int, rng: np.random.Generator,
               mode: str = "contaminate", targets: Optional[Sequence[int]] = None,
               horizon: int = 0) -> List[Dict]:
    """E1：单指标 vs 多指标混合 的 EDD（同一批联合流 + 同一次注入，配对比较）。

    `targets` 决定"漂移打在哪些指标上"：
      - None / 全部：所有指标按各自退化方向同时变差（理想情况，各检测器都占便宜）；
      - 只给一个 k：**只有第 k 个指标退化**——这是多指标融合的真正用武之地：
        针对别的指标调好的单指标检测器会完全失效，而混合仍能检出（漂移类型未知时的鲁棒性）。
    每个指标按自己的退化方向注入（拒答/分散度上升=变差，集中度/一致性下降=变差）。
    """
    K = pool.shape[1]
    tgts: List[Optional[int]] = list(range(K)) if targets is None else list(targets)
    pre_idx = block_bootstrap(np.arange(pool.shape[0], dtype=float), reps, T_pre, 20, rng).astype(int)
    post_idx = block_bootstrap(np.arange(pool.shape[0], dtype=float), reps, T_post, 20, rng).astype(int)
    pre = pool[pre_idx]                     # (R,T_pre,K)
    post = pool[post_idx]
    base = np.concatenate([pre, post], axis=1)
    out: List[Dict] = []
    for tgt in tgts:
        stream = np.array(base, copy=True)
        hits = range(K) if tgt is None else [tgt]
        for k in hits:
            # 注意：必须把**完整序列**与真实的变点位置传进去；
            # 早期版本传的是已经切好的 stream[:, at:, k] 且 at=0，而 at<=0 是不注入的早退分支，
            # 结果漂移根本没打进去（所有检出率恒为 0）。
            stream[:, :, k] = inject_drift(stream[:, :, k], at=at, delta=delta,
                                           direction=int(directions[k]), mode=mode, rng=rng)
        label = "全部指标" if tgt is None else f"仅指标#{tgt}"
        for k in range(K):
            M = build_multi_detector(stream, ms, lams, "one", single_k=k)
            out.append({"fuse": f"单指标#{k}", "k": k, "drift_target": label,
                        **_edd_stats(M, alpha, at, T_post, horizon)})
        for how, name in (("mix", "全指标混合"), ("max", "全指标取最大"), ("min", "全指标取最小")):
            M = build_multi_detector(stream, ms, lams, how)
            out.append({"fuse": name, "k": -1, "drift_target": label,
                        **_edd_stats(M, alpha, at, T_post, horizon)})
    for r in out:
        r["delta"] = delta
        r["inject_mode"] = mode
    return out


def is_inf(v) -> bool:
    """判断一个统计量是否为「未检出 / 无定义」。

    记录里的 inf 既可能是 `float('inf')`，也可能是字符串 `'inf'`（CSV 往返或显式转换），
    两种都要认——早期版本只判字符串，于是 `float('inf')` 的行被当成有效 EDD 参与比较，
    把「漂移打偏、完全没检出」的单指标和「打中了」的单指标混在一起取最小值，
    得出了混合比单指标慢 0.4× 这种错误结论。
    """
    if isinstance(v, str):
        return v.strip().lower() in ("", "inf", "infinity", "nan", "none")
    try:
        return not math.isfinite(float(v))
    except (TypeError, ValueError):
        return True


def _num(v, fmt: str = "{:.2f}") -> str:
    """报告里用的安全数值格式化：缺失/无定义一律显示「—」。"""
    if v is None or v == "" or is_inf(v):
        return "—"
    try:
        return fmt.format(float(v))
    except (TypeError, ValueError):
        return "—"


def _edd_stats(M: np.ndarray, alpha: float, at: int, T_post: int,
               horizon: int = 0, threshold: Optional[float] = None) -> Dict:
    """从 M 序列算 EDD 相关统计（变前误报、检出率、EDD 均值/中位/截尾）。

    `threshold` 不给时用 1/α；同误报水平的对比要用**各方案自己的阈值**（见 matched_threshold）。
    `horizon`（>0 时）= 只把「变点后 H 步内报警」算作检出。这很关键：不设视界时，
    零假设侧偶发的误报（例如某个检测器对该漂移本来就无感，只是碰巧在 500 步后响了一次）
    会被记成「检出」，于是出现「检出率 0.01、EDD 493 步」这种假象，还会把对照矩阵的
    对角线弄脏。超出视界的报警单列 `late_alarm_rate`，不隐瞒、也不算检出。
    """
    thr = float(threshold) if threshold is not None else 1.0 / alpha
    times = first_alarm(M, thr)
    pre_alarm = float(((times > 0) & (times <= at)).mean())
    after = times > at
    if horizon > 0:
        detected = after & (times <= at + horizon)
    else:
        detected = after
    late = after & ~detected
    delay = times[detected] - at
    cap = float(horizon if horizon > 0 else T_post)
    return {
        "pre_alarm_rate": pre_alarm,
        "detect_rate": float(detected.mean()),
        "late_alarm_rate": float(late.mean()),
        "edd_horizon": int(horizon),
        "edd_mean": float(delay.mean()) if delay.size else float("inf"),
        "edd_median": float(np.median(delay)) if delay.size else float("inf"),
        "edd_censored": float(np.where(detected, times - at, cap).mean()),
    }


def run_edd(pool: np.ndarray, m: float, lams: np.ndarray, how: str, alpha: float,
            at: int, delta: float, direction: int, reps: int, T_pre: int,
            T_post: int, rng: np.random.Generator, mode: str = "contaminate",
            horizon: int = 0) -> Dict:
    """注入漂移后的检测延迟（EDD）。

    统计口径全部交给 `_edd_stats`（曾经这里复制了一份，结果加 `--edd-horizon` 时漏改，
    单代理与 E1 两条路径口径不一致——直接复用以保证同源）：
      - 变点**之前**就报警 = 变前误报（单列 pre_alarm_rate，不算检出）；
      - 变点后 H 步内报警 = 检出；更晚 = 超时报警（late_alarm_rate）；
      - 一直没报警 = 截尾（计入 edd_censored）。
    """
    pre = block_bootstrap(pool, reps, T_pre, 20, rng)
    post = block_bootstrap(pool, reps, T_post, 20, rng)
    stream = inject_drift(np.concatenate([pre, post], axis=1), at, delta, direction,
                          mode=mode, rng=rng)
    M = build_detectors(stream, m, lams, how)
    return {"delta": delta, "direction": direction, "inject_mode": mode,
            **_edd_stats(M, alpha, at, T_post, horizon)}


# --------------------------------------------------------------------------
# E5：m 策略扫描 —— 保守性与检测延迟的权衡曲线
# --------------------------------------------------------------------------
NAMED_M_STRATEGIES = ("mean", "p95", "hoeffding", "binom", "max")


def m_grid_from_calib(calib: np.ndarray, extra_strategies: Sequence[str] = (),
                      n_grid: int = 13) -> List[Tuple[str, float]]:
    """要扫描的 (标签, m) 列表：分位数主网格 + 具名策略标记点，按 m 升序去重。

    曲线需要的是**连续**的 m，而不是几个具名策略；主网格用校准段的分位数 q50…q100
    （τ 越大越保守），再把工程上常用的具名估计（mean/p95/hoeffding/binom/max）
    作为曲线上的标记点，这样能直接读出「平时习惯用的那个估计落在曲线哪里」。
    同一取值只保留一个标签，优先保留具名策略。
    """
    a = np.asarray(calib, dtype=float)
    if a.size == 0:
        return []
    pts: List[Tuple[int, str, float]] = []
    for t in np.linspace(0.5, 1.0, max(2, int(n_grid))):
        pts.append((1, f"q{int(round(t * 100))}", float(np.quantile(a, float(t)))))
    for s in extra_strategies:
        s = (s or "").strip()
        if not s:
            continue
        if s == "binom" and not np.all((np.abs(a) < 1e-12) |
                                       (np.abs(a - 1.0) < 1e-12)):
            continue
        pts.append((0, s, float(estimate_m(a, s))))
    out: List[Tuple[str, float]] = []
    seen = set()
    for _prio, lab, m in sorted(pts, key=lambda x: (x[2], x[0], x[1])):
        key = round(m, 9)
        if key in seen:
            continue
        seen.add(key)
        out.append((lab, m))
    return out


def e5_curve_rows(null_streams: np.ndarray, drift_streams: Dict[float, np.ndarray],
                  points: Sequence[Tuple[str, float]], fuses: Sequence[str],
                  alpha: float, alpha_edd: float, at: int, horizon: int = 0,
                  m_floor: float = 1e-3, pool_mean: Optional[float] = None) -> List[Dict]:
    """单代理的 m 扫描：每个 m 点给出（变前报警率, 检出率, EDD）。

    **变前流与漂移流在 m 之间共用**（只重算检测器、不重新重采样），
    因此曲线上的点是配对的，点与点之间的差异来自 m 而不是抽样噪声。
    """
    rows: List[Dict] = []
    null_streams = np.asarray(null_streams, dtype=float)
    T = int(null_streams.shape[1])
    for lab, m_raw in points:
        m = float(max(m_raw, m_floor))
        lams = lambda_grid(m)
        arl_cache: Dict[str, Dict] = {}
        for how in fuses:
            M = build_detectors(null_streams, m, lams, how)
            times = first_alarm(M, 1.0 / alpha)
            ok = times > 0
            # 同时按**实际运行的那条报警线**（α_edd，E1 的检测实验用它）算一次误报率：
            # 选 m 应该按实际使用的阈值判断，只看 α=0.05 会把可选区间压得过窄。
            times_op = first_alarm(M, 1.0 / alpha_edd)
            arl_cache[how] = {
                "arl_alarm_rate": float(ok.mean()),
                "arl_alarm_rate_op": float((times_op > 0).mean()),
                "arl_mean": float(times[ok].mean()) if ok.any() else float("inf"),
                "arl_median": float(np.median(times[ok])) if ok.any() else float("inf"),
            }
        for d, stream in (drift_streams.items() if drift_streams else [(None, None)]):
            if stream is None:                      # 只扫 ARL（例如 --no-inject）
                for how in fuses:
                    rows.append({
                        "experiment": "e5_curve", "m_label": lab, "m": round(m, 4),
                        "pool_mean": (round(float(pool_mean), 4) if pool_mean is not None else ""),
                        "m_ge_pool_mean": (int(bool(pool_mean is not None and m >= pool_mean))
                                           if pool_mean is not None else ""),
                        "delta": "", "fuse": how, "alpha": alpha, "alpha_edd": alpha_edd,
                        "threshold": round(1.0 / alpha, 2), "K": int(len(lams)), "T": T,
                        "detect_rate": "", "late_alarm_rate": "", "edd_mean": "inf",
                        "edd_median": "inf", "edd_censored": "", "pre_alarm_rate": "",
                        **arl_cache[how]})
                continue
            T_post = int(stream.shape[1]) - at
            for how in fuses:
                M = build_detectors(np.asarray(stream, dtype=float), m, lams, how)
                rows.append({
                    "experiment": "e5_curve", "m_label": lab, "m": round(m, 4),
                    "pool_mean": (round(float(pool_mean), 4) if pool_mean is not None else ""),
                    "m_ge_pool_mean": (int(bool(pool_mean is not None and m >= pool_mean))
                                       if pool_mean is not None else ""),
                    "delta": d, "fuse": how, "alpha": alpha, "alpha_edd": alpha_edd,
                    "threshold": round(1.0 / alpha, 2),
                    "K": int(len(lams)), "T": T,
                    **arl_cache[how], **_edd_stats(M, alpha_edd, at, T_post, horizon),
                })
    return rows


def e5_mix_rows(null_multi: np.ndarray, drift_multi: Sequence[Tuple[str, float, np.ndarray]],
                ms_by_strategy: Sequence[Tuple[str, Sequence[float]]], alpha: float,
                alpha_edd: float, at: int, horizon: int = 0) -> List[Dict]:
    """混合检测器的 m 扫描：同一策略下把所有指标的 m 一起换掉。

    `drift_multi` 是 (漂移目标标签, Δ, 联合流) 的列表，流在各策略间共用（配对比较）。
    只跑「全指标混合」这一种融合——E5 关心的是 m，不是融合方式（后者是 E2 的事）。
    """
    rows: List[Dict] = []
    null_multi = np.asarray(null_multi, dtype=float)
    for strat, ms in ms_by_strategy:
        ms = [float(x) for x in ms]
        lams = lambda_grid(max(ms))
        M = build_multi_detector(null_multi, ms, lams, "mix")
        times = first_alarm(M, 1.0 / alpha)
        ok = times > 0
        rows.append({"experiment": "e5_mix", "m_strategy": strat, "m_label": strat,
                     "ms": "、".join(f"{x:.3f}" for x in ms),
                     "m": round(float(max(ms)), 4), "fuse": "全指标混合",
                     "drift_target": "（变前）", "delta": "", "alpha": alpha,
                     "threshold": round(1.0 / alpha, 2),
                     "arl_alarm_rate": float(ok.mean()),
                     "arl_mean": float(times[ok].mean()) if ok.any() else float("inf"),
                     "arl_median": float(np.median(times[ok])) if ok.any() else float("inf")})
        for label, d, stream in drift_multi:
            stream = np.asarray(stream, dtype=float)
            T_post = int(stream.shape[1]) - at
            M = build_multi_detector(stream, ms, lams, "mix")
            rows.append({"experiment": "e5_mix", "m_strategy": strat, "m_label": strat,
                         "ms": "、".join(f"{x:.3f}" for x in ms),
                         "m": round(float(max(ms)), 4), "fuse": "全指标混合",
                         "drift_target": label, "delta": d, "alpha_edd": alpha_edd,
                         "threshold": round(1.0 / alpha_edd, 2), "K": int(len(lams)),
                         **_edd_stats(M, alpha_edd, at, T_post, horizon)})
    return rows


# --------------------------------------------------------------------------
# E2：漂移类型归因（G4）——报警之后，判断是哪个指标在退化
# --------------------------------------------------------------------------
ATTRIBUTION_RULES = ("evalue", "recent", "zscore", "shift")


def component_series(x_multi: np.ndarray, ms: Sequence[float],
                     lams: np.ndarray) -> np.ndarray:
    """每个指标**自身**的凸混合 e-value 序列，形状 (R, T, K)。

    混合统计量 M_n = Σ_k ω_k M_k(n) 是按指标可分解的，这给了归因一个天然的抓手：
    报警时刻哪个指标的 M_k 更大，就更像是它在退化。注意 ω_k 中同一指标的多个 λ 分量
    权重相同，所以这里对每个指标在其 λ 网格上取平均。
    """
    x_multi = np.asarray(x_multi, dtype=float)
    if x_multi.ndim != 3:
        raise ValueError("x_multi 应为 (R,T,K) 形状")
    R, T, K = x_multi.shape
    out = np.empty((R, T, K), dtype=float)
    for k in range(K):
        comps = [cumulative_evalue_series(x_multi[:, :, k], float(ms[k]), float(lam))
                 for lam in lams]
        out[:, :, k] = np.mean(np.stack(comps, axis=0), axis=0)
    return out


def attribute(rule: str, M_comp: np.ndarray, x_multi: np.ndarray, ms: Sequence[float],
              t_eval: np.ndarray, window: int = 50,
              base_mean: Optional[Sequence[float]] = None,
              base_sd: Optional[Sequence[float]] = None) -> np.ndarray:
    """按规则给出每个流预测的漂移指标下标（在 t_eval 时刻判断）。

    - evalue：各指标自身 e-value 最大者（**累积**证据最多）；
    - recent：最近 window 步的**对数增量**最大者（新近证据）——
      累积量会被历史主导，长跑之后早期的小波动也可能压过刚发生的漂移；
    - zscore：最近 window 步均值相对**变前均值/标准差**的 z 值最大者
      （工程上最标准的控制图做法，方差归一化，是三条基线里最强的一条）；
    - shift ：最近 window 步均值相对上界 m_k 的原始偏移最大者（最朴素的做法）。
    """
    M_comp = np.asarray(M_comp, dtype=float)
    x_multi = np.asarray(x_multi, dtype=float)
    R, T, K = M_comp.shape
    t = np.clip(np.asarray(t_eval, dtype=int), 1, T)
    idx = np.arange(R)
    w = max(1, min(int(window), T - 1))
    t0 = np.maximum(0, t - 1 - w)
    if rule == "evalue":
        score = M_comp[idx, t - 1, :]
    elif rule == "recent":
        score = np.log(np.maximum(M_comp[idx, t - 1, :], 1e-12)) - \
            np.log(np.maximum(M_comp[idx, t0, :], 1e-12))
    elif rule in ("zscore", "shift"):
        bm = (np.asarray(base_mean, dtype=float) if base_mean is not None
              else np.asarray(ms, dtype=float))
        bs = (np.asarray(base_sd, dtype=float) if base_sd is not None else None)
        score = np.zeros((R, K), dtype=float)
        for k in range(K):
            recent = np.array([x_multi[i, t0[i]:t[i], k].mean() for i in range(R)])
            if rule == "zscore":
                denom = max(float(bs[k]) if bs is not None else 1.0, 1e-6)
            else:
                denom = max(float(ms[k]), 1e-9)
            score[:, k] = (recent - float(bm[k])) / denom
    else:
        raise ValueError(f"未知归因规则: {rule}")
    return np.argmax(score, axis=1)


def run_e2_attribution(pool: np.ndarray, ms: Sequence[float], lams: np.ndarray,
                       alpha: float, at: int, deltas: Sequence[float],
                       targets: Optional[Sequence[int]], directions: Sequence[int],
                       reps: int, T_pre: int, T_post: int, rng: np.random.Generator,
                       mode: str = "contaminate", horizon: int = 300,
                       window: int = 50, calib: Optional[np.ndarray] = None,
                       delays: Sequence[int] = (0,)) -> List[Dict]:
    """E2：漂移只打在某个指标上，看报警后能否把它指出来。

    每一行 =（漂移目标, Δ, 归因规则, 归因延迟）的准确率；只统计**在视界内报警**的流
    （没报警就无从归因，这部分单列报警率）。`delays` 是「报警后再等多少步才判断」——
    等得久证据更多，但如果算法只能在事后才准，实用性就打折。
    """
    K = int(pool.shape[1])
    tgts = list(range(K)) if targets is None else list(targets)
    pre_idx = block_bootstrap(np.arange(pool.shape[0], dtype=float), reps, T_pre, 20, rng).astype(int)
    post_idx = block_bootstrap(np.arange(pool.shape[0], dtype=float), reps, T_post, 20, rng).astype(int)
    base = np.concatenate([pool[pre_idx], pool[post_idx]], axis=1)     # (R, at+T_post, K)
    c = np.asarray(calib, dtype=float) if calib is not None else np.asarray(pool, dtype=float)
    base_mean = c.mean(axis=0) if c.size else np.asarray(ms, dtype=float)
    base_sd = c.std(axis=0, ddof=0) if c.size else np.ones(len(ms))
    out: List[Dict] = []
    for tgt in tgts:
        for d in deltas:
            stream = np.array(base, copy=True)
            stream[:, :, tgt] = inject_drift(stream[:, :, tgt], at=at, delta=d,
                                             direction=int(directions[tgt]), mode=mode,
                                             rng=rng)
            M = build_multi_detector(stream, ms, lams, "mix")
            times = first_alarm(M, 1.0 / alpha)
            pre_alarm_rate = float(((times > 0) & (times <= at)).mean())
            late = times > at + horizon
            times = np.where(late, 0, times)          # 超出视界＝没检出
            det = times > at
            M_comp = component_series(stream, ms, lams)
            T = stream.shape[1]
            for delay in delays:
                t_eval = np.clip(times + int(delay), 1, T)
                for rule in ATTRIBUTION_RULES:
                    if det.any():
                        pred = attribute(rule, M_comp[det], stream[det], ms, t_eval[det],
                                         window, base_mean, base_sd)
                        acc = float((pred == tgt).mean())
                        preds = pred.tolist()
                    else:
                        acc, preds = float("nan"), []
                    out.append({
                        "experiment": "e2_attr", "drift_target": f"仅指标#{tgt}",
                        "target_k": tgt, "delta": d, "rule": rule, "delay": int(delay),
                        "n_reps": int(reps), "n_alarmed": int(det.sum()),
                        "pre_alarm_rate": pre_alarm_rate,
                        "detect_rate": float(det.mean()),
                        "late_rate": float(late.mean()),
                        "acc": acc,
                        "acc_other": float(np.mean([p != tgt for p in preds])) if preds else float("nan"),
                        "window": int(window), "horizon": int(horizon), "alpha": alpha,
                    })
    return out


# --------------------------------------------------------------------------
# E3：权重方案（均匀 / 先验 / 自适应）
# --------------------------------------------------------------------------
def normalize_weights(w: Sequence[float]) -> np.ndarray:
    """权重归一（和为 1 是混合保持合法的前提）；全 0 或含负数时退化为均匀。"""
    a = np.asarray(w, dtype=float)
    a = np.clip(a, 0.0, None)
    s = float(a.sum())
    if not np.isfinite(s) or s <= 0:
        return np.full(a.shape, 1.0 / max(1, a.size))
    return a / s


def prior_from_values(values: Sequence[float], floor: float = 0.25) -> np.ndarray:
    """把「先验重要性」变成权重：按 |值| 归一后，**向均匀权重收缩** floor 比例。

    `values` 通常是 E0 里各代理与真值的相关性强度。收缩（0=纯先验、1=纯均匀）
    是为了两方面都守住：
      - 避免先验把某个指标直接归零——一旦漂移正好打在它身上，混合就彻底失效
        （E1 的对照矩阵已经说明单指标失效有多彻底）；
      - 让「先验有多可信」变成一个显式可调的量，而不是藏在归一化的细节里。
    凸组合之后权重和仍然恰好为 1（不需要再归一化），且每个指标至少保留 floor/n 的份额。
    """
    a = np.abs(np.asarray(values, dtype=float))
    n = max(1, a.size)
    frac = float(min(max(floor, 0.0), 1.0))
    if not np.isfinite(a).all() or a.sum() <= 0:
        return np.full(n, 1.0 / n)
    w_raw = a / a.sum()
    return normalize_weights((1.0 - frac) * w_raw + frac * np.full(n, 1.0 / n))


def weighted_mix(M_comp: np.ndarray, w: Optional[Sequence[float]] = None) -> np.ndarray:
    """按固定权重混合各指标自身的 e-value 序列：(R,T,K) → (R,T)。

    w=None 即均匀权重；此时结果与 `build_multi_detector(..., "mix")` 完全一致
    （后者是在 (k,λ) 分量上均匀，等价于「先在 λ 内平均、再在 k 上平均」）。
    """
    M = np.asarray(M_comp, dtype=float)
    k = M.shape[2]
    ww = normalize_weights(w) if w is not None else np.full(k, 1.0 / k)
    return np.tensordot(ww, M, axes=(0, 2))


def adaptive_mix(M_comp: np.ndarray, w_prior: Optional[Sequence[float]] = None,
                 eta: float = 0.5) -> np.ndarray:
    """自适应加权混合：(R,T,K) → (R,T)。

    权重按**上一时刻的证据**逐点调整：ω_k(n) ∝ w_k^prior · max(M_k(n−1), 1)^η，
    再归一到和为 1。两个性质要守住：
      - **可预测**（只用到 n−1 时刻的信息）——不能偷看本步的 e-value；
      - **任意时刻权重和为 1**——这是混合保持合法的前提（原文 §3 的自适应调度同此要求）。
    η 越大越激进（迅速倒向当前证据最多的指标），η → 0 退化为固定先验权重。
    理论上：固定权重混合的合法性由命题 2.3 保证；可预测时变权重在实践中的误报控制
    由 ARL 侧实测检验（本实验一并报告，见 E3 的误报率列）。
    """
    M = np.asarray(M_comp, dtype=float)
    R, T, K = M.shape
    if K == 1:
        return M[:, :, 0].copy()
    wp = normalize_weights(w_prior) if w_prior is not None else np.full(K, 1.0 / K)
    eta = float(max(0.0, eta))
    out = np.empty((R, T), dtype=float)
    prev = np.maximum(M[:, 0, :], 1.0)                 # 用 n−1 时刻的 e-value 定权重
    wp_b = np.broadcast_to(wp, (R, K))                 # eta=0 时也要保持 (R,K) 形状
    for t in range(T):
        w = wp_b * np.power(prev, eta) if eta > 0 else wp_b
        w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-300)
        out[:, t] = (w * M[:, t, :]).sum(axis=1)
        prev = np.maximum(M[:, t, :], 1.0)
    return out


def matched_threshold(M_null: np.ndarray, target: float = 0.01) -> float:
    """给出「变前误报率 ≈ target」的阈值：取变前流**运行最大值**的 (1−target) 分位数。

    自适应加权、取最大这类方案在同一个 1/α 下误报天然更高，直接比 EDD 是不公平的
    （等于拿不同误报水平的两条曲线比）。要公平就必须**在同一误报水平上比 EDD**，
    这个函数就是给每个方案各自定一条报警线。

    注意：这里用的是同一批变前流，实际部署应把这一步放在**独立校准样本**上。
    """
    M = np.asarray(M_null, dtype=float)
    if M.size == 0:
        return float("inf")
    runmax = np.maximum.accumulate(M, axis=1)[:, -1]
    t = float(np.quantile(runmax, 1.0 - float(target)))
    return max(float(np.min(runmax)), t)      # 至少要比最小值高，避免阈值低于所有流


def run_e3_weights(schemes: Sequence[Tuple[str, Dict]], null_multi: np.ndarray,
                   drift_list: Sequence[Tuple[str, float, np.ndarray]], ms: Sequence[float],
                   lams: np.ndarray, alpha: float, alpha_edd: float, at: int,
                   horizon: int = 0, match_far: float = 0.0,
                   null_eval_multi: Optional[np.ndarray] = None) -> List[Dict]:
    """E3：在同一批流上比较各权重方案（变前误报 + 各漂移目标的检出/延迟）。

    `schemes` 里每项是 (方案名, 说明字典)，字典形如
    {"kind": "fixed"|"adaptive", "w": 权重或 None, "eta": 自适应强度}。
    各指标自身的 e-value 序列（component_series）**只算一次**，供所有方案复用。

    `match_far` > 0 时，额外产出一组 `calib="matched"` 的行：每个方案用
    「把变前误报压到 match_far」的自己那条阈值（见 matched_threshold），
    从而在**同一误报水平**上比较 EDD——否则自适应方案会凭更高的误报率"看起来更快"。
    """
    rows: List[Dict] = []
    null_multi = np.asarray(null_multi, dtype=float)

    def _w_text(w, k):
        return "、".join(f"{x:.3f}" for x in (normalize_weights(w) if w is not None
                                             else np.full(k, 1.0 / k)))

    M_null = component_series(null_multi, ms, lams) if null_multi.size else None
    M_eval = (component_series(np.asarray(null_eval_multi, dtype=float), ms, lams)
              if null_eval_multi is not None else M_null)
    drift_cache = []
    for label, d, stream in drift_list:
        stream = np.asarray(stream, dtype=float)
        drift_cache.append((label, d, component_series(stream, ms, lams),
                            int(stream.shape[1]) - at))
    for name, spec in schemes:
        kind = spec.get("kind", "fixed")
        w = spec.get("w")
        eta = float(spec.get("eta", 0.0))
        k_dim = int(null_multi.shape[2]) if null_multi.ndim == 3 else len(ms)

        def _mix(Mcomp):
            return (adaptive_mix(Mcomp, w, eta) if kind == "adaptive"
                    else weighted_mix(Mcomp, w))

        thr_fixed = 1.0 / alpha_edd
        thr_matched = None
        if M_null is not None:
            M0_calib = _mix(M_null)
            M0 = _mix(M_eval)
            times = first_alarm(M0, 1.0 / alpha)
            times_op = first_alarm(M0, thr_fixed)
            if match_far > 0:
                thr_matched = matched_threshold(M0_calib, match_far)
            ok = times > 0
            rows.append({
                "experiment": "e3_weight", "scheme": name, "kind": kind, "calib": "fixed",
                "eta": eta, "weights": _w_text(w, k_dim),
                "drift_target": "（变前）", "delta": "",
                "arl_alarm_rate": float(ok.mean()),
                "arl_alarm_rate_op": float((times_op > 0).mean()),
                "arl_mean": float(times[ok].mean()) if ok.any() else float("inf"),
                "arl_median": float(np.median(times[ok])) if ok.any() else float("inf"),
                "alpha": alpha, "alpha_edd": alpha_edd,
                "threshold": round(thr_fixed, 2),
            })
            if match_far > 0 and thr_matched:
                # 匹配阈值下**实际达到**的误报率：同误报水平的对比必须能核验，
                # 否则读者无法确认"两条曲线是在同一个误报水平上比的"。
                times_m = first_alarm(M0, thr_matched)
                rows.append({
                    "experiment": "e3_weight", "scheme": name, "kind": kind, "calib": "matched",
                    "eta": eta, "weights": _w_text(w, k_dim),
                    "drift_target": "（变前）", "delta": "",
                    "arl_alarm_rate": float(ok.mean()),
                    "arl_alarm_rate_op": float((times_m > 0).mean()),
                    "arl_mean": float(times_m[times_m > 0].mean()) if (times_m > 0).any()
                    else float("inf"),
                    "arl_median": float(np.median(times_m[times_m > 0]))
                    if (times_m > 0).any() else float("inf"),
                    "alpha": alpha, "alpha_edd": alpha_edd,
                    "threshold": round(thr_matched, 2), "match_target": match_far,
                })
        for label, d, M_comp, T_post in drift_cache:
            M = _mix(M_comp)
            rows.append({
                "experiment": "e3_weight", "scheme": name, "kind": kind, "calib": "fixed",
                "eta": eta, "weights": _w_text(w, len(ms)),
                "drift_target": label, "delta": d,
                "alpha": alpha, "alpha_edd": alpha_edd, "threshold": round(thr_fixed, 2),
                **_edd_stats(M, alpha_edd, at, T_post, horizon),
            })
            if match_far > 0 and thr_matched:
                rows.append({
                    "experiment": "e3_weight", "scheme": name, "kind": kind, "calib": "matched",
                    "eta": eta, "weights": _w_text(w, len(ms)),
                    "drift_target": label, "delta": d,
                    "alpha": alpha, "alpha_edd": alpha_edd,
                    "threshold": round(thr_matched, 2),
                    **_edd_stats(M, alpha_edd, at, T_post, horizon, threshold=thr_matched),
                })
    return rows


def load_proxy_prior(proxies: Sequence[str], path: Optional[str] = None,
                     floor: float = 0.25) -> Tuple[np.ndarray, str]:
    """从 E0 的有效性结果里取「先验重要性」：各代理与真值的最大 |批次内 ρ|。

    返回 (归一化权重, 数据源文件名)。找不到文件或缺列时退化为均匀权重——
    先验可以来自实证，但**不能因为拿不到就跑不动**。
    """
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "proxy_validity_*.csv")), reverse=True)
    if path:
        files = [path] + [f for f in files if f != path]
    n = max(1, len(proxies))
    for f in files:
        try:
            with open(f, encoding="utf-8-sig", newline="") as fh:
                rows = list(csv.DictReader(fh))
        except (OSError, UnicodeDecodeError):
            continue
        if not rows or "rho_within" not in rows[0]:
            continue
        strength = {}
        for p in proxies:
            vals = []
            for r in rows:
                if r.get("proxy") != p:
                    continue
                try:
                    vals.append(abs(float(r["rho_within"])))
                except (TypeError, ValueError):
                    continue
            if vals:
                strength[p] = max(vals)
        if len(strength) == len(proxies):
            return prior_from_values([strength[p] for p in proxies], floor), os.path.basename(f)
    return np.full(n, 1.0 / n), "（未找到 E0 结果，退化为均匀权重）"


# --------------------------------------------------------------------------
# E6：与相关工作同口径对比（同一输入序列 + 同一误报水平）
# --------------------------------------------------------------------------
E6_METHODS = ("edetector", "cusum", "page_hinkley", "ks_window", "jsd_window",
              "frechet_window", "mmd_window", "devatwal_manual")

E6_METHOD_CN = {
    "edetector": "本文：e-detector（凸混合 + 显式阈值）",
    "cusum": "经典 CUSUM（标准化后 k=0.5）",
    "page_hinkley": "Page–Hinkley",
    "ks_window": "两样本 KS（Feldhans 式，分箱近似）",
    "jsd_window": "JSD（Gupta 式分布距离）",
    "frechet_window": "Fréchet 距离（Greco/DriftLens 式）",
    "mmd_window": "MMD（Li 式核两样本）",
    "devatwal_manual": "两窗口比对 + 人工阈值（Devatwal 式）",
}


def _window_hist(x: np.ndarray, window: int, edges: np.ndarray) -> np.ndarray:
    """滑窗直方图：(R,T) 序列 → (R,T,B) 的窗口计数（右端对齐当前时刻）。

    用累积和实现，避免逐窗循环——两样本类方法（KS/JSD/Fréchet/MMD）都要用它。
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    R, T = x.shape
    B = len(edges) - 1
    idx = np.clip(np.digitize(x, edges) - 1, 0, B - 1)          # (R,T)
    onehot = np.zeros((R, T, B), dtype=float)
    np.put_along_axis(onehot, idx[:, :, None], 1.0, axis=2)
    csum = np.cumsum(onehot, axis=1)
    out = np.empty((R, T, B), dtype=float)
    w = min(int(window), T)
    out[:, :w, :] = csum[:, :w, :]
    out[:, w:, :] = csum[:, w:, :] - csum[:, :-w, :]
    return out


def _window_moments(x: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """滑窗均值与标准差（Fréchet 距离用）：(R,T) → 两个 (R,T)。"""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    R, T = x.shape
    w = min(int(window), T)
    c1 = np.cumsum(np.concatenate([np.zeros((R, 1)), x], axis=1), axis=1)
    c2 = np.cumsum(np.concatenate([np.zeros((R, 1)), x * x], axis=1), axis=1)
    s1 = np.empty((R, T))
    s2 = np.empty((R, T))
    s1[:, :w] = c1[:, 1:w + 1]
    s2[:, :w] = c2[:, 1:w + 1]
    if T > w:
        s1[:, w:] = c1[:, w + 1:] - c1[:, 1:T - w + 1]
        s2[:, w:] = c2[:, w + 1:] - c2[:, 1:T - w + 1]
    n = np.minimum(np.arange(1, T + 1), w).astype(float)
    mean = s1 / n
    var = np.maximum(s2 / n - mean ** 2, 0.0)
    return mean, np.sqrt(var)


def _ks_2samp(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """两样本 KS 统计量（按流并行）：(R,Wa) vs (R,Wb) → (R,)。

    用「合并排序 + 累计计数」实现（O(n log n)），不用 (R,Wa,Wb) 的广播
    ——后者在 Wa=50、Wb=120、R=200 时要 1.2M 个布尔量 × 每次评估，整轮 E6 会跑到十几分钟。

    两个坑：① CDF 方向必须是 `mean(样本 ≤ 点)`（早先把操作数写反得到的是生存函数）；
    ② **并列值只在「取值变化处」取值**——经验 CDF 是阶梯函数，在同一取值的中间位置
    两个 CDF 的计数不同步，直接对排序位置取最大会把「同一个样本与自己比」也算出 1/W 的假差异
    （单元测试就是这样抓到的）。
    """
    Wa, Wb = a.shape[1], b.shape[1]
    both = np.concatenate([a, b], axis=1)
    lab = np.concatenate([np.zeros_like(a), np.ones_like(b)], axis=1)
    order = np.argsort(both, axis=1, kind="stable")
    sv = np.take_along_axis(both, order, axis=1)
    lab_s = np.take_along_axis(lab, order, axis=1)
    ca = np.cumsum(lab_s == 0.0, axis=1) / Wa
    cb = np.cumsum(lab_s == 1.0, axis=1) / Wb
    diff = np.abs(ca - cb)
    is_end = np.ones(sv.shape, dtype=bool)
    is_end[:, :-1] = sv[:, 1:] != sv[:, :-1]        # 每个并列取值组的最后一个位置
    return np.where(is_end, diff, -1.0).max(axis=1)


def _jsd_2samp(a: np.ndarray, b: np.ndarray, bins: int = 10) -> np.ndarray:
    """两样本 JSD（按流并行，分箱边界取自该流两窗的联合范围，避免饱和）。

    两个窗的长度可以不同（当前窗 W vs 参考窗 ref_len），所以计数要各按自己的列数归一
    ——早期版本统一用 a 的列数，参考窗更长时 `np.add.at` 的索引长度对不上直接崩。
    """
    R, Wa = a.shape
    Wb = b.shape[1]
    lo = np.minimum(a.min(axis=1), b.min(axis=1))
    hi = np.maximum(a.max(axis=1), b.max(axis=1))
    span = np.maximum(hi - lo, 1e-9)
    ia = np.clip(((a - lo[:, None]) / span[:, None] * bins).astype(int), 0, bins - 1)
    ib = np.clip(((b - lo[:, None]) / span[:, None] * bins).astype(int), 0, bins - 1)
    pa = np.zeros((R, bins))
    pb = np.zeros((R, bins))
    np.add.at(pa, (np.repeat(np.arange(R), Wa), ia.ravel()), 1.0)
    np.add.at(pb, (np.repeat(np.arange(R), Wb), ib.ravel()), 1.0)
    pa /= Wa
    pb /= Wb
    m = 0.5 * (pa + pb)
    eps = 1e-12
    jsd = 0.5 * (pa * np.log((pa + eps) / (m + eps))).sum(axis=1) + \
        0.5 * (pb * np.log((pb + eps) / (m + eps))).sum(axis=1)
    return np.sqrt(np.maximum(jsd, 0.0))


def _frechet_1d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """一维 Fréchet 距离：sqrt((Δμ)² + (Δσ)²)（Greco/DriftLens 用的分布距离族）。"""
    return np.sqrt((a.mean(axis=1) - b.mean(axis=1)) ** 2 +
                   (a.std(axis=1) - b.std(axis=1)) ** 2)


def _mmd_2samp(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """MMD²（RBF 核，带宽取两窗合并样本的中位数启发式），按流并行。"""
    both = np.concatenate([a, b], axis=1)
    d2 = (both[:, :, None] - both[:, None, :]) ** 2
    pos = d2[d2 > 0]
    bw = float(np.median(pos)) if pos.size else 1.0
    bw = max(bw, 1e-6)
    kaa = np.exp(-((a[:, :, None] - a[:, None, :]) ** 2) / bw).mean(axis=(1, 2))
    kbb = np.exp(-((b[:, :, None] - b[:, None, :]) ** 2) / bw).mean(axis=(1, 2))
    kab = np.exp(-((a[:, :, None] - b[:, None, :]) ** 2) / bw).mean(axis=(1, 2))
    return np.maximum(kaa - 2 * kab + kbb, 0.0)


def method_score(name: str, stream: np.ndarray, calib: np.ndarray,
                 m: Optional[float] = None, lams: Optional[np.ndarray] = None,
                 window: int = 50, stride: int = 5, bins: int = 10,
                 manual_threshold: float = 2.0, ref_len: int = 0) -> np.ndarray:
    """按方法名给出**打分序列**（越大越像发生了漂移），形状 (R,T)。

    统一口径：所有方法看**同一条输入序列**（被打漂移那个指标的归一化序列），
    报警规则统一为「首次 ≥ 阈值」，阈值由 `calibrate_threshold` 在同一批变前流上
    按目标误报率校准（`devatwal_manual` 例外：它用人工阈值，正是要被展示的那件事）。

    两样本类方法（KS / JSD / Fréchet / MMD）的参考分布取**每条流自己的前 W 个点**
    （即文献里的 baseline window），而不是校准段：校准段只有几十个点、方差常与整体差很多，
    拿它当参考会让统计量整体饱和（实测 KS 在所有流上都顶到 0.94，阈值校准直接失效）。
    方法名可以带参数，如 `cusum@0.25` 表示假设漂移幅度为 0.25σ。
    """
    x = np.atleast_2d(np.asarray(stream, dtype=float))
    R, T = x.shape
    c = np.asarray(calib, dtype=float).ravel()
    mu0 = float(c.mean()) if c.size else float(x.mean())
    sd0 = float(c.std(ddof=0)) if c.size else float(x.std() or 1.0)
    sd0 = sd0 if sd0 > 1e-9 else 1.0
    base = name.split("@")[0]

    if base == "edetector":
        mm = float(m) if m else max(float(np.quantile(c, 0.85)) if c.size else 0.5, 1e-3)
        ll = np.asarray(lams) if lams is not None else lambda_grid(mm)
        return build_detectors(x, mm, ll, "mix")

    if base == "cusum":
        # 经典 CUSUM：标准化后扣除参考偏移 k（@ 后的数字，默认 0.5σ），单侧上升
        k = float(name.split("@")[1]) if "@" in name else 0.5
        z = (x - mu0) / sd0
        out = np.empty((R, T))
        s = np.zeros(R)
        for t in range(T):
            s = np.maximum(0.0, s + (z[:, t] - k))
            out[:, t] = s
        return out

    if base == "page_hinkley":
        k = float(name.split("@")[1]) if "@" in name else 0.25
        z = (x - mu0) / sd0
        run = np.zeros(R)
        best = np.zeros(R)
        out = np.empty((R, T))
        for t in range(T):
            run = run + (z[:, t] - k)
            best = np.minimum(best, run)
            out[:, t] = run - best
        return out

    if base == "devatwal_manual":
        # Devatwal 式：当前窗均值 vs 基线均值，阈值**人工设定**（默认 2σ，不校准）
        mean, _ = _window_moments(x, window)
        return np.abs((mean - mu0) / sd0)

    # 两样本 / 分布距离族：参考窗 = 每条流自己的**前 ref_len 个点**（默认同窗长）
    w = min(int(window), max(1, T // 4))
    rl = int(ref_len) if ref_len and ref_len > w else w
    rl = min(rl, T)
    ref = x[:, :rl]
    # 参考窗未收齐前不能计算在线分数，否则 t<rl 时会偷看未来观测。
    step = max(1, int(stride))
    start = math.ceil(max(w, rl) / step) * step
    out = np.full((R, T), -np.inf)
    for t in range(start, T, step):
        win = x[:, t - w:t]
        if base == "ks_window":
            out[:, t] = _ks_2samp(win, ref)
        elif base == "jsd_window":
            out[:, t] = _jsd_2samp(win, ref, bins)
        elif base == "frechet_window":
            out[:, t] = _frechet_1d(win, ref)
        elif base == "mmd_window":
            # MMD 是 O(W²) 的核计算，为控制整轮耗时把两窗都截到 32 个点（近似）
            out[:, t] = _mmd_2samp(win[:, -min(w, 32):], ref[:, :min(rl, 64)])
        else:
            raise ValueError(f"未知对比方法: {name}")
    return _stride_fill(out, stride)


def _stride_fill(score: np.ndarray, stride: int) -> np.ndarray:
    """把「每隔 stride 步才更新」的稀疏统计量补齐（阶梯保持），便于统一报警判定。"""
    s = np.asarray(score, dtype=float)
    if stride <= 1:
        return s
    out = np.empty_like(s)
    last = np.full(s.shape[0], -np.inf)
    for t in range(s.shape[1]):
        if t % stride == 0:
            last = s[:, t]
        out[:, t] = last
    return out


def calibrate_threshold(score_null: np.ndarray, target: float = 0.01) -> float:
    """按变前流把阈值校到「运行最大值超过它的比例 ≈ target」。"""
    S = np.asarray(score_null, dtype=float)
    if S.size == 0:
        return float("inf")
    runmax = np.maximum.accumulate(S, axis=1)[:, -1]
    return float(np.quantile(runmax, 1.0 - float(target)))


def run_e6_compare(methods: Sequence[str], null_streams: np.ndarray,
                   drift_list: Sequence[Tuple[str, float, np.ndarray]],
                   calib: np.ndarray, ms: Sequence[float], lams: np.ndarray,
                   at: int, horizon: int = 300, target_far: float = 0.01,
                   window: int = 50, stride: int = 5,
                   manual_threshold: float = 2.0, ref_len: int = 0,
                   null_eval_streams: Optional[np.ndarray] = None) -> List[Dict]:
    """E6：各方法在**同一输入序列、同一误报水平**下的检出率与 EDD。

    每个方法先用变前流校准自己的阈值（把误报率对齐到 target_far），再在漂移流上看 EDD；
    `devatwal_manual` 用人工阈值，不校准——它的实际误报率会单独报出来。
    `drift_list` 里每项是 (漂移目标, Δ, 单指标流)，流是**单个指标**的序列，
    也就是说所有方法看到的输入完全相同（多指标方法在这里按单指标运行）。
    """
    rows: List[Dict] = []
    null_streams = np.asarray(null_streams, dtype=float)
    null_eval = (np.asarray(null_eval_streams, dtype=float)
                 if null_eval_streams is not None else null_streams)
    for name in methods:
        m_k = float(max(ms[0], 1e-3)) if ms else None
        s_null = method_score(name, null_streams, calib, m=m_k, lams=lams,
                              window=window, stride=stride, ref_len=ref_len)
        if name == "devatwal_manual":
            thr = float(manual_threshold)
            thr_kind = "人工设定"
        else:
            thr = calibrate_threshold(s_null, target_far)
            thr_kind = f"校准到误报 {target_far:.0%}"
        s_eval = method_score(name, null_eval, calib, m=m_k, lams=lams,
                              window=window, stride=stride, ref_len=ref_len)
        t_null = first_alarm(s_eval, thr)
        far = float((t_null > 0).mean())
        rows.append({
            "experiment": "e6_compare", "method": name, "method_cn": E6_METHOD_CN.get(name, name),
            "drift_target": "（变前）", "delta": "", "threshold": round(thr, 4),
            "threshold_kind": thr_kind, "target_far": target_far,
            "alarm_rate": far, "n_reps": int(null_eval.shape[0]),
            "n_calibration_reps": int(null_streams.shape[0]),
            "window": int(window), "stride": int(stride),
        })
        for label, d, stream in drift_list:
            st = np.asarray(stream, dtype=float)
            T_post = int(st.shape[1]) - at
            s = method_score(name, st, calib, m=m_k, lams=lams,
                             window=window, stride=stride, ref_len=ref_len)
            times = first_alarm(s, thr)
            late = times > at + horizon
            times = np.where(late, 0, times)
            det = times > at
            delay = times[det] - at
            rows.append({
                "experiment": "e6_compare", "method": name,
                "method_cn": E6_METHOD_CN.get(name, name),
                "drift_target": label, "delta": d, "threshold": round(thr, 4),
                "threshold_kind": thr_kind, "target_far": target_far,
                "alarm_rate": far, "detect_rate": float(det.mean()),
                "late_rate": float(late.mean()),
                "edd_mean": float(delay.mean()) if delay.size else float("inf"),
                "edd_median": float(np.median(delay)) if delay.size else float("inf"),
                "edd_censored": float(np.where(det, times - at, horizon).mean()),
                "n_reps": int(st.shape[0]), "window": int(window), "stride": int(stride),
            })
    return rows


def run_e6_cusum_sweep(null_streams: np.ndarray, drift_list: Sequence[Tuple[str, float, np.ndarray]],
                       calib: np.ndarray, assumed_ks: Sequence[float], m: float,
                       lams: np.ndarray, at: int, horizon: int = 300,
                       target_far: float = 0.01,
                       null_eval_streams: Optional[np.ndarray] = None) -> List[Dict]:
    """E6 的关键对照：CUSUM 的**参考偏移 k** 选得对不对，决定了它有多快。

    经典 CUSUM 要知道「变后均值大概移动多少」（参考偏移 k，最优取真实位移的一半），
    而本文的 e-detector 通过 λ 网格混合**不需要这个先验**。这条扫描把两者的差别量化：
    同一批流、同一误报水平，横轴是"假设的漂移幅度"，每一格是实际的 Δ。
    每个 Δ 上同时给出 e-detector 的结果作为参照。
    """
    rows: List[Dict] = []
    null_streams = np.asarray(null_streams, dtype=float)
    null_eval = (np.asarray(null_eval_streams, dtype=float)
                 if null_eval_streams is not None else null_streams)

    def _eval(method_label: str, assumed_k: float, s_null: np.ndarray,
              thr: float, far: float) -> None:
        for label, d, stream in drift_list:
            st = np.asarray(stream, dtype=float)
            s = method_score(method_label, st, calib, m=m, lams=lams)
            times = first_alarm(s, thr)
            late = times > at + horizon
            times = np.where(late, 0, times)
            det = times > at
            delay = times[det] - at
            rows.append({
                "experiment": "e6_cusum_k", "method": method_label,
                "assumed_k": (float("nan") if method_label == "edetector" else float(assumed_k)),
                "drift_target": label, "delta": d, "alarm_rate": far,
                "threshold": round(thr, 4), "detect_rate": float(det.mean()),
                "edd_mean": float(delay.mean()) if delay.size else float("inf"),
                "edd_median": float(np.median(delay)) if delay.size else float("inf"),
                "edd_censored": float(np.where(det, times - at, horizon).mean()),
                "target_far": target_far,
            })

    for k in assumed_ks:
        lab = f"cusum@{float(k):g}"
        s_null = method_score(lab, null_streams, calib, m=m, lams=lams)
        thr = calibrate_threshold(s_null, target_far)
        s_eval = method_score(lab, null_eval, calib, m=m, lams=lams)
        _eval(lab, float(k), s_null, thr, float((first_alarm(s_eval, thr) > 0).mean()))
    # 本文方法的参照：它没有「假设漂移多大」这个旋钮，只按 λ 网格混合
    s_null_e = method_score("edetector", null_streams, calib, m=m, lams=lams)
    thr_e = calibrate_threshold(s_null_e, target_far)
    s_eval_e = method_score("edetector", null_eval, calib, m=m, lams=lams)
    _eval("edetector", float("nan"), s_null_e, thr_e,
          float((first_alarm(s_eval_e, thr_e) > 0).mean()))
    return rows


# --------------------------------------------------------------------------
# E7：非 iid 稳健性（序列相关下 ARL 保证偏离多少）
# --------------------------------------------------------------------------
def ar1_streams(pool: np.ndarray, reps: int, T: int, rho: float,
                rng: np.random.Generator) -> np.ndarray:
    """生成「边缘分布与 pool 相同、但滞后 1 自相关为 rho」的流。

    做法（copula / 分位数映射）：先造高斯 AR(1) `z_t = ρ z_{t−1} + √(1−ρ²) ε_t`，
    再用 pool 的经验分位数把 z 映射回原分布——这样**边缘分布完全一致**，
    只有相依结构在变，才能把「相关性」这一个因素单独拎出来。
    ρ=0 即独立同分布，ρ→1 时局部均值会长时间偏离整体均值（正是有界构造最怕的情形：
    合法性只要求 μ_n ≤ m，而相依会让**条件均值**在片段上超过 m）。
    """
    rng = rng or np.random.default_rng(0)
    p = np.sort(np.asarray(pool, dtype=float).ravel())
    if p.size == 0:
        raise ValueError("空数据池")
    rho = float(min(max(rho, 0.0), 0.999))
    z = np.empty((reps, T), dtype=float)
    z[:, 0] = rng.standard_normal(reps)
    scale = math.sqrt(max(1.0 - rho ** 2, 1e-12))
    for t in range(1, T):
        z[:, t] = rho * z[:, t - 1] + scale * rng.standard_normal(reps)
    # 概率积分变换 → 经验分位数映射（保边缘、保相依）
    try:
        from scipy.special import erf as _erf
        u = 0.5 * (1.0 + _erf(z / math.sqrt(2.0)))
    except ImportError:                                   # 没装 scipy 也能跑（慢一些）
        u = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
    idx = np.clip((u * p.size).astype(int), 0, p.size - 1)
    return p[idx]


def dependence_diagnostics(x: np.ndarray, m: float, window: int = 20) -> Dict:
    """相依性描述：滞后自相关、ESS 近似和窗口均值超界比例。

    窗口均值超界不是 E[X_n | 历史] > m 的检验；独立零假设也会偶然超界。
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    R, T = x.shape
    xc = x - x.mean(axis=1, keepdims=True)
    denom = (xc ** 2).sum(axis=1)
    acf1 = float(np.mean((xc[:, :-1] * xc[:, 1:]).sum(axis=1) /
                         np.maximum(denom, 1e-12))) if T > 1 else 0.0
    # 有效样本量（Bartlett 近似）：ESS ≈ T (1−ρ)/(1+ρ)
    ess = float(T * max(1e-3, (1 - acf1) / (1 + max(acf1, 0.0))))
    w = max(1, min(int(window), T))
    csum = np.cumsum(np.concatenate([np.zeros((R, 1)), x], axis=1), axis=1)
    loc = np.empty((R, T - w + 1))
    loc[:, :] = (csum[:, w:] - csum[:, :-w]) / w if T >= w else x.mean(axis=1)[:, None]
    exceed = float((loc > m).mean()) if loc.size else 0.0
    return {"acf1": round(acf1, 4), "ess": round(ess, 1),
            "local_exceed_rate": round(exceed, 4),
            "local_mean_max": round(float(loc.max()), 4) if loc.size else float("nan")}


def dependence_aware_m(calib: np.ndarray, window: int = 20, margin: float = 0.5,
                       strategy: str = "q85") -> float:
    """经验余量：把 m 抬到校准段局部均值的高值附近，不提供理论保证。

    参数含义：先用 `strategy` 得到名义上界，再看校准段**滑窗均值**的最大值——
    取两者较大者再加 margin 倍（滑窗均值的标准差）。这只是敏感性分析用的启发式。
    """
    c = np.asarray(calib, dtype=float).ravel()
    if c.size == 0:
        return 1.0
    m_nom = estimate_m(c, strategy)
    w = max(1, min(int(window), c.size))
    if w < c.size:
        csum = np.cumsum(np.insert(c, 0, 0.0))
        loc = (csum[w:] - csum[:-w]) / w
    else:
        loc = np.array([c.mean()])
    m_aware = float(loc.max() + margin * (loc.std(ddof=0) if loc.size > 1 else 0.0))
    return float(min(1.0 + 1e-9, max(m_nom, m_aware, 1e-6)))


def run_e7_dependence(generators: Sequence[Tuple[str, str, float]], pool: np.ndarray,
                      calib: np.ndarray, m_base: float, m_aware: float, lams: np.ndarray,
                      drift_delta: float, reps: int, T: int, at: int, T_post: int,
                      alphas: Sequence[float], rng: np.random.Generator,
                      window: int = 20, horizon: int = 300) -> List[Dict]:
    """E7：扫相依强度，看 ARL 保证偏离多少、以及「相依感知的 m」能不能补回来。

    `generators` 里每项是 (生成器名, 参数名, 参数值)：`block` 系列用块自助（块长越大相关越强），
    `ar1` 系列用分位数映射的 AR(1)（ρ 直接控制相关）。
    """
    rows: List[Dict] = []
    for gname, pname, pval in generators:
        if gname == "block":
            null = block_bootstrap(pool, reps, T, int(pval), rng)
        elif gname == "ar1":
            null = ar1_streams(pool, reps, T, float(pval), rng)
        else:
            raise ValueError(f"未知生成器: {gname}")
        diag = dependence_diagnostics(null, m_base, window)
        diag_aware = dependence_diagnostics(null, m_aware, window)
        row = {"experiment": "e7_dependence", "generator": gname, "param": pname,
               "param_value": pval, "m_base": round(m_base, 4),
               "m_aware": round(m_aware, 4), "window": int(window),
               "reps": int(reps), "T": int(T),
               "acf1": diag["acf1"], "ess": diag["ess"],
               "local_exceed_rate": diag["local_exceed_rate"],
               "local_exceed_rate_aware": diag_aware["local_exceed_rate"],
               "local_mean_max": diag["local_mean_max"]}
        for a in alphas:
            for tag, mm in (("base", m_base), ("aware", m_aware)):
                ll = lambda_grid(mm)
                M = build_detectors(null, mm, ll, "mix")
                times = first_alarm(M, 1.0 / a)
                ok = times > 0
                row[f"alarm_rate_{tag}_{a:g}"] = round(float(ok.mean()), 4)
                row[f"arl_mean_{tag}_{a:g}"] = (round(float(times[ok].mean()), 1)
                                                if ok.any() else "inf")
                row[f"arl_restricted_mean_{tag}_{a:g}"] = round(
                    float(np.where(ok, times, T + 1).mean()), 1)
        # 相依感知的 m 的代价：同一漂移下的检出与延迟
        pre = block_bootstrap(pool, reps, at, int(window), rng)
        post = block_bootstrap(pool, reps, T_post, int(window), rng)
        stream = inject_drift(np.concatenate([pre, post], axis=1), at=at, delta=drift_delta,
                              direction=+1, mode="contaminate", rng=rng)
        for tag, mm in (("base", m_base), ("aware", m_aware)):
            ll = lambda_grid(mm)
            M = build_detectors(stream, mm, ll, "mix")
            st = _edd_stats(M, 1.0 / 1000.0, at, T_post, horizon)
            row[f"detect_rate_{tag}"] = round(float(st["detect_rate"]), 3)
            em = st["edd_mean"]
            row[f"edd_{tag}"] = round(float(em), 1) if not is_inf(em) else "inf"
        rows.append(row)
    return rows


def load_proxy_matrix(path: Optional[str] = None, need: Sequence[str] = ()) -> Tuple[str, List[Dict]]:
    """读 E0 的逐查询代理矩阵。

    自动选择时会**跳过不含所需代理的矩阵**（例如跳过 --no-embed 跑出来的、
    检索侧列为空的那些），避免"代理有效样本不足（0）"这种令人困惑的失败。
    """
    if path:
        with open(path, encoding="utf-8-sig") as f:
            return path, list(csv.DictReader(f))
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "proxy_table_*.csv")), reverse=True)
    if not files:
        raise FileNotFoundError("找不到 results/proxy_table_*.csv（先跑 E0：probe_validity.py）")
    best = None
    for fp in files:
        with open(fp, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        filled = {p: sum(1 for r in rows if (r.get(p) or "").strip() != "") for p in need}
        if all(v >= 20 for v in filled.values()):
            return fp, rows
        if best is None or sum(filled.values()) > best[0]:
            best = (sum(filled.values()), fp, rows)
    print(f"  [提示] 没有一份矩阵同时含 {list(need)} 的全部数据，改用最接近的一份；"
          f"若检索侧为空请先跑一次带嵌入的 probe_validity.py")
    return best[1], best[2]


def series_from_rows(rows: List[Dict], proxy: str, method: str = "hmm") -> np.ndarray:
    """取某代理的按题序列（保留原始行序；跳过空值与非目标分块方法）。"""
    out = []
    for r in rows:
        if (r.get("method") or method) != method:
            continue
        v = (r.get(proxy) or "").strip()
        if v == "":
            continue
        try:
            out.append(float(v))
        except ValueError:
            continue
    return np.asarray(out, dtype=float)


def series_matrix(rows: List[Dict], proxies: Sequence[str],
                  method: str = "hmm") -> Tuple[np.ndarray, List[str]]:
    """多指标联合矩阵：只保留**所有指标都有值**的题（否则同一时刻的指标对不齐）。

    返回 (X, qids)，X 形状 (n, K)。
    """
    vals: Dict[str, Dict[str, float]] = {p: {} for p in proxies}
    order: List[str] = []
    for r in rows:
        if (r.get("method") or method) != method:
            continue
        qid = r["qid"]
        got: Dict[str, float] = {}
        ok = True
        for p in proxies:
            s = (r.get(p) or "").strip()
            if s == "":
                ok = False
                break
            try:
                got[p] = float(s)
            except ValueError:
                ok = False
                break
        if not ok:
            continue
        for p in proxies:
            vals[p][qid] = got[p]
        order.append(qid)
    if not order:
        return np.zeros((0, len(proxies))), []
    X = np.array([[vals[p][q] for p in proxies] for q in order], dtype=float)
    return X, order


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="RAG 漂移检测：最小 e-detector 验证")
    ap.add_argument("--proxy", default="ret_unique_docs,ans_refusal",
                    help="用哪些代理（逗号分隔；默认 E0 里最稳健的两个）")
    ap.add_argument("--matrix", default=None, help="代理矩阵 CSV（默认取最新）")
    ap.add_argument("--alpha", type=float, default=0.05, help="误报水平（ARL 实验用；阈值 = 1/α）")
    ap.add_argument("--alpha-edd", type=float, default=0.001,
                    help="EDD 实验用的误报水平：必须小到 ARL ≫ 变点前的步数，否则"
                         "检测器在变点前就报警（α=0.05 → ARL≥20 步，生产上太吵；"
                         "实际部署通常取 1e-3 量级）")
    ap.add_argument("--m-strategy", default=None,
                    help="变前上界估计：mean/p95/qNN/hoeffding/binom/max（可多选）。"
                         "缺省时：连续代理用 q60,q85,q100（扫 m 画 ARL/EDD 曲线）；"
                         "二元化（--binarize）后用 binom,hoeffding（分位数在 0/1 数据上会退化成 1.0）")
    ap.add_argument("--m-floor", type=float, default=0.02,
                    help="m 的下界（稀有事件代理校准窗口内可能全为 0，m=0 会让检测器退化）")
    ap.add_argument("--window", type=int, default=1,
                    help="进入检测器前的窗口长度；窗口样本数过少时不宜估计 m")
    ap.add_argument("--window-mode", choices=("overlap", "disjoint"), default="overlap",
                    help="overlap=重叠滑窗（仅作经验对照）；disjoint=不重叠窗口。"
                         "理论口径还需验证原始查询的逐时条件均值假设")
    ap.add_argument("--binarize", type=float, default=0.0, metavar="Q",
                    help="把代理转成 0/1 指示量（归一化值 ≥ Q 记为 1，0=关闭）。"
                         "用于**饱和型代理**：如 ret_unique_docs 归一化后上界就顶在 1.0，"
                         "合法的 m 必须 ≥ 上界 ⇒ L≤1 恒成立 ⇒ 永不报警；"
                         "取 Q=0.5 即「top-5 覆盖 ≥3 篇文档」，m=p̂+Hoeffding 就有余量了")
    ap.add_argument("--fuse", default="mix,max,min,single", help="对比哪些融合方式")
    ap.add_argument("--reps", type=int, default=1000, help="ARL 模拟重复次数")
    ap.add_argument("--T", type=int, default=1500, help="每条流的长度")
    ap.add_argument("--calib-frac", type=float, default=0.4, help="用前多少比例的数据估 m")
    ap.add_argument("--inject", default="0.05,0.1,0.2", help="注入漂移幅度（逗号分隔）")
    ap.add_argument("--inject-mode", default="contaminate", choices=["contaminate", "shift"],
                    help="注入方式：contaminate=一部分查询变最差（不饱和）；shift=加性平移")
    ap.add_argument("--no-inject", action="store_true", help="跳过 EDD 实验")
    ap.add_argument("--reps-edd", type=int, default=300, help="EDD 实验重复次数")
    ap.add_argument("--shuffle", dest="shuffle", action="store_true", default=True,
                    help="切分校准段前先打乱序列（默认开）")
    ap.add_argument("--no-shuffle", dest="shuffle", action="store_false",
                    help="按原始行序切分（仅当数据确实带时间序时才用；否则会把批次差异当漂移）")
    ap.add_argument("--e1", dest="e1", action="store_true", default=True,
                    help="跑 E1：单指标 vs 多指标混合（默认开）")
    ap.add_argument("--no-e1", dest="e1", action="store_false", help="跳过 E1")
    ap.add_argument("--e1-proxies", default="ans_refusal,ret_doc_hhi,ret_unique_docs",
                    help="E1 用哪些指标做混合（默认三个零成本的稳健代理）")
    ap.add_argument("--e1-targets", default="each", choices=["each", "all"],
                    help="E1 的漂移注入方式：each=逐个指标单独退化（对照矩阵，默认）；"
                         "all=所有指标同时退化")
    ap.add_argument("--e1-m", default="",
                    help="手工指定 E1 各指标的上界 m（逗号分隔，顺序与 --e1-proxies 一致）；"
                         "留空则用 --m-strategy 的第一个策略。E5 给出推荐 m 后用它复跑 E1")
    ap.add_argument("--e5", dest="e5", action="store_true", default=True,
                    help="跑 E5：m 策略扫描（ARL/EDD 权衡曲线，默认开）")
    ap.add_argument("--no-e5", dest="e5", action="store_false", help="跳过 E5")
    ap.add_argument("--e5-proxies", default="",
                    help="E5 扫哪些代理（默认沿用 --proxy）")
    ap.add_argument("--e5-grid", type=int, default=13,
                    help="分位数主网格的点数（q50→q100，默认 13）")
    ap.add_argument("--e2", dest="e2", action="store_true", default=True,
                    help="跑 E2：漂移类型归因（默认开，需要 E1 的联合流）")
    ap.add_argument("--no-e2", dest="e2", action="store_false", help="跳过 E2")
    ap.add_argument("--e2-window", type=int, default=50,
                    help="归因窗口（就近证据/recent 与 zscore/shift 规则用，默认 50 步）")
    ap.add_argument("--e2-deltas", default="0.05,0.1,0.2",
                    help="E2 扫描的漂移幅度（默认 0.05,0.1,0.2——**故意取小**："
                         "大漂移下三条规则都能到 1.00，区分不出方法优劣）")
    ap.add_argument("--e2-delays", default="0,50",
                    help="报警后再等多少步才归因（默认 0,50）")
    ap.add_argument("--e3", dest="e3", action="store_true", default=True,
                    help="跑 E3：权重方案对比（默认开，需要 E1 的联合流）")
    ap.add_argument("--no-e3", dest="e3", action="store_false", help="跳过 E3")
    ap.add_argument("--e3-prior", default="",
                    help="先验权重用的 E0 结果文件（默认取 results/ 下最新的一份）")
    ap.add_argument("--e3-prior-floor", type=float, default=0.25,
                    help="先验向均匀权重收缩的比例（0=纯先验，1=纯均匀，默认 0.25）")
    ap.add_argument("--e3-etas", default="0.25,0.5,1.0",
                    help="自适应加权的 η 网格（越大越激进）；最后一个会用于「从错误先验出发」")
    ap.add_argument("--e3-wrong-share", type=float, default=0.8,
                    help="错误先验里压在第 1 个指标上的权重（默认 0.8）")
    ap.add_argument("--e3-match-far", type=float, default=0.01,
                    help="同误报水平对比的目标误报率（每个方案各自定阈值；0=关闭，默认 0.01）")
    ap.add_argument("--e6", dest="e6", action="store_true", default=True,
                    help="跑 E6：与相关工作同口径对比（默认开，需要 E1 的联合流）")
    ap.add_argument("--no-e6", dest="e6", action="store_false", help="跳过 E6")
    ap.add_argument("--e6-deltas", default="0.2,0.4", help="E6 用的漂移幅度（默认 0.2,0.4）")
    ap.add_argument("--e6-target-far", type=float, default=0.01,
                    help="E6 对齐的目标误报率（默认 0.01）")
    ap.add_argument("--e6-window", type=int, default=50, help="两样本类方法的窗口长度")
    ap.add_argument("--e6-stride", type=int, default=5,
                    help="两样本类方法的计算步长（每 stride 步算一次，阶梯保持）")
    ap.add_argument("--e6-manual-threshold", type=float, default=2.0,
                    help="Devatwal 式人工阈值（标准化单位，默认 2.0=两倍标准差）")
    ap.add_argument("--e6-ref-len", type=int, default=120,
                    help="两样本类方法的参考窗长度（默认 120＝pre-change 段长度）")
    ap.add_argument("--e6-k-sweep", dest="e6_k_sweep", action="store_true", default=True,
                    help="额外跑 CUSUM 的参考偏移 k 扫描（默认开）")
    ap.add_argument("--no-e6-k-sweep", dest="e6_k_sweep", action="store_false",
                    help="跳过 CUSUM 的 k 扫描")
    ap.add_argument("--e6-ks", default="0.1,0.25,0.5,1.0",
                    help="CUSUM 参考偏移扫描的取值（标准化单位，默认 0.1,0.25,0.5,1.0）")
    ap.add_argument("--e7", dest="e7", action="store_true", default=True,
                    help="跑 E7：非 iid 稳健性（默认开，需要 E1 的联合流）")
    ap.add_argument("--no-e7", dest="e7", action="store_false", help="跳过 E7")
    ap.add_argument("--e7-proxy", default="", help="E7 用哪个代理（默认取 E1 的第一个）")
    ap.add_argument("--e7-blocks", default="1,5,20,50,100",
                    help="块自助的块长扫描（越大相关性越强；1=近似独立）")
    ap.add_argument("--e7-rhos", default="0,0.3,0.6,0.9",
                    help="AR(1) 的 ρ 扫描（边缘分布经分位数映射保持不变）")
    ap.add_argument("--e7-margin", type=float, default=0.5,
                    help="相依感知上界的余量（滑窗均值标准差的倍数，默认 0.5）")
    ap.add_argument("--e5-strategies", default="mean,q60,q85,hoeffding,binom,max",
                    help="曲线上的具名策略标记点（也会用于混合版的 m 替换）")
    ap.add_argument("--edd-horizon", type=int, default=300,
                    help="EDD 视界：只把变点后 H 步内的报警算作检出（0=不限，默认 300）。"
                         "不设视界会把零假设侧的偶发误报记成检出（曾出现「检出率 0.01、EDD 493 步」的假象）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M")
    proxies = [p.strip() for p in args.proxy.split(",") if p.strip()]
    matrix_path, rows = load_proxy_matrix(args.matrix, need=proxies)
    fuses = [f.strip() for f in args.fuse.split(",") if f.strip()]
    deltas = [] if args.no_inject else [float(x) for x in args.inject.split(",") if x.strip()]
    strategies = [s.strip() for s in (args.m_strategy or "").split(",") if s.strip()]
    if not strategies:
        # 二元/稀有事件用分位数会退化（q60 在正例率 66% 的 0/1 序列上就是 1.0 → m=1 → 永不报警）
        strategies = ["binom", "hoeffding"] if args.binarize > 0 else ["q60", "q85", "q100"]
    # 各实验段用**独立的随机数发生器**：共用一个 rng 时，E1 的乱序 permutate（决定校准段/检验段
    # 的抽样位置，进而决定 m）会随前面逐代理实验消耗的随机数个数变化——于是同一个 seed 下，
    # 只要改一下 --reps/--T（本该只影响前面那段），E1 的上界 m 就整体偏移，
    # 实测能把 ARL 结论从"全部报警"翻成"一条都不报警"。分离后各段结果互不干扰、可复现。
    rng = np.random.default_rng([args.seed, 1])        # 逐代理 ARL/EDD
    rng_e1 = np.random.default_rng([args.seed, 2])     # E1 多指标对照
    rng_traj = np.random.default_rng([args.seed, 3])   # 示例轨迹
    threshold = 1.0 / args.alpha

    print(f"代理矩阵: {os.path.basename(matrix_path)}（{len(rows)} 行）")
    print(f"α={args.alpha} → 阈值 1/α={threshold:.1f}；流长 T={args.T}，重复 {args.reps}"
          f"；窗口聚合={args.window}")
    print(f"代理: {proxies}｜融合: {fuses}｜m 策略: {strategies}")

    # 先按校准段定尺度（避免前视），再做窗口聚合与归一化
    units: Dict[str, np.ndarray] = {}
    calib_idx: Dict[str, int] = {}

    def build_unit(p: str) -> bool:
        """构造一个代理的归一化序列（校准段定尺度 → 窗口聚合 → 方向翻正 → 可选二值化）。

        抽成函数是为了让 E5 也能请求 --proxy 之外的代理（早期版本 E5 只认 units 里已有的，
        于是 `--e5-proxies` 里多写的代理被静默丢掉，报告里少了一整个代理）。
        """
        raw = series_from_rows(rows, p)
        if raw.size < 20:
            print(f"  [跳过] {p}: 有效样本不足（{raw.size}）")
            return False
        # 标注集按批次排列而非时间排列；先打乱原始查询，再切段。
        # 滑窗必须分别在两段内部计算，否则校准窗口会包含检验查询。
        u, n_calib = prepare_proxy_split(
            raw, args.calib_frac, args.window, rng, args.shuffle,
            invert=DEGRADE_DIRECTION.get(p, +1) < 0,
            binarize=args.binarize, window_mode=args.window_mode)
        units[p] = u
        calib_idx[p] = n_calib
        extra = f"｜二元化(≥{args.binarize}) 正例率 {u.mean():.3f}" if args.binarize > 0 else ""
        print(f"  {p}: 原始 {raw.size} 条 → 均值 {u.mean():.3f}，范围 {u.min():.3f}~{u.max():.3f}"
              f"（尺度按前 {n_calib} 条定{('，已打乱顺序' if args.shuffle else '，按原始行序')}"
              f"{'，已按退化方向翻正' if DEGRADE_DIRECTION.get(p, +1) < 0 else ''}）{extra}")
        return True

    for p in proxies:
        build_unit(p)
    if not units:
        print("[错误] 没有可用代理")
        return 1

    records: List[Dict] = []
    # ---- 实验 1：ARL（变前同分布，应 ≥ 1/α；max 融合应显著低于）----
    print("\n[实验 1] 变前报警与右截尾时间（模拟诊断，不构成理论保证验证）")
    for strategy in strategies:
        for p, u in units.items():
            n_calib = calib_idx[p]
            m = estimate_m(u[:n_calib], strategy)
            if m < args.m_floor:
                m = args.m_floor
                print(f"  [提示] {p} 的 m 估计值过低（校准窗口内几乎没有事件），"
                      f"已抬到下界 {args.m_floor}；稀有事件代理建议 --window 20 做窗口聚合")
            pool = u[n_calib:] if u.size > n_calib else u
            # 描述性检查：池样本均值不是逐时刻条件均值，不能证明理论前提。
            m_ok = bool(pool.mean() <= m)
            if m >= 1.0 - 1e-9:
                print(f"  [警告] {p}/{strategy}: m≈1.0 → L≤1 恒成立，检测器**永不可能报警**（无功效）；"
                      f"饱和型代理请加 --binarize 0.5 转成 0/1 指示量")
            elif not m_ok:
                print(f"  [警告] {p}/{strategy}: m={m:.3f} < 检验池样本均值 {pool.mean():.3f} "
                      "→ 与理论前提相容性存疑；样本均值本身不能检验条件均值")
            streams = block_bootstrap(pool, args.reps, args.T, 20, rng)
            lams = lambda_grid(m)
            dl, du = delta_bounds(m)
            for how in fuses:
                r = run_arl(streams, m, lams, how, args.alpha)
                rec = {"experiment": "arl", "proxy": p, "m_strategy": strategy,
                       "m": round(m, 4), "pool_mean": round(float(pool.mean()), 4),
                       "m_ge_pool_mean": int(m_ok),
                       "delta_L": round(dl, 4), "delta_U": round(du, 1),
                       "window": args.window, "fuse": how, "alpha": args.alpha,
                       "threshold": round(threshold, 2), "K": int(len(lams)),
                       **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}}
                records.append(rec)
                print(f"  {p:<17} m={m:.3f}({strategy}) {how:<7} "
                      f"ARL均值={rec['arl_mean']:>8.1f} 中位={rec['arl_median']:>7.1f} "
                      f"报警率={rec['alarm_rate']:.2f}")
    # ---- 实验 2：EDD（注入漂移后的检测延迟）----
    if deltas:
        print("\n[实验 2] 漂移注入后的 EDD（遍历 m 策略 → 直接体现 ARL/EDD 权衡）")
        for p, u in units.items():
            direction = +1  # build_unit 已按退化方向翻正
            for strategy in strategies:
                m = max(estimate_m(u[:calib_idx[p]], strategy), args.m_floor)
                pool = u[calib_idx[p]:] if u.size > calib_idx[p] else u
                lams = lambda_grid(m)
                for d in deltas:
                    for how in fuses:
                        r = run_edd(pool, m, lams, how, args.alpha_edd, at=120, delta=d,
                                    direction=direction, reps=args.reps_edd,
                                    T_pre=120, T_post=args.T, rng=rng,
                                    mode=args.inject_mode, horizon=args.edd_horizon)
                        rec = {"experiment": "edd", "proxy": p, "m_strategy": strategy,
                               "m": round(m, 4), "window": args.window, "fuse": how,
                               "alpha": args.alpha_edd,
                               "threshold": round(1.0 / args.alpha_edd, 0),
                               **{k: round(v, 2) if isinstance(v, float) else v
                                  for k, v in r.items()}}
                        records.append(rec)
                        print(f"  {p:<17} {strategy:<10} m={m:.3f} Δ={d:<5} {how:<7} "
                              f"变前误报={rec['pre_alarm_rate']:.2f} "
                              f"检出率={rec['detect_rate']:.2f} "
                              f"EDD均值={rec['edd_mean']:>7.1f} 中位={rec['edd_median']:>6.1f}")

    # ---- 实验 3：E1 单指标 vs 多指标混合（联合重采样，配对比较）----
    e1_ok = False          # E5 的混合扫描要复用 E1 的归一化矩阵与联合流，用它判断是否可用
    e1_proxies: List[str] = []
    if getattr(args, "e1", True):
        e1_proxies = [p.strip() for p in (args.e1_proxies or "").split(",") if p.strip()]
        e1_proxies = [p for p in e1_proxies if p in DEGRADE_DIRECTION]
        X, qids = series_matrix(rows, e1_proxies)
        if X.shape[0] < 30:
            print(f"\n[实验 3/E1] 跳过：可用于联合分析的题数不足（{X.shape[0]}）")
        else:
            print(f"\n[实验 3/E1] 单指标 vs 多指标混合：{e1_proxies}（{X.shape[0]} 题对齐）")
            Xn, n_calib_e1 = prepare_proxy_split(
                X, args.calib_frac, args.window, rng_e1, args.shuffle,
                binarize=0.0, window_mode=args.window_mode)
            for k, p in enumerate(e1_proxies):
                if DEGRADE_DIRECTION.get(p, +1) < 0:      # 同样要把"下降=退化"的代理翻正
                    Xn[:, k] = 1.0 - Xn[:, k]
            if args.binarize > 0:
                Xn = (Xn >= args.binarize).astype(float)
            ms_e1 = [max(estimate_m(Xn[:n_calib_e1, k], strategies[0]), args.m_floor)
                     for k in range(Xn.shape[1])]
            if (args.e1_m or "").strip():
                # E5 的推荐 m 到手后，用它复跑 E1：验证「Δ 小的时候混合被稀释」到底是方法问题
                # 还是上界选得太保守（实测很大程度是后者）。
                manual = [float(x) for x in args.e1_m.split(",") if x.strip()]
                if len(manual) != len(e1_proxies):
                    print(f"  [警告] --e1-m 给了 {len(manual)} 个值，但代理有 {len(e1_proxies)} 个，已忽略")
                else:
                    ms_e1 = [max(v, args.m_floor) for v in manual]
            pool_e1 = Xn[n_calib_e1:]
            print(f"  各指标上界 m（E1 统一取 --m-strategy 的第一个：{strategies[0]}）：" + "、".join(
                f"{p}={m:.3f}" for p, m in zip(e1_proxies, ms_e1)))
            # 稀有代理（如拒答）在校准段常常一次都不出现 → 分位数上界退化成 m_floor。
            # 上界贴着下限会让该分量异常敏感：漂移时几拍就报警，但变前也会因偶发事件误报。
            for p, m in zip(e1_proxies, ms_e1):
                if m <= args.m_floor + 1e-12:
                    print(f"  [注意] {p} 的上界触到 m_floor={args.m_floor}：校准段太短或事件太稀有，"
                          f"该分量会偏敏感（可加大 --calib-frac 或改用 binom/hoeffding 上界）")
            lams_e1 = lambda_grid(max(ms_e1))
            # 变前 ARL（同一批联合流）
            streams_e1 = bootstrap_rows(pool_e1, args.reps, args.T, 20, rng_e1)
            # 阈值校准与实际误报评估使用独立流，避免 E3/E6 在训练流上报告 1%。
            streams_e1_eval = bootstrap_rows(
                pool_e1, args.reps, args.T, 20,
                np.random.default_rng([args.seed, 5]))
            # E5 的混合扫描复用这两个：同一批联合流（配对）+ 同一份注入基底流
            base_e1 = np.concatenate(
                [bootstrap_rows(pool_e1, args.reps_edd, 120, 20, rng_e1),
                 bootstrap_rows(pool_e1, args.reps_edd, args.T, 20, rng_e1)], axis=1)
            e1_ok = True
            for r in run_e1_arl(streams_e1, ms_e1, lams_e1, args.alpha):
                idx = r["k"]
                r.update({"experiment": "e1_arl", "proxy": (e1_proxies[idx] if idx >= 0 else "全指标"),
                          "m_strategy": strategies[0], "alpha": args.alpha,
                          "threshold": round(1.0 / args.alpha, 2), "window": args.window,
                          "binarize": args.binarize, "K": len(e1_proxies),
                          "n": int(X.shape[0])})
                if isinstance(r.get("arl_mean"), float) and r["arl_mean"] == float("inf"):
                    r["arl_mean"] = "inf"
                records.append({k: (round(v, 3) if isinstance(v, float) else v)
                                for k, v in r.items()})
                print(f"  {r['fuse']:<12} ARL均值={r['arl_mean']} 报警率={r['alarm_rate']:.2f}")
            # 漂移注入 EDD：既跑"全部指标同时退化"，也跑"只让某一个指标退化"
            # （后者才是多指标融合的价值所在：漂移类型未知时单指标会瞎）
            e1_targets = None if args.e1_targets == "all" else list(range(len(e1_proxies)))
            for d in (deltas or [0.2]):
                # 序列已按退化方向翻正 → 注入一律为"上升"（否则方向为 −1 的指标会被往下打，
                # 而有界构造只能检出上升，等于白注入）
                dirs = [+1] * len(e1_proxies)
                for r in run_e1_edd(pool_e1, ms_e1, lams_e1, args.alpha_edd, at=120,
                                    delta=d, directions=dirs, reps=args.reps_edd,
                                    T_pre=120, T_post=args.T, rng=rng_e1,
                                    mode=args.inject_mode, targets=e1_targets,
                                    horizon=args.edd_horizon):
                    idx = r["k"]
                    r.update({"experiment": "e1_edd",
                              "proxy": (e1_proxies[idx] if idx >= 0 else "全指标"),
                              "m_strategy": strategies[0], "alpha": args.alpha_edd,
                              "threshold": round(1.0 / args.alpha_edd, 0),
                              "window": args.window, "binarize": args.binarize,
                              "K": len(e1_proxies), "n": int(X.shape[0])})
                    records.append({k: (round(v, 3) if isinstance(v, float) else v)
                                    for k, v in r.items()})
                    em = r["edd_mean"]
                    print(f"  Δ={d} {r['fuse']:<12} 检出率={r['detect_rate']:.2f} "
                          f"EDD均值={'—' if em == float('inf') else f'{em:.1f}'}")

    # ---- 实验 4：E5 m 策略扫描（保守性 vs 检测延迟的权衡曲线）----
    scan_deltas = deltas or [0.2]          # E5 与 E2 共用（--no-e5 时 E2 也要用）
    if getattr(args, "e5", True):
        e5_proxies = [p.strip() for p in (args.e5_proxies or "").split(",") if p.strip()]
        e5_proxies = [p for p in e5_proxies if p in DEGRADE_DIRECTION] or list(units)
        for p in e5_proxies:                     # E5 允许请求 --proxy 之外的代理
            if p not in units:
                print(f"  [E5] 追加代理 {p}")
                build_unit(p)
        e5_proxies = [p for p in e5_proxies if p in units] or list(units)
        extra = [s.strip() for s in (args.e5_strategies or "").split(",") if s.strip()]
        print(f"\n[实验 4/E5] m 扫描：{e5_proxies}｜分位数网格 {args.e5_grid} 点 + "
              f"具名策略 {extra}｜Δ={scan_deltas}")
        for p in e5_proxies:
            # 与 E1 共用同一条序列与同一段校准：E1 走「先归一化再滑窗」（联合矩阵），
            # 逐代理路径走「先滑窗再归一化」，两条路线的 m 不可比。E1 的代理必须用 E1 的那份，
            # 否则同一份报告里同一个指标会出现两个 m，读者无法判断「E1 选的 m 落在曲线哪里」。
            k_e1 = e1_proxies.index(p) if (e1_ok and p in e1_proxies) else -1
            if k_e1 >= 0:
                u, n_calib = Xn[:, k_e1], n_calib_e1
            else:
                u, n_calib = units[p], calib_idx[p]
            calib, pool = u[:n_calib], (u[n_calib:] if u.size > n_calib else u)
            # 变前流与漂移流只重采样一次，各 m 点共用（配对比较，差异只来自 m）
            null_streams = block_bootstrap(pool, args.reps, args.T, 20, rng)
            drift_streams = {}
            for d in scan_deltas:
                pre = block_bootstrap(pool, args.reps_edd, 120, 20, rng)
                post = block_bootstrap(pool, args.reps_edd, args.T, 20, rng)
                drift_streams[d] = inject_drift(np.concatenate([pre, post], axis=1),
                                                at=120, delta=d,
                                                # 序列在上游已按退化方向翻正，注入一律为"上升"；
                                                # 早期版本这里沿用 DEGRADE_DIRECTION，
                                                # 于是 HHI 的漂移被往**下**打——有界构造只检上升，
                                                # 整条曲线会假装"任何 m 都无功效"（实测踩过）。
                                                direction=+1,
                                                mode=args.inject_mode, rng=rng)
            points = m_grid_from_calib(calib, extra, args.e5_grid)
            for r in e5_curve_rows(null_streams, drift_streams, points, ["mix"], args.alpha,
                                   args.alpha_edd, at=120, horizon=args.edd_horizon,
                                   m_floor=args.m_floor, pool_mean=float(pool.mean())):
                r.update({"proxy": p, "m_strategy": r["m_label"], "window": args.window,
                          "binarize": args.binarize, "mode_strategy": args.m_strategy,
                          # 记下 E1 当时用的 m，报告里直接把「E1 的点」标在曲线上
                          "e1_m": (round(float(ms_e1[k_e1]), 4) if k_e1 >= 0 else "")})
                records.append({k: (round(v, 3) if isinstance(v, float) else v)
                                for k, v in r.items()})
            sub = [r for r in records if r.get("experiment") == "e5_curve" and r["proxy"] == p]
            for r in sub:
                if r.get("delta") == scan_deltas[0]:
                    em = r["edd_mean"]
                    print(f"  {p:<17} m={r['m']:.3f}({r['m_label']:<10}) "
                          f"合法={'是' if r['m_ge_pool_mean'] else '否'} "
                          f"变前报警率={r['arl_alarm_rate']:.2f} "
                          f"检出率={r['detect_rate']:.2f} "
                          f"EDD={'—' if isinstance(em, str) or em == float('inf') else f'{em:.1f}'}")

        # 混合版本：同一策略下把所有指标的 m 一起换掉——回答「E1 选了 q85，换个策略会怎样」
        if getattr(args, "e1", True) and e1_ok:
            ms_by_strategy = []
            for s in extra:
                if s == "binom" and not np.all((np.abs(Xn[:n_calib_e1]) < 1e-12) |
                                                 (np.abs(Xn[:n_calib_e1] - 1.0) < 1e-12)):
                    print("  [E5] 跳过 binom：联合指标中包含连续值或滑窗均值")
                    continue
                ms_by_strategy.append((s, [max(estimate_m(Xn[:n_calib_e1, k], s), args.m_floor)
                                           for k in range(Xn.shape[1])]))
            drift_multi = []
            for k in range(len(e1_proxies)):
                for d in scan_deltas:
                    stream = np.array(base_e1, copy=True)
                    stream[:, :, k] = inject_drift(stream[:, :, k], at=120, delta=d,
                                                   direction=+1, mode=args.inject_mode,
                                                   rng=rng_e1)
                    drift_multi.append((f"仅指标#{k}", d, stream))
            print(f"  混合版本（同一策略下换所有指标的 m）："
                  f"{'、'.join(s for s, _ in ms_by_strategy)}")
            for r in e5_mix_rows(streams_e1, drift_multi, ms_by_strategy, args.alpha,
                                 args.alpha_edd, at=120, horizon=args.edd_horizon):
                r.update({"proxy": "全指标", "window": args.window, "binarize": args.binarize,
                          "n": int(X.shape[0]), "K": len(e1_proxies)})
                records.append({k: (round(v, 3) if isinstance(v, float) else v)
                                for k, v in r.items()})
                if r.get("drift_target") == "（变前）":
                    print(f"  {r['m_strategy']:<10} m={r['ms']} 变前报警率={r['arl_alarm_rate']:.2f}")
                else:
                    em = r["edd_mean"]
                    print(f"  {r['m_strategy']:<10} 漂移={r['drift_target']} Δ={r['delta']} "
                          f"检出率={r['detect_rate']:.2f} "
                          f"EDD={'—' if isinstance(em, str) or em == float('inf') else f'{em:.1f}'}")
    # ---- 实验 5：E2 漂移类型归因（报警之后，指出是哪个指标在退化）----
    if getattr(args, "e2", True) and e1_ok:
        # 注意：命令行给的是逗号分隔的字符串，必须先解析成数值列表——
        # 直接把字符串传下去会按字符迭代（'0','.','0'…），在 inject_drift 里炸成 TypeError。
        e2_deltas = [float(x) for x in str(args.e2_deltas).split(",") if x.strip()]
        e2_delays = [int(float(x)) for x in str(args.e2_delays).split(",") if x.strip()]
        print(f"\n[实验 5/E2] 漂移类型归因：报警后判断是哪一个指标在退化"
              f"（Δ={e2_deltas}，归因窗口 {args.e2_window} 步，归因延迟 {e2_delays} 步）")
        e2_rows = run_e2_attribution(
            pool_e1, ms_e1, lams_e1, args.alpha_edd, at=120, deltas=e2_deltas,
            targets=list(range(len(e1_proxies))), directions=[+1] * len(e1_proxies),
            reps=args.reps_edd, T_pre=120, T_post=args.T, rng=rng_e1,
            mode=args.inject_mode, horizon=args.edd_horizon, window=args.e2_window,
            calib=Xn[:n_calib_e1], delays=e2_delays)
        for r in e2_rows:
            r.update({"proxy": "全指标混合", "n": int(X.shape[0]), "K": len(e1_proxies),
                      "k_proxies": "、".join(e1_proxies)})
            records.append({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
        for delay in e2_delays:
            for rule in ATTRIBUTION_RULES:
                sub = [r for r in e2_rows if r["rule"] == rule and r["delay"] == delay
                       and r["acc"] == r["acc"]]
                if sub:
                    print(f"  延迟 {delay:>3} 步 规则 {rule:<7} "
                          f"平均命中率={np.mean([r['acc'] for r in sub]):.2f}"
                          f"（共 {len(sub)} 个 目标×Δ 格）")
    # ---- 实验 6：E3 权重方案（均匀 / 先验 / 自适应）----
    if getattr(args, "e3", True) and e1_ok:
        prior, prior_src = load_proxy_prior(e1_proxies, args.e3_prior, args.e3_prior_floor)
        k_dim = len(e1_proxies)
        # 「故意放错」的先验：把大部分权重压在第 1 个指标上（其余平分）——
        # 用来检验自适应能不能从一个错误先验出发自己纠正回来。
        wrong = np.full(k_dim, (1.0 - args.e3_wrong_share) / max(1, k_dim - 1))
        wrong[0] = args.e3_wrong_share
        schemes: List[Tuple[str, Dict]] = [
            ("均匀权重", {"kind": "fixed", "w": None}),
            ("先验权重（E0 相关强度）", {"kind": "fixed", "w": prior}),
            ("先验权重（故意放错）", {"kind": "fixed", "w": wrong}),
        ]
        for eta in [float(x) for x in str(args.e3_etas).split(",") if x.strip()]:
            schemes.append((f"自适应 η={eta:g}（从均匀出发）",
                            {"kind": "adaptive", "w": None, "eta": eta}))
        schemes.append((f"自适应 η={args.e3_etas.split(',')[-1].strip()}（从错误先验出发）",
                        {"kind": "adaptive", "w": wrong, "eta": float(args.e3_etas.split(",")[-1])}))
        print(f"\n[实验 6/E3] 权重方案对比：{k_dim} 个指标"
              f"｜先验来自 {prior_src}：{'、'.join(f'{p}={w:.2f}' for p, w in zip(e1_proxies, prior))}"
              f"｜错误先验：{'、'.join(f'{w:.2f}' for w in normalize_weights(wrong))}")
        drift_list = []
        for k in range(k_dim):
            for d in scan_deltas:
                stream = np.array(base_e1, copy=True)
                stream[:, :, k] = inject_drift(stream[:, :, k], at=120, delta=d,
                                               direction=+1, mode=args.inject_mode,
                                               rng=rng_e1)
                drift_list.append((f"仅指标#{k}", d, stream))
        e3_rows = run_e3_weights(schemes, streams_e1, drift_list, ms_e1, lams_e1,
                                 args.alpha, args.alpha_edd, at=120,
                                 horizon=args.edd_horizon, match_far=args.e3_match_far,
                                 null_eval_multi=streams_e1_eval)
        for r in e3_rows:
            r.update({"proxy": "全指标混合", "n": int(X.shape[0]), "K": k_dim,
                      "k_proxies": "、".join(e1_proxies), "prior_src": prior_src})
            records.append({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
        for name, _spec in schemes:
            arl = [float(r.get("arl_alarm_rate_op", float("nan"))) for r in e3_rows
                   if r["scheme"] == name and r["drift_target"] == "（变前）"]
            hit = [r for r in e3_rows if r["scheme"] == name
                   and r["drift_target"] != "（变前）" and r.get("delta") == scan_deltas[0]]
            det = np.mean([float(r["detect_rate"]) for r in hit]) if hit else float("nan")
            edds = [float(r["edd_mean"]) for r in hit if not is_inf(r["edd_mean"])]
            print(f"  {name:<26} 误报率@α_edd={arl[0]:.2f} "
                  f"Δ={scan_deltas[0]} 平均检出率={det:.2f} "
                  f"平均 EDD={'—' if not edds else f'{np.mean(edds):.1f}'}")
    # ---- 实验 7：E6 与相关工作同口径对比 ----
    if getattr(args, "e6", True) and e1_ok:
        e6_deltas = [float(x) for x in str(args.e6_deltas).split(",") if x.strip()]
        args.e6_ks = [float(x) for x in str(args.e6_ks).split(",") if x.strip()] \
            if isinstance(args.e6_ks, str) else list(args.e6_ks)
        print(f"\n[实验 7/E6] 与相关工作同口径对比：同一输入序列 + 同一误报水平"
              f"（目标误报 {args.e6_target_far:.0%}，窗口 {args.e6_window} 步）")
        drift_single = []
        for k, p_name in enumerate(e1_proxies):
            for d in e6_deltas:
                stream = np.array(base_e1, copy=True)
                stream[:, :, k] = inject_drift(stream[:, :, k], at=120, delta=d,
                                               direction=+1, mode=args.inject_mode,
                                               rng=rng_e1)
                drift_single.append((f"{p_name}", d, stream[:, :, k]))
        for k, p_name in enumerate(e1_proxies):
            single_null = streams_e1[:, :, k]
            rows_k = run_e6_compare(
                E6_METHODS, single_null,
                [(lab, d, st) for lab, d, st in drift_single if lab == p_name],
                calib=Xn[:n_calib_e1, k], ms=[ms_e1[k]], lams=lambda_grid(ms_e1[k]),
                at=120, horizon=args.edd_horizon, target_far=args.e6_target_far,
                window=args.e6_window, stride=args.e6_stride,
                manual_threshold=args.e6_manual_threshold, ref_len=args.e6_ref_len,
                null_eval_streams=streams_e1_eval[:, :, k])
            for r in rows_k:
                r.update({"proxy": p_name, "n": int(X.shape[0]), "K": 1,
                          "k_proxies": p_name, "e1_m": round(float(ms_e1[k]), 4)})
                records.append({kk: (round(v, 3) if isinstance(v, float) else v)
                                for kk, v in r.items()})
            for r in rows_k:
                if r["drift_target"] == "（变前）" or r.get("delta") != e6_deltas[0]:
                    continue
                em = r["edd_mean"]
                print(f"  {p_name:<17} {r['method']:<16} Δ={r['delta']} "
                      f"检出率={r['detect_rate']:.2f} "
                      f"EDD={'—' if is_inf(em) else f'{float(em):.1f}'} "
                      f"（误报率 {r['alarm_rate']:.3f}）")
            # CUSUM 的「假设漂移幅度」扫描：它要知道变后均值移了多少，本文方法不需要
            if getattr(args, "e6_k_sweep", True):
                sweep = run_e6_cusum_sweep(
                    single_null,
                    [(lab, d, st) for lab, d, st in drift_single if lab == p_name],
                    calib=Xn[:n_calib_e1, k], assumed_ks=args.e6_ks, m=ms_e1[k],
                    lams=lambda_grid(ms_e1[k]), at=120, horizon=args.edd_horizon,
                    target_far=args.e6_target_far,
                    null_eval_streams=streams_e1_eval[:, :, k])
                for r in sweep:
                    r.update({"proxy": p_name, "n": int(X.shape[0]),
                              "k_proxies": p_name, "e1_m": round(float(ms_e1[k]), 4)})
                    records.append({kk: (round(v, 3) if isinstance(v, float) else v)
                                    for kk, v in r.items()})
                print(f"  {p_name}：CUSUM 参考偏移 k 的作用（同一误报水平下的 EDD）")
                for r in sweep:
                    em = r["edd_mean"]
                    tag = ("本文 e-detector" if r["method"] == "edetector"
                           else f"假设位移 {r['assumed_k']:g}σ")
                    print(f"      Δ={r['delta']:<4} {tag:<18} "
                          f"检出率={r['detect_rate']:.2f} "
                          f"EDD={'—' if is_inf(em) else f'{float(em):.1f}'}")
    # ---- 实验 8：E7 非 iid 稳健性（序列相关下 ARL 保证偏离多少）----
    if getattr(args, "e7", True) and e1_ok:
        args.e7_blocks = [int(float(x)) for x in str(args.e7_blocks).split(",") if x.strip()]
        args.e7_rhos = [float(x) for x in str(args.e7_rhos).split(",") if x.strip()]
        e7_proxy = (args.e7_proxy or e1_proxies[0])
        if e7_proxy not in units:
            build_unit(e7_proxy)
        k7 = e1_proxies.index(e7_proxy) if (e1_ok and e7_proxy in e1_proxies) else -1
        if k7 >= 0:
            u7, n7 = Xn[:, k7], n_calib_e1
        else:
            u7, n7 = units[e7_proxy], calib_idx[e7_proxy]
        calib7, pool7 = u7[:n7], u7[n7:]
        # 基准上界必须与 E1/E5 用的那个一致：早先这里独立按 strategies[0]（默认 q60）算，
        # 得到一个**违反 μ ≤ m** 的值，于是所有相依强度下误报率都是 1.00，实验完全失效。
        m_base = (float(ms_e1[k7]) if k7 >= 0
                  else float(max(estimate_m(calib7, strategies[0]), args.m_floor)))
        m_aware = max(m_base, dependence_aware_m(calib7, window=args.window,
                                                margin=args.e7_margin,
                                                strategy=strategies[-1]))
        generators = ([("block", "块长", float(b)) for b in args.e7_blocks]
                      + [("ar1", "ρ", float(r)) for r in args.e7_rhos])
        print(f"\n[实验 8/E7] 合成序列相关性压力测试：代理={e7_proxy}｜名义上界 m={m_base:.3f}、"
              f"相依感知上界 m'={m_aware:.3f}｜α={args.alpha:g}/{args.alpha_edd:g}")
        e7_rows = run_e7_dependence(
            generators, pool7, calib7, m_base, m_aware, lambda_grid(m_base),
            drift_delta=(scan_deltas[0] if scan_deltas else 0.2), reps=args.reps,
            T=args.T, at=120, T_post=args.T, alphas=[args.alpha, args.alpha_edd],
            rng=np.random.default_rng([args.seed, 4]), window=args.window,
            horizon=args.edd_horizon)
        for r in e7_rows:
            r.update({"proxy": e7_proxy, "n": int(u7.size), "K": 1,
                      "m_strategy": strategies[0]})
            records.append({kk: (round(v, 3) if isinstance(v, float) else v)
                            for kk, v in r.items()})
            print(f"  {r['generator']}={r['param_value']:<6g} 自相关={r['acf1']:+.3f} "
                  f"局部超界比例={r['local_exceed_rate']:.3f}"
                  f"（用 m' 后 {r['local_exceed_rate_aware']:.3f}） "
                  f"报警率@α={r.get(f'alarm_rate_base_{args.alpha:g}')} "
                  f"检出(base/aware)={r.get('detect_rate_base')}/{r.get('detect_rate_aware')} "
                  f"EDD={r.get('edd_base')}/{r.get('edd_aware')}")
    # ---- 输出 ----
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_csv = os.path.join(RESULTS_DIR, f"edetector_{stamp}.csv")
    for record in records:
        record["window_mode"] = args.window_mode
    keys = sorted({k for r in records for k in r})
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)

    # 示例轨迹（用于画图）：挑**真正能检出**的那组配置（EDD 检出率最高、EDD 最短），
    # 否则画出来的曲线离阈值十万八千里（早期版本用 strategies[0] 固定生成，就踩过这个坑）。
    best = None
    for r in records:
        if r.get("experiment") != "edd" or not r.get("detect_rate"):
            continue
        key = (r["detect_rate"], -(r["edd_mean"] if r["edd_mean"] != float("inf") else 1e9))
        if best is None or key > best[0]:
            best = (key, r)
    if best is None:
        for r in records:
            if r.get("experiment") == "arl" and r.get("alarm_rate"):
                best = ((r["alarm_rate"], 0), r)
                break
    p0 = best[1]["proxy"] if best else list(units)[0]
    how0 = best[1]["fuse"] if best else "mix"
    strategy0 = best[1]["m_strategy"] if best else strategies[0]
    delta0 = best[1].get("delta", deltas[0] if deltas else 0.1) if best else 0.1
    u0 = units[p0]
    m0 = max(estimate_m(u0[:calib_idx[p0]], strategy0), args.m_floor)
    lams0 = lambda_grid(m0)
    thr_series = 1.0 / (args.alpha_edd if deltas else args.alpha)
    pre = block_bootstrap(u0[calib_idx[p0]:], 1, 120, 20, rng_traj)
    post = block_bootstrap(u0[calib_idx[p0]:], 1, 300, 20, rng_traj)
    drift = inject_drift(np.concatenate([pre, post], axis=1), 120, delta0,
                         +1, mode=args.inject_mode, rng=rng_traj)   # 已翻正 → 一律上升
    series_csv = os.path.join(RESULTS_DIR, f"edetector_series_{stamp}.csv")
    mix = build_detectors(drift, m0, lams0, "mix")[0]
    mx = build_detectors(drift, m0, lams0, "max")[0]
    t_mix = int(first_alarm(np.atleast_2d(mix), thr_series)[0])
    t_max = int(first_alarm(np.atleast_2d(mx), thr_series)[0])
    with open(series_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "x", "M_mix", "M_max", "threshold", "change_point",
                    "alarm_t_mix", "alarm_t_max"])
        for t in range(drift.shape[1]):
            w.writerow([t + 1, round(float(drift[0, t]), 4), round(float(mix[t]), 4),
                        round(float(mx[t]), 4), round(thr_series, 2), 120, t_mix, t_max])
    print(f"\n示例轨迹配置：代理={p0}｜m 策略={strategy0}(m={m0:.3f})｜融合对比 mix vs max｜Δ={delta0}"
          f"｜阈值={thr_series:.0f}｜报警步数 mix={t_mix or '未报警'} max={t_max or '未报警'}")

    report = write_report(records, args, proxies, fuses, strategies, matrix_path,
                          os.path.basename(out_csv), os.path.basename(series_csv), units)
    print(f"\n结果: {out_csv}\n轨迹: {series_csv}\n报告: {report}")
    return 0


def write_report(records: List[Dict], args, proxies, fuses, strategies, matrix_path,
                 out_csv, series_csv, units: Dict[str, np.ndarray]) -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    report = os.path.join(LOG_DIR, f"E-detector_最小验证_{time.strftime('%Y%m%d')}.md")
    arl = [r for r in records if r["experiment"] == "arl"]
    edd = [r for r in records if r["experiment"] == "edd"]
    thr = 1.0 / args.alpha
    # 章节号按「实际存在的实验」顺排（跳过某个实验时不留空号）
    _seq = ["五", "六", "七", "八", "九"]
    _state = {"i": 0}

    def next_sec() -> str:
        s = _seq[min(_state["i"], len(_seq) - 1)]
        _state["i"] += 1
        return s
    lines = [
        "# e-detector 代理序列模拟（非真实 RAG 故障验证）",
        "",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M')}；α={args.alpha} → 阈值 1/α={thr:.0f}",
        f"- 代理矩阵：`{os.path.basename(matrix_path)}`；代理：{'、'.join(proxies)}",
        f"- 流长 T={args.T}，模拟重复 {args.reps} 次；原始查询先切段，"
        f"各段内部再按 {args.window_mode} 模式聚合（窗口 {args.window}）",
        "- 漂移直接注入代理序列；块重采样反映合成序列的依赖，不能视为线上时间相关性的实测。",
        "- 重叠滑窗会重复使用原始查询；即使原始查询独立，窗口值的逐时条件均值也可能超过 m。"
        "不重叠窗口避免机械重叠，仍须另行验证正常查询的条件均值前提。",
        "",
        "## 一、变前报警与右截尾时间（模拟诊断）",
        "",
        "> 理论前提是每一步的条件均值 E[X_t|历史] ≤ m。样本均值检查只是诊断，"
        "不能证明该前提。未报警流在 T+1 处截尾；截尾均值估计的是 E[min(报警时间,T+1)]，"
        "已报警流的均值只是条件均值。有限 T 的报警率也不是 α。",
        "",
        "| 代理 | m 策略 | m | 检验池均值 | m ≥ 池均值 | 融合 | 已报警流均值 | 截尾均值 | T 内报警率 | 诊断 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(arl, key=lambda x: (x["proxy"], x["m_strategy"], x["fuse"])):
        ok_theory = r.get("m_ge_pool_mean", 1) == 1
        am = r["arl_mean"]
        if r.get("alarm_rate", 0) == 0:
            verdict = "本次未观察到报警"
        elif not ok_theory:
            verdict = "m 低于池样本均值"
        else:
            verdict = "需检查条件均值前提"
        lines.append(f"| {r['proxy']} | {r['m_strategy']} | {r['m']} | {r.get('pool_mean', '')} | "
                     f"{'是' if ok_theory else '否'} | {r['fuse']} | "
                     f"{'—' if is_inf(am) else f'{float(am):.1f}'} | "
                     f"{float(r.get('arl_restricted_mean', args.T + 1)):.1f} | "
                     f"{r['alarm_rate']:.2f} | {verdict} |")
    if arl and not any(r.get("alarm_rate", 0) > 0 for r in arl):
        lines += ["",
                   "> T 步内没有报警只说明本次模拟的截尾均值为 T+1；"
                   "不能据此证明条件均值前提或理论 ARL 保证。"]
    lines += ["", "## 二、漂移注入后的检测延迟（EDD，变点在第 120 步）", ""]
    if edd:
        lines += ["| 代理 | m 策略 | m | Δ | 融合 | 变前误报率 | 检出率 | EDD 均值 | EDD 中位 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(edd, key=lambda x: (x["proxy"], x["m_strategy"], x["delta"], x["fuse"])):
            em = "—" if r["edd_mean"] == float("inf") else f"{r['edd_mean']:.1f}"
            ed = "—" if r["edd_median"] == float("inf") else f"{r['edd_median']:.1f}"
            lines.append(f"| {r['proxy']} | {r.get('m_strategy', '')} | {r.get('m', '')} | "
                         f"{r['delta']} | {r['fuse']} | {r.get('pre_alarm_rate', 0):.2f} | "
                         f"{r['detect_rate']:.2f} | {em} | {ed} |")
    else:
        lines.append("（本次跳过：--no-inject）")
    # 融合对比小结：**用报警率比**（ARL 均值会被截尾成 inf，比较会得出"无差异"的错误结论）
    def _avg(rows_, fuse):
        """ARL 均值（仅统计报警的流）——被截尾的流不计入，所以只能作参考。"""
        v = [x["arl_mean"] for x in rows_
             if x["fuse"] == fuse and x.get("arl_mean", float("inf")) != float("inf")]
        return sum(v) / len(v) if v else float("nan")

    def _avg_rate(rows_, fuse):
        v = [x["alarm_rate"] for x in rows_ if x["fuse"] == fuse and "alarm_rate" in x]
        return sum(v) / len(v) if v else float("nan")

    def _avg_edd(rows_, fuse):
        v = [x["edd_mean"] for x in rows_ if x["fuse"] == fuse and x["edd_mean"] != float("inf")]
        return sum(v) / len(v) if v else float("nan")

    rate_mix, rate_max, rate_min = _avg_rate(arl, "mix"), _avg_rate(arl, "max"), _avg_rate(arl, "min")
    eff = [r for r in arl if r.get("alarm_rate", 0) > 0]
    no_power = [r for r in arl if r.get("alarm_rate", 0) == 0]
    lines += ["", "## 三、结论", "",
              f"- **误报控制（用报警率比，而不是 ARL 均值）**：同一阈值下 T 步内报警概率 "
              f"mix={rate_mix:.2f}、max={rate_max:.2f}、min={rate_min:.2f}"
              + (f" → **取最大的误报是混合的 {rate_max / rate_mix:.1f} 倍**（Remark 3.1 的实证）"
                 if rate_mix and rate_max > rate_mix else "（本次未观察到明显差异）"),
               f"- 已报警流的条件平均时间（不是 ARL）：mix={_avg(arl, 'mix'):.1f}、"
              f"max={_avg(arl, 'max'):.1f}、min={_avg(arl, 'min'):.1f}、single={_avg(arl, 'single'):.1f}",
               f"- **T 内无变前报警配置 {len(no_power)}/{len(arl)}**；这不等于无检出功效。"
               "若 m≥1 且 X≤1，则增量恒不超过 1，才可断定该构造不会报警。",
               "- **m 估计策略（E5）**：m 越保守通常越难报警；样本均值和分位数都不是条件均值的保证上界。",
               "- **α 的含义**：仅在 e-detector 构造及条件均值前提成立时，"
               "阈值 1/α 对应理论 ARL ≥ 1/α；本次模拟不能验证该前提。",
              ]
    if edd:
        big = max(r["delta"] for r in edd)
        sub = [r for r in edd if r["delta"] == big]
        lines += ["", f"**Δ={big} 时各融合方式的检测延迟**（越大越慢＝越保守）：", "",
                  "| 代理 | m 策略 | mix | max | min | single |", "|---|---|---|---|---|---|"]
        for p_ in sorted({r["proxy"] for r in sub}):
            for s_ in sorted({r["m_strategy"] for r in sub}):
                cells = []
                for fuse in ("mix", "max", "min", "single"):
                    vals = [x["edd_mean"] for x in sub
                            if x["proxy"] == p_ and x["m_strategy"] == s_ and x["fuse"] == fuse]
                    vals = [v for v in vals if v != float("inf")]
                    cells.append(f"{vals[0]:.0f}" if vals else "—")
                lines.append(f"| {p_} | {s_} | " + " | ".join(cells) + " |")
    # E1：单指标 vs 多指标混合
    e1_arl = [r for r in records if r.get("experiment") == "e1_arl"]
    e1_edd = [r for r in records if r.get("experiment") == "e1_edd"]
    if e1_arl:
        k_proxies = [r["proxy"] for r in e1_arl if r.get("k", -1) >= 0]
        lines += ["", "## 四、E1：单指标 vs 多指标混合（联合重采样，配对比较）", "",
                  "> 多个指标在**同一批流**上重采样（保持指标间相关结构），各自用自己的上界 m_k；",
                  "> 固定凸权重保持 e-detector 形式；理论保证仍要求每个代理的条件均值前提成立。", "",
                  "| 配置 | 分量数 | 已报警流均值 | T 内报警率 |", "|---|---|---|---|"]
        for r in e1_arl:
            am = r["arl_mean"]
            lines.append(f"| {r['fuse']}{('（' + r['proxy'] + '）') if r.get('k', -1) >= 0 else ''} | "
                         f"{r.get('n_comp', '')} | {'—' if is_inf(am) else f'{float(am):.1f}'} | "
                         f"{r['alarm_rate']:.2f} |")
        # 变前 ARL 全都不报警时，必须解释清楚：这不是「检测器没用」，而是上界保守的代价，
        # 否则读者会以为 ARL 保证没生效（保证只在报警时才有分辨率）。
        if not any(r["alarm_rate"] > 0 for r in e1_arl):
            lines += ["", "> T 步内均未报警；此结果不能证明理论 ARL 保证，也无法估计完整平均报警时间。"]
        if e1_edd:
            hz = e1_edd[0].get("edd_horizon", 0)
            lines += ["", f"| 配置 | 漂移目标 | Δ | 变前误报率 | 检出率（H={hz or '∞'} 步内） | 超时报警率 | EDD 均值 | EDD 中位 |",
                      "|---|---|---|---|---|---|---|---|"]
            for r in e1_edd:
                em = r["edd_mean"]
                ed = r["edd_median"]
                lines.append(f"| {r['fuse']}{('（' + r['proxy'] + '）') if r.get('k', -1) >= 0 else ''} | "
                             f"{r.get('drift_target', '')} | {r['delta']} | "
                             f"{r.get('pre_alarm_rate', 0.0):.2f} | {r['detect_rate']:.2f} | "
                             f"{r.get('late_alarm_rate', 0.0):.2f} | "
                             f"{'—' if is_inf(em) else f'{float(em):.1f}'} | "
                             f"{'—' if is_inf(ed) else f'{float(ed):.1f}'} |")
            if hz:
                lines += ["",
                          f"> 检出率只统计**变点后 {hz} 步内**的报警；更晚才响的那些计入「超时报警率」，"
                          "既不算检出也不隐瞒。不设视界（`--edd-horizon 0`）时，一个对该漂移本就无感的"
                          "检测器会因零假设侧的偶发误报被记成「检出」，出现「检出率 0.01、EDD 493 步」"
                          "这类假象，还会把对照矩阵的对角线弄脏。"]
            # 核心对照矩阵：检测器 × 漂移目标（对角线快 = 打中了它盯的指标；对角线外应失效）
            big = max(r["delta"] for r in e1_edd)
            sub = [r for r in e1_edd if r["delta"] == big]
            tgts = [t for t in dict.fromkeys(r.get("drift_target", "") for r in sub) if t]
            dets = [d for d in dict.fromkeys(r["fuse"] for r in sub) if d]
            if len(tgts) > 1 and len(dets) > 1:
                lines += ["", "### 检测器 × 漂移目标 对照矩阵（单元格 = EDD 均值 / 检出率）", "",
                          "> 对角线＝漂移正好打在它盯的指标上；对角线外＝打在别的指标上。",
                          "> 单指标检测器在「打偏」时应失效（—），而混合应当仍然检出——"
                          "这就是多指标融合在**漂移类型未知**时的价值。", "",
                          "| 检测器 \\ 漂移目标 | " + " | ".join(tgts) + " |",
                          "|---" * (len(tgts) + 1) + "|"]
                for d in dets:
                    cells = []
                    for t in tgts:
                        hit = [x for x in sub if x["fuse"] == d and x.get("drift_target") == t]
                        if not hit:
                            cells.append("—")
                        else:
                            x = hit[0]
                            em = x["edd_mean"]
                            cells.append(f"{'—' if is_inf(em) else f'{float(em):.0f}'} / {x['detect_rate']:.2f}")
                    lines.append(f"| {d} | " + " | ".join(cells) + " |")
            # 混合 vs 单指标：必须在**同一次漂移注入**下、且与**盯对指标的那个单指标**比。
            # 两个曾经的错误：① 跨漂移目标取「最快的单指标」（拿 A 漂移的成绩比 B 漂移的混合）；
            # ② 只看谁 EDD 小，于是把「对该漂移无感、只是零假设侧偶发误报」的单指标当成命中者
            #    （实测出现过 493 步 / 1% 的假命中）。现在按漂移目标序号锁定对照，并要求它真的检出。
            mix_rows = [r for r in e1_edd if r["k"] == -1 and r["fuse"] == "全指标混合"]
            single_rows = [r for r in e1_edd if r["k"] >= 0]
            if any(float(r.get("pre_alarm_rate", 0.0)) > 0.05 for r in e1_edd):
                lines.append("- ⚠️ 至少一个配置的变前误报率超过 5%；其检出率和 EDD 不宜用于方法优劣结论。")
            for mrow in sorted(mix_rows, key=lambda x: (x["delta"], str(x.get("drift_target")))):
                tgt = mrow.get("drift_target", "")
                if is_inf(mrow["edd_mean"]) or float(mrow.get("pre_alarm_rate", 0.0)) > 0.05:
                    continue
                want_k = None
                if "#" in tgt:
                    try:
                        want_k = int(tgt.split("#")[-1])
                    except ValueError:
                        want_k = None
                same = [r for r in single_rows if r["delta"] == mrow["delta"]
                        and r.get("drift_target") == tgt]
                matched = [r for r in same if want_k is not None and r["k"] == want_k]
                cand = [r for r in (matched or same)
                        if float(r.get("pre_alarm_rate", 0.0)) <= 0.05]
                if not cand:
                    continue
                good = [r for r in cand if r["detect_rate"] >= 0.5 and not is_inf(r["edd_mean"])]
                if good:
                    ref = min(good, key=lambda r: float(r["edd_mean"]))
                    ratio = float(mrow["edd_mean"]) / float(ref["edd_mean"])
                    if max(float(mrow.get("pre_alarm_rate", 0.0)),
                           float(ref.get("pre_alarm_rate", 0.0))) <= 0.05:
                        lines.append(
                            f"- Δ={mrow['delta']}｜{tgt}：混合 EDD={float(mrow['edd_mean']):.1f} 步 vs "
                            f"盯对指标的单指标（{ref['proxy']}）{float(ref['edd_mean']):.1f} 步 → "
                            f"混合耗时为 oracle 的 **{ratio:.2f}×**（oracle 事先知道漂移目标）")
                else:
                    ref_rate = float(cand[0]["detect_rate"]) if cand else float("nan")
                    if mrow["detect_rate"] >= 0.5:
                        lines.append(
                            f"- Δ={mrow['delta']}｜{tgt}：**盯对指标的单指标也没检出**"
                            f"（检出率 {ref_rate:.2f}），混合检出 EDD={float(mrow['edd_mean']):.1f} 步 "
                            f"→ 融合独有的鲁棒性")
                    else:
                        lines.append(
                            f"- Δ={mrow['delta']}｜{tgt}：该幅度下**两个都不可靠**"
                            f"（盯对指标的单指标 {ref_rate:.2f} vs 混合 {mrow['detect_rate']:.2f}）"
                            f"→ 漂移太小，混合的稀释效应压过了鲁棒性收益，要看更大 Δ")
            if single_rows:
                # 「失效」用检出率判定（有视界时 EDD 可能仍是有限值——那是零假设侧的超时误报）；
                # 按漂移幅度分开统计，否则强弱两种 Δ 会互相掩盖。
                parts = []
                for d_ in sorted({r["delta"] for r in single_rows}):
                    s_ = [r for r in single_rows if r["delta"] == d_]
                    m_ = [r for r in mix_rows if r["delta"] == d_]
                    parts.append(
                        f"Δ={d_}：单指标检出 {sum(1 for r in s_ if r['detect_rate'] >= 0.5)}/{len(s_)}，"
                        f"混合 {sum(1 for r in m_ if r['detect_rate'] >= 0.5)}/{len(m_)}")
                lines.append(
                    "- **模拟检出个数**（按漂移幅度分开）——" + "；".join(parts) +
                    "。须同时查看各格变前误报率，才可解释这些检出差异。")
    # 五、E5：m 策略扫描
    e5_curve = [r for r in records if r.get("experiment") == "e5_curve"]
    e5_mix = [r for r in records if r.get("experiment") == "e5_mix"]
    if e5_curve or e5_mix:
        lines += ["", f"## {next_sec()}、E5：m 的保守性与检测延迟（权衡曲线）", "",
                  "> m 控制灵敏度；理论要求的是条件均值上界，而非检验池样本均值。"
                  "下表的样本均值比较仅作描述性筛选，不是合法性证明。",
                  "> 横轴是 m（从校准段 q50 扫到 q100），具名策略是曲线上的标记点；"
                  "同一 m 下的变前流与漂移流在各点间**共用**，所以点与点的差异只来自 m。"]
        if e5_curve:
            for p_ in dict.fromkeys(r["proxy"] for r in e5_curve):
                sub = [r for r in e5_curve if r["proxy"] == p_]
                d0 = min(r["delta"] for r in sub)
                sub = sorted([r for r in sub if r["delta"] == d0], key=lambda x: x["m"])
                pm = sub[0].get("pool_mean", "")
                lines += ["", f"### {p_}（Δ={d0}，变前均值 {pm}）", "",
                          "| m 点 | m | m ≥ 检验池样本均值 | 变前报警率@α | 变前报警率@α_edd | 检出率 | EDD 均值 |",
                          "|---|---|---|---|---|---|---|"]
                for r in sub:
                    em = r["edd_mean"]
                    # 变前就大量报警时，「检出率」只是少数没在变点前触发的幸运流，
                    # 数字不可读——必须标出来，否则会被误当成「检出了」。
                    noisy = float(r["arl_alarm_rate"]) >= 0.5
                    lines.append(f"| {r['m_label']} | {r['m']:.3f} | "
                                 f"{'是' if r['m_ge_pool_mean'] else '否'} | "
                                 f"{r['arl_alarm_rate']:.2f}"
                                 f"{'（误报过高，功效列不可读）' if noisy else ''} | "
                                 f"{_num(r.get('arl_alarm_rate_op'))} | "
                                 f"{r['detect_rate']:.2f} | "
                                 f"{'—' if is_inf(em) else f'{float(em):.1f}'} |")
                # 只保留「合法」的点做取舍讨论：不合法的点误报率不可比。
                # 误报率按**实际运行的那条报警线**（α_edd）判，工程上容忍 5% 的余量；
                # 用 α=0.05 那条阈值会把可选区间压得过窄（拒答代理会整段被判成不可用）。
                legal = [r for r in sub if r.get("m_ge_pool_mean")]
                usable = [r for r in legal
                          if r["detect_rate"] >= 0.5 and not is_inf(r["edd_mean"])
                          and float(r.get("arl_alarm_rate_op", 1.0)) <= 0.05]
                if usable:
                    best = min(usable, key=lambda r: float(r["edd_mean"]))
                    least = max(usable, key=lambda r: float(r["edd_mean"]))
                    lines.append(
                        f"- 本次模拟的候选区间：m ∈ [{min(r['m'] for r in usable):.3f}, "
                        f"{max(r['m'] for r in usable):.3f}]；最快 {best['m_label']}"
                        f"（m={best['m']:.3f}，EDD {float(best['edd_mean']):.1f} 步）"
                        f"到最慢 {least['m_label']}（m={least['m']:.3f}，"
                        f"EDD {float(least['edd_mean']):.1f} 步）——"
                        f"**EDD 相差 {float(least['edd_mean']) / float(best['edd_mean']):.1f} 倍**。")
                else:
                    lines.append("- 本次扫描没有同时满足样本均值、模拟报警率和检出率筛选的点。")
                # E1 当时用的那个 m 落在曲线哪里？以及按规则推荐的更紧的点
                e1m = next((r for r in sub if r.get("e1_m") not in ("", None)
                            and abs(float(r["m"]) - float(r["e1_m"])) < 1e-9
                            and r["m_label"] not in ("q50",)), None)
                if e1m is not None:
                    em = e1m["edd_mean"]
                    lines.append(
                        f"- **E1 当时用的点**：{e1m['m_label']}（m={e1m['m']:.3f}）→ "
                        f"检出率 {e1m['detect_rate']:.2f}、"
                        f"EDD {'未检出' if is_inf(em) else f'{float(em):.1f} 步'}。")
                tight = [r for r in legal
                         if float(r.get("arl_alarm_rate_op", 1.0)) <= 0.05
                         and r["detect_rate"] >= 0.9 and not is_inf(r["edd_mean"])]
                if tight:
                    rec = min(tight, key=lambda r: r["m"])
                    em = float(rec["edd_mean"])
                    cmp_txt = ""
                    if e1m is not None and not is_inf(e1m["edd_mean"]):
                        cmp_txt = (f"，比 E1 的点快 "
                                   f"{float(e1m['edd_mean']) / em:.1f} 倍")
                    lines.append(
                        f"- **模拟候选点**（样本均值检查后，取「变前报警率@α_edd ≤ 0.05 且检出率 ≥ 0.9」的"
                        f"**最小** m）：{rec['m_label']}（m={rec['m']:.3f}）→ "
                        f"检出率 {rec['detect_rate']:.2f}、EDD {em:.1f} 步{cmp_txt}。")
                lines.append(
                    "- ⚠️ 候选点在同一批模拟流上筛选，存在选择偏差；"
                    "需要另留独立查询和模拟种子验证，不能直接当作理论上界。")
        if e5_mix:
            lines += ["", "### 混合检测器：整体换 m 策略", "",
                      "| m 策略 | 各指标 m | Δ | 漂移目标 | 变前报警率 | 检出率 | EDD 均值 |",
                      "|---|---|---|---|---|---|---|"]
            for r in sorted(e5_mix, key=lambda x: (x["m_strategy"], str(x.get("drift_target")))):
                em = r.get("edd_mean", float("inf"))
                baseline = next((x for x in e5_mix
                                 if x["m_strategy"] == r["m_strategy"]
                                 and x.get("drift_target") == "（变前）"), None)
                arl_rate = baseline.get("arl_alarm_rate") if baseline else None
                noisy = (arl_rate not in (None, "")) and float(arl_rate) >= 0.5
                dr = ("—" if r.get("drift_target") == "（变前）"
                      else (f"{float(r['detect_rate']):.2f}"
                            + ("（误报过高，不可读）" if noisy else "")))
                lines.append(f"| {r['m_strategy']} | {r.get('ms', '')} | {r.get('delta', '')} | "
                             f"{r.get('drift_target', '')} | {_num(arl_rate)} | "
                             f"{dr} | {'—' if is_inf(em) else f'{float(em):.1f}'} |")
            lines += ["",
                      "> 读法：变前报警率那一行是**误报侧**（越小越安全），检出率/EDD 是**功效侧**；"
                      "误报率接近 1 的策略（mean/q60 这类在短校准段上会跌破均值的估计）"
                      "其功效列没有意义，已标注「误报过高，不可读」。",
                      "> 口径提醒：混合版的 m 由**同一个策略名**逐指标计算，"
                      "只要有一个指标的 m 违反 μ ≤ m，整个混合的误报就会失控——"
                      "这与 E1 里「以最激进的指标为准」的观察一致。"]
            lines += ["",
                      "**选 m 的工程规则（本次数据）**：",
                      "1. 先检查 m 与独立检验池的样本均值；该检查不能代替条件均值前提；",
                      "2. `binom` 只用于原始 0/1 独立样本，不能用于连续代理或重叠滑窗；",
                      "3. 本轮模拟流仅用于筛选候选 m；最终报警率和检出延迟还须另留独立查询与模拟流核验；",
                      "4. T 步内零报警只表示本次观察被截尾，同时需要检查检出功效。"]
    # 六、E2：漂移类型归因
    e2 = [r for r in records if r.get("experiment") == "e2_attr"]
    if e2:
        hz = e2[0].get("horizon", "")
        win = e2[0].get("window", "")
        # 章节号随「E5 是否存在」浮动：跳过 E5 的报告里不应留下空号
        sec_no = next_sec()
        lines += ["", f"## {sec_no}、E2：漂移类型归因（报警之后，是哪一个指标在退化）", "",
                  "> E1 回答了「有没有变差」，E2 回答「是哪一类变差」。做法：混合统计量",
                  "> M_n = Σ_k ω_k M_k(n) 按指标可分解，报警时比较各指标自身的 e-value",
                  "> （规则 evalue）、最近一段的对数增量（规则 recent），",
                  "> 并与工程上最自然的「窗口均值偏移最大者」（规则 shift）对照。",
                  f"> 归因窗口 {win} 步；只统计变点后 {hz} 步内报警的流（没报警就无从归因）。", ""]
        if any(float(r.get("pre_alarm_rate", 0.0)) > 0.05 for r in e2):
            lines.append("> ⚠️ 部分配置的变前误报率超过 5%；下方命中率只描述幸存的报警流，不能直接概括整体归因效果。")
        rules = [r for r in dict.fromkeys(x["rule"] for x in e2) if r]
        deltas_ = [d for d in dict.fromkeys(x["delta"] for x in e2) if d != ""]
        delays_ = [d for d in dict.fromkeys(x["delay"] for x in e2) if d != ""]
        k_cls = max(1, int(e2[0].get("K", 1)))
        for delay in delays_:
            lines += ["", f"### 归因延迟 {int(delay)} 步"
                          + ("（报警当下就判断）" if int(delay) == 0 else "（报警后再等这么久）"),
                      "",
                      "| Δ | " + " | ".join(f"规则 {r}" for r in rules) + " | 平均报警率 |",
                      "|---" * (len(rules) + 2) + "|"]
            for d in deltas_:
                cells = []
                for r in rules:
                    vals = [float(x["acc"]) for x in e2
                            if x["delta"] == d and x["rule"] == r and x["delay"] == delay
                            and x["acc"] == x["acc"]]
                    cells.append(f"{np.mean(vals):.2f}" if vals else "—")
                ar = [float(x["detect_rate"]) for x in e2
                      if x["delta"] == d and x["delay"] == delay]
                lines.append(f"| {d} | " + " | ".join(cells)
                             + f" | {np.mean(ar):.2f} |" if ar else
                             f"| {d} | " + " | ".join(cells) + " | — |")
        lines += ["",
                  "> 口径：命中率只在**该格真正报了警**的流上统计（报警率那一列是分母），"
                  "所以小 Δ 下的命中率是「报警之后能否指对」，不是「能否报警」；"
                  "K=3 类，随机猜 = 0.33。"]
        best_rule = None
        for r in rules:
            vals = [float(x["acc"]) for x in e2 if x["rule"] == r and x["acc"] == x["acc"]]
            if vals and (best_rule is None or np.mean(vals) > best_rule[1]):
                best_rule = (r, float(np.mean(vals)))
        # 延迟维度单独给一条结论：等得久不一定更好（就近证据类规则会退化）
        delay_note = ""
        if len(delays_) >= 2:
            d0, d1 = min(delays_), max(delays_)
            parts = []
            for r in rules:
                a0 = [float(x["acc"]) for x in e2 if x["rule"] == r and x["delay"] == d0
                      and x["acc"] == x["acc"]]
                a1 = [float(x["acc"]) for x in e2 if x["rule"] == r and x["delay"] == d1
                      and x["acc"] == x["acc"]]
                if a0 and a1:
                    parts.append(f"{r} {np.mean(a0):.2f}→{np.mean(a1):.2f}")
            if parts:
                delay_note = (f"- 归因延迟的影响（延迟 {int(d0)} 步 → {int(d1)} 步，平均命中率）："
                              + "；".join(parts) + "。")
        if best_rule:
            lines += ["",
                      f"- 最好的归因规则是 **{best_rule[0]}**（平均命中率 {best_rule[1]:.2f}，"
                      f"随机猜 {1.0 / k_cls:.2f}，提升 {best_rule[1] * k_cls:.1f} 倍）。"]
        if delay_note:
            lines.append(delay_note)
        lines += [
            "- 口径提醒：本实验的漂移是**逐指标注入**的，所以「归因」= 找出被注入的那个指标；"
            "真实系统级漂移（换语料 / 换分块 / 换模型）往往会让多个指标同时移动，"
            "那时需要的是「漂移来源 → 指标特征」的联合推断，属于下一步（需要漂移场景库）。",
            "- 未检出率：报警率不到 1 的那些格子里，剩余部分不是「归因错」，而是**根本没报警**，"
            "两者要分开看。"]
    # E3：权重方案
    e3 = [r for r in records if r.get("experiment") == "e3_weight"]
    if e3:
        sec_no = next_sec()
        schemes_ = [s for s in dict.fromkeys(r["scheme"] for r in e3) if s]
        deltas_ = sorted({r["delta"] for r in e3 if r.get("drift_target") != "（变前）"})
        match_far = next((r.get("match_target") for r in e3 if r.get("match_target")), "")
        lines += ["", f"## {sec_no}、E3：权重方案（均匀 / 先验 / 自适应）", "",
                  "> 混合的权重是本文的实验变量之一：均匀（基线）、按 E0 实证的相关强度加先验、",
                  "> 以及按「上一时刻各指标自身的 e-value」逐点调整的自适应加权"
                  "（可预测、任意时刻权重和为 1）。",
                  "> 时变权重对完整统计量重新加权，不自动继承固定凸混合的 e-detector 理论保证；"
                  "这里仅检验模拟误报。",
                  "> **关键口径**：自适应加权在同一阈值下误报更高，直接比 EDD 等于拿两条不同"
                  "误报水平的曲线比。第二张表为**每个方案各自定阈值**，并在独立流上核对实际误报率。", ""]

        def _edd_cells(calib):
            out = {}
            for s in schemes_:
                row = []
                for d in deltas_:
                    hit = [r for r in e3 if r["scheme"] == s and r.get("calib") == calib
                           and r.get("drift_target") != "（变前）" and r.get("delta") == d]
                    edds = [float(r["edd_mean"]) for r in hit if not is_inf(r["edd_mean"])]
                    dets = [float(r["detect_rate"]) for r in hit]
                    if not hit:
                        row.append("—")
                    elif edds:
                        row.append(f"{np.mean(dets):.2f} / {np.mean(edds):.1f}")
                    else:
                        row.append(f"{np.mean(dets):.2f} / —")
                out[s] = row
            return out

        cells_fixed, cells_match = _edd_cells("fixed"), _edd_cells("matched")
        lines += ["### 表 1：同一阈值（1/α_edd）下的误报与延迟", "",
                  "| 方案 | 权重 | 变前误报率 | " + " | ".join(f"Δ={d} 检出/EDD" for d in deltas_)
                  + " | 阈值 |", "|---" * (len(deltas_) + 4) + "|"]
        for s in schemes_:
            null = next((r for r in e3 if r["scheme"] == s and r.get("calib") == "fixed"
                         and r.get("drift_target") == "（变前）"), None)
            if not null:
                continue
            lines.append(f"| {s} | {null.get('weights', '')} | "
                         f"{float(null.get('arl_alarm_rate_op', 0)):.3f} | "
                         + " | ".join(cells_fixed[s]) + f" | {null.get('threshold', '')} |")
        if match_far:
            lines += ["", f"### 表 2：目标误报率 {match_far} 下的延迟（独立流报告实际误报率）", "",
                      "| 方案 | 权重 | 匹配阈值 | 实际误报率 | "
                      + " | ".join(f"Δ={d} 检出/EDD" for d in deltas_) + " |",
                      "|---" * (len(deltas_) + 4) + "|"]
            for s in schemes_:
                null = next((r for r in e3 if r["scheme"] == s and r.get("calib") == "matched"
                             and r.get("drift_target") == "（变前）"), None)
                if not null:
                    continue
                lines.append(f"| {s} | {null.get('weights', '')} | {null.get('threshold', '')} | "
                             f"{float(null.get('arl_alarm_rate_op', 0)):.3f} | "
                             + " | ".join(cells_match[s]) + " |")
            matched_null = [r for r in e3 if r.get("calib") == "matched"
                            and r.get("drift_target") == "（变前）"]
            if matched_null:
                fars = [float(r["arl_alarm_rate_op"]) for r in matched_null]
                lines.append(
                    f"- 匹配阈值在独立变前流上的实际误报率范围为 "
                    f"{min(fars):.3f}–{max(fars):.3f}；"
                    "若各方案的实际误报率不同，表 2 的 EDD 仅作描述性比较。")
                if any(abs(far - match_far) > max(0.02, 2 * match_far) for far in fars):
                    lines.append(
                        f"- 本轮独立流的误报率未达到 {match_far:.1%} 目标；"
                        "不能把表 2 称为同目标误报率下的严格功效比较。")
            lines.append(
                "- 自适应权重、错误先验与均匀权重的效果应同时看检出率、EDD 和实际误报率；"
                "本轮代理序列模拟不足以证明某种权重方案在真实 RAG 故障中更优。")
    # E6：与相关工作同口径对比
    e6 = [r for r in records if r.get("experiment") == "e6_compare"]
    e6k = [r for r in records if r.get("experiment") == "e6_cusum_k"]
    if e6:
        sec_no = next_sec()
        far_t = float(e6[0].get("target_far", 0.01))
        n_eval = int(e6[0].get("n_reps", args.reps))
        lines += ["", f"## {sec_no}、E6：同一代理输入与目标误报率下的方法对比", "",
                  "> 四条相关工作与本方法都看**同一条输入序列**（被打漂移那个指标的归一化序列），"
                  "各方法在一批变前流上按同一目标误报率选阈值，在独立变前流上报告实际误报率。",
                  "> 两样本类方法（KS / JSD / Fréchet / MMD）的参考分布取每条流自己的前 120 个点"
                  "（文献里的 baseline window）——**不能拿校准段当参考**：它只有几十个点、"
                  "方差常与整体差一大截，统计量会整体饱和、阈值校准直接失效（实测 KS 在所有流上顶到 1.0）。",
                  f"> 目标误报率 {far_t:.0%}，各方法实际误报率可能不同；"
                  f"{n_eval} 条流在该目标下预期约 {n_eval * far_t:.1f} 次报警，估计较粗。"
                  "人工阈值方法不校准。", ""]
        proxies_ = [p for p in dict.fromkeys(r["proxy"] for r in e6) if p]
        deltas_ = sorted({r["delta"] for r in e6 if r.get("delta") != ""})
        for p_ in proxies_:
            lines += [f"### {p_}（单指标序列）", "",
                      "| 方法 | 变前误报率 | " + " | ".join(f"Δ={d} 检出/EDD" for d in deltas_)
                      + " |", "|---" * (len(deltas_) + 2) + "|"]
            for mname in dict.fromkeys(r["method"] for r in e6 if r["proxy"] == p_):
                head = [r for r in e6 if r["proxy"] == p_ and r["method"] == mname
                        and r["drift_target"] == "（变前）"]
                far = float(head[0]["alarm_rate"]) if head else float("nan")
                cells = []
                for d in deltas_:
                    hit = [r for r in e6 if r["proxy"] == p_ and r["method"] == mname
                           and r["drift_target"] != "（变前）" and r["delta"] == d]
                    if not hit:
                        cells.append("—")
                        continue
                    r0 = hit[0]
                    em = r0["edd_mean"]
                    cells.append(f"{r0['detect_rate']:.2f} / "
                                 f"{'—' if is_inf(em) else f'{float(em):.1f}'}")
                cn = next((r["method_cn"] for r in e6 if r["method"] == mname), mname)
                lines.append(f"| {cn} | {far:.3f} | " + " | ".join(cells) + " |")
        if e6k:
            lines += ["", "### CUSUM 的「假设漂移幅度」有多要紧（本文方法没有这个旋钮）", "",
                      "经典 CUSUM 需要事先知道变后均值移动了多少（参考偏移 k，最优约为真实位移的一半）。"
                       "下表按相同目标误报率扫描 k，并单独核对实际误报率；每一行是一种「假设」，列是真实的 Δ；"
                      "最后一行是本文方法（按 λ 网格混合，不需要这个先验）。", ""]
            for p_ in [p for p in dict.fromkeys(r["proxy"] for r in e6k) if p]:
                sub = [r for r in e6k if r["proxy"] == p_]
                deltas_k = sorted({r["delta"] for r in sub if r.get("delta") != ""})
                ks_ = sorted({float(r["assumed_k"]) for r in sub
                              if r.get("assumed_k") not in ("", None)
                              and not is_inf(r.get("assumed_k"))})
                lines += ["", f"**{p_}**", "",
                          "| 假设的位移 | " + " | ".join(f"Δ={d} EDD" for d in deltas_k)
                          + " |", "|---" * (len(deltas_k) + 1) + "|"]
                for k_ in ks_:
                    cells = []
                    for d in deltas_k:
                        hit = [r for r in sub if r["method"] == f"cusum@{k_:g}"
                               and r["delta"] == d and not is_inf(r["edd_mean"])]
                        cells.append(f"{float(hit[0]['edd_mean']):.1f}" if hit else "—")
                    lines.append(f"| 参考偏移 k={k_:g}σ | " + " | ".join(cells) + " |")
                cells = []
                for d in deltas_k:
                    hit = [r for r in sub if r["method"] == "edetector" and r["delta"] == d
                           and not is_inf(r["edd_mean"])]
                    cells.append(f"{float(hit[0]['edd_mean']):.1f}" if hit else "—")
                lines.append("| **本文 e-detector（无需指定）** | " + " | ".join(cells) + " |")
                spread = []
                for d in deltas_k:
                    vals = [float(r["edd_mean"]) for r in sub if r["method"].startswith("cusum@")
                            and r["delta"] == d and not is_inf(r["edd_mean"])]
                    if len(vals) >= 2 and min(vals) > 0:
                        spread.append(f"Δ={d}：{max(vals) / min(vals):.1f}×")
                if spread:
                    lines.append(f"- 同一 Δ 下，只因为「假设的位移」选得不同，CUSUM 的 EDD 就能相差 "
                                 f"{'；'.join(spread)}。")
        lines += ["",
                  "- 读表时先看独立流上的实际误报率；若差异明显，EDD 不是严格的同误报比较。",
                  "- 本轮仅比较标量代理序列上的简化检测器；各方法的原论文系统和输入对象没有完整复刻。",
                  "- 口径局限：相关工作的原始对象是嵌入/多维分布，这里把它们复现在**同一标量代理序列**上，"
                  "属于「同一输入下的检测方法对比」，不是原论文系统的完整复刻。"]
    # E7：非 iid 稳健性
    e7 = [r for r in records if r.get("experiment") == "e7_dependence"]
    if e7:
        sec_no = next_sec()
        alphas = sorted({k.split("_")[-1] for k in e7[0] if k.startswith("alarm_rate_base_")},
                        key=lambda x: -float(x))
        lines += ["", f"## {sec_no}、E7：合成序列相关性的压力测试", "",
                  "> 理论前提是 E[X_n|历史] ≤ m。窗口均值超过 m 只是描述性统计，"
                  "独立且满足该前提的序列也可能出现这种窗口，不能以它判定前提是否成立。",
                  "> 块生成器重采样的是打乱查询后形成的重叠滑窗，相关性部分来自滑窗重叠；"
                  "AR(1) 生成器保持该滑窗池的边缘分布。两者都不是线上时间相关性的实测。", ""]
        for a in alphas:
            lines += [f"### 报警率（α={a}，阈值 {1.0 / float(a):.0f}，T={e7[0].get('T')} 步）", "",
                      "| 生成器 | 参数 | 自相关 | 局部超界(m) | T 内报警率(m) | T 内报警率(m') | 局部超界(m') | 截尾均值(m/m') |",
                      "|---|---|---|---|---|---|---|---|"]
            for r in e7:
                lines.append(
                    f"| {r['generator']} | {r['param']}={r['param_value']:g} | {float(r['acf1']):+.3f} | "
                    f"{float(r['local_exceed_rate']):.3f} | "
                    f"**{float(r.get(f'alarm_rate_base_{a}', 0)):.3f}** | "
                    f"{float(r.get(f'alarm_rate_aware_{a}', 0)):.3f} | "
                    f"{float(r['local_exceed_rate_aware']):.3f} | "
                    f"{r.get(f'arl_restricted_mean_base_{a}', '—')} / "
                    f"{r.get(f'arl_restricted_mean_aware_{a}', '—')} |")
        lines += ["", f"（名义上界 m={e7[0].get('m_base')}，相依感知上界 m'={e7[0].get('m_aware')}；"
                      f"β 口径：α_edd=1e-3 下的检出与延迟见下表）", "",
                  "| 生成器 | 参数 | 检出率（m / m'） | EDD（m → m'） |", "|---|---|---|---|"]
        for r in e7:
            lines.append(f"| {r['generator']} | {r['param']}={r['param_value']:g} | "
                         f"{r.get('detect_rate_base')} / {r.get('detect_rate_aware')} | "
                         f"{r.get('edd_base')} → {r.get('edd_aware')} |")
        lines += ["",
                  "- 这些有限 T 的报警率显示对相依结构的敏感性；完整 ARL 仍是未观测尾部在内的平均首次报警时间。",
                  "- m' 是校准段滑窗均值给出的经验余量，不能恢复理论保证；应在独立时间序列上验证。",
                  "- E7 的变点仍直接注入代理值，未测量 RAG 链路故障对质量真值的影响。"]
    lines += [
        "",
        f"- 示例轨迹（可直接画图，已含阈值/变点/报警步数列）：`{os.path.basename(series_csv)}`；"
        f"逐配置结果：`{os.path.basename(out_csv)}`。",
        f"- 读表提示：已报警流均值显示「—」表示 T={args.T} 步内未观察到报警；完整 ARL 未估出。",
        "",
        "> 口径说明：原始记录在切分前打乱，是因为 v1→v2 批次行序不是时间序；"
        "校准与检验各自在段内做滑窗。变前流从检验池作块重采样，"
        "漂移默认直接污染代理值，Δ 是代理值的目标均值变化。"
        "真实线上日志应按时间切分并用 `--no-shuffle`；真实质量变点还需链路级故障注入与真值核验。"]
    open(report, "w", encoding="utf-8").write("\n".join(lines))
    return report


if __name__ == "__main__":
    sys.exit(main())
