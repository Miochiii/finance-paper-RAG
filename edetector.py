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
      - hoeffding : m = 均值 + sqrt(log(1/δ)/(2n)) —— 有限样本上界（概率 1−δ 成立），
                    但对**稀有事件太松**（p̂=0.02、n=58 时给到 0.18，约 9 倍均值）；
      - binom     : m = 二项比例的 Clopper-Pearson 上界（精确、保守但紧），
                    二元/稀有事件代理应当用这个（同例给到约 0.09，比 Hoeffding 紧一倍）；
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
        n = int(a.size)
        k = int(np.sum(a >= 0.5))                    # 二元指示量的正例数
        m = float(stats.beta.ppf(1.0 - delta, k + 1, max(1, n - k)))
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
        return np.full(x.shape, 0.5)
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
    """变前（同分布）流的 ARL：报警时间均值/中位数 + 截尾比例。"""
    M = build_detectors(x_streams, m, lams, how)
    times = first_alarm(M, 1.0 / alpha)
    censored = int((times == 0).sum())
    valid = times[times > 0]
    return {
        "n_reps": int(x_streams.shape[0]),
        "T": int(x_streams.shape[1]),
        "alarm_rate": float(1.0 - censored / x_streams.shape[0]),
        "arl_mean": float(valid.mean()) if valid.size else float("inf"),
        "arl_median": float(np.median(valid)) if valid.size else float("inf"),
        "censored": censored,
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
        ok = times > 0
        out.append({"fuse": f"单指标#{k}", "k": k,
                    "alarm_rate": float(ok.mean()),
                    "arl_mean": float(times[ok].mean()) if ok.any() else float("inf"),
                    "arl_median": float(np.median(times[ok])) if ok.any() else float("inf"),
                    "n_comp": int(len(lams))})
    for how, label in (("mix", "全指标混合"), ("max", "全指标取最大"), ("min", "全指标取最小")):
        M = build_multi_detector(streams, ms, lams, how)
        times = first_alarm(M, 1.0 / alpha)
        ok = times > 0
        out.append({"fuse": label, "k": -1,
                    "alarm_rate": float(ok.mean()),
                    "arl_mean": float(times[ok].mean()) if ok.any() else float("inf"),
                    "arl_median": float(np.median(times[ok])) if ok.any() else float("inf"),
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


def _edd_stats(M: np.ndarray, alpha: float, at: int, T_post: int,
               horizon: int = 0) -> Dict:
    """从 M 序列算 EDD 相关统计（变前误报、检出率、EDD 均值/中位/截尾）。

    `horizon`（>0 时）= 只把「变点后 H 步内报警」算作检出。这很关键：不设视界时，
    零假设侧偶发的误报（例如某个检测器对该漂移本来就无感，只是碰巧在 500 步后响了一次）
    会被记成「检出」，于是出现「检出率 0.01、EDD 493 步」这种假象，还会把对照矩阵的
    对角线弄脏。超出视界的报警单列 `late_alarm_rate`，不隐瞒、也不算检出。
    """
    times = first_alarm(M, 1.0 / alpha)
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
                    help="进入检测器前的滑窗聚合长度（二值/稀有代理建议 20，即监控'每20条的拒答率'）")
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
    for p in proxies:
        raw = series_from_rows(rows, p)
        if raw.size < 20:
            print(f"  [跳过] {p}: 有效样本不足（{raw.size}）")
            continue
        raw = rolling_mean(raw, args.window) if args.window > 1 else raw
        if args.shuffle:
            # 标注集的行序 = v1 批次在前、v2 批次在后，不是时间序；
            # 直接按行序切"校准段/检验段"会把批次差异误当成分布漂移
            # （实测：校准得的 m 反而小于检验段均值，违反 μ ≤ m 前提）。
            raw = rng.permutation(raw)
        n_calib = max(8, int(raw.size * args.calib_frac))
        u = fit_unit(raw[:n_calib], raw)
        # **方向统一**：有界构造只能检出"数值超过 m"（上升），所以把"下降=退化"的代理翻正
        # （如证据集中度 HHI、检索一致性：它们变小才是变差）。
        # 不做这一步的话，这类代理即使真的退化了也永远检不出来（实测检出率恒为 0）。
        if DEGRADE_DIRECTION.get(p, +1) < 0:
            u = 1.0 - u
        if args.binarize > 0:
            u = (u >= args.binarize).astype(float)
        units[p] = u
        calib_idx[p] = n_calib
        extra = f"｜二元化(≥{args.binarize}) 正例率 {u.mean():.3f}" if args.binarize > 0 else ""
        print(f"  {p}: 原始 {raw.size} 条 → 均值 {u.mean():.3f}，范围 {u.min():.3f}~{u.max():.3f}"
              f"（尺度按前 {n_calib} 条定{('，已打乱顺序' if args.shuffle else '，按原始行序')}"
              f"{'，已按退化方向翻正' if DEGRADE_DIRECTION.get(p, +1) < 0 else ''}）{extra}")
    if not units:
        print("[错误] 没有可用代理")
        return 1

    records: List[Dict] = []
    # ---- 实验 1：ARL（变前同分布，应 ≥ 1/α；max 融合应显著低于）----
    print("\n[实验 1] 变前 ARL 验证（目标 ARL ≥ 1/α）")
    for strategy in strategies:
        for p, u in units.items():
            n_calib = calib_idx[p]
            m = estimate_m(u[:n_calib], strategy)
            if m < args.m_floor:
                m = args.m_floor
                print(f"  [提示] {p} 的 m 估计值过低（校准窗口内几乎没有事件），"
                      f"已抬到下界 {args.m_floor}；稀有事件代理建议 --window 20 做窗口聚合")
            pool = u[n_calib:] if u.size > n_calib else u
            # 合法性前置检查：ARL ≥ 1/α 只在"变前 μ ≤ m"时成立；
            # 若变前数据均值已超过 m（采样估计不足/分布漂移），保证不适用，必须显式标出。
            m_ok = bool(pool.mean() <= m)
            if m >= 1.0 - 1e-9:
                print(f"  [警告] {p}/{strategy}: m≈1.0 → L≤1 恒成立，检测器**永不可能报警**（无功效）；"
                      f"饱和型代理请加 --binarize 0.5 转成 0/1 指示量")
            elif not m_ok:
                print(f"  [警告] {p}/{strategy}: m={m:.3f} < 变前数据均值 {pool.mean():.3f} "
                      f"→ 违反 μ ≤ m 前提，ARL 保证不适用（这正是 E5 要展示的权衡）")
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
            direction = DEGRADE_DIRECTION.get(p, +1)
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
    if getattr(args, "e1", True):
        e1_proxies = [p.strip() for p in (args.e1_proxies or "").split(",") if p.strip()]
        e1_proxies = [p for p in e1_proxies if p in DEGRADE_DIRECTION]
        X, qids = series_matrix(rows, e1_proxies)
        if X.shape[0] < 30:
            print(f"\n[实验 3/E1] 跳过：可用于联合分析的题数不足（{X.shape[0]}）")
        else:
            print(f"\n[实验 3/E1] 单指标 vs 多指标混合：{e1_proxies}（{X.shape[0]} 题对齐）")
            n_calib_e1 = max(8, int(X.shape[0] * args.calib_frac))
            if args.shuffle:
                perm = rng_e1.permutation(X.shape[0])
                X = X[perm]
            Xn = np.column_stack([fit_unit(X[:n_calib_e1, k], X[:, k])
                                  for k in range(X.shape[1])])
            for k, p in enumerate(e1_proxies):
                if DEGRADE_DIRECTION.get(p, +1) < 0:      # 同样要把"下降=退化"的代理翻正
                    Xn[:, k] = 1.0 - Xn[:, k]
            if args.window > 1:
                Xn = np.column_stack([rolling_mean(Xn[:, k], args.window)
                                      for k in range(Xn.shape[1])])
            if args.binarize > 0:
                Xn = (Xn >= args.binarize).astype(float)
            ms_e1 = [max(estimate_m(Xn[:n_calib_e1, k], strategies[0]), args.m_floor)
                     for k in range(Xn.shape[1])]
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

    # ---- 输出 ----
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_csv = os.path.join(RESULTS_DIR, f"edetector_{stamp}.csv")
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
    lines = [
        "# 最小 e-detector 验证（RAG 质量漂移的序贯检测）",
        "",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M')}；α={args.alpha} → 阈值 1/α={thr:.0f}",
        f"- 代理矩阵：`{os.path.basename(matrix_path)}`；代理：{'、'.join(proxies)}",
        f"- 流长 T={args.T}，ARL 重复 {args.reps} 次，块自助（block=20）保留短程自相关",
        "",
        "## 一、变前 ARL（合法性检验：ARL 应 ≥ 1/α）",
        "",
        "> 前置条件：**变前 μ ≤ m**。`m ≥ 变前均值` 列为 0 表示该配置已违反前提，"
        "此时 ARL 保证不适用（这正是「m 估计策略」要权衡的地方）。",
        "",
        "| 代理 | m 策略 | m | 变前均值 | m ≥ 均值 | 融合 | ARL 均值 | ARL 中位 | 报警率 | 结论 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(arl, key=lambda x: (x["proxy"], x["m_strategy"], x["fuse"])):
        ok_theory = r.get("m_ge_pool_mean", 1) == 1
        am = r["arl_mean"]
        amd = r.get("arl_median", am)
        if r.get("alarm_rate", 0) == 0:
            verdict = "⬜ 无功效（T 步内从未报警）"
        elif not ok_theory:
            verdict = "⚠️ 违反 μ ≤ m 前提"
        elif not is_inf(am) and float(am) >= thr:
            verdict = "✅ 达标"
        elif r["fuse"] == "max":
            verdict = "❌ 误报膨胀（Remark 3.1 预期）"
        else:
            verdict = "⚠️ 低于 1/α"
        lines.append(f"| {r['proxy']} | {r['m_strategy']} | {r['m']} | {r.get('pool_mean', '')} | "
                     f"{'是' if ok_theory else '否'} | {r['fuse']} | "
                     f"{'—' if is_inf(am) else f'{float(am):.1f}'} | "
                     f"{'—' if is_inf(amd) else f'{float(amd):.1f}'} | "
                     f"{r['alarm_rate']:.2f} | {verdict} |")
    if arl and not any(r.get("alarm_rate", 0) > 0 for r in arl):
        lines += ["",
                  "> 本节 ARL 全部为「—」＝ T 步内从未报警（ARL > T）：**误报侧没有分辨率**，"
                  "并非检测器失效。原因通常是三者同时偏保守：m 取 q85（远高于均值）、阈值 1/α 大、"
                  "变前流本身平稳。此时 ARL ≥ 1/α **平凡成立**（0 次误报），"
                  "要看功效请转第二节（EDD），或扫描更紧的 m（`--m-strategy mean,q60,q85,q100`）"
                  "让误报侧出现可比较的差别。"]
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
              f"- ARL 均值（仅统计报警的流，被截尾的流不计入，故仅供参考）：mix={_avg(arl, 'mix'):.1f}、"
              f"max={_avg(arl, 'max'):.1f}、min={_avg(arl, 'min'):.1f}、single={_avg(arl, 'single'):.1f}",
              f"- **无功效配置 {len(no_power)}/{len(arl)}**：m 太松（尤其 m≈1.0 时 L≤1 恒成立）或代理饱和"
              f"（归一化上界顶在 1.0）都会导致「永不报警」。饱和型代理请加 `--binarize 0.5` 转成 0/1 指示量。",
              f"- **m 估计策略（E5）**：m 越保守 ARL 越安全、EDD 越长；用样本均值当上界会违反 μ ≤ m 前提，"
              f"实测 ARL 不足 1/α（见第一节「m ≥ 均值」列）。",
              f"- **α 的工程含义**：ARL ≥ 1/α，α={args.alpha} 表示平均每 {thr:.0f} 步可能误报一次；"
              f"EDD 实验用 α={args.alpha_edd}（ARL ≥ {1 / args.alpha_edd:.0f} 步）。",
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
                  "> 「全指标混合」按命题 2.3 做凸组合（权重和为 1），因此合法性仍然成立。", "",
                  "| 配置 | 分量数 | ARL 均值 | 报警率 |", "|---|---|---|---|"]
        for r in e1_arl:
            am = r["arl_mean"]
            lines.append(f"| {r['fuse']}{('（' + r['proxy'] + '）') if r.get('k', -1) >= 0 else ''} | "
                         f"{r.get('n_comp', '')} | {'—' if is_inf(am) else f'{float(am):.1f}'} | "
                         f"{r['alarm_rate']:.2f} |")
        # 变前 ARL 全都不报警时，必须解释清楚：这不是「检测器没用」，而是上界保守的代价，
        # 否则读者会以为 ARL 保证没生效（保证只在报警时才有分辨率）。
        if not any(r["alarm_rate"] > 0 for r in e1_arl):
            lines += ["",
                      "> 变前 ARL 全部为「—」＝ T 步内从未报警（ARL > T）：这说明**误报侧根本没有分辨率**，"
                      "而不是检测器失效。原因是三者同时偏保守：m 取 q85（远高于均值）、阈值 1/α 很大、"
                      "变前流本身平稳。ARL ≥ 1/α 的保证此时**平凡成立**（0 次误报），"
                      "E1 的对照组要看**EDD 侧**；若要让 ARL 有分辨率，应回到 m 策略扫描"
                      "（`--m-strategy mean,q60,q85,q100`）——那里可见 mix 的误报率明显低于 max。"]
        if e1_edd:
            hz = e1_edd[0].get("edd_horizon", 0)
            lines += ["", f"| 配置 | 漂移目标 | Δ | 检出率（H={hz or '∞'} 步内） | 超时报警率 | EDD 均值 | EDD 中位 |",
                      "|---|---|---|---|---|---|---|"]
            for r in e1_edd:
                em = r["edd_mean"]
                ed = r["edd_median"]
                lines.append(f"| {r['fuse']}{('（' + r['proxy'] + '）') if r.get('k', -1) >= 0 else ''} | "
                             f"{r.get('drift_target', '')} | {r['delta']} | {r['detect_rate']:.2f} | "
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
            for mrow in sorted(mix_rows, key=lambda x: (x["delta"], str(x.get("drift_target")))):
                tgt = mrow.get("drift_target", "")
                if is_inf(mrow["edd_mean"]):
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
                cand = matched or same
                good = [r for r in cand if r["detect_rate"] >= 0.5 and not is_inf(r["edd_mean"])]
                if good:
                    ref = min(good, key=lambda r: float(r["edd_mean"]))
                    ratio = float(ref["edd_mean"]) / float(mrow["edd_mean"])
                    lines.append(
                        f"- Δ={mrow['delta']}｜{tgt}：混合 EDD={float(mrow['edd_mean']):.1f} 步 vs "
                        f"盯对指标的单指标（{ref['proxy']}）{float(ref['edd_mean']):.1f} 步 → "
                        f"混合慢 **{ratio:.2f}×**（对手是「事先知道漂移打在哪」的 oracle 基线）")
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
                    "- **鲁棒性口径**（「检测器 × 漂移目标」组合的检出个数，按漂移幅度分开看）——"
                    + "；".join(parts) + "。"
                    "单指标一旦「打偏」不是变慢，而是像没装一样失效（检出率 0.00）；"
                    "混合只需其中**任意一个**指标被漂移触及即可报警，所以对漂移类型不敏感。")
    lines += [
        "",
        f"- 示例轨迹（可直接画图，已含阈值/变点/报警步数列）：`{os.path.basename(series_csv)}`；"
        f"逐配置结果：`{os.path.basename(out_csv)}`。",
        f"- 读表提示：ARL 显示「—」= 在 T={args.T} 步内从未报警（ARL > T，说明配置过保守或无功效，不是错误）。",
        "",
        "> 口径说明：变前流由**块自助重采样**构造（分块内近似平稳），漂移注入默认用"
        "**污染模型**（一部分查询变最差状态，期望均值漂移 = Δ，不饱和）；"
        "切分校准段/检验段前会**打乱顺序**——标注集的行序是 v1→v2 两个批次、不是时间序，"
        "按行序切会把批次差异误当成分布漂移（实测会直接违反 μ ≤ m 前提）。"
        "接入真实线上日志（带时间戳）后用 `--no-shuffle` 才符合时间序语义。"]
    open(report, "w", encoding="utf-8").write("\n".join(lines))
    return report


if __name__ == "__main__":
    sys.exit(main())
