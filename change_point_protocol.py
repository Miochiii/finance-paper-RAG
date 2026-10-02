# -*- coding: utf-8 -*-
"""诊断变点实验的有限视界误报口径与重叠窗口影响。

这里的 Bernoulli 基准是已知生成机制的单元级模拟。其 p 默认取代理矩阵的
拒答比例，只用于说明窗口结构；重采样或参数模拟不构成真实 RAG 的独立验证。
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.stats import binomtest

import edetector as ed


MODES = ("raw", "overlap", "disjoint")


def aggregate_queries(raw: np.ndarray, mode: str, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """返回监控值和每个值对应的原始查询序号（从 1 开始）。"""
    x = np.atleast_2d(np.asarray(raw, dtype=float))
    if x.ndim != 2 or window < 1 or x.shape[1] < window:
        raise ValueError("输入须为二维序列，且查询长度不少于窗口长度")
    if mode == "raw":
        return x, np.arange(1, x.shape[1] + 1)
    if mode == "overlap":
        c = np.pad(np.cumsum(x, axis=1), ((0, 0), (1, 0)))
        return (c[:, window:] - c[:, :-window]) / window, np.arange(window, x.shape[1] + 1)
    if mode == "disjoint":
        n = x.shape[1] // window
        return x[:, :n * window].reshape(x.shape[0], n, window).mean(axis=2), \
            np.arange(1, n + 1) * window
    raise ValueError(f"未知聚合方式: {mode}")


def alarm_queries(scores: np.ndarray, threshold: float, query_steps: np.ndarray) -> np.ndarray:
    times = ed.first_alarm(scores, threshold)
    return np.where(times > 0, query_steps[np.maximum(times - 1, 0)], 0)


def exact_interval(alarms: int, n: int) -> Tuple[float, float]:
    ci = binomtest(int(alarms), int(n)).proportion_ci(confidence_level=0.95, method="exact")
    return float(ci.low), float(ci.high)


def null_summary(alarm_at: np.ndarray, horizon: int) -> Dict[str, float]:
    t = np.asarray(alarm_at, dtype=int)
    k = int(np.count_nonzero(t))
    lo, hi = exact_interval(k, t.size)
    return {
        "n_eval": int(t.size), "false_alarms": k,
        "far_h": k / t.size, "far_ci_low": lo, "far_ci_high": hi,
        "alarm_only_mean_queries": float(t[t > 0].mean()) if k else float("inf"),
        "restricted_mean_queries": float(np.where(t > 0, t, horizon + 1).mean()),
    }


def drift_summary(alarm_at: np.ndarray, change_at: int, post_horizon: int) -> Dict[str, float]:
    t = np.asarray(alarm_at, dtype=int)
    pre = (t > 0) & (t <= change_at)
    hit = (t > change_at) & (t <= change_at + post_horizon)
    delay = t[hit] - change_at
    return {
        "n_drift": int(t.size), "pre_alarm_rate": float(pre.mean()),
        "detect_rate": float(hit.mean()),
        "detect_rate_given_no_pre": float(hit.sum() / (~pre).sum()) if (~pre).any() else float("nan"),
        "edd_queries": float(delay.mean()) if delay.size else float("inf"),
    }


def choose_threshold(calibration_scores: np.ndarray, target_far: float) -> float:
    """仅用校准流选阈值；等于经验分位数的轨迹不计作报警。"""
    if not 0 < target_far < 1:
        raise ValueError("target_far 必须在 0 与 1 之间")
    maxima = np.asarray(calibration_scores).max(axis=1)
    q = float(np.quantile(maxima, 1 - target_far, method="higher"))
    return float(max(1.0 + 1e-9, np.nextafter(q, np.inf)))


def overlapping_conditional_exceed(raw: np.ndarray, p: float, m: float, window: int) -> float:
    """iid Bernoulli 假设下，已知前 w-1 个原始值时 E[窗口值|历史]>m 的比例。"""
    x = np.atleast_2d(np.asarray(raw, dtype=float))
    if window < 2:
        return float(p > m)
    c = np.pad(np.cumsum(x, axis=1), ((0, 0), (1, 0)))
    known = c[:, window - 1:-1] - c[:, :-(window)]
    return float(np.mean((known + p) / window > m))


def matrix_audit(path: Path) -> Dict:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "ans_refusal" not in rows[0]:
        raise ValueError("代理矩阵为空或缺少 ans_refusal 列")
    by_batch = {}
    for row in rows:
        label = row.get("batch", "")
        entry = by_batch.setdefault(label, {"queries": 0, "refusals": 0})
        entry["queries"] += 1
        entry["refusals"] += int(float(row["ans_refusal"]) >= 0.5)
    n = len(rows)
    positives = sum(x["refusals"] for x in by_batch.values())
    return {"queries": n, "unique_qids": len({r.get("qid", "") for r in rows}),
            "refusals": positives, "p_empirical": positives / n, "by_batch": by_batch}


def run_protocol(matrix: Path, output_dir: Path, report_dir: Path, *,
                 seed: int = 20260927, horizon: int = 1000, window: int = 20,
                 m: float = 0.05, target_far: float = 0.01,
                 n_calibration: int = 500, n_eval: int = 1000, n_drift: int = 500,
                 change_at: int = 300, post_horizon: int = 200,
                 post_p: float = 0.20) -> Tuple[Path, Path]:
    audit = matrix_audit(matrix)
    p = audit["p_empirical"]
    if not (0 < p < m < 1 and p < post_p < 1):
        raise ValueError("此基准需要 0 < 经验拒答率 < m < 1 且变后概率高于变前概率")
    if horizon < window or not 0 < change_at < horizon or post_horizon < 1:
        raise ValueError("窗口、变点与视界参数不合法")
    generators = [np.random.default_rng([seed, k]) for k in (1, 2, 3)]
    raw_cal = (generators[0].random((n_calibration, horizon)) < p).astype(float)
    raw_eval = (generators[1].random((n_eval, horizon)) < p).astype(float)
    u = generators[2].random((n_drift, horizon))
    raw_drift = (u < np.where(np.arange(horizon) < change_at, p, post_p)).astype(float)
    lams = ed.lambda_grid(m)
    rows = []
    for mode in MODES:
        x_cal, query_steps = aggregate_queries(raw_cal, mode, window)
        x_eval, _ = aggregate_queries(raw_eval, mode, window)
        x_drift, _ = aggregate_queries(raw_drift, mode, window)
        s_cal = ed.build_detectors(x_cal, m, lams, "mix")
        threshold = choose_threshold(s_cal, target_far)
        s_eval = ed.build_detectors(x_eval, m, lams, "mix")
        s_drift = ed.build_detectors(x_drift, m, lams, "mix")
        for label, thr in (("fixed_1_over_alpha_edd", 1000.0),
                           ("empirical_far_target", threshold)):
            null = null_summary(alarm_queries(s_eval, thr, query_steps), horizon)
            drift = drift_summary(alarm_queries(s_drift, thr, query_steps),
                                  change_at, post_horizon)
            rows.append({"mode": mode, "threshold_rule": label, "threshold": thr,
                         "target_far_h": target_far if label == "empirical_far_target" else "",
                         "queries_per_value": window if mode != "raw" else 1,
                         "monitoring_values": x_eval.shape[1],
                         "p_iid": p, "m": m, **null, **drift})
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "change_point_protocol_iid.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    exceed = overlapping_conditional_exceed(raw_eval, p, m, window)
    report_path = report_dir / "变点协议诊断_iid.md"
    lines = ["# 变点协议诊断：有限视界误报与窗口结构", "",
             f"- 输入矩阵：`{matrix.name}`，查询 {audit['queries']} 条，其中拒答 {audit['refusals']} 条；"
             f"经验拒答比例 {p:.4f}。此比例仅用于设定模拟参数。",
             f"- 生成机制：iid Bernoulli({p:.4f})，m={m:.3f}，窗口 {window} 条，"
             f"每条流 {horizon} 条查询；变点在第 {change_at} 条后，变后拒答概率 {post_p:.2f}。",
             f"- 阈值校准流 {n_calibration} 条、正常评估流 {n_eval} 条、漂移流 {n_drift} 条；"
             "三批模拟使用独立随机数。各模式使用配对的原始查询流。", "",
             "## 样本覆盖", "",
             "| 批次 | 查询 | 拒答 |", "|---|---:|---:|"]
    for batch, counts in audit["by_batch"].items():
        lines.append(f"| {batch} | {counts['queries']} | {counts['refusals']} |")
    lines += ["", f"- 重叠窗口的逐时条件均值超过 m 的模拟时间比例：{exceed:.3f}。"
              "计算使用已知前 w−1 条原始查询与 iid 参数 p；原始查询和不重叠窗口在该生成机制下条件均值为 p≤m。",
              "- 这是已知分布的结构诊断，不能验证当前 RAG 查询分布的条件均值前提。", "",
              "## 同一查询视界的配对比较", "",
              "| 输入 | 阈值 | 实测 FAR_H（95% 精确区间） | 变点前报警 | 检出率 | 无提前报警时检出率 | EDD（查询数） |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['mode']} / {row['threshold_rule']} | {row['threshold']:.1f} | "
                     f"{row['far_h']:.3f} [{row['far_ci_low']:.3f}, {row['far_ci_high']:.3f}] | "
                     f"{row['pre_alarm_rate']:.3f} | {row['detect_rate']:.3f} | "
                     f"{row['detect_rate_given_no_pre']:.3f} | "
                     f"{row['edd_queries']:.1f} |")
    lines += ["", "`FAR_H` 是 H 条查询内至少一次报警的比例；它与 ARL 是不同指标。"
              "检出率只计变点后指定视界内的首次报警，提前报警单独列出。",
              "经验阈值只在校准流上选择，表中误报用另一批流计算；它仍共享同一个由 166 条查询估出的 p。",
              "当前只有 5 次真实拒答，不能用这组模拟的窄置信区间推断真实线上误报率。",
              "检测统计量计算时封顶为 1e12；重叠窗口所需的经验阈值已接近该封顶值，"
              "这一路径的性能高度依赖截断实现，不应据此声称可部署。",
              "原始查询、重叠窗口和不重叠窗口的报警时间都换算到查询数；不重叠窗口监控点更少。", ""]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, report_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path(ed.RESULTS_DIR))
    ap.add_argument("--report-dir", type=Path, default=Path(ed.LOG_DIR))
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--horizon", type=int, default=1000)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--m", type=float, default=0.05)
    ap.add_argument("--target-far", type=float, default=0.01)
    ap.add_argument("--n-calibration", type=int, default=500)
    ap.add_argument("--n-eval", type=int, default=1000)
    ap.add_argument("--n-drift", type=int, default=500)
    ap.add_argument("--change-at", type=int, default=300)
    ap.add_argument("--post-horizon", type=int, default=200)
    ap.add_argument("--post-p", type=float, default=0.20)
    args = ap.parse_args()
    csv_path, report_path = run_protocol(
        args.matrix, args.output_dir, args.report_dir, seed=args.seed,
        horizon=args.horizon, window=args.window, m=args.m,
        target_far=args.target_far, n_calibration=args.n_calibration,
        n_eval=args.n_eval, n_drift=args.n_drift, change_at=args.change_at,
        post_horizon=args.post_horizon, post_p=args.post_p)
    print(csv_path)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
