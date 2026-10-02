# -*- coding: utf-8 -*-
"""把配对 RAG 答案拼接成受控变点流，仅诊断可观测拒答代理。

同一批题目被重新排序多次，只反映次序敏感性，不能当成独立线上流或
用于估计真实 FAR。金标准答案和 judge 分数仅用于验证故障效果，不进检测器。
"""

import argparse
import csv
from pathlib import Path

import numpy as np

import analyze_rag_fault_pairs as paired
import change_point_protocol as protocol
import edetector as ed
from probe_validity import answer_side_proxies
from rag_refusal import explicit_refusal


MODES = ("raw", "overlap", "disjoint")


def build_streams(pairs, orders: np.ndarray, change_at: int,
                  proxy: str = "explicit"):
    if proxy not in ("explicit", "legacy"):
        raise ValueError("proxy 必须为 explicit 或 legacy")
    get_value = (lambda r: explicit_refusal(r["pred_answer"])) if proxy == "explicit" \
        else (lambda r: float(r["ans_refusal"]))
    baseline = np.array([get_value(b) for b, _ in pairs])
    fault = np.array([get_value(f) for _, f in pairs])
    normal = baseline[orders]
    changed = normal.copy()
    changed[:, change_at:] = fault[orders[:, change_at:]]
    return normal, changed


def calibration_refusals(answers: Path, used_qids: set, proxy: str):
    with answers.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    get_value = explicit_refusal if proxy == "explicit" \
        else (lambda answer: answer_side_proxies(answer)["ans_refusal"])
    values = [get_value(r["pred_answer"]) for r in rows
              if r.get("qid") not in used_qids and r.get("pred_answer")]
    if len(values) < 30:
        raise ValueError("独立于配对题目的正常校准集不足 30 条")
    return np.asarray(values, dtype=float)


def run(pairs_path: Path, answers_path: Path, output_dir: Path, *,
        seed=20260927, n_orders=200, change_at=20, window=5,
        threshold=100.0, proxy="explicit"):
    pairs = paired.load_pairs(pairs_path)
    if (len(pairs) < change_at + window or change_at < window
            or n_orders < 1 or window < 2 or threshold <= 1):
        raise ValueError("题目数、变点、窗口或阈值不合法")
    if any(not b.get("pred_answer") or not f.get("pred_answer")
           or b.get("error") or f.get("error") for b, f in pairs):
        raise ValueError("配对实验存在未完成的生成或评分")
    used_qids = {b["qid"] for b, _ in pairs}
    calibration = calibration_refusals(answers_path, used_qids, proxy)
    m = ed.estimate_m(calibration, "binom", delta=0.05)
    lams = ed.lambda_grid(m)
    rng = np.random.default_rng(seed)
    orders = np.array([rng.permutation(len(pairs)) for _ in range(n_orders)])
    normal, changed = build_streams(pairs, orders, change_at, proxy)
    rows = []
    for mode in MODES:
        norm_x, steps = protocol.aggregate_queries(normal, mode, window)
        change_x, _ = protocol.aggregate_queries(changed, mode, window)
        norm_scores = ed.build_detectors(norm_x, m, lams, "mix")
        change_scores = ed.build_detectors(change_x, m, lams, "mix")
        norm_alarm = protocol.alarm_queries(norm_scores, threshold, steps)
        change_alarm = protocol.alarm_queries(change_scores, threshold, steps)
        before = (change_alarm > 0) & (change_alarm <= change_at)
        after = (change_alarm > change_at) & (change_alarm <= len(pairs))
        rows.append({
            "mode": mode, "window": 1 if mode == "raw" else window,
            "refusal_proxy": proxy,
            "monitoring_values": norm_x.shape[1], "n_orders": n_orders,
            "threshold": threshold, "calibration_n": len(calibration),
            "calibration_events": int(calibration.sum()), "m_upper_95": m,
            "change_at_query": change_at,
            "normal_order_alarm_fraction": float(np.mean(norm_alarm > 0)),
            "changed_pre_alarm_fraction": float(before.mean()),
            "changed_post_alarm_fraction": float(after.mean()),
            "post_alarm_given_no_pre": float(after.sum() / (~before).sum()),
            "mean_delay_queries_if_detected": float(np.mean(change_alarm[after] - change_at))
              if after.any() else float("inf"),
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = output_dir / "rag_fault_change_stream.csv"
    report = output_dir / "rag_fault_change_stream.md"
    series = output_dir / "rag_fault_change_stream_one.csv"
    if any(p.exists() for p in (summary, report, series)):
        raise FileExistsError("变点流输出已存在，拒绝覆盖")
    with summary.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    raw_normal = ed.build_detectors(normal[:1], m, lams, "mix")[0]
    raw_changed = ed.build_detectors(changed[:1], m, lams, "mix")[0]
    with series.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_index", "qid", "segment",
            "normal_refusal", "changed_refusal", "normal_M_raw", "changed_M_raw"])
        writer.writeheader()
        for t, pair_index in enumerate(orders[0], 1):
            writer.writerow({"query_index": t, "qid": pairs[pair_index][0]["qid"],
                             "segment": "pre" if t <= change_at else "post",
                             "normal_refusal": normal[0, t - 1],
                             "changed_refusal": changed[0, t - 1],
                             "normal_M_raw": raw_normal[t - 1],
                             "changed_M_raw": raw_changed[t - 1]})
    lines = ["# 配对答案构造的变点流：拒答代理", "",
             f"- 使用 {len(pairs)} 道既有正式题的配对答案；第 {change_at} 道之后将上下文切换为移除正确文献的版本。",
             f"- 同一批题随机重排 {n_orders} 次（种子 {seed}）；这些顺序共享答案，不能当成 {n_orders} 条独立运行流。",
             f"- 检测器输入仅为答案侧拒答代理（{proxy}）；金标准文献、F1 和 LLM 正确性评分不进入检测器。",
             f"- 用未进入配对实验的 {len(calibration)} 条既有答案估计拒答均值："
             f"{int(calibration.sum())} 次代理命中，二项分布单侧 95% 上界 m={m:.4f}。",
             f"- 统一使用 E-detector 混合统计量、固定阈值 {threshold:g}。校准上界的独立同分布前提"
             "及检测器的逐时条件均值前提在此数据上都未验证。", "",
             "| 输入 | 正常重排中报警比例 | 变点前报警比例 | 变点后报警比例 | 无提前报警时检出比例 | 检出时平均延迟（查询数） |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        delay = (f"{r['mean_delay_queries_if_detected']:.1f}"
                 if np.isfinite(r["mean_delay_queries_if_detected"]) else "—")
        lines.append(f"| {r['mode']} | {r['normal_order_alarm_fraction']:.3f} | "
                     f"{r['changed_pre_alarm_fraction']:.3f} | "
                     f"{r['changed_post_alarm_fraction']:.3f} | "
                     f"{r['post_alarm_given_no_pre']:.3f} | {delay} |")
    lines += ["", "这些数字只描述固定题集的**顺序敏感性**，不构成真实误报率、检出率置信区间或线上检测延迟。",
              "重叠窗口重复使用查询，其条件均值可能超过 m；此列只作为工程对照。",
              "explicit 规则是本轮修订的可观测文本代理；旧规则会把 fin_009 正常答案误判为拒答。", ""]
    report.write_text("\n".join(lines), encoding="utf-8")
    return summary, report, series


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--answers", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--n-orders", type=int, default=200)
    ap.add_argument("--change-at", type=int, default=20)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=100.0)
    ap.add_argument("--proxy", choices=("explicit", "legacy"), default="explicit")
    args = ap.parse_args()
    for path in run(args.pairs, args.answers, args.output_dir, seed=args.seed,
                    n_orders=args.n_orders, change_at=args.change_at,
                    window=args.window, threshold=args.threshold, proxy=args.proxy):
        print(path)


if __name__ == "__main__":
    main()
