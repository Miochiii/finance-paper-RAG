"""Post hoc method comparison on frozen fresh streams, with calibration-only tuning.

This comparison is exploratory: the held-out results were seen before choosing
the additional methods. Zero-refusal calibration cannot establish low FAR.
"""
import argparse
import math
from pathlib import Path

import numpy as np

import edetector as ed
from rag_fresh_change_experiment import (digest, dump, empirical_threshold, load,
    matrices, observations, summarize_alarms, verified_inputs, write_csv)


def consecutive_score(x, length=2):
    x = np.atleast_2d(np.asarray(x, float))
    out = np.zeros_like(x)
    for t in range(length-1, x.shape[1]):
        out[:, t] = np.prod(x[:, t-length+1:t+1], axis=1)
    return out


def window_count(x, window=5):
    x = np.atleast_2d(np.asarray(x, float))
    out = np.zeros_like(x)
    for t in range(window-1, x.shape[1]):
        out[:, t] = x[:, t-window+1:t+1].sum(axis=1)
    return out


def bernoulli_cusum(x, p0, p1):
    if not 0 < p0 < p1 < 1:
        raise ValueError("requires 0 < p0 < p1 < 1")
    x = np.atleast_2d(np.asarray(x, float))
    if not np.all((x == 0) | (x == 1)):
        raise ValueError("Bernoulli CUSUM requires binary observations")
    increments = x * math.log(p1/p0) + (1-x) * math.log((1-p1)/(1-p0))
    previous = np.zeros(x.shape[0])
    out = np.zeros_like(x)
    for t in range(x.shape[1]):
        previous = np.maximum(0, previous + increments[:, t])
        out[:, t] = previous
    return out


def page_hinkley(x, delta=0.05):
    """One-sided PH; causal running mean includes the current observation."""
    x = np.atleast_2d(np.asarray(x, float))
    total = np.zeros(x.shape[0])
    cumulative = np.zeros(x.shape[0])
    minimum = np.zeros(x.shape[0])
    out = np.zeros_like(x)
    for t in range(x.shape[1]):
        total += x[:, t]
        cumulative += x[:, t] - total/(t+1) - delta
        minimum = np.minimum(minimum, cumulative)
        out[:, t] = cumulative - minimum
    return out


def calibration_threshold(scores, target=0.1):
    maxima = np.asarray(scores).max(axis=1)
    rank = math.ceil((len(maxima)+1)*(1-target))
    if rank > len(maxima):
        raise ValueError("insufficient calibration streams")
    return max(1e-12, float(np.nextafter(np.sort(maxima)[rank-1], np.inf)))


def compare(source, output):
    manifest, _, _ = verified_inputs(source)
    cfg = manifest["config"]
    lock = load(source / "calibration_lock.json")["frozen"]
    raw = {p: matrices(observations(source, p))["ans_refusal"]
           for p in ("threshold", "normal_test", "fault_test")}
    m = lock["bounds"]["ans_refusal"]
    methods = [
        ("first_refusal", lambda x: x, "fixed", 1.0, "first positive"),
        ("two_consecutive", consecutive_score, "fixed", 1.0, "2 consecutive positives"),
        ("window5_two", window_count, "fixed", 2.0, "at least 2 of trailing 5; starts at query 5"),
        ("window5_calibrated", window_count, "calibration", None, "trailing 5; normal-only rank threshold"),
        ("bernoulli_cusum_p1_0.10_fixed", lambda x: bernoulli_cusum(x, m, 0.10), "fixed", math.log(100), "p0=m, p1=.10"),
        ("bernoulli_cusum_p1_0.30_fixed", lambda x: bernoulli_cusum(x, m, 0.30), "fixed", math.log(100), "p0=m, p1=.30"),
        ("bernoulli_cusum_p1_0.10_calibrated", lambda x: bernoulli_cusum(x, m, 0.10), "calibration", None, "p0=m, p1=.10"),
        ("bernoulli_cusum_p1_0.30_calibrated", lambda x: bernoulli_cusum(x, m, 0.30), "calibration", None, "p0=m, p1=.30"),
        ("page_hinkley_calibrated", page_hinkley, "calibration", None, "running mean; delta=.05"),
        ("edetector_fixed_100", lambda x: ed.build_detectors(x, m, ed.lambda_grid(m), "mix"), "fixed", 100.0, "original primary configuration"),
        ("edetector_calibrated", lambda x: ed.build_detectors(x, m, ed.lambda_grid(m), "mix"), "e_calibration", None, "original normal-calibration configuration"),
    ]
    results, alarms, maxcal = [], [], []
    for name, score, rule, threshold, detail in methods:
        series = {p: score(x) for p, x in raw.items()}
        if rule == "calibration":
            threshold = calibration_threshold(series["threshold"], cfg["target_far"])
        elif rule == "e_calibration":
            threshold = empirical_threshold(series["threshold"], cfg["target_far"])
        na, fa = [ed.first_alarm(series[p], threshold) for p in ("normal_test", "fault_test")]
        results.append(dict(method=name, threshold_rule=rule, threshold=threshold, detail=detail,
                            **summarize_alarms(na, fa, cfg["horizon"], cfg["change_at"])))
        for phase, values in (("normal_test", na), ("fault_test", fa)):
            for run, at in enumerate(values):
                alarms.append(dict(method=name, phase=phase, run=run, alarm_query=int(at)))
        maxcal.append(dict(method=name, normal_calibration_max=float(series["threshold"].max()),
                           threshold=threshold))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "method_comparison.csv", results)
    write_csv(output / "method_alarm_times.csv", alarms)
    write_csv(output / "calibration_maxima.csv", maxcal)
    dump(output / "comparison_manifest.json", dict(scope="post_hoc_exploratory_same_frozen_observations",
         source=str(source.resolve()), source_manifest_sha256=digest(source / "manifest.json"),
         calibration_lock_sha256=digest(source / "calibration_lock.json"), code_sha256=digest(Path(__file__)),
         parameters_from_test=False, methods_selected_after_original_results_seen=True,
         normal_calibration_refusals=int(raw["threshold"].sum()),
         calibration_threshold_warning="all-zero null data cannot validate a rare false-positive budget"))
    lines = ["# 新生成序列上的方法对照（事后探索）", "",
        "使用同一批新生成观测，没有重排、重采样或新增API调用。追加方法是在原测试结果已知后选定，因此本表是事后探索；后续正式验证须提前冻结方法。",
        "所有方法看同一个逐请求拒答0/1序列，首次报警、正常视界30次、变点后20次、提前报警和漏检的口径相同。阈值或参考均值只使用正常校准，未用测试调参。",
        "固定规则与正常校准阈值分列。相同的经验误报并不等于已经匹配真实误报预算，不同方法也不需要使用相同数值阈值。", "",
        "|方法|阈值|正常报警/20|提前报警/10|检出/10|检出者平均延迟|受限延迟|",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        delay = "未检出" if r["mean_detected_delay"] is None else f'{r["mean_detected_delay"]:.2f}'
        lines.append(f'|{r["method"]}|{r["threshold"]:.4g}|{r["false_alarms"]}|{r["pre_alarms"]}|'
                     f'{r["detected"]}|{delay}|{r["restricted_post_delay"]:.2f}|')
    lines += ["", "## 如何解释", "",
        "正常校准300次拒答均为0，正常测试600次亦均为0。因此首次拒答规则可在本批数据同时得到0次正常报警和较短检测延迟；不能据此断言其在自然流量中更好，也不能宣称e-detector已有经验优势。",
        "多个校准阈值退化到约0，实质接近第一次正分数就报警。小样本正常校准没有覆盖稀有拒答、正常难题或无害分布变化。",
        "Bernoulli CUSUM的变前p0取已冻结的拒答上界m，变后p1预设.10/.30，不使用实际故障拒答率.70。固定阈值log(100)在本表是预定对照，不凭本实验声称理论误报保证。",
        "Page–Hinkley使用在线运行均值，与旧method_score中固定参考均值的同名实现区分；后者与固定偏移CUSUM等价，不能作为独立算法证据。",
        "全部正常0/20对应的二项95%区间上限约16.8%（假设流间独立同分布）；这批数据无法区分低误报方法。",
        "下一轮先核查隐蔽故障的质量与信号关系，再使用弱变化和含正常拒答的序列验证检测器的收益。", ""]
    (output / "method_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[6:21]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    compare(a.source, a.output)
