# -*- coding: utf-8 -*-
"""核对并汇总 RAG 正常／移除金标准证据的配对结果。

没有生成答案的离线预演只输出干预完整性。配对结果只说明机制敏感度，
不把题目顺序当成真实到达时间，也不从这些配对推断 FAR 或检测延迟。
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from rag_refusal import explicit_refusal


def _number(value):
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def load_pairs(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    grouped = {}
    for row in rows:
        qid = row.get("qid")
        condition = row.get("condition")
        if not qid or condition not in ("baseline", "remove_gold"):
            raise ValueError("缺少 qid 或存在未知条件")
        if condition in grouped.setdefault(qid, {}):
            raise ValueError(f"重复的题目和条件: {qid}/{condition}")
        grouped[qid][condition] = row
    pairs = []
    for qid, conditions in grouped.items():
        if set(conditions) != {"baseline", "remove_gold"}:
            raise ValueError(f"题目缺少配对条件: {qid}")
        base, fault = conditions["baseline"], conditions["remove_gold"]
        if (int(base["gold_hit"]) != 1 or int(fault["gold_hit"]) != 0
                or int(base["evidence_count"]) != int(fault["evidence_count"])):
            raise ValueError(f"证据移除或条数配平失败: {qid}")
        pairs.append((base, fault))
    return pairs


def document_map(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {row["qid"]: row.get("gold_sources") or ""
                for row in csv.DictReader(f)}


def cluster_bootstrap_mean(pairs, docs, field, *, seed=20260927, reps=3000):
    """以金标准文献为重采样单位，返回描述性均值差及分位区间。"""
    valid = [(b, f) for b, f in pairs if _number(b.get(field)) is not None
             and _number(f.get(field)) is not None and not b.get("error") and not f.get("error")]
    if not valid:
        return None
    groups = {}
    for base, fault in valid:
        doc = docs.get(base["qid"], base["qid"])
        groups.setdefault(doc, []).append(_number(fault[field]) - _number(base[field]))
    keys = list(groups)
    if len(keys) < 2:
        return None
    rng = np.random.default_rng(seed)
    deltas = [np.mean([d for key in rng.choice(keys, size=len(keys), replace=True)
                       for d in groups[key]]) for _ in range(reps)]
    raw = np.array([d for values in groups.values() for d in values])
    return float(raw.mean()), tuple(float(x) for x in np.quantile(deltas, [0.025, 0.975]))


def analyze(pairs_path: Path, answers_path: Path, output_dir: Path):
    pairs = load_pairs(pairs_path)
    docs = document_map(answers_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = output_dir / "paired_fault_analysis.md"
    detail = output_dir / "paired_fault_analysis.csv"
    if report.exists() or detail.exists():
        raise FileExistsError("分析输出已存在，拒绝覆盖")
    detail_rows = []
    for base, fault in pairs:
        complete = bool(base.get("pred_answer") and fault.get("pred_answer")
                        and not base.get("error") and not fault.get("error"))
        base_explicit = explicit_refusal(base.get("pred_answer", "")) if complete else None
        fault_explicit = explicit_refusal(fault.get("pred_answer", "")) if complete else None
        base_success = (int(_number(base.get("judge_corr")) >= 4 and base_explicit == 0)
                        if complete and _number(base.get("judge_corr")) is not None else None)
        fault_success = (int(_number(fault.get("judge_corr")) >= 4 and fault_explicit == 0)
                         if complete and _number(fault.get("judge_corr")) is not None else None)
        detail_rows.append({
            "qid": base["qid"], "gold_sources": docs.get(base["qid"], ""),
            "matched_evidence_count": base["evidence_count"],
            "complete": int(complete), "baseline_f1": base.get("f1", ""),
            "fault_f1": fault.get("f1", ""),
            "delta_f1": (_number(fault.get("f1")) - _number(base.get("f1")))
                        if complete and _number(fault.get("f1")) is not None
                        and _number(base.get("f1")) is not None else "",
            "baseline_correctness": base.get("judge_corr", ""),
            "fault_correctness": fault.get("judge_corr", ""),
            "baseline_refusal": base.get("ans_refusal", ""),
            "fault_refusal": fault.get("ans_refusal", ""),
            "baseline_explicit_refusal": base_explicit,
            "fault_explicit_refusal": fault_explicit,
            "baseline_answer_success": base_success,
            "fault_answer_success": fault_success,
            "error": " | ".join(x for x in (base.get("error"), fault.get("error")) if x),
        })
    with detail.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)
    done = [(b, f) for b, f in pairs if b.get("pred_answer") and f.get("pred_answer")
            and not b.get("error") and not f.get("error")]
    lines = ["# RAG 证据移除配对分析", "",
             f"- 证据审计：{len(pairs)} 对；正常条件 gold_hit=1、故障条件 gold_hit=0，且每对证据条数相同。",
             f"- 无生成或评分错误的完整配对：{len(done)}/{len(pairs)}。",
             f"- 涉及金标准文献 {len({docs.get(b['qid'], '') for b, _ in pairs})} 篇。",
             "- 样本按既有答案质量与正常检索命中筛选；结果是条件性机制实验。", ""]
    if done:
        explicit_done = [({**b, "explicit_refusal": explicit_refusal(b["pred_answer"])},
                          {**f, "explicit_refusal": explicit_refusal(f["pred_answer"])})
                         for b, f in done]
        for key, label, lower_is_worse in (
            ("f1", "答案 F1", True),
            ("judge_corr", "DeepSeek 正确性评分", True),
            ("explicit_refusal", "显式拒答代理", False),
        ):
            usable = [(b, f) for b, f in explicit_done if _number(b.get(key)) is not None
                      and _number(f.get(key)) is not None]
            if not usable:
                continue
            before = np.array([_number(b[key]) for b, _ in usable])
            after = np.array([_number(f[key]) for _, f in usable])
            delta = after - before
            adverse = int((delta < 0).sum() if lower_is_worse else (delta > 0).sum())
            unchanged = int((delta == 0).sum())
            lines.append(f"- {label}：正常均值 {before.mean():.3f}，故障均值 {after.mean():.3f}；"
                         f"退化 {adverse}/{len(usable)}，不变 {unchanged}/{len(usable)}。")
            boot = cluster_bootstrap_mean(usable, docs, key)
            if boot:
                lines.append(f"  - 故障−正常的均值差 {boot[0]:+.3f}；按文献重采样的描述性区间 "
                             f"[{boot[1][0]:+.3f}, {boot[1][1]:+.3f}]。")
        usable = [(b, f) for b, f in done if _number(b.get("judge_corr")) is not None
                  and _number(f.get("judge_corr")) is not None]
        if usable:
            good_base = sum(_number(b["judge_corr"]) >= 4 for b, _ in usable)
            good_fault = sum(_number(f["judge_corr"]) >= 4 for _, f in usable)
            lines.append(f"- 正确性评分≥4：正常 {good_base}/{len(usable)}，"
                         f"故障 {good_fault}/{len(usable)}。")
            conflicts = [f["qid"] for _, f in usable if _number(f["judge_corr"]) >= 4
                         and _number(f.get("ans_refusal")) == 1]
            if conflicts:
                lines.append(f"- 故障组有 {len(conflicts)} 条拒答仍被正确性评分判为≥4："
                             f"{', '.join(conflicts)}。该评分不能直接等同于答出金标准。")
            base_success = sum(_number(b["judge_corr"]) >= 4
                               and not explicit_refusal(b["pred_answer"]) for b, _ in usable)
            fault_success = sum(_number(f["judge_corr"]) >= 4
                                and not explicit_refusal(f["pred_answer"]) for _, f in usable)
            lines.append(f"- 纠正后的答题成功代理（评分≥4 且明确未拒答）："
                         f"正常 {base_success}/{len(usable)}，故障 {fault_success}/{len(usable)}。")
        legacy_base = sum(_number(b.get("ans_refusal")) == 1 for b, _ in done)
        legacy_fault = sum(_number(f.get("ans_refusal")) == 1 for _, f in done)
        explicit_base = sum(explicit_refusal(b["pred_answer"]) for b, _ in done)
        explicit_fault = sum(explicit_refusal(f["pred_answer"]) for _, f in done)
        lines.append(f"- 旧拒答规则：正常 {legacy_base}/{len(done)}、故障 {legacy_fault}/{len(done)}；"
                     f"显式拒答规则：正常 {int(explicit_base)}/{len(done)}、"
                     f"故障 {int(explicit_fault)}/{len(done)}。")
    else:
        lines.append("- 本次仅为离线预演，没有调用生成模型；答案效应尚未评估。")
    lines += ["", "## 解释边界", "",
              "该干预替换的是进入生成器的检索上下文，没有重建索引。模型生成随机性与 LLM 自评分可能影响观测值。",
              "gold_hit=0 仅表示金标准文献被移除；其他论文可能仍含通用答案，模型也可能凭已有知识回答。",
              "题目次序不是真实请求到达时间；本报告不估计误报率、平均报警间隔或检测延迟。",
              "候选新题仍为 pending，本实验只使用既有正式题。", ""]
    report.write_text("\n".join(lines), encoding="utf-8")
    return detail, report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--answers", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    detail, report = analyze(args.pairs, args.answers, args.output_dir)
    print(detail)
    print(report)


if __name__ == "__main__":
    main()
