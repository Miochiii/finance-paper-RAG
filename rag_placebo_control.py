# -*- coding: utf-8 -*-
"""在保留全部金标准证据时，等量替换非金标准片段作负对照。"""

import argparse
import csv
import json
import random
import re
from pathlib import Path

from rag_fault_replay import gold_hit, gold_sources, remove_gold_evidence
from rag_refusal import explicit_refusal


def make_placebo(base, gold, donor_pool, *, seed=20260927, qid=""):
    """替换与金标准片段数相同的非金标准片段；不足则返回 None。"""
    gold_indices = [i for i, e in enumerate(base) if e.get("source") in gold]
    nongold_indices = [i for i, e in enumerate(base) if e.get("source") not in gold]
    n = len(gold_indices)
    if n == 0 or len(nongold_indices) < n:
        return None
    base_sources = {e.get("source") for e in base}
    seen = {(e.get("source"), e.get("text")) for e in base}
    candidates = [e for e in donor_pool if e.get("source") not in base_sources
                  and e.get("text") and (e.get("source"), e.get("text")) not in seen]
    random.Random(f"{seed}:{qid}").shuffle(candidates)
    result = [dict(e) for e in base]
    donor_sources = []
    for i, donor in zip(nongold_indices[:n], candidates):
        result[i] = dict(donor)
        donor_sources.append(donor.get("source", ""))
    if len(donor_sources) != n:
        return None
    if len(result) != len(base) or not gold_hit(result, gold):
        raise AssertionError("负对照未保留金标准证据或证据条数")
    if any(result[i] != base[i] for i in gold_indices):
        raise AssertionError("金标准证据被改动")
    return result, nongold_indices[:n], donor_sources


def _read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def prompt_sources(context):
    """从 evaluate.make_context 的实际提示头读取进入模型的来源。"""
    sources = []
    for line in context.splitlines():
        match = re.fullmatch(r"\[来源\d+\]（来源: (.*?)）", line)
        if match:
            sources.append(match.group(1))
    return sources


def run(pairs_path: Path, answers_path: Path, evidence_cache: Path,
        output_dir: Path, *, seed=20260927, live=False, judge=False):
    if judge and not live:
        raise ValueError("--judge 只能与 --live 同用")
    from evaluate import make_context

    original = _read_rows(pairs_path)
    baseline = {r["qid"]: r for r in original if r.get("condition") == "baseline"}
    fault = {r["qid"]: r for r in original if r.get("condition") == "remove_gold"}
    if len(baseline) != len(fault) or set(baseline) != set(fault):
        raise ValueError("配对结果不完整")
    answers = {r["qid"]: r for r in _read_rows(answers_path)}
    cache = json.loads(evidence_cache.read_text(encoding="utf-8"))
    if any(qid not in answers or qid not in cache for qid in baseline):
        raise ValueError("题目或证据缓存缺失")
    # 仅使用这 40 题原本的检索片段作 donor，不引入新的论文材料。
    pool = [(qid, e) for qid in baseline for e in cache[qid]]
    full_pool = [(qid, e) for qid, evidence in cache.items() for e in evidence]

    def context_for(evidence):
        return make_context([{"text": e["text"],
                              "metadata": {"source": e.get("source", "")}}
                             for e in evidence])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "placebo_manifest.csv"
    replay_path = output_dir / "placebo_replay.csv"
    report_path = output_dir / "placebo_report.md"
    targets = (manifest_path, replay_path, report_path) if live else (manifest_path, report_path)
    if any(path.exists() for path in targets):
        raise FileExistsError("负对照输出已存在，请指定新目录")

    manifest, replay = [], []
    for qid, row in baseline.items():
        base = cache[qid]
        gold = gold_sources(answers[qid])
        if not gold_hit(base, gold) or len(base) != int(row["evidence_count"]):
            raise ValueError(f"基线证据与旧结果不一致：{qid}")
        if len({e.get("source") for e in base}) != int(row["unique_docs"]):
            raise ValueError(f"基线来源数与旧结果不一致：{qid}")
        gold_count = sum(e.get("source") in gold for e in base)
        nongold_count = len(base) - gold_count
        baseline_context = context_for(base)
        baseline_prompt_sources = prompt_sources(baseline_context)
        true_fault = remove_gold_evidence(
            base, gold, [e for source_qid, e in full_pool if source_qid != qid])
        old_fault = fault[qid]
        if (gold_hit(true_fault, gold)
                or len(true_fault) != int(old_fault["evidence_count"])
                or len({e.get("source") for e in true_fault}) != int(old_fault["unique_docs"])):
            raise ValueError(f"重建故障证据与旧结果不一致：{qid}")
        fault_prompt_sources = prompt_sources(context_for(true_fault))
        donors = [e for source_qid, e in pool if source_qid != qid]
        chosen = make_placebo(base, gold, donors, seed=seed, qid=qid)
        placebo_context = context_for(chosen[0]) if chosen else ""
        placebo_prompt_sources = prompt_sources(placebo_context) if chosen else []
        manifest.append({
            "qid": qid, "gold_count": gold_count, "nongold_count": nongold_count,
            "matched_placebo_feasible": int(chosen is not None),
            "baseline_prompt_gold_hit": int(bool(gold & set(baseline_prompt_sources))),
            "fault_prompt_gold_hit": int(bool(gold & set(fault_prompt_sources))),
            "baseline_prompt_blocks": len(baseline_prompt_sources),
            "fault_prompt_blocks": len(fault_prompt_sources),
            "placebo_prompt_gold_hit": int(bool(gold & set(placebo_prompt_sources))) if chosen else "",
            "placebo_prompt_blocks": len(placebo_prompt_sources) if chosen else "",
            "placebo_prompt_changed": int(baseline_context != placebo_context) if chosen else "",
            "replacement_indices": "|".join(map(str, chosen[1])) if chosen else "",
            "donor_sources": "|".join(chosen[2]) if chosen else "",
            "gold_hit_after": gold_hit(chosen[0], gold) if chosen else "",
            "evidence_count_after": len(chosen[0]) if chosen else "",
        })
        if not live or chosen is None:
            continue
        from evaluate import em_f1, generate, llm_judge
        context = placebo_context
        pred, error, corr, faith = "", "", None, None
        try:
            pred = generate(row["question"], context)
            if judge and pred:
                corr, faith, reason = llm_judge(
                    row["question"], row["gold_answer"], pred, context)
                if reason.startswith(("judge失败", "judge解析失败")):
                    error = reason
        except Exception as exc:
            error = f"{type(exc).__name__}: {str(exc)[:160]}"
        em, f1 = em_f1(pred, row["gold_answer"]) if pred else (None, None)
        replay.append({
            "qid": qid, "condition": "replace_nongold", "gold_hit": 1,
            "evidence_count": len(chosen[0]), "replacement_count": gold_count,
            "question": row["question"], "gold_answer": row["gold_answer"],
            "pred_answer": pred, "em": em, "f1": f1,
            "judge_corr": corr, "judge_faith": faith,
            "ans_refusal_explicit": explicit_refusal(pred) if pred else None,
            "baseline_f1": row.get("f1", ""),
            "baseline_judge_corr": row.get("judge_corr", ""),
            "baseline_refusal_explicit": explicit_refusal(row["pred_answer"]),
            "remove_gold_f1": fault[qid].get("f1", ""),
            "remove_gold_judge_corr": fault[qid].get("judge_corr", ""),
            "remove_gold_refusal_explicit": explicit_refusal(fault[qid]["pred_answer"]),
            "error": error,
        })
        print(f"{qid}: 负对照生成{'成功' if not error else '失败'}", flush=True)

    with manifest_path.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    if live:
        with replay_path.open("x", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(replay[0]))
            writer.writeheader()
            writer.writerows(replay)
    feasible = sum(int(r["matched_placebo_feasible"]) for r in manifest)
    complete = [r for r in replay if r["pred_answer"] and not r["error"]]
    lines = ["# 保留金标准证据的等量替换负对照", "",
             f"- 原配对题目 {len(baseline)} 道；含非金标准片段 "
             f"{sum(int(r['nongold_count']) > 0 for r in manifest)} 道；"
             f"非金标准片段数不少于金标准片段数、可等量替换 {feasible} 道。",
             f"- 实际生成提示审计：基线可见金标准来源 "
             f"{sum(int(r['baseline_prompt_gold_hit']) for r in manifest)}/{len(manifest)}；"
             f"原故障可见金标准来源 {sum(int(r['fault_prompt_gold_hit']) for r in manifest)}/{len(manifest)}。",
             "- 对可行题替换与原故障移除数量相同的非金标准片段；所有金标准片段原位置保留，证据总条数不变。",
             f"- 负对照提示中可见金标准来源 "
             f"{sum(int(r['placebo_prompt_gold_hit']) for r in manifest if r['matched_placebo_feasible'])}/{feasible}；"
             f"实际提示发生变化 {sum(int(r['placebo_prompt_changed']) for r in manifest if r['matched_placebo_feasible'])}/{feasible}。",
             "- donor 仅来自这 40 题的原始缓存证据，且来源文献不在该题基线中。",
             "- 这 7 题是结构筛选子集；与全部 40 题的比较不具有同等代表性。",
             f"- 生成：{'DeepSeek 生成并' + ('评分' if judge else '未评分') if live else '未调用模型'}；"
             f"完成 {len(complete)}/{feasible} 条。"]
    if complete and judge:
        good = sum(float(r["judge_corr"]) >= 4 and not int(r["ans_refusal_explicit"])
                   for r in complete if r["judge_corr"] not in (None, ""))
        refusal = sum(int(r["ans_refusal_explicit"]) for r in complete)
        lines += [f"- 负对照显式拒答 {refusal}/{len(complete)}；"
                  f"正确性评分≥4 且未明确拒答 {good}/{len(complete)}。"]
        lines += ["", "| 7 题结构子集 | 原基线 | 非金标准替换 | 移除金标准 |",
                  "|---|---:|---:|---:|"]
        for label, keys in (
            ("F1 均值", ("baseline_f1", "f1", "remove_gold_f1")),
            ("正确性评分均值", ("baseline_judge_corr", "judge_corr", "remove_gold_judge_corr")),
        ):
            means = [sum(float(r[key]) for r in complete) / len(complete) for key in keys]
            lines.append(f"| {label} | {means[0]:.3f} | {means[1]:.3f} | {means[2]:.3f} |")
        fault_refuses = sum(int(float(r["remove_gold_refusal_explicit"])) for r in complete)
        lines.append("")
        lines.append(f"- 该子集原故障答案显式拒答 {fault_refuses}/{len(complete)}；"
                     "因此不能用这 7 题验证全样本的拒答变化。")
    lines += ["", "本对照只评估保留金标准证据时的上下文扰动，不校准真实误报率。",
              "模型生成存在随机性；这些题目按证据结构筛出，不能把差异解释为总体因果效应。", ""]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return targets


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--answers", type=Path, required=True)
    ap.add_argument("--evidence-cache", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--judge", action="store_true")
    args = ap.parse_args()
    for path in run(args.pairs, args.answers, args.evidence_cache,
                    args.output_dir, seed=args.seed, live=args.live, judge=args.judge):
        print(path)


if __name__ == "__main__":
    main()
