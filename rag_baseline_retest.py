# -*- coding: utf-8 -*-
"""对原 40 题的相同基线提示重复生成，估计模型与评判器自身波动。"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from rag_refusal import explicit_refusal


def _rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def run(pairs_path: Path, evidence_cache: Path, output_dir: Path, *, live=False):
    from evaluate import em_f1, generate, llm_judge, make_context

    paired = _rows(pairs_path)
    baseline = [r for r in paired if r.get("condition") == "baseline"]
    faults = {r["qid"]: r for r in paired if r.get("condition") == "remove_gold"}
    if not baseline or len(baseline) != len(faults) or {r["qid"] for r in baseline} != set(faults):
        raise ValueError("配对结果不完整")
    cache = json.loads(evidence_cache.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "baseline_retest.csv"
    report_path = output_dir / "baseline_retest.md"
    if csv_path.exists() or report_path.exists():
        raise FileExistsError("输出已存在，请指定新目录")
    out = []
    for row in baseline:
        qid = row["qid"]
        evidence = cache[qid]
        if len(evidence) != int(row["evidence_count"]):
            raise ValueError(f"缓存证据条数不匹配：{qid}")
        context = make_context([{"text": e["text"],
                                 "metadata": {"source": e.get("source", "")}}
                                for e in evidence])
        pred, error, corr, faith = "", "", None, None
        if live:
            try:
                pred = generate(row["question"], context)
                if pred:
                    corr, faith, reason = llm_judge(row["question"], row["gold_answer"],
                                                    pred, context)
                    if reason.startswith(("judge失败", "judge解析失败")):
                        error = reason
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc)[:160]}"
        em, f1 = em_f1(pred, row["gold_answer"]) if pred else (None, None)
        out.append({
            "qid": qid,
            "prompt_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
            "question": row["question"], "gold_answer": row["gold_answer"],
            "retest_answer": pred, "retest_em": em, "retest_f1": f1,
            "retest_judge_corr": corr, "retest_judge_faith": faith,
            "retest_refusal_explicit": explicit_refusal(pred) if pred else None,
            "baseline_answer": row["pred_answer"],
            "baseline_f1": row.get("f1", ""),
            "baseline_judge_corr": row.get("judge_corr", ""),
            "baseline_refusal_explicit": explicit_refusal(row["pred_answer"]),
            "remove_gold_answer": faults[qid]["pred_answer"],
            "remove_gold_f1": faults[qid].get("f1", ""),
            "remove_gold_judge_corr": faults[qid].get("judge_corr", ""),
            "remove_gold_refusal_explicit": explicit_refusal(faults[qid]["pred_answer"]),
            "error": error,
        })
        if live:
            print(f"[{len(out)}/{len(baseline)}] {qid}: {'成功' if not error else '失败'}", flush=True)
    with csv_path.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    complete = [r for r in out if r["retest_answer"] and r["retest_judge_corr"] is not None
                and not r["error"]]
    lines = ["# 相同基线提示重复生成", "",
             f"- 配对题目 {len(baseline)} 道；"
             f"{'重复生成并评分' if live else '只完成提示预检，未调用模型'}；"
             f"完整 {len(complete)}/{len(baseline)}。",
             "- 每题复用原问题、原金标准答案和原检索缓存；生成提示由同一 evaluate.make_context 构造。",
             "- 模型温度为 0，但 API 和 LLM 评判器仍可能产生波动。"]
    if complete:
        lines += ["", "| 指标 | 原基线 | 相同提示重跑 | 移除金标准 |",
                  "|---|---:|---:|---:|"]
        for label, keys in (
            ("F1 均值", ("baseline_f1", "retest_f1", "remove_gold_f1")),
            ("正确性评分均值", ("baseline_judge_corr", "retest_judge_corr", "remove_gold_judge_corr")),
        ):
            means = [sum(float(r[key]) for r in complete) / len(complete) for key in keys]
            lines.append(f"| {label} | {means[0]:.3f} | {means[1]:.3f} | {means[2]:.3f} |")
        for label, score_key, refusal_key in (
            ("原基线", "baseline_judge_corr", "baseline_refusal_explicit"),
            ("相同提示重跑", "retest_judge_corr", "retest_refusal_explicit"),
            ("移除金标准", "remove_gold_judge_corr", "remove_gold_refusal_explicit"),
        ):
            refuses = sum(int(float(r[refusal_key])) for r in complete)
            success = sum(float(r[score_key]) >= 4 and not int(float(r[refusal_key]))
                          for r in complete)
            lines.append(f"- {label}：显式拒答 {refuses}/{len(complete)}；"
                         f"答题成功代理 {success}/{len(complete)}。")
        lines.append(f"- 重跑答案与原基线答案完全相同 "
                     f"{sum(r['retest_answer'] == r['baseline_answer'] for r in complete)}/{len(complete)}。")
        score_shift = [float(r["retest_judge_corr"]) - float(r["baseline_judge_corr"])
                       for r in complete]
        mean_abs_f1 = sum(abs(float(r["retest_f1"]) - float(r["baseline_f1"]))
                          for r in complete) / len(complete)
        lines.append(f"- 重跑正确性评分发生变化 {sum(d != 0 for d in score_shift)}/{len(complete)} 题；"
                     f"下降 {sum(d < 0 for d in score_shift)} 题、上升 {sum(d > 0 for d in score_shift)} 题。")
        lines.append(f"- 重跑与原基线的 F1 平均绝对差 {mean_abs_f1:.3f}；"
                     f"F1 下降 {sum(float(r['retest_f1']) < float(r['baseline_f1']) for r in complete)}/{len(complete)} 题。")
    lines += ["", "相同提示重跑是模型波动对照，不是新的独立用户请求；"
              "不能用来估计线上误报率。", ""]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, report_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--evidence-cache", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()
    for path in run(args.pairs, args.evidence_cache, args.output_dir, live=args.live):
        print(path)


if __name__ == "__main__":
    main()
