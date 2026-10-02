"""Describe a frozen, gold-free context signal against previous automatic labels.

No classification threshold is fitted; these are historical descriptive results,
not independent validation of the new experiment.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def run(scores_path, feature_path, output):
    with scores_path.open(encoding="utf-8-sig", newline="") as f:
        scores = list(csv.DictReader(f))
    features = json.loads(feature_path.read_text(encoding="utf-8"))["scores"]
    groups = defaultdict(list)
    rows = []
    for row in scores:
        key = row["qid"] + ":" + row["condition"]
        feature = features[key]
        quality = row["answer_quality"]
        group = ("uncertain" if quality == "U" else
                 "complete" if quality == "2" else
                 "refusal_incomplete" if float(row["rule_refusal"]) == 1 else
                 "nonrefusal_incomplete")
        groups[(row["condition"], group)].append(feature["relevance_gap"])
        rows.append({k: row[k] for k in ("review_id", "qid", "condition", "answer_quality", "rule_refusal")} |
                    dict(group=group, context_gap=feature["relevance_gap"],
                         context_sha256=feature["context_sha256"], truncated=feature["truncated"]))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "historical_context_proxy.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# 上下文相关性辅助信号：历史答案的描述性核查", "",
             "分组来自此前已冻结的自动质量标签；BGE只看问题和可见上下文，不看答案或标准答案。此处没有拟合分类阈值，也不改变新生成实验的任何参数。", "",
             "|条件|历史答案组|数量|平均缺口|最小|最大|", "|---|---|---:|---:|---:|---:|"]
    for (condition, group), values in sorted(groups.items()):
        lines.append(f"|{condition}|{group}|{len(values)}|{mean(values):.4f}|{min(values):.4f}|{max(values):.4f}|")
    lines += ["", "缺口大表示模型认为证据不切题，不是答案错误概率。即使低缺口，答案仍可遗漏要点或答错。",
              "本核查使用旧答案质量和同一固定题池；它不能提供独立泛化准确率，也不能证明所有非拒答错误都可检出。",
              "所有80个上下文在512token上限发生截断，因此数值仅反映截断后输入的相关性。", ""]
    (output / "historical_context_proxy.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    run(args.scores, args.features, args.output)
