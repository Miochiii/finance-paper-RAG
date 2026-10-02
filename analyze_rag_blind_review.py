# -*- coding: utf-8 -*-
"""校验双人盲审进度；锁定后才揭盲，并生成分歧及最终配对摘要。"""

import argparse
import csv
from collections import Counter
from pathlib import Path


LABELS = {
    "reference_adequate_0_1_U": {"0", "1", "U"},
    "explicit_refusal_0_1_U": {"0", "1", "U"},
    "answer_quality_0_1_2_U": {"0", "1", "2", "U"},
}
SHORT = {"reference_adequate_0_1_U": "reference_adequate",
         "explicit_refusal_0_1_U": "explicit_refusal",
         "answer_quality_0_1_2_U": "answer_quality"}


def _read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _index_review(path):
    rows = _read_csv(path)
    if not rows:
        raise ValueError(f"盲审表为空：{path}")
    if not {"review_id", "question", "reference_answer", "model_answer", *LABELS}.issubset(rows[0]):
        raise ValueError(f"盲审表缺少列：{path}")
    if "condition" in rows[0] or "qid" in rows[0]:
        raise ValueError("审阅文件包含实验条件或原题号，盲审已破坏")
    indexed = {}
    for row in rows:
        rid = row["review_id"]
        if not rid or rid in indexed:
            raise ValueError(f"审阅 ID 缺失或重复：{rid}")
        for name, allowed in LABELS.items():
            value = (row.get(name) or "").strip().upper()
            if value and value not in allowed:
                raise ValueError(f"{path.name} {rid} 的 {name} 标记无效：{value}")
            row[name] = value
        indexed[rid] = row
    return indexed


def _kappa(rows, name):
    valid = [(a[name], b[name]) for a, b in rows
             if a[name] != "U" and b[name] != "U"]
    if not valid:
        return None, 0
    n = len(valid)
    observed = sum(a == b for a, b in valid) / n
    left, right = Counter(a for a, _ in valid), Counter(b for _, b in valid)
    expected = sum(left[label] * right[label] for label in LABELS[name] if label != "U") / n**2
    kappa = (observed - expected) / (1 - expected) if expected < 1 else None
    return kappa, n


def _load_key(path, ids):
    rows = _read_csv(path)
    key = {}
    by_qid = {}
    for row in rows:
        rid, qid, condition = row.get("review_id"), row.get("qid"), row.get("condition")
        if rid in key or not qid or condition not in {"baseline", "remove_gold"}:
            raise ValueError("揭盲密钥存在重复或未知条件")
        key[rid] = row
        by_qid.setdefault(qid, Counter())[condition] += 1
    if set(key) != ids or any(v != Counter({"baseline": 1, "remove_gold": 1}) for v in by_qid.values()):
        raise ValueError("密钥与审阅 ID 或配对条件不一致")
    return key


def _adjudicated_labels(path, disagreement_ids):
    rows = _read_csv(path)
    final = {}
    for row in rows:
        rid = row.get("review_id")
        if rid in final or rid not in disagreement_ids:
            raise ValueError(f"仲裁 ID 重复或不在分歧表：{rid}")
        labels = {}
        for name, allowed in LABELS.items():
            value = (row.get("final_" + SHORT[name]) or "").strip().upper()
            if value not in allowed:
                raise ValueError(f"仲裁未完成或标签无效：{rid}/{name}")
            labels[name] = value
        final[rid] = labels
    if set(final) != disagreement_ids:
        raise ValueError("仲裁表未覆盖全部分歧 ID")
    return final


def analyze(reviewer_1: Path, reviewer_2: Path, key_path: Path,
            output_dir: Path, adjudication: Path = None, template: Path = None):
    left = _index_review(reviewer_1)
    right = _index_review(reviewer_2)
    if set(left) != set(right):
        raise ValueError("两位审阅者的 review_id 不一致")
    if any((left[rid][f] != right[rid][f]) for rid in left
           for f in ("question", "reference_answer", "model_answer")):
        raise ValueError("两份盲审表的题目或答案被修改")
    if template is not None:
        frozen = _index_review(template)
        if set(frozen) != set(left) or any(left[rid][f] != frozen[rid][f] for rid in left
                                        for f in ("question", "reference_answer", "model_answer")):
            raise ValueError("审阅文本与冻结模板不一致")
    output_dir.mkdir(parents=True, exist_ok=True)
    ids = sorted(left)
    complete_1 = sum(all(left[rid][f] for f in LABELS) for rid in ids)
    complete_2 = sum(all(right[rid][f] for f in LABELS) for rid in ids)
    progress_path = output_dir / "review_progress.md"
    progress_path.write_text(
        "# 盲审进度\n\n"
        f"- 审阅条目 {len(ids)}；审阅者 1 完成 {complete_1}/{len(ids)}；"
        f"审阅者 2 完成 {complete_2}/{len(ids)}。\n"
        "- 两份表都全部填写后，脚本才读取条件密钥并生成揭盲摘要。\n",
        encoding="utf-8")
    if complete_1 < len(ids) or complete_2 < len(ids):
        if adjudication:
            raise ValueError("双人盲审尚未完成，不能仲裁或揭盲")
        return [progress_path]

    paired = [(left[rid], right[rid]) for rid in ids]
    disagreement_ids = {rid for rid in ids if any(
        left[rid][name] != right[rid][name]
        or left[rid][name] == "U" or right[rid][name] == "U"
        for name in LABELS)}
    disagreements_path = output_dir / "blind_disagreements.csv"
    agreement_path = output_dir / "blind_agreement.md"
    disagreement_rows = []
    for rid in ids:
        if rid not in disagreement_ids:
            continue
        item = {f: left[rid][f] for f in ("review_id", "question", "reference_answer", "model_answer")}
        for name in LABELS:
            short = SHORT[name]
            item["reviewer_1_" + short] = left[rid][name]
            item["reviewer_2_" + short] = right[rid][name]
            item["final_" + short] = ""
        disagreement_rows.append(item)
    fields = ["review_id", "question", "reference_answer", "model_answer"]
    for name in LABELS:
        fields.extend(("reviewer_1_" + SHORT[name], "reviewer_2_" + SHORT[name],
                       "final_" + SHORT[name]))
    if not disagreements_path.exists():
        _write_csv(disagreements_path, disagreement_rows, fields)
    lines = ["# 双人盲审一致性", "",
             f"- 审阅条目 {len(ids)}；有分歧或 U 的条目 {len(disagreement_ids)}。",
             "- Kappa 仅在双方均非 U 的条目上计算；U 和分歧进入不含条件的仲裁表。", ""]
    for name in LABELS:
        agreement = sum(left[rid][name] == right[rid][name] for rid in ids)
        kappa, n = _kappa(paired, name)
        lines.append(f"- {SHORT[name]}：完全一致 {agreement}/{len(ids)}；"
                     f"非 U 配对 {n}；Cohen κ={kappa:.3f}。" if kappa is not None else
                     f"- {SHORT[name]}：完全一致 {agreement}/{len(ids)}；"
                     f"非 U 配对 {n}；Cohen κ 无法计算。")
    agreement_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    outputs = [progress_path, agreement_path, disagreements_path]

    if disagreement_ids and adjudication is None:
        return outputs
    final = _adjudicated_labels(adjudication, disagreement_ids) if disagreement_ids else {}
    for rid in ids:
        if rid not in final:
            final[rid] = {name: left[rid][name] for name in LABELS}
    key = _load_key(key_path, set(ids))
    reference_by_qid = {}
    for rid in ids:
        qid = key[rid]["qid"]
        adequate = final[rid]["reference_adequate_0_1_U"]
        if qid in reference_by_qid and reference_by_qid[qid] != adequate:
            raise ValueError(f"同一题标准答案充分性不一致：{qid}；请按问题统一核验后再统计")
        reference_by_qid[qid] = adequate
    by_condition = {"baseline": [], "remove_gold": []}
    by_qid = {}
    for rid in ids:
        row = final[rid]
        condition = key[rid]["condition"]
        adequate = row["reference_adequate_0_1_U"] == "1"
        decidable = (adequate and row["answer_quality_0_1_2_U"] != "U"
                     and row["explicit_refusal_0_1_U"] != "U")
        success = int(row["answer_quality_0_1_2_U"] == "2"
                      and row["explicit_refusal_0_1_U"] == "0") if decidable else None
        by_condition[condition].append((row, success))
        by_qid.setdefault(key[rid]["qid"], {})[condition] = success
    summary_path = output_dir / "blind_condition_summary.md"
    lines = ["# 仲裁后盲审结果", "",
             "答题成功定义为：标准答案足够、模型实质正确且未明确拒答；"
             "标准答案不足或不确定的条目不计入成功率分母。", ""]
    for condition, items in by_condition.items():
        valid = [success for _, success in items if success is not None]
        refusal_valid = [row["explicit_refusal_0_1_U"] for row, _ in items
                         if row["explicit_refusal_0_1_U"] != "U"]
        refusals = sum(value == "1" for value in refusal_valid)
        lines.append(f"- {condition}：可判 {len(valid)}/{len(items)}；"
                     f"成功 {sum(valid)}/{len(valid)}；明确拒答 {refusals}/{len(refusal_valid)}。")
    comparable = [conditions for conditions in by_qid.values()
                  if all(conditions.get(c) is not None for c in by_condition)]
    lines.append(f"- 双条件均可判的配对 {len(comparable)}/{len(by_qid)}；"
                 f"成功从基线降至故障 {sum(x['baseline'] > x['remove_gold'] for x in comparable)} 对。")
    lines += ["", "本结果依赖盲审和仲裁质量；这些固定题目仍不能估计真实线上误报率。", ""]
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    outputs.append(summary_path)
    return outputs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reviewer-1", type=Path, required=True)
    ap.add_argument("--reviewer-2", type=Path, required=True)
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--adjudication", type=Path)
    ap.add_argument("--template", type=Path, help="冻结盲审模板，用于防止双方文件同时修改文本")
    args = ap.parse_args()
    for path in analyze(args.reviewer_1, args.reviewer_2, args.key,
                        args.output_dir, args.adjudication, args.template):
        print(path)


if __name__ == "__main__":
    main()
