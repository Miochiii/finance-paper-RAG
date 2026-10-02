# -*- coding: utf-8 -*-
"""审计盲审导入、初步一致性并生成独立复核表；不读取条件密钥。"""

import argparse
import csv
import difflib
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

from analyze_rag_blind_review import LABELS, SHORT

TEXT_FIELDS = ("question", "reference_answer", "model_answer")
FIELDS = ["review_id", *TEXT_FIELDS, *LABELS, "review_notes"]


def read_review(path):
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"无法严格解码：{path}")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    fields = reader.fieldnames or []
    if len(fields) != len(set(fields)) or not set(FIELDS).issubset(fields):
        raise ValueError(f"列名缺失或重复：{path}")
    if {"condition", "qid"} & set(fields):
        raise ValueError(f"复核输入包含实验条件或原题号：{path}")
    indexed = {}
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"CSV 列数不一致：{path}")
        rid = row["review_id"].strip()
        if not rid or rid in indexed:
            raise ValueError(f"审阅 ID 缺失或重复：{path}/{rid}")
        row["review_id"] = rid
        for name in LABELS:
            row[name] = row[name].strip().upper()
        indexed[rid] = row
    if not indexed:
        raise ValueError(f"空表：{path}")
    return indexed, {"path": str(path.resolve()), "encoding": encoding,
                     "sha256": hashlib.sha256(raw).hexdigest(), "rows": len(indexed)}


def _excel_gbk_loss(text):
    """仅识别旧 Excel 导出中不可编码字符对应的 UTF-16 问号替换。"""
    result = []
    for char in text:
        try:
            char.encode("gbk")
            result.append(char)
        except UnicodeEncodeError:
            result.append("?" * (len(char.encode("utf-16-le")) // 2))
    return "".join(result)


def agreement(left, right, name):
    allowed = LABELS[name]
    pairs = [(left[rid][name], right[rid][name]) for rid in sorted(left)]
    valid = [(a, b) for a, b in pairs if a in allowed and b in allowed]
    certain = [(a, b) for a, b in valid if a != "U" and b != "U"]
    n = len(certain)
    exact = sum(a == b for a, b in certain)
    lc = Counter(a for a, _ in certain)
    rc = Counter(b for _, b in certain)
    expected = sum(lc[x] * rc[x] for x in allowed - {"U"}) / n**2 if n else None
    kappa = ((exact / n - expected) / (1 - expected)
             if n and expected < 1 else None)
    return {"joint_valid_n": len(valid), "exact_including_U": sum(a == b for a, b in valid),
            "joint_certain_n": n, "exact_certain": exact, "cohen_kappa": kappa,
            "confusion": [{"reviewer_1": a, "reviewer_2": b, "n": count}
                          for (a, b), count in sorted(Counter(pairs).items())]}


def _write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit(template_path, reviewer_paths, output_dir, recheck_refusal=False, recheck_reference=False):
    if len(reviewer_paths) != 2:
        raise ValueError("需要两份审阅表")
    template, template_meta = read_review(template_path)
    reviews, metadata = [], []
    issues = []
    content_recheck = [set(), set()]
    for index, path in enumerate(reviewer_paths):
        rows, meta = read_review(path)
        if set(rows) != set(template):
            raise ValueError(f"ID 与冻结模板不一致：{path}")
        reviews.append(rows)
        metadata.append(meta)
        for rid in sorted(template):
            row = rows[rid]
            for name, allowed in LABELS.items():
                value = row[name]
                if value not in allowed:
                    issues.append({"reviewer": index + 1, "review_id": rid,
                                   "field": name, "issue": "missing_label" if not value else "invalid_label",
                                   "observed": value, "frozen": ""})
            for name in TEXT_FIELDS:
                original, supplied = template[rid][name], row[name]
                if original == supplied:
                    continue
                emoji_only = supplied == original.replace("📚", "??")
                export_loss = supplied == _excel_gbk_loss(original)
                issue = ("decorative_encoding_loss" if emoji_only else
                         "symbol_encoding_loss" if export_loss else "content_changed")
                if not emoji_only:
                    content_recheck[index].add(rid)
                changes = [(original[i:j], supplied[k:l]) for tag, i, j, k, l in
                           difflib.SequenceMatcher(None, original, supplied, autojunk=False).get_opcodes()
                           if tag != "equal"]
                issues.append({"reviewer": index + 1, "review_id": rid, "field": name,
                               "issue": issue,
                               "observed": json.dumps([b for _, b in changes], ensure_ascii=False),
                               "frozen": json.dumps([a for a, _ in changes], ensure_ascii=False)})
    stats = {SHORT[name]: agreement(*reviews, name) for name in LABELS}
    groups = defaultdict(list)
    for rid, row in template.items():
        groups[(row["question"], row["reference_answer"])].append(rid)
    repeated = [ids for ids in groups.values() if len(ids) > 1]
    reference_inconsistent = [sum(len({rows[rid]["reference_adequate_0_1_U"] for rid in ids}) > 1
                                  for ids in repeated) for rows in reviews]
    unresolved = sum(any(reviews[0][rid][name] != reviews[1][rid][name]
                         or reviews[0][rid][name] not in allowed - {"U"}
                         or reviews[1][rid][name] not in allowed - {"U"}
                         for name, allowed in LABELS.items()) for rid in template)
    summary = {"status": "import_audit_only", "template": template_meta, "reviewers": metadata,
               "agreement": stats, "unresolved_rows": unresolved,
               "issue_counts": dict(Counter(item["issue"] for item in issues)),
               "content_recheck": [sorted(x) for x in content_recheck],
               "recheck_all_refusal": recheck_refusal,
               "recheck_all_reference": recheck_reference,
               "repeated_question_reference_groups": len(repeated),
               "reference_inconsistent_groups": reference_inconsistent,
               "label_counts": [{SHORT[name]: dict(Counter(rows[rid][name] for rid in rows))
                                 for name in LABELS} for rows in reviews]}
    names = ["import_audit.json", "import_issues.csv", "audit_report.md", "复核填写说明.md",
             "reviewer_1_recheck.csv", "reviewer_2_recheck.csv"]
    output_dir.mkdir(parents=True, exist_ok=True)
    if any((output_dir / name).exists() for name in names):
        raise FileExistsError("输出文件已存在；请使用新目录，避免覆盖已填写的复核表")
    (output_dir / names[0]).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(output_dir / names[1], issues, ["reviewer", "review_id", "field", "issue", "observed", "frozen"])
    for index, rows in enumerate(reviews):
        repaired = []
        for rid in template:
            item = {field: template[rid][field] for field in FIELDS}
            item["review_notes"] = rows[rid]["review_notes"]
            reasons = []
            if rid in content_recheck[index]:
                reasons.append("本行输入文本与冻结版本有内容或公式符号差异；请重新判定三个标签")
            if recheck_refusal:
                reasons.append("请按明确拒答定义重新核对拒答列")
            if recheck_reference:
                reasons.append("请判断标准答案本身是否足够；不要按模型回答是否正确来填此列")
            for name, allowed in LABELS.items():
                value = rows[rid][name]
                item["previous_" + SHORT[name]] = value
                needs_check = (rid in content_recheck[index] or value not in allowed
                               or recheck_refusal and name == "explicit_refusal_0_1_U"
                               or recheck_reference and name == "reference_adequate_0_1_U")
                item[name] = "" if needs_check else value
                if value not in allowed:
                    reasons.append(f"{SHORT[name]} 原标签缺失或无效")
            item["recheck_reason"] = "；".join(reasons)
            repaired.append(item)
        fields = FIELDS + ["previous_" + SHORT[name] for name in LABELS] + ["recheck_reason"]
        _write_csv(output_dir / names[4 + index], repaired, fields)
    lines = ["# 盲审导入审计与初步一致性", "",
             "本报告只使用冻结盲表及两份提交表，不读取实验条件密钥。原文件不修改。",
             "原标签尚有缺失、无效值及文本差异；以下是修正前描述性审计，不是正式双人盲审或仲裁结果。", "",
             f"- 两表各 {len(template)} 行，ID 均与模板一致；没有审阅备注。"
             if not any(r["review_notes"].strip() for rows in reviews for r in rows.values())
             else f"- 两表各 {len(template)} 行，ID 均与模板一致。",
             f"- 问题分类：{summary['issue_counts']}。",
             f"- 任一标签不同、U、缺失或无效的回答：{unresolved}/{len(template)}。", "",
             "| 标签 | 双方合法标签（含 U） | 完全一致（含 U） | 双方确定标签 | 确定标签一致 | Cohen κ |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, stat in stats.items():
        kappa = f"{stat['cohen_kappa']:.3f}" if stat["cohen_kappa"] is not None else "不可计算"
        lines.append(f"| {name} | {stat['joint_valid_n']} | {stat['exact_including_U']} | "
                     f"{stat['joint_certain_n']} | {stat['exact_certain']} | {kappa} |")
    refusal_stat = stats["explicit_refusal"]
    lines += ["", "分母逐列计算；缺失/非法值排除，U 不参与 κ。含 U 的一致不等于已确定。"
              "80 条回答来自 40 个问题，未把它们当作 80 个独立实验样本进行显著性检验。", "",
              "## 标签理解需核实", "",
              f"- 拒答确定标签一致 {refusal_stat['exact_certain']}/{refusal_stat['joint_certain_n']}；"
              "应核实双方对拒答标签含义的理解，不能自动翻转任何人的整列标签。",
              f"- 冻结文本中相同问题与相同标准答案的重复组共 {len(repeated)} 组。"
              f"两位审阅者在其中分别有 {reference_inconsistent[0]}、{reference_inconsistent[1]} 组"
              "给了不同的充分性标签或漏填；该字段判断标准答案本身，不应由模型回答改变。",
              "- 两份表的 U 均缺少备注，后续核论文时需要补充具体疑点。", "",
              "## 输入与追溯", ""]
    for index, meta in enumerate(metadata):
        lines.append(f"- 审阅者 {index + 1}：{meta['encoding']}；SHA-256 `{meta['sha256']}`。"
                     f"需要核对内容的 ID：{', '.join(sorted(content_recheck[index])) or '无'}。")
    lines += ["", "## 下一步", "",
              "1. 两名原审阅者分别填写各自 recheck 表的空白标签，保持独立，不交换标签或查看条件密钥。",
              "2. 两人统一理解三个字段，重新核对拒答与标准答案充分性列；文本有差异的行重新判定，无法判定时填 U 并说明原因。",
              "3. 锁定复核表后重新计算一致性，再生成最终分歧仲裁表。标准答案不足的题按 PDF 做题目级核验，"
              "同一问题的标准答案充分性统一判定；保留原基准及所有修改版本。",
              "4. 仲裁完成后计算两条件的配对成功率、质量变化及拒答变化，核验自动拒答代理。"
              "这 40 题的离线结果仍不能替代独立在线序列的变点验证。", ""]
    (output_dir / names[2]).write_text("\n".join(lines), encoding="utf-8")
    guide = """# 给原审阅者的独立复核说明

请只打开你自己的 reviewer_N_recheck.csv 和本说明，不查看另一位审阅者的结果或实验条件密钥。
本轮是校正导入问题和标签理解的独立复核。保留 previous_* 列作为原提交记录。

## 三个字段的含义

- reference_adequate_0_1_U：标准答案足以判定问题=1；明显不足=0；需核原论文=U。
- explicit_refusal_0_1_U：回答明确表示当前无法作答或材料不足以作答=1；未明确拒答=0；确实无法判断=U。答错也可能是 0。解释某个概念中一般性的“无法判断”，不直接视为拒答。
- answer_quality_0_1_2_U：错误或未回答=0；部分回答=1；实质正确=2；标准答案不足、需要核论文才能判断=U。拒答措辞清晰并不代表答题质量为 2。

拒答判断示例（虚构示例，与实验题无关）：

| 问题与回答 | 拒答标签 | 说明 |
|---|---|---|
| 问：圆的面积公式？答：面积为 2πr。 | 0 | 作出了回答；错误体现在质量标签。 |
| 问：圆的面积公式？答：所给资料没有面积公式，无法回答。 | 1 | 明确表示无法回答。 |
| 问：随机变量是什么意思？答：取值无法事先确定的变量。 | 0 | “无法确定”是在解释概念。 |

## 本轮填写

1. 重新核对已留空的标签。本批要求两位审阅者核对全部回答的拒答和标准答案充分性。拒答 1 的含义是明确拒答；标准答案充分性只依据问题与标准答案判断，不按模型是否答对来填。相同问题及标准答案应使用同一判定，难以确定时填 U。
2. 空白的其他标签必须重新判定；未留空的其他标签为原提交值，若发现误填可以修正并在 review_notes 说明。
3. 审阅者 1 特别核对 R0013（公式下标）和 R0022（整段标准答案）：以本表恢复的冻结文本重新判定三个标签，并在备注说明原审阅时是否能看到完整文本。
4. 审阅者 2 补全 R0018；R0014 的旧拒答标签 2 不合法，按规则重新填写。
5. 对 U 和标准答案不足的判断，尽可能写明疑点，方便下一轮核对论文。
6. 另存为 UTF-8 CSV，例如 reviewer_1_recheck_ok.csv；不要覆盖原始 *_ok.csv，不改 question/reference_answer/model_answer 列。

所有复核表均不含条件和原题号；纠正后才锁定并进入一致性与仲裁分析。
"""
    (output_dir / names[3]).write_text(guide, encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--template", type=Path, required=True)
    ap.add_argument("--reviewer-1", type=Path, required=True)
    ap.add_argument("--reviewer-2", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--recheck-refusal", action="store_true")
    ap.add_argument("--recheck-reference", action="store_true")
    args = ap.parse_args()
    result = audit(args.template, [args.reviewer_1, args.reviewer_2], args.output_dir,
                   args.recheck_refusal, args.recheck_reference)
    print(json.dumps({"output_dir": str(args.output_dir), "issues": result["issue_counts"],
                      "unresolved_rows": result["unresolved_rows"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
