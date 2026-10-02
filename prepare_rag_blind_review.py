# -*- coding: utf-8 -*-
"""把证据移除配对答案随机打散，生成不透露条件的人工盲审表。"""

import argparse
import csv
import hashlib
import random
import shutil
from pathlib import Path


BLIND_FIELDS = [
    "review_id", "question", "reference_answer", "model_answer",
    "reference_adequate_0_1_U", "explicit_refusal_0_1_U",
    "answer_quality_0_1_2_U", "review_notes",
]
KEY_FIELDS = ["review_id", "qid", "condition"]


def prepare(pairs_path: Path, output_dir: Path, seed: int = 20260927):
    with pairs_path.open(encoding="utf-8-sig", newline="") as f:
        original = list(csv.DictReader(f))
    if not original:
        raise ValueError("配对结果为空")
    grouped = {}
    for row in original:
        qid, condition = row.get("qid"), row.get("condition")
        if not qid or condition not in {"baseline", "remove_gold"}:
            raise ValueError("缺少题号或条件异常")
        if not row.get("question") or not row.get("gold_answer") or not row.get("pred_answer"):
            raise ValueError(f"题目、标准答案或模型答案缺失：{qid}/{condition}")
        if row.get("error"):
            raise ValueError(f"模型或评分错误：{qid}/{condition}")
        conditions = grouped.setdefault(qid, set())
        if condition in conditions:
            raise ValueError(f"重复条件：{qid}/{condition}")
        conditions.add(condition)
    if any(conditions != {"baseline", "remove_gold"} for conditions in grouped.values()):
        raise ValueError("存在未配对题目")

    rows = list(original)
    random.Random(seed).shuffle(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    blind_path = output_dir / "blind_answer_review.csv"
    reviewer_1_path = output_dir / "reviewer_1.csv"
    reviewer_2_path = output_dir / "reviewer_2.csv"
    key_path = output_dir / "blind_answer_key.csv"
    guide_path = output_dir / "README.md"
    if any(path.exists() for path in (blind_path, reviewer_1_path, reviewer_2_path,
                                      key_path, guide_path)):
        raise FileExistsError("盲审文件已存在；请指定新目录，避免覆盖已填写的标注")
    blinded, key = [], []
    for i, row in enumerate(rows, 1):
        review_id = f"R{i:04d}"
        blinded.append({
            "review_id": review_id,
            "question": row["question"],
            "reference_answer": row["gold_answer"],
            "model_answer": row["pred_answer"],
            "reference_adequate_0_1_U": "",
            "explicit_refusal_0_1_U": "",
            "answer_quality_0_1_2_U": "",
            "review_notes": "",
        })
        key.append({"review_id": review_id, "qid": row["qid"],
                    "condition": row["condition"]})
    for path, fields, data in ((blind_path, BLIND_FIELDS, blinded),
                               (key_path, KEY_FIELDS, key)):
        with path.open("x", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
    shutil.copyfile(blind_path, reviewer_1_path)
    shutil.copyfile(blind_path, reviewer_2_path)
    digest = hashlib.sha256(pairs_path.read_bytes()).hexdigest()
    guide_path.write_text(
        "# RAG 配对答案盲审说明\n\n"
        f"- 来源文件 SHA-256：`{digest}`；随机种子：`{seed}`；"
        f"{len(grouped)} 道题、{len(blinded)} 条答案。\n"
        "- `reviewer_1.csv` 与 `reviewer_2.csv` 是相同的空白盲审表，分别交给两名审阅者；"
        "`blind_answer_review.csv` 是未填写模板；"
        "完成并锁定标注前不要打开 `blind_answer_key.csv`。\n"
        "- 每行独立审阅，不根据重复的问题猜测 A/B 条件；"
        "盲表不包含检索证据，因此本轮只评回答质量，不评证据忠实性。\n\n"
        "## 填写规则\n\n"
        "- `reference_adequate_0_1_U`：标准答案是否足以判定问题，1=足够，0=明显不足，U=需核 PDF。\n"
        "- `explicit_refusal_0_1_U`：模型是否明确表示无法回答，1=是，0=否，U=不确定。\n"
        "- `answer_quality_0_1_2_U`：0=错误或未回答，1=部分回答，2=实质正确，U=标准答案不足或需核 PDF。\n"
        "- 对标准答案不足、引用原文存疑或回答看似正确但依据不足的情况，写明疑点并核对 PDF。\n"
        "- 两名审阅者应分别填写盲表；锁定后才用密钥表揭盲和统计一致性。\n"
        "- `analyze_rag_blind_review.py` 在填写未完成时只输出进度，不读取密钥；"
        "两份都完成后输出一致性和不含条件的分歧仲裁表。\n"
        "- 独立仲裁者填写分歧表的 `final_*` 列后，使用 `--adjudication` 重跑，"
        "才得到正式的分条件配对摘要。\n",
        encoding="utf-8")
    return blind_path, reviewer_1_path, reviewer_2_path, key_path, guide_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260927)
    args = ap.parse_args()
    for path in prepare(args.pairs, args.output_dir, args.seed):
        print(path)


if __name__ == "__main__":
    main()
