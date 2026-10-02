# -*- coding: utf-8 -*-
"""保留原始候选，另存一份经原文片段补全引文的待审核副本。"""

import csv
from collections import Counter
from pathlib import Path

import gen_annotations as ga
import generate_change_point_candidates as gen


RAW = Path(ga.ANNOT_DIR) / "finance_annotations_v3_candidates_20260927_clean.csv"
OUT = Path(ga.ANNOT_DIR) / "finance_annotations_v3_candidates_20260927_evidence_checked.csv"
REPORT = OUT.with_suffix(".md")

# 问题关键词 -> 原文片段内逐字起止点。只用本轮已经发送过的四段本地摘录。
REPAIRS = {
    "哪些数据集来比较": (
        "因此本文选择曾经最大的两家P2P机构", "德国、澳大利亚数据集作为对比。"),
    "阈值范围和步长": (
        "具体而言，本文采用遍历的办法进行阈值调整", "0.\n99）。"),
    "特征工程归纳为哪几部分": (
        "综合各类文献，本文将特征工程归纳为以下几部分：", 'A --> D["特征监控"]\n```'),
    "特征工程做得好": (
        "（1）特征越好，灵活性越强。", "（3）特征越好，模型的性能越出色。"),
    "未来可从哪些方向": (
        "将来的研究可以从以下几个方向进行探索：", "将会是未来A股市场投资的发展方向。"),
    "已实现波动率进行建模": (
        "目前应用最普遍的 RV 建模方法是由 Corsi", "这也是本文第二个实证章节的建模框架。"),
    "哪些因子模型下": (
        "实证结果显示，RF-Text-TSMOM 策略在 Fama", "均能取得显著的风险溢价表现；"),
    "排序最高的文本变量类别": (
        "最后，本章分析了不同类文本变量", "U\\_W；"),
}


def main() -> None:
    if OUT.exists() or REPORT.exists():
        raise FileExistsError("审核副本已存在，拒绝覆盖")
    with RAW.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    with Path(ga.MAIN_CSV).open(encoding="utf-8-sig", newline="") as f:
        old = list(csv.DictReader(f))
    kb = ga.load_kb_by_doc()
    plan = {doc: excerpts for doc, excerpts, _ in gen.plan_documents(
        kb, old, seed=20260927, limit=10)}
    repaired = []
    for row in rows:
        q = row["question"]
        matches = [key for key in REPAIRS if key in q]
        if len(matches) > 1:
            raise ValueError(f"引文规则重复命中: {q}")
        if matches:
            start, end = REPAIRS[matches[0]]
            pages = set(row["evidence_pages"].split("|"))
            candidates = [e["text"] for e in plan[row["gold_docs"]]
                          if str(e["page_start"]) in pages]
            spans = []
            for source in candidates:
                if start in source and end in source:
                    left = source.index(start)
                    right = source.index(end, left) + len(end)
                    spans.append(source[left:right])
            if len(spans) != 1:
                raise ValueError(f"未能唯一定位完整原文: {q}")
            row["gold_chunks"] = spans[0]
            row["notes"] += "；助手据原文补全兜底引文，仍需人工审核"
            repaired.append(q)
        if "哪三种机器学习模型进行股票预测" in q:
            row["answer"] = "支持向量机、XGBoost 和神经网络。"
            row["notes"] += "；答案删去原文未写出的缩写 SVM"
        assert row["status"] == "pending"
        assert any(row["gold_chunks"] in c["text"] for c in kb[row["gold_docs"]])
    if len(repaired) != len(REPAIRS):
        raise ValueError(f"引文修正数量异常: {len(repaired)}")
    with OUT.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ga.CAND_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tiers = Counter(r["tier"] for r in rows)
    lines = ["# 新生成候选的证据核对", "",
             f"- 来源：10 篇论文，{len(rows)} 条候选；旧正式题库未改动。",
             f"- 机器分级：{tiers['绿']} 绿、{tiers['黄']} 黄、{tiers['红']} 红；全部仍为 pending。",
             f"- {len(repaired)} 条原始兜底引文已按所选原文片段补全；保留 quote_fallback=1 作为生成问题记录。",
             "- 1 条答案删去未在逐字引文中写出的 SVM 缩写。",
             "- 本轮机器质检最初依据全部四段摘录；因此 verify=ok 不能单独证明所引句子完整支持答案。已逐条对照候选引文与答案，黄色条目仍需 PDF 原件审核。",
             "- 每条引文均为对应论文知识库原文子串；这只能证明引文真实，不等于已完成独立人工审题。",
             "- 黄色候选优先核对 PDF 原件，尤其是 OCR 把 0.01 拆成两行及图示提取文本的两题。",
             "- 新题来自同一批论文，不能视作独立真实用户流量来估计 1% 误报率。", "",
             "## 已补全引文的候选", ""]
    lines.extend(f"- {q}" for q in repaired)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT)
    print(REPORT)


if __name__ == "__main__":
    main()
