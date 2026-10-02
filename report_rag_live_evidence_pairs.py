"""Report real paired generations under V9 numerical evidence corruption."""
import argparse
import csv
import os
from pathlib import Path
from rag_fresh_change_experiment import load


def report(output, log):
    s = load(output / "analysis_summary.json")
    audit = load(output / "integrity_audit.json")
    normal, fault = s["metrics"]
    n = s["qualified_pairs"]
    lines = ["# V9：生成端数值证据故障的真实配对实验", "", "## 结果", "",
        f"预设机制验证门槛：**{s['gate']['status']}**。输入{s['input_questions']}题，结构筛查可用{s['structural_eligible']}题，经证据盲提取与两轮检查准入{n}题/{s['qualified_sources']}篇论文；{s['excluded_before_answers']}题在生成前排除，全部保留原因。",
        f"生成{s['fresh_generations']}条真实新回答，无合成答案。正常完整{normal['complete']}/{n}，故障完整{fault['complete']}/{n}；正常完整→故障不完整{s['complete_to_bad']}/{n}对，配对质量U为{s['paired_quality_u']}。",
        f"故障中自动参考评分错误/不完整{fault['bad']}条，正常证据核对信号检出{fault['bad_flagged']}，确定漏检{fault['bad_missed']}，未定{fault['bad_unresolved']}。正常完整答案误报{normal['complete_false_flags']}/{normal['complete']}，完整答案核对U为{normal['complete_proxy_u']}。",
        f"故障错误/不完整且未拒答{fault['bad_without_refusal']}条，其中证据核对信号检出{fault['bad_without_refusal_flagged']}；故障回答拒答{fault['refusals']}条。", "",
        "## 预先冻结的实验方法", "",
        "- 采用V8.2的17题，在既有单例开发中澄清过的P0002表格列方向沿用至输入题库。论文和问题已经见过，不属于新题或新论文验证。",
        "- 正常检索片段直接采用冻结缓存，不重新排序。正常原文提取的临时核对依据及参考答案固定；故障检查用的临时值不进入监测评分。",
        "- 固定修改第1个数值字段：整数加1，小数加最后一个展示位的一个单位，例如0.0032→0.0033，保留正负号意义和百分号。这是人为设计的小数值扰动，不代表自然故障的分布。",
        "- 仅定位原文引文中的唯一数值位置；表格保留其他单元格，所有检索片段中同一引文的重复副本同步修改。仍有原值字面量的题目保守排除，包含无关表格中碰巧同值的情况；不对全文所有相同数字直接替换。",
        "- 生成前只给问题、公开字段和变更后的证据做盲提取，再做两轮来源检查。必须提取到变更值，其他必答字段与正常依据等价，才能准入。没有按生成答案挑选成功故障。",
        "- 通过强制预检才可调用生成；正常/故障载荷顺序按固定种子打散，生成模型配置完全相同。载荷仅有问题和上下文，不含故障条件、参考或评分计划。",
        "- 评分沿用冻结的两轮字段比较、实际回答引文核验及拒答规则。所有U保留在相应分母，按论文分组报告，不将同文献问题当作完全独立样本。", "",
        "## 全部回答指标", "", "|条件|N|完整|错误/不完整|质量U|核对U|检出|漏检|错误U|完整误报|拒答|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in s["metrics"]:
        lines.append("|" + "|".join(str(r[k]) for k in ("condition", "n", "complete", "bad", "quality_u", "proxy_u", "bad_flagged", "bad_missed", "bad_unresolved", "complete_false_flags", "refusals")) + "|")
    lines += ["", "## 生成前筛查", "", "|题号|结构可用|准入|修改副本数|残留原值数|结构原因|", "|---|---:|---:|---:|---:|---|"]
    for r in load(output / "intervention_screening.json"):
        lines.append("|" + "|".join(str(r[k]) for k in ("question_key", "structural", "qualified", "changed_copies", "residual_values", "structural_reason")) + "|")
    lines += ["", "## 预设门槛", "", "准入至少8对/6篇来源；正常完整比例≥80%；正常完整→故障不完整比例≥60%；故障实际错误/不完整检出比例≥80%；正常完整误报为0；回答质量U比例≤10%。", ""]
    for key, passed in s["gate"]["checks"].items():
        lines.append(f"- {key}: {passed}")
    lines += ["", "## 按论文分组", "", "|来源|条件|N|完整|不完整|质量U|检出|核对U|", "|---|---|---:|---:|---:|---:|---:|---:|"]
    with (output / "source_groups.csv").open(encoding="utf-8-sig", newline="") as handle:
        for r in csv.DictReader(handle):
            lines.append("|" + "|".join(str(r[k]) for k in ("source", "condition", "n", "complete", "bad", "quality_u", "bad_flagged", "proxy_u")) + "|")
    lines += ["", "## 结论范围与下一步", "",
        "本轮验证生成端证据数值被改动后的回答表现，以及来自正常原文的核对信号能否识别质量变化。正常知识库和监测依据仍可信；若整库、监测依据也同步遭到污染，结论不适用。筛查淘汰含重复正文/相同值的题目，因此也不能代表全部17题或自然流量。",
        "这是同一供应商自动提取/评分与MinerU来源条件下的结果，多轮一致不等于独立人工真值。离线参考和监测依据同出于正常原文，二者评分不是相互独立的证据。",
        "已验证真实故障回答，不是长序列变点检验。结果不能写成e-detector检出率，也不能证明其优于首次信号、滑窗或CUSUM。",
        "若本轮门槛通过，下一阶段先冻结新的序列协议与阈值校准/测试划分，再采集新的正常/故障回答，比较拒答、首次核对异常、滑窗、CUSUM与e-detector；不把本轮配对重复拼接当作新的独立流量。若未通过，保留本轮，另立开发版本处理失败原因。",
        f"实际模型{s['actual_models']}；请求模型deepseek-chat；响应{s['api_responses']}次，tokens {s['total_tokens']}，失败记录{s['failed_api_records']}，生成结束{s['generation_finishes']}。",
        f"数据流与评分重放审计：{audit['status']}。", ""]
    os.environ.setdefault("MPLCONFIGDIR", str((output / "mplconfig").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    sets = [([normal["complete"], fault["complete"], s["complete_to_bad"]], ["Normal complete", "Fault complete", "Complete to bad"], "Fresh paired answer quality"),
            ([fault["bad_flagged"], fault["bad_missed"], fault["bad_unresolved"], normal["complete_false_flags"]], ["Bad flagged", "Bad missed", "Bad unresolved", "Normal false flags"], "Fixed normal-evidence monitoring")]
    for ax, (values, labels, title) in zip(axes, sets):
        ax.bar(range(len(values)), values, color=["#277da1", "#e07a5f", "#909090", "#d4a373"][:len(values)])
        ax.set_xticks(range(len(values)), labels, fontsize=8, rotation=10)
        ax.set(title=title, ylabel="Count", ylim=(0, n+2))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.spines[["top", "right"]].set_visible(False)
        for i, value in enumerate(values):
            ax.text(i, value+.15, str(value), ha="center")
    fig.suptitle(f"V9: {n} qualified questions, {s['qualified_sources']} existing papers, {s['fresh_generations']} fresh answers", fontsize=12)
    fig.supxlabel("Controlled numeric evidence corruption; conditional automatic quality; no stream detector claim.", fontsize=9)
    for ext in ("png", "svg"):
        fig.savefig(output / ("live_evidence_results." + ext), dpi=180)
    plt.close(fig)
    content = "\n".join(lines) + "\n"
    (output / "live_evidence_report.md").write_text(content, encoding="utf-8")
    log.mkdir(parents=True, exist_ok=True)
    (log / "真实数值证据故障V9结果_20261002.md").write_text(content, encoding="utf-8")
    print("Saved", output / "live_evidence_report.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--log", type=Path, required=True)
    a = ap.parse_args()
    report(a.output, a.log)
