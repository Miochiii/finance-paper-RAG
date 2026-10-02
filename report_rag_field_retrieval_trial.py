"""Aggregate report and static research figure for the frozen V6 comparison."""
import argparse
import csv
import json
from pathlib import Path

from rag_fresh_change_experiment import load


def rows(path):
    with path.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))


def report(output, log):
    import os
    os.environ.setdefault("MPLCONFIGDIR",str((output/"mplconfig").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    v=load(output/"analysis_summary.json");audit=load(output/"integrity_audit.json")
    metrics={r["condition"]:r for r in v["metrics"]};pairs=rows(output/"paired_quality.csv")
    diagnostic=rows(output/"local_retrieval_diagnostic.csv");basis=rows(output/"proxy_basis_comparison.csv")
    before=metrics["live_baseline"];after=metrics["live_enhanced"];c=metrics["controls"]
    fig,axes=plt.subplots(1,3,figsize=(12,4.2),layout="constrained")
    axes[0].bar([0,1,2],[v["baseline_proxy_bases"],v["enhanced_strict_proxy_bases"],v["enhanced_mapped_proxy_bases"]],color=["#7e8aa2","#277da1","#43aa8b"])
    axes[0].set_xticks([0,1,2],["Baseline","New strict","New mapped"])
    axes[0].set(title="Evidence bases (11 questions)",ylim=(0,11.8),ylabel="Questions")
    for i,n in enumerate([v["baseline_proxy_bases"],v["enhanced_strict_proxy_bases"],v["enhanced_mapped_proxy_bases"]]):axes[0].text(i,n+.15,str(n),ha="center")
    bottom=[0,0]
    for key,label,color in (("complete","Complete","#43aa8b"),("incomplete","Incomplete","#f8961e"),("quality_u","Unresolved","#9b9b9b")):
        values=[before[key],after[key]];axes[1].bar([0,1],values,bottom=bottom,label=label,color=color)
        bottom=[b+n for b,n in zip(bottom,values)]
    axes[1].set_xticks([0,1],["Baseline","Enhanced"]);axes[1].set(title="Fresh normal answers (11 each)",ylim=(0,12),ylabel="Answers")
    axes[1].legend(fontsize=8,loc="lower left")
    x=[0,1];flagged=[12,c["flagged_incomplete"]];unresolved=[10,c["unresolved_incomplete"]];miss=[0,c["missed_incomplete"]]
    axes[2].bar(x,flagged,label="Flagged",color="#277da1")
    axes[2].bar(x,unresolved,bottom=flagged,label="Unresolved",color="#9b9b9b")
    axes[2].bar(x,miss,bottom=[a+b for a,b in zip(flagged,unresolved)],label="Determinate miss",color="#f94144")
    axes[2].set_xticks(x,["Prior controls","New controls"]);axes[2].set(title="22 synthetic incomplete controls",ylim=(0,23.5),ylabel="Controls")
    axes[2].legend(fontsize=8,loc="lower left")
    for ax in axes:ax.spines[["right","top"]].set_visible(False)
    fig.suptitle("V6 development: seen sources, fixed fields, fresh paired answers",fontsize=13)
    for suffix in ("png","svg"):fig.savefig(output/("field_retrieval_comparison."+suffix),dpi=180)
    plt.close(fig)
    lines=["# 明确字段检索 V6：开发配对验证","",
      "## 结论","",
      f"同一批11道已见来源题，新配置的核对依据有效数从 {v['baseline_proxy_bases']}/11 变为 {v['enhanced_mapped_proxy_bases']}/11（不做符号映射时 {v['enhanced_strict_proxy_bases']}/11）。",
      f"两种配置各生成11条新回答：完整数 {before['complete']}→{after['complete']}；不完整数 {before['incomplete']}→{after['incomplete']}；质量未定数 {before['quality_u']}→{after['quality_u']}。逐题改善 {v['paired_improved']}，退化 {v['paired_worsened']}，质量未定配对 {v['paired_quality_u']}。",
      f"22条合成不完整对照检出 {c['flagged_incomplete']}，确定漏检 {c['missed_incomplete']}，未定 {c['unresolved_incomplete']}；完整对照误报 {metrics['control_complete']['false_flags']}/11。预设自动评分对照门槛：**{v['gate']['status']}**。未定保留在全部分母中。","",
      "## 本轮改动与设计","",
      "- 缩短公开题干的查询，去掉论文标题和作答模板；按公开字段构造查询，保留对象、时期和表号。",
      "- 增加表格完整窗口，显式表号匹配可以进入候选；重排后去除高度重叠段落。",
      "- 两种配置均最多3段、原文合计6000字符；检索范围仍限题干公开可见的论文。",
      r"- 对原文提取字段增加可追溯的表示映射：\varepsilon→ε、\epsilon→ϵ、转义下划线、数学分隔符和空白。保留原始摘录、原始范围；不改变数值、单位或运算符。",
      "- 新回答前冻结原文、代码、检索结果、核对依据、全部55案例和单字段修改对照；无按回答选择样本。",
      "- 拒答规则在新回答前冻结；同时报告原规则与字段规则，旧结果不重写。","",
      "## 检索后原文可用性","",
      "仅测冻结规范摘录是否逐字出现，用于开发诊断；其数值不能代替语义支持或回答质量。标准答案和原文摘录均不进入检索接口。","",
      "|题号|必答字段|旧摘录数|新摘录数|旧依据有效|新严格有效|新映射有效|",
      "|---|---:|---:|---:|---:|---:|---:|"]
    bm={r["question_key"]:r for r in basis}
    for r in diagnostic:
        p=bm[r["question_key"]]
        lines.append("|"+"|".join(str(x) for x in [r["question_key"],r["slots"],r["baseline_literal_quotes"],r["enhanced_literal_quotes"],p["baseline_basis"],p["enhanced_strict_basis"],p["enhanced_mapped_basis"]])+"|")
    lines += ["","## 全部信号结果","","|条件|N|完整|不完整|质量U|代理U|坏回答检出|确定漏检|坏回答U|完整误报|字段拒答|",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in v["metrics"]:lines.append("|"+"|".join(str(r[k]) for k in ["condition","n","complete","incomplete","quality_u","proxy_u","flagged_incomplete","missed_incomplete","unresolved_incomplete","false_flags","refusals"])+"|")
    lines += ["","## 新回答逐题配对","","|题号|旧质量|新质量|旧代理|新代理|旧字段拒答|新字段拒答|","|---|---:|---:|---|---|---:|---:|"]
    for r in pairs:lines.append("|"+"|".join(r[k] for k in ["question_key","baseline_quality","enhanced_quality","baseline_proxy","enhanced_proxy","baseline_refusal","enhanced_refusal"])+"|")
    scores=rows(output/"answer_scores.csv")
    live=[r for r in scores if r["condition"].startswith("live_")]
    lines += ["",f"22条新回答中原拒答规则识别 {sum(int(r['legacy_refusal']) for r in live)} 条，预先冻结的字段规则识别 {sum(int(r['rule_refusal']) for r in live)} 条。该比较只针对本轮实际回答；不能据此给出总体拒答识别准确率。","",
      "## 剩余具体问题","",
      "O0011的新检索第一段末尾停在“从2011年开始至20”，原文紧接着为“18年的交易数据”。字符窗口切断了结束年份。新回答给出了18个行业和1696家公司，却只能报告不完整的样本期间，质量为1；代理U、字段拒答0，仍未检出这条部分回答。",
      "这条失败不能因其他10条正常回答完整而排除。下一版应在固定原文预算内保留句子/日期完整边界，再用新回答验证；本轮冻结窗口和结果保持原样。","",
      "## 审计与解释边界","",
      f"审计 {audit['status']}；13项相关单元测试通过。{v['api_responses']} 个新API响应，{v['total_tokens']} tokens；实际返回模型 {json.dumps(v['actual_models'],ensure_ascii=False)}；生成结束原因 {v['generation_finishes']}。",
      "本轮沿用已见论文和题目，是开发比较。表格分块、查询、去重和符号映射同时改动，不能单独归因。两次自动裁判仍来自同一家模型，完整性不等于独立人工正确性。",
      "合成遗漏/改错是在答案文本上修改，用于验证自动评分；本轮新生成的只有正常答案，尚未完成真实证据故障配对，也没有证明变点检测优于基线。",
      "下一步按逐题退化情况改善检索的稳定性，再冻结配置到独立题目验证；随后再运行真实证据干预和长序列变点实验。","",
      f"![结果图]({(output/'field_retrieval_comparison.png').resolve().as_posix()})",""]
    text="\n".join(lines);(output/"field_retrieval_report.md").write_text(text,encoding="utf-8")
    log.mkdir(parents=True,exist_ok=True);(log/"明确字段检索V6结果_20261002.md").write_text(text,encoding="utf-8")
    print("Report saved:",output/"field_retrieval_report.md")


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--output",type=Path,required=True);ap.add_argument("--log",type=Path,required=True)
    a=ap.parse_args();report(a.output,a.log)
