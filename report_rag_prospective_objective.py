"""Prospective question/source-group report, with preparation correction ledger."""
import argparse
import csv
from pathlib import Path

from rag_fresh_change_experiment import load


def rows(path):
    with path.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))


def report(output,log):
    import os
    os.environ.setdefault("MPLCONFIGDIR",str((output/"mplconfig").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    v=load(output/"analysis_summary.json");audit=load(output/"integrity_audit.json");m=load(output/"protocol_lock.json")
    metrics={r["condition"]:r for r in v["metrics"]};live=metrics["live_normal"];control=metrics["controls"]
    n=v["qualified_questions"];fig,axes=plt.subplots(1,3,figsize=(12,4.2),layout="constrained")
    axes[0].bar([0,1],[n,v["proxy_bases"]],color=["#7e8aa2","#277da1"])
    axes[0].set_xticks([0,1],["Admitted questions","Available bases"]);axes[0].set(title=f"Prospective evidence ({v['qualified_sources']} source groups)",ylim=(0,n+2),ylabel="Questions")
    for i,value in enumerate([n,v["proxy_bases"]]):axes[0].text(i,value+.15,str(value),ha="center")
    values=[live["complete"],live["incomplete"],live["quality_u"]]
    axes[1].bar([0,1,2],values,color=["#43aa8b","#f8961e","#9b9b9b"])
    axes[1].set_xticks([0,1,2],["Complete","Incomplete","Unresolved"]);axes[1].set(title=f"{n} fresh normal answers",ylim=(0,n+2),ylabel="Answers")
    for i,value in enumerate(values):axes[1].text(i,value+.15,str(value),ha="center")
    conditions=["control_complete","control_omitted","control_corrupted"]
    clear=[metrics[c]["n"]-metrics[c]["proxy_u"]-metrics[c]["false_flags"] if c=="control_complete" else metrics[c]["missed_incomplete"] for c in conditions]
    flag=[metrics[c]["false_flags"] if c=="control_complete" else metrics[c]["flagged_incomplete"] for c in conditions]
    unresolved=[metrics[c]["proxy_u"] for c in conditions]
    axes[2].bar([0,1,2],clear,label="Complete verdict",color="#43aa8b")
    axes[2].bar([0,1,2],flag,bottom=clear,label="Incomplete verdict",color="#277da1")
    axes[2].bar([0,1,2],unresolved,bottom=[a+b for a,b in zip(clear,flag)],label="Unresolved",color="#9b9b9b")
    axes[2].set_xticks([0,1,2],["Full controls","Omitted","Corrupted"]);axes[2].set(title="Synthetic evaluator controls",ylim=(0,n+2),ylabel="Controls")
    axes[2].legend(fontsize=8,loc="lower left")
    for ax in axes:ax.spines[["right","top"]].set_visible(False)
    fig.suptitle("Prospective new questions: frozen V6.1, existing corpus, exact source cards",fontsize=13)
    for ext in ("png","svg"):fig.savefig(output/("prospective_objective_comparison."+ext),dpi=180)
    plt.close(fig)
    screen=rows(output/"candidate_screening.csv");groups=rows(output/"live_source_groups.csv")
    scores=rows(output/"answer_scores.csv");bases=rows(output/"proxy_basis_availability.csv")
    lines=["# 前瞻新题验证V7.1：冻结V6.1方法","","## 结论","",
      f"从8篇候选来源各尝试2题，{v['qualified_questions']}/{v['maximum_candidates']}题经逐字来源校验、两轮来源检查及去重进入实验，覆盖{v['qualified_sources']}篇论文。候选状态保持pending，未进入正式标注集。",
      f"新生成正常回答：完整 {live['complete']}/{n}，不完整 {live['incomplete']}，质量未定 {live['quality_u']}；核对依据有效 {v['proxy_bases']}/{n}。",
      f"{2*n}条合成遗漏/改错对照：检出 {control['flagged_incomplete']}，确定漏检 {control['missed_incomplete']}，未定 {control['unresolved_incomplete']}；完整对照误报 {metrics['control_complete']['false_flags']}/{n}。",
      f"本轮预先锁定的验证门槛：**{v['gate']['status']}**。所有未定和失效依据均计入分母。","",
      "## 来源与新题范围","",
      "8篇论文未进入此次覆盖代理的开发或筛选来源集，但均属于原知识库且有旧标注题。本轮使用新问题，冻结检索和评分后再生成回答，属于前瞻验证；不是整个项目从未见过这些论文，也不是独立人工真值。",
      "每篇最多2题，报告按论文分组；不能把同文献的题目和合成对照视为完全独立观测。本轮未计算依赖独立同分布假设的置信区间。","",
      "## 测试回答生成前的来源纠正","",
      "首版选题用统一[TABLE_START]标签识别表格，但该标签也包裹MinerU的chart重建内容。首版11道机器审查通过题中，有4道使用了chart证据，其中部分含约数/~标记。首版未生成任何待测回答，也未用于评分结果调参。",
      "首版完整记录保留。随后在8篇来源上统一重建候选：依据MinerU原始block.type选择table/text，保留原块索引、页码、原文位置；拒绝chart和明确的数字近似标记。每篇最多4段、每段900字符。",
      "这次改动纠正候选证据来源。检索参数、句子边界规则、生成提示词/温度/长度、代理提取/核查、两轮比较、拒答规则以及预设评价门槛均沿用冻结V6.1方法。","",
      "## 预设门槛","",
      "- 至少10题、6篇来源；正常新回答完整比例至少80%，质量U至多10%，代理U至多20%。",
      "- 合成对照代理U至多20%；完整对照误报0；遗漏和改错各至少检出80%。",
      "- 来源核验、题库、检索结果、核对依据及全部案例均先于新回答冻结；结果出现后不调参数、重选题目或降低门槛。","",
      "|检查项|通过|","|---|---|"]
    for k,value in v["gate"]["checks"].items():lines.append(f"|{k}|{value}|")
    lines += ["","## 全部信号结果","","|条件|N|完整|不完整|质量U|代理U|坏回答检出|确定漏检|坏回答U|完整误报|",
      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in v["metrics"]:lines.append("|"+"|".join(str(r[k]) for k in ["condition","n","complete","incomplete","quality_u","proxy_u","flagged_incomplete","missed_incomplete","unresolved_incomplete","false_flags"])+"|")
    lines += ["","## 正常新回答：按来源分组","","|论文|新回答N|完整|不完整|质量U|代理U|","|---|---:|---:|---:|---:|---:|"]
    for r in groups:lines.append("|"+"|".join(r[k] for k in ["source","n","complete","incomplete","quality_u","proxy_u"])+"|")
    lines += ["","## 正常新回答：逐题保留","","|题号|质量|代理|字段拒答|核对依据有效|","|---|---:|---|---:|---:|"]
    bm={r["question_key"]:r for r in bases}
    for r in scores:
        if r["condition"]=="live_normal":lines.append("|"+"|".join([r["question_key"],r["quality"],r["proxy_status"],r["rule_refusal"],bm[r["question_key"]]["valid"]])+"|")
    lines += ["","## 解释边界和下一步","",
      "新题、来源分组和冻结配置使结果比反复调试11道开发题更有说服力；参考字段和评分仍由同一家模型自动检查，以MinerU表格/正文为条件，不是独立人工或逐页PDF复核。",
      "合成对照修改答案文本，检验的是评分能力。本轮没有生成真实证据干预后的回答，没有长序列检测结果，因此不能据此宣称变点检测的检出率或误报率。",
      "若本轮门槛通过，下一步用冻结的正常检索建立核对依据，再单独干预生成端证据，做真实正常/故障配对；随后才进入长序列变点及基线比较。若门槛未通过，按全部失败和未定原因开下一轮开发，原验证结果保留。",
      f"审计 {audit['status']}。来源选择及新协议相关测试通过；新API响应 {v['api_responses']}，tokens {v['total_tokens']}，实际模型 {v['actual_models']}，生成结束原因 {v['generation_finishes']}。","",
      f"![结果图]({(output/'prospective_objective_comparison.png').resolve().as_posix()})",""]
    content="\n".join(lines);(output/"prospective_objective_report.md").write_text(content,encoding="utf-8")
    log.mkdir(parents=True,exist_ok=True);(log/"前瞻新题验证V7_1结果_20261002.md").write_text(content,encoding="utf-8")
    print("Saved prospective report",output/"prospective_objective_report.md")


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--output",type=Path,required=True);ap.add_argument("--log",type=Path,required=True)
    a=ap.parse_args();report(a.output,a.log)
