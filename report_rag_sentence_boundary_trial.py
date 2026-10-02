"""V6.1 report: boundary development fix and untouched unresolved outcomes."""
import argparse
import csv
import json
from pathlib import Path

from rag_fresh_change_experiment import load,dump,write_csv


def rows(path):
    with path.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))


def report(output,log):
    import os
    os.environ.setdefault("MPLCONFIGDIR",str((output/"mplconfig").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    v=load(output/"analysis_summary.json");audit=load(output/"integrity_audit.json")
    m=load(output/"protocol_lock.json");gate=load(output/"boundary_success_lock.json")
    metrics={r["condition"]:r for r in v["metrics"]};pairs=rows(output/"paired_quality.csv")
    before=metrics["live_baseline"];after=metrics["live_enhanced"];controls=metrics["controls"]
    target=next(r for r in pairs if r["question_key"]==gate["target_question"])
    checks=dict(target_basis=load(output/"proxy_plans"/(gate["target_question"]+".json"))["basis_complete"],
                target_new_answer_complete=target["enhanced_quality"]=="2",
                no_paired_quality_regression=v["paired_worsened"]==0 and v["paired_quality_u"]==0,
                no_repaired_quality_u=after["quality_u"]==0,
                synthetic_control_gate=v["gate"]["status"]=="passed",all22_new_answers=v["fresh_paired_answers"]==22)
    outcome=dict(status="passed" if all(checks.values()) else "not_passed",checks=checks,
                 scope="development_regression_repair_on_seen_questions_not_external_validation")
    dump(output/"boundary_success_summary.json",outcome)
    fig,axes=plt.subplots(1,3,figsize=(12,4.2),layout="constrained")
    axes[0].bar([0,1],[v["baseline_proxy_bases"],v["enhanced_mapped_proxy_bases"]],color=["#7e8aa2","#43aa8b"])
    axes[0].set_xticks([0,1],["V6 original","V6.1 repaired"]);axes[0].set(title="Evidence bases (11 questions)",ylim=(0,12),ylabel="Questions")
    for i,n in enumerate([v["baseline_proxy_bases"],v["enhanced_mapped_proxy_bases"]]):axes[0].text(i,n+.15,str(n),ha="center")
    bottom=[0,0]
    for key,label,color in (("complete","Complete","#43aa8b"),("incomplete","Incomplete","#f8961e"),("quality_u","Unresolved","#9b9b9b")):
        values=[before[key],after[key]];axes[1].bar([0,1],values,bottom=bottom,label=label,color=color)
        bottom=[a+b for a,b in zip(bottom,values)]
    axes[1].set_xticks([0,1],["V6 original","V6.1 repaired"]);axes[1].set(title="Fresh normal answers (11 each)",ylim=(0,12),ylabel="Answers")
    axes[1].legend(fontsize=8,loc="lower left")
    previous=load(Path(m["previous"])/"analysis_summary.json");prior=next(r for r in previous["metrics"] if r["condition"]=="controls")
    x=[0,1];flag=[prior["flagged_incomplete"],controls["flagged_incomplete"]];u=[prior["unresolved_incomplete"],controls["unresolved_incomplete"]]
    axes[2].bar(x,flag,label="Flagged",color="#277da1");axes[2].bar(x,u,bottom=flag,label="Unresolved",color="#9b9b9b")
    miss=[prior["missed_incomplete"],controls["missed_incomplete"]]
    axes[2].bar(x,miss,bottom=[a+b for a,b in zip(flag,u)],label="Determinate miss",color="#f94144")
    axes[2].set_xticks(x,["Prior V6 controls","V6.1 controls"]);axes[2].set(title="22 synthetic incomplete controls",ylim=(0,24),ylabel="Controls")
    axes[2].legend(fontsize=8,loc="lower left")
    for ax in axes:ax.spines[["right","top"]].set_visible(False)
    fig.suptitle("V6.1 development: fixed query/ranks, same 6000-character budget",fontsize=13)
    for ext in ("png","svg"):fig.savefig(output/("sentence_boundary_comparison."+ext),dpi=180)
    plt.close(fig)
    b=rows(output/"boundary_repair_summary.csv");repair_counts={status:sum(r["status"]==status for r in b) for status in sorted({r["status"] for r in b})}
    write_csv(output/"boundary_status_counts.csv",[dict(status=k,count=n) for k,n in repair_counts.items()])
    lines=["# 句子边界修复V6.1：开发回归验证","","## 结论","",
      f"固定同一批11题，检索核对依据有效数 {v['baseline_proxy_bases']}→{v['enhanced_mapped_proxy_bases']}；新生成完整回答 {before['complete']}→{after['complete']}；逐题改善 {v['paired_improved']}，退化 {v['paired_worsened']}，质量未定配对 {v['paired_quality_u']}。",
      f"针对上一轮年份截断题O0011，新回答质量 {target['baseline_quality']}→{target['enhanced_quality']}，代理 {target['baseline_proxy']}→{target['enhanced_proxy']}；预先锁定的修复成功门槛：**{outcome['status']}**。",
      f"22条合成遗漏/改错对照检出 {controls['flagged_incomplete']}，确定漏检 {controls['missed_incomplete']}，未定 {controls['unresolved_incomplete']}；完整对照误报 {metrics['control_complete']['false_flags']}/11。原自动评分门槛 **{v['gate']['status']}**。","",
      "## 具体改动","",
      "保留V6选定的3个段落、来源、排名和重排分数，仅在原文内修复段尾边界；每段最多2000字符，每题合计不超过6000字符。",
      "段尾在句中时，向后最多寻找256字符，补齐至句末/表格行末。在长度不足时，只从开头裁掉到一个完整边界；保护原窗口内完整包含的表格。数字小数点不当句号，成对数学分隔符内的换行不当边界。",
      "无法在预算内安全补齐则保留原窗口，并明确记录未解决。选择和边界修复均不接收规范字段值、规范摘录、答案或质量标签。",
      f"33段的边界状态：{json.dumps(repair_counts,ensure_ascii=False)}。未解决段尾的存在不直接等同于回答缺失，仍由语义核查判断。","",
      "## 设计与冻结","",
      "- 沿用已见11题及原参考字段。两种配置均重新生成11条正常回答，另有33条合成完整/遗漏/改错对照。",
      "- 检索结果、全部核对依据、55案例及修复成功门槛在本轮新回答前冻结。",
      "- 问题和来源范围、模型请求别名、生成温度/长度、评分提示词、符号映射及拒答规则沿用V6。",
      "- 两种配置的排名固定，本轮只比较预算内的边界修复；每题都保留，不依据回答重新选配置。","",
      "## 全部结果","","|条件|N|完整|不完整|质量U|代理U|坏回答检出|坏回答U|确定漏检|完整误报|",
      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in v["metrics"]:lines.append("|"+"|".join(str(r[k]) for k in ["condition","n","complete","incomplete","quality_u","proxy_u","flagged_incomplete","unresolved_incomplete","missed_incomplete","false_flags"])+"|")
    lines += ["","## 逐题新回答配对","","|题号|V6质量|V6.1质量|V6代理|V6.1代理|","|---|---:|---:|---|---|"]
    for r in pairs:lines.append("|"+"|".join(r[k] for k in ["question_key","baseline_quality","enhanced_quality","baseline_proxy","enhanced_proxy"])+"|")
    lines += ["","## 解释边界及下一步","",
      "这是针对已发现退化的开发回归验证，11题和论文都已见过。自动参考和两轮评分仍使用同一供应商，不能作为独立人工真值。",
      "合成遗漏/改错只检查答案评分能力，本轮新生成均为正常回答；尚未证明真实证据故障或长序列变点的检出效果。",
      "接下来冻结当前配置到来源或题目留出的验证集；通过后再做真实证据干预配对，并比较变点检测与简单阈值等基线。",
      f"审计 {audit['status']}；6项新边界测试通过。新增API响应 {v['api_responses']}，tokens {v['total_tokens']}，实际返回模型 {v['actual_models']}，结束原因 {v['generation_finishes']}。","",
      f"![结果图]({(output/'sentence_boundary_comparison.png').resolve().as_posix()})",""]
    text="\n".join(lines);(output/"sentence_boundary_report.md").write_text(text,encoding="utf-8")
    log.mkdir(parents=True,exist_ok=True);(log/"句子边界修复V6_1结果_20261002.md").write_text(text,encoding="utf-8")
    print("Repair success:",outcome);print("Report:",output/"sentence_boundary_report.md")


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--output",type=Path,required=True);ap.add_argument("--log",type=Path,required=True)
    a=ap.parse_args();report(a.output,a.log)
