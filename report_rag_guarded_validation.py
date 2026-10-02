"""Report V8 prospective new-question/source-consistency validation."""
import argparse
import csv
from pathlib import Path

from rag_fresh_change_experiment import load


def report(output,log):
    summary=load(output/"analysis_summary.json");audit=load(output/"integrity_audit.json")
    by={r["condition"]:r for r in summary["metrics"]};live=by["live_normal"];bad=by["controls"]
    n=summary["qualified_questions"];attempts=summary["admission_attempts"]
    lines=["# 新题验证V8：来源盲提取准入与严格单位规则","","## 结论","",
        f"预设自动验证门槛：**{summary['gate']['status']}**。尝试{attempts}道候选，程序有效{summary['program_valid_candidates']}，来源盲提取与参考一致{summary['blind_consistency_pass']}，最终实验准入{n}题/{summary['qualified_sources']}篇来源。",
        f"正常新回答：自动参考评分完整{live['complete']}/{n}，不完整{live['incomplete']}，质量未定{live['quality_u']}。检索核对依据有效{summary['proxy_bases']}/{n}。",
        f"合成遗漏/改错对照：检出{bad['flagged_incomplete']}/{2*n}，确定漏检{bad['missed_incomplete']}，未定{bad['unresolved_incomplete']}。完整对照误报{by['control_complete']['false_flags']}/{n}，完整对照代理未定{by['control_complete']['proxy_u']}。",
        "全部未定保留在分母。没有按待测回答结果换题、改参考、重做检索或调整评分规则。候选继续保持pending，未进入正式标注集。","",
        "## 本轮新增且预先冻结的规则","",
        "- 候选来源来自原始MinerU table/text块，拒绝chart重建和明确近似数字；每段不超过900字符，每篇最多4段和2道候选。",
        "- 选题前整理历史结构化证据/引文的原文区间；候选来源块不得与已登记的区间重叠。不能保证项目其他未登记处理或供应商从未见过这些片段。",
        "- 来源卡片提取请求只有问题、公共字段与原文，没有拟定参考值、单位或参考引文；两轮来源检查后才与参考比较，最后再做两轮带参考的来源审查。",
        "- 不支持或不一致的候选不通过准入，不用拟定参考补齐盲提取。严格单位拆分只接受逐字来源中的单个数字与已知单位；不换算、不舍入。",
        "- 百分号仅输出一次，5%与无单位0.05不混同；公开字段、BGE检索和句子边界规则、生成配置、回答比较规则与拒答规则沿用上一版。","",
        "## 全部指标","","|条件|N|完整|不完整|质量U|代理U|错误检出|错误U|完整误报|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary["metrics"]:lines.append("|"+"|".join(str(r[k]) for k in (
        "condition","n","complete","incomplete","quality_u","proxy_u","flagged_incomplete","unresolved_incomplete","false_flags"))+"|")
    lines += ["","## 按论文分组","","|来源|新正常回答N|完整|不完整|质量U|代理U|","|---|---:|---:|---:|---:|---:|"]
    with (output/"live_source_groups.csv").open(encoding="utf-8-sig",newline="") as handle:
        for r in csv.DictReader(handle):lines.append("|"+"|".join(r[k] for k in ("source","n","complete","incomplete","quality_u","proxy_u"))+"|")
    lines += ["","## 解释边界","",
        "这是新题、已有论文来源的前瞻验证。来源和题目先冻结再生成待测回答，但论文不是整个项目未见的新文献。每篇最多2题，同文献题目与合成对照不视为独立样本。",
        "来源一致性、提取和答案评分仍使用同一家服务。多轮一致是自动检查，不是独立人工真值；MinerU文本未逐页独立核对。自动完整数不能写成独立确认的真实正确率。",
        "本轮正常回答是真实新生成，故障对照是在答案上改一个字段；没有生成真实证据故障回答，没有长序列变点试验。不能把错误对照检出率当作变点检测检出率。",
        "上一轮开发题的单位修复回放不混入本轮成绩，本轮也不改写上一轮的28/32及4个未定。",
        f"请求模型deepseek-chat；实际响应模型{summary['actual_models']}。新响应{summary['api_responses']}，tokens {summary['total_tokens']}；生成结束{summary['generation_finishes']}。",
        f"完整性审计{audit['status']}，覆盖来源位置、公开载荷、准入、时间顺序、原始请求和评分重放。审计不证明参考语义绝对正确。","",
        "## 下一步","",
        "若门槛通过，固定本轮正常检索建立核对依据，然后仅干预生成端证据，生成真实正常/故障配对。先核查故障是否确实降低质量，再冻结长序列与简单拒答、滑窗、CUSUM和e-detector比较。",
        "若未通过，先按全部未定和失败原因另立开发版本，不改写本轮题库、门槛和成绩。无需追加人工复核，结论保留自动评价的限定。","",
        f"![V8结果]({(output/'guarded_validation_results.png').resolve().as_posix()})",""]
    import os
    os.environ.setdefault("MPLCONFIGDIR",str((output/"mplconfig").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    fig,axes=plt.subplots(1,3,figsize=(12,4.4),layout="constrained")
    values=([attempts,summary["blind_consistency_pass"],n],[live["complete"],live["incomplete"],live["quality_u"]],
            [bad["flagged_incomplete"],bad["missed_incomplete"],bad["unresolved_incomplete"]])
    labels=(["Attempts","Consistent","Admitted"],["Complete","Incomplete","Unresolved"],["Flagged","Missed","Unresolved"])
    titles=["Pre-answer candidate screening","Conditional automatic quality","Synthetic bad-answer controls"]
    for ax,v,labels,title in zip(axes,values,labels,titles):
        ax.bar(range(3),v,color=["#277da1","#43aa8b","#9b9b9b"])
        ax.set_xticks(range(3),labels,fontsize=9);ax.set(title=title,ylabel="Count",ylim=(0,max(v)+3))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True));ax.spines[["right","top"]].set_visible(False)
        for i,x in enumerate(v):ax.text(i,x+.2,str(x),ha="center")
    fig.suptitle("V8: new questions, existing sources, frozen blind admission and unit rules",fontsize=13)
    fig.supxlabel("Source-conditioned automatic checks; no independent truth or live evidence-fault/stream claims.",fontsize=9)
    for ext in ("png","svg"):fig.savefig(output/("guarded_validation_results."+ext),dpi=180)
    plt.close(fig)
    content="\n".join(lines);(output/"guarded_validation_report.md").write_text(content,encoding="utf-8")
    log.mkdir(parents=True,exist_ok=True);(log/"新题来源盲提取验证V8结果_20261002.md").write_text(content,encoding="utf-8")
    print("Saved guarded new-question report",output/"guarded_validation_report.md")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True);parser.add_argument("--log",type=Path,required=True)
    args=parser.parse_args();report(args.output,args.log)
