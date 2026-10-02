"""Readable Chinese development report; includes all unresolved cases."""
import argparse
import csv
from pathlib import Path

from rag_fresh_change_experiment import load


def read_csv(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def report(output,log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei","DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"]=False
    summary=load(output/"analysis_summary.json");audit=load(output/"integrity_audit.json")
    scopes=read_csv(output/"scope_evidence_summary.csv");rows=read_csv(output/"answer_scores.csv")
    pairs=read_csv(output/"paired_quality.csv");metrics={m["condition"]:m for m in summary["metrics"]}
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    conditions=("cached","mineru_retrieved")
    labels=["原缓存","全文检索"]
    for ax,title,key,states,names,colors in (
       (axes[0],"证据语义支持（12题）","evidence",("complete","incomplete","U"),("完整","不完整","U"),("#329e78","#d58b36","#b1b8c1")),
       (axes[1],"新答案质量（各12条）","quality",("2","1","0","U"),("完整","部分","错误/未答","U"),("#329e78","#e6b458","#c56161","#b1b8c1")),
       (axes[2],"新代理状态（各12条）","proxy_status",("complete","incomplete","U"),("完整","不完整","U"),("#329e78","#d58b36","#b1b8c1"))):
        bottom=[0,0]
        for state,name,color in zip(states,names,colors):
            if key=="evidence":
                values=[sum(r["cached_evidence" if c=="cached" else "retrieved_evidence"]==state for r in scopes) for c in conditions]
            else:
                values=[sum(r["condition"]==c and r[key]==state for r in rows) for c in conditions]
            ax.bar(labels,values,bottom=bottom,label=name,color=color,width=.6)
            for i,v in enumerate(values):
                if v:
                    ax.text(i,bottom[i]+v/2,str(v),ha="center",va="center",fontsize=11)
            bottom=[a+b for a,b in zip(bottom,values)]
        ax.set_title(title,fontsize=12);ax.set_ylim(0,13);ax.set_ylabel("数量")
        ax.legend(loc="upper center",bbox_to_anchor=(.5,-.09),ncol=2,fontsize=9,frameon=False)
    fig.suptitle("V4 开发验证：已用题的新回答；灰色 U 全部保留",fontsize=13)
    for extension in ("png","svg"):
        fig.savefig(output/("retrieval_comparison."+extension),dpi=180)
    plt.close(fig)
    lines=["# MinerU 全文检索与覆盖代理 V4：开发验证结果","",
       "## 本轮范围","",
       "沿用上一轮12道已用题，固定原缓存和实际可见论文范围内的全文检索证据各生成12条新回答，另加1条固定缺参数对照。全部问题、依据、检索和引用校验规则在生成前冻结，未按回答结果选题。全文检索沿用2000字符窗口、1500步长、字符TF-IDF前20候选、BGE 4096 token重排、前3段的配置。标准事实不参与检索。",
       "",
       "本轮属于已用题上的开发验证，不能作为独立泛化结果，也不能推导在线变点检测的误报保证。新方法未替换已冻结的V3结果或修改线上RAG服务。",
       "","## 引用校验与判分修改","",
       "V4先核对真实原文子串，再识别配对的Markdown展示标记，并保存回答原始位置。裸AUC/F1等短实体仅在边界正确且唯一时可定位，随后扩展真实上下文；虚构句子、改变数值和数学运算符不会被近似匹配。每轮最多补做一次引用复制，原语义状态和遗漏列表不允许改变。两轮未达成一致仍为U。",
       "",
       "离线质量总分由验证后的逐点状态导出，消除了‘总分完整但逐点部分回答’的内部矛盾：全部covered为2，有部分覆盖为1；有实质矛盾或全部missing为0；未定/分歧/截断为U。依据由问题和原始MinerU片段重新提取、两轮检查，生成时不可见。它仍是自动代理依据，尚不能视为人类金标准。",
       "","## 结果","",
       "|条件|新答案数|完整答案|不完整答案|质量U|代理U|完整答案误报|不完整检出|不完整漏检|不完整未定|拒答|",
       "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c,label in (("cached","原缓存"),("mineru_retrieved","全文检索"),("known_omission_control","缺参数对照"),("all","全部")):
        v=metrics[c]
        lines.append("|"+"|".join([label]+[str(v[k]) for k in ("n","complete","incomplete","quality_u","proxy_u","false_flags","flagged_incomplete","missed_incomplete","unresolved_incomplete","refusals")])+"|")
    lines += ["",f"两轮依据审查一致：{summary['scope_valid']}/12题；临时检索代理依据有效：{summary['proxy_plans_valid']}/12题。依据一致的题中，语义完整证据为原缓存 {summary['cached_semantic_complete']}/{summary['scope_valid']}，全文检索 {summary['retrieved_semantic_complete']}/{summary['scope_valid']}。",
       "",f"可确定的配对答案中：质量提升 {summary['quality_improved_pairs']} 对，降低 {summary['quality_worsened_pairs']} 对；另有 {summary['pair_quality_u']} 对至少一侧质量U。",
       "",f"![固定条件对照]({(output/'retrieval_comparison.png').resolve().as_posix()})","",
       "|题号|新依据有效|原缓存证据|全文检索证据|原缓存质量|全文检索质量|临时代理依据有效|",
       "|---|---:|---|---|---|---|---:|"]
    by_pair={p["qid"]:p for p in pairs}
    for s in scopes:
        p=by_pair[s["qid"]]
        lines.append("|"+"|".join([s["qid"],s["reference_valid"],s["cached_evidence"],s["retrieved_evidence"],p["cached_quality"],p["retrieved_quality"],s["proxy_valid"]])+"|")
    lines += ["","## 两个范围争议的处理","",
       "这些审查只查看问题和原文，未查看本轮新答案。原来的规范依据保持不变，以下是新版本的开发依据：",""]
    private=load(output/"private_question_key.json")
    for qid in ("fin_007","fin_022"):
        bid=next(b for b,k in private.items() if k["qid"]==qid)
        ref=load(output/"references"/(bid+".json"))
        lines += [f"### {qid}","",ref["question"],"",f"新依据有效：{ref['basis_complete']}。",""]
        for point in ref["points"]:
            lines.append("- "+point["point"])
        if not ref["basis_complete"]:
            lines += ["", "审查未达成一致，保留U。"]
        lines.append("")
    lines += ["## 固定门槛与结论","",f"本轮预设开发门槛：**{summary['gate']['status']}**。",""]
    for k,v in summary["gate"]["checks"].items():
        lines.append(f"- {k}: {v}")
    causes=load(output/"failure_cause_summary.json")
    lines += ["","本门槛包含依据可用性、全文证据完整率、代理U比例、完整回答误报、至少5条不完整答案及其检出率、固定缺参数对照。U始终留在分母中，没有把未定项算作检出成功。即使判定题上没有误报，仍须同时报告代理U和样本量。",
       "",f"质量U原因：{causes['quality_u_causes']}；代理U原因：{causes['proxy_u_causes']}。本轮逐字引用没有造成最终U，主要限制已转为依据范围、逐点输出格式和两轮语义分歧。6条自动判定完整回答中有{causes['good_proxy_u']}条代理U。",
       "","1条表观漏检是fin_019全文检索回答。其离线依据含5个点，临时代理含4个点，涉及小波分解、不同频率空间/子序列及去噪的相近表述；离线审查判部分，临时代理判完整。由于两侧事实粒度和同义判定不同，该项只能作为相对于自动依据的表观漏检，尚无独立真值支持。详见failure_causes.md及保留的原始逐点判断。",
       "","## 后续安排","",
       "优先把定义/优点/如何建模等范围易变的题改为范围明确的客观问题，作为新候选题而非覆盖旧题：明确论文对象、需要列出的步骤/指标集合、参数前后口径，避免用非必要细节判错。候选应由MinerU原文支持并在生成答案前通过两轮范围审查。仍无法一致的候选留为pending。",
       "",
       "全文检索有改善的题继续保留配对比较；全文有事实但前3段没有充分支持的题，下一轮单独评估召回和重排。仅在固定新题上验证代理可用性与误报/漏检后，再开展新信号的长序列变点实验。",
       "","## 审计与限制","",
       f"审计：{audit['status']}；{summary['answers']}条新回答，{summary['api_responses']}个唯一新API响应，{summary['total_tokens']} tokens，实际模型：{summary['models']}。原文哈希、检索片段位置、公开输入白名单、生成前冻结、逐字回答位置和从原始响应重算分数均已检查。",
       "",
       "请求别名是deepseek-chat，实际返回deepseek-flash。生成、依据审查与评分来自同一供应商；两轮一致只能说明内部稳定，不能代替外部独立评价。全文检索限于原缓存实际可见论文范围，本轮没有测量跨全库来源选择能力。只有小样本、已用题和固定故障，不支持真实线上总体性能结论。",""]
    text="\n".join(lines)
    (output/"v4_development_report.md").write_text(text,encoding="utf-8")
    log.parent.mkdir(parents=True,exist_ok=True);log.write_text(text,encoding="utf-8")
    print("Report saved",log)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,required=True);ap.add_argument("--log",type=Path,required=True)
    a=ap.parse_args();report(a.output,a.log)
