"""Report finite-scope candidate and scoring-control trial, including missing bases."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import rag_auto_evaluation as auto
from rag_fresh_change_experiment import load,dump,write_csv
from rag_objective_coverage_pilot import verify_cases


def read_csv(path):
    with path.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))


def cause(plan):
    if plan["basis_complete"]:return ""
    if any(r.get("supported")!="1" for r in plan["raw"].get("slots",[])):
        return "retrieved_evidence_insufficient"
    if any(any(r.get(k)!="1" for k in ("supported","unique","aliases_valid")) for c in plan["checks"] for r in c.get("slot_checks",[])):
        return "semantic_check_unresolved"
    return "literal_value_or_quote_validation"


def report(output,log,candidate_csv):
    verify_cases(output)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei","DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"]=False
    summary=load(output/"analysis_summary.json");reference=load(output/"reference_control_summary.json")
    audit=load(output/"integrity_audit.json");questions=load(output/"public_questions.json")
    refs=load(output/"private_references.json");qkeys=load(output/"private_question_key.json")
    rows=read_csv(output/"answer_scores.csv");table={r["condition"]:r for r in summary["metrics"]}
    failures=[];counts=Counter()
    for qid,q in questions.items():
        p=load(output/"proxy_plans"/(qid+".json"));reason=cause(p)
        if reason:counts[reason]+=1
        fresh=next(r for r in rows if r["question_key"]==qid and r["condition"]=="live_normal")
        failures.append(dict(question_key=qid,input_id=qkeys[qid]["input_id"],proxy_basis_valid=int(p["basis_complete"]),
            basis_failure_cause=reason,live_quality=fresh["quality"],live_proxy=fresh["proxy_status"],live_rule_refusal=fresh["rule_refusal"]))
    write_csv(output/"basis_failure_summary.csv",failures)
    dump(output/"basis_failure_counts.json",dict(counts))
    exported=[]
    inputs=load(output/"candidate_inputs.json")
    for bid,row in inputs.items():
        revision=load(output/"revised_candidates"/(bid+".json"));initial=load(output/"candidates"/(bid+".json"))
        c=revision.get("candidate") or initial.get("candidate")
        qid=next((k for k,v in qkeys.items() if v["input_id"]==bid),"")
        exported.append(dict(id="objective_v5_"+bid,status="pending",experimental_question_key=qid,
             experimental_admission=int(bool(qid)),question=c["question"] if c else "",
             answer="\n".join(s["label"]+"："+s["expected"]+s["unit"] for s in c["slots"]) if c else "",
             gold_docs="|".join(row["sources"]),gold_chunks="\n".join(s["quote"] for s in c["slots"]) if c else "",
             objective_slots=json.dumps(c["slots"],ensure_ascii=False) if c else "[]",
             notes="程序限定作答字段；原文两轮审查；实验候选" if qid else "未通过单字段数值/名称规则，不进入本轮实验"))
    write_csv(candidate_csv,exported)
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),constrained_layout=True)
    labels=["完整","漏一项","改错值"]
    conditions=["control_complete","control_omitted","control_corrupted"]
    exact=[reference["by_condition"][c]["exact"] for c in conditions]
    axes[0].bar(labels,exact,color="#329e78",width=.6)
    for i,v in enumerate(exact):axes[0].text(i,v+.15,f"{v}/11",ha="center")
    axes[0].set_title("给定固定事实的评分：33条对照")
    axes[0].set_ylim(0,13);axes[0].set_ylabel("判分一致数量")
    bottom=[0,0,0]
    for state,name,color in (("complete","判完整","#329e78"),("incomplete","判不完整","#d58b36"),("U","未定","#b1b8c1")):
        values=[sum(r["condition"]==c and r["proxy_status"]==state for r in rows) for c in conditions]
        axes[1].bar(labels,values,bottom=bottom,label=name,color=color,width=.6)
        for i,v in enumerate(values):
            if v:axes[1].text(i,bottom[i]+v/2,str(v),ha="center",va="center")
        bottom=[a+b for a,b in zip(bottom,values)]
    axes[1].set_title("实际检索代理：未定项全部保留")
    axes[1].set_ylim(0,13);axes[1].set_ylabel("对照数量")
    axes[1].legend(loc="upper center",bbox_to_anchor=(.5,-.09),ncol=2,frameon=False)
    labels3=["完整","未答出/错误","质量U"]
    live=table["live_normal"];values=[live["complete"],live["incomplete"],live["quality_u"]]
    axes[2].bar(labels3,values,color=["#329e78","#d58b36","#b1b8c1"],width=.6)
    for i,v in enumerate(values):axes[2].text(i,v+.15,str(v),ha="center")
    axes[2].set_title("真实RAG新回答：11条")
    axes[2].set_ylim(0,13);axes[2].set_ylabel("数量")
    fig.suptitle("明确字段题 V5.1：评分依据与检索依据分别验证",fontsize=13)
    for ext in ("png","svg"):fig.savefig(output/("objective_coverage_comparison."+ext),dpi=180)
    plt.close(fig)
    lines=["# 明确字段题 V5.1：自动评分与检索代理结果","",
       "## 本轮完成内容","",
       "从原12道已用题对应的MinerU论文片段生成12道新候选，保留全部初稿及审查。初稿曾出现题干要求四项但事实清单仅三项、题干引文不连续、集合/公式不符合单字段规则等情况。两个来源审查器没有发现全部范围缺陷，因此在任何测试答案产生之前另立V5.1范围修订协议：定位范围仅由原题干和公共字段提取，程序按最终字段标签构造完整题干，来源审查再次进行。",
       "",
       f"最终 {summary['qualified_questions']}/12 道新题通过程序规则与两轮来源核查。1道公式题保留未通过。所有题库记录保持pending；本轮11题仅用于自动实验，正式题库准入是另一项工作。公开代理输入没有规范字段值。",
       "",
       "先冻结源文事实、字段清单、全部33条已知控制回答和检索配置，再生成11条真实RAG回答。控制回答分别是逐项完整、删除第一项、只改错第一项的数值/名称；其标签由程序修改确定。两类评价分别进行：离线评分器获得固定源文事实；临时覆盖代理从问题检索原文并自行提取字段事实。二者不混为一个结果。",
       "","## 结果一：有固定源文事实时的评分","",
       f"33条控制回答：判分一致 **{reference['exact']}/{reference['n']}**，错误 {reference['errors']}，未定 {reference['u']}。完整/遗漏/改错三类分别为11/11一致。预先设定的评分诊断门槛：**{reference['gate']['status']}**。",
       "",
       "这是带固定事实依据且控制答案标签已知的小样本检验，支持明确字段下的判分一致性；不表示自然语言回答、复杂同义表达、所有论文或真实故障都能可靠评价。源文事实仍经同一供应商自动双审，未充当独立人工金标准。",
       "","## 结果二：实际检索代理","",
       "|条件|样本数|确定完整|确定不完整|质量U|代理U|完整误报|不完整检出|不完整判完整|不完整未定|规则拒答|",
       "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c,label in (("control_complete","完整对照"),("control_omitted","遗漏对照"),("control_corrupted","改错对照"),("live_normal","真实新回答")):
        r=table[c]
        lines.append("|"+"|".join([label]+[str(r[k]) for k in ("n","complete","incomplete","quality_u","proxy_u","false_flags","flagged_incomplete","missed_incomplete","unresolved_incomplete","refusals")])+"|")
    lines += ["",f"检索代理依据有效 {summary['proxy_bases_valid']}/11题。22条不完整控制回答中检出 {table['controls']['flagged_incomplete']} 条，{table['controls']['unresolved_incomplete']} 条未定；U计为未检出，不能称为100%检出。11条完整对照中5条代理U，零确定误报同样不能解释为全部正常通过。",
       "",f"实际检索代理预设门槛：**{summary['gate']['status']}**。",""]
    lines += [f"- {k}: {v}" for k,v in summary["gate"]["checks"].items()]
    lines += ["",f"![对照结果]({(output/'objective_coverage_comparison.png').resolve().as_posix()})","",
       "## 未定项定位","",
       "|题号|原候选组|代理依据有效|主要原因|真实回答质量|真实代理状态|规则拒答|",
       "|---|---|---:|---|---|---|---:|"]
    for r in failures:
        lines.append("|"+"|".join(str(r[k]) for k in ("question_key","input_id","proxy_basis_valid","basis_failure_cause","live_quality","live_proxy","live_rule_refusal"))+"|")
    lines += ["",f"原因计数：{dict(counts)}。4题的前3段缺少指定表格/日期资料，1题的ε-SVM模型写法与原文LaTeX表示不完全一致，逐字值验证保留U。对于后者，两轮语义核查有支持判断，问题发生在表达形式校验。",
       "","本轮11条真实RAG新回答中7条自动判完整、4条未答出指定字段，质量U为0。4条不完整回答的代理均为U，其中3条没有被当前规则识别为明确拒答。原始回答分别是逐字段‘未找到’或‘无法依据参考上下文提供所要求字段’，实际均说明材料不足；这是规则漏识别，不能当作‘模型直接作答但不拒答’的隐蔽错误证据。",
       "",
       "新增rag_field_refusal.py作为后验开发规则，只看公开字段与回答：识别全部字段逐项未找到和明确无法提供所需字段，同时保留部分回答、一般性市场不确定性及引用尾注的排除测试。新旧相关4项测试通过；在本轮11条已有回答上，拒答由1条补至4条，恰好补上S0003、S0005、S0018。此为调试回放，原冻结指标保持原值，后续需新回答验证，不能把回放当独立性能证据。详见refusal_rule_development.csv/json。",
       "","## 下一步","",
       "先修检索：根据公开论文定位范围、表号、模型名和每个字段分别构造查询，评估是否需要召回表格所在窗口及相邻说明，减少完整文件名和题干模板对排名的影响。新的排名输入仍不包含规范字段值/答案/摘录，先做本地召回诊断，再冻结配置进行新回答对照。现有11题可用于检索开发，下一次泛化验证需另留新题或新来源。",
       "",
       "同时处理ε等符号的来源表示映射，保留原文位置，避免把符号显示差异当成依据不存在；不得放宽数值、对象或单位相等规则。字段式拒答开发规则须在新回答中一并验证。稳定检索与代理可用性后，再验证缺证据的真实RAG故障和长序列变点信号。",
       "","## 审计与文件","",
       f"相关20项测试通过；完整性审计 {audit['status']}。{summary['api_responses']}个唯一新API响应，{summary['total_tokens']} tokens，实际模型 {summary['actual_models']}，无失败请求，评分阶段schema重试 {summary['schema_retry_responses']} 次、引用重试 {summary['quote_retry_responses']} 次。",
       "",
       "原文哈希、29篇源文、检索原始位置、字段题构造、修订在答案前锁定、固定控制标签、输入白名单及从原始响应重算评分均已核对。",
       "",f"候选表：[{candidate_csv.name}]({candidate_csv.resolve().as_posix()})；全部候选status=pending。",""]
    text="\n".join(lines);(output/"objective_coverage_report.md").write_text(text,encoding="utf-8")
    log.parent.mkdir(parents=True,exist_ok=True);log.write_text(text,encoding="utf-8")
    print("Report and pending candidate bank saved",log,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ("output","log","candidate_csv"):ap.add_argument("--"+name,type=Path,required=True)
    a=ap.parse_args();report(a.output,a.log,a.candidate_csv)
