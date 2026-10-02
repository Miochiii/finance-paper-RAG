"""Describe unresolved judgments without overwriting frozen scores."""
import argparse
from collections import Counter
from pathlib import Path

from rag_fresh_change_experiment import load,dump,write_csv


def unresolved_cause(plan,result):
    if not plan["basis_complete"]:
        return "basis_scope_unresolved"
    count=len(plan["points"])
    for raw in result["rounds"]:
        rows=raw.get("point_results",[])
        if not isinstance(rows,list) or len(rows)!=count or {r.get("index") for r in rows}!=set(range(1,count+1)):
            return "point_schema_mismatch"
    if any(result["signal"]["invalid_quotes"]):
        return "literal_quote_unresolved"
    if any(r["status"]=="U" for r in result["signal"]["rounds"]):
        return "point_semantics_unresolved"
    return "two_round_disagreement"


def run(output):
    keys=load(output/"private_case_key.json");qkeys=load(output/"private_question_key.json")
    rows=[];quality_causes=Counter();proxy_causes=Counter();misses=[]
    for aid,key in keys.items():
        bid=key["question_key"];score=load(output/"scores"/(aid+".json"))
        ref=load(output/"references"/(bid+".json"));proxy=load(output/"proxy_plans"/(bid+".json"))
        qcause=unresolved_cause(ref,score["reference_comparison"]) if score["quality"]=="U" else ""
        pcause=unresolved_cause(proxy,score["proxy_comparison"]) if score["proxy_comparison"]["signal"]["status"]=="U" else ""
        if qcause:quality_causes[qcause]+=1
        if pcause:proxy_causes[pcause]+=1
        rows.append(dict(aid=aid,qid=qkeys[bid]["qid"],condition=key["condition"],quality=score["quality"],
             proxy_status=score["proxy_comparison"]["signal"]["status"],quality_u_cause=qcause,proxy_u_cause=pcause))
        if score["quality"] in {"0","1"} and score["proxy_comparison"]["signal"]["status"]=="complete":
            misses.append(dict(aid=aid,qid=qkeys[bid]["qid"],condition=key["condition"],
                reference_points=[p["point"] for p in ref["points"]],proxy_points=[p["point"] for p in proxy["points"]],
                reference_verdicts=score["reference_comparison"]["rounds"],
                interpretation="apparent_miss_relative_to_automatic_reference_not_independent_human_error_label"))
    good=[r for r in rows if r["quality"]=="2"]
    value=dict(quality_u_causes=dict(quality_causes),proxy_u_causes=dict(proxy_causes),
               good_proxy_u=sum(r["proxy_status"]=="U" for r in good),good=len(good),apparent_misses=len(misses))
    dump(output/"failure_cause_summary.json",value)
    dump(output/"private_apparent_misses.json",misses)
    write_csv(output/"failure_causes.csv",rows)
    lines=["# V4 未定项与表观漏检原因","",
           f"质量U原因：{dict(quality_causes)}。代理U原因：{dict(proxy_causes)}。",
           "",f"6条自动判定完整回答中，代理U有{value['good_proxy_u']}条；零确定误报不能解释为这些回答全部正常通过。",
           "","## 表观漏检","",
           "本处的漏检以自动依据为参照，没有独立人工真值。逐点状态显示可能受事实重复、同义表达和范围边界影响，不更改冻结标签。",""]
    for case in misses:
        lines += [f"### {case['aid']} / {case['qid']}","","离线必要点：",""]
        lines += ["- "+p for p in case["reference_points"]]
        lines += ["","临时代理必要点：",""]+["- "+p for p in case["proxy_points"]]+[""]
    lines += ["## 后续处理","",
       "对于basis_scope_unresolved，先将问题改写为明确对象/名单/参数口径的新候选题并在生成前审查；对于point_schema_mismatch，另立版本约束逐点索引和数量，避免模型重新合并事实；对于two_round_disagreement和同义表达争议，保留原始两轮记录，增加来源支持的等价表述核查。以上后续修订均需新冻结和新回答验证。",""]
    (output/"failure_causes.md").write_text("\n".join(lines),encoding="utf-8")
    print(value)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--output",type=Path,required=True)
    run(ap.parse_args().output)
