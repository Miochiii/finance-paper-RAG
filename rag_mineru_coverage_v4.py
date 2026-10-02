"""Fixed v4 development trial: answer-hidden scope and fresh retrieval pairs.

Original 12 questions are already seen development cases. This trial measures
new responses, not independent held-out generalization. Proxy requests receive
only question, query-only retrieved evidence and the actual answer.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import rag_auto_evaluation as auto
from rag_fresh_change_experiment import digest, dump, load, sha, utc, write_csv
from rag_subtle_fault_pilot import batch, sources, COVERAGE_PROMPT, coverage_result
from rag_quote_span_repair import enrich_short_quotes, answer_payload
from rag_coverage_refinement import EXTRACT, MINIMUM, minimum_plan
from rag_answer_quote_v4 import COMPARE_V4, QUOTE_REPAIR, validated_comparison, repair_quotes, agreement_score
from rag_prospective_coverage_validation import verify_hashes, literal_failure
from rag_refusal import explicit_refusal

EXTRACT_V4 = EXTRACT + """\n将最低必要事实拆成可分别核对的原子点。不得把核心定义和可选实现细节混在一点。
定义题只要求核心机制与概念关系；除非问题明确要求算法名单/版本/更新形式，否则这些细节作为可选补充。
枚举题仍须保留材料明确列出的完整集合，不得漏掉独立指标。不要因为便于评价就缩小问题范围。
"""


def evidence_projection(rows):
    return [dict(eid=f"C{i:04d}", text=r["text"]) for i,r in enumerate(rows,1)]


def source_context(rows):
    return "\n\n".join(f'[来源{i}]（来源: {r["source"]}）\n{r["text"]}' for i,r in enumerate(rows,1))


def prepare(validation, prior, output):
    import evaluate
    output.mkdir(parents=True, exist_ok=True)
    diagnostic = validation / "mineru_retrieval_diagnostic"
    paths = [validation/"private_screen_inputs.json", validation/"protocol_lock.json",
             diagnostic/"local_retrieval_records.json", diagnostic/"manifest.json", prior/"blind_questions.json",
             prior/"reference_basis_lock.json", Path(__file__)]
    paths += [Path(__file__).with_name(n) for n in ("rag_answer_quote_v4.py", "rag_auto_evaluation.py",
              "rag_quote_span_repair.py", "rag_coverage_refinement.py", "rag_semantic_coverage_pilot.py",
              "rag_prospective_coverage_validation.py", "rag_subtle_fault_pilot.py",
              "rag_fresh_change_experiment.py", "rag_refusal.py", "evaluate.py")]
    rows = load(validation/"private_screen_inputs.json")
    records = load(diagnostic/"local_retrieval_records.json")
    old_questions = load(prior/"blind_questions.json")
    questions, reference_inputs, private = {}, {}, {}
    for i,(qid,row) in enumerate(rows.items(),1):
        bid = f"T{i:04d}"
        retrieved = records[qid]["retrieved"]
        if records[qid]["question"] != row["question"] or set(records[qid]["actual_visible_sources"]) != sources(row["context"]):
            raise ValueError("retrieval metadata/query changed")
        questions[bid] = dict(question=row["question"], cached_context=row["context"],
               retrieved_context=source_context(retrieved), evidence=evidence_projection(retrieved),
               actual_sources=sorted(sources(row["context"])))
        reference_inputs[bid] = dict(question=row["question"], retrieved_evidence=[dict(eid=s["eid"],text=s["text"])
                                     for s in old_questions[row["bid"]]["snippets"]])
        private[bid] = dict(qid=qid, original_points=row["points"])
        paths.append(prior/"references"/(row["bid"]+".json"))
    paths.append(validation/"construction/fin_013.json")
    control = load(validation/"construction/fin_013.json")["chosen"]
    if not control or control["target_index"] != 2:
        raise ValueError("fixed omission control changed")
    bid = next(b for b,p in private.items() if p["qid"] == "fin_013")
    questions[bid]["omission_context"] = control["context"]
    hashes = {str(p.resolve()):digest(p) for p in paths}
    if (output/"protocol_lock.json").exists():
        if load(output/"protocol_lock.json")["input_hashes"] != hashes:
            raise ValueError("v4 inputs changed")
        verify(output)
        return
    dump(output/"public_questions.json",questions)
    dump(output/"private_reference_inputs.json",reference_inputs)
    dump(output/"private_question_key.json",private)
    dump(output/"protocol_lock.json",dict(version="mineru_coverage_v4_development",locked_at=utc(), input_hashes=hashes,
          source_hashes=load(diagnostic/"manifest.json")["mineru_hashes"],
          initial_hashes={str((output/n).resolve()):digest(output/n) for n in
                         ("public_questions.json","private_reference_inputs.json","private_question_key.json")},
          config=dict(workers=4, seed=20261002, model="deepseek-chat", generation_temperature=0,
                      generation_max_tokens=512, system_prompt=evaluate.EVAL_PROMPT),
          retrieval=load(diagnostic/"manifest.json")["config"],
          gate=dict(minimum_scope_valid=10, minimum_retrieved_complete_fraction=.8,
                    maximum_proxy_u_fraction=.2, maximum_complete_flags=0, minimum_incomplete_flag_fraction=.8,
                    minimum_incomplete_answers=5, omission_control_flagged=True),
          interpretation="development_on_seen_12_questions_new_responses_not_held_out_test",
          reference_scope_answer_hidden=True, strict_literal_quote_repair_maximum=1,
          original_frozen_results_unchanged=True))
    print("V4 frozen: 12 questions; 24 retrieval-paired answers + 1 fixed omission control",flush=True)


def verify(output):
    m=load(output/"protocol_lock.json")
    for field in ("input_hashes","initial_hashes","source_hashes"):
        verify_hashes(m[field])
    return m,load(output/"public_questions.json")


def build_plan(output, name, question, evidence):
    payload=dict(question=question,retrieved_evidence=evidence)
    raw=auto.request(output,"extract_"+name,EXTRACT_V4,payload)
    enriched,changes=enrich_short_quotes(raw,evidence)
    checks=[auto.request(output,f"minimum_{name}_r{n}",MINIMUM+
                        ("\n从题目明确要求出发检查。" if n==1 else "\n逐项检查必要与可选细节的边界及集合完整性。"),
                        dict(**payload,candidate_points=enriched.get("points",[])),n) for n in (1,2)]
    plans=[minimum_plan(enriched,c,evidence) for c in checks]
    valid=all(p["basis_complete"] for p in plans) and plans[0]["points"]==plans[1]["points"]
    return dict(question=question,evidence=evidence,original_extraction=raw,extraction=enriched,
                quote_expansions=changes,checks=checks,basis_complete=valid,points=plans[0]["points"] if valid else [])


def plans(output):
    m,questions=verify(output)
    if (output/"plans_lock.json").exists():
        verify_hashes(load(output/"plans_lock.json")["hashes"])
        return
    refs=load(output/"private_reference_inputs.json")
    def one(item):
        bid,q=item
        for kind,evidence in (("references",refs[bid]["retrieved_evidence"]),("proxy_plans",q["evidence"])):
            path=output/kind/(bid+".json")
            if not path.exists():
                dump(path,build_plan(output,kind+"_"+bid,q["question"],evidence))
        return {k:load(output/k/(bid+".json"))["basis_complete"] for k in ("references","proxy_plans")}
    batch(list(questions.items()),one,m["config"]["workers"],"V4_PLAN")
    paths=sorted((output/"references").glob("*.json"))+sorted((output/"proxy_plans").glob("*.json"))
    dump(output/"plans_lock.json",dict(locked_at=utc(),hashes={str(p.resolve()):digest(p) for p in paths},
                                      before_new_generation=True, answer_hidden_scope=True))


def contexts(q):
    values=[("cached",q["cached_context"]),("mineru_retrieved",q["retrieved_context"])]
    if "omission_context" in q:
        values.append(("known_omission_control",q["omission_context"]))
    return values


def audit_evidence(output):
    m,questions=verify(output)
    verify_hashes(load(output/"plans_lock.json")["hashes"])
    def one(item):
        bid,q=item
        ref=load(output/"references"/(bid+".json"))
        result={}
        for condition,context in contexts(q):
            name=bid+"_"+condition
            runs=[]
            if ref["basis_complete"]:
                points=[p["point"] for p in ref["points"]]
                payload=dict(question=q["question"],necessary_points=points,visible_context=context)
                for n in (1,2):
                    raw=auto.request(output,f"support_{name}_r{n}",COVERAGE_PROMPT,payload,n)
                    statuses=coverage_result(raw,dict(necessary_points=points,context=context))
                    if literal_failure(raw,statuses):
                        raw=auto.request(output,f"support_{name}_r{n}_copy_retry",COVERAGE_PROMPT+
                                         "\nsupported时直接复制真实原文，不能改标点、拼接或补字；无法找到逐字证据用U。",payload,n)
                        statuses=coverage_result(raw,dict(necessary_points=points,context=context))
                    runs.append(statuses)
            determinate=len(runs)==2 and all(all(s!="U" for s in run) for run in runs)
            full=len(runs)==2 and all(all(s=="supported" for s in run) for run in runs)
            result[condition]=dict(rounds=runs,status="complete" if full else "incomplete" if determinate else "U")
        dump(output/"evidence_audits"/(bid+".json"),result)
        return {k:v["status"] for k,v in result.items()}
    batch(list(questions.items()),one,m["config"]["workers"],"V4_SUPPORT")


def freeze_cases(output):
    m,questions=verify(output)
    verify_hashes(load(output/"plans_lock.json")["hashes"])
    if (output/"cases_lock.json").exists():
        verify_hashes(load(output/"cases_lock.json")["hashes"])
        return
    temporary=[(bid,condition,context,q) for bid,q in questions.items() for condition,context in contexts(q)]
    random.Random(m["config"]["seed"]).shuffle(temporary)
    cases,keys={},{}
    for i,(bid,condition,context,q) in enumerate(temporary,1):
        aid=f"N{i:04d}"
        cases[aid]=dict(question=q["question"],context=context,context_sha256=sha(context))
        keys[aid]=dict(question_key=bid,condition=condition)
    dump(output/"public_cases.json",cases)
    dump(output/"private_case_key.json",keys)
    paths=[output/n for n in ("public_cases.json","private_case_key.json","plans_lock.json")]
    paths+=sorted((output/"evidence_audits").glob("*.json"))
    if len(paths)!=15:
        raise ValueError("missing pre-generation audits")
    dump(output/"cases_lock.json",dict(locked_at=utc(),hashes={str(p.resolve()):digest(p) for p in paths},
                                       cases=len(cases),no_answer_quality_based_selection=True))


def verify_cases(output):
    m,_=verify(output)
    verify_hashes(load(output/"plans_lock.json")["hashes"])
    verify_hashes(load(output/"cases_lock.json")["hashes"])
    return m,load(output/"public_cases.json")


def generate(output):
    import evaluate
    m,cases=verify_cases(output)
    cfg=m["config"]
    def one(item):
        aid,case=item
        payload=dict(model=cfg["model"],temperature=cfg["generation_temperature"],max_tokens=cfg["generation_max_tokens"],
                     messages=[dict(role="system",content=cfg["system_prompt"]),dict(role="user",content=
                     f'问题：{case["question"]}\n\n参考上下文：\n{case["context"]}')])
        hashed=sha(json.dumps(payload,ensure_ascii=False,sort_keys=True))
        path=output/"generations"/(aid+".json")
        if path.exists():
            if load(path)["payload_sha256"]!=hashed:
                raise ValueError("generation payload changed")
            return "cached"
        client=evaluate._get_client().with_options(timeout=90,max_retries=1)
        try:
            started=utc();response=client.chat.completions.create(**payload);choice=response.choices[0]
            answer=(choice.message.content or "").strip()
            if not answer:
                raise RuntimeError("empty generation")
            dump(path,dict(payload=payload,payload_sha256=hashed,answer=answer,started_at=started,finished_at=utc(),
                 response_id=response.id,actual_model=response.model,finish_reason=choice.finish_reason,
                 usage=response.usage.model_dump(),raw_response=response.model_dump(),rule_refusal=int(explicit_refusal(answer))))
            return choice.finish_reason
        finally:
            client.close()
    batch(list(cases.items()),one,cfg["workers"],"V4_GENERATE")


def compare(output,name,question,answer,plan):
    rounds,originals,repairs=[],[],[]
    if plan["basis_complete"]:
        payload=answer_payload(question,answer,plan)
        for n in (1,2):
            raw=auto.request(output,f"compare_{name}_r{n}",COMPARE_V4+
                  ("\n先对照明确作答文字。" if n==1 else "\n先检查实际遗漏，再核对覆盖。"),payload,n)
            original=raw
            invalid=validated_comparison(raw,len(plan["points"]),answer)[2]
            repair=None
            if invalid:
                repair=auto.request(output,f"quote_repair_{name}_r{n}",QUOTE_REPAIR,
                                    dict(**payload,point_results=raw.get("point_results",[]),invalid_indices=invalid),n)
                raw=repair_quotes(raw,repair,invalid)
            originals.append(original);repairs.append(repair);rounds.append(raw)
    return dict(original_rounds=originals,quote_repairs=repairs,rounds=rounds,
                signal=agreement_score(rounds,len(plan["points"]),answer,plan["basis_complete"]))


def point_quality(result, finish_reason="stop"):
    """Derive totals from validated point verdicts; never trust a separate total."""
    if finish_reason!="stop" or len(result["signal"]["rounds"])!=2:
        return "U"
    labels=[]
    for r in result["signal"]["rounds"]:
        statuses=r["point_statuses"]
        if r["status"]=="U":
            labels.append("U")
        elif all(s=="covered" for s in statuses):
            labels.append("2")
        elif "contradicted" in statuses or all(s=="missing" for s in statuses):
            labels.append("0")
        else:
            labels.append("1")
    return labels[0] if len(set(labels))==1 else "U"


def score(output):
    m,cases=verify_cases(output)
    keys=load(output/"private_case_key.json")
    if any(not (output/"generations"/(aid+".json")).exists() for aid in cases):
        raise ValueError("fresh answers incomplete")
    def one(item):
        aid,case=item
        bid=keys[aid]["question_key"];gen=load(output/"generations"/(aid+".json"))
        quality=compare(output,"reference_"+aid,case["question"],gen["answer"],load(output/"references"/(bid+".json")))
        proxy=compare(output,"proxy_"+aid,case["question"],gen["answer"],load(output/"proxy_plans"/(bid+".json")))
        label=point_quality(quality,gen["finish_reason"])
        if gen["finish_reason"]!="stop":
            proxy["signal"].update(status="U",score=None)
        dump(output/"scores"/(aid+".json"),dict(quality=label,reference_comparison=quality,proxy_comparison=proxy))
        return dict(quality=label,proxy=proxy["signal"]["status"])
    batch(list(cases.items()),one,m["config"]["workers"],"V4_SCORE")


def summarize(rows):
    good=[r for r in rows if r["quality"]=="2"]
    bad=[r for r in rows if r["quality"] in {"0","1"}]
    return dict(n=len(rows),complete=len(good),incomplete=len(bad),quality_u=sum(r["quality"]=="U" for r in rows),
                proxy_u=sum(r["proxy_status"]=="U" for r in rows),
                false_flags=sum(r["proxy_status"]=="incomplete" for r in good),
                flagged_incomplete=sum(r["proxy_status"]=="incomplete" for r in bad),
                missed_incomplete=sum(r["proxy_status"]=="complete" for r in bad),
                unresolved_incomplete=sum(r["proxy_status"]=="U" for r in bad),refusals=sum(r["rule_refusal"] for r in rows))


def analyze(output):
    m,cases=verify_cases(output)
    keys=load(output/"private_case_key.json");qkeys=load(output/"private_question_key.json")
    rows,scope_rows=[],[]
    for aid,case in cases.items():
        bid=keys[aid]["question_key"];gen=load(output/"generations"/(aid+".json"));s=load(output/"scores"/(aid+".json"))
        rows.append(dict(aid=aid,qid=qkeys[bid]["qid"],**keys[aid],quality=s["quality"],
                    proxy_status=s["proxy_comparison"]["signal"]["status"],proxy_score=s["proxy_comparison"]["signal"]["score"],
                    rule_refusal=gen["rule_refusal"],finish_reason=gen["finish_reason"]))
    for bid,key in qkeys.items():
        ref=load(output/"references"/(bid+".json"));proxy=load(output/"proxy_plans"/(bid+".json"))
        ev=load(output/"evidence_audits"/(bid+".json"))
        scope_rows.append(dict(qid=key["qid"],reference_valid=int(ref["basis_complete"]),original_points=len(key["original_points"]),
                       revised_points=len(ref["points"]),proxy_valid=int(proxy["basis_complete"]),proxy_points=len(proxy["points"]),
                       cached_evidence=ev["cached"]["status"],retrieved_evidence=ev["mineru_retrieved"]["status"]))
    metrics=[dict(condition=c,**summarize([r for r in rows if c=="all" or r["condition"]==c]))
             for c in ("all","cached","mineru_retrieved","known_omission_control")]
    groups={qid:{r["condition"]:r for r in rows if r["qid"]==qid} for qid in qkeys_to_qids(qkeys)}
    paired=[]
    for qid,g in groups.items():
        a,b=g["cached"],g["mineru_retrieved"]
        paired.append(dict(qid=qid,cached_quality=a["quality"],retrieved_quality=b["quality"],
                           quality_delta=int(b["quality"])-int(a["quality"]) if a["quality"]!="U" and b["quality"]!="U" else None))
    allmetrics=metrics[0];gate=m["gate"]
    valid=[r for r in scope_rows if r["reference_valid"]]
    control=[r for r in rows if r["condition"]=="known_omission_control"][0]
    checks=dict(scope_valid=len(valid)>=gate["minimum_scope_valid"],
       retrieved_semantic_complete=bool(valid) and sum(r["retrieved_evidence"]=="complete" for r in valid)/len(valid)>=gate["minimum_retrieved_complete_fraction"],
       proxy_available=allmetrics["proxy_u"]/len(rows)<=gate["maximum_proxy_u_fraction"],
       no_complete_false_flags=allmetrics["false_flags"]<=gate["maximum_complete_flags"],
       enough_incomplete=allmetrics["incomplete"]>=gate["minimum_incomplete_answers"],
       incomplete_detection=bool(allmetrics["incomplete"]) and allmetrics["flagged_incomplete"]/allmetrics["incomplete"]>=gate["minimum_incomplete_flag_fraction"],
       omission_control=control["quality"] in {"0","1"} and control["proxy_status"]=="incomplete")
    records=[load(p) for p in (output/"api_records").glob("*.json")]
    gens=[load(p) for p in (output/"generations").glob("*.json")]
    value=dict(questions=len(scope_rows),answers=len(rows),metrics=metrics,scope_valid=len(valid),
       proxy_plans_valid=sum(r["proxy_valid"] for r in scope_rows),cached_semantic_complete=sum(r["cached_evidence"]=="complete" for r in valid),
       retrieved_semantic_complete=sum(r["retrieved_evidence"]=="complete" for r in valid),
       quality_improved_pairs=sum(r["quality_delta"] is not None and r["quality_delta"]>0 for r in paired),
       quality_worsened_pairs=sum(r["quality_delta"] is not None and r["quality_delta"]<0 for r in paired),
       pair_quality_u=sum(r["quality_delta"] is None for r in paired),
       gate=dict(status="passed" if all(checks.values()) else "not_passed",checks=checks),
       api_responses=len(records)+len(gens),failed_api_records=sum(not r.get("ok") for r in records),
       total_tokens=sum(r.get("usage",{}).get("total_tokens",0) for r in records+gens),
       models=dict(Counter([r.get("model_returned","error") for r in records]+[g["actual_model"] for g in gens])),
       interpretation=m["interpretation"])
    write_csv(output/"answer_scores.csv",rows);write_csv(output/"scope_evidence_summary.csv",scope_rows)
    write_csv(output/"paired_quality.csv",paired);write_csv(output/"signal_metrics.csv",metrics)
    dump(output/"analysis_summary.json",value)
    print(json.dumps(value,ensure_ascii=False,indent=2),flush=True)


def qkeys_to_qids(qkeys):
    return [key["qid"] for key in qkeys.values()]


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action",choices=["prepare","plans","audit_evidence","freeze","generate","score","analyze"])
    for name in ("validation","prior","output"):
        ap.add_argument("--"+name,type=Path,required=name=="output")
    a=ap.parse_args()
    if a.action=="prepare":
        prepare(a.validation,a.prior,a.output)
    else:
        {"plans":plans,"audit_evidence":audit_evidence,"freeze":freeze_cases,"generate":generate,"score":score,"analyze":analyze}[a.action](a.output)
