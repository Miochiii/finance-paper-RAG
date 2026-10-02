"""Audit objective candidates, public scope revision and known control truth."""
import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import rag_auto_evaluation as auto
from rag_fresh_change_experiment import load,dump,sha,utc
from rag_core.mineru_loader import blocks_to_text
from rag_prospective_coverage_validation import verify_hashes
from rag_objective_coverage_pilot import verify_cases,proxy_plan,point_quality,evidence_projection,source_context,metrics
from rag_objective_slots import (CANDIDATE_PROMPT,CANDIDATE_CHECK,EXTRACT_SLOTS,CHECK_SLOTS,COMPARE_SLOTS,
       validate_candidate,checks_pass,public_slots,plan_facts,controls,schema_valid,validated_comparison,comparison_payload,agreement_score)
from rag_objective_scope_revision import SCOPE_PROMPT,scalar_slots,make_question
from rag_answer_quote_v4 import QUOTE_REPAIR,repair_quotes
from rag_refusal import explicit_refusal


def audit(output):
    m,cases=verify_cases(output);verify_hashes(m["model_hashes"])
    revlock=load(output/"scope_revision_lock.json");verify_hashes(revlock["hashes"])
    inputs=load(output/"candidate_inputs.json");old=load(output/"old_questions.json")
    questions=load(output/"public_questions.json");qkeys=load(output/"private_question_key.json")
    refs=load(output/"private_references.json");keys=load(output/"private_case_key.json");known=load(output/"private_control_answers.json")
    diagnostic_lock=load(output/"reference_control_lock.json");verify_hashes(diagnostic_lock["hashes"])
    retrieved=load(output/"public_retrieval.json")
    api={p.stem:load(p) for p in (output/"api_records").glob("*.json")};expected=set()
    def expect(name,prompt,payload,n=0):
        r=api[name];h=sha(json.dumps([auto.MODEL,prompt,payload,n],ensure_ascii=False,sort_keys=True))
        if not r.get("ok") or r["prompt"]!=prompt or r["payload"]!=payload or r["round"]!=n or r["input_sha256"]!=h or json.loads(r["raw"])!=r["parsed"]:
            raise ValueError("raw request mismatch: "+name)
        expected.add(name);return r["parsed"]
    accepted=[]
    for bid,row in inputs.items():
        stored=load(output/"candidates"/(bid+".json"))
        raw=expect("candidate_"+bid,CANDIDATE_PROMPT,dict(paper_sources=row["sources"],old_question=row["question"],mineru_evidence=row["evidence"]))
        v=validate_candidate(raw,row["evidence"],old)
        if any(stored[k]!=v[k] for k in v):raise ValueError("initial validation changed")
        checks=[]
        if v["valid"]:
            c=v["candidate"];payload=dict(question=c["question"],slots=c["slots"],mineru_evidence=row["evidence"])
            checks=[expect(f"candidate_check_{bid}_r{n}",CANDIDATE_CHECK+
                     ("\n优先检查题干范围和口径。" if n==1 else "\n优先检查字段与原文值的唯一对应及简称等价。"),payload,n) for n in (1,2)]
        if checks!=stored["checks"]:raise ValueError("initial checks changed")
        revised=load(output/"revised_candidates"/(bid+".json"));slots=scalar_slots(stored["candidate"]) if stored["candidate"] else []
        if not 2<=len(slots)<=3:
            if revised["qualified"]:raise ValueError("non scalar candidate admitted")
            continue
        payload=dict(paper_sources=row["sources"],original_question=stored["candidate"]["question"],requested_slots=public_slots(dict(slots=slots)))
        locator=expect("scope_"+bid,SCOPE_PROMPT,payload)
        c=make_question(row["sources"],locator["scope"],slots);valid=validate_candidate(dict(candidate=c),row["evidence"],old)
        if any(revised[k]!=valid[k] for k in valid):raise ValueError("constructed scope differs from public fields")
        newchecks=[]
        if valid["valid"]:
            payload=dict(question=valid["candidate"]["question"],slots=valid["candidate"]["slots"],mineru_evidence=row["evidence"])
            newchecks=[expect(f"revised_check_{bid}_r{n}",CANDIDATE_CHECK+
                      ("\n按程序列出的字段逐项检查对象与口径。" if n==1 else "\n检查字段序号、原表顺序、唯一值、单位和简称对应。"),payload,n) for n in (1,2)]
        qualified=valid["valid"] and len(newchecks)==2 and all(checks_pass(c,len(valid["candidate"]["slots"]),
                           ("explicit_required","expected_supported","unique","aliases_valid"),("question_clear","scope_complete")) for c in newchecks)
        if revised["checks"]!=newchecks or revised["qualified"]!=qualified:raise ValueError("revised source checks changed")
        if qualified:accepted.append(bid)
    if accepted!=[k["input_id"] for k in qkeys.values()]:raise ValueError("post answer candidate selection")
    texts={name:blocks_to_text(load(Path(r["mineru_path"]))) for name,r in load(output/"source_manifest.json").items()}
    for qid,q in questions.items():
        bid=qkeys[qid]["input_id"];c=load(output/"revised_candidates"/(bid+".json"))["candidate"]
        if q!=dict(question=c["question"],requested_slots=public_slots(c),sources=inputs[bid]["sources"]):raise ValueError("public question/reference leak")
        if refs[qid]!=dict(basis_complete=True,slots=c["slots"],facts=plan_facts(c["slots"])):raise ValueError("reference facts changed")
        for s in c["slots"]:
            if not any(auto.norm(s["quote"]) in auto.norm(texts[p]) for p in q["sources"]):
                raise ValueError("reference quote absent from original MinerU sources")
        r=retrieved[qid]
        if r["question"]!=q["question"] or r["requested_slots"]!=q["requested_slots"] or r["sources"]!=q["sources"]:
            raise ValueError("retrieval query/scope changed")
        for span in r["retrieved"]:
            if span["source"] not in q["sources"] or texts[span["source"]][span["start"]:span["end"]]!=span["text"] or sha(span["text"])!=span["text_sha256"]:
                raise ValueError("retrieval span not original")
        if r["evidence"]!=evidence_projection(r["retrieved"]) or r["context"]!=source_context(r["retrieved"]):
            raise ValueError("retrieval evidence projection contaminated")
        payload=dict(question=q["question"],requested_slots=q["requested_slots"],retrieved_evidence=r["evidence"])
        raw=expect("proxy_extract_"+qid,EXTRACT_SLOTS,payload)
        checks=[expect(f"proxy_check_{qid}_r{n}",CHECK_SLOTS+
                 ("\n核对字段与值的对应关系。" if n==1 else "\n核对实验口径和别名是否有来源支持。"),
                 dict(**payload,temporary_slots=raw.get("slots",[])),n) for n in (1,2)]
        plan=load(output/"proxy_plans"/(qid+".json"))
        wanted=dict(question=q["question"],evidence=r["evidence"],raw=raw,checks=checks,**proxy_plan(raw,checks,q,r["evidence"]))
        if plan!=wanted:raise ValueError("proxy plan changed")
    groups={};gens=[];metrics_rows=[]
    for aid,case in cases.items():
        qid=keys[aid]["question_key"];condition=keys[aid]["condition"];q=questions[qid]
        groups.setdefault(qid,set()).add(condition)
        if case!=dict(question=q["question"],context=retrieved[qid]["context"],context_sha256=sha(retrieved[qid]["context"])):
            raise ValueError("case input changed")
        if condition.startswith("control_"):
            expected_control=controls(dict(slots=refs[qid]["slots"]))[condition.removeprefix("control_")]
            if known[aid]!=expected_control:raise ValueError("control truth or single slot edit changed")
            answer=expected_control["answer"];finish="stop"
        else:
            gen=load(output/"generations"/(aid+".json"));cfg=m["config"]
            payload=dict(model=cfg["model"],temperature=cfg["generation_temperature"],max_tokens=cfg["generation_max_tokens"],messages=[
                dict(role="system",content=cfg["system_prompt"]),dict(role="user",content=f'问题：{case["question"]}\n\n参考上下文：\n{case["context"]}')])
            if gen["payload"]!=payload or gen["payload_sha256"]!=sha(json.dumps(payload,ensure_ascii=False,sort_keys=True)):
                raise ValueError("live generation payload changed")
            raw=gen["raw_response"]
            if gen["answer"]!=raw["choices"][0]["message"]["content"].strip() or gen["response_id"]!=raw["id"] or gen["finish_reason"]!=raw["choices"][0]["finish_reason"]:
                raise ValueError("live answer changed")
            answer=gen["answer"];finish=gen["finish_reason"];gens.append(gen)
        score=load(output/"scores"/(aid+".json"));derived={}
        for kind,plan in (("reference",refs[qid]),("proxy",load(output/"proxy_plans"/(qid+".json")))):
            reference_control=kind=="reference" and condition.startswith("control_")
            facts=plan["facts"];count=len(facts);payload=comparison_payload(case["question"],answer,facts)
            originals=[];schemas=[];quotes=[];rounds=[]
            if plan["basis_complete"]:
                for n in (1,2):
                    prompt=COMPARE_SLOTS+("\n按所问对象逐字段检查回答。" if n==1 else "\n逐字段检查遗漏、值与对象是否错配。")
                    name=("reference_control" if reference_control else kind)+"_"+aid
                    original=expect(f"compare_{name}_r{n}",prompt,payload,n);value=original;schema=None;quote=None
                    if not schema_valid(value,count):
                        schema=expect(f"schema_retry_{name}_r{n}",prompt+
                            "\n上一份输出的行数/索引无效。请重新按required_indices各返回一行，不能增删或合并事实。",payload,n);value=schema
                    invalid=validated_comparison(value,count,answer)[2]
                    if invalid:
                        quote=expect(f"quote_retry_{name}_r{n}",QUOTE_REPAIR,dict(**payload,point_results=value.get("point_results",[]),invalid_indices=invalid),n)
                        value=repair_quotes(value,quote,invalid)
                    originals.append(original);schemas.append(schema);quotes.append(quote);rounds.append(value)
            signal=agreement_score(rounds,count,answer,plan["basis_complete"])
            if kind=="proxy" and finish!="stop":signal.update(status="U",score=None)
            result=dict(original_rounds=originals,schema_retries=schemas,quote_retries=quotes,rounds=rounds,signal=signal)
            if reference_control:
                stored_check=load(output/"reference_control_scores"/(aid+".json"))
                if stored_check!=dict(result=result,predicted_quality=point_quality(result),known_quality=known[aid]["quality"]):
                    raise ValueError("reference scorer diagnostic edited")
            else:
                derived[kind+"_comparison"]=result
            for anchors in signal["anchors"]:
                for a in anchors:
                    if a["ok"] and answer[a["start"]:a["end"]]!=a["raw_quote"]:raise ValueError("answer anchor changed")
        if condition.startswith("control_"):
            derived.update(quality=known[aid]["quality"],quality_provenance="source_supported_control_by_construction",reference_comparison=None)
        else:
            derived.update(quality=point_quality(derived["reference_comparison"],finish),quality_provenance="two_round_automatic_reference")
        derived["rule_refusal"]=int(explicit_refusal(answer))
        if derived!=score:raise ValueError("recomputed scoring differs")
        metrics_rows.append(dict(condition=condition,quality=score["quality"],proxy_status=score["proxy_comparison"]["signal"]["status"],rule_refusal=score["rule_refusal"]))
    expected_conditions={"control_complete","control_omitted","control_corrupted","live_normal"}
    if set(groups)!=set(questions) or any(v!=expected_conditions for v in groups.values()):raise ValueError("case missing")
    if expected!=set(api):raise ValueError("unaccounted API records")
    dates=[datetime.fromisoformat(load(output/n)["locked_at"]) for n in
           ("protocol_lock.json","scope_revision_lock.json","questions_lock.json","retrieval_lock.json","proxy_lock.json","answer_cases_lock.json")]
    first=min(datetime.fromisoformat(g["started_at"]) for g in gens);last=max(datetime.fromisoformat(g["finished_at"]) for g in gens)
    if not all(a<b for a,b in zip(dates,dates[1:])) or dates[-1]>=first:raise ValueError("phase locks invalid")
    if not dates[-1]<datetime.fromisoformat(diagnostic_lock["locked_at"])<first:
        raise ValueError("scorer diagnostic added after seeing test answers")
    for name,r in api.items():
        t=datetime.fromisoformat(r["started_utc"])
        if name.startswith("candidate_") and not dates[0]<t<dates[1]:raise ValueError("initial draft outside phase")
        if name.startswith(("scope_","revised_check_")) and not dates[1]<t<dates[2]:raise ValueError("scope revised after questions frozen")
        if name.startswith(("proxy_extract_","proxy_check_")) and not dates[3]<t<dates[4]:raise ValueError("proxy plans outside phase")
        if name.startswith(("compare_","schema_retry_","quote_retry_")) and t<=last:raise ValueError("scored before live generation finished")
    ids=[r["response_id"] for r in api.values()]+[g["response_id"] for g in gens]
    if len(ids)!=len(set(ids)):raise ValueError("response ID reused")
    summary=load(output/"analysis_summary.json")
    for row in summary["metrics"]:
        selected=[r for r in metrics_rows if (row["condition"]=="controls" and r["condition"].startswith("control_")) or r["condition"]==row["condition"]]
        if {k:v for k,v in row.items() if k!="condition"}!=metrics(selected):raise ValueError("summary edited")
    value=dict(status="passed",checked_at=utc(),initial_candidates=len(inputs),admitted_questions=len(questions),
       known_control_answers=len(known),live_answers=len(gens),unique_new_api_responses=len(ids),source_hashes_checked=len(m["source_hashes"]),
       scope_constructed_before_any_test_answer=True,initial_drafts_and_failed_checks_retained=True,
       control_labels_recomputed_from_single_slot_changes=True,public_only_proxy_payloads_verified=True,
       reference_scorer_diagnostic_prespecified_and_recomputed=True,
       retrieved_spans_and_answer_anchors_verified=True,all_scores_recomputed=True,
       new_questions_on_seen_sources=True,live_fault_detection_test=False,
       actual_models=dict(Counter([r["model_returned"] for r in api.values()]+[g["actual_model"] for g in gens])))
    dump(output/"integrity_audit.json",value);print(json.dumps(value,ensure_ascii=False,indent=2))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--output",type=Path,required=True)
    audit(ap.parse_args().output)
