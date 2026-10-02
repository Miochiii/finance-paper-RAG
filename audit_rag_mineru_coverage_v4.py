"""Recompute v4 plans, literal anchors, point totals and public-only payloads."""
import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import rag_auto_evaluation as auto
from rag_fresh_change_experiment import load,dump,sha,utc
from rag_mineru_coverage_v4 import (verify_cases,evidence_projection,source_context,EXTRACT_V4,MINIMUM,
           minimum_plan,enrich_short_quotes,contexts,literal_failure,point_quality,summarize)
from rag_answer_quote_v4 import (COMPARE_V4,QUOTE_REPAIR,validated_comparison,repair_quotes,agreement_score)
from rag_quote_span_repair import answer_payload
from rag_subtle_fault_pilot import COVERAGE_PROMPT,coverage_result,sources
from rag_prospective_coverage_validation import verify_hashes
from rag_core.mineru_loader import blocks_to_text
from rag_refusal import explicit_refusal


def audit(output):
    m,cases=verify_cases(output)
    questions=load(output/"public_questions.json");refs=load(output/"private_reference_inputs.json")
    keys=load(output/"private_case_key.json");private=load(output/"private_question_key.json")
    plans_lock=load(output/"plans_lock.json");case_lock=load(output/"cases_lock.json")
    api={p.stem:load(p) for p in (output/"api_records").glob("*.json")}
    expected={}
    def expect(name,prompt,payload,n=0):
        r=api[name]
        request_hash=sha(json.dumps([auto.MODEL,prompt,payload,n],ensure_ascii=False,sort_keys=True))
        if not r.get("ok") or r["prompt"]!=prompt or r["payload"]!=payload or r["round"]!=n or r["input_sha256"]!=request_hash:
            raise ValueError("request payload/prompt/cache mismatch: "+name)
        if json.loads(r["raw"])!=r["parsed"]:
            raise ValueError("raw response parsed fields changed")
        expected[name]=True
        return r["parsed"]
    validation=Path(next(p for p in m["input_hashes"] if p.endswith("prospective_coverage_validation_20261001\\protocol_lock.json"))).parent
    diagnostic=validation/"mineru_retrieval_diagnostic"
    old_records=load(diagnostic/"local_retrieval_records.json")
    source_manifest_path=next(Path(p) for p in load(diagnostic/"manifest.json")["input_hashes"] if p.endswith("source_manifest.json"))
    fulltexts={name:blocks_to_text(load(Path(r["mineru_path"]))) for name,r in load(source_manifest_path).items()}
    for bid,q in questions.items():
        qid=private[bid]["qid"];record=old_records[qid]
        found=record["retrieved"]
        if q["evidence"]!=evidence_projection(found) or q["retrieved_context"]!=source_context(found):
            raise ValueError("gold leaked through retrieval projection")
        if set(q["actual_sources"])!=sources(q["cached_context"]) or q["question"]!=record["question"]:
            raise ValueError("public retrieval scope/query changed")
        for r in found:
            if r["source"] not in q["actual_sources"] or fulltexts[r["source"]][r["start"]:r["end"]]!=r["text"] or sha(r["text"])!=r["text_sha256"]:
                raise ValueError("retrieval raw span changed")
        for kind,evidence in (("references",refs[bid]["retrieved_evidence"]),("proxy_plans",q["evidence"])):
            plan=load(output/kind/(bid+".json"));name=kind+"_"+bid
            payload=dict(question=q["question"],retrieved_evidence=evidence)
            raw=expect("extract_"+name,EXTRACT_V4,payload)
            enriched,changes=enrich_short_quotes(raw,evidence)
            checks=[expect(f"minimum_{name}_r{n}",MINIMUM+
                         ("\n从题目明确要求出发检查。" if n==1 else "\n逐项检查必要与可选细节的边界及集合完整性。"),
                         dict(**payload,candidate_points=enriched.get("points",[])),n) for n in (1,2)]
            derived=[minimum_plan(enriched,c,evidence) for c in checks]
            valid=all(p["basis_complete"] for p in derived) and derived[0]["points"]==derived[1]["points"]
            wanted=dict(question=q["question"],evidence=evidence,original_extraction=raw,extraction=enriched,
                        quote_expansions=changes,checks=checks,basis_complete=valid,points=derived[0]["points"] if valid else [])
            if plan!=wanted:
                raise ValueError("frozen reference/proxy plan differs from raw calls")
        reference=load(output/"references"/(bid+".json"))
        evidence_audit=load(output/"evidence_audits"/(bid+".json"))
        for condition,context in contexts(q):
            runs=[]
            if reference["basis_complete"]:
                points=[p["point"] for p in reference["points"]]
                payload=dict(question=q["question"],necessary_points=points,visible_context=context)
                for n in (1,2):
                    name=f"support_{bid}_{condition}_r{n}"
                    raw=expect(name,COVERAGE_PROMPT,payload,n)
                    statuses=coverage_result(raw,dict(necessary_points=points,context=context))
                    if literal_failure(raw,statuses):
                        raw=expect(name+"_copy_retry",COVERAGE_PROMPT+
                            "\nsupported时直接复制真实原文，不能改标点、拼接或补字；无法找到逐字证据用U。",payload,n)
                        statuses=coverage_result(raw,dict(necessary_points=points,context=context))
                    runs.append(statuses)
            determinate=len(runs)==2 and all(all(s!="U" for s in run) for run in runs)
            full=len(runs)==2 and all(all(s=="supported" for s in run) for run in runs)
            if evidence_audit[condition]!=dict(rounds=runs,status="complete" if full else "incomplete" if determinate else "U"):
                raise ValueError("semantic evidence audit changed")
    groups={};gens=[];rows=[]
    for aid,case in cases.items():
        key=keys[aid];bid=key["question_key"]
        groups.setdefault(bid,[]).append(key["condition"])
        expected_context=dict(contexts(questions[bid]))[key["condition"]]
        if case!=dict(question=questions[bid]["question"],context=expected_context,context_sha256=sha(expected_context)):
            raise ValueError("case/input condition changed")
        gen=load(output/"generations"/(aid+".json"));score=load(output/"scores"/(aid+".json"));cfg=m["config"]
        payload=dict(model=cfg["model"],temperature=cfg["generation_temperature"],max_tokens=cfg["generation_max_tokens"],messages=[
             dict(role="system",content=cfg["system_prompt"]),dict(role="user",content=f'问题：{case["question"]}\n\n参考上下文：\n{case["context"]}')])
        raw=gen["raw_response"]
        if gen["payload"]!=payload or gen["payload_sha256"]!=sha(json.dumps(payload,ensure_ascii=False,sort_keys=True)):
            raise ValueError("fresh generation input changed")
        if gen["answer"]!=raw["choices"][0]["message"]["content"].strip() or gen["response_id"]!=raw["id"] or gen["actual_model"]!=raw["model"]:
            raise ValueError("fresh response changed")
        if gen["finish_reason"]!=raw["choices"][0]["finish_reason"] or gen["rule_refusal"]!=int(explicit_refusal(gen["answer"])):
            raise ValueError("generation flags changed")
        computed={}
        for kind,short in (("references","reference"),("proxy_plans","proxy")):
            plan=load(output/kind/(bid+".json"));rounds=[];originals=[];repairs=[]
            payload=answer_payload(case["question"],gen["answer"],plan)
            if plan["basis_complete"]:
                for n in (1,2):
                    name=short+"_"+aid
                    original=expect(f"compare_{name}_r{n}",COMPARE_V4+
                            ("\n先对照明确作答文字。" if n==1 else "\n先检查实际遗漏，再核对覆盖。"),payload,n)
                    invalid=validated_comparison(original,len(plan["points"]),gen["answer"])[2]
                    repair=None;value=original
                    if invalid:
                        repair=expect(f"quote_repair_{name}_r{n}",QUOTE_REPAIR,
                            dict(**payload,point_results=original.get("point_results",[]),invalid_indices=invalid),n)
                        value=repair_quotes(original,repair,invalid)
                    rounds.append(value);originals.append(original);repairs.append(repair)
            signal=agreement_score(rounds,len(plan["points"]),gen["answer"],plan["basis_complete"])
            if short=="proxy" and gen["finish_reason"]!="stop":
                signal.update(status="U",score=None)
            computed[short+"_comparison"]=dict(original_rounds=originals,quote_repairs=repairs,rounds=rounds,signal=signal)
            for anchors in signal["anchors"]:
                for anchor in anchors:
                    if anchor["ok"] and gen["answer"][anchor["start"]:anchor["end"]]!=anchor["raw_quote"]:
                        raise ValueError("literal answer anchor changed")
        computed["quality"]=point_quality(computed["reference_comparison"],gen["finish_reason"])
        if computed!=score:
            raise ValueError("quality/proxy score differs from recomputed raw responses")
        rows.append(dict(quality=score["quality"],proxy_status=score["proxy_comparison"]["signal"]["status"],rule_refusal=gen["rule_refusal"]))
        gens.append(gen)
    if set(groups)!=set(questions) or any(sorted(v)!=sorted(k for k,_ in contexts(questions[b])) for b,v in groups.items()):
        raise ValueError("paired case dropped/duplicated")
    if set(expected)!=set(api):
        raise ValueError("unaccounted API calls")
    protocol_date=datetime.fromisoformat(m["locked_at"]);plans_date=datetime.fromisoformat(plans_lock["locked_at"])
    case_date=datetime.fromisoformat(case_lock["locked_at"])
    first_gen=min(datetime.fromisoformat(g["started_at"]) for g in gens)
    last_gen=max(datetime.fromisoformat(g["finished_at"]) for g in gens)
    if not protocol_date<plans_date<case_date<first_gen:
        raise ValueError("phase order invalid")
    for name,r in api.items():
        started=datetime.fromisoformat(r["started_utc"])
        if started<=protocol_date:
            raise ValueError("API predates frozen protocol")
        if name.startswith(("extract_","minimum_")) and started>=plans_date:
            raise ValueError("scope established after freeze")
        if name.startswith("support_") and not plans_date<started<case_date:
            raise ValueError("evidence audit outside pre-generation phase")
        if name.startswith(("compare_","quote_repair_")) and started<=last_gen:
            raise ValueError("scoring before generation finished")
    ids=[r["response_id"] for r in api.values()]+[g["response_id"] for g in gens]
    if len(ids)!=len(set(ids)):
        raise ValueError("response ID reused")
    old_ids={load(p).get("response_id") for d in (validation/"generations",validation/"api_records") for p in d.glob("*.json")}
    if set(ids)&old_ids:
        raise ValueError("old response reused as new")
    summary=load(output/"analysis_summary.json")
    if {k:v for k,v in summary["metrics"][0].items() if k!="condition"}!=summarize(rows):
        raise ValueError("aggregate metric changed")
    value=dict(status="passed",checked_at=utc(),questions=len(questions),fresh_answers=len(cases),
       unique_api_responses=len(ids),source_hashes_checked=len(m["source_hashes"]),
       actual_models=dict(Counter([r["model_returned"] for r in api.values()]+[g["actual_model"] for g in gens])),
       source_spans_and_public_scope_verified=True,reference_scope_locked_before_answers=True,
       proxy_input_whitelists_verified=True,point_quality_and_proxy_scores_recomputed=True,
       quote_repairs_change_only_quoted_text=True,u_cases_retained=True,external_unseen_test=False)
    dump(output/"integrity_audit.json",value)
    print(json.dumps(value,ensure_ascii=False,indent=2))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--output",type=Path,required=True)
    audit(ap.parse_args().output)
