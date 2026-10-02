"""Offline audit of provenance, pre-answer locks, payloads and derived scores."""
import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import rag_auto_evaluation as auto
from rag_fresh_change_experiment import load, dump, sha, utc
from rag_prospective_coverage_validation import verify_hashes
from rag_core.mineru_loader import blocks_to_text
from rag_field_retrieval_trial import verify_cases, mapped_proxy
from rag_objective_slots import (EXTRACT_SLOTS, CHECK_SLOTS, COMPARE_SLOTS, controls, exact_source_quote,
       schema_valid, validated_comparison, comparison_payload, agreement_score)
from rag_answer_quote_v4 import QUOTE_REPAIR, repair_quotes
from rag_mineru_coverage_v4 import evidence_projection, source_context, point_quality
from rag_public_field_retrieval import public_query, source_surface_anchor, CONFIG
from rag_field_refusal import field_refusal
from rag_refusal import explicit_refusal


def audit(output):
    m=verify_cases(output);verify_hashes(m["model_hashes"])
    previous=Path(m["previous"]);qs=load(output/"public_questions.json")
    refs=load(output/"private_references.json");retrieved=load(output/"public_retrieval.json")
    old=load(output/"baseline_retrieval.json");cases=load(output/"public_cases.json")
    keys=load(output/"private_case_key.json");known=load(output/"private_control_answers.json")
    if qs!=load(previous/"public_questions.json") or refs!=load(previous/"private_references.json"):
        raise ValueError("frozen bank changed")
    if old!=load(previous/"public_retrieval.json"):raise ValueError("baseline context changed")
    texts={source:blocks_to_text(load(Path(row["mineru_path"])))
           for source,row in load(output/"source_manifest.json").items()}
    api={p.stem:load(p) for p in (output/"api_records").glob("*.json")};expected=set()
    def expect(name,prompt,payload,n=0):
        r=api[name];h=sha(json.dumps([auto.MODEL,prompt,payload,n],ensure_ascii=False,sort_keys=True))
        if not r.get("ok") or r["prompt"]!=prompt or r["payload"]!=payload or r["round"]!=n or r["input_sha256"]!=h or json.loads(r["raw"])!=r["parsed"]:
            raise ValueError("raw request mismatch: "+name)
        expected.add(name);return r["parsed"]
    for qid,q in qs.items():
        base=load(output/"baseline_plans"/(qid+".json"))
        if base!=load(previous/"proxy_plans"/(qid+".json")):raise ValueError("baseline plan changed")
        row=retrieved[qid];trace=public_query(q)
        if any(row["query_trace"][k]!=v for k,v in trace.items()):raise ValueError("private retrieval query")
        for bank in (old,retrieved):
            r=bank[qid]
            if any(r[k]!=q[k] for k in ("question","requested_slots","sources")):raise ValueError("public input changed")
            if len(r["retrieved"])>3 or sum(len(c["text"]) for c in r["retrieved"])>6000:
                raise ValueError("retrieval budget changed")
            for c in r["retrieved"]:
                if c["source"] not in q["sources"] or texts[c["source"]][c["start"]:c["end"]]!=c["text"] or sha(c["text"])!=c["text_sha256"]:
                    raise ValueError("span provenance mismatch")
            if r["evidence"]!=evidence_projection(r["retrieved"]) or r["context"]!=source_context(r["retrieved"]):
                raise ValueError("context reconstruction mismatch")
        payload=dict(question=q["question"],requested_slots=q["requested_slots"],retrieved_evidence=row["evidence"])
        raw=expect("proxy_extract_"+qid,EXTRACT_SLOTS,payload)
        checks=[expect(f"proxy_check_{qid}_r{n}",CHECK_SLOTS+
                ("\n核对字段与值的对应关系。" if n==1 else "\n核对实验口径和别名是否有来源支持。"),
                dict(**payload,temporary_slots=raw.get("slots",[])),n) for n in (1,2)]
        plan=load(output/"proxy_plans"/(qid+".json"))
        wanted=dict(question=q["question"],evidence=row["evidence"],raw=raw,checks=checks,**mapped_proxy(raw,checks,q,row["evidence"]))
        if plan!=wanted:raise ValueError("proxy plan changed")
        for mapping in plan["source_mappings"]:
            slot=next(s for s in raw["slots"] if s["index"]==mapping["index"])
            by={e["eid"]:e["text"] for e in row["evidence"]}
            quote=exact_source_quote(slot["quote"],by[slot["eid"]])["quote"]
            if source_surface_anchor(slot["expected"],quote)!=dict((k,v) for k,v in mapping.items() if k!="index"):
                raise ValueError("source presentation mapping changed")
    def replay(name,question,answer,plan):
        originals=[];schemas=[];quotes=[];rounds=[];count=len(plan["facts"])
        payload=comparison_payload(question,answer,plan["facts"])
        if plan["basis_complete"]:
            for n in (1,2):
                prompt=COMPARE_SLOTS+("\n按所问对象逐字段检查回答。" if n==1 else "\n逐字段检查遗漏、值与对象是否错配。")
                original=expect(f"compare_{name}_r{n}",prompt,payload,n);value=original;schema=None;quote=None
                if not schema_valid(value,count):
                    schema=expect(f"schema_retry_{name}_r{n}",prompt+
                         "\n上一份输出的行数/索引无效。请重新按required_indices各返回一行，不能增删或合并事实。",payload,n);value=schema
                invalid=validated_comparison(value,count,answer)[2]
                if invalid:
                    quote=expect(f"quote_retry_{name}_r{n}",QUOTE_REPAIR,
                         dict(**payload,point_results=value.get("point_results",[]),invalid_indices=invalid),n)
                    value=repair_quotes(value,quote,invalid)
                originals.append(original);schemas.append(schema);quotes.append(quote);rounds.append(value)
        return dict(original_rounds=originals,schema_retries=schemas,quote_retries=quotes,rounds=rounds,
                    signal=agreement_score(rounds,count,answer,plan["basis_complete"]))
    groups={};gens=[]
    for aid,case in cases.items():
        k=keys[aid];qid=k["question_key"];condition=k["condition"];q=qs[qid]
        groups.setdefault(qid,Counter())[condition]+=1
        bank=old if condition=="live_baseline" else retrieved
        if case!=dict(question=q["question"],context=bank[qid]["context"],context_sha256=sha(bank[qid]["context"])):
            raise ValueError("case inputs changed")
        if aid in known:
            control=controls(dict(slots=refs[qid]["slots"]))[condition.removeprefix("control_")]
            if control!=known[aid]:raise ValueError("control truth changed")
            answer=control["answer"];finish="stop";reference=None;quality=control["quality"]
        else:
            gen=load(output/"generations"/(aid+".json"));cfg=m["config"]
            payload=dict(model=cfg["model"],temperature=cfg["generation_temperature"],max_tokens=cfg["generation_max_tokens"],
                    messages=[dict(role="system",content=cfg["system_prompt"]),dict(role="user",content=f'问题：{case["question"]}\n\n参考上下文：\n{case["context"]}')])
            if gen["payload"]!=payload or gen["payload_sha256"]!=sha(json.dumps(payload,ensure_ascii=False,sort_keys=True)):
                raise ValueError("generation payload mismatch")
            raw=gen["raw_response"]
            if gen["answer"]!=raw["choices"][0]["message"]["content"].strip() or gen["actual_model"]!=raw["model"] or gen["response_id"]!=raw["id"] or gen["finish_reason"]!=raw["choices"][0]["finish_reason"]:
                raise ValueError("generation raw mismatch")
            answer=gen["answer"];finish=gen["finish_reason"];gens.append(gen)
            reference=replay("reference_"+aid,case["question"],answer,refs[qid]);quality=point_quality(reference,finish)
        folder="baseline_plans" if condition=="live_baseline" else "proxy_plans"
        proxy=replay("proxy_"+aid,case["question"],answer,load(output/folder/(qid+".json")))
        if finish!="stop":proxy["signal"].update(status="U",score=None)
        wanted=dict(quality=quality,reference_comparison=reference,proxy_comparison=proxy,
                    legacy_refusal=int(explicit_refusal(answer)),
                    field_refusal=field_refusal(answer,[s["label"] for s in q["requested_slots"]]))
        if wanted!=load(output/"scores"/(aid+".json")):raise ValueError("score derivation mismatch")
    wanted=Counter({"live_baseline":1,"live_enhanced":1,"control_complete":1,"control_omitted":1,"control_corrupted":1})
    if len(groups)!=11 or any(group!=wanted for group in groups.values()):raise ValueError("case attrition or condition selection")
    if set(api)!=expected:raise ValueError("unexpected API records")
    locks=[load(output/name)["locked_at"] for name in ("protocol_lock.json","retrieval_lock.json","proxy_lock.json","cases_lock.json")]
    if locks!=sorted(locks) or max(locks)>=min(g["started_at"] for g in gens):raise ValueError("locks after test answers")
    if any(datetime.fromisoformat(api[name]["started_utc"])>=datetime.fromisoformat(locks[2]) for name in api if name.startswith("proxy_")):
        raise ValueError("proxy evidence checks after proxy lock")
    ids=[r["response_id"] for r in api.values()]+[g["response_id"] for g in gens]
    if len(ids)!=len(set(ids)):raise ValueError("response reused")
    result=dict(status="passed",checked_at=utc(),questions=len(qs),source_documents=len(texts),cases=len(cases),
                fresh_generations=len(gens),api_records=len(api),raw_payloads_reconstructed=True,
                raw_spans_source_scope_and_budgets_checked=True,normative_ranking_inputs_excluded=True,
                fixed_control_truth_rederived=True,all_scores_recomputed=True,pre_answer_phase_locks_checked=True,
                frozen_parent_files_unchanged=True,all_unresolved_cases_retained=True)
    dump(output/"integrity_audit.json",result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--output",type=Path,required=True)
    audit(ap.parse_args().output)
