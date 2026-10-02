"""Offline source, qualification, no-feedback, request and score replay audit."""
import argparse
import json
from collections import Counter
from pathlib import Path

import rag_auto_evaluation as auto
from rag_fresh_change_experiment import load,dump,sha,utc
from rag_prospective_coverage_validation import verify_hashes
from rag_exact_source_validation import verify
from rag_exact_source_cards import exact_cards,block_regions,APPROXIMATE
from rag_core.mineru_loader import blocks_to_text
from rag_prospective_objective_validation import question_signature
from rag_objective_slots import (CANDIDATE_PROMPT,CANDIDATE_CHECK,validate_candidate,checks_pass,
       public_slots,plan_facts,controls,EXTRACT_SLOTS,CHECK_SLOTS,COMPARE_SLOTS,schema_valid,
       validated_comparison,comparison_payload,agreement_score)
from rag_objective_scope_revision import SCOPE_PROMPT,scalar_slots,make_question
from rag_field_retrieval_trial import mapped_proxy
from rag_sentence_boundary_repair import repair_selected
from rag_public_field_retrieval import public_query
from rag_mineru_coverage_v4 import evidence_projection,source_context,point_quality
from rag_answer_quote_v4 import QUOTE_REPAIR,repair_quotes
from rag_field_refusal import field_refusal
from rag_refusal import explicit_refusal


def audit(output):
    verify(output);m=load(output/"protocol_lock.json");verify_hashes(m["model_hashes"])
    for name in ("questions_lock.json","retrieval_lock.json","proxy_lock.json","cases_lock.json","validation_tools_lock.json"):
        verify_hashes(load(output/name)["hashes"])
    parent=load(Path(m["previous"])/"protocol_lock.json")
    for key in ("config","retrieval_config","boundary_config","gate"):
        if m[key]!=parent[key]:raise ValueError("frozen V6.1 method changed")
    qs=load(output/"public_questions.json");refs=load(output/"private_references.json");qkeys=load(output/"private_question_key.json")
    inputs=load(output/"candidate_inputs.json");old=load(output/"old_questions.json");keys=load(output/"private_case_key.json")
    known=load(output/"private_control_answers.json");cases=load(output/"public_cases.json");retrieved=load(output/"public_retrieval.json")
    texts={};blocks={}
    for source,row in load(output/"source_manifest.json").items():
        blocks[source]=load(Path(row["mineru_path"]));texts[source]=blocks_to_text(blocks[source])
        if sha(texts[source])!=row["text_sha256"]:raise ValueError("source text changed")
    development=set(load(output/"development_source_registry.json"))
    if any(k["source"] in development for k in qkeys.values()):raise ValueError("source overlap with method development")
    for source,cards in load(output/"candidate_source_cards.json").items():
        if cards!=exact_cards(texts[source],source,blocks[source]):raise ValueError("source-only card selection differs")
        if any(c["mineru_block_type"] not in {"table","text"} or APPROXIMATE.search(c["text"]) or len(c["text"])>900 for c in cards):
            raise ValueError("chart or approximate card admitted")
    api={p.stem:load(p) for p in (output/"api_records").glob("*.json")};expected=set()
    def expect(name,prompt,payload,n=0):
        r=api[name];h=sha(json.dumps([auto.MODEL,prompt,payload,n],ensure_ascii=False,sort_keys=True))
        if not r.get("ok") or r["prompt"]!=prompt or r["payload"]!=payload or r["round"]!=n or r["input_sha256"]!=h or json.loads(r["raw"])!=r["parsed"]:
            raise ValueError("request differs from raw: "+name)
        expected.add(name);return r["parsed"]
    accepted=[];seen={(r["source"],question_signature(q)) for r in inputs.values() for q in r["old_questions"]}
    for bid,row in inputs.items():
        stored=load(output/"candidates"/(bid+".json"));raw=None;locator=None;checks=[]
        validated=dict(valid=False,candidate=None,errors=["no_valid_scalar_candidate"])
        if len(row["evidence"])<2:
            if stored["qualified"]:raise ValueError("too few cards admitted")
            continue
        raw=expect("candidate_"+bid,CANDIDATE_PROMPT,dict(paper_sources=[row["source"]],old_questions=row["old_questions"],mineru_evidence=row["evidence"]))
        proposed=raw.get("candidate")
        if isinstance(proposed,dict) and isinstance(proposed.get("question"),str):
            slots=scalar_slots(proposed)
            if 2<=len(slots)<=3 and all(all(k in s for k in ("index","label","query_quote","type")) for s in slots):
                public=dict(paper_sources=[row["source"]],original_question=proposed["question"],requested_slots=public_slots(dict(slots=slots)))
                locator=expect("scope_"+bid,SCOPE_PROMPT,public)
                if isinstance(locator.get("scope"),str) and 0<len(locator["scope"])<=300:
                    built=make_question([row["source"]],locator["scope"],slots)
                    validated=validate_candidate(dict(candidate=built),row["evidence"],old)
                    if validated["valid"]:
                        payload=dict(question=built["question"],slots=validated["candidate"]["slots"],mineru_evidence=row["evidence"])
                        checks=[expect(f"candidate_check_{bid}_r{n}",CANDIDATE_CHECK+
                             ("\n按程序列出的字段逐项检查对象与口径。" if n==1 else "\n检查字段序号、原表顺序、唯一值、单位和简称对应。"),payload,n) for n in (1,2)]
        qualified=validated["valid"] and len(checks)==2 and all(checks_pass(c,len(validated["candidate"]["slots"]),
                       ("explicit_required","expected_supported","unique","aliases_valid"),("question_clear","scope_complete")) for c in checks)
        if any(stored[k]!=v for k,v in dict(**validated,raw=raw,locator=locator,checks=checks,qualified=qualified).items()):
            raise ValueError("candidate checks or source qualification changed")
        signature=(row["source"],question_signature(validated["candidate"]["question"])) if validated["candidate"] else None
        if qualified and signature not in seen:accepted.append(bid);seen.add(signature)
    if accepted!=[k["input_id"] for k in qkeys.values()]:raise ValueError("response-dependent question selection")
    for qid,q in qs.items():
        bid=qkeys[qid]["input_id"];row=inputs[bid];c=load(output/"candidates"/(bid+".json"))["candidate"]
        if q!=dict(question=c["question"],requested_slots=public_slots(c),sources=[row["source"]]):raise ValueError("public projection contains reference fields")
        if refs[qid]!=dict(basis_complete=True,slots=c["slots"],facts=plan_facts(c["slots"])):raise ValueError("reference changed")
        by={e["eid"]:e for e in row["evidence"]}
        for slot in c["slots"]:
            card=by[slot["eid"]]
            if auto.norm(slot["quote"]) not in auto.norm(card["text"]) or card["mineru_block_type"] not in {"table","text"}:
                raise ValueError("reference quote wrong block provenance")
        r=retrieved[qid];trace=public_query(q)
        if any(r["query_trace"][k]!=v for k,v in trace.items()):raise ValueError("private fields in retrieval query")
        if repair_selected(r["pre_repair_retrieved"],texts)!=r["retrieved"]:raise ValueError("boundary replay mismatch")
        if len(r["retrieved"])>3 or sum(len(e["text"]) for e in r["retrieved"])>6000:raise ValueError("budget changed")
        for span in r["pre_repair_retrieved"]+r["retrieved"]:
            if span["source"] not in q["sources"] or texts[span["source"]][span["start"]:span["end"]]!=span["text"] or sha(span["text"])!=span["text_sha256"]:
                raise ValueError("retrieved raw provenance mismatch")
        if r["evidence"]!=evidence_projection(r["retrieved"]) or r["context"]!=source_context(r["retrieved"]):raise ValueError("context projection changed")
        payload=dict(question=q["question"],requested_slots=q["requested_slots"],retrieved_evidence=r["evidence"])
        raw=expect("proxy_extract_"+qid,EXTRACT_SLOTS,payload)
        checks=[expect(f"proxy_check_{qid}_r{n}",CHECK_SLOTS+
                      ("\n核对字段与值的对应关系。" if n==1 else "\n核对实验口径和别名是否有来源支持。"),
                      dict(**payload,temporary_slots=raw.get("slots",[])),n) for n in (1,2)]
        wanted=dict(question=q["question"],evidence=r["evidence"],raw=raw,checks=checks,**mapped_proxy(raw,checks,q,r["evidence"]))
        if load(output/"proxy_plans"/(qid+".json"))!=wanted:raise ValueError("proxy plan changed")
    def replay(name,question,answer,plan):
        originals=[];schemas=[];quotes=[];rounds=[];count=len(plan["facts"]);payload=comparison_payload(question,answer,plan["facts"])
        if plan["basis_complete"]:
            for n in (1,2):
                prompt=COMPARE_SLOTS+("\n按所问对象逐字段检查回答。" if n==1 else "\n逐字段检查遗漏、值与对象是否错配。")
                original=expect(f"compare_{name}_r{n}",prompt,payload,n);value=original;schema=None;quote=None
                if not schema_valid(value,count):
                    schema=expect(f"schema_retry_{name}_r{n}",prompt+"\n上一份输出的行数/索引无效。请重新按required_indices各返回一行，不能增删或合并事实。",payload,n);value=schema
                invalid=validated_comparison(value,count,answer)[2]
                if invalid:
                    quote=expect(f"quote_retry_{name}_r{n}",QUOTE_REPAIR,dict(**payload,point_results=value.get("point_results",[]),invalid_indices=invalid),n)
                    value=repair_quotes(value,quote,invalid)
                originals.append(original);schemas.append(schema);quotes.append(quote);rounds.append(value)
        return dict(original_rounds=originals,schema_retries=schemas,quote_retries=quotes,rounds=rounds,
                    signal=agreement_score(rounds,count,answer,plan["basis_complete"]))
    groups={};gens=[]
    for aid,case in cases.items():
        k=keys[aid];qid=k["question_key"];q=qs[qid];groups.setdefault(qid,Counter())[k["condition"]]+=1
        if case!=dict(question=q["question"],context=retrieved[qid]["context"],context_sha256=sha(retrieved[qid]["context"])):
            raise ValueError("answer input changed")
        if aid in known:
            control=controls(dict(slots=refs[qid]["slots"]))[k["condition"].removeprefix("control_")]
            if control!=known[aid]:raise ValueError("control single-field edit changed")
            answer=control["answer"];finish="stop";quality=control["quality"];reference=None
        else:
            gen=load(output/"generations"/(aid+".json"));cfg=m["config"]
            payload=dict(model=cfg["model"],temperature=cfg["generation_temperature"],max_tokens=cfg["generation_max_tokens"],messages=[
                 dict(role="system",content=cfg["system_prompt"]),dict(role="user",content=f'问题：{case["question"]}\n\n参考上下文：\n{case["context"]}')])
            if gen["payload"]!=payload or gen["payload_sha256"]!=sha(json.dumps(payload,ensure_ascii=False,sort_keys=True)):
                raise ValueError("generation payload changed")
            raw=gen["raw_response"]
            if gen["answer"]!=raw["choices"][0]["message"]["content"].strip() or gen["actual_model"]!=raw["model"] or gen["response_id"]!=raw["id"] or gen["finish_reason"]!=raw["choices"][0]["finish_reason"]:
                raise ValueError("generation raw response mismatch")
            answer=gen["answer"];finish=gen["finish_reason"];gens.append(gen)
            reference=replay("reference_"+aid,case["question"],answer,refs[qid]);quality=point_quality(reference,finish)
        proxy=replay("proxy_"+aid,case["question"],answer,load(output/"proxy_plans"/(qid+".json")))
        if finish!="stop":proxy["signal"].update(status="U",score=None)
        wanted=dict(quality=quality,reference_comparison=reference,proxy_comparison=proxy,
            legacy_refusal=int(explicit_refusal(answer)),field_refusal=field_refusal(answer,[s["label"] for s in q["requested_slots"]]))
        if load(output/"scores"/(aid+".json"))!=wanted:raise ValueError("score derivation changed")
    wanted=Counter(live_normal=1,control_complete=1,control_omitted=1,control_corrupted=1)
    if any(g!=wanted for g in groups.values()) or set(groups)!=set(qs):raise ValueError("case attrition")
    if set(api)!=expected:raise ValueError("unaccounted API requests")
    phase=[load(output/n)["locked_at"] for n in ("protocol_lock.json","source_integrity_lock.json","questions_lock.json","retrieval_lock.json","proxy_lock.json","cases_lock.json","validation_tools_lock.json")]
    if phase!=sorted(phase) or max(phase)>=min(g["started_at"] for g in gens):raise ValueError("freeze phase after test answers")
    ids=[r["response_id"] for r in api.values()]+[g["response_id"] for g in gens]
    if len(set(ids))!=len(ids):raise ValueError("API response reused")
    result=dict(status="passed",checked_at=utc(),questions=len(qs),sources=len({k["source"] for k in qkeys.values()}),
         development_source_overlap=0,previously_in_corpus_and_old_annotation_banks=True,cases=len(cases),fresh_answers=len(gens),
         source_block_types_and_approximation_filter_verified=True,all_candidate_qualification_replayed=True,
         all_raw_requests_scores_and_controls_replayed=True,raw_retrieval_budget_and_boundary_replay_verified=True,
         frozen_method_unchanged=True,pre_answer_locks_checked=True,unresolved_cases_retained=True,
         api_records=len(api),legacy_preparation_had_no_test_answers=True)
    dump(output/"integrity_audit.json",result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--output",type=Path,required=True)
    audit(ap.parse_args().output)
