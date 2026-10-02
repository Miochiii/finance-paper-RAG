"""Audit frozen-method validation, exact public proxy payloads and fresh responses."""
import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import rag_auto_evaluation as auto
from rag_fresh_change_experiment import load, dump, sha, utc
from rag_prospective_coverage_validation import (SELECTED, verify_cases, verify_hashes, delete_spans,
                                               qualified, EXTRACT, MINIMUM, COMPARE_V3)
from rag_quote_span_repair import answer_payload, agreement_score, enrich_short_quotes
from rag_coverage_refinement import minimum_plan
from rag_refusal import explicit_refusal
from rag_subtle_fault_pilot import sources
from rag_semantic_coverage_pilot import retrieval_candidates


def audit(output):
    m,cases=verify_cases(output)
    verify_hashes(m["model_hashes"])
    plan_lock,case_lock=load(output/"proxy_lock.json"),load(output/"cases_lock.json")
    verify_hashes(plan_lock["hashes"])
    private,truth,keys,selected=[load(output/n) for n in ("private_case_key.json","private_truth.json","public_proxy_plan_keys.json","private_screen_inputs.json")]
    if set(selected)!=set(SELECTED) or not set(cases)==set(private)==set(truth)==set(keys):
        raise ValueError("screening or public case set differs")
    corpus=load(output/"retrieval_corpus.json")
    if len(corpus)!=40 or any(set(c)!={"text","sources","text_sha256"} for c in corpus.values()):
        raise ValueError("retrieval corpus contaminated")
    groups={}
    for aid,key in private.items():groups.setdefault(key["qid"],{})[key["condition"]]=aid
    constructions={q:load(output/"construction"/(q+".json")) for q in selected}
    eligible=[q for q in selected if constructions[q]["qualified"]][:m["config"]["maximum_paired_questions"]]
    if set(eligible)!=set(groups):raise ValueError("post-generation question selection")
    for qid,group in groups.items():
        if set(group)!={"baseline","semantic_partial","retained_evidence_prefix"}:raise ValueError("unbalanced paired conditions")
        row,result=selected[qid],constructions[qid];choice=result["chosen"]
        if not qualified(result["baseline_statuses"]) or not qualified(choice["statuses"],choice["target_index"]):raise ValueError("semantic gate failed")
        context=row["context"]
        for edit in choice["edits"]:
            if edit["input_context"]!=context or sha(context)!=edit["input_sha256"]:raise ValueError("edit history mismatch")
            if edit["removed_texts"]!=[context[a:b] for a,b in edit["spans"]]:raise ValueError("removed raw spans changed")
            context=delete_spans(context,edit["spans"])
            if context!=edit["output_context"]:raise ValueError("deletion altered other content")
        if context!=choice["context"] or context!=cases[group["semantic_partial"]]["context"]:raise ValueError("semantic context changed")
        if cases[group["baseline"]]["context"]!=row["context"]:raise ValueError("baseline changed")
        renumber=lambda text:re.sub(r"\[来源\d+\]","[来源]",text)
        if not renumber(cases[group["retained_evidence_prefix"]]["context"]).endswith(renumber(row["context"])):raise ValueError("retention control lost baseline")
    api={p.stem:load(p) for p in (output/"api_records").glob("*.json")}
    expected={}
    for scope in set(keys.values()):
        plan=load(output/"proxy_plans"/(scope+".json"))
        evidence=[dict(eid=e["eid"],text=e["text"]) for e in plan["evidence"]]
        for e in evidence:
            if e["eid"] not in corpus or e["text"]!=corpus[e["eid"]]["text"]:raise ValueError("retrieved evidence mismatch")
        payload=dict(question=plan["question"],retrieved_evidence=evidence)
        expected["extract_"+scope]=(payload,EXTRACT)
        expected["minimum_"+scope]=(dict(**payload,candidate_points=plan["original_extraction"].get("points",[])),MINIMUM)
        if api["extract_"+scope]["parsed"]!=plan["original_extraction"] or api["minimum_"+scope]["parsed"]!=plan["scope_check"]:raise ValueError("proxy plans differ from raw extraction")
        enriched,changes=enrich_short_quotes(plan["original_extraction"],evidence)
        derived=minimum_plan(enriched,plan["scope_check"],evidence)
        if changes!=plan["quote_changes"] or any(plan[k]!=v for k,v in derived.items()):raise ValueError("frozen v3 plan processing changed")
    gens=[]
    for aid,case in cases.items():
        if set(case)!={"question","context","context_sha256"} or sha(case["context"])!=case["context_sha256"]:raise ValueError("public case contaminated")
        plan=load(output/"proxy_plans"/(keys[aid]+".json"))
        if plan["actual_sources"]!=sorted(sources(case["context"])) or plan["question"]!=case["question"]:raise ValueError("public retrieval scope mismatch")
        candidates={eid for eid,_ in retrieval_candidates(corpus,case["context"])}
        if not {e["eid"] for e in plan["evidence"]}<=candidates:raise ValueError("hidden source filter used")
        gen,score=load(output/"generations"/(aid+".json")),load(output/"scores"/(aid+".json"));cfg=m["config"]
        payload=dict(model=cfg["model"],temperature=cfg["generation_temperature"],max_tokens=cfg["generation_max_tokens"],messages=[
            dict(role="system",content=cfg["system_prompt"]),dict(role="user",content=f'问题：{case["question"]}\n\n参考上下文：\n{case["context"]}')])
        if gen["payload"]!=payload or gen["payload_sha256"]!=sha(json.dumps(payload,ensure_ascii=False,sort_keys=True)):raise ValueError("generation payload mismatch")
        raw=gen["raw_response"]
        if gen["answer"]!=raw["choices"][0]["message"]["content"].strip() or gen["response_id"]!=raw["id"] or gen["finish_reason"]!="stop":raise ValueError("fresh generation missing or truncated")
        if gen["rule_refusal"]!=int(explicit_refusal(gen["answer"])):raise ValueError("refusal rule edited")
        gens.append(gen)
        quality_payload=dict(question=case["question"],model_answer=gen["answer"],necessary_points=truth[aid]["necessary_points"],basis_complete=True)
        for n in (1,2):
            expected[f"quality_{aid}_r{n}"]=(quality_payload,auto.QUALITY_PROMPT+("\n先检查遗漏和实质错误。" if n==1 else "\n先逐项对照必要要点。"))
        rounds=[api[f"quality_{aid}_r{n}"]["parsed"] for n in (1,2)]
        consensus=auto.quality_consensus(rounds,gen["rule_refusal"],True,len(truth[aid]["necessary_points"]))
        if rounds!=score["rounds"] or any(score[k]!=v for k,v in consensus.items()):raise ValueError("offline quality fields edited")
        compare_payload=answer_payload(case["question"],gen["answer"],plan)
        if plan["basis_complete"]:
            for n in (1,2):expected[f"compare_{aid}_r{n}"]=(compare_payload,COMPARE_V3+("\n先对照回答里的明确文字。" if n==1 else "\n先检查遗漏了哪些具体对象，再判覆盖。"))
            comparisons=[api[f"compare_{aid}_r{n}"]["parsed"] for n in (1,2)]
        else:comparisons=[]
        if comparisons!=score["comparisons"] or score["signal"]!=agreement_score(comparisons,len(plan["points"]),gen["answer"],plan["basis_complete"]):raise ValueError("frozen proxy verdict edited")
        expected["faith_"+aid]=(dict(question=case["question"],model_answer=gen["answer"],visible_context=case["context"]),auto.FAITH_PROMPT)
        if api["faith_"+aid]["parsed"]!=score["faith"] or score["support_status"]!=auto.validate_faith(score["faith"],case["context"]):raise ValueError("faithfulness edited")
    for name,(payload,prompt) in expected.items():
        r=api[name]
        if not r.get("ok") or r["payload"]!=payload or r["prompt"]!=prompt:raise ValueError("proxy/evaluation payload or frozen prompt changed")
    if any(not r.get("ok") for r in api.values()):raise ValueError("incomplete API record")
    dates=[datetime.fromisoformat(x) for x in (m["locked_at"],case_lock["locked_at"],plan_lock["locked_at"])]
    first_gen=min(datetime.fromisoformat(g["started_at"]) for g in gens);last_gen=max(datetime.fromisoformat(g["finished_at"]) for g in gens)
    if not dates[0]<dates[1]<dates[2]<first_gen:raise ValueError("phase locks invalid")
    if any(datetime.fromisoformat(r["completed_at"])>=dates[1] for r in constructions.values()):raise ValueError("construction after freeze")
    for name,r in api.items():
        start=datetime.fromisoformat(r["started_utc"])
        if start<=dates[0]:raise ValueError("API before protocol lock")
        if name.startswith(("map_","audit_")) and start>=dates[1]:raise ValueError("intervention after case freeze")
        if name.startswith(("extract_","minimum_")) and not dates[1]<start<dates[2]:raise ValueError("proxy plan after generation")
        if name.startswith(("quality_","compare_","faith_")) and start<=last_gen:raise ValueError("evaluation before all generation")
    ids=[r["response_id"] for r in api.values()]+[g["response_id"] for g in gens]
    models=Counter([r["model_returned"] for r in api.values()]+[g["actual_model"] for g in gens])
    if len(ids)!=len(set(ids)) or len(models)!=1:raise ValueError("duplicate IDs or changed model")
    development=Path(next(p for p in m["input_hashes"] if p.endswith("coverage_refinement_v3\\manifest.json"))).parent.parent
    old_ids={load(p)["response_id"] for directory in (development/"generations",development/"api_records",development/"coverage_refinement_v2/api_records",development/"coverage_refinement_v3/api_records") for p in directory.glob("*.json")}
    if set(ids)&old_ids:raise ValueError("old development response reused")
    value=dict(status="passed",checked_at=utc(),screened_questions=len(selected),paired_questions=len(groups),generated_answers=len(cases),
         unique_new_api_response_ids=len(ids),actual_models=dict(models),code_input_model_hashes_checked=len(m["input_hashes"])+len(m["model_hashes"]),
         source_selection_before_generation=True,all_screening_failures_retained=True,exact_local_deletion_spans_verified=True,
         normative_facts_not_sent_to_proxies=True,proxy_prompts_and_judgement_functions_unchanged=True,
         proxy_plans_completed_before_generation=True,all_new_responses=True,all_derived_quality_and_signal_scores_recomputed=True,
         external_unseen_test=False)
    dump(output/"integrity_audit.json",value);print(value)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--output",type=Path,required=True)
    audit(ap.parse_args().output)
