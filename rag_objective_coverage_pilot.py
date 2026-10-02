"""New finite-scope questions and evaluator controls, all frozen before answers.

New questions use previously seen source papers. Synthetic answer changes are
known by construction and test scoring only, not live evidence-fault detection.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import rag_auto_evaluation as auto
from rag_fresh_change_experiment import digest,dump,load,sha,utc,write_csv
from rag_subtle_fault_pilot import batch,LocalReranker
from rag_core.mineru_loader import blocks_to_text
from rag_prospective_coverage_validation import verify_hashes
from rag_mineru_coverage_v4 import point_quality,source_context,evidence_projection
from rag_answer_quote_v4 import QUOTE_REPAIR,repair_quotes
from diagnose_mineru_evidence_coverage import make_chunks,retrieve
from rag_refusal import explicit_refusal
from rag_objective_slots import (CANDIDATE_PROMPT,CANDIDATE_CHECK,EXTRACT_SLOTS,CHECK_SLOTS,COMPARE_SLOTS,
        validate_candidate,checks_pass,public_slots,plan_facts,controls,exact_source_quote,numeric_value,
        schema_valid,validated_comparison,comparison_payload,agreement_score)


def prepare(previous,validation,prior,output):
    import csv
    import evaluate
    output.mkdir(parents=True,exist_ok=True)
    paths=[previous/n for n in ("private_reference_inputs.json","private_question_key.json","public_questions.json")]
    paths += [validation/"protocol_lock.json",prior/"source_manifest.json",Path(__file__)]
    paths += [Path(__file__).parent/n for n in ("rag_objective_slots.py","rag_auto_evaluation.py","rag_answer_quote_v4.py",
               "rag_mineru_coverage_v4.py","rag_prospective_coverage_validation.py","rag_subtle_fault_pilot.py",
               "rag_semantic_coverage_pilot.py","rag_fresh_change_experiment.py","diagnose_mineru_evidence_coverage.py",
               "rag_core/mineru_loader.py","rag_refusal.py","evaluate.py")]
    annotations=sorted(Path("data/annotations").glob("*.csv"))
    old_questions=[]
    for path in annotations:
        with path.open(encoding="utf-8-sig",newline="") as f:
            old_questions.extend(r["question"] for r in csv.DictReader(f) if r.get("question"))
    paths+=annotations
    refs=load(previous/"private_reference_inputs.json");pub=load(previous/"public_questions.json")
    inputs={bid:dict(question=r["question"],evidence=r["retrieved_evidence"],sources=pub[bid]["actual_sources"])
            for bid,r in refs.items()}
    source_manifest=load(prior/"source_manifest.json")
    source_hashes={r["mineru_path"]:r["mineru_sha256"] for r in source_manifest.values()}
    hashes={str(p.resolve()):digest(p) for p in paths}
    if (output/"protocol_lock.json").exists():
        if load(output/"protocol_lock.json")["input_hashes"]!=hashes:raise ValueError("objective protocol input changed")
        verify(output);return
    dump(output/"candidate_inputs.json",inputs);dump(output/"old_questions.json",sorted(set(old_questions)))
    dump(output/"source_manifest.json",source_manifest)
    dump(output/"protocol_lock.json",dict(version="objective_slots_v5_pilot",locked_at=utc(),input_hashes=hashes,
          source_hashes=source_hashes,model_hashes=load(validation/"protocol_lock.json")["model_hashes"],
          initial_hashes={str((output/n).resolve()):digest(output/n) for n in ("candidate_inputs.json","old_questions.json","source_manifest.json")},
          config=dict(workers=4,seed=20261002,model="deepseek-chat",generation_temperature=0,generation_max_tokens=512,
                      system_prompt=evaluate.EVAL_PROMPT,reranker_path=load(validation/"protocol_lock.json")["config"]["reranker_path"]),
          retrieval=dict(window_chars=2000,stride_chars=1500,lexical_candidates=20,max_tokens=4096,top_k=3,
                         scope="public_named_paper_sources_only"),
          gate=dict(minimum_qualified_questions=10,maximum_control_proxy_u_fraction=.2,maximum_complete_false_flags=0,
                    minimum_omitted_flag_fraction=.8,minimum_corrupted_flag_fraction=.8,minimum_reference_check_pass_fraction=.8),
          maximum_candidates=12,maximum_slots=3,schema_retry_limit=1,quote_retry_limit=1,
          interpretation="new_questions_on_seen_papers_scoring_controls_and_live_normal_answers_not_live_fault_benchmark",
          candidate_status="pending_experimental_admission_only"))
    print("Frozen: 12 new candidate questions; 2-3 explicit slots; at most 36 controls +12 live answers",flush=True)


def verify(output):
    m=load(output/"protocol_lock.json")
    for key in ("input_hashes","source_hashes","initial_hashes"):verify_hashes(m[key])
    return m


def candidates(output):
    m=verify(output);inputs=load(output/"candidate_inputs.json");old=load(output/"old_questions.json")
    def one(item):
        bid,row=item;path=output/"candidates"/(bid+".json")
        if path.exists():return load(path)["qualified"]
        payload=dict(paper_sources=row["sources"],old_question=row["question"],mineru_evidence=row["evidence"])
        raw=auto.request(output,"candidate_"+bid,CANDIDATE_PROMPT,payload)
        validated=validate_candidate(raw,row["evidence"],old)
        checks=[]
        if validated["valid"]:
            candidate=validated["candidate"]
            payload=dict(question=candidate["question"],slots=candidate["slots"],mineru_evidence=row["evidence"])
            checks=[auto.request(output,f"candidate_check_{bid}_r{n}",CANDIDATE_CHECK+
                        ("\n优先检查题干范围和口径。" if n==1 else "\n优先检查字段与原文值的唯一对应及简称等价。"),payload,n) for n in (1,2)]
        qualified=validated["valid"] and len(checks)==2 and all(checks_pass(c,len(validated["candidate"]["slots"]),
                         ("explicit_required","expected_supported","unique","aliases_valid"),("question_clear","scope_complete")) for c in checks)
        dump(path,dict(raw=raw,**validated,checks=checks,qualified=qualified,finished_at=utc()))
        return qualified
    batch(list(inputs.items()),one,m["config"]["workers"],"OBJECTIVE_CANDIDATES")


def freeze_questions(output):
    verify(output)
    if (output/"questions_lock.json").exists():verify_hashes(load(output/"questions_lock.json")["hashes"]);return
    inputs=load(output/"candidate_inputs.json");questions={};refs={};keys={};seen=set();screen=[]
    for bid,row in inputs.items():
        r=load(output/"candidates"/(bid+".json"));c=r["candidate"]
        duplicate=bool(c and auto.norm(c["question"]) in seen)
        accepted=r["qualified"] and not duplicate
        screen.append(dict(input_id=bid,status="pending",program_valid=int(r["valid"]),two_checks_pass=int(r["qualified"]),
                           exact_duplicate=int(duplicate),experimental_admission=int(accepted),errors="|".join(r["errors"])))
        if accepted:
            qid=f"O{len(questions)+1:04d}";seen.add(auto.norm(c["question"]))
            questions[qid]=dict(question=c["question"],requested_slots=public_slots(c),sources=row["sources"])
            refs[qid]=dict(basis_complete=True,slots=c["slots"],facts=plan_facts(c["slots"]))
            keys[qid]=dict(input_id=bid,candidate_status="pending")
    for name,value in (("public_questions.json",questions),("private_references.json",refs),("private_question_key.json",keys)):
        dump(output/name,value)
    write_csv(output/"candidate_screening.csv",screen)
    paths=[output/n for n in ("public_questions.json","private_references.json","private_question_key.json")]+sorted((output/"candidates").glob("*.json"))
    dump(output/"questions_lock.json",dict(locked_at=utc(),hashes={str(p.resolve()):digest(p) for p in paths},
                                         qualified=len(questions),no_answer_based_selection=True))
    print("Admitted new objective questions:",len(questions),"/12",flush=True)


def verify_questions(output):
    m=verify(output);verify_hashes(load(output/"questions_lock.json")["hashes"])
    return m,load(output/"public_questions.json")


def retrieval(output):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    m,questions=verify_questions(output)
    if (output/"retrieval_lock.json").exists():verify_hashes(load(output/"retrieval_lock.json")["hashes"]);return
    verify_hashes(m["model_hashes"])
    chunks=[]
    for source,row in sorted(load(output/"source_manifest.json").items()):
        chunks.extend(make_chunks(blocks_to_text(load(Path(row["mineru_path"]))),source))
    vectorizer=TfidfVectorizer(analyzer="char",ngram_range=(2,3),max_features=60000,lowercase=True,dtype=np.float32)
    matrix=vectorizer.fit_transform([c["text"] for c in chunks]);model=LocalReranker(m["config"]["reranker_path"])
    rows={}
    for qid,q in questions.items():
        found=retrieve(q["question"],q["sources"],chunks,vectorizer,matrix,model)
        rows[qid]=dict(question=q["question"],requested_slots=q["requested_slots"],sources=q["sources"],
                       retrieved=found,evidence=evidence_projection(found),context=source_context(found))
        print("OBJECTIVE_RETRIEVE",qid,len(found),flush=True)
    del model
    dump(output/"public_retrieval.json",rows)
    dump(output/"retrieval_lock.json",dict(locked_at=utc(),indexed_chunks=len(chunks),
               hashes={str((output/"public_retrieval.json").resolve()):digest(output/"public_retrieval.json")},
               normative_fields_excluded_from_ranking=True))


def proxy_plan(raw,checks,question,evidence):
    requested=question["requested_slots"];rows=raw.get("slots",[]);count=len(requested)
    valid=isinstance(rows,list) and len(rows)==count and all(isinstance(r,dict) and type(r.get("index")) is int for r in rows)
    valid=valid and {r["index"] for r in rows}==set(range(1,count+1))
    by={e["eid"]:e["text"] for e in evidence};alltext=auto.norm("\n".join(by.values()));slots=[]
    if valid:
        indexed={r["index"]:r for r in rows}
        for req in requested:
            r=indexed[req["index"]]
            expected=r.get("expected");unit=r.get("unit");aliases=r.get("aliases")
            ok=r.get("supported")=="1" and isinstance(expected,str) and bool(expected) and isinstance(unit,str) and isinstance(aliases,list)
            ok=ok and all(isinstance(a,str) and bool(a) and auto.norm(a) in alltext for a in aliases)
            ok=ok and (req["type"]!="number" or (numeric_value(expected) and not aliases))
            quote=exact_source_quote(r.get("quote",""),by.get(r.get("eid"),""))
            ok=ok and bool(quote) and auto.norm(expected) in auto.norm(quote["quote"])
            valid=valid and ok
            if ok:slots.append(dict(**req,expected=expected,unit=unit,aliases=aliases,eid=r["eid"],quote=quote["quote"],source_span=quote))
    valid=valid and len(checks)==2 and all(checks_pass(c,count,("supported","unique","aliases_valid")) for c in checks)
    return dict(basis_complete=bool(valid),slots=slots if valid else [],facts=plan_facts(slots) if valid else [])


def plans(output):
    m,questions=verify_questions(output);verify_hashes(load(output/"retrieval_lock.json")["hashes"])
    if (output/"proxy_lock.json").exists():verify_hashes(load(output/"proxy_lock.json")["hashes"]);return
    retrieved=load(output/"public_retrieval.json")
    def one(item):
        qid,q=item;row=retrieved[qid];payload=dict(question=q["question"],requested_slots=q["requested_slots"],retrieved_evidence=row["evidence"])
        raw=auto.request(output,"proxy_extract_"+qid,EXTRACT_SLOTS,payload)
        checks=[auto.request(output,f"proxy_check_{qid}_r{n}",CHECK_SLOTS+
                             ("\n核对字段与值的对应关系。" if n==1 else "\n核对实验口径和别名是否有来源支持。"),
                             dict(**payload,temporary_slots=raw.get("slots",[])),n) for n in (1,2)]
        plan=dict(question=q["question"],evidence=row["evidence"],raw=raw,checks=checks,**proxy_plan(raw,checks,q,row["evidence"]))
        dump(output/"proxy_plans"/(qid+".json"),plan)
        return plan["basis_complete"]
    batch(list(questions.items()),one,m["config"]["workers"],"OBJECTIVE_PLANS")
    dump(output/"proxy_lock.json",dict(locked_at=utc(),hashes={str(p.resolve()):digest(p) for p in sorted((output/"proxy_plans").glob("*.json"))},
                                       before_new_answers=True))


def freeze_answers(output):
    m,questions=verify_questions(output)
    verify_hashes(load(output/"retrieval_lock.json")["hashes"]);verify_hashes(load(output/"proxy_lock.json")["hashes"])
    if (output/"answer_cases_lock.json").exists():verify_hashes(load(output/"answer_cases_lock.json")["hashes"]);return
    refs=load(output/"private_references.json");retrieved=load(output/"public_retrieval.json")
    temporary=[]
    for qid,q in questions.items():
        known=controls(dict(slots=refs[qid]["slots"]))
        for condition,row in known.items():temporary.append((qid,"control_"+condition,row))
        temporary.append((qid,"live_normal",None))
    random.Random(m["config"]["seed"]).shuffle(temporary)
    cases={};keys={};known={}
    for i,(qid,condition,row) in enumerate(temporary,1):
        aid=f"S{i:04d}";q=questions[qid];context=retrieved[qid]["context"]
        cases[aid]=dict(question=q["question"],context=context,context_sha256=sha(context))
        keys[aid]=dict(question_key=qid,condition=condition)
        if row:known[aid]=row
    for name,value in (("public_cases.json",cases),("private_case_key.json",keys),("private_control_answers.json",known)):
        dump(output/name,value)
    paths=[output/n for n in ("public_cases.json","private_case_key.json","private_control_answers.json","questions_lock.json","proxy_lock.json","retrieval_lock.json")]
    dump(output/"answer_cases_lock.json",dict(locked_at=utc(),hashes={str(p.resolve()):digest(p) for p in paths},cases=len(cases),
                                              control_truth_known_before_scoring=True,no_response_based_selection=True))


def verify_cases(output):
    m,_=verify_questions(output)
    for name in ("retrieval_lock.json","proxy_lock.json","answer_cases_lock.json"):verify_hashes(load(output/name)["hashes"])
    return m,load(output/"public_cases.json")


def generate(output):
    import evaluate
    m,cases=verify_cases(output);keys=load(output/"private_case_key.json");cfg=m["config"]
    def one(item):
        aid,case=item;payload=dict(model=cfg["model"],temperature=cfg["generation_temperature"],max_tokens=cfg["generation_max_tokens"],
             messages=[dict(role="system",content=cfg["system_prompt"]),dict(role="user",content=f'问题：{case["question"]}\n\n参考上下文：\n{case["context"]}')])
        hashed=sha(json.dumps(payload,ensure_ascii=False,sort_keys=True));path=output/"generations"/(aid+".json")
        if path.exists():
            if load(path)["payload_sha256"]!=hashed:raise ValueError("generation payload changed")
            return "cached"
        client=evaluate._get_client().with_options(timeout=90,max_retries=1)
        try:
            started=utc();response=client.chat.completions.create(**payload);choice=response.choices[0];answer=(choice.message.content or "").strip()
            if not answer:raise ValueError("empty answer")
            dump(path,dict(payload=payload,payload_sha256=hashed,answer=answer,started_at=started,finished_at=utc(),response_id=response.id,
                 actual_model=response.model,finish_reason=choice.finish_reason,usage=response.usage.model_dump(),raw_response=response.model_dump(),
                 rule_refusal=int(explicit_refusal(answer))))
            return choice.finish_reason
        finally:client.close()
    batch([(aid,c) for aid,c in cases.items() if keys[aid]["condition"]=="live_normal"],one,cfg["workers"],"OBJECTIVE_GENERATE")


def compare(output,name,question,answer,plan):
    originals=[];schema_retries=[];quote_retries=[];rounds=[]
    if plan["basis_complete"]:
        facts=plan["facts"];payload=comparison_payload(question,answer,facts);count=len(facts)
        for n in (1,2):
            prompt=COMPARE_SLOTS+("\n按所问对象逐字段检查回答。" if n==1 else "\n逐字段检查遗漏、值与对象是否错配。")
            original=auto.request(output,f"compare_{name}_r{n}",prompt,payload,n);raw=original;schema_retry=None;quote_retry=None
            if not schema_valid(raw,count):
                schema_retry=auto.request(output,f"schema_retry_{name}_r{n}",prompt+
                    "\n上一份输出的行数/索引无效。请重新按required_indices各返回一行，不能增删或合并事实。",payload,n)
                raw=schema_retry
            invalid=validated_comparison(raw,count,answer)[2]
            if invalid:
                quote_retry=auto.request(output,f"quote_retry_{name}_r{n}",QUOTE_REPAIR,
                             dict(**payload,point_results=raw.get("point_results",[]),invalid_indices=invalid),n)
                raw=repair_quotes(raw,quote_retry,invalid)
            originals.append(original);schema_retries.append(schema_retry);quote_retries.append(quote_retry);rounds.append(raw)
    return dict(original_rounds=originals,schema_retries=schema_retries,quote_retries=quote_retries,rounds=rounds,
                signal=agreement_score(rounds,len(plan["facts"]),answer,plan["basis_complete"]))


def score(output):
    m,cases=verify_cases(output);keys=load(output/"private_case_key.json");refs=load(output/"private_references.json")
    controls_locked=load(output/"private_control_answers.json")
    if any(not (output/"generations"/(aid+".json")).exists() for aid in cases if keys[aid]["condition"]=="live_normal"):
        raise ValueError("fresh normal answers incomplete")
    def one(item):
        aid,case=item;qid=keys[aid]["question_key"];known=controls_locked.get(aid)
        gen=known if known else load(output/"generations"/(aid+".json"));answer=gen["answer"]
        quality_result=None
        if known:quality=known["quality"];provenance="source_supported_control_by_construction"
        else:
            quality_result=compare(output,"reference_"+aid,case["question"],answer,refs[qid])
            quality=point_quality(quality_result,gen["finish_reason"]);provenance="two_round_automatic_reference"
        proxy=compare(output,"proxy_"+aid,case["question"],answer,load(output/"proxy_plans"/(qid+".json")))
        if not known and gen["finish_reason"]!="stop":proxy["signal"].update(status="U",score=None)
        dump(output/"scores"/(aid+".json"),dict(quality=quality,quality_provenance=provenance,reference_comparison=quality_result,
                    proxy_comparison=proxy,rule_refusal=int(explicit_refusal(answer))))
        return dict(quality=quality,proxy=proxy["signal"]["status"])
    batch(list(cases.items()),one,m["config"]["workers"],"OBJECTIVE_SCORE")


def metrics(rows):
    good=[r for r in rows if r["quality"]=="2"];bad=[r for r in rows if r["quality"] in {"0","1"}]
    return dict(n=len(rows),complete=len(good),incomplete=len(bad),quality_u=sum(r["quality"]=="U" for r in rows),
       proxy_u=sum(r["proxy_status"]=="U" for r in rows),false_flags=sum(r["proxy_status"]=="incomplete" for r in good),
       flagged_incomplete=sum(r["proxy_status"]=="incomplete" for r in bad),missed_incomplete=sum(r["proxy_status"]=="complete" for r in bad),
       unresolved_incomplete=sum(r["proxy_status"]=="U" for r in bad),refusals=sum(r["rule_refusal"] for r in rows))


def analyze(output):
    m,cases=verify_cases(output);questions=load(output/"public_questions.json");keys=load(output/"private_case_key.json")
    refs=load(output/"private_references.json");retrieved=load(output/"public_retrieval.json")
    rows=[];scope=[]
    for aid,case in cases.items():
        s=load(output/"scores"/(aid+".json"));proxy=s["proxy_comparison"]["signal"]
        rows.append(dict(aid=aid,**keys[aid],quality=s["quality"],quality_provenance=s["quality_provenance"],
                          proxy_status=proxy["status"],proxy_score=proxy["score"],rule_refusal=s["rule_refusal"]))
    for qid,q in questions.items():
        p=load(output/"proxy_plans"/(qid+".json"));evidence=retrieved[qid]["evidence"]
        # Availability is evaluation-only, after question-only ranking.
        normtexts=[auto.norm(e["text"]) for e in evidence]
        scope.append(dict(question_key=qid,slots=len(refs[qid]["slots"]),proxy_basis_valid=int(p["basis_complete"]),
            reference_quotes_present=sum(any(auto.norm(s["quote"]) in t for t in normtexts) for s in refs[qid]["slots"]),
            interpretation="literal_availability_only_not_semantic_support"))
    table=[dict(condition=c,**metrics([r for r in rows if c=="controls" and r["condition"].startswith("control_") or r["condition"]==c]))
           for c in ("controls","control_complete","control_omitted","control_corrupted","live_normal")]
    by={r["condition"]:r for r in table};gate=m["gate"]
    cc=by["controls"];om=by["control_omitted"];bad=by["control_corrupted"]
    checks=dict(enough_questions=len(questions)>=gate["minimum_qualified_questions"],
        control_proxy_available=bool(cc["n"]) and cc["proxy_u"]/cc["n"]<=gate["maximum_control_proxy_u_fraction"],
        complete_false_flags=by["control_complete"]["false_flags"]<=gate["maximum_complete_false_flags"],
        omitted_detection=bool(om["n"]) and om["flagged_incomplete"]/om["n"]>=gate["minimum_omitted_flag_fraction"],
        corrupted_detection=bool(bad["n"]) and bad["flagged_incomplete"]/bad["n"]>=gate["minimum_corrupted_flag_fraction"],
        candidate_scope_acceptance=len(questions)/12>=gate["minimum_reference_check_pass_fraction"])
    api=[load(p) for p in (output/"api_records").glob("*.json")];gens=[load(p) for p in (output/"generations").glob("*.json")]
    value=dict(candidates=12,qualified_questions=len(questions),controlled_answers=cc["n"],fresh_normal_answers=len(gens),metrics=table,
          proxy_bases_valid=sum(r["proxy_basis_valid"] for r in scope),
          gate=dict(status="passed" if all(checks.values()) else "not_passed",checks=checks,scope="scoring_controls_only_not_live_fault_detection"),
          api_responses=len(api)+len(gens),failed_api_records=sum(not r.get("ok") for r in api),
          total_tokens=sum(r.get("usage",{}).get("total_tokens",0) for r in api+gens),
          actual_models=dict(Counter([r.get("model_returned","error") for r in api]+[g["actual_model"] for g in gens])),
          schema_retry_responses=sum(p.name.startswith("schema_retry_") for p in (output/"api_records").glob("*.json")),
          quote_retry_responses=sum(p.name.startswith("quote_retry_") for p in (output/"api_records").glob("*.json")),interpretation=m["interpretation"])
    write_csv(output/"answer_scores.csv",rows);write_csv(output/"signal_metrics.csv",table);write_csv(output/"retrieval_availability.csv",scope)
    dump(output/"analysis_summary.json",value);print(json.dumps(value,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action",choices=["prepare","candidates","freeze_questions","retrieve","plans","freeze_answers","generate","score","analyze"])
    for name in ("previous","validation","prior","output"):ap.add_argument("--"+name,type=Path,required=name=="output")
    a=ap.parse_args()
    if a.action=="prepare":prepare(a.previous,a.validation,a.prior,a.output)
    else:{"candidates":candidates,"freeze_questions":freeze_questions,"retrieve":retrieval,"plans":plans,
          "freeze_answers":freeze_answers,"generate":generate,"score":score,"analyze":analyze}[a.action](a.output)
