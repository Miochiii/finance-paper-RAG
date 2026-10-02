"""Frozen development comparison on 11 seen-source objective questions.

Fresh baseline/enhanced pairs, synthetic evaluator controls, no answer selection.
This is not a new-source validation or a live change-point trial.
"""
import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path

import rag_auto_evaluation as auto
from rag_fresh_change_experiment import dump, load, digest, sha, utc, write_csv
from rag_prospective_coverage_validation import verify_hashes
from rag_core.mineru_loader import blocks_to_text
from rag_subtle_fault_pilot import batch, LocalReranker
from rag_objective_coverage_pilot import proxy_plan, compare, metrics
from rag_objective_slots import (EXTRACT_SLOTS, CHECK_SLOTS, controls, agreement_score,
                                 exact_source_quote)
from rag_mineru_coverage_v4 import evidence_projection, source_context, point_quality
from rag_field_refusal import field_refusal
from rag_refusal import explicit_refusal
from rag_public_field_retrieval import (CONFIG, field_chunks, retrieve_fields,
                                       surface_map, source_surface_anchor, public_query)


def prepare(previous, output):
    import evaluate
    output.mkdir(parents=True, exist_ok=True)
    parent=load(previous/"protocol_lock.json")
    for name in ("input_hashes", "source_hashes", "initial_hashes"): verify_hashes(parent[name])
    for name in ("questions_lock.json", "retrieval_lock.json", "proxy_lock.json", "answer_cases_lock.json"):
        verify_hashes(load(previous/name)["hashes"])
    paths=[previous/n for n in ("protocol_lock.json", "public_questions.json", "private_references.json",
          "source_manifest.json", "public_retrieval.json", "analysis_summary.json", "private_case_key.json")]
    paths+=sorted((previous/"proxy_plans").glob("*.json"))+sorted((previous/"scores").glob("*.json"))
    modules=["rag_field_retrieval_trial.py", "rag_public_field_retrieval.py", "rag_field_refusal.py",
             "rag_objective_slots.py", "rag_objective_coverage_pilot.py", "rag_auto_evaluation.py",
             "rag_answer_quote_v4.py", "rag_refusal.py", "rag_mineru_coverage_v4.py", "evaluate.py",
             "rag_core/mineru_loader.py", "rag_subtle_fault_pilot.py", "rag_fresh_change_experiment.py",
             "rag_semantic_coverage_pilot.py", "diagnose_mineru_evidence_coverage.py",
             "rag_prospective_coverage_validation.py"]
    paths += [Path(__file__).parent/n for n in modules]
    hashes={str(p.resolve()):digest(p) for p in paths}
    if (output/"protocol_lock.json").exists():
        if load(output/"protocol_lock.json")["input_hashes"]!=hashes: raise ValueError("protocol changed")
        verify(output); return
    for name in ("public_questions.json", "private_references.json", "source_manifest.json"):
        dump(output/name,load(previous/name))
    dump(output/"baseline_retrieval.json",load(previous/"public_retrieval.json"))
    for path in sorted((previous/"proxy_plans").glob("*.json")):
        dump(output/"baseline_plans"/path.name,load(path))
    dump(output/"protocol_lock.json",dict(version="public_field_retrieval_v6_development",
         locked_at=utc(),previous=str(previous.resolve()),input_hashes=hashes,
         source_hashes=parent["source_hashes"],model_hashes=parent["model_hashes"],
         config=dict(workers=4,seed=20261002,model="deepseek-chat",generation_temperature=0,
                     generation_max_tokens=512,system_prompt=evaluate.EVAL_PROMPT,
                     reranker_path=parent["config"]["reranker_path"]),retrieval_config=CONFIG,
         gate=dict(minimum_questions=10,maximum_control_proxy_u_fraction=.2,
                   maximum_complete_false_flags=0,minimum_omitted_flag_fraction=.8,
                   minimum_corrupted_flag_fraction=.8),
         controls_scope="synthetic_answers_not_live_evidence_faults",
         interpretation="development_seen_questions_seen_sources_new_paired_answers_multiple_changes_no_causal_ablation",
         fixed_question_bank=True,maximum_new_generations=22,all_unresolved_retained=True,
         no_answer_based_tuning=True,refusal_rule="pre_frozen_rag_field_refusal"))
    print("V6 frozen: 11 seen questions, 22 fresh paired answers, 33 synthetic controls",flush=True)


def verify(output):
    m=load(output/"protocol_lock.json")
    for name in ("input_hashes", "source_hashes"): verify_hashes(m[name])
    return m


def retrieval(output):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    m=verify(output)
    if (output/"retrieval_lock.json").exists(): verify_hashes(load(output/"retrieval_lock.json")["hashes"]); return
    verify_hashes(m["model_hashes"])
    chunks=[]
    for source,row in sorted(load(output/"source_manifest.json").items()):
        chunks.extend(field_chunks(blocks_to_text(load(Path(row["mineru_path"]))),source))
    vectorizer=TfidfVectorizer(analyzer="char",ngram_range=(2,3),max_features=60000,lowercase=True,dtype=np.float32)
    matrix=vectorizer.fit_transform([surface_map(c["text"])[0] for c in chunks])
    model=LocalReranker(m["config"]["reranker_path"]);rows={}
    for qid,q in load(output/"public_questions.json").items():
        found,trace=retrieve_fields(q,chunks,vectorizer,matrix,model)
        rows[qid]=dict(**q,retrieved=found,evidence=evidence_projection(found),
                       context=source_context(found),query_trace=trace)
        print("FIELD_RETRIEVE",qid,len(found),trace["candidate_count"],flush=True)
    del model
    dump(output/"public_retrieval.json",rows)
    dump(output/"retrieval_lock.json",dict(locked_at=utc(),indexed_chunks=len(chunks),
          hashes={str((output/"public_retrieval.json").resolve()):digest(output/"public_retrieval.json")},
          no_private_reference_in_ranker=True,maximum_raw_characters=6000))
    # Gold quotes are used only after all retrieval results have been frozen.
    references=load(output/"private_references.json");old=load(output/"baseline_retrieval.json");diagnostic=[]
    for qid,row in rows.items():
        quote_counts=[]
        for passages in (old[qid]["retrieved"],row["retrieved"]):
            quote_counts.append(sum(any(auto.norm(s["quote"]) in auto.norm(p["text"]) for p in passages)
                                    for s in references[qid]["slots"]))
        diagnostic.append(dict(question_key=qid,slots=len(references[qid]["slots"]),
                               baseline_literal_quotes=quote_counts[0],enhanced_literal_quotes=quote_counts[1],
                               baseline_characters=sum(len(p["text"]) for p in old[qid]["retrieved"]),
                               enhanced_characters=sum(len(p["text"]) for p in row["retrieved"])))
    write_csv(output/"local_retrieval_diagnostic.csv",diagnostic)
    print("Literal diagnostic (not semantic validation):",diagnostic,flush=True)


def mapped_proxy(raw, checks, question, evidence):
    """Keep raw quotes intact; trace exact, limited source presentation mapping."""
    strict=proxy_plan(raw,checks,question,evidence)
    changed=copy.deepcopy(raw);mappings=[]
    by={e["eid"]:e["text"] for e in evidence}
    for row in changed.get("slots",[]):
        if not isinstance(row,dict) or not isinstance(row.get("expected"),str): continue
        anchor=exact_source_quote(row.get("quote",""),by.get(row.get("eid"),""))
        if not anchor or auto.norm(row["expected"]) in auto.norm(anchor["quote"]): continue
        equivalent=source_surface_anchor(row["expected"],anchor["quote"])
        if equivalent:
            mappings.append(dict(index=row.get("index"),**equivalent))
            row["expected"]=equivalent["literal"]
    normalized=proxy_plan(changed,checks,question,evidence)
    return dict(**normalized,strict_basis_complete=strict["basis_complete"],mapped_extraction=changed,source_mappings=mappings)


def plans(output):
    m=verify(output);verify_hashes(load(output/"retrieval_lock.json")["hashes"])
    if (output/"proxy_lock.json").exists():verify_hashes(load(output/"proxy_lock.json")["hashes"]);return
    questions=load(output/"public_questions.json");retrieved=load(output/"public_retrieval.json")
    def one(item):
        qid,q=item;evidence=retrieved[qid]["evidence"]
        payload=dict(question=q["question"],requested_slots=q["requested_slots"],retrieved_evidence=evidence)
        raw=auto.request(output,"proxy_extract_"+qid,EXTRACT_SLOTS,payload)
        checks=[auto.request(output,f"proxy_check_{qid}_r{n}",CHECK_SLOTS+
               ("\n核对字段与值的对应关系。" if n==1 else "\n核对实验口径和别名是否有来源支持。"),
               dict(**payload,temporary_slots=raw.get("slots",[])),n) for n in (1,2)]
        p=dict(question=q["question"],evidence=evidence,raw=raw,checks=checks,**mapped_proxy(raw,checks,q,evidence))
        dump(output/"proxy_plans"/(qid+".json"),p)
        return dict(basis=p["basis_complete"],strict=p["strict_basis_complete"])
    batch(list(questions.items()),one,m["config"]["workers"],"FIELD_PLANS")
    dump(output/"proxy_lock.json",dict(locked_at=utc(),before_answers=True,
          hashes={str(p.resolve()):digest(p) for p in sorted((output/"proxy_plans").glob("*.json"))}))


def freeze_cases(output):
    m=verify(output)
    for name in ("retrieval_lock.json","proxy_lock.json"):verify_hashes(load(output/name)["hashes"])
    if (output/"cases_lock.json").exists():verify_hashes(load(output/"cases_lock.json")["hashes"]);return
    qs=load(output/"public_questions.json");refs=load(output/"private_references.json")
    old=load(output/"baseline_retrieval.json");new=load(output/"public_retrieval.json")
    pending=[]
    for qid,q in qs.items():
        for condition,bank in (("live_baseline",old),("live_enhanced",new)):
            pending.append((qid,condition,bank[qid]["context"],None))
        for condition,row in controls(dict(slots=refs[qid]["slots"])).items():
            pending.append((qid,"control_"+condition,new[qid]["context"],row))
    random.Random(m["config"]["seed"]).shuffle(pending)
    cases={};keys={};known={}
    for n,(qid,condition,context,row) in enumerate(pending,1):
        aid=f"V{n:04d}"
        cases[aid]=dict(question=qs[qid]["question"],context=context,context_sha256=sha(context))
        keys[aid]=dict(question_key=qid,condition=condition)
        if row:known[aid]=row
    for name,value in (("public_cases.json",cases),("private_case_key.json",keys),("private_control_answers.json",known)):
        dump(output/name,value)
    paths=[output/n for n in ("public_cases.json","private_case_key.json","private_control_answers.json",
            "public_questions.json","private_references.json","baseline_retrieval.json","proxy_lock.json","retrieval_lock.json")]
    paths+=sorted((output/"baseline_plans").glob("*.json"))
    dump(output/"cases_lock.json",dict(locked_at=utc(),cases=len(cases),new_generations=22,
         hashes={str(p.resolve()):digest(p) for p in paths},no_response_based_selection=True))
    print("Frozen cases:",len(cases),flush=True)


def verify_cases(output):
    m=verify(output)
    for name in ("retrieval_lock.json","proxy_lock.json","cases_lock.json"):verify_hashes(load(output/name)["hashes"])
    return m


def generate(output):
    import evaluate
    m=verify_cases(output);cases=load(output/"public_cases.json");keys=load(output/"private_case_key.json");cfg=m["config"]
    def one(item):
        aid,case=item;payload=dict(model=cfg["model"],temperature=cfg["generation_temperature"],max_tokens=cfg["generation_max_tokens"],
           messages=[dict(role="system",content=cfg["system_prompt"]),dict(role="user",content=f'问题：{case["question"]}\n\n参考上下文：\n{case["context"]}')])
        hashed=sha(json.dumps(payload,ensure_ascii=False,sort_keys=True));path=output/"generations"/(aid+".json")
        if path.exists():
            if load(path)["payload_sha256"]!=hashed:raise ValueError("generation input changed")
            return "cached"
        client=evaluate._get_client().with_options(timeout=90,max_retries=1)
        try:
            started=utc();response=client.chat.completions.create(**payload);choice=response.choices[0];answer=(choice.message.content or "").strip()
            if not answer:raise ValueError("empty answer")
            dump(path,dict(payload=payload,payload_sha256=hashed,answer=answer,started_at=started,finished_at=utc(),
                 response_id=response.id,actual_model=response.model,finish_reason=choice.finish_reason,
                 usage=response.usage.model_dump(),raw_response=response.model_dump()))
            return choice.finish_reason
        finally:client.close()
    batch([(a,c) for a,c in cases.items() if keys[a]["condition"].startswith("live_")],one,cfg["workers"],"FIELD_GENERATE")


def score(output):
    m=verify_cases(output);cases=load(output/"public_cases.json");keys=load(output/"private_case_key.json")
    known=load(output/"private_control_answers.json");refs=load(output/"private_references.json");qs=load(output/"public_questions.json")
    def one(item):
        aid,case=item;k=keys[aid];qid=k["question_key"];path=output/"scores"/(aid+".json")
        if path.exists():return "cached"
        gen=known.get(aid) or load(output/"generations"/(aid+".json"));answer=gen["answer"]
        reference=None
        if aid in known:quality=known[aid]["quality"]
        else:
            reference=compare(output,"reference_"+aid,case["question"],answer,refs[qid])
            quality=point_quality(reference,gen["finish_reason"])
        folder="baseline_plans" if k["condition"]=="live_baseline" else "proxy_plans"
        proxy=compare(output,"proxy_"+aid,case["question"],answer,load(output/folder/(qid+".json")))
        if aid not in known and gen["finish_reason"]!="stop":proxy["signal"].update(status="U",score=None)
        refusal=field_refusal(answer,[s["label"] for s in qs[qid]["requested_slots"]])
        dump(path,dict(quality=quality,reference_comparison=reference,proxy_comparison=proxy,
             legacy_refusal=int(explicit_refusal(answer)),field_refusal=refusal))
        return dict(quality=quality,proxy=proxy["signal"]["status"],refusal=refusal["refusal"])
    batch(list(cases.items()),one,m["config"]["workers"],"FIELD_SCORE")


def analyze(output):
    m=verify_cases(output);qs=load(output/"public_questions.json");keys=load(output/"private_case_key.json");rows=[];plans=[]
    for aid,k in keys.items():
        s=load(output/"scores"/(aid+".json"));p=s["proxy_comparison"]["signal"]
        rows.append(dict(aid=aid,**k,quality=s["quality"],proxy_status=p["status"],proxy_score=p["score"],
                         rule_refusal=s["field_refusal"]["refusal"],legacy_refusal=s["legacy_refusal"]))
    conditions=("controls","control_complete","control_omitted","control_corrupted","live_baseline","live_enhanced")
    table=[dict(condition=c,**metrics([r for r in rows if
              (c=="controls" and r["condition"].startswith("control_")) or r["condition"]==c])) for c in conditions]
    by={r["condition"]:r for r in table};gate=m["gate"]
    checks=dict(enough_questions=len(qs)>=gate["minimum_questions"],
         controls_available=by["controls"]["proxy_u"]/by["controls"]["n"]<=gate["maximum_control_proxy_u_fraction"],
         complete_false_flags=by["control_complete"]["false_flags"]<=gate["maximum_complete_false_flags"],
         omitted_detection=by["control_omitted"]["flagged_incomplete"]/by["control_omitted"]["n"]>=gate["minimum_omitted_flag_fraction"],
         corrupted_detection=by["control_corrupted"]["flagged_incomplete"]/by["control_corrupted"]["n"]>=gate["minimum_corrupted_flag_fraction"])
    pairs=[]
    for qid in qs:
        a=next(r for r in rows if r["question_key"]==qid and r["condition"]=="live_baseline")
        b=next(r for r in rows if r["question_key"]==qid and r["condition"]=="live_enhanced")
        p=load(output/"proxy_plans"/(qid+".json"));old=load(output/"baseline_plans"/(qid+".json"))
        plans.append(dict(question_key=qid,baseline_basis=int(old["basis_complete"]),
                          enhanced_strict_basis=int(p["strict_basis_complete"]),enhanced_mapped_basis=int(p["basis_complete"]),
                          source_mappings=len(p["source_mappings"])))
        determinate=a["quality"]!="U" and b["quality"]!="U"
        pairs.append(dict(question_key=qid,baseline_quality=a["quality"],enhanced_quality=b["quality"],
                          baseline_proxy=a["proxy_status"],enhanced_proxy=b["proxy_status"],
                          baseline_refusal=a["rule_refusal"],enhanced_refusal=b["rule_refusal"],
                          improved=int(determinate and int(b["quality"])>int(a["quality"])),
                          worsened=int(determinate and int(b["quality"])<int(a["quality"])),quality_u=int(not determinate)))
    api=[load(p) for p in (output/"api_records").glob("*.json")];gens=[load(p) for p in (output/"generations").glob("*.json")]
    value=dict(questions=len(qs),fresh_paired_answers=len(gens),synthetic_controls=33,metrics=table,
         baseline_proxy_bases=sum(p["baseline_basis"] for p in plans),
         enhanced_strict_proxy_bases=sum(p["enhanced_strict_basis"] for p in plans),
         enhanced_mapped_proxy_bases=sum(p["enhanced_mapped_basis"] for p in plans),
         paired_improved=sum(p["improved"] for p in pairs),paired_worsened=sum(p["worsened"] for p in pairs),
         paired_quality_u=sum(p["quality_u"] for p in pairs),
         gate=dict(status="passed" if all(checks.values()) else "not_passed",checks=checks,
                   scope="evaluator_synthetic_controls_only"),
         api_responses=len(api)+len(gens),failed_api_records=sum(not r.get("ok") for r in api),
         total_tokens=sum(r.get("usage",{}).get("total_tokens",0) for r in api+gens),
         actual_models=dict(Counter([r.get("model_returned","error") for r in api]+[g["actual_model"] for g in gens])),
         generation_finishes=dict(Counter(g["finish_reason"] for g in gens)),
         interpretation=m["interpretation"])
    for name,r in (("answer_scores.csv",rows),("signal_metrics.csv",table),("paired_quality.csv",pairs),("proxy_basis_comparison.csv",plans)):
        write_csv(output/name,r)
    dump(output/"analysis_summary.json",value);print(json.dumps(value,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action",choices=["prepare","retrieve","plans","freeze_cases","generate","score","analyze"])
    ap.add_argument("--previous",type=Path);ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    if a.action=="prepare":prepare(a.previous,a.output)
    else:{"retrieve":retrieval,"plans":plans,"freeze_cases":freeze_cases,"generate":generate,
          "score":score,"analyze":analyze}[a.action](a.output)
