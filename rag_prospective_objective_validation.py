"""Prospective new-question validation of frozen V6.1, on 8 corpus papers.

Papers are excluded from the current coverage-method development/screening set;
they already exist in the corpus and old annotation banks. This is not a wholly
unseen-corpus or independent-human evaluation. No test-outcome-based tuning.
"""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import rag_auto_evaluation as auto
import rag_field_retrieval_trial as experiment
from rag_fresh_change_experiment import load,dump,digest,sha,utc,write_csv
from rag_prospective_coverage_validation import verify_hashes
from rag_core.mineru_loader import blocks_to_text
from rag_subtle_fault_pilot import batch,LocalReranker
from rag_public_field_retrieval import field_chunks,retrieve_fields,surface_map,public_query
from rag_sentence_boundary_repair import repair_selected
from rag_objective_slots import (CANDIDATE_PROMPT,CANDIDATE_CHECK,validate_candidate,
          checks_pass,public_slots,plan_facts,controls)
from rag_objective_scope_revision import SCOPE_PROMPT,scalar_slots,make_question
from rag_mineru_coverage_v4 import evidence_projection,source_context


def question_signature(question):
    """Exact request/scope duplicate check; paper titles and formatting excluded."""
    text=re.sub(r"《[^》]+》", "",question)
    if "针对" in text:text=text.split("针对",1)[1]
    return auto.norm(text)


def choose_cards(text,source,count=4):
    """Public source-only fixed sample; no old/new answers or gold values."""
    candidates=[]
    for match in re.finditer(r"\[TABLE_START\].*?\[/TABLE_END\]",text,re.S):
        part=match.group()
        if 80<=len(part)<=900 and len(re.findall(r"\d+(?:\.\d+)?",part))>=3:
            score=min(30,len(re.findall(r"\d+(?:\.\d+)?",part)))+sum(
                word in part for word in ("样本","参数","准确","收益","性能","描述","数据"))*3
            candidates.append((score,match.start(),match.end(),"intact_table"))
    # Fixed fallback: paragraph windows with numeric context, bounded to 900 chars.
    if len(candidates)<count:
        for match in re.finditer(r"[^\n]+(?:\n[^\n]+){0,2}",text):
            part=match.group()[:900]
            if not (180<=len(part)<=900) or "[TABLE" in part or "参考文献" in part[:80]:continue
            digits=len(re.findall(r"\d+(?:\.\d+)?",part))
            if digits<3 or not any(w in part for w in ("本文","样本","实验","数据","实证")):continue
            candidates.append((min(20,digits),match.start(),match.start()+len(part),"numeric_prose"))
    selected=[]
    for score,start,end,kind in sorted(candidates,key=lambda c:(-c[0],c[1])):
        if any(max(0,min(end,x["end"])-max(start,x["start"]))>0 for x in selected):continue
        raw=text[start:end]
        selected.append(dict(eid=f"E{len(selected)+1}",text=raw,source=source,start=start,end=end,
                             text_sha256=sha(raw),kind=kind,selection_score=score))
        if len(selected)==count:break
    return selected


def prepare(previous,original,objective,v4,output):
    parent=experiment.verify_cases(previous)
    verify_hashes(load(previous/"boundary_validation_lock.json")["hashes"])
    output.mkdir(parents=True,exist_ok=True)
    source_old=load(original/"source_manifest.json")
    development=set(source_old)
    for row in load(objective/"candidate_inputs.json").values():development.update(row["sources"])
    for row in load(v4/"public_questions.json").values():development.update(row["actual_sources"])
    cache_path=Path("data/docs_cache_v2.json");cache=load(cache_path)
    available=sorted(set(cache)-development)
    banks=sorted(Path("data/annotations").glob("*.csv"));prior_rows=[]
    for path in banks:
        with path.open(encoding="utf-8-sig",newline="") as f:prior_rows+=list(csv.DictReader(f))
    known_questions=sorted({r["question"] for r in prior_rows if r.get("question")})
    manifest={};texts={}
    for source in sorted(cache):
        matches=list(Path("data/corpora/金融论文/mineru_out/batch").glob(Path(source).stem+"/vlm/*_content_list.json"))
        if len(matches)!=1:raise ValueError("MinerU source path ambiguous: "+source)
        path=matches[0].resolve();texts[source]=blocks_to_text(load(path))
        manifest[source]=dict(mineru_path=str(path),mineru_sha256=digest(path),text_sha256=sha(texts[source]))
    inputs={};exposure=[];cards_inventory={}
    for n,source in enumerate(available,1):
        cards=choose_cards(texts[source],source);cards_inventory[source]=cards
        old=[r for r in prior_rows if source in (r.get("gold_docs") or "").split("|")]
        exposure.append(dict(source=source,old_annotation_rows=len(old),old_system_answer_rows=sum(bool(r.get("system_answer")) for r in old),
                             coverage_method_development_exposure=0,available_cards=len(cards),already_in_corpus=1))
        # Exactly two attempts per eligible source, disjoint pairs of at most 900 chars.
        for offset in (0,2):
            bid=f"H{n:03d}_{offset//2+1}"
            inputs[bid]=dict(source=source,evidence=cards[offset:offset+2],
                    old_questions=sorted({r["question"] for r in old if r.get("question")}))
    modules=[Path(p) for p in parent["input_hashes"] if Path(p).suffix==".py"]
    paths=modules+[Path(__file__),cache_path,original/"source_manifest.json",objective/"candidate_inputs.json",
                  v4/"public_questions.json",previous/"protocol_lock.json",previous/"boundary_success_summary.json"]+banks
    hashes={str(p.resolve()):digest(p) for p in paths}
    if (output/"protocol_lock.json").exists():
        if load(output/"protocol_lock.json")["input_hashes"]!=hashes:raise ValueError("prospective inputs changed")
        experiment.verify(output);return
    for name,value in (("source_manifest.json",manifest),("candidate_inputs.json",inputs),("old_questions.json",known_questions),
          ("development_source_registry.json",sorted(development)),("candidate_source_cards.json",cards_inventory)):
        dump(output/name,value)
    write_csv(output/"source_exposure.csv",exposure)
    copied=[output/n for n in ("source_manifest.json","candidate_inputs.json","old_questions.json",
                              "development_source_registry.json","candidate_source_cards.json")]
    dump(output/"protocol_lock.json",dict(version="prospective_objective_v7_validation",locked_at=utc(),previous=str(previous.resolve()),
          input_hashes=hashes,initial_hashes={str(p.resolve()):digest(p) for p in copied},
          source_hashes={r["mineru_path"]:r["mineru_sha256"] for r in manifest.values()},model_hashes=parent["model_hashes"],
          config=parent["config"],retrieval_config=parent["retrieval_config"],boundary_config=parent["boundary_config"],gate=parent["gate"],
          validation_gate=dict(minimum_questions=10,minimum_sources=6,minimum_live_complete_fraction=.8,
                     maximum_live_quality_u_fraction=.1,maximum_live_proxy_u_fraction=.2),
          source_candidates=available,maximum_candidates=len(inputs),maximum_attempts_per_source=2,maximum_excerpt_characters=900,
          candidate_status="pending",candidate_admission="two_source_checks_before_test_answers",
          refusal_rule=parent["refusal_rule"],method_frozen_before_candidate_generation=True,
          question_bank_is_new=True,papers_previously_in_corpus_and_annotation_banks=True,
          interpretation="prospective_new_questions_current_method_development_disjoint_sources_not_wholly_unseen_corpus_or_independent_human_truth"))
    print("Prospective frozen:",len(available),"sources;",len(inputs),"maximum new candidates",flush=True)


def verify(output):
    m=experiment.verify(output);verify_hashes(m["initial_hashes"]);return m


def candidate(output,bid,row):
    path=output/"candidates"/(bid+".json")
    if path.exists():return load(path)["qualified"]
    if len(row["evidence"])<2:
        dump(path,dict(qualified=False,candidate=None,errors=["insufficient_source_cards"],checks=[],raw=None,locator=None));return False
    payload=dict(paper_sources=[row["source"]],old_questions=row["old_questions"],mineru_evidence=row["evidence"])
    raw=auto.request(output,"candidate_"+bid,CANDIDATE_PROMPT,payload)
    proposed=raw.get("candidate");locator=None;checks=[];validated=dict(valid=False,candidate=None,errors=["no_valid_scalar_candidate"])
    if isinstance(proposed,dict) and isinstance(proposed.get("question"),str):
        slots=scalar_slots(proposed)
        if 2<=len(slots)<=3 and all(all(k in s for k in ("index","label","query_quote","type")) for s in slots):
            public=dict(paper_sources=[row["source"]],original_question=proposed["question"],requested_slots=public_slots(dict(slots=slots)))
            locator=auto.request(output,"scope_"+bid,SCOPE_PROMPT,public)
            if isinstance(locator.get("scope"),str) and 0<len(locator["scope"])<=300:
                constructed=make_question([row["source"]],locator["scope"],slots)
                validated=validate_candidate(dict(candidate=constructed),row["evidence"],load(output/"old_questions.json"))
                if validated["valid"]:
                    payload=dict(question=constructed["question"],slots=validated["candidate"]["slots"],mineru_evidence=row["evidence"])
                    checks=[auto.request(output,f"candidate_check_{bid}_r{n}",CANDIDATE_CHECK+
                        ("\n按程序列出的字段逐项检查对象与口径。" if n==1 else "\n检查字段序号、原表顺序、唯一值、单位和简称对应。"),payload,n) for n in (1,2)]
    qualified=validated["valid"] and len(checks)==2 and all(checks_pass(c,len(validated["candidate"]["slots"]),
                ("explicit_required","expected_supported","unique","aliases_valid"),("question_clear","scope_complete")) for c in checks)
    dump(path,dict(**validated,qualified=qualified,raw=raw,locator=locator,checks=checks,finished_at=utc()))
    return qualified


def candidates(output):
    m=verify(output)
    batch(list(load(output/"candidate_inputs.json").items()),lambda item:candidate(output,*item),m["config"]["workers"],"PROSPECTIVE_CANDIDATES")


def freeze_questions(output):
    verify(output)
    if (output/"questions_lock.json").exists():verify_hashes(load(output/"questions_lock.json")["hashes"]);return
    inputs=load(output/"candidate_inputs.json")
    questions={};refs={};keys={};screen=[]
    seen={(row["source"],question_signature(q)) for row in inputs.values() for q in row["old_questions"]}
    for bid,row in inputs.items():
        r=load(output/"candidates"/(bid+".json"));c=r["candidate"]
        signature=(row["source"],question_signature(c["question"])) if c else None
        duplicate=bool(signature and signature in seen);accepted=r["qualified"] and not duplicate
        screen.append(dict(input_id=bid,source=row["source"],status="pending",source_checks_pass=int(r["qualified"]),
                 exact_request_duplicate=int(duplicate),experimental_admission=int(accepted),errors="|".join(r["errors"])))
        if accepted:
            qid=f"P{len(questions)+1:04d}";seen.add(signature)
            questions[qid]=dict(question=c["question"],requested_slots=public_slots(c),sources=[row["source"]])
            refs[qid]=dict(basis_complete=True,slots=c["slots"],facts=plan_facts(c["slots"]))
            keys[qid]=dict(input_id=bid,source=row["source"],candidate_status="pending")
    for name,value in (("public_questions.json",questions),("private_references.json",refs),("private_question_key.json",keys)):
        dump(output/name,value)
    write_csv(output/"candidate_screening.csv",screen)
    paths=[output/n for n in ("public_questions.json","private_references.json","private_question_key.json")]+sorted((output/"candidates").glob("*.json"))
    dump(output/"questions_lock.json",dict(locked_at=utc(),hashes={str(p.resolve()):digest(p) for p in paths},questions=len(questions),
            sources=len({k["source"] for k in keys.values()}),no_test_response_selection=True))
    print("Frozen prospective questions:",len(questions),"from",len({k["source"] for k in keys.values()}),"sources",flush=True)


def retrieval(output):
    from sklearn.feature_extraction.text import TfidfVectorizer
    m=verify(output);verify_hashes(load(output/"questions_lock.json")["hashes"])
    if (output/"retrieval_lock.json").exists():verify_hashes(load(output/"retrieval_lock.json")["hashes"]);return
    verify_hashes(m["model_hashes"])
    docs={s:blocks_to_text(load(Path(r["mineru_path"]))) for s,r in load(output/"source_manifest.json").items()}
    chunks=[c for s,text in sorted(docs.items()) for c in field_chunks(text,s)]
    vectorizer=TfidfVectorizer(analyzer="char",ngram_range=(2,3),max_features=60000,lowercase=True,dtype=np.float32)
    matrix=vectorizer.fit_transform([surface_map(c["text"])[0] for c in chunks])
    model=LocalReranker(m["config"]["reranker_path"]);rows={}
    for qid,q in load(output/"public_questions.json").items():
        found,trace=retrieve_fields(q,chunks,vectorizer,matrix,model);repaired=repair_selected(found,docs)
        rows[qid]=dict(**q,pre_repair_retrieved=found,retrieved=repaired,evidence=evidence_projection(repaired),
                       context=source_context(repaired),query_trace=trace)
        print("PROSPECTIVE_RETRIEVE",qid,len(repaired),flush=True)
    del model
    dump(output/"public_retrieval.json",rows)
    dump(output/"retrieval_lock.json",dict(locked_at=utc(),indexed_documents=len(docs),indexed_chunks=len(chunks),
         hashes={str((output/"public_retrieval.json").resolve()):digest(output/"public_retrieval.json")},
         no_private_reference_in_ranker=True,maximum_raw_characters=6000))


def freeze_cases(output):
    import random
    m=verify(output)
    for name in ("questions_lock.json","retrieval_lock.json","proxy_lock.json"):verify_hashes(load(output/name)["hashes"])
    if (output/"cases_lock.json").exists():verify_hashes(load(output/"cases_lock.json")["hashes"]);return
    qs=load(output/"public_questions.json");refs=load(output/"private_references.json");retrieved=load(output/"public_retrieval.json")
    pending=[]
    for qid,q in qs.items():
        pending.append((qid,"live_normal",None))
        for condition,row in controls(dict(slots=refs[qid]["slots"])).items():pending.append((qid,"control_"+condition,row))
    random.Random(m["config"]["seed"]).shuffle(pending);cases={};keys={};known={}
    for n,(qid,condition,row) in enumerate(pending,1):
        aid=f"N{n:04d}";context=retrieved[qid]["context"]
        cases[aid]=dict(question=qs[qid]["question"],context=context,context_sha256=sha(context))
        keys[aid]=dict(question_key=qid,condition=condition)
        if row:known[aid]=row
    for name,value in (("public_cases.json",cases),("private_case_key.json",keys),("private_control_answers.json",known)):
        dump(output/name,value)
    paths=[output/n for n in ("public_cases.json","private_case_key.json","private_control_answers.json","questions_lock.json",
          "proxy_lock.json","retrieval_lock.json","public_questions.json","private_references.json")]
    dump(output/"cases_lock.json",dict(locked_at=utc(),hashes={str(p.resolve()):digest(p) for p in paths},
          new_generations=len(qs),cases=len(cases),no_response_based_selection=True))
    print("Frozen all prospective cases:",len(cases),flush=True)


def analyze(output):
    m=experiment.verify_cases(output);qs=load(output/"public_questions.json");qkeys=load(output/"private_question_key.json")
    keys=load(output/"private_case_key.json");rows=[];basis=[]
    for aid,k in keys.items():
        s=load(output/"scores"/(aid+".json"));signal=s["proxy_comparison"]["signal"]
        rows.append(dict(aid=aid,**k,source=qkeys[k["question_key"]]["source"],quality=s["quality"],proxy_status=signal["status"],
               proxy_score=signal["score"],rule_refusal=s["field_refusal"]["refusal"],legacy_refusal=s["legacy_refusal"]))
    conditions=("controls","control_complete","control_omitted","control_corrupted","live_normal")
    table=[dict(condition=c,**experiment.metrics([r for r in rows if
           (c=="controls" and r["condition"].startswith("control_")) or r["condition"]==c])) for c in conditions]
    by={r["condition"]:r for r in table};g=m["gate"];vg=m["validation_gate"];sources={k["source"] for k in qkeys.values()}
    n=len(qs);control=by["controls"];live=by["live_normal"]
    frac=lambda a,b:a/b if b else 1
    checks=dict(enough_questions=n>=vg["minimum_questions"],enough_sources=len(sources)>=vg["minimum_sources"],
          controls_available=frac(control["proxy_u"],control["n"])<=g["maximum_control_proxy_u_fraction"],
          complete_false_flags=by["control_complete"]["false_flags"]<=g["maximum_complete_false_flags"],
          omitted_detection=frac(by["control_omitted"]["flagged_incomplete"],n)>=g["minimum_omitted_flag_fraction"] if n else False,
          corrupted_detection=frac(by["control_corrupted"]["flagged_incomplete"],n)>=g["minimum_corrupted_flag_fraction"] if n else False,
          live_complete=frac(live["complete"],n)>=vg["minimum_live_complete_fraction"] if n else False,
          live_quality_available=frac(live["quality_u"],n)<=vg["maximum_live_quality_u_fraction"],
          live_proxy_available=frac(live["proxy_u"],n)<=vg["maximum_live_proxy_u_fraction"])
    for qid,q in qs.items():
        p=load(output/"proxy_plans"/(qid+".json"));basis.append(dict(question_key=qid,source=qkeys[qid]["source"],
          valid=int(p["basis_complete"]),strict_valid=int(p["strict_basis_complete"]),source_mappings=len(p["source_mappings"])))
    grouped=[]
    for source in sorted(sources):
        r=[x for x in rows if x["source"]==source and x["condition"]=="live_normal"]
        grouped.append(dict(source=source,**experiment.metrics(r)))
    api=[load(p) for p in (output/"api_records").glob("*.json")];gens=[load(p) for p in (output/"generations").glob("*.json")]
    value=dict(maximum_candidates=m["maximum_candidates"],qualified_questions=n,qualified_sources=len(sources),
          fresh_normal_answers=len(gens),synthetic_controls=control["n"],metrics=table,proxy_bases=sum(r["valid"] for r in basis),
          gate=dict(status="passed" if all(checks.values()) else "not_passed",checks=checks,scope="prospective_new_question_objective_scoring_and_normal_generation"),
          api_responses=len(api)+len(gens),failed_api_records=sum(not r.get("ok") for r in api),
          total_tokens=sum(r.get("usage",{}).get("total_tokens",0) for r in api+gens),
          actual_models=dict(Counter([r.get("model_returned","error") for r in api]+[g["actual_model"] for g in gens])),
          generation_finishes=dict(Counter(g["finish_reason"] for g in gens)),
          interpretation=m["interpretation"],questions_per_source_maximum=2,source_grouped_reporting=True)
    for name,r in (("answer_scores.csv",rows),("signal_metrics.csv",table),("proxy_basis_availability.csv",basis),("live_source_groups.csv",grouped)):
        write_csv(output/name,r)
    dump(output/"analysis_summary.json",value);print(json.dumps(value,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action",choices=["prepare","candidates","freeze_questions","retrieve","plans","freeze_cases","generate","score","analyze"])
    for name in ("previous","original","objective","v4","output"):ap.add_argument("--"+name,type=Path,required=name=="output")
    a=ap.parse_args()
    if a.action=="prepare":prepare(a.previous,a.original,a.objective,a.v4,a.output)
    else:{"candidates":candidates,"freeze_questions":freeze_questions,"retrieve":retrieval,"plans":experiment.plans,
          "freeze_cases":freeze_cases,"generate":experiment.generate,"score":experiment.score,"analyze":analyze}[a.action](a.output)
