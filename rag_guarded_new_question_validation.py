"""V8: frozen admission guard and numeric-unit rules, new questions/old corpus.

Historical source spans are excluded before candidate sampling. Blind extraction
from source cards precedes reference-aware source checks and any test answer.
All 24 attempts and all unresolved test cases remain in the experiment record.
"""
import argparse
import copy
import csv
import json
from collections import Counter
from pathlib import Path

import rag_auto_evaluation as auto
import rag_field_retrieval_trial as experiment
import rag_prospective_objective_validation as previous_validation
from rag_fresh_change_experiment import load,dump,digest,sha,utc,write_csv
from rag_prospective_coverage_validation import verify_hashes
from rag_exact_source_cards import exact_cards
from rag_core.mineru_loader import blocks_to_text
from rag_subtle_fault_pilot import batch,normalized_map
from rag_objective_slots import (CANDIDATE_PROMPT,CANDIDATE_CHECK,EXTRACT_SLOTS,CHECK_SLOTS,
    public_slots,validate_candidate,checks_pass)
from rag_objective_scope_revision import scalar_slots,SCOPE_PROMPT,make_question
from rag_objective_admission_guard import blind_source_payload,source_reference_consistency
from rag_numeric_unit_mapping import numeric_unit_proxy


CANDIDATE_V8 = CANDIDATE_PROMPT + """\n字段名须准确描述所问量：预测间隔/持股时间与价格不同，不得把日期、价格、标签编码、比例或样本数互换。
number的expected仅数字（可含百分号），单位单列；expected若已含%，unit留空，不能重复符号。
两个字段应来自原文明确的行/列/对象，不能用题干直接提供所问数值。原文不充分或含多个不清楚口径时返回null。
"""
CHECK_SUFFIX = {
    1:"\n核对字段与值的对应关系。",
    2:"\n核对实验口径和别名是否有来源支持。",
}
GOLD_SUFFIX = {
    1:"\n按程序列出的字段逐项检查对象与口径。",
    2:"\n检查字段序号、原表顺序、唯一值、单位和简称对应。",
}
HISTORY_NAMES = {
    "candidate_inputs.json","candidate_source_cards.json","public_retrieval.json",
    "baseline_retrieval.json","public_questions.json","blind_questions.json",
    "public_cases.json","private_references.json","_evidence_cache_live.json",
    "_evidence_cache.json",
}


def collect_exposures(value, sources, texts, questions, inherited=()):
    """Structured source attribution only; ignore answers and judgment prose."""
    if isinstance(value,list):
        for row in value:collect_exposures(row,sources,texts,questions,inherited)
        return
    if not isinstance(value,dict):return
    owners=set(inherited)
    for key in ("source","sources","actual_sources","paper_sources","gold_docs"):
        item=value.get(key)
        names=item.split("|") if isinstance(item,str) else item if isinstance(item,list) else []
        owners.update(n for n in names if isinstance(n,str) and n in sources)
    question=value.get("question")
    if isinstance(question,str):
        owners.update(s for s in sources if "《"+Path(s).stem+"》" in question)
        for source in owners:questions[source].add(question)
    for key in ("text","context","quote"):
        text=value.get(key)
        if isinstance(text,str) and len(auto.norm(text))>=40:
            for source in owners:texts[source].add(text)
    for key,child in value.items():
        if key in {"answer","system_answer","raw_response","point_results","reason"}:continue
        collect_exposures(child,sources,texts,questions,(key,) if key in sources else tuple(owners))


def exposure_intervals(text, exposures):
    normalized,positions=normalized_map(text);spans=set();unmatched=0
    for exposed in sorted(exposures):
        needle=auto.norm(exposed)
        cursor=0;found=False
        while True:
            start=normalized.find(needle,cursor)
            if start<0:break
            found=True;spans.add((positions[start],positions[start+len(needle)-1]+1))
            cursor=start+len(needle)
        if not found:unmatched+=1
    merged=[]
    for start,end in sorted(spans):
        if merged and start<=merged[-1][1]:merged[-1][1]=max(end,merged[-1][1])
        else:merged.append([start,end])
    return merged,unmatched


def unused_cards(text,source,blocks,intervals,count=4):
    cards=exact_cards(text,source,blocks,count=len(blocks))
    selected=[c for c in cards if not any(max(c["start"],a)<min(c["end"],b) for a,b in intervals)][:count]
    for n,card in enumerate(selected,1):card["eid"]=f"E{n}"
    return selected


def canonical_candidate(candidate):
    value=copy.deepcopy(candidate)
    for slot in value["slots"]:
        if slot.get("type")=="number" and isinstance(slot.get("expected"),str) and slot["expected"].endswith(("%","％")) and slot.get("unit") in ("%","％"):
            slot["unit"]=""
    return value


def prepare(previous,output):
    project=Path(__file__).parent
    parent=experiment.verify_cases(previous)
    review=load(previous/"validity_review/review_inputs_lock.json")
    verify_hashes(review["hashes"]);verify_hashes(review["code_hashes"])
    if (output/"protocol_lock.json").exists():verify(output);return
    output.mkdir(parents=True,exist_ok=True)
    manifest=load(previous/"source_manifest.json");sources=set(manifest)
    blocks={s:load(Path(r["mineru_path"])) for s,r in manifest.items()}
    docs={s:blocks_to_text(b) for s,b in blocks.items()}
    exposures={s:set() for s in sources};old={s:set() for s in sources}
    history=sorted(p for p in (project/"results").rglob("*.json")
          if p.name in HISTORY_NAMES and not p.is_relative_to(output.resolve())
          and not any(part in {"api_records","generations","scores","pdf_text_cache"} for part in p.parts))
    for path in history:collect_exposures(load(path),sources,exposures,old)
    annotations=sorted((project/"data/annotations").glob("*.csv"))
    for path in annotations:
        with path.open(encoding="utf-8-sig",newline="") as handle:
            for row in csv.DictReader(handle):
                for source in (row.get("gold_docs") or "").split("|"):
                    if source in sources and row.get("question"):old[source].add(row["question"])
    intervals={};inventory={};exposure_rows=[]
    for source in sorted(sources):
        intervals[source],unmatched=exposure_intervals(docs[source],exposures[source])
        inventory[source]=unused_cards(docs[source],source,blocks[source],intervals[source])
        exposure_rows.append(dict(source=source,previously_in_corpus=1,known_question_count=len(old[source]),
            history_excerpts=len(exposures[source]),matched_intervals=len(intervals[source]),
            unmatched_whole_excerpts=unmatched,unused_exact_cards=len(inventory[source])))
    # Deterministic source-only ordering; exactly two attempts for each of 12 sources.
    available=sorted((s for s in sources if len(inventory[s])>=4),key=lambda s:(sha("v8-20261002|"+s),s))
    chosen=available[:12]
    if len(chosen)<6:raise ValueError("not enough sources with unused exact cards")
    inputs={}
    for n,source in enumerate(chosen,1):
        for offset in (0,2):
            inputs[f"J{n:03d}_{offset//2+1}"]=dict(source=source,evidence=inventory[source][offset:offset+2],old_questions=sorted(old[source]))
    for name,value in (("source_manifest.json",manifest),("candidate_inputs.json",inputs),
         ("candidate_source_cards.json",{s:inventory[s] for s in chosen}),
         ("historical_exposure_intervals.json",intervals),("old_questions.json",sorted({q for group in old.values() for q in group})),
         ("historical_questions_by_source.json",{s:sorted(q) for s,q in old.items()})):
        dump(output/name,value)
    write_csv(output/"source_exposure.csv",exposure_rows)
    codes=[Path(p) for p in parent["input_hashes"] if Path(p).suffix==".py"]
    codes.extend(project/name for name in (
        "rag_guarded_new_question_validation.py","audit_rag_guarded_validation.py","report_rag_guarded_validation.py",
        "rag_prospective_objective_validation.py","rag_exact_source_cards.py","rag_numeric_unit_mapping.py",
        "rag_objective_admission_guard.py","tests/test_rag_guarded_new_question_validation.py"))
    paths=codes+history+annotations+[previous/"protocol_lock.json",previous/"validity_review/validity_summary.json"]
    copied=[output/name for name in ("source_manifest.json","candidate_inputs.json","candidate_source_cards.json",
        "historical_exposure_intervals.json","old_questions.json","historical_questions_by_source.json","source_exposure.csv")]
    dump(output/"protocol_lock.json",dict(version="guarded_new_questions_v8",locked_at=utc(),previous=str(previous.resolve()),
        input_hashes={str(p.resolve()):digest(p) for p in paths},initial_hashes={str(p.resolve()):digest(p) for p in copied},
        source_hashes=parent["source_hashes"],model_hashes=parent["model_hashes"],config=parent["config"],
        retrieval_config=parent["retrieval_config"],boundary_config=parent["boundary_config"],gate=parent["gate"],
        validation_gate=parent["validation_gate"],refusal_rule=parent["refusal_rule"],
        source_candidates=chosen,maximum_candidates=len(inputs),maximum_attempts_per_source=2,
        maximum_excerpt_characters=900,maximum_excerpt_cards_per_source=4,candidate_status="pending",
        candidate_admission="blind_source_extraction_two_checks_then_reference_consistency_then_two_gold_aware_checks",
        historical_registry_files=len(history),historical_registry_scope="known_structured_excerpts_not_all_project_or_vendor_exposure",
        interpretation="prospective_new_questions_unused_registered_source_spans_existing_sources_same_vendor_automatic_truth",
        no_answer_based_tuning=True,method_frozen_before_candidates=True,not_wholly_unseen_sources=True,
        question_bank_is_new=True,all_unresolved_retained=True))
    print("Frozen V8:",len(chosen),"sources;",len(inputs),"candidate attempts; history files",len(history),flush=True)


def verify(output):
    m=experiment.verify(output);verify_hashes(m["initial_hashes"]);return m


def evaluate_candidate(output,bid,row,request):
    raw=request("candidate_"+bid,CANDIDATE_V8,dict(paper_sources=[row["source"]],old_questions=row["old_questions"],mineru_evidence=row["evidence"]))
    locator=None;blind=None;blind_checks=[];checks=[];guard=None
    valid=dict(valid=False,candidate=None,errors=["no_valid_scalar_candidate"])
    proposed=raw.get("candidate")
    if isinstance(proposed,dict) and isinstance(proposed.get("question"),str):
        slots=scalar_slots(proposed)
        if 2<=len(slots)<=3 and all(all(k in s for k in ("index","label","query_quote","type")) for s in slots):
            locator=request("scope_"+bid,SCOPE_PROMPT,dict(paper_sources=[row["source"]],original_question=proposed["question"],requested_slots=public_slots(dict(slots=slots))))
            scope=locator.get("scope")
            if isinstance(scope,str) and 0<len(scope)<=300:
                built=canonical_candidate(make_question([row["source"]],scope,slots))
                valid=validate_candidate(dict(candidate=built),row["evidence"],load(output/"old_questions.json"))
                if valid["valid"]:
                    c=valid["candidate"];payload=blind_source_payload(c,row["evidence"])
                    blind=request("admission_extract_"+bid,EXTRACT_SLOTS,payload)
                    blind_checks=[request(f"admission_check_{bid}_r{n}",CHECK_SLOTS+CHECK_SUFFIX[n],
                          dict(**payload,temporary_slots=blind.get("slots",[])),n) for n in (1,2)]
                    guard=source_reference_consistency(c,blind,blind_checks,row["evidence"])
                    if guard["admit"]:
                        payload=dict(question=c["question"],slots=c["slots"],mineru_evidence=row["evidence"])
                        checks=[request(f"candidate_check_{bid}_r{n}",CANDIDATE_CHECK+GOLD_SUFFIX[n],payload,n) for n in (1,2)]
    qualified=bool(valid["valid"] and guard and guard["admit"] and len(checks)==2 and all(
        checks_pass(c,len(valid["candidate"]["slots"]),("explicit_required","expected_supported","unique","aliases_valid"),
        ("question_clear","scope_complete")) for c in checks))
    return dict(**valid,qualified=qualified,raw=raw,locator=locator,blind_extraction=blind,
                blind_checks=blind_checks,admission_guard=guard,checks=checks)


def candidates(output):
    m=verify(output)
    def one(item):
        bid,row=item;path=output/"candidates"/(bid+".json")
        if path.exists():return load(path)["qualified"]
        def request(name,prompt,payload,n=0):return auto.request(output,name,prompt,payload,n)
        result=evaluate_candidate(output,bid,row,request)
        dump(path,dict(**result,finished_at=utc()));return result["qualified"]
    batch(list(load(output/"candidate_inputs.json").items()),one,m["config"]["workers"],"GUARDED_CANDIDATES")


def freeze_questions(output):
    verify(output);previous_validation.freeze_questions(output)
    rows=[]
    inputs=load(output/"candidate_inputs.json")
    for bid,row in inputs.items():
        candidate=load(output/"candidates"/(bid+".json"));guard=candidate["admission_guard"]
        rows.append(dict(input_id=bid,source=row["source"],program_valid=int(candidate["valid"]),
            blind_consistency_pass=int(bool(guard and guard["admit"])),two_reference_checks_pass=int(candidate["qualified"]),
            reasons="|".join(candidate["errors"]+[r["reason"] for r in guard["reasons"]] if guard else candidate["errors"])))
    write_csv(output/"admission_diagnostic.csv",rows)
    qs=load(output/"public_questions.json");m=load(output/"protocol_lock.json")
    if len(qs)<m["validation_gate"]["minimum_questions"] or len({s for q in qs.values() for s in q["sources"]})<m["validation_gate"]["minimum_sources"]:
        raise ValueError("pre-answer bank gate failed; retain all attempts, do not generate test answers")


def plans(output):
    m=verify(output);verify_hashes(load(output/"questions_lock.json")["hashes"])
    verify_hashes(load(output/"retrieval_lock.json")["hashes"])
    if (output/"proxy_lock.json").exists():verify_hashes(load(output/"proxy_lock.json")["hashes"]);return
    qs=load(output/"public_questions.json");retrieved=load(output/"public_retrieval.json")
    def one(item):
        qid,q=item;evidence=retrieved[qid]["evidence"]
        payload=dict(question=q["question"],requested_slots=q["requested_slots"],retrieved_evidence=evidence)
        raw=auto.request(output,"proxy_extract_"+qid,EXTRACT_SLOTS,payload)
        checks=[auto.request(output,f"proxy_check_{qid}_r{n}",CHECK_SLOTS+CHECK_SUFFIX[n],dict(**payload,temporary_slots=raw.get("slots",[])),n) for n in (1,2)]
        plan=dict(question=q["question"],evidence=evidence,raw=raw,checks=checks,**numeric_unit_proxy(raw,checks,q,evidence))
        dump(output/"proxy_plans"/(qid+".json"),plan);return plan["basis_complete"]
    batch(list(qs.items()),one,m["config"]["workers"],"GUARDED_PLANS")
    dump(output/"proxy_lock.json",dict(locked_at=utc(),before_answers=True,
         hashes={str(p.resolve()):digest(p) for p in sorted((output/"proxy_plans").glob("*.json"))}))


def freeze_cases(output):
    verify(output);previous_validation.freeze_cases(output)
    paths=[Path(__file__).with_name(n) for n in ("audit_rag_guarded_validation.py","report_rag_guarded_validation.py")]
    paths.extend(output/n for n in ("protocol_lock.json","questions_lock.json","retrieval_lock.json","proxy_lock.json","cases_lock.json"))
    path=output/"validation_tools_lock.json"
    if path.exists():verify_hashes(load(path)["hashes"]);return
    dump(path,dict(locked_at=utc(),before_answers=True,hashes={str(p.resolve()):digest(p) for p in paths}))


def analyze(output):
    verify(output);previous_validation.analyze(output)
    summary=load(output/"analysis_summary.json");qs=load(output/"public_questions.json")
    inputs=load(output/"candidate_inputs.json")
    all_candidates=[load(output/"candidates"/(bid+".json")) for bid in inputs]
    keys=load(output/"private_question_key.json")
    summary.update(admission_attempts=len(inputs),blind_consistency_pass=sum(bool(c["admission_guard"] and c["admission_guard"]["admit"]) for c in all_candidates),
        program_valid_candidates=sum(c["valid"] for c in all_candidates),
        admission_rejected=len(inputs)-len(qs),admitted_references_all_blind_consistent=all(
            load(output/"candidates"/(k["input_id"]+".json"))["admission_guard"]["admit"] for k in keys.values()),
        reference_semantics_scope="automatic_source_consistency_not_independent_truth",
        source_exposure_scope="new_questions_unused_registered_spans_existing_papers",
        controls_scope="synthetic_answer_edits_not_live_evidence_faults")
    dump(output/"analysis_summary.json",summary)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["prepare","candidates","freeze_questions","retrieve","plans","freeze_cases","generate","score","analyze"])
    parser.add_argument("--previous",type=Path);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.action=="prepare":prepare(args.previous,args.output)
    else:
        verify(args.output)
        {"candidates":candidates,"freeze_questions":freeze_questions,"retrieve":previous_validation.retrieval,
         "plans":plans,"freeze_cases":freeze_cases,"generate":experiment.generate,"score":experiment.score,
         "analyze":analyze}[args.action](args.output)
