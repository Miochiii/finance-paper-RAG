"""V6.1: isolate sentence-tail repair with unchanged frozen V6 query/ranks.

All 11 seen-source questions retained. Same 3 passages/6000-character budget,
22 fresh paired normal answers, and 33 synthetic evaluator controls.
"""
import argparse
from pathlib import Path

import rag_field_retrieval_trial as parent_trial
from rag_fresh_change_experiment import load,dump,digest,utc,write_csv
from rag_prospective_coverage_validation import verify_hashes
from rag_core.mineru_loader import blocks_to_text
from rag_mineru_coverage_v4 import evidence_projection,source_context
from rag_sentence_boundary_repair import CONFIG,repair_selected
import rag_auto_evaluation as auto


def prepare(previous,output):
    m=parent_trial.verify_cases(previous)
    output.mkdir(parents=True,exist_ok=True)
    paths=[previous/n for n in ("protocol_lock.json","public_questions.json","private_references.json",
          "source_manifest.json","public_retrieval.json","analysis_summary.json","integrity_audit.json","private_case_key.json")]
    paths+=sorted((previous/"proxy_plans").glob("*.json"))+sorted((previous/"scores").glob("*.json"))
    paths+=[Path(p) for p in m["input_hashes"] if Path(p).suffix==".py"]
    paths+=[Path(__file__),Path(__file__).with_name("rag_sentence_boundary_repair.py"),
            Path(__file__).parent/"tests/test_rag_sentence_boundary_repair.py"]
    hashes={str(p.resolve()):digest(p) for p in paths}
    if (output/"protocol_lock.json").exists():
        if load(output/"protocol_lock.json")["input_hashes"]!=hashes:raise ValueError("boundary protocol changed")
        parent_trial.verify(output);return
    for name in ("public_questions.json","private_references.json","source_manifest.json"):
        dump(output/name,load(previous/name))
    dump(output/"baseline_retrieval.json",load(previous/"public_retrieval.json"))
    for path in sorted((previous/"proxy_plans").glob("*.json")):dump(output/"baseline_plans"/path.name,load(path))
    dump(output/"protocol_lock.json",dict(version="sentence_boundary_v6_1_development",locked_at=utc(),
         previous=str(previous.resolve()),input_hashes=hashes,source_hashes=m["source_hashes"],model_hashes=m["model_hashes"],
         config=m["config"],retrieval_config=m["retrieval_config"],boundary_config=CONFIG,gate=m["gate"],
         controls_scope=m["controls_scope"],interpretation="seen_question_development_boundary_only_ranks_fixed_not_new_source_or_live_fault_validation",
         fixed_question_bank=True,maximum_new_generations=22,all_unresolved_retained=True,
         no_answer_based_tuning=True,refusal_rule=m["refusal_rule"],ranking_unchanged=True))
    print("V6.1 frozen: same ranks and 6000-character budget; 22 fresh paired answers +33 controls",flush=True)


def retrieval(output):
    m=parent_trial.verify(output)
    if (output/"retrieval_lock.json").exists():verify_hashes(load(output/"retrieval_lock.json")["hashes"]);return
    docs={source:blocks_to_text(load(Path(row["mineru_path"])))
          for source,row in load(output/"source_manifest.json").items()}
    rows={};boundaries=[]
    for qid,row in load(output/"baseline_retrieval.json").items():
        found=repair_selected(row["retrieved"],docs)
        rows[qid]=dict(question=row["question"],requested_slots=row["requested_slots"],sources=row["sources"],
                       retrieved=found,evidence=evidence_projection(found),context=source_context(found),query_trace=row["query_trace"])
        for rank,c in enumerate(found,1):
            trace=c["boundary_repair"]
            boundaries.append(dict(question_key=qid,rank=rank,status=trace["status"],
                  added_tail_characters=trace["added_tail_characters"],removed_head_characters=trace["removed_head_characters"],
                  before_characters=trace["original_end"]-trace["original_start"],after_characters=c["end"]-c["start"]))
        print("BOUNDARY_REPAIR",qid,[c["boundary_repair"]["status"] for c in found],flush=True)
    dump(output/"public_retrieval.json",rows)
    write_csv(output/"boundary_repair_summary.csv",boundaries)
    dump(output/"retrieval_lock.json",dict(locked_at=utc(),
         hashes={str((output/"public_retrieval.json").resolve()):digest(output/"public_retrieval.json")},
         no_private_reference_in_ranker=True,maximum_raw_characters=6000,ranking_unchanged=True))
    # Offline diagnostics start after the new retrieval is locked.
    refs=load(output/"private_references.json");old=load(output/"baseline_retrieval.json");diagnostic=[]
    for qid,row in rows.items():
        counts=[sum(any(auto.norm(s["quote"]) in auto.norm(c["text"]) for c in bank[qid]["retrieved"])
                    for s in refs[qid]["slots"]) for bank in (old,rows)]
        diagnostic.append(dict(question_key=qid,slots=len(refs[qid]["slots"]),
             baseline_literal_quotes=counts[0],enhanced_literal_quotes=counts[1],
             baseline_characters=sum(len(c["text"]) for c in old[qid]["retrieved"]),
             enhanced_characters=sum(len(c["text"]) for c in row["retrieved"])))
    write_csv(output/"local_retrieval_diagnostic.csv",diagnostic)
    print("Locked boundary diagnostic:",diagnostic,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action",choices=["prepare","retrieve","plans","freeze_cases","generate","score","analyze"])
    ap.add_argument("--previous",type=Path);ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    if a.action=="prepare":prepare(a.previous,a.output)
    elif a.action=="retrieve":retrieval(a.output)
    else:getattr(parent_trial,a.action)(a.output)
