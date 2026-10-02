"""Audit V6.1 unchanged ranking, exact boundary replay and prospective gates."""
import argparse
from pathlib import Path

from audit_rag_field_retrieval_trial import audit as audit_common
from rag_fresh_change_experiment import load,dump,utc
from rag_prospective_coverage_validation import verify_hashes
from rag_core.mineru_loader import blocks_to_text
from rag_sentence_boundary_repair import repair_selected


def audit(output):
    extension=load(output/"boundary_validation_lock.json")
    verify_hashes(extension["hashes"])
    audit_common(output)
    m=load(output/"protocol_lock.json");old=load(output/"baseline_retrieval.json")
    new=load(output/"public_retrieval.json")
    docs={source:blocks_to_text(load(Path(r["mineru_path"]))) for source,r in load(output/"source_manifest.json").items()}
    changes=0;unresolved=0
    for qid,row in old.items():
        expected=repair_selected(row["retrieved"],docs)
        if new[qid]["retrieved"]!=expected or new[qid]["query_trace"]!=row["query_trace"]:
            raise ValueError("ranking changed or boundary replay mismatch")
        for old_span,new_span in zip(row["retrieved"],expected):
            if old_span["source"]!=new_span["source"] or old_span["rerank"]!=new_span["rerank"]:
                raise ValueError("boundary repair selected or reranked a new passage")
            changes+=new_span["boundary_repair"]["status"]=="repaired"
            unresolved+=new_span["boundary_repair"]["status"].startswith("unresolved")
    success=load(output/"boundary_success_lock.json");verify_hashes(success["hashes"])
    starts=[load(p)["started_at"] for p in (output/"generations").glob("*.json")]
    if max(success["locked_at"],extension["locked_at"])>=min(starts):raise ValueError("repair success gate after test answers")
    result=load(output/"integrity_audit.json")
    result.update(boundary_config=m["boundary_config"],frozen_ranking_replayed=True,source_slice_boundary_repair_replayed=True,
                  repaired_passages=changes,unresolved_passage_tails=unresolved,repair_success_gate_before_answers=True)
    dump(output/"integrity_audit.json",result)
    print("Boundary audit passed:",changes,"repaired passages;",unresolved,"unresolved tails retained")


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--output",type=Path,required=True)
    audit(ap.parse_args().output)
