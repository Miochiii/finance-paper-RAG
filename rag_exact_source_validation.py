"""Prospective bank preparation corrected before any test answers were generated.

Frozen V6.1 retrieval/answer scoring remain unchanged. Preserve first bank; use
uniform table/text-only candidate evidence for all 8 eligible sources.
"""
import argparse
from pathlib import Path

import rag_prospective_objective_validation as validation
import rag_field_retrieval_trial as experiment
from rag_exact_source_cards import exact_cards
from rag_fresh_change_experiment import load,dump,digest,utc
from rag_prospective_coverage_validation import verify_hashes


def prepare(previous,original,objective,v4,legacy,output):
    if list((legacy/"generations").glob("*.json")):raise ValueError("source admission correction after test answers")
    def select(text,source,count=4):
        matches=list(Path("data/corpora/金融论文/mineru_out/batch").glob(Path(source).stem+"/vlm/*_content_list.json"))
        if len(matches)!=1:raise ValueError("source path ambiguous")
        return exact_cards(text,source,load(matches[0]),count)
    # Runtime injection is explicit and separately hash-locked, never edit frozen code.
    old=validation.choose_cards;validation.choose_cards=select
    try:validation.prepare(previous,original,objective,v4,output)
    finally:validation.choose_cards=old
    paths=[Path(__file__),Path(__file__).with_name("rag_exact_source_cards.py"),
           Path(__file__).parent/"tests/test_rag_exact_source_cards.py",legacy/"protocol_lock.json",legacy/"questions_lock.json"]
    paths+=sorted((legacy/"candidates").glob("*.json"))
    hashes={str(p.resolve()):digest(p) for p in paths};path=output/"source_integrity_lock.json"
    if path.exists():
        if load(path)["hashes"]!=hashes:raise ValueError("source integrity extension changed")
        return
    dump(path,dict(locked_at=utc(),hashes=hashes,legacy_preparation=str(legacy.resolve()),
       reason="table_marker_also_wraps_chart_reconstructions_four_admitted_questions_not_exact_table_references",
       legacy_test_answers=0,allowed_mineru_block_types=["table","text"],maximum_card_characters=900,
       approximation_markers_rejected=True,uniform_new_attempts_for_all8_sources=True,
       retrieval_and_scoring_method_unchanged=True,primary_validation_gate_unchanged=True))
    print("Frozen original-block source integrity correction; no test answers seen",flush=True)


def verify(output):
    validation.verify(output);verify_hashes(load(output/"source_integrity_lock.json")["hashes"])


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action",choices=["prepare","candidates","freeze_questions","retrieve","plans","freeze_cases","generate","score","analyze"])
    for name in ("previous","original","objective","v4","legacy","output"):ap.add_argument("--"+name,type=Path,required=name=="output")
    a=ap.parse_args()
    if a.action=="prepare":prepare(a.previous,a.original,a.objective,a.v4,a.legacy,a.output)
    else:
        verify(a.output)
        {"candidates":validation.candidates,"freeze_questions":validation.freeze_questions,"retrieve":validation.retrieval,
         "plans":experiment.plans,"freeze_cases":validation.freeze_cases,"generate":experiment.generate,
         "score":experiment.score,"analyze":validation.analyze}[a.action](a.output)
