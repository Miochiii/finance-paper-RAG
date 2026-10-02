"""V8.2 pre-answer revision: old questions only in local duplicate checks.

V8.1 admitted 5/24 and generated no test answers. Preserve it; reconstruct every
candidate on the same prepared cards, with no old questions in creation requests.
"""
import argparse
from pathlib import Path

import rag_guarded_percent_validation as percent
import rag_guarded_new_question_validation as base
import rag_prospective_objective_validation as validation
import rag_field_retrieval_trial as experiment
from rag_fresh_change_experiment import load,dump,digest,utc
from rag_prospective_coverage_validation import verify_hashes


NEW_CANDIDATE_PROMPT=base.CANDIDATE_V8.replace("旧问题仅用于避重复。","只依据当前给定原文创建问题。")+"""\n输入不包含旧题或已有作答要求。直接为所给新原文的任意明确表/对象构造一道题，不能假设必须问其他未提供的表或论文核心结论。
只依据当前证据中实际存在的行、列和字段确定问题范围。题库去重由程序完成，不需要猜测哪些主题已经被问过。
"""
original_evaluate_candidate=base.evaluate_candidate


def evaluate_candidate(output,bid,row,request):
    def create_without_old_questions(name,prompt,payload,n=0):
        if name=="candidate_"+bid:
            payload={k:payload[k] for k in ("paper_sources","mineru_evidence")}
            prompt=NEW_CANDIDATE_PROMPT
        return request(name,prompt,payload,n)
    return original_evaluate_candidate(output,bid,row,create_without_old_questions)


base.evaluate_candidate=evaluate_candidate
unused_cards=base.unused_cards
CANDIDATE_V8=NEW_CANDIDATE_PROMPT


def prepare(previous,output):
    percent.verify(previous)
    if list((previous/"generations").glob("*.json")):raise ValueError("context revision must precede any test answer")
    if (output/"protocol_lock.json").exists():verify(output);return
    output.mkdir(parents=True,exist_ok=True)
    parent=load(previous/"protocol_lock.json")
    names=("source_manifest.json","candidate_inputs.json","candidate_source_cards.json","historical_exposure_intervals.json",
           "old_questions.json","historical_questions_by_source.json","source_exposure.csv")
    for name in names:
        (output/name).write_bytes((previous/name).read_bytes())
    project=Path(__file__).parent
    files=[Path(__file__),project/"audit_rag_candidate_context_validation.py",project/"report_rag_candidate_context_validation.py",
        project/"tests/test_rag_candidate_context_revision.py",previous/"protocol_lock.json",previous/"percent_extension_lock.json",
        previous/"pre_answer_bank_disposition.json"]+sorted((previous/"candidates").glob("*.json"))
    hashes={str(p.resolve()):digest(p) for p in files}
    m=dict(parent,version="guarded_new_questions_v8_2_context_revision",locked_at=utc(),previous=str(previous.resolve()),
        input_hashes=dict(parent["input_hashes"],**hashes),initial_hashes={str((output/n).resolve()):digest(output/n) for n in names},
        interpretation="prospective_new_test_questions_existing_sources_previously_prepared_cards_same_vendor_automatic_truth",
        old_question_generation_context_removed=True,programmatic_deduplication_retained=True,
        source_cards_seen_in_preparation_no_test_answers=True,all24_attempts_recreated_before_test_answers=True)
    dump(output/"protocol_lock.json",m)
    (output/"percent_extension_lock.json").write_bytes((previous/"percent_extension_lock.json").read_bytes())
    dump(output/"candidate_context_revision_lock.json",dict(locked_at=utc(),hashes=hashes,
        original_preparation=str(previous.resolve()),original_test_answers=0,
        revision="creation_payload_no_old_questions_local_dedup_unchanged",all_attempts_uniformly_recreated=True))
    print("V8.2 context rule frozen; same prepared cards; 24 new attempts; prior bank has 0 test answers",flush=True)


def verify(output):
    percent.verify(output);verify_hashes(load(output/"candidate_context_revision_lock.json")["hashes"])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["prepare","candidates","freeze_questions","retrieve","plans","freeze_cases","preflight","generate","score","analyze"])
    parser.add_argument("--previous",type=Path);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.action=="prepare":prepare(args.previous,args.output)
    else:
        verify(args.output)
        {"candidates":base.candidates,"freeze_questions":base.freeze_questions,"retrieve":validation.retrieval,
         "plans":base.plans,"freeze_cases":base.freeze_cases,"preflight":percent.preflight,
         "generate":percent.generate,"score":experiment.score,"analyze":base.analyze}[args.action](args.output)
