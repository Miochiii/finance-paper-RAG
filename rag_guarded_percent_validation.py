"""V8.1: new questions/spans after V8 preflight failure; fail before generating.

The V8 records remain unchanged. This separately freezes exact percent-unit
projection and a mandatory successful preflight lock for new test generations.
"""
import argparse
from pathlib import Path

import rag_guarded_new_question_validation as base
import rag_prospective_objective_validation as validation
import rag_field_retrieval_trial as experiment
from rag_percent_fact_guard import percent_unit_proxy,percent_source_consistency,fact_projection_issues
from rag_fresh_change_experiment import load,dump,digest,utc
from rag_prospective_coverage_validation import verify_hashes

# Explicit runtime extension, separately locked; historical modules stay intact.
base.numeric_unit_proxy=percent_unit_proxy
base.source_reference_consistency=percent_source_consistency
evaluate_candidate=base.evaluate_candidate
unused_cards=base.unused_cards
CANDIDATE_V8=base.CANDIDATE_V8


def prepare(previous,legacy,output):
    base.prepare(previous,output)
    if not (legacy/"preflight_failure_disposition.json").exists():raise ValueError("missing preserved V8 disposition")
    files=[Path(__file__),Path(__file__).with_name("rag_percent_fact_guard.py"),
           Path(__file__).with_name("audit_rag_guarded_percent_validation.py"),
           Path(__file__).with_name("report_rag_guarded_percent_validation.py"),
           Path(__file__).parent/"tests/test_rag_percent_fact_guard.py",legacy/"preflight_failure_disposition.json",
           output/"protocol_lock.json"]
    path=output/"percent_extension_lock.json";hashes={str(p.resolve()):digest(p) for p in files}
    if path.exists():
        if load(path)["hashes"]!=hashes:raise ValueError("percent extension changed")
        return
    dump(path,dict(locked_at=utc(),version="guarded_new_questions_v8_1",hashes=hashes,
         previous_failed_workflow=str(legacy.resolve()),new_candidate_bank_and_registered_unused_spans=True,
         percent_value_magnitude_unchanged=True,mandatory_preflight=True,method_frozen_before_candidates=True))


def verify(output):
    base.verify(output);verify_hashes(load(output/"percent_extension_lock.json")["hashes"])


def preflight(output):
    verify(output)
    for name in ("questions_lock.json","retrieval_lock.json","proxy_lock.json","cases_lock.json","validation_tools_lock.json"):
        verify_hashes(load(output/name)["hashes"])
    path=output/"preflight_lock.json"
    if path.exists():verify_hashes(load(path)["hashes"]);return load(path)
    if list((output/"generations").glob("*.json")):raise ValueError("preflight must precede every test generation")
    references=load(output/"private_references.json")
    plans={p.stem:load(p) for p in (output/"proxy_plans").glob("*.json")}
    issues=fact_projection_issues(references,plans)
    if issues:
        dump(output/"preflight_failed.json",dict(checked_at=utc(),status="failed",issues=issues))
        raise ValueError("numeric fact preflight failed; no test generations allowed")
    files=[output/"private_references.json",output/"cases_lock.json",output/"percent_extension_lock.json"]
    files.extend(sorted((output/"proxy_plans").glob("*.json")))
    record=dict(locked_at=utc(),status="passed",before_answers=True,issues=[],
         questions=len(references),proxy_bases=sum(p["basis_complete"] for p in plans.values()),
         hashes={str(p.resolve()):digest(p) for p in files})
    dump(path,record);print("Mandatory preflight passed:",len(references),"questions",flush=True)
    return record


def generate(output):
    record=preflight(output)
    if record["status"]!="passed":raise ValueError("preflight not passed")
    experiment.generate(output)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["prepare","candidates","freeze_questions","retrieve","plans","freeze_cases","preflight","generate","score","analyze"])
    for name in ("previous","legacy","output"):parser.add_argument("--"+name,type=Path,required=name=="output")
    args=parser.parse_args()
    if args.action=="prepare":prepare(args.previous,args.legacy,args.output)
    else:
        verify(args.output)
        {"candidates":base.candidates,"freeze_questions":base.freeze_questions,"retrieve":validation.retrieval,
         "plans":base.plans,"freeze_cases":base.freeze_cases,"preflight":preflight,"generate":generate,
         "score":experiment.score,"analyze":base.analyze}[args.action](args.output)
