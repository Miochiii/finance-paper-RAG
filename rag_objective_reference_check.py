"""Isolate scoring reliability given source-verified facts, before any retuning."""
import argparse
from collections import Counter
from pathlib import Path

from rag_fresh_change_experiment import load,dump,digest,utc,write_csv
from rag_objective_coverage_pilot import verify_cases,compare,point_quality
from rag_prospective_coverage_validation import verify_hashes
from rag_subtle_fault_pilot import batch


def prepare(output):
    verify_cases(output)
    if list((output/"generations").glob("*.json")) or list((output/"scores").glob("*.json")):
        raise ValueError("reference scoring diagnostic must be prespecified before live answers")
    paths=[Path(__file__),output/"protocol_lock.json",output/"questions_lock.json",output/"proxy_lock.json",output/"answer_cases_lock.json"]
    hashes={str(p.resolve()):digest(p) for p in paths};path=output/"reference_control_lock.json"
    if path.exists():
        if load(path)["hashes"]!=hashes:raise ValueError("reference diagnostic lock changed")
        return
    dump(path,dict(locked_at=utc(),hashes=hashes,minimum_controls=30,minimum_exact_quality_fraction=.95,maximum_u_fraction=.1,
                  purpose="isolate_scorer_with_source_verified_facts_not_annotation_free_proxy",no_answers_seen=True,
                  comparison_prompts_and_validation_functions_unchanged=True))


def evaluate(output):
    m,cases=verify_cases(output);verify_hashes(load(output/"reference_control_lock.json")["hashes"])
    refs=load(output/"private_references.json");keys=load(output/"private_case_key.json");known=load(output/"private_control_answers.json")
    def one(item):
        aid,row=item;qid=keys[aid]["question_key"]
        result=compare(output,"reference_control_"+aid,cases[aid]["question"],row["answer"],refs[qid])
        quality=point_quality(result)
        dump(output/"reference_control_scores"/(aid+".json"),dict(result=result,predicted_quality=quality,known_quality=row["quality"]))
        return dict(expected=row["quality"],predicted=quality)
    batch(list(known.items()),one,m["config"]["workers"],"REFERENCE_CONTROL_SCORE")


def analyze(output):
    verify_cases(output);lock=load(output/"reference_control_lock.json");verify_hashes(lock["hashes"])
    keys=load(output/"private_case_key.json");known=load(output/"private_control_answers.json");rows=[]
    for aid,row in known.items():
        r=load(output/"reference_control_scores"/(aid+".json"))
        rows.append(dict(aid=aid,**keys[aid],known_quality=row["quality"],predicted_quality=r["predicted_quality"],exact=int(row["quality"]==r["predicted_quality"])))
    correct=sum(r["exact"] for r in rows);unresolved=sum(r["predicted_quality"]=="U" for r in rows)
    checks=dict(enough_controls=len(rows)>=lock["minimum_controls"],exact_quality=bool(rows) and correct/len(rows)>=lock["minimum_exact_quality_fraction"],
                available=bool(rows) and unresolved/len(rows)<=lock["maximum_u_fraction"])
    summary=dict(n=len(rows),exact=correct,errors=len(rows)-correct-unresolved,u=unresolved,
                 predicted=dict(Counter(r["predicted_quality"] for r in rows)),
                 by_condition={c:dict(n=sum(r["condition"]==c for r in rows),exact=sum(r["condition"]==c and r["exact"] for r in rows))
                               for c in ("control_complete","control_omitted","control_corrupted")},
                 gate=dict(status="passed" if all(checks.values()) else "not_passed",checks=checks),
                 interpretation="scoring_given_verified_facts_only_not_retrieval_or_live_fault_detection")
    write_csv(output/"reference_control_quality.csv",rows);dump(output/"reference_control_summary.json",summary)
    print(summary,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("action",choices=["prepare","evaluate","analyze"])
    ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
    {"prepare":prepare,"evaluate":evaluate,"analyze":analyze}[a.action](a.output)
