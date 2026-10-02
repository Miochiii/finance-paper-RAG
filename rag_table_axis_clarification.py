"""Focused development pair: clarify a transposed sample table, same evidence.

V8.2 results stay unchanged. Derive the explicit target from the source matrix
before two fresh generations. This is one known-case development comparison.
"""
import argparse
import copy
from pathlib import Path

import rag_field_retrieval_trial as experiment
import rag_guarded_new_question_validation as base
from rag_fresh_change_experiment import load,dump,digest,sha,utc
from rag_objective_scope_revision import make_question
from rag_prospective_coverage_validation import verify_hashes
from rag_percent_fact_guard import fact_projection_issues


def table_cell(text,row_label,data_column):
    rows=[]
    for line in text.splitlines():
        if not line.strip().startswith("|"):continue
        cells=[c.strip() for c in line.strip().split("|")[1:-1]]
        if cells and cells[0]==row_label:rows.append((line,cells))
    if len(rows)!=1 or type(data_column)is not int or not 1<=data_column<len(rows[0][1]):
        raise ValueError("table row or data-column target not unique")
    return dict(value=rows[0][1][data_column],quote=rows[0][0],row_label=row_label,data_column=data_column)


def prepare(parent,output):
    if (output/"protocol_lock.json").exists():experiment.verify_cases(output);return
    output.mkdir(parents=True,exist_ok=True);qid="P0002"
    q=load(parent/"public_questions.json")[qid];ref=load(parent/"private_references.json")[qid]
    key=load(parent/"private_question_key.json")[qid];inputs=load(parent/"candidate_inputs.json")[key["input_id"]]
    cards=[c for c in inputs["evidence"] if c["eid"] in {s["eid"] for s in ref["slots"]}]
    if len(cards)!=1:raise ValueError("expected one transposed source table")
    card=cards[0];cells=[table_cell(card["text"],label,3) for label in ("Score","评级")]
    if [r["value"] for r in cells]!=[s["expected"] for s in ref["slots"]]:raise ValueError("reference differs from explicit matrix cells")
    page=card["page_idx"]+1
    scope=f"MinerU第{page}页的续表5-6；在Score行和评级行中从左到右数第3个数据列（不含最左侧行标题）对应的样本"
    fixed=make_question(q["sources"],scope,ref["slots"])
    clarified=dict(question=fixed["question"],requested_slots=q["requested_slots"],sources=q["sources"])
    qs=dict(original=q,clarified=clarified);retrieval=load(parent/"public_retrieval.json")[qid]
    refs={k:copy.deepcopy(ref) for k in qs};cases={};keys={};retrieved={}
    for name,question in qs.items():
        retrieved[name]=dict(retrieval,**question)
        cases[name]=dict(question=question["question"],context=retrieval["context"],context_sha256=sha(retrieval["context"]))
        keys[name]=dict(question_key=name,condition="live_normal")
    for name,value in (("public_questions.json",qs),("private_references.json",refs),("public_retrieval.json",retrieved),
        ("public_cases.json",cases),("private_case_key.json",keys),("private_control_answers.json",{}),
        ("private_matrix_target.json",dict(source=card["source"],block_index=card["mineru_block_index"],page_idx=card["page_idx"],cells=cells))):dump(output/name,value)
    files=[parent/n for n in ("protocol_lock.json","public_questions.json","private_references.json","candidate_inputs.json","public_retrieval.json")]
    files.extend((Path(__file__),Path(__file__).parent/"tests/test_rag_table_axis_clarification.py"))
    files.extend(output/n for n in ("public_questions.json","private_references.json","public_cases.json","private_case_key.json","private_control_answers.json","private_matrix_target.json"))
    m=load(parent/"protocol_lock.json")
    dump(output/"protocol_lock.json",dict(m,version="table_axis_scope_development_pair",locked_at=utc(),previous=str(parent.resolve()),
        input_hashes={str(p.resolve()):digest(p) for p in files},scope="one_seen_case_two_fresh_answers_same_evidence",
        interpretation="development_not_new_question_or_unseen_source_validation",original_quality_condition="third_data_column_interpretation_not_unambiguous_truth"))
    for name in ("questions_lock.json","retrieval_lock.json"):
        file=output/("public_questions.json" if name.startswith("questions") else "public_retrieval.json")
        dump(output/name,dict(locked_at=utc(),hashes={str(file.resolve()):digest(file)}))
    ready=load(parent/"public_questions.json");ready[qid]=clarified
    dump(output/"next_pair_public_questions.json",ready)
    print("Frozen one-case scope clarification; explicit source cells unchanged; two fresh generations planned")


def freeze(output):
    experiment.verify(output)
    for name in ("questions_lock.json","retrieval_lock.json","proxy_lock.json"):verify_hashes(load(output/name)["hashes"])
    refs=load(output/"private_references.json");plans={p.stem:load(p) for p in (output/"proxy_plans").glob("*.json")}
    issues=fact_projection_issues(refs,plans)
    if issues:raise ValueError("preflight failed; generation prohibited")
    if (output/"cases_lock.json").exists():verify_hashes(load(output/"cases_lock.json")["hashes"]);return
    if list((output/"generations").glob("*.json")):raise ValueError("case lock after answer generation")
    files=[output/n for n in ("public_cases.json","private_case_key.json","private_control_answers.json","private_references.json","proxy_lock.json")]
    files.extend(sorted((output/"proxy_plans").glob("*.json")))
    dump(output/"cases_lock.json",dict(locked_at=utc(),hashes={str(p.resolve()):digest(p) for p in files},preflight="passed",new_generations=2))


def analyze(output):
    m=experiment.verify_cases(output);rows=[]
    for name in ("original","clarified"):
        gen=load(output/"generations"/(name+".json"));score=load(output/"scores"/(name+".json"));plan=load(output/"proxy_plans"/(name+".json"))
        rows.append(dict(condition=name,quality=score["quality"],proxy=score["proxy_comparison"]["signal"]["status"],
            source_basis=plan["basis_complete"],actual_model=gen["actual_model"],finish=gen["finish_reason"]))
    if load(output/"cases_lock.json")["locked_at"]>=min(load(p)["started_at"] for p in (output/"generations").glob("*.json")):raise ValueError("case lock followed test answers")
    dump(output/"axis_pair_summary.json",dict(checked_at=utc(),scope=m["scope"],rows=rows,
        raw_matrix_targets_programmatically_checked=True,source_text_not_pdf_truth=True,
        primary_v8_2_results_unchanged=True,original_quality_is_conditional_on_third_column=True,
        no_claim_of_general_improvement=True,no_change_to_retrieval_context=True,preflight="passed"))
    print(rows)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("action",choices=["prepare","plans","freeze","generate","score","analyze"])
    parser.add_argument("--parent",type=Path);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    if args.action=="prepare":prepare(args.parent,args.output)
    else:
        experiment.verify(args.output)
        {"plans":base.plans,"freeze":freeze,"generate":experiment.generate,"score":experiment.score,"analyze":analyze}[args.action](args.output)
