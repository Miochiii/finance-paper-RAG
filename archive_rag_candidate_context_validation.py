"""Archive V8.2 code/aggregates, retaining the V8 and V8.1 dispositions."""
import argparse
import shutil
from pathlib import Path

from rag_fresh_change_experiment import load,dump,digest,utc
from rag_prospective_coverage_validation import verify_hashes


def archive(project,output,backup):
    project,output,backup=[p.resolve() for p in (project,output,backup)]
    if not output.is_relative_to(project) or project==backup or backup.is_relative_to(project) or project.is_relative_to(backup):raise ValueError("project paths invalid")
    m=load(output/"protocol_lock.json");verify_hashes(m["input_hashes"])
    verify_hashes(load(output/"preflight_lock.json")["hashes"])
    code=["rag_guarded_new_question_validation.py","audit_rag_guarded_validation.py","report_rag_guarded_validation.py",
        "archive_rag_guarded_validation.py","rag_percent_fact_guard.py","rag_guarded_percent_validation.py",
        "audit_rag_guarded_percent_validation.py","report_rag_guarded_percent_validation.py",
        "rag_candidate_context_revision.py","audit_rag_candidate_context_validation.py","report_rag_candidate_context_validation.py",
        "archive_rag_candidate_context_validation.py","tests/test_rag_guarded_new_question_validation.py",
        "tests/test_rag_percent_fact_guard.py","tests/test_rag_candidate_context_revision.py",
        "rag_table_axis_clarification.py","tests/test_rag_table_axis_clarification.py"]
    aggregates=["analysis_summary.json","integrity_audit.json","targeted_tests.json","candidate_screening.csv",
        "admission_diagnostic.csv","source_exposure.csv","answer_scores.csv","signal_metrics.csv",
        "proxy_basis_availability.csv","live_source_groups.csv","workflow_summary.json"]
    aggregates.extend(["table_axis_clarification_20261002/axis_pair_summary.json",
                       "table_axis_clarification_20261002/integrity_audit.json"])
    destination=backup/"docs/experiments/guarded_new_questions_v8_2_20261002";files=[]
    def save(source,target,content=None):
        target=target.resolve()
        if not target.is_relative_to(backup):raise ValueError("backup destination escapes project")
        target.parent.mkdir(parents=True,exist_ok=True)
        if content is None:
            shutil.copy2(source,target)
            if digest(source)!=digest(target):raise ValueError("copy mismatch")
        else:target.write_text(content,encoding="utf-8")
        files.append(dict(source=str(source.resolve()),destination=str(target),sha256=digest(target),transformed=content is not None))
    for name in code:save(project/name,backup/name)
    for name in aggregates:save(output/name,destination/name)
    for ext in ("png","svg"):save(output/("guarded_validation_results."+ext),backup/"docs/assets"/("rag_guarded_v8_2_20261002."+ext))
    for source,name in ((project/"log/新题来源盲提取验证V8_2协议_20261002.md","rag_guarded_v8_2_protocol_20261002.md"),
        (output/"guarded_validation_report.md","rag_guarded_v8_2_report_20261002.md")):
        content=source.read_text(encoding="utf-8").replace((output/"guarded_validation_results.png").as_posix(),"assets/rag_guarded_v8_2_20261002.png")
        save(source,backup/"docs"/name,content)
    summary=load(output/"analysis_summary.json")
    record=dict(saved_at=utc(),copied_files=len(files),files=files,
        primary_gate=summary["gate"]["status"],mandatory_preflight="passed",
        interpretation=m["interpretation"],previous_failed_and_preparation_records_preserved=True,
        raw_questions_references_answers_source_cards_and_api_not_copied=True,git_commit=False,github_push=False)
    dump(output/"backup_manifest.json",record);dump(destination/"backup_manifest.json",record)
    paths=[p for p in output.rglob("*") if p.is_file() and "mplconfig" not in p.parts and p.name!="artifact_manifest.json" and p.suffix!=".tmp"]
    dump(output/"artifact_manifest.json",dict(saved_at=utc(),
        output_hashes={str(p.resolve()):digest(p) for p in sorted(paths)},code_hashes={str((project/name).resolve()):digest(project/name) for name in code},
        tests=load(output/"targeted_tests.json"),audit=load(output/"integrity_audit.json"),
        visual_review="passed",all_candidates_and_unresolved_retained=True,
        primary_numeric_gate=summary["gate"]["status"],mandatory_preflight_before_all_answers=True,
        references_scope="same_vendor_automatic_consistency_not_independent_truth"))
    print("Archived V8.2:",len(files),"files;",len(paths),"output hashes")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("project","output","backup"):parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();archive(args.project,args.output,args.backup)
