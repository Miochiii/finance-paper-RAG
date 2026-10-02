"""Archive V8 code, aggregate data and figures locally; no commit or upload."""
import argparse
import shutil
from pathlib import Path

from rag_fresh_change_experiment import load,dump,digest,utc
from rag_prospective_coverage_validation import verify_hashes


def archive(project,output,backup):
    project,output,backup=[p.resolve() for p in (project,output,backup)]
    if not output.is_relative_to(project) or project==backup or backup.is_relative_to(project) or project.is_relative_to(backup):
        raise ValueError("active and backup project paths invalid")
    verify_hashes(load(output/"protocol_lock.json")["input_hashes"])
    names=["rag_guarded_new_question_validation.py","audit_rag_guarded_validation.py",
           "report_rag_guarded_validation.py","archive_rag_guarded_validation.py",
           "tests/test_rag_guarded_new_question_validation.py"]
    aggregates=["analysis_summary.json","integrity_audit.json","targeted_tests.json",
        "candidate_screening.csv","admission_diagnostic.csv","source_exposure.csv",
        "answer_scores.csv","signal_metrics.csv","proxy_basis_availability.csv","live_source_groups.csv"]
    workflow_failure=output/"preflight_failure_disposition.json"
    if workflow_failure.exists():aggregates.append(workflow_failure.name)
    destination=backup/"docs/experiments/guarded_new_questions_v8_20261002"
    copied=[]
    def save(source,target,content=None):
        target=target.resolve()
        if not target.is_relative_to(backup):raise ValueError("backup path escapes project")
        target.parent.mkdir(parents=True,exist_ok=True)
        if content is None:
            shutil.copy2(source,target)
            if digest(source)!=digest(target):raise ValueError("backup copy mismatch")
        else:target.write_text(content,encoding="utf-8")
        copied.append(dict(source=str(source.resolve()),destination=str(target),sha256=digest(target),transformed=content is not None))
    for name in names:save(project/name,backup/name)
    for name in aggregates:save(output/name,destination/name)
    for ext in ("png","svg"):
        save(output/("guarded_validation_results."+ext),backup/"docs/assets"/("rag_guarded_v8_20261002."+ext))
    reports=[(project/"log/新题来源盲提取验证V8协议_20261002.md","rag_guarded_v8_protocol_20261002.md"),
             (output/"guarded_validation_report.md","rag_guarded_v8_report_20261002.md")]
    for source,name in reports:
        content=source.read_text(encoding="utf-8").replace((output/"guarded_validation_results.png").as_posix(),"assets/rag_guarded_v8_20261002.png")
        save(source,backup/"docs"/name,content)
    summary=load(output/"analysis_summary.json")
    manifest=dict(saved_at=utc(),copied_files=len(copied),files=copied,
        primary_numeric_gate=summary["gate"]["status"],interpretation=summary["interpretation"],
        workflow_validity="preflight_failed_confirmatory_use_blocked" if workflow_failure.exists() else "not_evaluated_here",
        raw_questions_answers_references_source_cards_and_api_not_copied=True,git_commit=False,github_push=False)
    dump(output/"backup_manifest.json",manifest);dump(destination/"backup_manifest.json",manifest)
    files=[p for p in output.rglob("*") if p.is_file() and "mplconfig" not in p.parts and p.name!="artifact_manifest.json" and p.suffix!=".tmp"]
    dump(output/"artifact_manifest.json",dict(saved_at=utc(),output_hashes={str(p.resolve()):digest(p) for p in sorted(files)},
        code_hashes={str((project/name).resolve()):digest(project/name) for name in names},
        tests=load(output/"targeted_tests.json"),audit=load(output/"integrity_audit.json"),
        visualization_checked=True,all_attempts_cases_and_unresolved_retained=True,
        primary_numeric_gate=summary["gate"]["status"],reference_semantics_scope=summary["reference_semantics_scope"]))
    print("Archived",len(copied),"backup files;",len(files),"output hashes")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("project","output","backup"):parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();archive(args.project,args.output,args.backup)
