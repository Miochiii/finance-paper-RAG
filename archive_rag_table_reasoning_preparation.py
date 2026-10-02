"""Archive offline V12/V12.1 aggregates and code without raw research data."""
import argparse
import shutil
from pathlib import Path
from rag_fresh_change_experiment import load,dump,digest,utc
from rag_prospective_coverage_validation import verify_hashes
import rag_table_reasoning_stress as base


def archive(project,output,failed,backup):
    project,output,failed,backup=[p.resolve() for p in (project,output,failed,backup)]
    if not output.is_relative_to(project) or not failed.is_relative_to(project) or project==backup or backup.is_relative_to(project):raise ValueError('invalid roots')
    m=base.verify(output)
    if load(output/'preparation_integrity_audit.json')['status']!='passed':raise ValueError('preparation audit required')
    files=[]
    def save(source,target):
        target=target.resolve()
        if not target.is_relative_to(backup):raise ValueError('backup escaped')
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        if digest(source)!=digest(target):raise ValueError('copy mismatch')
        files.append(dict(source=str(source.resolve()),destination=str(target),sha256=digest(target)))
    code=base.NEW_CODE+['rag_table_reasoning_revision.py','audit_rag_table_reasoning_revision.py',
                       'tests/test_rag_table_reasoning_revision.py','archive_rag_table_reasoning_preparation.py']
    for n in code:save(project/n,backup/n)
    dest=backup/'docs/experiments/table_reasoning_stress_v12_1_20261002'
    for n in ['targeted_tests.json','preparation_integrity_audit.json','preparation_disposition.json','distractor_strength_summary.json']:
        save(output/n,dest/n)
    for n in ['preparation_integrity_audit.json','offline_preflight_failure.json','preparation_disposition.json']:
        save(failed/n,backup/'docs/experiments/table_reasoning_stress_v12_20261002'/n)
    save(project/'log/表格计算新题V12_1协议_20261002.md',backup/'docs/rag_table_v12_1_protocol_20261002.md')
    save(output/'preparation_report.md',backup/'docs/rag_table_v12_1_preparation_20261002.md')
    record=dict(saved_at=utc(),copied_files=len(files),files=files,raw_questions_references_evidence_not_copied=True,git_commit=False,github_push=False)
    dump(output/'preparation_backup_manifest.json',record);dump(dest/'preparation_backup_manifest.json',record)
    paths=[p for p in output.rglob('*') if p.is_file() and p.suffix!='.tmp' and p.name!='preparation_checkpoint_manifest.json']
    checkpoint=dict(saved_at=utc(),output_hashes={str(p.resolve()):digest(p) for p in paths},
        code_and_input_hashes=dict(m['input_hashes'],**{str((project/n).resolve()):digest(project/n) for n in code}),API_requests=0,fresh_generations=0)
    dump(output/'preparation_checkpoint_manifest.json',checkpoint)
    verify_hashes(checkpoint['output_hashes']);verify_hashes(checkpoint['code_and_input_hashes'])
    print('V12.1 offline preparation archived;',len(files),'files;',len(paths),'output hashes verified',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ['project','output','failed','backup']:ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();archive(a.project,a.output,a.failed,a.backup)
