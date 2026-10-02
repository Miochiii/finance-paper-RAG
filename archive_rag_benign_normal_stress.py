"""Archive V11 aggregates and code, keeping all raw research data in project."""
import argparse
import shutil
from pathlib import Path
from rag_fresh_change_experiment import load,dump,digest,utc
from rag_prospective_coverage_validation import verify_hashes
from rag_benign_normal_stress import verify,NEW_CODE


def archive(project,output,backup):
    project,output,backup=[p.resolve() for p in (project,output,backup)]
    if not output.is_relative_to(project) or project==backup or project.is_relative_to(backup) or backup.is_relative_to(project):
        raise ValueError('invalid archive roots')
    m=verify(output)
    complete=(output/'integrity_audit.json').exists() and (output/'analysis_summary.json').exists()
    audit=load(output/('integrity_audit.json' if complete else 'preparation_integrity_audit.json'))
    if audit['status']!='passed':
        raise ValueError('audit required')
    destination=backup/'docs/experiments/benign_normal_stress_v11_20261002'
    copied=[]
    def save(source,target):
        target=target.resolve()
        if not target.is_relative_to(backup):
            raise ValueError('backup path escaped')
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        if digest(source)!=digest(target):
            raise ValueError('copy mismatch')
        copied.append(dict(source=str(source.resolve()),destination=str(target),sha256=digest(target)))
    code=NEW_CODE+['audit_rag_benign_preparation.py','archive_rag_benign_normal_stress.py']
    for n in code:
        save(project/n,backup/n)
    names=['targeted_tests.json','preparation_integrity_audit.json','preparation_disposition.json']
    if complete:
        names+=['workflow_summary.json','analysis_summary.json','integrity_audit.json','signal_metrics.csv','paired_quality.csv',
                'source_groups.csv','answer_scores.csv','context_screening.csv']
    for n in names:
        save(output/n,destination/n)
    save(project/'log/正常证据压力V11协议_20261002.md',backup/'docs/rag_benign_v11_protocol_20261002.md')
    if complete:
        save(output/'normal_stress_report.md',backup/'docs/rag_benign_v11_report_20261002.md')
        for ext in ('png','svg'):
            save(output/('normal_stress_quality.'+ext),backup/'docs/assets'/('rag_benign_v11_20261002.'+ext))
    phase='complete' if complete else 'offline_preparation'
    record=dict(saved_at=utc(),phase=phase,copied_files=len(copied),files=copied,
                raw_questions_evidence_answers_references_and_API_records_not_copied=True,git_commit=False,github_push=False)
    name='backup_manifest.json' if complete else 'preparation_backup_manifest.json'
    dump(output/name,record)
    dump(destination/name,record)
    manifest_name='artifact_manifest.json' if complete else 'preparation_checkpoint_manifest.json'
    paths=[p for p in output.rglob('*') if p.is_file() and 'mplconfig' not in p.parts and
           p.suffix!='.tmp' and p.name not in {'artifact_manifest.json','preparation_checkpoint_manifest.json'}]
    manifest=dict(saved_at=utc(),phase=phase,output_hashes={str(p.resolve()):digest(p) for p in sorted(paths)},
        code_and_input_hashes=dict(m['input_hashes'],**{str((project/n).resolve()):digest(project/n) for n in code}),
        audit=audit['status'],tests=load(output/'targeted_tests.json')['passed'],interpretation=m['interpretation'])
    dump(output/manifest_name,manifest)
    verify_hashes(manifest['output_hashes'])
    verify_hashes(manifest['code_and_input_hashes'])
    print('Archived',phase,len(copied),'files;',len(paths),'output hashes verified',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('project','output','backup'):
        ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args()
    archive(a.project,a.output,a.backup)
