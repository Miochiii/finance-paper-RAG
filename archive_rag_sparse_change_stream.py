"""Archive V10 checkpoints or completed aggregates; keep raw data local."""
import argparse
import shutil
from pathlib import Path
from rag_fresh_change_experiment import load,dump,digest,utc
from rag_prospective_coverage_validation import verify_hashes
from rag_sparse_change_stream import verify,NEW_CODE


def archive(project,output,backup):
    project,output,backup=[p.resolve() for p in (project,output,backup)]
    if not output.is_relative_to(project) or project==backup or project.is_relative_to(backup) or backup.is_relative_to(project):
        raise ValueError('invalid project roots')
    protocol=verify(output)
    complete=(output/'integrity_audit.json').exists() and (output/'analysis_summary.json').exists()
    audit=load(output/('integrity_audit.json' if complete else 'pretest_integrity_audit.json'))
    if audit['status']!='passed':raise ValueError('audit did not pass')
    destination=backup/'docs/experiments/sparse_change_stream_v10_20261002'
    files=[]
    def save(source,target):
        target=target.resolve()
        if not target.is_relative_to(backup):raise ValueError('backup destination escaped root')
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        if digest(source)!=digest(target):raise ValueError('copy hash mismatch')
        files.append(dict(source=str(source.resolve()),destination=str(target),sha256=digest(target)))
    code=NEW_CODE+['audit_rag_sparse_pretest.py','archive_rag_sparse_change_stream.py']
    for name in code:save(project/name,backup/name)
    aggregates=['targeted_tests.json','source_split.json','pretest_integrity_audit.json','pretest_disposition.json']
    if complete:
        aggregates+=['analysis_summary.json','integrity_audit.json','observations.csv','detector_results.csv',
                     'alarm_times.csv','prespecified_traces.csv','signal_metrics.csv','stream_realizations.csv',
                     'source_groups.csv','workflow_summary.json']
    for name in aggregates:save(output/name,destination/name)
    docs=[(project/'log/稀疏故障新生成序列V10协议_20261002.md','rag_sparse_v10_protocol_20261002.md'),
          (output/'pretest_progress.md','rag_sparse_v10_pretest_20261002.md')]
    if complete:docs.append((output/'sparse_stream_report.md','rag_sparse_v10_report_20261002.md'))
    for source,name in docs:save(source,backup/'docs'/name)
    if complete:
        for ext in ('png','svg'):save(output/('sparse_stream_results.'+ext),backup/'docs/assets'/('rag_sparse_v10_20261002.'+ext))
    record=dict(saved_at=utc(),phase='complete' if complete else 'mean_phase_checkpoint',copied_files=len(files),files=files,
        raw_questions_evidence_answers_references_schedule_and_score_cache_not_copied=True,git_commit=False,github_push=False)
    name='backup_manifest.json' if complete else 'pretest_backup_manifest.json'
    dump(output/name,record);dump(destination/name,record)
    paths=[p for p in output.rglob('*') if p.is_file() and 'mplconfig' not in p.parts and p.suffix!='.tmp'
           and p.name not in {'artifact_manifest.json','pretest_checkpoint_manifest.json'}]
    manifest=dict(saved_at=utc(),phase=record['phase'],output_hashes={str(p.resolve()):digest(p) for p in sorted(paths)},
        code_and_input_hashes=dict(protocol['input_hashes'],**{str((project/n).resolve()):digest(project/n) for n in code}),
        audit=audit['status'],tests=25,interpretation=protocol['interpretation'])
    manifest_name='artifact_manifest.json' if complete else 'pretest_checkpoint_manifest.json'
    dump(output/manifest_name,manifest);verify_hashes(manifest['output_hashes']);verify_hashes(manifest['code_and_input_hashes'])
    print('Archived',record['phase'],len(files),'files;',len(paths),'output hashes verified')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('project','output','backup'):ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();archive(a.project,a.output,a.backup)
