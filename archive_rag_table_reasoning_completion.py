"""Copy completed V12.1 code and aggregates; freeze local raw artifacts in place."""
import argparse
import shutil
from pathlib import Path
from rag_fresh_change_experiment import load, dump, digest, utc
from rag_prospective_coverage_validation import verify_hashes
import rag_table_reasoning_stress as base


def archive(project, output, backup):
    project, output, backup = [p.resolve() for p in (project, output, backup)]
    if (not output.is_relative_to(project) or project == backup or
            project.is_relative_to(backup) or backup.is_relative_to(project)):
        raise ValueError('invalid archive roots')
    protocol = base.verify_ready(output)
    checkpoint = load(output/'preparation_checkpoint_manifest.json')
    verify_hashes(checkpoint['output_hashes']); verify_hashes(checkpoint['code_and_input_hashes'])
    for name in ['integrity_audit.json', 'completion_integrity_audit.json']:
        if load(output/name)['status'] != 'passed': raise ValueError('passed audits required')
    if load(output/'workflow_summary.json')['status'] != 'completed_v12_1':
        raise ValueError('completion report required')
    tests = load(output/'supplementary_tests.json')
    if tests['status'] != 'passed': raise ValueError('targeted tests required')
    code = base.NEW_CODE + ['rag_table_reasoning_revision.py', 'audit_rag_table_reasoning_revision.py',
        'tests/test_rag_table_reasoning_revision.py', 'archive_rag_table_reasoning_preparation.py',
        'diagnose_rag_table_calculation_literals.py', 'tests/test_rag_table_calculation_literals.py',
        'finalize_rag_table_reasoning_stress.py', 'tests/test_finalize_rag_table_reasoning_stress.py',
        'archive_rag_table_reasoning_completion.py']
    copied = []
    def save(source, target):
        target = target.resolve()
        if not target.is_relative_to(backup): raise ValueError('backup path escaped')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if digest(source) != digest(target): raise ValueError('copy hash mismatch')
        copied.append(dict(source=str(source.resolve()), destination=str(target), sha256=digest(target)))
    for name in code: save(project/name, backup/name)
    destination = backup/'docs/experiments/table_reasoning_stress_v12_1_20261002'
    aggregates = ['analysis_summary.json', 'integrity_audit.json', 'completion_integrity_audit.json',
        'workflow_summary.json', 'signal_metrics.csv', 'task_metrics.csv', 'paired_quality.csv',
        'answer_scores.csv', 'candidate_screening.csv', 'distractor_strength_summary.json',
        'distractor_stratified_metrics.csv', 'calculation_literal_diagnostic.json',
        'calculation_literal_diagnostic.csv', 'filtered_literal_diagnostic.json',
        'filtered_literal_diagnostic.csv', 'supplementary_tests.json', 'expanded_api_authorization.json',
        'api_approval_rejection.json', 'calculation_diagnostic_lock.json']
    for name in aggregates: save(output/name, destination/name)
    for stem in ['report', 'conclusions']:
        save(output/('table_reasoning_'+stem+'.md'), backup/'docs'/('rag_table_v12_1_'+stem+'_20261002.md'))
    for ext in ['png', 'svg']:
        save(output/('table_reasoning_summary.'+ext), backup/'docs/assets'/('rag_table_v12_1_20261002.'+ext))
    record = dict(saved_at=utc(), copied_files=len(copied), files=copied,
        raw_questions_source_evidence_references_answers_API_records_not_copied=True,
        git_commit=False, github_push=False)
    dump(output/'backup_manifest.json', record); dump(destination/'backup_manifest.json', record)
    code_hashes = dict(protocol['input_hashes'])
    code_hashes.update(checkpoint['code_and_input_hashes'])
    code_hashes.update(load(output/'calculation_diagnostic_lock.json')['hashes'])
    code_hashes.update({str((project/n).resolve()): digest(project/n) for n in code})
    paths = sorted(p for p in output.rglob('*') if p.is_file() and p.suffix != '.tmp' and p.name != 'artifact_manifest.json')
    manifest = dict(saved_at=utc(), phase='completed_v12_1',
        output_hashes={str(p.resolve()): digest(p) for p in paths},
        code_and_input_hashes=code_hashes, source_hashes=protocol['ranking_source_hashes'],
        tests_passed=tests['passed'], integrity_audit='passed', completion_audit='passed',
        automatic_quality_errors=7, monitor_U=1, conclusions_scope='point_level_development_batch_not_change_point_superiority')
    dump(output/'artifact_manifest.json', manifest)
    verify_hashes(manifest['output_hashes']); verify_hashes(manifest['code_and_input_hashes'])
    verify_hashes(manifest['source_hashes'])
    for item in copied:
        if digest(Path(item['destination'])) != item['sha256']: raise ValueError('backup changed during archive')
    print('V12.1 archived:', len(copied), 'verified copies;', len(paths), 'output hashes;', len(code_hashes), 'code/input hashes;', tests['passed'], 'tests', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['project', 'output', 'backup']: ap.add_argument('--'+name, type=Path, required=True)
    args = ap.parse_args()
    archive(args.project, args.output, args.backup)
