"""Offline replay of public construction, admission, fresh payloads and scores."""
import argparse
import json
import random
from pathlib import Path

import rag_auto_evaluation as auto
import rag_benign_normal_stress as v11
from rag_core.mineru_loader import blocks_to_text
from rag_fresh_change_experiment import load, dump, sha, utc
from rag_objective_coverage_pilot import compare
from rag_mineru_coverage_v4 import point_quality
from rag_field_refusal import field_refusal
from rag_refusal import explicit_refusal
from rag_prospective_coverage_validation import verify_hashes


def audit(output):
    m = v11.verify_ready(output)
    parent = Path(m['previous'])
    old = load(parent/'artifact_manifest.json')
    verify_hashes(old['output_hashes'])
    verify_hashes(old['code_hashes'])
    qs = load(output/'public_questions.json')
    refs = load(output/'private_references.json')
    sources = load(output/'source_manifest.json')
    variants = load(output/'context_variants.json')
    if qs != load(parent/'table_axis_clarification_20261002/next_pair_public_questions.json') or refs != load(parent/'private_references.json'):
        raise ValueError('question or reference changed')
    original = load(parent/'public_retrieval.json')
    api = {p.stem:load(p) for p in (output/'api_records').glob('*.json')}
    expected = set()
    def expect(out,name,prompt,payload,n=0):
        r = api[name]
        hashed = sha(json.dumps([auto.MODEL,prompt,payload,n],ensure_ascii=False,sort_keys=True))
        if not r.get('ok') or r['prompt']!=prompt or r['payload']!=payload or r['round']!=n or r['input_sha256']!=hashed or json.loads(r['raw'])!=r['parsed']:
            raise ValueError('API input/raw mismatch '+name)
        expected.add(name)
        return r['parsed']
    saved = auto.request
    auto.request = expect  # Missing records raise locally; audit cannot call API.
    try:
        screening = []
        for qid,q in qs.items():
            path = parent/('table_axis_clarification_20261002/proxy_plans/clarified.json' if qid=='P0002' else 'proxy_plans/'+qid+'.json')
            normal = load(path)
            if normal != load(output/'proxy_plans'/(qid+'.json')):
                raise ValueError('normal monitor changed')
            retrieval = dict(original[qid],**q)
            if retrieval != load(output/'public_retrieval.json')[qid]:
                raise ValueError('original retrieval changed')
            source = q['sources'][0]
            text = blocks_to_text(load(Path(sources[source]['mineru_path'])))
            if sha(text)!=sources[source]['text_sha256']:
                raise ValueError('source conversion changed')
            bank = v11.construct(q,retrieval,text,m['config'])
            for c,row in bank.items():
                row['normal_literal_basis_preserved'] = v11.literal_basis_preserved(normal,row['evidence'])
                row['structurally_valid'] &= row['normal_literal_basis_preserved']
                if not row['normal_literal_basis_preserved']:
                    row['reason'] = 'normal_literal_basis_missing'
                if row != variants[qid][c]:
                    raise ValueError('public construction changed')
                admitted = bool(row['structurally_valid'])
                if admitted and c!='live_normal':
                    plan = v11.extract_plan(output,qid+'_'+c,q,row['evidence'])
                    result = load(output/'stress_plans'/(qid+'_'+c+'.json'))
                    admitted = v11.qualify(normal,plan)
                    if result['plan']!=plan or result['qualified']!=admitted:
                        raise ValueError('qualification derivation changed')
                screening.append(dict(question_key=qid,condition=c,source=source,structural=int(row['structurally_valid']),
                    qualified=int(admitted),structural_reason=row['reason'],text_characters=row['text_characters'],
                    baseline_text_characters=row['original_text_characters']))
        if screening!=load(output/'context_screening.json'):
            raise ValueError('screening attrition or outcome changed')
        pending = [(r['question_key'],r['condition']) for r in screening if r['qualified']]
        random.Random(m['config']['seed']).shuffle(pending)
        cases,keys = {},{}
        for i,(q,c) in enumerate(pending,1):
            aid = f'N{i:04d}'
            context = variants[q][c]['context']
            cases[aid] = dict(question=qs[q]['question'],context=context,context_sha256=sha(context))
            keys[aid] = dict(question_key=q,condition=c)
        if cases!=load(output/'public_cases.json') or keys!=load(output/'private_case_key.json') or load(output/'private_control_answers.json'):
            raise ValueError('case schedule changed or synthetic answers included')
        generations = []
        cfg = m['config']
        for aid,case in cases.items():
            qid = keys[aid]['question_key']
            gen = load(output/'generations'/(aid+'.json'))
            payload = dict(model=cfg['model'],temperature=cfg['generation_temperature'],max_tokens=cfg['generation_max_tokens'],
                messages=[dict(role='system',content=cfg['system_prompt']),
                          dict(role='user',content=f'问题：{case["question"]}\n\n参考上下文：\n{case["context"]}')])
            if gen['payload']!=payload or gen['payload_sha256']!=sha(json.dumps(payload,ensure_ascii=False,sort_keys=True)):
                raise ValueError('generation payload changed or private information leaked')
            raw = gen['raw_response']
            if gen['answer']!=raw['choices'][0]['message']['content'].strip() or gen['response_id']!=raw['id'] or gen['actual_model']!=raw['model'] or gen['finish_reason']!=raw['choices'][0]['finish_reason'] or gen['usage']!=raw['usage']:
                raise ValueError('generation raw response mismatch')
            reference = compare(output,'reference_'+aid,case['question'],gen['answer'],refs[qid])
            proxy = compare(output,'proxy_'+aid,case['question'],gen['answer'],load(output/'proxy_plans'/(qid+'.json')))
            if gen['finish_reason']!='stop':
                proxy['signal'].update(status='U',score=None)
            wanted = dict(quality=point_quality(reference,gen['finish_reason']),reference_comparison=reference,
                proxy_comparison=proxy,legacy_refusal=int(explicit_refusal(gen['answer'])),
                field_refusal=field_refusal(gen['answer'],[s['label'] for s in qs[qid]['requested_slots']]))
            if wanted!=load(output/'scores'/(aid+'.json')):
                raise ValueError('score changed or monitor uses stress plan')
            generations.append(gen)
    finally:
        auto.request = saved
    if set(api)!=expected:
        raise ValueError('unaccounted API records')
    for folder in ['generations','scores']:
        if {p.stem for p in (output/folder).glob('*.json')}!=set(cases):
            raise ValueError('answer or score attrition')
    times = [load(output/n)['locked_at'] for n in ['protocol_lock.json','qualification_lock.json','cases_lock.json','preflight_lock.json']]
    if times!=sorted(times) or max(times)>=min(g['started_at'] for g in generations):
        raise ValueError('answers preceded mandatory preflight')
    if any(r['started_utc']<=times[0] or r['started_utc']>=times[1] for name,r in api.items() if name.startswith('stress_')):
        raise ValueError('qualification chronology changed')
    if any(load(p)['finished_at']>=times[1] for p in (output/'stress_plans').glob('*.json')):
        raise ValueError('qualification followed admission lock')
    ids = [r['response_id'] for r in api.values()]+[g['response_id'] for g in generations]
    if len(ids)!=len(set(ids)):
        raise ValueError('response IDs reused')
    result = dict(status='passed',checked_at=utc(),checked_questions=len(qs),checked_screening_records=len(screening),
       checked_fresh_generations=len(generations),checked_api_records=len(api),all_response_ids_unique=True,
       exact_MinerU_source_spans_replayed=True,public_only_construction_replayed=True,
       all_original_source_positions_preserved=True,no_numbers_or_units_edited=True,
       qualification_before_answers=True,all_exclusions_retained=True,
       normal_monitor_and_reference_unchanged=True,stress_plans_excluded_from_monitor=True,
       generation_payload_question_and_context_only=True,scoring_derivation_replayed=True,
       scope='dataflow_integrity_not_independent_reference_truth_or_change_point_validation')
    dump(output/'integrity_audit.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    audit(ap.parse_args().output)
