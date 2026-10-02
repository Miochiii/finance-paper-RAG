"""Offline replay of V12 source cells, formula answers and optional API trial."""
import argparse
import json
from pathlib import Path

import rag_auto_evaluation as auto
import rag_table_reasoning_stress as v12
from rag_core.mineru_loader import blocks_to_text
from rag_fresh_change_experiment import load,dump,sha,utc
from rag_prospective_coverage_validation import verify_hashes
from rag_objective_coverage_pilot import compare
from rag_mineru_coverage_v4 import point_quality
from rag_field_refusal import field_refusal
from rag_refusal import explicit_refusal


def preparation(output):
    m=v12.verify(output);parent=Path(m['previous'])
    prior=load(parent/'artifact_manifest.json')
    verify_hashes(prior['output_hashes']);verify_hashes(prior['code_and_input_hashes'])
    qs=load(output/'public_questions.json');tasks=load(output/'private_task_rules.json');refs=load(output/'private_references.json')
    banks=load(output/'table_banks.json');sources=load(output/'source_manifest.json');old=load(parent/'public_questions.json')
    if len(qs)!=27 or len(banks)!=9:raise ValueError('candidate attrition at preparation')
    public_specs={s['bank_id']:s for s in v12.specifications()}
    for bid,bank in banks.items():
        source=bank['source'];text=blocks_to_text(load(Path(sources[source]['mineru_path'])))
        if sha(text)!=sources[source]['text_sha256']:raise ValueError('converted source changed')
        spec=public_specs[bid]
        if old[spec['seed_qid']]['sources']!=[source]:raise ValueError('public seed source changed')
        all_tables=v12.tables(text,source)
        target=[t for t in all_tables if t['table_id']==spec['table_id']]
        if len(target)!=1 or target[0]!=bank['target_table']:raise ValueError('target table locator changed')
        table=target[0];grid=v12.parse_grid(spec,table)
        if grid!=bank['private_source_grid']:raise ValueError('source grid changed')
        for row in grid:
            for cell in row['values'].values():
                a,b=cell['source_span']
                if text[a:b]!=cell['literal']:raise ValueError('source cell coordinate mismatch')
        distractor,trace=v12.select_distractor(spec,table,all_tables)
        if trace!=bank['distractor_selection_trace']:raise ValueError('distractor selection changed')
        for c,rows in zip(v12.CONDITIONS,[[table],[distractor,table] if distractor else []]):
            variant=dict(retrieved=rows,evidence=v12.evidence_projection(rows),context=v12.source_context(rows),
              structurally_valid=bool(rows) and sum(len(r['text']) for r in rows)<=v12.CONFIG['maximum_text_characters'],
              text_characters=sum(len(r['text']) for r in rows))
            if variant!=bank['variants'][c]:raise ValueError('context changed')
            if any(text[r['start']:r['end']]!=r['text'] for r in rows):raise ValueError('evidence edited')
        for kind in v12.TYPES:
            qid=bid+'_'+kind;public,task=v12.make_question(spec,table,kind)
            if public!=qs[qid] or dict(bank_id=bid,**task)!=tasks[qid]:raise ValueError('question or rule changed')
            plan=v12.make_plan(public,task,grid,table)
            if plan!=refs[qid]:raise ValueError('arithmetic reference changed')
            if any(auto.norm(public['question'])==auto.norm(q['question']) for q in old.values()):raise ValueError('old question repeated')
    paired_banks=sum(b['variants'][v12.CONDITIONS[1]]['structurally_valid'] for b in banks.values())
    result=dict(status='passed',checked_at=utc(),new_questions=len(qs),source_papers=len(sources),
       task_types=v12.TYPES,source_tables=len(banks),structural_paired_questions=paired_banks*3,
       intact_field_matched_distractor_tables=paired_banks,all_source_cell_coordinates_verified=True,
       Decimal_arithmetic_checked_by_independent_Fraction_rounding=True,all_original_source_tables_preserved=True,
       no_reference_values_or_paper_titles_in_distractor_ranking=True,all_candidates_pending=True,
       test_generations_collected=len(list((output/'generations').glob('*.json'))),
       scope='source_coordinates_and_calculation_derivation_not_independent_semantic_truth')
    dump(output/'preparation_integrity_audit.json',result)
    print('V12 preparation audit passed;',len(qs),'new questions;',paired_banks,'intact distractors',flush=True)
    return result


def audit(output):
    v12.verify_ready(output)
    # Do not overwrite the frozen offline preparation audit.
    m=load(output/'protocol_lock.json');qs=load(output/'public_questions.json');refs=load(output/'private_references.json')
    tasks=load(output/'private_task_rules.json');banks=load(output/'table_banks.json');keys=load(output/'private_case_key.json');cases=load(output/'public_cases.json')
    api={p.stem:load(p) for p in (output/'api_records').glob('*.json')};expected=set()
    def expect(out,name,prompt,payload,n=0):
        r=api[name];h=sha(json.dumps([auto.MODEL,prompt,payload,n],ensure_ascii=False,sort_keys=True))
        if not r.get('ok') or r['input_sha256']!=h or r['prompt']!=prompt or r['payload']!=payload or r['round']!=n or json.loads(r['raw'])!=r['parsed']:
            raise ValueError('raw API payload mismatch: '+name)
        expected.add(name);return r['parsed']
    saved=auto.request;auto.request=expect
    try:
        for qid,q in qs.items():
            task=tasks[qid];bank=banks[task['bank_id']]
            payload=dict(public_question=q,public_rule=task['rule'],candidate_facts=refs[qid]['facts'],source_grid=bank['private_source_grid'],source_table=bank['target_table']['text'])
            rounds=[expect(output,f'candidate_check_{qid}_r{n}',v12.REFERENCE_CHECK+('\n优先核对行列、表号、单位。' if n==1 else '\n独立复算算术、过滤条件及唯一性。'),payload,n) for n in (1,2)]
            row=load(output/'candidate_checks'/(qid+'.json'))
            if row['checks']!=rounds or row['qualified']!=all(v12.flags_pass(r,v12.REFERENCE_FLAGS) for r in rounds):raise ValueError('candidate checks changed')
        for bid,bank in banks.items():
            for c in v12.CONDITIONS:
                if not bank['variants'][c]['structurally_valid']:continue
                evidence=bank['variants'][c]['evidence'];payload=dict(public_schema=bank['public_schema'],retrieved_evidence=evidence)
                raw=expect(output,'grid_extract_'+bid+'_'+c,v12.GRID_EXTRACT,payload)
                rounds=[expect(output,f'grid_check_{bid}_{c}_r{n}',v12.GRID_CHECK+('\n优先核对行列对应。' if n==1 else '\n优先核对表号、单位与完整性。'),dict(**payload,temporary_grid=raw),n) for n in (1,2)]
                stored=load(output/'grid_checks'/(bid+'_'+c+'.json'))
                if stored['raw']!=raw or stored['checks']!=rounds or stored['plan']!=v12.grid_plan(raw,rounds,bank,evidence):raise ValueError('blind grid admission changed')
        gens=[]
        for aid,case in cases.items():
            qid=keys[aid]['question_key'];task=tasks[qid];bank=banks[task['bank_id']]
            grid=load(output/'grid_checks'/(task['bank_id']+'_live_normal.json'))['plan']['grid']
            normal=v12.make_plan(qs[qid],task,grid,bank['target_table'])
            if normal!=load(output/'proxy_plans'/(qid+'.json')):raise ValueError('monitor altered or uses stress/reference facts')
            gen=load(output/'generations'/(aid+'.json'));cfg=m['config']
            payload=dict(model=cfg['model'],temperature=cfg['generation_temperature'],max_tokens=cfg['generation_max_tokens'],
                messages=[dict(role='system',content=cfg['system_prompt']),dict(role='user',content=f'问题：{case["question"]}\n\n参考上下文：\n{case["context"]}')])
            raw=gen['raw_response']
            if gen['payload']!=payload or gen['payload_sha256']!=sha(json.dumps(payload,ensure_ascii=False,sort_keys=True)) or gen['answer']!=raw['choices'][0]['message']['content'].strip() or gen['response_id']!=raw['id']:
                raise ValueError('generation payload or raw changed')
            reference=compare(output,'reference_'+aid,case['question'],gen['answer'],refs[qid]);proxy=compare(output,'proxy_'+aid,case['question'],gen['answer'],normal)
            if gen['finish_reason']!='stop':proxy['signal'].update(status='U',score=None)
            score=dict(quality=point_quality(reference,gen['finish_reason']),reference_comparison=reference,proxy_comparison=proxy,
               legacy_refusal=int(explicit_refusal(gen['answer'])),field_refusal=field_refusal(gen['answer'],[s['label'] for s in qs[qid]['requested_slots']]))
            if score!=load(output/'scores'/(aid+'.json')):raise ValueError('score derivation changed')
            gens.append(gen)
    finally:auto.request=saved
    if set(api)!=expected:raise ValueError('unaccounted API records')
    for d in ['generations','scores']:
        if {p.stem for p in (output/d).glob('*.json')}!=set(cases):raise ValueError('answer/score attrition')
    if load(output/'preflight_lock.json')['locked_at']>=min(g['started_at'] for g in gens):raise ValueError('generation before preflight')
    ids=[r['response_id'] for r in api.values()]+[g['response_id'] for g in gens]
    if len(ids)!=len(set(ids)):raise ValueError('response IDs reused')
    result=dict(status='passed',checked_at=utc(),checked_fresh_generations=len(gens),checked_API_records=len(api),
       candidate_checks_and_blind_grid_derivations_replayed=True,normal_monitor_grid_independent_of_candidate_reference_payload=True,
       shared_public_arithmetic_rule=True,all_response_IDs_unique=True,all_U_and_preanswer_exclusions_retained=True,
       mandatory_preflight_before_answers=True,scope='dataflow_integrity_not_independent_truth_or_change_point_detection')
    dump(output/'integrity_audit.json',result);print('V12 trial integrity audit passed',flush=True);return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=['preparation','audit']);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();globals()[a.action](a.output)
