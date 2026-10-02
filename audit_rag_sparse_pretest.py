"""Audit the completed mean phase without collecting blocked future phases."""
import argparse
import json
from pathlib import Path
import rag_auto_evaluation as auto
import rag_sparse_change_stream as v10
from rag_fresh_change_experiment import load,dump,sha,utc
from rag_objective_coverage_pilot import compare


def audit(output):
    m=v10.verify(output);qs=load(output/"public_questions.json")
    refs=load(output/"private_references.json");plans=load(output/"normal_monitor_plans.json")
    cases=load(output/"public_cases.json");split=v10.split_sources(qs,m['config']['seed'])
    if split!=load(output/"source_split.json") or v10.schedule(split,m['config'])!=load(output/"private_schedule.json"):
        raise ValueError("source split/schedule changed")
    slots=[s for s in load(output/"private_schedule.json") if s['phase']=='mean']
    api={p.stem:load(p) for p in (output/"api_records").glob("*.json")}
    cache={p.stem:load(p) for p in (output/"score_cache").glob("*.json")}
    used=set();checked=set();rows=[];groups={};ids=[]
    def expect(out,name,prompt,payload,n=0):
        r=api[name]
        if (not r.get('ok') or r['prompt']!=prompt or r['payload']!=payload or r['round']!=n
                or r['input_sha256']!=sha(json.dumps([auto.MODEL,prompt,payload,n],ensure_ascii=False,sort_keys=True))
                or json.loads(r['raw'])!=r['parsed']):raise ValueError('API record mismatch')
        used.add(name);return r['parsed']
    request=auto.request;auto.request=expect
    try:
        for slot in slots:
            rid=slot['request_id'];qid=slot['question_key']
            gen=load(output/"responses"/(rid+".json"));obs=load(output/"observations"/(rid+".json"))
            payload=v10.generation_payload(cases[qid+':'+slot['condition']],m['config'])
            if gen['slot']!=slot or gen['payload']!=payload or gen['payload_sha256']!=sha(json.dumps(payload,ensure_ascii=False,sort_keys=True)):
                raise ValueError('generation inputs changed')
            raw=gen['raw_response']
            if (gen['answer']!=raw['choices'][0]['message']['content'].strip() or gen['response_id']!=raw['id']
                or gen['actual_model']!=raw['model'] or gen['finish_reason']!=raw['choices'][0]['finish_reason']
                or gen['usage']!=raw['usage']):raise ValueError('raw generation mismatch')
            inputs=v10.grade_input(qs[qid]['question'],gen,refs[qid],plans[qid]);key=v10.score_key(inputs)
            c=cache[key]
            if c['inputs']!=inputs or c['key']!=key:raise ValueError('cache inputs changed')
            if key not in checked:
                ref=compare(output,'reference_'+key,inputs['question'],inputs['answer'],inputs['reference'])
                proxy=compare(output,'proxy_'+key,inputs['question'],inputs['answer'],inputs['monitor'])
                if c['grade']!=v10.derive_grade(ref,proxy,inputs['answer'],inputs['finish_reason'],qs[qid]):
                    raise ValueError('cache score replay differs')
                checked.add(key)
            if any(obs[k]!=v for k,v in v10.signal_row(slot,key,c['grade']).items()):raise ValueError('observation changed')
            if gen['started_at']<=m['locked_at'] or gen['finished_at']>obs['processed_at'] or c['finished_at']>obs['processed_at']:
                raise ValueError('chronology changed')
            groups.setdefault(slot['stream'],[]).append((gen,obs));rows.append(obs);ids.append(gen['response_id'])
    finally:auto.request=request
    if used!=set(api) or checked!=set(cache) or len(ids)!=len(set(ids)):raise ValueError('freshness/API/cache attrition')
    for pairs in groups.values():
        pairs.sort(key=lambda pair:pair[0]['slot']['step'])
        if any(a[1]['processed_at']>b[0]['started_at'] for a,b in zip(pairs,pairs[1:])):
            raise ValueError('stream not serial')
    result=dict(status='passed',checked_at=utc(),scope='completed_mean_phase_only_not_full_stream_experiment',
                fresh_generations=len(rows),unique_generation_ids=len(set(ids)),distinct_score_inputs=len(cache),
                scoring_api_responses=len(api),quality_metrics=v10.quality_metrics(rows),
                future_phases_not_evaluated=True,source_split_schedule_payload_cache_and_chronology_replayed=True)
    dump(output/"pretest_integrity_audit.json",result);print(result)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    audit(ap.parse_args().output)
