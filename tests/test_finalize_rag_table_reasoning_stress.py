import copy
import pytest
from finalize_rag_table_reasoning_stress import (
    numbered_fields, rational_filter_winner, verify_generation_metadata,
    verify_scope_and_chronology, expected_cases,
)


def generation():
    raw = dict(model='actual-model', id='unique-id', usage=dict(total_tokens=10),
               choices=[dict(finish_reason='stop', message=dict(content='answer'))])
    return dict(raw_response=raw, actual_model='actual-model', response_id='unique-id',
                finish_reason='stop', usage=dict(total_tokens=10), answer='answer',
                started_at='2026-10-02T01:00:00+00:00', finished_at='2026-10-02T01:00:01+00:00')


def test_raw_generation_metadata_tampering_rejected():
    verify_generation_metadata(generation())
    for field,value in [('actual_model','requested-model'), ('finish_reason','length'),
                        ('usage',dict(total_tokens=0)), ('response_id','reused-id')]:
        changed = copy.deepcopy(generation()); changed[field] = value
        with pytest.raises(ValueError): verify_generation_metadata(changed)


def test_scoring_chronology_unknown_record_and_authorized_cap_rejected():
    gen = {'R0001': generation()}
    api = {'candidate_check_x': dict(ok=True,started_utc='2026-10-02T00:00:00+00:00'),
           'compare_proxy_R0001_r2': dict(ok=True,started_utc='2026-10-02T01:00:02+00:00')}
    auth = dict(maximum_admission_requests=1, maximum_fresh_answers=1,
                maximum_logical_requests_including_repairs=3)
    time = '2026-10-02T00:30:00+00:00'
    assert verify_scope_and_chronology(api,gen,auth,time)['logical_request_records']==3
    early = copy.deepcopy(api); early['compare_proxy_R0001_r2']['started_utc']='2026-10-02T01:00:00+00:00'
    with pytest.raises(ValueError): verify_scope_and_chronology(early,gen,auth,time)
    unknown = dict(api); unknown['other']=unknown.pop('compare_proxy_R0001_r2')
    with pytest.raises(ValueError): verify_scope_and_chronology(unknown,gen,auth,time)
    small = dict(auth,maximum_logical_requests_including_repairs=2)
    with pytest.raises(ValueError): verify_scope_and_chronology(api,gen,small,time)


def test_post_hoc_field_reading_does_not_choose_gold_or_footer():
    assert numbered_fields('1）组合6\n2）0.7467\n来源: [来源2]')==('6','0.7467')
    assert numbered_fields('1. 实体：AG\n2. ADF：-14.5448')==('AG','-14.5448')
    assert numbered_fields('1. AG\n1. AU\n2. -14.5448') is None
    assert numbered_fields('1. AG\n2. 1e-3') is None


def test_fraction_filter_handles_negative_max_and_joint_constraints():
    grid = [dict(entity=e,values={k:dict(value=v) for k,v in values.items()})
            for e,values in [('a',dict(metric='-3',location='A',limit='1')),
                            ('b',dict(metric='-10',location='A',limit='1')),
                            ('c',dict(metric='10',location='B',limit='1'))]]
    rule = dict(filters=[('location','==','A'),('limit','<=','1')],target='metric')
    entity,value = rational_filter_winner(rule,grid)
    assert entity=='a' and value==-3
    grid.append(copy.deepcopy(grid[0])); grid[-1]['entity']='d'
    with pytest.raises(ValueError): rational_filter_winner(rule,grid)


def test_randomized_cases_keep_only_preanswer_admitted_pairs():
    rows = [dict(question_key=q,experimental_admission=int(q!='excluded')) for q in ['q1','excluded','q2']]
    questions = {q:dict(question=q+'?') for q in ['q1','excluded','q2']}
    tasks = {q:dict(bank_id=q) for q in questions}
    banks = {q:dict(variants={c:dict(context=q+c) for c in ['live_normal','live_table_distractor']}) for q in questions}
    cases,keys = expected_cases(rows,questions,tasks,banks,42)
    assert len(cases)==4
    assert {(k['question_key'],k['condition']) for k in keys.values()}=={
        (q,c) for q in ['q1','q2'] for c in ['live_normal','live_table_distractor']}
    assert (cases,keys)==expected_cases(rows,questions,tasks,banks,42)
