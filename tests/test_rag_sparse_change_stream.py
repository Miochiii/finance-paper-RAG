import numpy as np
import pytest
from pathlib import Path
import rag_sparse_change_stream as v10


def questions():
    return {f"Q{i}{j}":dict(sources=[f"paper{i}"]) for i in range(8) for j in range(2 if i<2 else 1)}


def test_source_split_has_no_paper_overlap_and_balances_questions():
    split=v10.split_sources(questions(),123)
    assert not set(split["calibration"]) & set(split["test"])
    assert all(len(g)==4 and sum(map(len,g.values()))==5 for g in split.values())


def test_schedule_keeps_all_fixed_positions_and_no_prechange_injection():
    split=v10.split_sources(questions(),123)
    cfg=v10.CONFIG
    plan=v10.schedule(split,cfg)
    assert len(plan)==2480
    assert plan==v10.schedule(split,cfg)
    assert len({r['request_id'] for r in plan})==len(plan)
    assert all(r['condition']=='live_normal' for r in plan if r['phase']!='test' or r['step']<=cfg['change_at'])
    for r in plan:
        pool=split['test' if r['phase']=='test' else 'calibration']
        assert r['question_key'] in pool[r['source']]


def test_cache_identity_includes_finish_and_both_bases():
    a=dict(question="q",answer="a",finish_reason="stop",reference={'x':1},monitor={'x':2})
    assert v10.score_key(a)==v10.score_key(dict(a))
    assert v10.score_key(a)!=v10.score_key(dict(a,finish_reason='length'))
    assert v10.score_key(a)!=v10.score_key(dict(a,monitor={'x':3}))


def test_unknown_is_retained_and_alarm_positive():
    grade=dict(quality='U',proxy_comparison={'signal':{'status':'U','score':None}},field_refusal={'refusal':0},legacy_refusal=0)
    row=v10.signal_row({'request_id':'x'},'key',grade)
    assert row['coverage_flag']==1 and row['coverage_u']==1 and row['quality']=='U'


def test_cusum_and_e_statistics_are_causal_and_restart():
    cfg=v10.CONFIG
    a=np.array([[0,1,0,0,0]],float);b=np.array([[0,1,0,1,1]],float)
    sa=v10.method_series(a,a,.05,.3,cfg);sb=v10.method_series(b,b,.05,.3,cfg)
    for name in v10.METHODS:np.testing.assert_allclose(sa[name][:,:3],sb[name][:,:3])
    assert sa['edetector_fixed100'][0,0]<1
    assert sa['cusum_fixed'][0,0]==0


def test_test_collection_blocked_before_client_construction(monkeypatch):
    monkeypatch.setattr(v10,'verify',lambda _: {})
    def fail(_):raise ValueError('calibration lock missing')
    monkeypatch.setattr(v10,'verify_calibration',fail)
    with pytest.raises(ValueError,match='calibration lock'):
        v10.collect(Path('unused_output'),'test')


def test_prealarm_is_not_detection_and_misses_count_in_restricted_delay():
    result=v10.summarize_alarms([0,0],[5,0,18],40,16)
    assert result['pre_alarms']==1 and result['detected']==1
    assert result['restricted_post_delay']==(25+2)/2
