import pytest
import rag_benign_normal_stress as v11
from rag_numeric_unit_mapping import numeric_signature


Q = dict(question='针对表2的训练样本量与准确率，请按序回答。',sources=['paper.pdf'],
         requested_slots=[dict(index=1,label='训练样本量',type='number',query_quote='训练样本量')])


def test_union_keeps_all_source_positions_and_table_axes():
    text='abc[TABLE_START]\n表2\n| 指标 | 值 |\n| 样本 | 45 |\n[/TABLE_END]xyz'
    rows=[v11.source_row(text,'paper.pdf',0,35,'test'),v11.source_row(text,'paper.pdf',20,len(text),'test')]
    result=v11.construct(Q,dict(retrieved=rows),text)
    assert result['live_dedup']['retrieved'][0]['text']==text
    assert result['live_reverse']['retrieved']==rows[::-1]
    assert sum(len(r['text']) for r in result['live_dedup']['retrieved'])<sum(len(r['text']) for r in rows)


def test_private_reference_cannot_affect_construction():
    with pytest.raises(ValueError,match='public question'):
        v11.public_grams(dict(Q,expected='45'))


def test_modified_cached_text_rejected():
    with pytest.raises(ValueError,match='exact MinerU'):
        v11.construct(Q,dict(retrieved=[dict(source='paper.pdf',start=0,end=4,text='fake')]),'real')


def test_intact_distractor_table_outside_original_union():
    table='[TABLE_START]\n表9\n| 训练样本量 | 准确率 |\n'+'| 777 | 95.1% |\n'*8+'[/TABLE_END]'
    text='x'*500+table+'z'*500+'target original'
    row,trace=v11.select_distractor(Q,text,'paper.pdf',[[len(text)-15,len(text)]],1800)
    assert row['text']==table and row['kind']=='distractor_intact_table'
    assert trace['public_bigram_hits']>0


def test_no_distractor_is_retained_as_preanswer_failure():
    row,trace=v11.select_distractor(Q,'tiny','paper.pdf',[[0,4]],1800)
    assert row is None and trace['candidate_count']==0


def test_qualification_requires_same_numeric_unit_and_all_slots():
    normal=dict(slots=[dict(index=1,type='number',expected='5',unit='%')])
    assert v11.qualify(normal,dict(basis_complete=True,slots=[dict(index=1,type='number',expected='5%',unit='')]))
    assert not v11.qualify(normal,dict(basis_complete=True,slots=[dict(index=1,type='number',expected='.05',unit='')]))
    assert not v11.qualify(normal,dict(basis_complete=True,slots=[]))
    assert numeric_signature('5','%')!=numeric_signature('.05','')


def test_generation_blocked_before_any_client_call(monkeypatch):
    def blocked(_):
        raise ValueError('preflight failed')
    monkeypatch.setattr(v11,'verify_ready',blocked)
    monkeypatch.setattr(v11.experiment,'generate',lambda _:pytest.fail('generation called after failed gate'))
    with pytest.raises(ValueError,match='preflight failed'):
        v11.generate(None)


def test_quality_error_not_counted_as_false_flag():
    rows=[dict(quality='0',proxy='incomplete',refusal=0),dict(quality='2',proxy='incomplete',refusal=0)]
    result=v11.metrics(rows)
    assert result['bad_flagged']==1 and result['complete_false_flags']==1
