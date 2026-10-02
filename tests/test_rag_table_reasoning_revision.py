import rag_table_reasoning_revision as v


def row(source,table_id,text):
    return dict(source=source,table_id=table_id,text=text,start=10,caption='表'+table_id)


def test_same_paper_field_matched_table_preferred_then_corpus_fallback():
    spec=dict(fields=[dict(name='收益均值',header_token='收益均值(%)')],pair=['1','10'])
    target=row('a','4-1','original')
    own=row('a','5-2','平均收益0.4')
    other=row('b','4-1','收益均值(%)1.2')
    chosen,trace=v.select(spec,target,[target,own,other])
    assert chosen==own and trace['same_paper']
    chosen,trace=v.select(spec,target,[target,other])
    assert chosen==other and not trace['same_paper']


def test_ranking_normalization_does_not_mutate_raw_table():
    raw=r'$Gain(Y,X_{4})$ | 0.00061'
    assert 'gain(y,x4)' in v.rank_view(raw)
    assert raw==r'$Gain(Y,X_{4})$ | 0.00061'
