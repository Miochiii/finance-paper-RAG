from decimal import Decimal
import pytest
import rag_table_reasoning_stress as v12


def table(text):
    return dict(text=text,start=100,end=100+len(text),source='paper.pdf',table_id='1-1',caption='表1-1',mineru_page=4)


def grid(a,b):
    return [dict(entity='A',values={'值':dict(value=a)}),dict(entity='B',values={'值':dict(value=b)})]


def test_exact_cell_spans_and_no_trailing_empty_column():
    t=table('[TABLE_START]\n表1-1\n| 对象 | 值 |\n| --- | --- |\n| A | -1.23 |\n[/TABLE_END]')
    rows=v12.lines(t)
    assert rows[0]['cells']==['对象','值'] and rows[1]['cells']==['A','-1.23']
    a,b=rows[1]['spans'][1]
    assert t['text'][a-t['start']:b-t['start']]=='-1.23'


def test_percent_display_values_do_not_become_fractions():
    assert v12.scalar('11.2%')==Decimal('11.2')
    assert v12.calculate(dict(kind='difference',pair=['A','B'],metric='值',places=4),grid('11.2','11.08'))[-1]=='0.1200'


@pytest.mark.parametrize('a,b,wanted',[('1','3','0.3333'),('-1','3','-0.3333'),('1.00005','1','1.0001'),('-1.00005','1','-1.0001')])
def test_decimal_rounding_agrees_with_independent_rational(a,b,wanted):
    assert v12.calculate(dict(kind='ratio',pair=['A','B'],metric='值',places=4),grid(a,b))[-1]==wanted


def test_zero_division_and_tied_maximum_block_admission():
    with pytest.raises(ValueError,match='zero denominator'):
        v12.calculate(dict(kind='ratio',pair=['A','B'],metric='值',places=4),grid('2','0'))
    with pytest.raises(ValueError,match='tied'):
        v12.calculate(dict(kind='filtered_max',target='值',filters=[['值','>','0']]),grid('2','2'))


def test_negative_values_compared_without_absolute_value():
    assert v12.calculate(dict(kind='filtered_max',target='值',filters=[['值','<','-1']]),grid('-14','-23'))==['A','-14']


def test_double_block_source_axis_mapping():
    t=table('[TABLE_START]\n表1-1\n| 组合 | 值 | 组合 | 值 |\n| --- | --- | --- | --- |\n| 1 | .9 | 2 | .8 |\n[/TABLE_END]')
    s=dict(format='double_block',entity_header='组合',fields=[v12.field('值',1)])
    g=v12.parse_grid(s,t)
    assert v12.grid_values(g)=={'1':{'值':'0.9'},'2':{'值':'0.8'}}
    assert g[1]['values']['值']['column_index']==3


def test_feature_header_ragged_shape_requires_explicit_axis():
    t=table('[TABLE_START]\n表1-1\n| 股票名称 | 特征 |\n| --- | --- |\n| $X_1$ | $X_2$ |\n| A | .1 | .2 |\n[/TABLE_END]')
    s=dict(format='feature_header',entity_header='股票名称',fields=[v12.field('X1',1,'X_1'),v12.field('X2',2,'X_2')])
    assert v12.grid_values(v12.parse_grid(s,t))=={'A':{'X1':'0.1','X2':'0.2'}}


def test_approximate_and_bracketed_source_values_rejected():
    for value in ('~2.3','2.18[2.36]','1e-3','4/5'):
        with pytest.raises(ValueError):v12.scalar(value)
    assert v12.scalar('0.73***')==Decimal('.73')


def test_chart_and_cover_are_not_distractor_tables():
    text='【第1页】\n[TABLE_START]\n图1-1\n| A | 1 |\n[/TABLE_END]\n[TABLE_START]\n表2-1\n| A | 2 |\n[/TABLE_END]'
    assert [t['table_id'] for t in v12.tables(text,'paper.pdf')]==['2-1']


def test_incomplete_gate_blocks_generation_before_client(monkeypatch):
    def fail(_):raise ValueError('failed preflight')
    monkeypatch.setattr(v12,'verify_ready',fail)
    monkeypatch.setattr(v12.experiment,'generate',lambda _:pytest.fail('generation called'))
    with pytest.raises(ValueError,match='preflight'):
        v12.generate(None)


def test_real_quality_error_is_not_a_false_flag():
    result=v12.metrics([dict(quality='0',proxy='incomplete',refusal=0),dict(quality='2',proxy='incomplete',refusal=0)])
    assert result['bad_flagged']==1 and result['complete_false_flags']==1
