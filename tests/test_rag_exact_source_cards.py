from rag_exact_source_cards import exact_cards,block_regions,APPROXIMATE
from rag_core.mineru_loader import blocks_to_text


def test_chart_and_approximate_table_are_excluded_by_original_metadata():
    body='<table><tr><td>参数</td><td>值</td></tr><tr><td>a1</td><td>12</td></tr><tr><td>b2</td><td>34</td></tr></table>'
    blocks=[dict(type='chart',page_idx=0,chart_caption=['图1-1'],content='| 1 | 0.12 | 0.34 |\n'*8),
            dict(type='table',page_idx=1,table_caption=['表2-1 参数'],table_body=body),
            dict(type='table',page_idx=2,table_caption=['表3-1'],table_body=body.replace('12','~12'))]
    text=blocks_to_text(blocks);cards=exact_cards(text,'x.pdf',blocks)
    assert len(cards)==1 and cards[0]['mineru_block_type']=='table'
    assert cards[0]['mineru_block_index']==1
    assert cards[0]['text']==text[cards[0]['start']:cards[0]['end']]
    assert len(block_regions(text,blocks))==3


def test_approximation_markers_do_not_confuse_default_risk_word():
    assert not APPROXIMATE.search('违约样本数量为123，共456家公司')
    assert APPROXIMATE.search('数量约 123')
    assert APPROXIMATE.search('~0.12')
