import pytest

from rag_sentence_boundary_repair import repair_passage,repair_selected,boundaries
from rag_fresh_change_experiment import sha


def passage(text,start,end,source="x.pdf"):
    return dict(source=source,start=start,end=end,text=text[start:end],text_sha256=sha(text[start:end]),rank=1)


def test_cut_year_completed_without_budget_increase():
    head="已经截断的上一句在这段开头结束并附带较长的背景说明。"
    prefix="从2011年开始至20"
    body="景"*(2000-len(head)-len(prefix))
    full=head+body+prefix+"18年的交易数据进行了实证研究。"
    end=2000
    start=0
    old=passage(full,start,end)
    new=repair_passage(old,full)
    assert new["boundary_repair"]["status"]=="repaired"
    assert "2018年的交易数据进行了实证研究。" in new["text"]
    assert len(new["text"])<=2000 and new["rank"]==old["rank"]
    assert new["text"]==full[new["start"]:new["end"]]
    assert sha(new["text"])==new["text_sha256"]


def test_numeric_decimal_dates_are_not_sentence_boundaries():
    text="数值0.312，日期2026.10.02。后文"
    assert boundaries(text,0,len(text))==[text.index("。")+1]
    end=text.index("10.02")+2
    new=repair_passage(passage(text,0,end),text)
    assert new["text"].endswith("2026.10.02。")


def test_table_contents_are_preserved_when_margin_is_extended():
    table="[TABLE_START]\n表5-1\n| n_heads | 2 |\n[/TABLE_END]"
    full="开头。"+table+"后文说明被截断在这里。"
    old=passage(full,0,len(full)-6)
    new=repair_passage(old,full)
    assert table in new["text"] and new["boundary_repair"]["status"]=="repaired"


def test_unrepairable_long_sentence_remains_explicitly_unresolved():
    full="字"*2500
    old=passage(full,0,2000)
    new=repair_passage(old,full)
    assert new["text"]==old["text"]
    assert new["boundary_repair"]["status"]=="unresolved_no_tail_boundary"


def test_already_complete_and_math_are_not_cut():
    full="公式 $a；\nb$ 接着正文。然后结束。"
    assert full.index("\n")+1 not in boundaries(full,0,len(full))
    end=full.index("正文。")+3
    new=repair_passage(passage(full,0,end),full)
    assert new["boundary_repair"]["status"]=="unchanged_already_complete"


def test_source_mismatch_and_budget_overflow_are_rejected():
    full="完整句子。"*500
    old=passage(full,0,1998)
    with pytest.raises(ValueError):repair_passage(dict(old,text="编造原文"),full)
    with pytest.raises(ValueError):repair_selected([old]*4,{"x.pdf":full})
