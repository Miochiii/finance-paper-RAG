import pytest

from rag_public_field_retrieval import (public_query, surface_map, source_surface_anchor,
                                       field_chunks, table_ids, overlap)


def test_projection_rejects_gold_and_conditions():
    q=dict(question="依据论文《测试》，针对表5-9的参数，请仅逐项给出以下字段：1）max_depth。",
           requested_slots=[dict(index=1,label="max_depth",query_quote="max_depth",type="number")],sources=["x.pdf"])
    assert public_query(q)["table_ids"] == ["5-9"]
    assert "论文" not in public_query(q)["queries"][0]
    with pytest.raises(ValueError): public_query(dict(**q, expected="3"))
    with pytest.raises(ValueError): public_query(dict(q,requested_slots=[dict(q["requested_slots"][0],expected="3")]))


def test_symbol_anchor_preserves_raw_and_numbers():
    quote=r"第二阶段采用 $\varepsilon$ -SVM模型，与GARCH不同。"
    anchor=source_surface_anchor("ε-SVM模型",quote)
    assert quote[anchor["start"]:anchor["end"]] == anchor["literal"]
    assert surface_map(anchor["literal"])[0] == "ε-SVM模型"
    assert source_surface_anchor("0.3","参数=3") is None
    assert source_surface_anchor("ϵ-SVM模型",quote) is None
    assert surface_map(r"\varepsilonfoo")[0] == r"\varepsilonfoo"


def test_table_chunks_are_intact_exact_slices():
    text="背景"*500+"[TABLE_START]\n续表 5-1 参数\n| n_heads | 2 |\n[/TABLE_END]"+"结尾"*500
    rows=field_chunks(text,"x.pdf")
    tables=[r for r in rows if r["kind"]=="table"]
    assert len(tables)==1 and tables[0]["table_ids"]==["5-1"]
    for r in rows:
        assert text[r["start"]:r["end"]]==r["text"] and len(r["text"])<=2000
    assert "[/TABLE_END]" in tables[0]["text"]
    assert table_ids("表4.2；表 4-2；5-9") == ["4-2"]
    assert overlap(tables[0],dict(tables[0],source="y.pdf"))==0
