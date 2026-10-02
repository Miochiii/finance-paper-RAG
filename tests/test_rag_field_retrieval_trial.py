from rag_field_retrieval_trial import mapped_proxy


def test_mapping_requires_raw_source_quote_and_preserves_value():
    q=dict(requested_slots=[dict(index=1,label="模型",query_quote="模型",type="entity")])
    raw=dict(slots=[dict(index=1,supported="1",expected="ε-SVM模型",unit="",aliases=[],
                        eid="C1",quote=r"第二阶段采用 $\varepsilon$ -SVM模型。")])
    checks=[dict(slot_checks=[dict(index=1,supported="1",unique="1",aliases_valid="1")])]*2
    evidence=[dict(eid="C1",text=raw["slots"][0]["quote"])]
    p=mapped_proxy(raw,checks,q,evidence)
    assert not p["strict_basis_complete"] and p["basis_complete"]
    assert len(p["source_mappings"])==1 and raw["slots"][0]["expected"]=="ε-SVM模型"
    assert p["slots"][0]["expected"] in evidence[0]["text"]
    assert not mapped_proxy(raw,checks,q,[dict(eid="C1",text="没有对应证据")])["basis_complete"]


def test_mapping_does_not_relax_numeric_or_wrong_greek_values():
    checks=[dict(slot_checks=[dict(index=1,supported="1",unique="1",aliases_valid="1")])]*2
    q=dict(requested_slots=[dict(index=1,label="数值",query_quote="数值",type="number")])
    raw=dict(slots=[dict(index=1,supported="1",expected="0.3",unit="",aliases=[],eid="C1",quote="该模型参数被设定为3。")])
    p=mapped_proxy(raw,checks,q,[dict(eid="C1",text=raw["slots"][0]["quote"])])
    assert not p["basis_complete"] and not p["source_mappings"]
