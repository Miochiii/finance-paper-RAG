from rag_objective_coverage_pilot import proxy_plan,metrics


def test_proxy_requires_all_explicit_slots_supported_twice():
    q=dict(requested_slots=[dict(index=1,label="深度",query_quote="深度",type="number"),dict(index=2,label="树数",query_quote="树数",type="number")])
    evidence=[dict(eid="C1",text="该模型参数为深度3，树数100。")]
    raw=dict(slots=[dict(index=i,supported="1",expected=v,unit="",aliases=[],eid="C1",quote=evidence[0]["text"]) for i,v in ((1,"3"),(2,"100"))])
    checks=[dict(slot_checks=[dict(index=i,supported="1",unique="1",aliases_valid="1") for i in (1,2)]) for _ in (1,2)]
    assert proxy_plan(raw,checks,q,evidence)["basis_complete"]
    raw["slots"][1]["expected"]="200"
    assert not proxy_plan(raw,checks,q,evidence)["basis_complete"]
    raw["slots"]=raw["slots"][:1]
    assert not proxy_plan(raw,checks,q,evidence)["basis_complete"]


def test_proxy_source_check_disagreement_is_unresolved():
    q=dict(requested_slots=[dict(index=1,label="参数",query_quote="参数",type="number")])
    raw=dict(slots=[dict(index=1,supported="1",expected="100",unit="",aliases=[],eid="C1",quote="本模型的参数取值为100。")])
    evidence=[dict(eid="C1",text="本模型的参数取值为100。")]
    good=dict(slot_checks=[dict(index=1,supported="1",unique="1",aliases_valid="1")])
    uncertain=dict(slot_checks=[dict(index=1,supported="U",unique="1",aliases_valid="1")])
    assert not proxy_plan(raw,[good,uncertain],q,evidence)["basis_complete"]


def test_unknown_controls_count_as_unflagged_and_good_u_is_visible():
    rows=[dict(quality="1",proxy_status="U",rule_refusal=0),dict(quality="2",proxy_status="U",rule_refusal=0)]
    m=metrics(rows)
    assert m["n"]==2 and m["proxy_u"]==2 and m["flagged_incomplete"]==0 and m["unresolved_incomplete"]==1
