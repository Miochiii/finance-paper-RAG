from rag_objective_slots import (validate_candidate,checks_pass,public_slots,controls,literal_anchor,
                                schema_valid,agreement_score,comparison_payload)


def candidate():
    return dict(question="模型的深度与树数分别是多少？",slots=[
       dict(index=1,label="深度",query_quote="深度",type="number",expected="3",unit="",aliases=[],eid="E1",quote="模型的深度为3，树数为100。"),
       dict(index=2,label="树数",query_quote="树数",type="number",expected="100",unit="",aliases=[],eid="E1",quote="模型的深度为3，树数为100。")])


def test_candidate_support_request_and_pending():
    evidence=[dict(eid="E1",text="文中：模型的深度为3，树数为100。")]
    v=validate_candidate(dict(candidate=candidate()),evidence,[])
    assert v["valid"] and v["candidate"]["status"]=="pending"
    s=v["candidate"]["slots"][0];span=s["source_span"]
    assert evidence[0]["text"][span["start"]:span["end"]]==s["quote"]
    c=candidate();c["slots"][0]["expected"]="4"
    assert not validate_candidate(dict(candidate=c),evidence,[])["valid"]
    c=candidate();c["slots"][0]["query_quote"]="准确率"
    assert not validate_candidate(dict(candidate=c),evidence,[])["valid"]
    assert not validate_candidate(dict(candidate=candidate()),evidence,[candidate()["question"]])["valid"]


def test_controls_change_one_mapping_and_have_known_quality():
    c=controls(candidate())
    assert c["complete"]["answer"]=="深度：3\n树数：100"
    assert c["omitted"]["answer"]=="树数：100" and c["omitted"]["quality"]=="1"
    assert c["corrupted"]["answer"]=="深度：4\n树数：100" and c["corrupted"]["quality"]=="0"
    row=candidate();row["slots"][0].update(expected="21.93%")
    assert "22.93%" in controls(row)["corrupted"]["answer"]


def test_public_projection_excludes_reference_values_and_quotes():
    public=public_slots(candidate())
    assert all(set(s)=={"index","label","query_quote","type"} for s in public)
    assert "expected" not in str(public) and "100" not in str(public)


def test_two_source_checks_and_fixed_indices_required():
    rows=[dict(index=i,supported="1",unique="1",aliases_valid="1") for i in (1,2)]
    assert checks_pass(dict(slot_checks=rows),2,("supported","unique","aliases_valid"))
    rows[1]["unique"]="U"
    assert not checks_pass(dict(slot_checks=rows),2,("supported","unique","aliases_valid"))
    assert not schema_valid(dict(point_results=[dict(index=1),dict(index=1)]),2)


def test_format_does_not_bypass_short_entity_boundary_or_numeric_rule():
    assert literal_anchor("指标为**AUC**。","**AUC**")["ok"]
    assert not literal_anchor("**AUC**出现，**AUC**重复。","**AUC**")["ok"]
    assert not literal_anchor("参数为**10**。","**10**")["ok"]
    assert not literal_anchor("乘积x*y*z=6。","xyz=6")["ok"]


def test_answer_quotes_and_disagreement_keep_u():
    covered=dict(point_results=[dict(index=1,status="covered",answer_quote="深度：3",missing_elements=[])])
    wrong=dict(point_results=[dict(index=1,status="contradicted",answer_quote="深度：3",missing_elements=[])])
    assert agreement_score([covered,covered],1,"深度：3",True)["status"]=="complete"
    assert agreement_score([covered,wrong],1,"深度：3",True)["status"]=="U"
    covered["point_results"][0]["answer_quote"]="深度：4"
    assert agreement_score([covered,covered],1,"深度：3",True)["status"]=="U"
    payload=comparison_payload("问题","回答",[dict(index=1,expected="3")])
    assert set(payload)=={"question","answer_to_check","facts","required_indices"}
