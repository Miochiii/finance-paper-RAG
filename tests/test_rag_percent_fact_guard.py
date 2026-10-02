from rag_percent_fact_guard import normalize_percent_units,fact_projection_issues,percent_unit_proxy


def test_duplicate_percent_unit_drops_once_with_literal_source_proof():
    req=[dict(index=1,label="年化收益",query_quote="年化收益",type="number")]
    raw=dict(slots=[dict(index=1,supported="1",expected="9.07%",unit="%",aliases=[],eid="E1",quote="| 策略年化收益 | 9.07% |")])
    evidence=[dict(eid="E1",text=raw["slots"][0]["quote"])]
    changed,mapping=normalize_percent_units(raw,req,evidence)
    assert changed["slots"][0]["expected"]=="9.07%" and changed["slots"][0]["unit"]==""
    assert raw["slots"][0]["unit"]=="%" and len(mapping)==1
    checks=[dict(slot_checks=[dict(index=1,supported="1",unique="1",aliases_valid="1")]) for _ in range(2)]
    plan=percent_unit_proxy(raw,checks,dict(question="年化收益？",requested_slots=req),evidence)
    assert plan["basis_complete"] and plan["facts"][0]["point"]=="年化收益：9.07%"


def test_fabricated_or_unsupported_percent_value_is_not_repaired():
    req=[dict(index=1,type="number")]
    raw=dict(slots=[dict(index=1,supported="1",expected="19.07%",unit="%",eid="E1",quote="| 策略年化收益 | 9.07% |")])
    evidence=[dict(eid="E1",text=raw["slots"][0]["quote"])]
    changed,mapping=normalize_percent_units(raw,req,evidence)
    assert changed==raw and not mapping
    raw["slots"][0].update(expected="9.07%",supported="U")
    assert normalize_percent_units(raw,req,evidence)==(raw,[])


def test_preflight_detects_duplicate_in_reference_or_proxy_even_if_rendered_once():
    fact=dict(index=1,label="收益",type="number",expected="9.07%",unit="%",point="收益：9.07%")
    assert fact_projection_issues({"Q":dict(facts=[fact])},{})[0]["reason"]=="duplicated_percent_unit"
    fact["unit"]="";fact["point"]="收益：9.07%%"
    assert fact_projection_issues({}, {"Q":dict(facts=[fact])})[0]["reason"]=="numeric_point_projection_mismatch"
    fact["point"]="收益：9.07%"
    assert not fact_projection_issues({}, {"Q":dict(facts=[fact])})


def test_generation_never_runs_after_failed_preflight(monkeypatch):
    import pytest
    import rag_guarded_percent_validation as validation
    generated=[]
    monkeypatch.setattr(validation,"preflight",lambda output:dict(status="failed"))
    monkeypatch.setattr(validation.experiment,"generate",lambda output:generated.append(output))
    with pytest.raises(ValueError,match="preflight not passed"):validation.generate(None)
    assert not generated
