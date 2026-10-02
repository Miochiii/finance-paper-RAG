from rag_objective_admission_guard import blind_source_payload, source_reference_consistency, scalar_equal


def fixture(label="持股时间",value="5",extracted="5个交易日",supported="1"):
    slot=dict(index=1,label=label,query_quote=label,type="number",expected=value,unit="个交易日",
              aliases=[],eid="E1",quote="| 持股时间 | 5个交易日 |")
    candidate=dict(question="请给出"+label,slots=[slot])
    raw=dict(slots=[dict(index=1,supported=supported,expected=extracted,unit="",aliases=[],eid="E1",quote=slot["quote"])])
    checks=[dict(slot_checks=[dict(index=1,supported="1",unique="1",aliases_valid="1")]) for _ in range(2)]
    return candidate,raw,checks,[dict(eid="E1",text=slot["quote"])]


def test_blind_payload_contains_only_public_question_fields_and_source():
    candidate,_,_,evidence=fixture()
    payload=blind_source_payload(candidate,evidence)
    assert set(payload)=={"question","requested_slots","retrieved_evidence"}
    assert set(payload["requested_slots"][0])=={"index","label","query_quote","type"}


def test_unit_split_can_pass_but_reference_cannot_rescue_missing_source():
    candidate,raw,checks,evidence=fixture()
    assert source_reference_consistency(candidate,raw,checks,evidence)["admit"]
    raw["slots"][0].update(supported="U",expected="",quote="")
    result=source_reference_consistency(candidate,raw,checks,evidence)
    assert not result["admit"] and not result["plan"]["basis_complete"]


def test_price_field_time_unit_is_rejected_even_with_agreeing_extraction():
    result=source_reference_consistency(*fixture(label="交易日后价格"))
    assert not result["admit"]
    assert "price_field_reference_uses_time_unit" in {r["reason"] for r in result["reasons"]}


def test_reference_disagreement_is_not_changed_to_match():
    result=source_reference_consistency(*fixture(value="10"))
    assert not result["admit"] and result["plan"]["slots"][0]["expected"]=="5"


def test_scalar_comparison_preserves_percent_magnitude_and_unit_semantics():
    assert scalar_equal("5.0%","5％","number")
    assert not scalar_equal("5%",".05","number")
    assert not scalar_equal("LightGBM","LGBM","entity")


def test_percent_unit_representation_is_compared_once():
    candidate,raw,checks,evidence=fixture()
    candidate["slots"][0].update(label="年化收益",expected="9.07%",unit="%")
    raw["slots"][0].update(expected="9.07%",quote="| 策略年化收益 | 9.07% |")
    evidence[0]["text"]=raw["slots"][0]["quote"]
    assert source_reference_consistency(candidate,raw,checks,evidence)["admit"]
    candidate["slots"][0].update(expected=".0907",unit="")
    assert not source_reference_consistency(candidate,raw,checks,evidence)["admit"]
