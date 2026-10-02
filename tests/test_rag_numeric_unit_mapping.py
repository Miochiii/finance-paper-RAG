import copy

import pytest

from rag_numeric_unit_mapping import separate_numeric_units, numeric_unit_proxy, numeric_signature, render_numeric


def setup(value="5个交易日", unit="", supported="1", kind="number", quote=None):
    text="| 持股时间 | 5个交易日 |\n| 回测样本 | 100个 |"
    raw=dict(slots=[dict(index=1,supported=supported,expected=value,unit=unit,aliases=[],eid="E1",
                        quote=quote or "| 持股时间 | 5个交易日 |")])
    req=[dict(index=1,label="持股时间",query_quote="持股时间",type=kind)]
    checks=[dict(slot_checks=[dict(index=1,supported="1",unique="1",aliases_valid="1")]) for _ in range(2)]
    return raw,req,[dict(eid="E1",text=text)],checks


def test_exact_suffix_split_is_nonmutating_and_preserves_source_span():
    raw,req,evidence,checks=setup();before=copy.deepcopy(raw)
    plan=numeric_unit_proxy(raw,checks,dict(question="持股时间？",requested_slots=req),evidence)
    assert raw==before and not plan["original_basis_complete"] and plan["basis_complete"]
    assert (plan["slots"][0]["expected"],plan["slots"][0]["unit"])==("5","个交易日")
    mapping=plan["numeric_unit_mappings"][0]
    assert evidence[0]["text"][mapping["start"]:mapping["end"]]=="5个交易日"


@pytest.mark.parametrize("value,unit",[("约5个交易日",""),("5-10个交易日",""),("5或10个交易日",""),
                                      ("5个交易日","天"),("5千个交易日",""),("100个","")])
def test_ranges_approximations_conflicts_and_uncited_values_are_not_repaired(value,unit):
    raw,req,evidence,_=setup(value,unit)
    changed,mappings,rejected=separate_numeric_units(raw,req,evidence)
    assert changed==raw and not mappings and rejected


@pytest.mark.parametrize("supported,kind",[("U","number"),("1","entity")])
def test_unsupported_and_entity_slots_are_not_repaired(supported,kind):
    raw,req,evidence,_=setup(supported=supported,kind=kind)
    changed,mappings,_=separate_numeric_units(raw,req,evidence)
    assert changed==raw and not mappings


def test_plain_numeric_percent_is_preserved():
    raw,req,evidence,_=setup(value="5%")
    changed,mappings,_=separate_numeric_units(raw,req,evidence)
    assert changed==raw and not mappings


def test_percent_forms_are_equivalent_but_not_converted_or_duplicated():
    assert numeric_signature("9.07%","%")==numeric_signature("9.070","％")
    assert numeric_signature("5%","")!=numeric_signature(".05","")
    assert render_numeric("9.07%","%")=="9.07%"
    with pytest.raises(ValueError):numeric_signature("5%","元")
