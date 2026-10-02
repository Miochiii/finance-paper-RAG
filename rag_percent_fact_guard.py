"""Strict percent projection and pre-generation invariant checks.

Drop only a duplicated unit marker that is already represented in a supported
numeric value with a literal source anchor. Do not convert numeric magnitudes.
"""
import copy

import rag_auto_evaluation as auto
from rag_objective_slots import numeric_value,exact_source_quote
from rag_numeric_unit_mapping import numeric_unit_proxy,render_numeric
from rag_objective_admission_guard import source_reference_consistency


def normalize_percent_units(raw,requested,evidence):
    changed=copy.deepcopy(raw);mappings=[]
    types={s["index"]:s["type"] for s in requested};by={e["eid"]:e["text"] for e in evidence}
    rows=changed.get("slots",[])
    if not isinstance(rows,list):return changed,mappings
    for row in rows:
        if not isinstance(row,dict):continue
        value=row.get("expected")
        if types.get(row.get("index"))!="number" or row.get("supported")!="1" or not isinstance(value,str):continue
        if not numeric_value(value) or not value.endswith(("%","％")) or row.get("unit") not in ("%","％"):continue
        quote=row.get("quote")
        anchor=exact_source_quote(quote,by.get(row.get("eid"),"")) if isinstance(quote,str) else None
        if not anchor or auto.norm(value) not in auto.norm(anchor["quote"]):continue
        mappings.append(dict(index=row["index"],eid=row["eid"],original_unit=row["unit"],unit="",
            unchanged_expected=value,source_span=anchor,rule="drop_duplicate_percent_unit_already_in_literal_numeric_value"))
        row["unit"]=""
    return changed,mappings


def percent_unit_proxy(raw,checks,question,evidence):
    changed,mappings=normalize_percent_units(raw,question["requested_slots"],evidence)
    return dict(**numeric_unit_proxy(changed,checks,question,evidence),percent_unit_mappings=mappings)


def percent_source_consistency(candidate,raw,checks,evidence):
    requested=[{k:s[k] for k in ("index","label","query_quote","type")} for s in candidate["slots"]]
    changed,mappings=normalize_percent_units(raw,requested,evidence)
    return dict(**source_reference_consistency(candidate,changed,checks,evidence),percent_unit_mappings=mappings)


def fact_projection_issues(references,plans):
    issues=[]
    for name,bank in (("reference",references),("proxy",plans)):
        for qid,plan in bank.items():
            for fact in plan.get("facts",[]):
                if fact.get("type")!="number":continue
                value,unit=fact.get("expected"),fact.get("unit")
                try:correct=f'{fact["label"]}：{render_numeric(value,unit)}'
                except (ValueError,KeyError):
                    issues.append(dict(question_key=qid,basis=name,index=fact.get("index"),reason="invalid_numeric_unit"));continue
                if value.endswith(("%","％")) and unit in ("%","％"):
                    issues.append(dict(question_key=qid,basis=name,index=fact["index"],reason="duplicated_percent_unit"))
                if fact["point"]!=correct:
                    issues.append(dict(question_key=qid,basis=name,index=fact["index"],reason="numeric_point_projection_mismatch"))
    return issues
