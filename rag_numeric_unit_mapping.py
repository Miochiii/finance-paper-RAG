"""Development-only scalar/unit separation with an exact source trace.

No unit conversion, rounding, synonym substitution or unsupported-value repair.
Callers keep frozen historical plans unchanged and revalidate with fresh cases.
"""
import copy
import re
from decimal import Decimal

from rag_objective_slots import exact_source_quote, numeric_value
from rag_field_retrieval_trial import mapped_proxy


UNITS = (
    "个交易日", "交易日", "个工作日", "工作日", "个月", "年", "月", "日", "天",
    "小时", "分钟", "秒", "万元", "亿元", "元", "万股", "股", "个", "次", "倍",
    "ms", "s", "MB", "GB", "bps",
)
NUMBER_WITH_UNIT = re.compile(
    r"(?P<number>[+-]?(?:\d+(?:\.\d+)?|\.\d+))(?P<space>\s*)(?P<unit>"
    + "|".join(re.escape(u) for u in UNITS) + r")"
)


def numeric_signature(value, unit):
    """Equivalent representations only; 5% never becomes .05 without a unit.

    A percent suffix and a separate percent unit describe one marker. A percent
    suffix with another unit is rejected. The numeric magnitude is unchanged.
    """
    if not isinstance(value, str) or not numeric_value(value) or not isinstance(unit, str):
        raise ValueError("invalid numeric scalar/unit")
    percent = value.endswith(("%", "％"))
    if percent and unit not in ("", "%", "％"):
        raise ValueError("percent suffix conflicts with explicit unit")
    canonical_unit = "%" if percent or unit in ("%", "％") else unit
    return Decimal(value.rstrip("%％")), canonical_unit


def render_numeric(value, unit):
    """For future controls only: append an explicit unit once."""
    numeric_signature(value, unit)
    return value if value.endswith(("%", "％")) else value + unit


def separate_numeric_units(raw, requested, evidence):
    """Change only supported numeric fields with a known literal unit suffix.

    The raw value and its unit must occur together in the exact cited source
    quotation. Conflicting existing units, approximate/range/multi-value strings,
    entities and unsupported fields are preserved without repair.
    """
    changed = copy.deepcopy(raw)
    mappings = []
    rejected = []
    types = {s["index"]: s["type"] for s in requested}
    by_eid = {e["eid"]: e["text"] for e in evidence}
    rows = changed.get("slots", [])
    if not isinstance(rows, list):
        return changed, mappings, [dict(index=None, reason="invalid_slots")]
    for row in rows:
        if not isinstance(row, dict):
            continue
        index = row.get("index")
        value = row.get("expected")
        if types.get(index) != "number" or row.get("supported") != "1" or not isinstance(value, str):
            continue
        if numeric_value(value):
            continue
        match = NUMBER_WITH_UNIT.fullmatch(value)
        if not match:
            rejected.append(dict(index=index, reason="not_single_scalar_with_known_literal_unit"))
            continue
        number, unit = match["number"], match["unit"]
        existing = row.get("unit")
        if not isinstance(existing, str) or (existing and existing != unit):
            rejected.append(dict(index=index, reason="conflicting_or_invalid_unit"))
            continue
        text = by_eid.get(row.get("eid"), "")
        quote = row.get("quote")
        anchor = exact_source_quote(quote, text) if isinstance(quote, str) else None
        if not anchor or value not in anchor["quote"]:
            rejected.append(dict(index=index, reason="combined_value_not_literal_cited_source"))
            continue
        offset = anchor["quote"].index(value)
        start = anchor["start"] + offset
        end = start + len(value)
        if text[start:end] != value:
            raise ValueError("numeric/unit source trace changed")
        mappings.append(dict(index=index, eid=row["eid"], original_expected=value,
                             original_unit=existing, expected=number, unit=unit,
                             start=start, end=end, literal=text[start:end],
                             rule="single_number_exact_literal_unit_suffix_no_value_conversion"))
        row.update(expected=number, unit=unit)
    return changed, mappings, rejected


def numeric_unit_proxy(raw, checks, question, evidence):
    changed, mappings, rejected = separate_numeric_units(raw, question["requested_slots"], evidence)
    plan = mapped_proxy(changed, checks, question, evidence)
    return dict(**plan, numeric_unit_mappings=mappings, numeric_unit_rejections=rejected,
                original_basis_complete=mapped_proxy(raw, checks, question, evidence)["basis_complete"])
