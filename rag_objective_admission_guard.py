"""Conservative development guard before test-answer generation.

First extract from source cards without references or answers; only afterwards
compare the validated extraction with the proposed reference. Same-provider
agreement remains an automatic consistency check, not independent truth.
"""
from decimal import Decimal
import re

from rag_numeric_unit_mapping import numeric_unit_proxy, numeric_signature
from rag_objective_slots import public_slots


TIME_UNIT = re.compile(r"(?:个)?(?:交易日|工作日|月)|年|日|天|小时|分钟|秒")


def blind_source_payload(candidate, evidence):
    """Allowlist public fields: expected values, units and gold quotes excluded."""
    return dict(question=candidate["question"], requested_slots=public_slots(candidate),
                retrieved_evidence=evidence)


def field_unit_conflicts(candidate):
    """A narrow price-versus-time alarm; absence is not a semantic certification."""
    return [dict(index=s["index"], reason="price_field_reference_uses_time_unit")
            for s in candidate["slots"]
            if s["type"] == "number" and re.search(r"(?:价格|股价|收盘价)$", s["label"])
            and TIME_UNIT.fullmatch(s["unit"])]


def scalar_equal(left, right, kind):
    if kind == "entity":
        return left == right
    if left.endswith(("%", "％")) != right.endswith(("%", "％")):
        return False
    try:
        return Decimal(left.rstrip("%％")) == Decimal(right.rstrip("%％"))
    except Exception:
        return False


def source_reference_consistency(candidate, blind_raw, blind_checks, evidence):
    """Decision after blind source extraction and two source checks have finished.

    No rescue using the proposed reference. Reject/U items stay out of the next
    experimental bank and are reported in the admission denominator.
    """
    question = dict(question=candidate["question"], requested_slots=public_slots(candidate))
    plan = numeric_unit_proxy(blind_raw, blind_checks, question, evidence)
    conflicts = field_unit_conflicts(candidate)
    reasons = list(conflicts)
    if not plan["basis_complete"]:
        reasons.append(dict(index=None, reason="blind_source_basis_unresolved"))
    else:
        by_index = {s["index"]: s for s in plan["slots"]}
        for reference in candidate["slots"]:
            extracted = by_index[reference["index"]]
            if reference["type"] == "number":
                try:
                    left = numeric_signature(reference["expected"], reference["unit"])
                    right = numeric_signature(extracted["expected"], extracted["unit"])
                except ValueError:
                    reasons.append(dict(index=reference["index"], reason="invalid_scalar_unit_representation"))
                    continue
                value_equal, unit_equal = left[0] == right[0], left[1] == right[1]
            else:
                value_equal = reference["expected"] == extracted["expected"]
                unit_equal = reference["unit"] == extracted["unit"]
            if not value_equal:
                reasons.append(dict(index=reference["index"], reason="reference_and_blind_value_disagree"))
            if not unit_equal:
                reasons.append(dict(index=reference["index"], reason="reference_and_blind_unit_disagree"))
    return dict(admit=not reasons, reasons=reasons, plan=plan,
                scope="automatic_source_consistency_not_independent_truth")
