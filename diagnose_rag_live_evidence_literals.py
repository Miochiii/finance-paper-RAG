"""Posthoc labeled numeric magnitude check; not an independent source truth.

Preserve whitespace so the next item's index cannot join the preceding value.
This adds a diagnostic and never changes the frozen primary scoring or gate.
"""
import argparse
import re
import unicodedata
from decimal import Decimal
from pathlib import Path

from rag_fresh_change_experiment import load, dump, utc


def labeled_number(answer, label):
    text = unicodedata.normalize("NFKC", answer)
    pattern = re.escape(unicodedata.normalize("NFKC", label)).replace(r"\ ", r"\s*")
    hit = re.search(pattern + r"\s*:\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:%)?)", text)
    return hit.group(1) if hit else None


def diagnose(output):
    refs = load(output / "private_references.json")
    faults = load(output / "private_fault_interventions.json")
    rows = []
    for aid, key in load(output / "private_case_key.json").items():
        qid = key["question_key"]
        answer = load(output / "generations" / (aid + ".json"))["answer"]
        slots = []
        for slot in refs[qid]["slots"]:
            if slot["type"] != "number":
                continue
            observed = labeled_number(answer, slot["label"])
            wanted = (faults[qid]["new_value"] if key["condition"] == "live_corrupted" and slot["index"] == 1
                      else slot["expected"])
            match = observed is not None and Decimal(observed.rstrip("%％")) == Decimal(wanted.rstrip("%％"))
            slots.append(dict(index=slot["index"], label=slot["label"], observed=observed,
                              source_normal=slot["expected"], expected_for_condition=wanted,
                              displayed_magnitude_matches=match))
        rows.append(dict(aid=aid, **key, slots=slots,
                         all_displayed_magnitudes_match=all(s["displayed_magnitude_matches"] for s in slots)))
    summary = dict(checked_at=utc(), answers=len(rows),
        all_magnitudes_match=sum(r["all_displayed_magnitudes_match"] for r in rows),
        normal_target_matches=sum(r["condition"] == "live_normal" and r["slots"][0]["displayed_magnitude_matches"] for r in rows),
        fault_target_matches=sum(r["condition"] == "live_corrupted" and r["slots"][0]["displayed_magnitude_matches"] for r in rows),
        other_numeric_fields_unchanged=sum(all(s["displayed_magnitude_matches"] for s in r["slots"][1:]) for r in rows),
        no_additional_api=True, whitespace_and_line_boundaries_preserved=True,
        scope="posthoc_deterministic_labeled_displayed_magnitude_check_not_unit_semantics_or_independent_source_truth",
        primary_scores_unchanged=True, first_attempt_parser_error_retained=True)
    dump(output / "private_literal_diagnostic.json", dict(summary=summary, rows=rows))
    dump(output / "literal_diagnostic_summary.json", summary)
    print(summary)
    if summary["all_magnitudes_match"] != len(rows):
        raise ValueError("literal diagnostic needs investigation; primary results unchanged")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    diagnose(ap.parse_args().output)
