"""V9: fresh paired generations under a source-anchored numeric evidence fault.

Cached normal retrieval and answer-hidden normal monitoring plans stay fixed.
Qualification uses evidence only, before test answers; all failures are retained.
"""
import argparse
import copy
import json
import random
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path

import rag_auto_evaluation as auto
import rag_field_retrieval_trial as experiment
from rag_fresh_change_experiment import load, dump, digest, sha, utc, write_csv
from rag_prospective_coverage_validation import verify_hashes
from rag_subtle_fault_pilot import normalized_map, batch
from rag_numeric_unit_mapping import numeric_signature
from rag_percent_fact_guard import percent_unit_proxy, fact_projection_issues
from rag_objective_slots import EXTRACT_SLOTS, CHECK_SLOTS, numeric_value
from rag_mineru_coverage_v4 import source_context, evidence_projection

NUMBER = re.compile(r"(?<![\d.])[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[%％])?(?![\d.])")
CHECK_SUFFIX = {1: "\n核对字段与值的对应关系。", 2: "\n核对实验口径和别名是否有来源支持。"}
MODULES = ["rag_live_evidence_pairs.py", "audit_rag_live_evidence_pairs.py",
           "report_rag_live_evidence_pairs.py", "tests/test_rag_live_evidence_pairs.py"]
COMMON_MODULES = ["rag_field_retrieval_trial.py", "rag_auto_evaluation.py", "rag_objective_coverage_pilot.py",
                  "rag_objective_slots.py", "rag_percent_fact_guard.py", "rag_numeric_unit_mapping.py",
                  "rag_objective_admission_guard.py", "rag_mineru_coverage_v4.py", "rag_subtle_fault_pilot.py",
                  "rag_fresh_change_experiment.py", "rag_prospective_coverage_validation.py", "rag_answer_quote_v4.py",
                  "rag_refusal.py", "rag_field_refusal.py", "rag_semantic_coverage_pilot.py", "evaluate.py"]


def display_tick(value):
    """Increment one least displayed decimal place, retaining the percent marker."""
    if not numeric_value(value):
        raise ValueError("target must be a numeric scalar")
    mantissa = value.rstrip("%％")
    places = len(mantissa.split(".")[1]) if "." in mantissa else 0
    delta = Decimal(1).scaleb(-places)
    return format(Decimal(mantissa) + delta, f".{places}f") + value[len(mantissa):]


def value_spans(text, value):
    """Match complete literal magnitudes, also finding a following percent marker.

    No substring match in another number, no sign stripping, no rounding.
    Unit identity is checked separately when qualifying extracted fields.
    """
    wanted = value.rstrip("%％")
    return [(m.start(), m.start() + len(wanted)) for m in NUMBER.finditer(text)
            if m.group().rstrip("%％") == wanted]


def corrupt_evidence(retrieval, plan):
    """Edit all exact copies of one cited numeric target; reject residual old values.

    The first requested field is fixed in advance. The cited row must contain
    exactly one occurrence. Extra old-value occurrences, including unrelated
    coincidental values, cause conservative rejection instead of broad edits.
    """
    first = plan["slots"][0]
    if first["index"] != 1 or first["type"] != "number":
        return dict(structurally_valid=False, reason="first_slot_not_numeric")
    old = first["expected"]
    new = display_tick(old)
    quote = first["quote"]
    nq, qp = normalized_map(quote)
    targets = value_spans(nq, old)
    if len(targets) != 1:
        return dict(structurally_valid=False, reason="target_not_unique_in_cited_quote")
    target_start, target_end = targets[0]
    altered = copy.deepcopy(retrieval["retrieved"])
    edits = []
    for i, passage in enumerate(altered):
        text = passage["text"]
        nt, positions = normalized_map(text)
        spans = set()
        for match in re.finditer(re.escape(nq), nt):
            a = positions[match.start() + target_start]
            b = positions[match.start() + target_end - 1] + 1
            if text[a:b] != old.rstrip("%％"):
                return dict(structurally_valid=False, reason="target_presentation_not_exact")
            spans.add((a, b))
        for a, b in sorted(spans):
            edits.append(dict(eid=f"C{i+1:04d}", start=a, end=b, old=text[a:b],
                              new=new.rstrip("%％"), source=passage["source"],
                              original_source_start=passage["start"] + a,
                              original_source_end=passage["start"] + b))
        for a, b in sorted(spans, reverse=True):
            text = text[:a] + new.rstrip("%％") + text[b:]
        passage["text"] = text
        # These spans describe the unmodified source; new text is a private fault.
        passage["original_text_sha256"] = passage.get("text_sha256", sha(retrieval["retrieved"][i]["text"]))
        passage["text_sha256"] = sha(text)
    remaining = [dict(eid=f"C{i+1:04d}", start=a, end=b)
                 for i, row in enumerate(altered) for a, b in value_spans(row["text"], old)]
    valid = bool(edits) and not remaining
    return dict(structurally_valid=valid,
                reason="passed" if valid else "residual_original_value" if remaining else "no_cited_quote_copy",
                target_index=1, old_value=old, new_value=new, unit=first["unit"],
                edits=edits, residual_original_value_spans=remaining,
                retrieved=altered, evidence=evidence_projection(altered), context=source_context(altered))


def same_slot(left, right):
    if left["type"] != right["type"] or left["index"] != right["index"]:
        return False
    if left["type"] == "number":
        try:
            return numeric_signature(left["expected"], left["unit"]) == numeric_signature(right["expected"], right["unit"])
        except ValueError:
            return False
    return auto.norm(left["expected"]) == auto.norm(right["expected"])


def qualify_fault(fault, normal_plan, fault_plan):
    checks = dict(structural=fault["structurally_valid"],
                  fault_basis_complete=bool(fault_plan.get("basis_complete")),
                  exact_single_field_change=False, valid_numeric_projection=False)
    if checks["structural"] and checks["fault_basis_complete"]:
        normal = normal_plan["slots"]
        changed = fault_plan["slots"]
        target = dict(normal[0], expected=fault["new_value"])
        checks["exact_single_field_change"] = (len(normal) == len(changed) and same_slot(target, changed[0])
                                                and all(same_slot(a, b) for a, b in zip(normal[1:], changed[1:])))
        checks["valid_numeric_projection"] = not fact_projection_issues({}, {"fault": fault_plan})
    return dict(qualified=all(checks.values()), checks=checks)


def prepare(parent, output):
    if (output / "protocol_lock.json").exists():
        verify(output)
        return
    manifest = load(parent / "artifact_manifest.json")
    verify_hashes(manifest["output_hashes"])
    verify_hashes(manifest["code_hashes"])
    previous = load(parent / "protocol_lock.json")
    verify_hashes(previous["source_hashes"])
    questions = load(parent / "table_axis_clarification_20261002/next_pair_public_questions.json")
    references = load(parent / "private_references.json")
    retrieval = load(parent / "public_retrieval.json")
    output.mkdir(parents=True, exist_ok=True)
    plans = {}
    inputs = [parent / n for n in ("artifact_manifest.json", "protocol_lock.json", "private_references.json",
                                   "public_retrieval.json", "table_axis_clarification_20261002/next_pair_public_questions.json")]
    for qid, question in questions.items():
        path = (parent / "table_axis_clarification_20261002/proxy_plans/clarified.json" if qid == "P0002"
                else parent / "proxy_plans" / (qid + ".json"))
        inputs.append(path)
        plans[qid] = load(path)
        if plans[qid]["question"] != question["question"] or not plans[qid]["basis_complete"]:
            raise ValueError("normal source plan does not match public question")
        retrieval[qid] = dict(retrieval[qid], **question)
        dump(output / "proxy_plans" / (qid + ".json"), plans[qid])
    if fact_projection_issues(references, plans):
        raise ValueError("input fact projection invalid")
    for name, data in (("public_questions.json", questions), ("private_references.json", references),
                       ("public_retrieval.json", retrieval), ("source_manifest.json", load(parent / "source_manifest.json"))):
        dump(output / name, data)
    code = [Path(__file__).parent / n for n in MODULES + COMMON_MODULES]
    dump(output / "protocol_lock.json", dict(version="live_numeric_evidence_pairs_v9", locked_at=utc(),
         previous=str(parent.resolve()), input_hashes={str(p.resolve()): digest(p) for p in inputs + code},
         source_hashes=previous["source_hashes"], config=dict(previous["config"], seed=2026100209),
         interpretation="seen_questions_existing_sources_cached_normal_context_same_vendor_automatic_truth",
         maximum_new_generations=2*len(questions), fault_rule="first_numeric_slot_plus_one_least_displayed_decimal_place",
         qualification="exact_cited_quote_copies_only_no_old_magnitude_residual_then_blind_extraction_two_source_checks",
         minimum_pairs=8, minimum_sources=6, all_preanswer_failures_retained=True,
         monitor_basis="frozen_unmodified_normal_retrieval_no_fault_plan_or_reference_in_monitor",
         gate=dict(minimum_normal_complete_fraction=.8, minimum_paired_complete_to_bad_fraction=.6,
                   minimum_bad_answer_flag_fraction=.8, maximum_normal_complete_false_flags=0,
                   maximum_quality_u_fraction=.1), no_answer_based_selection=True))
    faults = {qid: corrupt_evidence(retrieval[qid], plans[qid]) for qid in questions}
    dump(output / "private_fault_interventions.json", faults)
    files = [output / n for n in ("public_questions.json", "private_references.json", "public_retrieval.json",
                                  "source_manifest.json", "private_fault_interventions.json")]
    files.extend(sorted((output / "proxy_plans").glob("*.json")))
    dump(output / "preparation_lock.json", dict(locked_at=utc(), hashes={str(p.resolve()): digest(p) for p in files}))
    print("V9 prepared:", len(questions), "input questions; structurally eligible",
          sum(f["structurally_valid"] for f in faults.values()), "before any test answers", flush=True)


def verify(output):
    m = experiment.verify(output)
    verify_hashes(load(output / "preparation_lock.json")["hashes"])
    return m


def check_faults(output):
    m = verify(output)
    if (output / "intervention_check_lock.json").exists():
        verify_hashes(load(output / "intervention_check_lock.json")["hashes"])
        return
    if list((output / "generations").glob("*.json")):
        raise ValueError("cannot qualify after test answers")
    questions = load(output / "public_questions.json")
    faults = load(output / "private_fault_interventions.json")
    def one(item):
        qid, fault = item
        q = questions[qid]
        payload = dict(question=q["question"], requested_slots=q["requested_slots"], retrieved_evidence=fault["evidence"])
        raw = auto.request(output, "fault_extract_" + qid, EXTRACT_SLOTS, payload)
        checks = [auto.request(output, f"fault_check_{qid}_r{n}", CHECK_SLOTS + CHECK_SUFFIX[n],
                  dict(**payload, temporary_slots=raw.get("slots", [])), n) for n in (1, 2)]
        plan = dict(question=q["question"], evidence=fault["evidence"], raw=raw, checks=checks,
                    **percent_unit_proxy(raw, checks, q, fault["evidence"]))
        normal = load(output / "proxy_plans" / (qid + ".json"))
        result = qualify_fault(fault, normal, plan)
        dump(output / "fault_plans" / (qid + ".json"), dict(plan=plan, **result, finished_at=utc()))
        return result
    batch([(q, f) for q, f in faults.items() if f["structurally_valid"]], one, m["config"]["workers"], "V9_FAULT_CHECK")
    screen = []
    for qid, fault in faults.items():
        check = load(output / "fault_plans" / (qid + ".json")) if fault["structurally_valid"] else None
        screen.append(dict(question_key=qid, source=questions[qid]["sources"][0], structural=int(fault["structurally_valid"]),
                           structural_reason=fault["reason"], qualified=int(bool(check and check["qualified"])),
                           changed_copies=len(fault.get("edits", [])), residual_values=len(fault.get("residual_original_value_spans", []))))
    dump(output / "intervention_screening.json", screen)
    write_csv(output / "intervention_screening.csv", screen)
    files = [output / "intervention_screening.json"] + sorted((output / "fault_plans").glob("*.json"))
    files += sorted((output / "api_records").glob("*.json"))
    dump(output / "intervention_check_lock.json", dict(locked_at=utc(), before_test_answers=True,
         hashes={str(p.resolve()): digest(p) for p in files}))


def preflight_checks(m, questions, references, plans, screen):
    admitted = [r for r in screen if r["qualified"]]
    checks = dict(enough_pairs=len(admitted) >= m["minimum_pairs"],
                  enough_sources=len({r["source"] for r in admitted}) >= m["minimum_sources"],
                  numeric_projection_valid=not fact_projection_issues(references, plans),
                  complete_normal_bases=all(plans[q]["basis_complete"] for q in questions),
                  all_input_questions_retained=len(screen) == len(questions)
                  and {r["question_key"] for r in screen} == set(questions))
    return dict(status="passed" if all(checks.values()) else "failed", checks=checks,
                admitted=[r["question_key"] for r in admitted])


def freeze_cases(output):
    m = verify(output)
    verify_hashes(load(output / "intervention_check_lock.json")["hashes"])
    questions = load(output / "public_questions.json")
    plans = {p.stem: load(p) for p in (output / "proxy_plans").glob("*.json")}
    preflight = preflight_checks(m, questions, load(output / "private_references.json"), plans,
                                 load(output / "intervention_screening.json"))
    if preflight["status"] != "passed":
        dump(output / "preflight_failure.json", dict(checked_at=utc(), **preflight, generation_prohibited=True))
        raise ValueError("mandatory preflight failed: no test generation permitted")
    if (output / "cases_lock.json").exists():
        verify_ready(output)
        return
    if list((output / "generations").glob("*.json")):
        raise ValueError("case lock would follow test answers")
    normal = load(output / "public_retrieval.json")
    faults = load(output / "private_fault_interventions.json")
    pending = [(qid, c, normal[qid]["context"] if c == "live_normal" else faults[qid]["context"])
               for qid in preflight["admitted"] for c in ("live_normal", "live_corrupted")]
    random.Random(m["config"]["seed"]).shuffle(pending)
    cases, keys = {}, {}
    for i, (qid, condition, context) in enumerate(pending, 1):
        aid = f"L{i:04d}"
        cases[aid] = dict(question=questions[qid]["question"], context=context, context_sha256=sha(context))
        keys[aid] = dict(question_key=qid, condition=condition)
    for n, data in (("public_cases.json", cases), ("private_case_key.json", keys), ("private_control_answers.json", {})):
        dump(output / n, data)
    # Compatibility locks for the immutable generation/scoring functions.
    for name, files in (("retrieval_lock.json", [output / "public_retrieval.json"]),
                        ("proxy_lock.json", sorted((output / "proxy_plans").glob("*.json"))),
                        ("cases_lock.json", [output / n for n in ("public_cases.json", "private_case_key.json",
                            "private_control_answers.json", "intervention_check_lock.json")])):
        dump(output / name, dict(locked_at=utc(), hashes={str(p.resolve()): digest(p) for p in files}))
    files = [output / n for n in ("preparation_lock.json", "intervention_check_lock.json", "cases_lock.json")]
    dump(output / "preflight_lock.json", dict(locked_at=utc(), **preflight,
         hashes={str(p.resolve()): digest(p) for p in files}, maximum_new_generations=len(cases)))
    print("Mandatory preflight passed:", len(preflight["admitted"]), "pairs;", len(cases), "fresh generations frozen", flush=True)


def verify_ready(output):
    m = verify(output)
    lock = load(output / "preflight_lock.json")
    if lock["status"] != "passed":
        raise ValueError("generation prohibited without passed preflight")
    verify_hashes(lock["hashes"])
    experiment.verify_cases(output)
    questions = load(output / "public_questions.json")
    plans = {p.stem: load(p) for p in (output / "proxy_plans").glob("*.json")}
    fresh = preflight_checks(m, questions, load(output / "private_references.json"), plans,
                             load(output / "intervention_screening.json"))
    if any(lock[k] != fresh[k] for k in ("status", "checks", "admitted")):
        raise ValueError("preflight replay differs")
    return m


def generate(output):
    verify_ready(output)  # Failure raises before any client construction.
    experiment.generate(output)


def score(output):
    verify_ready(output)
    experiment.score(output)


def metrics(rows):
    return dict(n=len(rows), complete=sum(r["quality"] == "2" for r in rows),
        bad=sum(r["quality"] in {"0", "1"} for r in rows), quality_u=sum(r["quality"] == "U" for r in rows),
        proxy_u=sum(r["proxy"] == "U" for r in rows),
        bad_flagged=sum(r["quality"] in {"0", "1"} and r["proxy"] == "incomplete" for r in rows),
        bad_missed=sum(r["quality"] in {"0", "1"} and r["proxy"] == "complete" for r in rows),
        bad_unresolved=sum(r["quality"] in {"0", "1"} and r["proxy"] == "U" for r in rows),
        complete_false_flags=sum(r["quality"] == "2" and r["proxy"] == "incomplete" for r in rows),
        complete_proxy_u=sum(r["quality"] == "2" and r["proxy"] == "U" for r in rows),
        refusals=sum(r["refusal"] for r in rows),
        bad_without_refusal=sum(r["quality"] in {"0", "1"} and not r["refusal"] for r in rows),
        bad_without_refusal_flagged=sum(r["quality"] in {"0", "1"} and not r["refusal"] and r["proxy"] == "incomplete" for r in rows))


def analyze(output):
    m = verify_ready(output)
    questions = load(output / "public_questions.json")
    keys = load(output / "private_case_key.json")
    rows = []
    for aid, key in keys.items():
        s = load(output / "scores" / (aid + ".json"))
        rows.append(dict(aid=aid, **key, source=questions[key["question_key"]]["sources"][0], quality=s["quality"],
                         proxy=s["proxy_comparison"]["signal"]["status"], proxy_score=s["proxy_comparison"]["signal"]["score"],
                         refusal=s["field_refusal"]["refusal"], legacy_refusal=s["legacy_refusal"]))
    pairs = []
    for qid in load(output / "preflight_lock.json")["admitted"]:
        a = next(r for r in rows if r["question_key"] == qid and r["condition"] == "live_normal")
        b = next(r for r in rows if r["question_key"] == qid and r["condition"] == "live_corrupted")
        determinate = a["quality"] != "U" and b["quality"] != "U"
        pairs.append(dict(question_key=qid, source=a["source"], normal_quality=a["quality"], fault_quality=b["quality"],
             normal_proxy=a["proxy"], fault_proxy=b["proxy"], normal_refusal=a["refusal"], fault_refusal=b["refusal"],
             complete_to_bad=int(a["quality"] == "2" and b["quality"] in {"0", "1"}),
             worsened=int(determinate and int(b["quality"]) < int(a["quality"])),
             improved=int(determinate and int(b["quality"]) > int(a["quality"])), quality_u=int(not determinate)))
    table = [dict(condition=c, **metrics([r for r in rows if r["condition"] == c])) for c in ("live_normal", "live_corrupted")]
    normal, fault = table
    gate = m["gate"]
    checks = dict(normal_quality=normal["complete"] / normal["n"] >= gate["minimum_normal_complete_fraction"],
         paired_quality_drop=sum(p["complete_to_bad"] for p in pairs) / len(pairs) >= gate["minimum_paired_complete_to_bad_fraction"],
         bad_detection=fault["bad"] > 0 and fault["bad_flagged"] / max(fault["bad"], 1) >= gate["minimum_bad_answer_flag_fraction"],
         normal_false_flags=normal["complete_false_flags"] <= gate["maximum_normal_complete_false_flags"],
         quality_u=(normal["quality_u"] + fault["quality_u"]) / len(rows) <= gate["maximum_quality_u_fraction"])
    groups = [dict(source=s, condition=c, **metrics([r for r in rows if r["source"] == s and r["condition"] == c]))
              for s in sorted({r["source"] for r in rows}) for c in ("live_normal", "live_corrupted")]
    api = [load(p) for p in (output / "api_records").glob("*.json")]
    gens = [load(p) for p in (output / "generations").glob("*.json")]
    screen = load(output / "intervention_screening.json")
    summary = dict(version=m["version"], interpretation=m["interpretation"], input_questions=len(questions),
        structural_eligible=sum(r["structural"] for r in screen), qualified_pairs=len(pairs),
        qualified_sources=len({p["source"] for p in pairs}), excluded_before_answers=len(questions)-len(pairs),
        fresh_generations=len(gens), synthetic_answers=0, metrics=table,
        complete_to_bad=sum(p["complete_to_bad"] for p in pairs), paired_worsened=sum(p["worsened"] for p in pairs),
        paired_improved=sum(p["improved"] for p in pairs), paired_quality_u=sum(p["quality_u"] for p in pairs),
        gate=dict(status="passed" if all(checks.values()) else "not_passed", checks=checks,
                  scope="numeric_evidence_fault_mechanism_gate_not_stream_change_point_validation"),
        api_responses=len(api)+len(gens), failed_api_records=sum(not r.get("ok") for r in api),
        total_tokens=sum(r.get("usage", {}).get("total_tokens", 0) for r in api+gens),
        actual_models=dict(Counter([r.get("model_returned", "error") for r in api]+[g["actual_model"] for g in gens])),
        generation_finishes=dict(Counter(g["finish_reason"] for g in gens)),
        coverage_monitor_source="unmodified_normal_evidence", fault_plans_used_for_answer_scoring=False)
    for name, data in (("answer_scores.csv", rows), ("paired_quality.csv", pairs), ("signal_metrics.csv", table), ("source_groups.csv", groups)):
        write_csv(output / name, data)
    dump(output / "analysis_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["prepare", "check_faults", "freeze_cases", "generate", "score", "analyze"])
    ap.add_argument("--parent", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.action == "prepare":
        prepare(a.parent, a.output)
    else:
        globals()[a.action](a.output)
