"""Replay V9 intervention qualification, blind API payloads and paired scores.

Offline only: the request reader cannot send a network request. Integrity does
not imply independent reference truth or a validated change-point detector.
"""
import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path

import rag_auto_evaluation as auto
import rag_live_evidence_pairs as v9
from rag_fresh_change_experiment import load, dump, sha, utc
from rag_objective_slots import EXTRACT_SLOTS, CHECK_SLOTS
from rag_percent_fact_guard import percent_unit_proxy
from rag_objective_coverage_pilot import compare
from rag_mineru_coverage_v4 import point_quality
from rag_field_refusal import field_refusal
from rag_refusal import explicit_refusal
from rag_prospective_coverage_validation import verify_hashes


def audit(output):
    m = v9.verify_ready(output)
    parent = Path(m["previous"])
    manifest = load(parent / "artifact_manifest.json")
    verify_hashes(manifest["output_hashes"])
    verify_hashes(manifest["code_hashes"])
    qs = load(output / "public_questions.json")
    refs = load(output / "private_references.json")
    retrieval = load(output / "public_retrieval.json")
    faults = load(output / "private_fault_interventions.json")
    if qs != load(parent / "table_axis_clarification_20261002/next_pair_public_questions.json"):
        raise ValueError("question bank changed")
    if refs != load(parent / "private_references.json"):
        raise ValueError("normal references changed")
    old_retrieval = load(parent / "public_retrieval.json")
    api = {p.stem: load(p) for p in (output / "api_records").glob("*.json")}
    expected = set()
    def expect(name, prompt, payload, n=0):
        record = api[name]
        hashed = sha(json.dumps([auto.MODEL, prompt, payload, n], ensure_ascii=False, sort_keys=True))
        if (not record.get("ok") or record["prompt"] != prompt or record["payload"] != payload
                or record["round"] != n or record["input_sha256"] != hashed
                or json.loads(record["raw"]) != record["parsed"]):
            raise ValueError("raw request mismatch: " + name)
        expected.add(name)
        return record["parsed"]
    normal_plans = {}
    screen = []
    for qid, q in qs.items():
        path = (parent / "table_axis_clarification_20261002/proxy_plans/clarified.json" if qid == "P0002"
                else parent / "proxy_plans" / (qid + ".json"))
        normal = load(path)
        normal_plans[qid] = normal
        if load(output / "proxy_plans" / (qid + ".json")) != normal:
            raise ValueError("normal monitoring plan altered")
        if retrieval[qid] != dict(old_retrieval[qid], **q):
            raise ValueError("normal cached retrieval altered")
        fault = v9.corrupt_evidence(retrieval[qid], normal)
        if faults[qid] != fault:
            raise ValueError("intervention replay mismatch")
        qualified = False
        if fault["structurally_valid"]:
            payload = dict(question=q["question"], requested_slots=q["requested_slots"], retrieved_evidence=fault["evidence"])
            raw = expect("fault_extract_" + qid, EXTRACT_SLOTS, payload)
            checks = [expect(f"fault_check_{qid}_r{n}", CHECK_SLOTS + v9.CHECK_SUFFIX[n],
                             dict(**payload, temporary_slots=raw.get("slots", [])), n) for n in (1, 2)]
            plan = dict(question=q["question"], evidence=fault["evidence"], raw=raw, checks=checks,
                        **percent_unit_proxy(raw, checks, q, fault["evidence"]))
            result = v9.qualify_fault(fault, normal, plan)
            stored = load(output / "fault_plans" / (qid + ".json"))
            if any(stored[k] != value for k, value in dict(plan=plan, **result).items()):
                raise ValueError("fault admission replay changed")
            qualified = result["qualified"]
        screen.append(dict(question_key=qid, source=q["sources"][0], structural=int(fault["structurally_valid"]),
                           structural_reason=fault["reason"], qualified=int(qualified),
                           changed_copies=len(fault.get("edits", [])), residual_values=len(fault.get("residual_original_value_spans", []))))
    if screen != load(output / "intervention_screening.json"):
        raise ValueError("question attrition or screening altered")
    preflight = v9.preflight_checks(m, qs, refs, normal_plans, screen)
    normal_contexts = load(output / "public_retrieval.json")
    pending = [(qid, c, normal_contexts[qid]["context"] if c == "live_normal" else faults[qid]["context"])
               for qid in preflight["admitted"] for c in ("live_normal", "live_corrupted")]
    random.Random(m["config"]["seed"]).shuffle(pending)
    cases, keys = {}, {}
    for i, (qid, condition, context) in enumerate(pending, 1):
        aid = f"L{i:04d}"
        cases[aid] = dict(question=qs[qid]["question"], context=context, context_sha256=sha(context))
        keys[aid] = dict(question_key=qid, condition=condition)
    if cases != load(output / "public_cases.json") or keys != load(output / "private_case_key.json"):
        raise ValueError("condition pairs or randomized schedule changed")
    if load(output / "private_control_answers.json"):
        raise ValueError("synthetic answers included in live trial")
    cfg = m["config"]
    generations = []
    original_request = auto.request
    # Audit never invokes the API, including when a required response is missing.
    auto.request = lambda out, name, prompt, payload, n=0: expect(name, prompt, payload, n)
    try:
        for aid, case in cases.items():
            qid = keys[aid]["question_key"]
            gen = load(output / "generations" / (aid + ".json"))
            payload = dict(model=cfg["model"], temperature=cfg["generation_temperature"], max_tokens=cfg["generation_max_tokens"],
                  messages=[dict(role="system", content=cfg["system_prompt"]),
                            dict(role="user", content=f'问题：{case["question"]}\n\n参考上下文：\n{case["context"]}')])
            if gen["payload"] != payload or gen["payload_sha256"] != sha(json.dumps(payload, ensure_ascii=False, sort_keys=True)):
                raise ValueError("generation payload changed or has private labels")
            raw = gen["raw_response"]
            if (gen["answer"] != raw["choices"][0]["message"]["content"].strip()
                    or gen["actual_model"] != raw["model"] or gen["response_id"] != raw["id"]
                    or gen["finish_reason"] != raw["choices"][0]["finish_reason"] or gen["usage"] != raw["usage"]):
                raise ValueError("generation/raw response mismatch")
            reference = compare(output, "reference_" + aid, case["question"], gen["answer"], refs[qid])
            proxy = compare(output, "proxy_" + aid, case["question"], gen["answer"], normal_plans[qid])
            if gen["finish_reason"] != "stop":
                proxy["signal"].update(status="U", score=None)
            wanted = dict(quality=point_quality(reference, gen["finish_reason"]), reference_comparison=reference,
                          proxy_comparison=proxy, legacy_refusal=int(explicit_refusal(gen["answer"])),
                          field_refusal=field_refusal(gen["answer"], [s["label"] for s in qs[qid]["requested_slots"]]))
            if load(output / "scores" / (aid + ".json")) != wanted:
                raise ValueError("scoring or independent normal-plan dataflow changed")
            generations.append(gen)
    finally:
        auto.request = original_request
    if set(api) != expected:
        raise ValueError("unaccounted API records")
    for folder in ("generations", "scores"):
        if {p.stem for p in (output / folder).glob("*.json")} != set(cases):
            raise ValueError("answer or score attrition")
    times = [load(output / n)["locked_at"] for n in ("protocol_lock.json", "preparation_lock.json",
             "intervention_check_lock.json", "retrieval_lock.json", "proxy_lock.json", "cases_lock.json", "preflight_lock.json")]
    if times != sorted(times) or max(times) >= min(g["started_at"] for g in generations):
        raise ValueError("test answers preceded qualification or mandatory preflight")
    qualification_api = [r for name, r in api.items() if name.startswith("fault_")]
    if min(r["started_utc"] for r in qualification_api) <= times[1]:
        raise ValueError("fault rule not frozen before extraction")
    if max(load(p)["finished_at"] for p in (output / "fault_plans").glob("*.json")) >= min(g["started_at"] for g in generations):
        raise ValueError("fault qualification followed test answers")
    ids = [r["response_id"] for r in api.values()] + [g["response_id"] for g in generations]
    if len(ids) != len(set(ids)):
        raise ValueError("response IDs reused")
    result = dict(status="passed", checked_at=utc(), input_questions=len(qs), retained_screening_records=len(screen),
         qualified_pairs=len(preflight["admitted"]), checked_api_records=len(api), checked_fresh_generations=len(generations),
         all_response_ids_unique=True, intervention_spans_replayed=True, all_duplicate_quote_copies_replayed=True,
         residual_old_magnitudes_conservatively_rejected=True, blind_fault_qualification_before_test_answers=True,
         normal_references_and_monitor_plans_unchanged=True, fault_plans_excluded_from_monitor=True,
         generation_inputs_only_question_context=True, scoring_inputs_only_question_answer_basis_facts=True,
         mandatory_preflight_before_all_answers=True, all_unresolved_and_screening_failures_retained=True,
         scope="dataflow_and_derivation_integrity_not_independent_reference_truth_or_stream_detector_validation")
    dump(output / "integrity_audit.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    audit(ap.parse_args().output)
