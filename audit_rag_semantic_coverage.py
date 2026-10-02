"""Audit phase locks and the boundary between normative truth and public proxies."""
import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path

import rag_auto_evaluation as auto
from rag_fresh_change_experiment import digest, dump, load, utc
from rag_semantic_coverage_pilot import (verify, verify_proxy_plans, qualified,
    proxy_payloads, retrieval_candidates)
from rag_subtle_fault_pilot import sources


def audit(output):
    manifest, cases = verify(output)
    verify_proxy_plans(output)
    for path, expected in manifest["model_input_hashes"].items():
        if digest(Path(path)) != expected:
            raise ValueError("local reranker model changed")
    corpus = load(output / "retrieval_corpus.json")
    if len(corpus) != 40 or any(set(r) != {"text", "sources", "text_sha256"} for r in corpus.values()):
        raise ValueError("retrieval corpus schema contaminated")
    if any(auto.sha(r["text"]) != r["text_sha256"] or set(r["sources"]) != sources(r["text"]) for r in corpus.values()):
        raise ValueError("retrieval corpus mismatch")
    selected = load(output / "private_screen_inputs.json")
    private, truth = load(output / "private_case_key.json"), load(output / "private_truth.json")
    plan_keys = load(output / "public_proxy_plan_keys.json")
    groups = {}
    for aid, key in private.items():
        groups.setdefault(key["qid"], {})[key["condition"]] = aid
    if set(cases) != set(private) or set(cases) != set(truth) or set(cases) != set(plan_keys):
        raise ValueError("case set mismatch")
    for qid, group in groups.items():
        if set(group) != {"baseline", "semantic_partial", "retained_evidence_prefix"}:
            raise ValueError("missing paired condition")
        construction = load(output / "construction" / (qid + ".json"))
        choice = construction["chosen"]
        if not qualified(construction["baseline_statuses"]) or not qualified(choice["statuses"], choice["target_index"]):
            raise ValueError("semantic gate failed")
        if cases[group["baseline"]]["context"] != selected[qid]["context"] or cases[group["semantic_partial"]]["context"] != choice["context"]:
            raise ValueError("construction context mismatch")
        retained = cases[group["retained_evidence_prefix"]]["context"]
        import re
        renumber = lambda text: re.sub(r"\[来源\d+\]", "[来源]", text)
        if not renumber(retained).endswith(renumber(selected[qid]["context"])):
            raise ValueError("retention control lost baseline bytes")
    # Every selected screening question, including failures, has a preserved record.
    constructions = [load(output / "construction" / (qid + ".json")) for qid in selected]
    if {r["qid"] for r in constructions if r["qualified"]} != set(groups):
        raise ValueError("post-generation exclusions detected")
    plan_expected = {}
    for scope in set(plan_keys.values()):
        plan = load(output / "proxy_plans" / (scope + ".json"))
        evidence = [dict(eid=e["eid"], text=e["text"]) for e in plan["evidence"]]
        for e in plan["evidence"]:
            if e["eid"] not in corpus or e["text"] != corpus[e["eid"]]["text"]:
                raise ValueError("retrieved evidence changed")
        payload = dict(question=plan["question"], retrieved_evidence=evidence)
        plan_expected["extract_" + scope] = payload
        plan_expected["verify_extract_" + scope] = dict(**payload, candidate_points=plan["extraction"].get("points", []))
    expected, gens = dict(plan_expected), []
    for aid, case in cases.items():
        if set(case) != {"question", "context", "context_sha256"} or auto.sha(case["context"]) != case["context_sha256"]:
            raise ValueError("public case contains private fields or mismatched context")
        plan = load(output / "proxy_plans" / (plan_keys[aid] + ".json"))
        if plan["question"] != case["question"] or plan["actual_sources"] != sorted(sources(case["context"])):
            raise ValueError("retrieval scope mismatch")
        available = {eid for eid, _ in retrieval_candidates(corpus, case["context"])}
        if not {e["eid"] for e in plan["evidence"]} <= available:
            raise ValueError("retrieval used hidden gold source filter")
        gen = load(output / "generations" / (aid + ".json"))
        cfg = manifest["config"]
        payload = dict(model=cfg["model"], temperature=cfg["generation_temperature"], max_tokens=cfg["generation_max_tokens"],
                      messages=[dict(role="system", content=cfg["system_prompt"]),
                                dict(role="user", content=f'问题：{case["question"]}\n\n参考上下文：\n{case["context"]}')])
        if gen["payload"] != payload or gen["finish_reason"] != "stop" or not gen["answer"]:
            raise ValueError("generation invalid or truncated")
        gens.append(gen)
        direct, comparison = proxy_payloads(case, gen["answer"], plan)
        expected["direct_" + aid] = direct
        if plan["basis_complete"]:
            expected["compare_" + aid] = comparison
        for n in (1, 2):
            expected[f"quality_{aid}_r{n}"] = dict(question=case["question"], model_answer=gen["answer"],
                       necessary_points=truth[aid]["necessary_points"], basis_complete=True)
        load(output / "scores" / (aid + ".json"))
    records = {p.stem: load(p) for p in sorted((output / "api_records").glob("*.json"))}
    for name, payload in expected.items():
        if name not in records or records[name].get("payload") != payload or not records[name].get("ok"):
            raise ValueError("evaluation/proxy payload contaminated or incomplete")
    if any(not r.get("ok") for r in records.values()):
        raise ValueError("failed request retained")
    creation = datetime.fromisoformat(manifest["prepared_at"])
    experiment_lock = datetime.fromisoformat(load(output / "experiment_lock.json")["locked_at"])
    plan_lock = datetime.fromisoformat(load(output / "proxy_plan_lock.json")["locked_at"])
    first_generation = min(datetime.fromisoformat(g["started_at"]) for g in gens)
    final_generation = max(datetime.fromisoformat(g["finished_at"]) for g in gens)
    if not creation < experiment_lock < plan_lock < first_generation:
        raise ValueError("phase lock order invalid")
    if any(datetime.fromisoformat(r["completed_at"]) >= experiment_lock for r in constructions):
        raise ValueError("construction finished after case freeze")
    for name, record in records.items():
        started = datetime.fromisoformat(record["started_utc"])
        if started <= creation:
            raise ValueError("API before construction lock")
        if name in plan_expected and not experiment_lock < started < plan_lock:
            raise ValueError("proxy plan not extracted before generation")
        if (name.startswith(("quality_", "direct_", "compare_")) and started <= final_generation):
            raise ValueError("answer evaluation before all generation completed")
        if name.startswith(("map_", "audit_")) and started >= experiment_lock:
            raise ValueError("intervention constructed after experimental lock")
    ids = [r["response_id"] for r in records.values()] + [g["response_id"] for g in gens]
    models = Counter([r["model_returned"] for r in records.values()] + [g["actual_model"] for g in gens])
    if len(ids) != len(set(ids)) or len(models) != 1:
        raise ValueError("duplicate response IDs or changed model version")
    value = dict(status="passed", checked_at=utc(), screened_questions=len(selected), qualified_questions=len(groups),
          paired_cases=len(cases), api_response_ids=len(ids), unique_response_ids=len(set(ids)), actual_models=dict(models),
          annotated_facts_not_sent_to_proxies=True, retrieval_uses_actual_source_metadata=True,
          public_schema_and_exact_proxy_payloads_checked=True, all_screening_failures_retained=True,
          semantic_absence_and_other_fact_retention_checked_twice=True, generation_after_plan_freeze=True,
          all_paired_cases_retained=True, code_input_model_hashes_checked=len(manifest["input_hashes"])+len(manifest["model_input_hashes"]))
    dump(output / "integrity_audit.json", value)
    print(value)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    audit(ap.parse_args().output)
