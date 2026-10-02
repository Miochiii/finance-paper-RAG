"""Verify immutable inputs, phase order and judge payload whitelists."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from rag_subtle_fault_pilot import CONDITIONS, verify
from rag_fresh_change_experiment import digest, dump, load


def audit(output):
    manifest, cases = verify(output)
    for path, expected in (manifest["input_hashes"] | manifest["model_input_hashes"]).items():
        if digest(Path(path)) != expected:
            raise ValueError("frozen input/code/model changed")
    private = load(output / "private_case_key.json")
    condition_counts = Counter(k["condition"] for k in private.values())
    if set(condition_counts) != set(CONDITIONS) or len(set(condition_counts.values())) != 1:
        raise ValueError("unbalanced paired cases")
    cfg = manifest["config"]
    response_ids, generations, api = [], [], []
    coverage_finished = []
    for aid, case in cases.items():
        gen = load(output / "generations" / (aid + ".json"))
        expected_messages = [dict(role="system", content=cfg["system_prompt"]),
                             dict(role="user", content=f'问题：{case["question"]}\n\n参考上下文：\n{case["context"]}')]
        if gen["payload"]["messages"] != expected_messages or not gen["answer"]:
            raise ValueError("generation input mismatch")
        generations.append(gen)
        response_ids.append(gen["response_id"])
        cov_path = output / "coverage" / (aid + ".json")
        coverage_finished.append(datetime.fromtimestamp(cov_path.stat().st_mtime, timezone.utc))
        for name, expected in (
            ("coverage_" + aid, dict(question=case["question"], necessary_points=case["necessary_points"], visible_context=case["context"])),
            ("faith_" + aid, dict(question=case["question"], model_answer=gen["answer"], visible_context=case["context"])),
            ("quality_" + aid + "_r1", dict(question=case["question"], model_answer=gen["answer"], necessary_points=case["necessary_points"], basis_complete=True)),
            ("quality_" + aid + "_r2", dict(question=case["question"], model_answer=gen["answer"], necessary_points=case["necessary_points"], basis_complete=True))):
            record = load(output / "api_records" / (name + ".json"))
            if not record["ok"] or record["payload"] != expected:
                raise ValueError("incomplete or contaminated judge request")
            response_ids.append(record["response_id"])
            api.append(record)
        load(output / "scores" / (aid + ".json"))
    gen_start = min(datetime.fromisoformat(g["started_at"]) for g in generations)
    if max(coverage_finished) >= gen_start:
        raise ValueError("evidence audit not complete before generation")
    if datetime.fromisoformat(manifest["prepared_at"]) >= min(datetime.fromisoformat(r["started_utc"]) for r in api):
        raise ValueError("API started before input freeze")
    judge_started = [datetime.fromisoformat(r["started_utc"]) for r in api if "model_answer" in r["payload"]]
    if min(judge_started) <= max(datetime.fromisoformat(g["finished_at"]) for g in generations):
        raise ValueError("answer evaluation started before all generation completed")
    if len(response_ids) != len(set(response_ids)):
        raise ValueError("duplicate response IDs")
    models = Counter([r["actual_model"] for r in generations] + [r["model_returned"] for r in api])
    if len(models) != 1:
        raise ValueError("model identifier changed; separate versions required")
    value = dict(status="passed", checked_at=datetime.now(timezone.utc).isoformat(),
        paired_condition_counts=dict(condition_counts), generations=len(generations), evaluations=len(api),
        unique_response_ids=len(set(response_ids)), actual_models=dict(models),
        input_code_model_hashes_checked=len(manifest["input_hashes"])+len(manifest["model_input_hashes"]),
        conditions_not_sent_to_judges=True, full_case_whitelists_checked=True,
        evidence_audit_completed_before_all_generation=True, generation_completed_before_quality_evaluation=True,
        all_selected_cases_retained=True)
    dump(output / "integrity_audit.json", value)
    print(value)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    audit(ap.parse_args().output)
