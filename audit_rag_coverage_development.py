"""Cross-phase integrity checks for semantic intervention and proxy revisions."""
import argparse
from collections import Counter
from pathlib import Path

import rag_auto_evaluation as auto
from rag_fresh_change_experiment import load, dump, utc
from rag_refusal import explicit_refusal
from rag_semantic_coverage_pilot import verify, verify_proxy_plans, direct_score, retrieval_score
from rag_quote_span_repair import agreement_score


def audit(output):
    _, cases = verify(output)
    verify_proxy_plans(output)
    v2, v3 = output / "coverage_refinement_v2", output / "coverage_refinement_v3"
    if any(load(p / "integrity_audit.json")["status"] != "passed" for p in (output, v2, v3)):
        raise ValueError("required phase audit missing")
    truth = load(output / "private_truth.json")
    keys = load(output / "public_proxy_plan_keys.json")
    main_api = {p.stem: load(p) for p in (output / "api_records").glob("*.json")}
    v2_api = {p.stem: load(p) for p in (v2 / "api_records").glob("*.json")}
    v3_api = {p.stem: load(p) for p in (v3 / "api_records").glob("*.json")}
    ids, models, total_tokens = [], Counter(), 0
    for r in list(main_api.values()) + list(v2_api.values()) + list(v3_api.values()):
        if not r.get("ok"):
            raise ValueError("failed API record")
        ids.append(r["response_id"])
        models[r["model_returned"]] += 1
        total_tokens += r["usage"]["total_tokens"]
    for aid, case in cases.items():
        gen = load(output / "generations" / (aid + ".json"))
        response = gen["raw_response"]
        if (gen["answer"] != response["choices"][0]["message"]["content"].strip() or
            gen["response_id"] != response["id"] or gen["actual_model"] != response["model"] or
            gen["usage"] != response["usage"] or gen["finish_reason"] != "stop" or
            gen["rule_refusal"] != int(explicit_refusal(gen["answer"]))):
            raise ValueError("generation derived fields differ from raw response")
        ids.append(gen["response_id"])
        models[gen["actual_model"]] += 1
        total_tokens += gen["usage"]["total_tokens"]
        score = load(output / "scores" / (aid + ".json"))
        rounds = [main_api[f"quality_{aid}_r{n}"]["parsed"] for n in (1, 2)]
        if rounds != score["rounds"]:
            raise ValueError("quality rounds differ from raw judge records")
        consensus = auto.quality_consensus(rounds, gen["rule_refusal"], True, len(truth[aid]["necessary_points"]))
        if any(score[k] != v for k, v in consensus.items()):
            raise ValueError("quality consensus edited")
        direct = main_api["direct_" + aid]["parsed"]
        if direct != score["direct"] or score["direct_signal"] != direct_score(direct, case["question"], gen["answer"]):
            raise ValueError("question-answer signal edited")
        plan = load(output / "proxy_plans" / (keys[aid] + ".json"))
        comparison = main_api["compare_" + aid]["parsed"] if plan["basis_complete"] else {}
        if score["comparison"] != comparison or score["retrieval_signal"] != retrieval_score(comparison, len(plan["points"]), plan["basis_complete"]):
            raise ValueError("first retrieval signal edited")
        second = load(v2 / "scores" / (aid + ".json"))
        plan2 = load(v2 / "plans" / (keys[aid] + ".json"))
        compare2 = v2_api["compare_" + aid]["parsed"] if plan2["basis_complete"] else {}
        if second["comparison"] != compare2 or second["signal"] != retrieval_score(compare2, len(plan2["points"]), plan2["basis_complete"]):
            raise ValueError("second retrieval signal edited")
        faith = v2_api["faith_" + aid]["parsed"]
        if faith != second["faith"] or second["support_status"] != auto.validate_faith(faith, case["context"]):
            raise ValueError("visible-context support score edited")
        third = load(v3 / "scores" / (aid + ".json"))
        plan3 = load(v3 / "plans.json")[keys[aid]]
        compare3 = [v3_api[f"compare_{aid}_r{n}"]["parsed"] for n in (1, 2)] if plan3["basis_complete"] else []
        if third["comparison"] != compare3 or third["signal"] != agreement_score(compare3, len(plan3["points"]), gen["answer"], plan3["basis_complete"]):
            raise ValueError("third retrieval signal edited")
    if len(ids) != len(set(ids)) or len(models) != 1:
        raise ValueError("reused responses or changed model identifier")
    value = dict(status="passed", checked_at=utc(), total_unique_api_responses=len(ids), generated_answers=len(cases),
                 actual_models=dict(models), total_tokens=total_tokens, generation_fields_match_raw_response=True,
                 every_quality_and_proxy_score_recomputed=True, all_phase_audits_passed=True,
                 revisions_use_same_development_answers=True, independent_validation=False)
    dump(output / "development_integrity_audit.json", value)
    print(value)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    audit(ap.parse_args().output)
