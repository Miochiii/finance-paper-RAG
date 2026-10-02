"""Offline replay of V10 source split, schedule, causal cache and calibration."""
import argparse
import json
from pathlib import Path
import rag_auto_evaluation as auto
import rag_sparse_change_stream as v10
from rag_fresh_change_experiment import load, dump, sha, utc
from rag_prospective_coverage_validation import verify_hashes
from rag_objective_coverage_pilot import compare


def audit(output):
    m, calibration = v10.verify_calibration(output)
    qs = load(output/"public_questions.json")
    refs = load(output/"private_references.json")
    plans = load(output/"normal_monitor_plans.json")
    cases = load(output/"public_cases.json")
    cfg = m["config"]
    split = v10.split_sources(qs,cfg["seed"])
    if split != load(output/"source_split.json"):
        raise ValueError("source split changed")
    schedule = v10.schedule(split,cfg)
    if schedule != load(output/"private_schedule.json"):
        raise ValueError("schedule or conditions changed")
    parent = Path(m["previous"])
    old = load(parent/"artifact_manifest.json")
    verify_hashes(old["output_hashes"])
    verify_hashes(old["code_and_input_hashes"])
    original_qs = load(parent/"public_questions.json")
    original_refs = load(parent/"private_references.json")
    normal = load(parent/"public_retrieval.json")
    faults = load(parent/"private_fault_interventions.json")
    for qid,q in qs.items():
        if q != original_qs[qid] or refs[qid] != original_refs[qid] or plans[qid] != load(parent/"proxy_plans"/(qid+".json")):
            raise ValueError("normal question/reference/monitor modified")
        for c in ("live_normal","live_corrupted"):
            context = normal[qid]["context"] if c == "live_normal" else faults[qid]["context"]
            if cases[qid+":"+c] != dict(question=q["question"],context=context,context_sha256=sha(context)):
                raise ValueError("generation context differs from V9 frozen intervention")
    api = {p.stem:load(p) for p in (output/"api_records").glob("*.json")}
    used_api = set()
    def expect(out,name,prompt,payload,n=0):
        r = api[name]
        if (not r.get("ok") or r["prompt"]!=prompt or r["payload"]!=payload or r["round"]!=n
            or r["input_sha256"]!=sha(json.dumps([auto.MODEL,prompt,payload,n],ensure_ascii=False,sort_keys=True))
            or json.loads(r["raw"])!=r["parsed"]):
            raise ValueError("API payload/raw mismatch: "+name)
        used_api.add(name)
        return r["parsed"]
    cache = {p.stem:load(p) for p in (output/"score_cache").glob("*.json")}
    consumers = {}
    groups = {}
    gens = []
    rows = []
    original_request = auto.request
    auto.request = expect  # Missing records raise; audits can never invoke APIs.
    try:
        for slot in schedule:
            rid = slot["request_id"]
            qid = slot["question_key"]
            gen = load(output/"responses"/(rid+".json"))
            obs = load(output/"observations"/(rid+".json"))
            payload = v10.generation_payload(cases[qid+":"+slot["condition"]],cfg)
            if gen["slot"]!=slot or gen["payload"]!=payload or gen["payload_sha256"]!=sha(json.dumps(payload,ensure_ascii=False,sort_keys=True)):
                raise ValueError("generator sees changed inputs or private fields")
            raw = gen["raw_response"]
            if (raw["id"]!=gen["response_id"] or raw["model"]!=gen["actual_model"]
                or raw["choices"][0]["message"]["content"].strip()!=gen["answer"]
                or raw["choices"][0]["finish_reason"]!=gen["finish_reason"] or raw["usage"]!=gen["usage"]):
                raise ValueError("raw generation mismatch")
            inputs = v10.grade_input(qs[qid]["question"],gen,refs[qid],plans[qid])
            key = v10.score_key(inputs)
            stored = cache[key]
            if stored["key"]!=key or stored["inputs"]!=inputs:
                raise ValueError("score cache identity changed")
            if key not in consumers:
                ref = compare(output,"reference_"+key,inputs["question"],inputs["answer"],inputs["reference"])
                proxy = compare(output,"proxy_"+key,inputs["question"],inputs["answer"],inputs["monitor"])
                wanted = v10.derive_grade(ref,proxy,inputs["answer"],inputs["finish_reason"],qs[qid])
                if wanted!=stored["grade"]:
                    raise ValueError("cached score replay mismatch")
            expected = v10.signal_row(slot,key,stored["grade"])
            if any(obs[k]!=v for k,v in expected.items()) or obs["generation_response_id"]!=gen["response_id"]:
                raise ValueError("observation changed or U deleted")
            if gen["finished_at"]>obs["processed_at"] or stored["finished_at"]>obs["processed_at"]:
                raise ValueError("future score used in current observation")
            if slot["phase"]=="test" and gen["started_at"]<=calibration["locked_at"]:
                raise ValueError("test generation preceded calibration lock")
            if gen["started_at"]<=m["locked_at"]:
                raise ValueError("protocol followed data")
            consumers.setdefault(key,[]).append((gen,obs))
            groups.setdefault(slot["stream"],[]).append((gen,obs))
            gens.append(gen)
            rows.append(obs)
    finally:
        auto.request = original_request
    for key,items in consumers.items():
        if cache[key]["started_at"]<min(g["finished_at"] for g,o in items):
            raise ValueError("score cache created before any new answer")
        if cache[key]["finished_at"]>min(o["processed_at"] for g,o in items):
            raise ValueError("score cache was unavailable to a consumer")
    for sid,items in groups.items():
        items.sort(key=lambda item:item[0]["slot"]["step"])
        if [g["slot"]["step"] for g,o in items]!=list(range(1,len(items)+1)):
            raise ValueError("stream position attrition")
        if any(a[1]["processed_at"]>b[0]["started_at"] for a,b in zip(items,items[1:])):
            raise ValueError("within-stream generation/scoring not causal and serial")
    ids = [g["response_id"] for g in gens]+[r["response_id"] for r in api.values()]
    old_ids = {load(p)["response_id"] for p in (parent/"generations").glob("*.json")}
    if len(ids)!=len(set(ids)) or set(ids)&old_ids:
        raise ValueError("old or duplicated responses used")
    if set(api)!=used_api or set(cache)!=set(consumers):
        raise ValueError("unaccounted score/API input")
    wanted_ids = {s["request_id"] for s in schedule}
    for folder in ("responses","observations"):
        if {p.stem for p in (output/folder).glob("*.json")}!=wanted_ids:
            raise ValueError("scheduled response/observation count differs")
    mean = [r for r in rows if r["phase"]=="mean"]
    threshold = [r for r in rows if r["phase"]=="threshold"]
    if calibration["frozen"]!=v10.calibration_values(mean,threshold,cfg):
        raise ValueError("threshold or mean uses changed data")
    if max(o["processed_at"] for g,o in [item for items in groups.values() for item in items] if g["slot"]["phase"]!="test")>=calibration["locked_at"]:
        raise ValueError("calibration locked before its observations")
    result = dict(status="passed",checked_at=utc(),fresh_generations=len(gens),schedule_positions=len(schedule),
        scoring_api_records=len(api),distinct_score_inputs=len(cache),response_ids_unique_and_new=True,
        source_split_and_schedule_replayed=True,normal_references_and_monitor_plans_unchanged=True,
        score_cache_identity_and_derivation_replayed=True,within_stream_generation_scoring_serial=True,
        cache_created_from_new_answers_and_available_before_next_query=True,all_tests_follow_calibration_lock=True,
        calibration_replayed_from_normal_calibration_only=True,all_scheduled_no_injection_streams_retained=True,
        scope="dataflow_derivation_and_chronology_not_independent_truth_or_cloud_independence")
    dump(output/"integrity_audit.json",result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,required=True)
    audit(ap.parse_args().output)
