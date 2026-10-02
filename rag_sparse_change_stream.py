"""V10: fresh sparse-fault streams, source-group split and causal score cache.

All generation slots are new. Exact question/answer/basis matches share a frozen
two-round evaluation; each stream processes its current answer before the next
generation. Test collection is blocked until normal calibration is frozen.
"""
import argparse
import json
import math
import random
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import edetector as ed
import rag_auto_evaluation as auto
from rag_objective_coverage_pilot import compare
from rag_fresh_change_experiment import load, dump, digest, sha, utc, write_csv, summarize_alarms
from rag_prospective_coverage_validation import verify_hashes
from rag_percent_fact_guard import fact_projection_issues
from rag_mineru_coverage_v4 import point_quality
from rag_field_refusal import field_refusal
from rag_refusal import explicit_refusal
from compare_rag_fresh_detectors import window_count, bernoulli_cusum, calibration_threshold

CONFIG = dict(seed=2026100210, mean_n=80, mean_streams=4, calibration_runs=12, normal_runs=30,
              fault_runs_per_rate=6, rates=[.1, .3, .6], horizon=40, change_at=16,
              target_far_h=.1, delta=.025, window=5, fixed_e_threshold=100., workers=4)
METHODS = ["first_refusal", "first_coverage", "window5_two", "window5_calibrated",
           "cusum_fixed", "cusum_calibrated", "edetector_fixed100", "edetector_calibrated"]
NEW_CODE = ["rag_sparse_change_stream.py", "audit_rag_sparse_change_stream.py",
            "report_rag_sparse_change_stream.py", "tests/test_rag_sparse_change_stream.py"]
EXTRA_CODE = ["edetector.py", "change_point_protocol.py", "compare_rag_fresh_detectors.py"]
_CACHE_LOCK = threading.Lock()
_KEY_LOCKS = {}


def split_sources(questions, seed):
    """Balance source counts and question counts using only pre-existing metadata."""
    by = {}
    for qid, q in questions.items():
        if len(q["sources"]) != 1:
            raise ValueError("one source per question required")
        by.setdefault(q["sources"][0], []).append(qid)
    if len(by) != 8:
        raise ValueError("expected eight V9-qualified sources")
    names = sorted(by)
    random.Random(seed).shuffle(names)
    names.sort(key=lambda name: -len(by[name]))  # Stable random ties.
    groups = {"calibration": {}, "test": {}}
    for name in names:
        options = [g for g in groups if len(groups[g]) < 4]
        g = min(options, key=lambda g: (sum(len(v) for v in groups[g].values()), len(groups[g]), g))
        groups[g][name] = sorted(by[name])
    return groups


def schedule(split, cfg):
    specs = [("mean", f"mean_{i:02d}", "calibration", 0., cfg["mean_n"]//cfg["mean_streams"], i)
             for i in range(cfg["mean_streams"])]
    specs += [("threshold", f"threshold_{i:02d}", "calibration", 0., cfg["horizon"], i)
              for i in range(cfg["calibration_runs"])]
    specs += [("test", f"normal_{i:02d}", "test", 0., cfg["horizon"], i)
              for i in range(cfg["normal_runs"])]
    specs += [("test", f"fault_{int(rate*100):02d}_{i:02d}", "test", rate, cfg["horizon"], i)
              for rate in cfg["rates"] for i in range(cfg["fault_runs_per_rate"])]
    rows = []
    for phase, sid, group, rate, length, run in specs:
        # Separate seed for each stream; all schedule draws occur before APIs.
        seed = int(sha(f'{cfg["seed"]}:{phase}:{sid}')[:16], 16)
        rng = random.Random(seed)
        sources = sorted(split[group])
        for step in range(1, length+1):
            source = rng.choice(sources)
            qid = rng.choice(split[group][source])
            draw = rng.random()
            corrupted = rate > 0 and step > cfg["change_at"] and draw < rate
            rows.append(dict(request_id=f"{sid}_{step:03d}", phase=phase, stream=sid, run=run,
                 step=step, rate=rate, source=source, question_key=qid,
                 condition="live_corrupted" if corrupted else "live_normal"))
    return rows


def prepare(parent, output):
    if (output/"protocol_lock.json").exists():
        verify(output)
        return
    old = load(parent/"artifact_manifest.json")
    verify_hashes(old["output_hashes"])
    verify_hashes(old["code_and_input_hashes"])
    previous = load(parent/"protocol_lock.json")
    if load(parent/"analysis_summary.json")["gate"]["status"] != "passed":
        raise ValueError("V9 mechanism gate not passed")
    admitted = load(parent/"preflight_lock.json")["admitted"]
    questions = {q: load(parent/"public_questions.json")[q] for q in admitted}
    refs = {q: load(parent/"private_references.json")[q] for q in admitted}
    normal = load(parent/"public_retrieval.json")
    faults = load(parent/"private_fault_interventions.json")
    plans = {q: load(parent/"proxy_plans"/(q+".json")) for q in admitted}
    if fact_projection_issues(refs, plans) or not all(p["basis_complete"] for p in plans.values()):
        raise ValueError("mandatory numeric/basis preflight failed")
    cases = {}
    for qid, q in questions.items():
        for c in ("live_normal", "live_corrupted"):
            context = normal[qid]["context"] if c == "live_normal" else faults[qid]["context"]
            cases[qid+":"+c] = dict(question=q["question"], context=context, context_sha256=sha(context))
    cfg = dict(CONFIG, generation=previous["config"])
    split = split_sources(questions, cfg["seed"])
    plan = schedule(split, cfg)
    output.mkdir(parents=True, exist_ok=True)
    for n, value in (("public_questions.json", questions), ("private_references.json", refs),
                     ("normal_monitor_plans.json", plans), ("public_cases.json", cases),
                     ("source_split.json", split), ("private_schedule.json", plan)):
        dump(output/n, value)
    inputs = [parent/n for n in ("artifact_manifest.json", "protocol_lock.json", "preflight_lock.json",
        "public_questions.json", "private_references.json", "private_fault_interventions.json", "public_retrieval.json")]
    inputs += [parent/"proxy_plans"/(q+".json") for q in admitted]
    inputs += [Path(__file__).parent/n for n in NEW_CODE+EXTRA_CODE]
    # Freeze all common method code inherited from V9, without changing it.
    inherited = {p: h for p, h in previous["input_hashes"].items() if p.endswith(".py")}
    hashes = dict(inherited, **{str(p.resolve()): digest(p) for p in inputs})
    files = [output/n for n in ("public_questions.json", "private_references.json", "normal_monitor_plans.json",
             "public_cases.json", "source_split.json", "private_schedule.json")]
    dump(output/"protocol_lock.json", dict(version="sparse_numeric_fault_stream_v10", locked_at=utc(),
        previous=str(parent.resolve()), input_hashes=hashes, source_hashes=previous["source_hashes"],
        initial_hashes={str(p.resolve()): digest(p) for p in files}, config=cfg, methods=METHODS,
        maximum_new_generations=len(plan), preflight="passed", all_model_answers_fresh=True,
        score_cache_rule="exact_current_question_answer_finish_reference_and_normal_monitor_facts_new_stage_cache_only",
        cache_evaluation="two_rounds_per_distinct_input_no_cached_answer_generation",
        unknown_signal_policy="proxy_U_is_alarm_positive; retain_quality_U; no_unknown_deletion",
        source_sampling="uniform_source_then_uniform_question_with_replacement",
        source_split="4_calibration_papers_4_test_papers_all_previously_seen_in_development",
        calibration_rule="m_binomial_upper_mean_phase_only_rank_threshold_threshold_phase_only",
        methods_selected_before_new_stream_data=True, all_test_data_blocked_until_calibration_lock=True,
        interpretation="prospective_fresh_fixed_bank_cached_context_source_separated_stream_pilot_not_natural_traffic",
        theory_source="https://nejsds.nestat.org/journal/NEJSDS/article/59/text",
        theory_scope="ARL_threshold_not_finite_horizon_false_alarm_probability; conditional_null_bound_unverified"))
    print("V10 frozen:", len(plan), "fresh slots; split question counts",
          {k:sum(len(v) for v in group.values()) for k,group in split.items()}, flush=True)


def verify(output):
    m = load(output/"protocol_lock.json")
    for n in ("input_hashes", "source_hashes", "initial_hashes"):
        verify_hashes(m[n])
    if m["preflight"] != "passed":
        raise ValueError("failed mandatory preflight")
    refs = load(output/"private_references.json")
    plans = load(output/"normal_monitor_plans.json")
    if fact_projection_issues(refs, plans):
        raise ValueError("numeric/basis preflight changed")
    return m


def generation_payload(case, cfg):
    c = cfg["generation"]
    return dict(model=c["model"], temperature=c["generation_temperature"], max_tokens=c["generation_max_tokens"],
        messages=[dict(role="system",content=c["system_prompt"]),
                  dict(role="user",content=f'问题：{case["question"]}\n\n参考上下文：\n{case["context"]}')])


def grade_input(question, gen, reference, plan):
    return dict(question=question, answer=gen["answer"], finish_reason=gen["finish_reason"],
                reference=dict(basis_complete=reference["basis_complete"], facts=reference["facts"]),
                monitor=dict(basis_complete=plan["basis_complete"], facts=plan["facts"]))


def score_key(inputs):
    return "K" + sha(json.dumps(inputs, ensure_ascii=False, sort_keys=True))


def derive_grade(reference, proxy, answer, finish, question):
    quality = point_quality(reference, finish)
    if finish != "stop":
        proxy["signal"].update(status="U", score=None)
    return dict(quality=quality, reference_comparison=reference, proxy_comparison=proxy,
        field_refusal=field_refusal(answer, [s["label"] for s in question["requested_slots"]]),
        legacy_refusal=int(explicit_refusal(answer)))


def cached_grade(output, question, gen, reference, plan):
    inputs = grade_input(question["question"], gen, reference, plan)
    key = score_key(inputs)
    with _CACHE_LOCK:
        lock = _KEY_LOCKS.setdefault(key, threading.Lock())
    with lock:
        path = output/"score_cache"/(key+".json")
        if path.exists():
            stored = load(path)
            if stored["inputs"] != inputs:
                raise ValueError("score cache input mismatch")
            return key, stored["grade"], True
        started = utc()
        try:
            ref = compare(output, "reference_"+key, inputs["question"], inputs["answer"], inputs["reference"])
            proxy = compare(output, "proxy_"+key, inputs["question"], inputs["answer"], inputs["monitor"])
        except Exception as exc:
            # Preserve failed API records before a later resume can retry them.
            records = {p.name:load(p) for p in (output/"api_records").glob("*"+key+"*.json")}
            dump(output/"evaluation_failures"/(key+"_"+sha(utc())[:12]+".json"),
                 dict(recorded_at=utc(), error_type=type(exc).__name__, records=records))
            raise
        grade = derive_grade(ref, proxy, inputs["answer"], inputs["finish_reason"], question)
        dump(path, dict(inputs=inputs, key=key, grade=grade, started_at=started, finished_at=utc()))
        return key, grade, False


def signal_row(slot, key, grade):
    status = grade["proxy_comparison"]["signal"]["status"]
    return dict(**slot, score_key=key, quality=grade["quality"], proxy_status=status,
                proxy_score=grade["proxy_comparison"]["signal"]["score"],
                coverage_flag=int(status != "complete"), coverage_u=int(status == "U"),
                refusal=int(grade["field_refusal"]["refusal"]), legacy_refusal=grade["legacy_refusal"])


def verify_calibration(output):
    m = verify(output)
    lock = load(output/"calibration_lock.json")
    verify_hashes(lock["hashes"])
    return m, lock


def collect(output, phase):
    import evaluate
    m = verify(output)
    if phase == "test":
        verify_calibration(output)  # Before any API client or cached test response.
    cfg = m["config"]
    cases = load(output/"public_cases.json")
    qs = load(output/"public_questions.json")
    refs = load(output/"private_references.json")
    plans = load(output/"normal_monitor_plans.json")
    groups = {}
    for slot in load(output/"private_schedule.json"):
        if slot["phase"] == phase:
            groups.setdefault(slot["stream"], []).append(slot)
    order = sorted(groups)
    random.Random(cfg["seed"] + {"mean":1, "threshold":2, "test":3}[phase]).shuffle(order)
    count = 0
    def stream(sid):
        client = evaluate._get_client().with_options(timeout=90, max_retries=1)
        hits = 0
        try:
            for slot in groups[sid]:
                rid = slot["request_id"]
                qid = slot["question_key"]
                case = cases[qid+":"+slot["condition"]]
                payload = generation_payload(case, cfg)
                path = output/"responses"/(rid+".json")
                if path.exists():
                    gen = load(path)
                    if gen["slot"] != slot or gen["payload"] != payload:
                        raise ValueError("cached generation input differs")
                else:
                    started = utc()
                    try:
                        raw = client.chat.completions.create(**payload)
                    except Exception as exc:
                        dump(output/"failed_attempts"/(rid+".json"), dict(slot=slot, started_at=started,
                             finished_at=utc(), error_type=type(exc).__name__))
                        raise RuntimeError(f"generation failed: {rid}/{type(exc).__name__}") from None
                    choice = raw.choices[0]
                    gen = dict(slot=slot, payload=payload, payload_sha256=sha(json.dumps(payload,ensure_ascii=False,sort_keys=True)),
                        started_at=started, finished_at=utc(), answer=(choice.message.content or "").strip(),
                        response_id=raw.id, actual_model=raw.model, finish_reason=choice.finish_reason,
                        usage=raw.usage.model_dump(), raw_response=raw.model_dump())
                    dump(path, gen)
                key, grade, reused = cached_grade(output, qs[qid], gen, refs[qid], plans[qid])
                obs_path = output/"observations"/(rid+".json")
                row = signal_row(slot, key, grade)
                if obs_path.exists():
                    old = load(obs_path)
                    if any(old[k] != v for k,v in row.items()):
                        raise ValueError("observation changed on resume")
                else:
                    dump(obs_path, dict(**row, generation_response_id=gen["response_id"],
                                       processed_at=utc(), score_cache_hit=reused))
                hits += int(reused)
        finally:
            client.close()
        return sid, len(groups[sid]), hits
    print("Collecting", phase, sum(len(v) for v in groups.values()), "fresh positions; serial generation + scoring within streams", flush=True)
    with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
        futures = [pool.submit(stream, sid) for sid in order]
        for f in as_completed(futures):
            sid, n, hits = f.result()
            count += n
            print(phase, sid, "finished", n, "positions; cumulative", count, "; cache hits", hits, flush=True)


def observations(output, phase):
    rows = []
    for slot in load(output/"private_schedule.json"):
        if slot["phase"] != phase:
            continue
        rid = slot["request_id"]
        obs = load(output/"observations"/(rid+".json"))
        if any(obs[k] != v for k,v in slot.items()):
            raise ValueError("slot/observation mismatch")
        rows.append(obs)
    return rows


def matrix(rows, feature):
    groups = {}
    for row in rows:
        groups.setdefault(row["stream"], []).append(row)
    ordered = [sorted(v, key=lambda r:r["step"]) for k,v in sorted(groups.items())]
    if not ordered or len({len(v) for v in ordered}) != 1:
        raise ValueError("unequal or empty monitoring streams")
    return np.asarray([[r[feature] for r in stream] for stream in ordered], float)


def method_series(coverage, refusal, m, p1, cfg):
    window = window_count(coverage, cfg["window"])
    cusum = bernoulli_cusum(coverage, m, p1)
    e = ed.build_detectors(coverage, m, ed.lambda_grid(m), "mix")
    return dict(first_refusal=refusal, first_coverage=coverage,
        window5_two=window, window5_calibrated=window, cusum_fixed=cusum,
        cusum_calibrated=cusum, edetector_fixed100=e, edetector_calibrated=e)


def calibration_values(mean, threshold, cfg):
    m = ed.estimate_m(np.asarray([r["coverage_flag"] for r in mean]), "binom", cfg["delta"])
    if not 0 < m < 1:
        raise ValueError("mean upper bound unsuitable; retain failed calibration without test generation")
    p1 = .3 if m < .3 else (m+1)/2
    series = method_series(matrix(threshold,"coverage_flag"), matrix(threshold,"refusal"), m, p1, cfg)
    thresholds = dict(first_refusal=1., first_coverage=1., window5_two=2.,
                      cusum_fixed=math.log(100.), edetector_fixed100=cfg["fixed_e_threshold"])
    for name in ("window5_calibrated", "cusum_calibrated", "edetector_calibrated"):
        value = calibration_threshold(series[name], cfg["target_far_h"])
        thresholds[name] = max(1.+1e-9,value) if name.startswith("edetector") else value
    return dict(m=m, cusum_p1=p1, thresholds=thresholds,
                mean_positives=sum(r["coverage_flag"] for r in mean), mean_u=sum(r["coverage_u"] for r in mean),
                threshold_positives=sum(r["coverage_flag"] for r in threshold), threshold_u=sum(r["coverage_u"] for r in threshold),
                calibration_maxima={n:float(s.max()) for n,s in series.items()})


def freeze_calibration(output):
    m = verify(output)
    mean, threshold = observations(output,"mean"), observations(output,"threshold")
    frozen = calibration_values(mean, threshold, m["config"])
    path = output/"calibration_lock.json"
    if path.exists():
        verify_calibration(output)
        if load(path)["frozen"] != frozen:
            raise ValueError("calibration replay changed")
        return
    tests = [s for s in load(output/"private_schedule.json") if s["phase"] == "test"]
    if any((output/"responses"/(s["request_id"]+".json")).exists() for s in tests):
        raise ValueError("test data already existed before threshold freeze")
    files = [output/folder/(r["request_id"]+".json") for r in mean+threshold for folder in ("responses","observations")]
    files += sorted((output/"score_cache").glob("*.json")) + sorted((output/"api_records").glob("*.json"))
    dump(path, dict(locked_at=utc(), frozen=frozen, calibration_only=True,
                   hashes={str(p.resolve()):digest(p) for p in files}))
    print("Calibration frozen:", json.dumps(frozen), flush=True)


def quality_metrics(rows):
    return dict(n=len(rows), complete=sum(r["quality"] == "2" for r in rows),
        bad=sum(r["quality"] in {"0","1"} for r in rows), quality_u=sum(r["quality"] == "U" for r in rows),
        proxy_u=sum(r["coverage_u"] for r in rows), positives=sum(r["coverage_flag"] for r in rows),
        bad_flagged=sum(r["quality"] in {"0","1"} and r["coverage_flag"] for r in rows),
        bad_refusals=sum(r["quality"] in {"0","1"} and r["refusal"] for r in rows),
        complete_false_flags=sum(r["quality"] == "2" and r["proxy_status"] == "incomplete" for r in rows),
        complete_u=sum(r["quality"] == "2" and r["coverage_u"] for r in rows))


def detector_analysis(rows, frozen, cfg):
    normal = [r for r in rows if r["rate"] == 0]
    normal_series = method_series(matrix(normal,"coverage_flag"), matrix(normal,"refusal"), frozen["m"], frozen["cusum_p1"], cfg)
    results, alarms, traces = [], [], []
    for rate in cfg["rates"]:
        fault = [r for r in rows if r["rate"] == rate]
        fs = method_series(matrix(fault,"coverage_flag"), matrix(fault,"refusal"), frozen["m"], frozen["cusum_p1"], cfg)
        sids = sorted({r["stream"] for r in fault})
        for method in METHODS:
            threshold = frozen["thresholds"][method]
            na = ed.first_alarm(normal_series[method], threshold)
            fa = ed.first_alarm(fs[method], threshold)
            results.append(dict(rate=rate, method=method, threshold=threshold,
                                **summarize_alarms(na,fa,cfg["horizon"],cfg["change_at"])))
            for phase, ids, values in (("normal",sorted({r["stream"] for r in normal}),na),("fault",sids,fa)):
                for sid, at in zip(ids,values):
                    alarms.append(dict(rate=rate, method=method, stream=sid, group=phase, alarm_query=int(at)))
            for step, value in enumerate(fs[method][0],1):
                traces.append(dict(rate=rate,stream=sids[0],step=step,method=method,statistic=float(value),threshold=threshold))
    return results, alarms, traces


def analyze(output):
    m, lock = verify_calibration(output)
    rows = [r for phase in ("mean","threshold","test") for r in observations(output,phase)]
    test = [r for r in rows if r["phase"] == "test"]
    results, alarms, traces = detector_analysis(test, lock["frozen"], m["config"])
    metrics = [dict(phase=phase,condition=condition,**quality_metrics([r for r in rows if r["phase"]==phase and r["condition"]==condition]))
               for phase in ("mean","threshold","test") for condition in ("live_normal","live_corrupted")
               if any(r["phase"]==phase and r["condition"]==condition for r in rows)]
    streams = []
    for sid in sorted({r["stream"] for r in test}):
        rr = [r for r in test if r["stream"]==sid]
        post = [r for r in rr if r["step"]>m["config"]["change_at"]]
        streams.append(dict(stream=sid, rate=rr[0]["rate"], injected=sum(r["condition"]=="live_corrupted" for r in post),
                            post_bad=sum(r["quality"] in {"0","1"} for r in post), post_u=sum(r["quality"]=="U" for r in post),
                            post_coverage_flags=sum(r["coverage_flag"] for r in post)))
    source_groups = [dict(source=source,phase=phase,**quality_metrics([r for r in rows if r["source"]==source and r["phase"]==phase]))
                     for source in sorted({r["source"] for r in rows}) for phase in ("mean","threshold","test")
                     if any(r["source"]==source and r["phase"]==phase for r in rows)]
    gens = [load(p) for p in (output/"responses").glob("*.json")]
    api = [load(p) for p in (output/"api_records").glob("*.json")]
    ids = [g["response_id"] for g in gens]
    if len(ids)!=len(set(ids)) or len(gens)!=m["maximum_new_generations"]:
        raise ValueError("generation count/freshness mismatch")
    summary = dict(version=m["version"], interpretation=m["interpretation"],
        fresh_generations=len(gens), unique_generation_response_ids=len(set(ids)),
        distinct_score_inputs=len(list((output/"score_cache").glob("*.json"))),
        score_cache_hits=sum(r["score_cache_hit"] for r in rows), scoring_api_responses=len(api),
        failed_api_records=sum(not a.get("ok") for a in api),
        actual_models=dict(Counter([g["actual_model"] for g in gens]+[r.get("model_returned","error") for r in api])),
        generation_finishes=dict(Counter(g["finish_reason"] for g in gens)),
        total_tokens=sum(r.get("usage",{}).get("total_tokens",0) for r in gens+api),
        calibration=lock["frozen"], metrics=metrics, detector_results=results,
        fault_streams_without_injection=sum(r["rate"]>0 and r["injected"]==0 for r in streams),
        all_scheduled_zero_fault_streams_retained=True, source_groups_separated=True,
        unknown_signal_policy=m["unknown_signal_policy"], references_scope="same_vendor_MinerU_automatic_truth",
        no_independence_or_conditional_null_bound_claim=True, completed_at=utc())
    for name, data in (("observations.csv",rows),("detector_results.csv",results),("alarm_times.csv",alarms),
                       ("prespecified_traces.csv",traces),("signal_metrics.csv",metrics),
                       ("stream_realizations.csv",streams),("source_groups.csv",source_groups)):
        write_csv(output/name,data)
    dump(output/"analysis_summary.json",summary)
    print(json.dumps({k:v for k,v in summary.items() if k!="detector_results"},ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action",choices=["prepare","collect","freeze_calibration","analyze"])
    ap.add_argument("--parent",type=Path)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--phase",choices=["mean","threshold","test"])
    a=ap.parse_args()
    if a.action=="prepare":prepare(a.parent,a.output)
    elif a.action=="collect":collect(a.output,a.phase)
    else:globals()[a.action](a.output)
