"""Fresh API observations on a frozen question pool; calibration never uses tests.

This is a controlled cached-context experiment, not natural RAG traffic. Fresh
response IDs do not prove independence or a stationary cloud model.
"""
import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import edetector as ed
from change_point_protocol import aggregate_queries, alarm_queries, exact_interval
from rag_refusal import explicit_refusal

VERSION = "fresh_fixed_pool_v1"
PHASES = ("mean", "threshold", "normal_test", "fault_test")
DEFAULTS = dict(seed=20261001, mean_n=100, calibration_runs=10, normal_runs=20,
                fault_runs=10, horizon=30, change_at=10, window=5,
                target_far=0.1, fixed_threshold=100.0, delta_each=0.025,
                workers=4, model="deepseek-chat", temperature=0.0, max_tokens=512,
                rerank_max_length=512, fusion_weights=[0.5, 0.5])


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode("utf-8")).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def schedule(qids, config):
    """Independent PRNG substreams for phases/runs; all draws frozen in advance."""
    if not qids or len(set(qids)) != len(qids):
        raise ValueError("question pool must contain distinct IDs")
    h, change = config["horizon"], config["change_at"]
    if not 0 < change < h or h % config["window"] or change % config["window"]:
        raise ValueError("horizon and change must align with disjoint windows")
    sizes = {"mean": (1, config["mean_n"]),
             "threshold": (config["calibration_runs"], h),
             "normal_test": (config["normal_runs"], h),
             "fault_test": (config["fault_runs"], h)}
    out = []
    for phase_index, phase in enumerate(PHASES):
        runs, length = sizes[phase]
        for run in range(runs):
            rng = np.random.default_rng([config["seed"], phase_index, run])
            for step, qid in enumerate(rng.choice(qids, length), 1):
                condition = "remove_gold" if phase == "fault_test" and step > change else "baseline"
                out.append(dict(request_id=f"{phase}_{run:03d}_{step:04d}", phase=phase,
                                run=run, step=step, qid=str(qid), condition=condition))
    return out


def prepare(root, prior, output, model_path):
    import evaluate
    output.mkdir(parents=True, exist_ok=True)
    paths = [prior / "blind_answers.json", prior / "private_answer_key.json",
             Path(__file__), root / "edetector.py", root / "rag_refusal.py"]
    model_files = [model_path / n for n in ("model.safetensors", "config.json", "tokenizer.json",
                                          "tokenizer_config.json", "sentencepiece.bpe.model")]
    hashes = {str(p): digest(p) for p in paths + model_files}
    if (output / "manifest.json").exists():
        if load(output / "manifest.json")["input_hashes"] != hashes:
            raise ValueError("frozen inputs or code changed; use a new experiment directory")
        return
    answers, keys = [load(p) for p in paths[:2]]
    cases = {}
    for rid, k in keys.items():
        a = answers[rid]
        # Explicit whitelist: no historical answer or gold answer is carried forward.
        case = dict(question=a["question"], context=a["visible_context"])
        case["context_sha256"] = sha(case["context"])
        if case["context_sha256"] != a["context_sha256"]:
            raise ValueError("context hash mismatch")
        cid = k["qid"] + ":" + k["condition"]
        if cid in cases:
            raise ValueError("duplicate case")
        cases[cid] = case
    qids = sorted({cid.split(":")[0] for cid in cases})
    if len(qids) != 40 or len(cases) != 80:
        raise ValueError("requires the authorized 40 paired questions")
    for qid in qids:
        if cases[qid + ":baseline"]["question"] != cases[qid + ":remove_gold"]["question"]:
            raise ValueError("paired questions differ")
    config = {**DEFAULTS, "system_prompt": evaluate.EVAL_PROMPT}
    plan = schedule(qids, config)
    dump(output / "cases.json", cases)
    dump(output / "schedule.json", plan)
    dump(output / "manifest.json", dict(version=VERSION, created_at=utc(), config=config,
         input_hashes=hashes, case_sha256=digest(output / "cases.json"),
         schedule_sha256=digest(output / "schedule.json"), reranker_path=str(model_path),
         scope="fresh_generation_on_fixed_cached_contexts_not_natural_online_traffic"))
    print(f"Frozen: {len(qids)} questions, {len(plan)} fresh generation slots", flush=True)


def verified_inputs(output):
    manifest = load(output / "manifest.json")
    for name, field in (("cases.json", "case_sha256"), ("schedule.json", "schedule_sha256")):
        if digest(output / name) != manifest[field]:
            raise ValueError("frozen schedule/cases changed")
    return manifest, load(output / "cases.json"), load(output / "schedule.json")


def relevance(output):
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    manifest, cases, _ = verified_inputs(output)
    target = output / "context_relevance.json"
    if target.exists():
        return
    config = manifest["config"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(manifest["reranker_path"], local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        manifest["reranker_path"], local_files_only=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32).eval().to(device)
    rows = {}
    ids = sorted(cases)
    for start in range(0, len(ids), 2):
        batch = ids[start:start+2]
        pairs = [(cases[c]["question"], cases[c]["context"]) for c in batch]
        full = tokenizer(pairs, truncation=False)["input_ids"]
        inputs = tokenizer(pairs, padding=True, truncation="only_second",
                           max_length=config["rerank_max_length"], return_tensors="pt").to(device)
        with torch.inference_mode():
            logits = model(**inputs).logits.reshape(-1).float().cpu().numpy()
        for cid, logit, tokens in zip(batch, logits, full):
            gap = 1.0 / (1.0 + math.exp(float(np.clip(logit, -700, 700))))
            rows[cid] = dict(logit=float(logit), relevance_gap=gap,
                             full_input_tokens=len(tokens), truncated=len(tokens) > config["rerank_max_length"],
                             context_sha256=cases[cid]["context_sha256"])
    dump(target, dict(created_at=utc(), device=device, torch_version=torch.__version__,
                     transformers_version=transformers.__version__,
                     interpretation="1-sigmoid(logit), bounded score, not a correctness probability",
                     scores=rows))
    print(f"Locally scored {len(rows)} contexts on {device}", flush=True)


def messages(case, config):
    return [{"role": "system", "content": config["system_prompt"]},
            {"role": "user", "content": f'问题：{case["question"]}\n\n参考上下文：\n{case["context"]}'}]


def valid_record(record, slot, case, config):
    expected = sha(json.dumps(messages(case, config), ensure_ascii=False, sort_keys=True))
    return (record["slot"] == slot and record["payload_sha256"] == expected
            and record["status"] == "ok" and bool(record["response_id"]))


def collect(output, phase):
    import evaluate
    manifest, cases, plan = verified_inputs(output)
    config = manifest["config"]
    features = load(output / "context_relevance.json")["scores"]
    if phase in ("normal_test", "fault_test") and not (output / "calibration_lock.json").exists():
        raise ValueError("freeze normal calibration before generating tests")
    slots = [s for s in plan if s["phase"] == phase]
    groups = {}
    for slot in slots:
        # Mean observations can be concurrent; within each monitoring stream requests are serial.
        group = slot["step"] % config["workers"] if phase == "mean" else slot["run"]
        groups.setdefault(group, []).append(slot)
    records = output / "responses"
    records.mkdir(exist_ok=True)

    def stream(items):
        client = evaluate._get_client().with_options(timeout=90, max_retries=0)
        try:
            for slot in items:
                path = records / (slot["request_id"] + ".json")
                cid = slot["qid"] + ":" + slot["condition"]
                case = cases[cid]
                if path.exists():
                    if not valid_record(load(path), slot, case, config):
                        raise ValueError("invalid cached response; do not replace silently")
                    continue
                payload = dict(model=config["model"], messages=messages(case, config),
                               temperature=config["temperature"], max_tokens=config["max_tokens"])
                value = dict(slot=slot, payload_sha256=sha(json.dumps(payload["messages"], ensure_ascii=False, sort_keys=True)),
                             payload=payload, context_sha256=case["context_sha256"], attempts=[])
                for attempt in range(3):
                    started = utc()
                    try:
                        response = client.chat.completions.create(**payload)
                        finished = utc()
                        choice = response.choices[0]
                        answer = (choice.message.content or "").strip()
                        if not answer:
                            raise ValueError("empty_response")
                        value["attempts"].append(dict(started_at=started, finished_at=finished, status="ok"))
                        value.update(status="ok", response_id=response.id, actual_model=response.model,
                                     started_at=started, finished_at=finished, answer=answer,
                                     finish_reason=choice.finish_reason, usage=response.usage.model_dump(),
                                     raw_response=response.model_dump(), ans_refusal=explicit_refusal(answer),
                                     context_gap=features[cid]["relevance_gap"])
                        dump(path, value)
                        break
                    except Exception as exc:
                        value["attempts"].append(dict(started_at=started, finished_at=utc(),
                                                      status="error", error_type=type(exc).__name__))
                        dump(output / "failed_attempts" / path.name, value)
                        if attempt == 2:
                            raise RuntimeError(f"generation failed: {slot['request_id']}, {type(exc).__name__}") from None
                        time.sleep(2)
        finally:
            client.close()
        return len(items)

    print(f"Collecting {phase}: {len(slots)} slots; serial within each stream", flush=True)
    with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        futures = [pool.submit(stream, items) for items in groups.values()]
        for future in as_completed(futures):
            print(f"{phase}: finished group of {future.result()} slots", flush=True)


def observations(output, phase):
    manifest, cases, plan = verified_inputs(output)
    out = []
    for slot in (s for s in plan if s["phase"] == phase):
        path = output / "responses" / (slot["request_id"] + ".json")
        if not path.exists():
            raise ValueError(f"incomplete phase: {slot['request_id']}")
        r = load(path)
        if not valid_record(r, slot, cases[slot["qid"] + ":" + slot["condition"]], manifest["config"]):
            raise ValueError("response integrity failed")
        out.append(r)
    return out


def matrices(records):
    groups = {}
    for r in records:
        groups.setdefault(r["slot"]["run"], []).append(r)
    rows = [sorted(groups[k], key=lambda r: r["slot"]["step"]) for k in sorted(groups)]
    if len({len(row) for row in rows}) != 1:
        raise ValueError("unequal stream lengths")
    for row in rows:
        for a, b in zip(row, row[1:]):
            if a["finished_at"] > b["started_at"]:
                raise ValueError("stream requests are not chronological")
    return {f: np.array([[r[f] for r in row] for row in rows])
            for f in ("ans_refusal", "context_gap")}


def statistics(features, mode, config, bounds):
    parts = {}
    for feature in ("ans_refusal", "context_gap"):
        x, steps = aggregate_queries(features[feature], mode, config["window"])
        parts[feature] = ed.build_detectors(x, bounds[feature], ed.lambda_grid(bounds[feature]), "mix")
    parts["fusion"] = sum(w * parts[f] for w, f in zip(config["fusion_weights"], ("ans_refusal", "context_gap")))
    return parts, steps


def empirical_threshold(scores, target_far):
    """Finite calibration rank, frozen in advance; no unconditional FAR guarantee."""
    maxima = np.asarray(scores).max(axis=1)
    rank = math.ceil((len(maxima) + 1) * (1 - target_far))
    if rank > len(maxima):
        raise ValueError("not enough calibration streams for the specified rank")
    return max(1.0 + 1e-9, float(np.nextafter(np.sort(maxima)[rank-1], np.inf)))


def freeze_calibration(output):
    manifest, _, _ = verified_inputs(output)
    mean, cal = observations(output, "mean"), observations(output, "threshold")
    config = manifest["config"]
    bounds = {"ans_refusal": ed.estimate_m(np.array([r["ans_refusal"] for r in mean]), "binom", config["delta_each"]),
              "context_gap": ed.estimate_m(np.array([r["context_gap"] for r in mean]), "hoeffding", config["delta_each"])}
    thresholds = {}
    for mode in ("raw", "disjoint"):
        scores, _ = statistics(matrices(cal), mode, config, bounds)
        for feature, s in scores.items():
            thresholds[mode + ":" + feature] = empirical_threshold(s, config["target_far"])
    record_hashes = {r["slot"]["request_id"]: digest(output / "responses" / (r["slot"]["request_id"] + ".json"))
                     for r in mean + cal}
    value = dict(bounds=bounds, thresholds=thresholds, calibration_response_hashes=record_hashes,
                 relevance_sha256=digest(output / "context_relevance.json"))
    path = output / "calibration_lock.json"
    if path.exists():
        if load(path)["frozen"] != value:
            raise ValueError("calibration changed after freezing")
        return
    if any((output / "responses" / (s["request_id"] + ".json")).exists()
           for s in load(output / "schedule.json") if s["phase"] in ("normal_test", "fault_test")):
        raise ValueError("test already generated before calibration lock")
    dump(path, dict(locked_at=utc(), frozen=value))
    print(f"Calibration frozen: m={bounds}", flush=True)


def safe_mean(values):
    return float(np.mean(values)) if len(values) else None


def summarize_alarms(normal, fault, horizon, change):
    normal, fault = np.asarray(normal), np.asarray(fault)
    false, pre = int(np.count_nonzero(normal)), (fault > 0) & (fault <= change)
    hit = (fault > change) & (fault <= horizon)
    delays = fault[hit] - change
    lo, hi = exact_interval(false, len(normal))
    dlo, dhi = exact_interval(int(hit.sum()), len(fault))
    at_risk = ~pre
    restricted = np.where(hit[at_risk], fault[at_risk] - change, horizon-change+1)
    return dict(n_normal=len(normal), false_alarms=false, far_h=false/len(normal),
                far_ci_low=lo, far_ci_high=hi, n_fault=len(fault), pre_alarms=int(pre.sum()),
                detected=int(hit.sum()), undetected=int(((fault == 0)).sum()),
                detect_rate=float(hit.mean()), detect_ci_low=dlo, detect_ci_high=dhi,
                detect_given_no_pre=float(hit.sum()/at_risk.sum()) if at_risk.any() else None,
                mean_detected_delay=safe_mean(delays),
                restricted_post_delay=safe_mean(restricted))


def analyze(output):
    freeze_calibration(output)
    manifest, _, _ = verified_inputs(output)
    config = manifest["config"]
    lock = load(output / "calibration_lock.json")
    frozen = lock["frozen"]
    all_records = [r for p in PHASES for r in observations(output, p)]
    ids = [r["response_id"] for r in all_records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate response IDs; freshness audit failed")
    models = Counter(r["actual_model"] for r in all_records)
    if len(models) != 1:
        raise ValueError("model identifier changed during experiment; split/version audit required")
    normal = matrices([r for r in all_records if r["slot"]["phase"] == "normal_test"])
    fault = matrices([r for r in all_records if r["slot"]["phase"] == "fault_test"])
    results, alarms, trajectories = [], [], []
    for mode in ("raw", "disjoint"):
        ns, steps = statistics(normal, mode, config, frozen["bounds"])
        fs, _ = statistics(fault, mode, config, frozen["bounds"])
        for feature in ns:
            for rule, threshold in (("fixed_100", config["fixed_threshold"]),
                                    ("normal_calibration", frozen["thresholds"][mode+":"+feature])):
                na, fa = [alarm_queries(s, threshold, steps) for s in (ns[feature], fs[feature])]
                results.append(dict(mode=mode, feature=feature, threshold_rule=rule,
                                    threshold=threshold, horizon=config["horizon"],
                                    **summarize_alarms(na, fa, config["horizon"], config["change_at"])))
                for phase, times in (("normal_test", na), ("fault_test", fa)):
                    for run, at in enumerate(times):
                        alarms.append(dict(mode=mode, feature=feature, threshold_rule=rule,
                                           phase=phase, run=run, alarm_query=int(at)))
            for t, score in zip(steps, fs[feature][0]):
                trajectories.append(dict(mode=mode, feature=feature, phase="fault_test", run=0,
                                         query=int(t), score=float(score)))
    write_csv(output / "detector_results.csv", results)
    write_csv(output / "alarm_times.csv", alarms)
    write_csv(output / "prespecified_example.csv", trajectories)
    rows = [dict(**r["slot"], response_id=r["response_id"], actual_model=r["actual_model"],
                 started_at=r["started_at"], finished_at=r["finished_at"], finish_reason=r["finish_reason"],
                 ans_refusal=r["ans_refusal"], context_gap=r["context_gap"],
                 context_sha256=r["context_sha256"], answer_sha256=sha(r["answer"])) for r in all_records]
    write_csv(output / "fresh_observations.csv", rows)
    usage = dict(total_responses=len(all_records), unique_response_ids=len(set(ids)), actual_models=dict(models),
                 finish_reasons=dict(Counter(r["finish_reason"] for r in all_records)),
                 recorded_error_attempts=sum(len(r["attempts"])-1 for r in all_records),
                 prompt_tokens=sum(r["usage"]["prompt_tokens"] for r in all_records),
                 completion_tokens=sum(r["usage"]["completion_tokens"] for r in all_records))
    summaries = {}
    for phase in PHASES:
        records = [r for r in all_records if r["slot"]["phase"] == phase]
        for condition in sorted({r["slot"]["condition"] for r in records}):
            subset = [r for r in records if r["slot"]["condition"] == condition]
            summaries[phase+":"+condition] = dict(n=len(subset), refusals=sum(r["ans_refusal"] for r in subset),
                                                    mean_gap=safe_mean([r["context_gap"] for r in subset]))
    dump(output / "analysis_summary.json", dict(usage=usage, phase_signals=summaries,
          calibration=frozen, detector_results=results, completed_at=utc()))
    report(output, config, usage, summaries, frozen, results)
    plot(output, config, frozen, trajectories)
    print(json.dumps(dict(usage=usage, phase_signals=summaries), ensure_ascii=False), flush=True)


def report(output, config, usage, summaries, frozen, results):
    truncated = sum(v["truncated"] for v in load(output / "context_relevance.json")["scores"].values())
    lines = ["# 固定题池的新生成序列：变点验证", "",
        "## 范围与冻结协议", "",
        f"使用此前授权的40题及80份冻结上下文，每个观测位置单独调用生成API；共{usage['total_responses']}次有效响应，响应ID全部唯一。未重用旧答案。",
        "这是缓存证据的受控生成试验；没有经过本轮实时检索，也不是自然线上流量。题池包含同题及同文献，云模型的独立性、稳定性、条件均值前提均未被证明。",
        f"均值校准{config['mean_n']}次；阈值校准{config['calibration_runs']}条流；正常测试{config['normal_runs']}条流；故障测试{config['fault_runs']}条流。每条监控流{config['horizon']}次请求，故障在第{config['change_at']}次之后开始。",
        "题目均匀有放回抽取，阶段/流使用不同PRNG子序列，全部请求计划在生成前冻结。流内请求串行；多条流同时采集。校准锁在任何测试生成前写入；测试从不参与m或阈值选择。",
        f"请求别名{config['model']}，实际响应模型{list(usage['actual_models'])[0]}，temperature={config['temperature']}，max_tokens={config['max_tokens']}。完成原因{usage['finish_reasons']}；截断回答若有也保留于观测，不选择性删除。",
        f"输入token {usage['prompt_tokens']}，输出token {usage['completion_tokens']}；可见失败尝试{usage['recorded_error_attempts']}次。API凭据没有写入结果。", "",
        "## 可观测信号与校准", "",
        "主信号为固定explicit_refusal规则。辅助为本地BGE的1−sigmoid(logit)，只反映问题与可见上下文的相关性，不是正确率或事实覆盖概率；不输入标准答案。",
        f"重排输入最长{config['rerank_max_length']}token，80个上下文中{truncated}个发生截断；分数只依据截断后实际模型输入。模型权重/分词器已哈希冻结。",
        f"正常均值校准所得m：拒答{frozen['bounds']['ans_refusal']:.6f}（二项上界），相关性缺口{frozen['bounds']['context_gap']:.6f}（Hoeffding上界）；每项delta={config['delta_each']}。这些上界依赖独立/稳定观测假设，不自动约束任意未来条件均值。",
        "各信号使用7个预设λ的累积统计量均匀混合；融合为两信号统计量各占0.5。原始请求为主，5条不重叠平均为预设对照。阈值100为主；正常校准流最大统计量的预定有限样本秩为辅助阈值。",
        f"校准目标FAR_H={config['target_far']}，只有{config['calibration_runs']}条校准流，因此辅助阈值取最大校准轨迹值的下一浮点数（最低1+1e-9）。不宣称真实误报率达到该目标。阈值100也不等价于FAR_H=1%。", "",
        "## 信号观测", "", "|阶段/条件|观测|拒答|平均相关性缺口|", "|---|---:|---:|---:|"]
    for key, v in summaries.items():
        lines.append(f"|{key}|{v['n']}|{int(v['refusals'])}|{v['mean_gap']:.4f}|")
    lines += ["", "## 正常误报与故障检测", "",
        "|输入/信号/阈值|阈值|正常报警/20（95%区间）|提前报警/10|检出/10|检出者平均延迟|受限变后延迟|",
        "|---|---:|---|---:|---:|---:|---:|"]
    for r in results:
        delay = "未检出" if r['mean_detected_delay'] is None else f"{r['mean_detected_delay']:.2f}"
        restricted = "无在险流" if r['restricted_post_delay'] is None else f"{r['restricted_post_delay']:.2f}"
        lines.append(f"|{r['mode']}/{r['feature']}/{r['threshold_rule']}|{r['threshold']:.3g}|"
                     f"{r['false_alarms']}/{r['n_normal']} [{r['far_ci_low']:.3f},{r['far_ci_high']:.3f}]|"
                     f"{r['pre_alarms']}|{r['detected']}|{delay}|{restricted}|")
    lines += ["", "FAR_H指30次请求内至少一次首次报警。精确二项区间只描述流之间独立同分布假设下的抽样误差，不涵盖同文献聚类、云服务时变等影响。",
              "提前报警单独计数，不算变后检出。检出者平均延迟排除了漏检；受限变后延迟仅在无提前报警的流计算，漏检记为21次，避免只报告成功流。它不是无限视界EDD/ARL。",
              "12个配置全部预先指定并完整报告；同一批观测上的配置不是独立实验，没有依据测试结果挑选最佳配置。",
              "run=0为预先固定的示例轨迹，不按检测效果挑选。", "",
              "## 后续", "", "先依据主配置判断本受控故障是否可检出；将辅助相关性作为检索失效信号分析，不能据此声称已解决直接答错。",
              "自然线上正常请求仍需单独采集并校准；长视界、低误报目标及新题/新文献分布需要新的冻结协议和新请求。现有20条正常测试流即使0报警，95%区间上限仍约16.8%，不足以验证1%误报目标。", ""]
    (output / "fresh_change_report.md").write_text("\n".join(lines), encoding="utf-8")


def plot(output, config, frozen, trajectories):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for ax, mode in zip(axes, ("raw", "disjoint")):
        for feature in ("ans_refusal", "context_gap", "fusion"):
            rows = [r for r in trajectories if r["mode"] == mode and r["feature"] == feature]
            ax.plot([r["query"] for r in rows], [max(r["score"], 1e-8) for r in rows], label=feature)
        ax.axhline(config["fixed_threshold"], color="black", ls=":", label="threshold 100")
        ax.axvline(config["change_at"], color="gray", ls="--")
        ax.set_yscale("log")
        ax.set_ylabel(f"Statistic ({mode})")
        ax.legend(loc="upper left", ncol=2, fontsize=8)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Fresh request index; fault starts after request 10")
    fig.suptitle("Prespecified fault stream 0 (fixed question pool)")
    fig.tight_layout()
    fig.savefig(output / "prespecified_example.png", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("prepare", "relevance", "collect", "freeze", "analyze"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path(__file__).parent)
    ap.add_argument("--prior", type=Path)
    ap.add_argument("--model-path", type=Path)
    ap.add_argument("--phase", choices=PHASES)
    args = ap.parse_args()
    if args.command == "prepare":
        prepare(args.root, args.prior, args.output, args.model_path)
    elif args.command == "relevance":
        relevance(args.output)
    elif args.command == "collect":
        collect(args.output, args.phase)
    elif args.command == "freeze":
        freeze_calibration(args.output)
    else:
        analyze(args.output)


if __name__ == "__main__":
    main()
