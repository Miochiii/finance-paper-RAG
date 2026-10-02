# -*- coding: utf-8 -*-
"""用已缓存的真实检索证据做配对的证据移除实验。

默认只构造并审计检索上下文，不请求模型。--live 会用 evaluate.py 的
deepseek-chat 入口把题目和证据上下文发往 api.deepseek.com；--judge 还会发送
金标准答案和模型答案。这与桌面 Harness 的在线服务不同。
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def gold_sources(row: Dict) -> set:
    return {s for s in (row.get("gold_sources") or "").split("|") if s}


def gold_hit(evidence: List[Dict], gold: set) -> int:
    return int(bool(gold & {e.get("source") for e in evidence}))


def remove_gold_evidence(evidence: List[Dict], gold: set,
                         donor_pool: List[Dict]) -> List[Dict]:
    """保留非金标准证据，以不同文献的缓存证据补足原来 top-k 的数量。"""
    if not evidence or not gold_hit(evidence, gold):
        raise ValueError("基线必须包含金标准文献")
    fault = [dict(e) for e in evidence if e.get("source") not in gold]
    seen = {(e.get("source"), e.get("text")) for e in fault}
    for e in donor_pool:
        key = (e.get("source"), e.get("text"))
        if e.get("source") in gold or key in seen or not e.get("text"):
            continue
        fault.append(dict(e))
        seen.add(key)
        if len(fault) >= len(evidence):
            break
    if len(fault) != len(evidence) or gold_hit(fault, gold):
        raise ValueError("无足够异源证据完成干预")
    return fault


def select_cases(answer_rows: List[Dict], evidence_by_qid: Dict[str, List[Dict]],
                 limit: int) -> List[Tuple[Dict, List[Dict]]]:
    cases = []
    for row in answer_rows:
        qid = row.get("qid", "")
        evidence = evidence_by_qid.get(qid, [])
        if (not row.get("question") or not row.get("gold_answer")
                or not gold_hit(evidence, gold_sources(row))):
            continue
        if float(row.get("judge_corr") or 0) < 4:
            continue
        cases.append((row, evidence))
        if len(cases) >= limit:
            break
    return cases


def run_replay(answers: Path, evidence_cache: Path, output_dir: Path,
               limit: int = 3, live: bool = False, judge: bool = False) -> Tuple[Path, Path]:
    from evaluate import em_f1, generate, llm_judge, make_context
    from probe_validity import answer_side_proxies

    with answers.open(encoding="utf-8-sig", newline="") as f:
        answer_rows = list(csv.DictReader(f))
    with evidence_cache.open(encoding="utf-8") as f:
        cache = json.load(f)
    cases = select_cases(answer_rows, cache, limit)
    if len(cases) < limit:
        raise ValueError(f"只找到 {len(cases)} 条有金标准证据和合格基线的查询")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "rag_remove_gold_replay.csv"
    report_path = output_dir / "rag_remove_gold_replay.md"
    if csv_path.exists() or report_path.exists():
        raise FileExistsError("实验输出已存在，拒绝覆盖；请指定新的 output-dir")
    rows = []
    for case_number, (row, base) in enumerate(cases, 1):
        qid = row["qid"]
        gold = gold_sources(row)
        donors = [e for other_qid, evs in cache.items() if other_qid != qid
                  for e in evs]
        fault = remove_gold_evidence(base, gold, donors)
        for condition, evidence in (("baseline", base), ("remove_gold", fault)):
            context = make_context([{"text": e["text"],
                                    "metadata": {"source": e.get("source", "")}}
                                   for e in evidence])
            pred, error = "", ""
            if live:
                try:
                    pred = generate(row["question"], context)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {str(exc)[:160]}"
            em, f1 = em_f1(pred, row["gold_answer"]) if pred else (None, None)
            corr, faith = None, None
            if pred and judge:
                corr, faith, reason = llm_judge(row["question"], row["gold_answer"],
                                                pred, context)
                if reason.startswith("judge失败") or reason.startswith("judge解析失败"):
                    error = reason
            proxy = answer_side_proxies(pred) if pred else {}
            rows.append({
                "qid": qid, "condition": condition, "gold_hit": gold_hit(evidence, gold),
                "evidence_count": len(evidence),
                "unique_docs": len({e.get("source") for e in evidence}),
                "question": row["question"], "gold_answer": row["gold_answer"],
                "pred_answer": pred, "em": em, "f1": f1,
                "judge_corr": corr, "judge_faith": faith,
                "ans_refusal": proxy.get("ans_refusal"),
                "ans_length": proxy.get("ans_length"), "error": error,
            })
        print(f"[{case_number}/{len(cases)}] {qid}: "
              f"正常/故障生成完成，错误 {sum(bool(r['error']) for r in rows[-2:])}", flush=True)
    with csv_path.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    completed = [r for r in rows if r["pred_answer"]]
    lines = ["# RAG 证据移除配对实验", "",
             f"- 来源：`{answers.name}` 的题目与金标准答案，`{evidence_cache.name}` 的已缓存真实检索证据。",
             f"- 查询 {len(cases)} 条；每题保留正常证据池，故障组仅移除金标准文献证据，"
             "再用其他查询缓存的异源证据补足相同数量。",
             "- 干预发生在检索结果进入上下文的位置；没有重建索引，也没有模拟线上查询时间序列。",
             f"- 生成方式：{'evaluate.py 的 deepseek-chat 入口' if live else '未调用模型，仅核对证据干预'}；"
             f"成功生成 {len(completed)}/{len(rows)} 条。", "",
             "| qid | 条件 | gold 文献命中 | 证据数 | F1 | 正确性评分 | 拒答代理 | 错误 |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        fmt = lambda x: "—" if x is None else f"{x:.2f}"
        lines.append(f"| {r['qid']} | {r['condition']} | {r['gold_hit']} | "
                     f"{r['evidence_count']} | {fmt(r['f1'])} | {fmt(r['judge_corr'])} | "
                     f"{fmt(r['ans_refusal'])} | {r['error'] or '—'} |")
    if len(completed) == len(rows):
        by_qid = {qid: {r["condition"]: r for r in rows if r["qid"] == qid}
                  for qid in {r["qid"] for r in rows}}
        pairs = [(v["baseline"], v["remove_gold"]) for v in by_qid.values()]
        f1_down = sum(f["f1"] < b["f1"] for b, f in pairs)
        refusal_up = sum(f["ans_refusal"] > b["ans_refusal"] for b, f in pairs)
        lines += ["", "## 配对摘要", "",
                  f"- 答案 F1 下降 {f1_down}/{len(pairs)} 题；"
                  f"拒答代理上升 {refusal_up}/{len(pairs)} 题。"]
        if judge and all(b["judge_corr"] is not None and f["judge_corr"] is not None
                         for b, f in pairs):
            corr_down = sum(f["judge_corr"] < b["judge_corr"] for b, f in pairs)
            lines.append(f"- DeepSeek 正确性评分下降 {corr_down}/{len(pairs)} 题。")
    lines += ["", "这是小样本机制试验，不估计检测器的误报率或检出延迟。"
              "如果故障未稳定降低答案质量，应先调整故障强度和真值，再做变点流实验。", ""]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, report_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--answers", type=Path, required=True)
    ap.add_argument("--evidence-cache", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--live", action="store_true",
                    help="将题目和证据上下文发往 DeepSeek API，实际生成答案")
    ap.add_argument("--judge", action="store_true",
                    help="额外发送金标准和模型答案给 DeepSeek API 评判")
    args = ap.parse_args()
    if args.limit < 1 or (args.judge and not args.live):
        ap.error("limit 至少为 1；--judge 必须与 --live 同用")
    csv_path, report_path = run_replay(args.answers, args.evidence_cache,
                                       args.output_dir, args.limit, args.live, args.judge)
    print(csv_path)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
