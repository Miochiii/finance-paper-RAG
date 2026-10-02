# -*- coding: utf-8 -*-
"""从现有论文中生成新的、证据可追溯的 RAG 评测候选。

默认仅预览选中的本地片段；--live 才把每篇最多四段、每段最多 900 字
发送到 api.deepseek.com 出题和核验。候选始终为 pending，不修改正式标注。
"""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

import gen_annotations as ga

QUOTE_VERIFY_PROMPT = """你是评测题证据核查员。只根据给出的【逐字引文】判断：
1. answer_supported：引文本身是否支持标准答案的所有要点；只支持一部分即为 false。
2. unique：引文对这个问题是否给出唯一明确的答案。
不要借助其他知识或未提供的原文。仅输出 JSON：
{"answer_supported":true,"unique":true,"issue":""}"""


def _norm(text: str) -> str:
    return "".join((text or "").split())


def _quality(text: str) -> float:
    sample = (text or "")[:900]
    han = len(re.findall(r"[\u4e00-\u9fff]", sample))
    if (len(sample) < 180 or han / max(1, len(sample)) < 0.45
            or "�" in sample or sample.count("|") > 3
            or "[TABLE" in sample or sample.count("$") > 2
            or "参考文献" in sample[:150] or "目录" in sample[:80]):
        return float("-inf")
    sentences = sum(sample.count(p) for p in "。！？；")
    if sentences < 2:
        return float("-inf")
    terms = sum(w in sample for w in ("本文", "本研究", "研究结果", "实验结果",
                                      "实证", "模型", "方法", "数据集", "准确率"))
    return min(sentences, 8) + 2 * terms + min(len(re.findall(r"\d", sample)), 6) / 3


def existing_by_doc(rows: Sequence[Dict]) -> Dict[str, Dict[str, List[str]]]:
    out: Dict[str, Dict[str, List[str]]] = {}
    for row in rows:
        for doc in (row.get("gold_docs") or "").split("|"):
            doc = doc.strip()
            if not doc:
                continue
            entry = out.setdefault(doc, {"questions": [], "quotes": []})
            if row.get("question"):
                entry["questions"].append(row["question"])
            if row.get("gold_chunks"):
                entry["quotes"].append(row["gold_chunks"])
    return out


def select_fresh_excerpts(chunks: Sequence[Dict], old_quotes: Sequence[str],
                          rng: np.random.Generator, k: int = 4,
                          max_chars: int = 900) -> List[Dict]:
    """优先选择未含已有 gold 引文的不同页片段。"""
    used = [_norm(q)[:50] for q in old_quotes if len(_norm(q)) >= 50]
    eligible = [c for c in chunks if _quality(c.get("text") or "") > float("-inf")]
    if not eligible:
        return []
    order = sorted(rng.permutation(len(eligible)),
                   key=lambda i: -_quality(eligible[int(i)].get("text") or ""))
    fresh, reused = [], []
    for i in order:
        c = eligible[int(i)]
        text = _norm(c.get("text") or "")
        (reused if any(q in text for q in used) else fresh).append(c)
    chosen, pages = [], set()
    for pool in (fresh, reused):
        for c in pool:
            page = c.get("page_start")
            if page in pages:
                continue
            chosen.append({**c, "text": (c.get("text") or "")[:max_chars]})
            pages.add(page)
            if len(chosen) >= k:
                return chosen
    return chosen


def plan_documents(kb: Dict[str, List[Dict]], existing_rows: Sequence[Dict],
                   *, seed: int, limit: int, excerpts_per_doc: int = 4,
                   max_chars: int = 900) -> List[Tuple[str, List[Dict], int]]:
    old = existing_by_doc(existing_rows)
    rng = np.random.default_rng(seed)
    plan = []
    for doc, chunks in sorted(kb.items()):
        prior = old.get(doc, {"questions": [], "quotes": []})
        excerpts = select_fresh_excerpts(chunks, prior["quotes"], rng,
                                         excerpts_per_doc, max_chars)
        if len(excerpts) < 2:
            continue
        plan.append((doc, excerpts, len(prior["questions"])))
    # 优先扩充既有题目较少的文献；同数量时文件名固定排序。
    plan.sort(key=lambda item: (item[2], item[0]))
    return plan[:limit]


def verbatim_quote(item: Dict, excerpts: List[Dict]) -> Tuple[str, bool]:
    quote, fallback = ga.verify_quote(item["quote"], excerpts, item["evidence"])
    if not any(quote in e["text"] for e in excerpts):
        # 上游的宽松空白归一化可能把非逐字引文判为匹配；此处再做原文子串检查。
        source = next((excerpts[i - 1]["text"] for i in item["evidence"]
                       if 1 <= i <= len(excerpts)), excerpts[0]["text"])
        quote, fallback = source[:300], True
    return quote, fallback


def candidate_from_item(item: Dict, doc: str, excerpts: List[Dict],
                        all_chunks: List[Dict], prior_questions: Sequence[str],
                        verifier: Dict) -> Dict:
    quote, fallback = verbatim_quote(item, excerpts)
    pages = sorted({excerpts[i - 1].get("page_start") for i in item["evidence"]
                    if 1 <= i <= len(excerpts) and excerpts[i - 1].get("page_start") is not None})
    missing_count, missing = ga.fact_check(item["a"], all_chunks, [str(p) for p in pages])
    sim = max((ga.question_similarity(item["q"], old) for old in prior_questions),
              default=0.0)
    ok = bool(verifier.get("answer_supported")) and bool(verifier.get("unique", True))
    row = {
        "id": "", "status": "pending", "question": item["q"], "answer": item["a"],
        "gold_docs": doc, "gold_chunks": quote,
        "notes": f"生成类型:{item['type']}；机器质检:{'通过' if ok else '存疑'}",
        "gen_type": item["type"], "evidence_pages": "|".join(str(p) for p in pages),
        "quote_fallback": int(fallback), "dup_sim": round(sim, 3),
        "title_leak": ga.compute_title_leak(item["q"], doc),
        "fact_missing": missing_count, "missing_facts": missing,
        "retrieval_hit": "", "system_answer": "", "system_supports_gold": "",
        "verify": "ok" if ok else f"存疑:{str(verifier.get('issue') or '')[:30]}",
    }
    row["tier"] = ga.triage(row)
    return row


def run(*, limit: int, per_doc: int, seed: int, output: Path,
        live: bool, duplicate_threshold: float = 0.60) -> Tuple[Path, Path]:
    with Path(ga.MAIN_CSV).open(encoding="utf-8-sig", newline="") as f:
        existing = list(csv.DictReader(f))
    kb = ga.load_kb_by_doc()
    plan = plan_documents(kb, existing, seed=seed, limit=limit)
    if not plan:
        raise ValueError("没有足够的本地原文片段")
    output.parent.mkdir(parents=True, exist_ok=True)
    preview = output.with_suffix(".preview.md")
    if output.exists():
        raise FileExistsError(f"候选输出已存在，拒绝覆盖: {output}")
    preview_lines = ["# 新题候选的本地片段预览", "",
                     f"- 既有标注 {sum(r.get('status') == 'done' for r in existing)} 题；"
                     f"计划处理 {len(plan)} 篇文献，每篇至多 {per_doc} 题。",
                     "- 下表只列片段页号与哈希；原文不复制进预览。", "",
                     "| 文献 | 已有题 | 新片段页号 | 片段 SHA-256 前 12 位 |",
                     "|---|---:|---|---|"]
    for doc, excerpts, prior_count in plan:
        pages = ",".join(str(e.get("page_start") or "?") for e in excerpts)
        hashes = ",".join(hashlib.sha256(e["text"].encode()).hexdigest()[:12]
                          for e in excerpts)
        preview_lines.append(f"| {doc} | {prior_count} | {pages} | {hashes} |")
    preview_text = "\n".join(preview_lines) + "\n"
    if preview.exists():
        if preview.read_text(encoding="utf-8") != preview_text:
            raise FileExistsError(f"预览与本轮参数不一致，拒绝覆盖: {preview}")
    else:
        with preview.open("x", encoding="utf-8") as f:
            f.write(preview_text)
    if not live:
        return preview, preview

    client = ga.get_client()
    questions = [r["question"] for r in existing if r.get("question")]
    accepted, skipped_similar, skipped_invalid_evidence = [], 0, 0
    tokens_in = tokens_out = 0
    for doc_number, (doc, excerpts, _) in enumerate(plan, 1):
        ev_text = "\n\n".join(f'【片段{i}】\n{e["text"]}'
                              for i, e in enumerate(excerpts, 1))
        payload = f"论文文件名：{doc}\n\n{ev_text}"
        data, ti, to = ga.llm_json(client, ga.GEN_PROMPT.format(n=per_doc), payload)
        tokens_in += ti
        tokens_out += to
        items = ga.parse_items(json.dumps(data, ensure_ascii=False))[:per_doc]
        for item in items:
            if not any(1 <= i <= len(excerpts) for i in item["evidence"]):
                skipped_invalid_evidence += 1
                continue
            similarity = max((ga.question_similarity(item["q"], old)
                              for old in questions), default=0.0)
            if similarity >= duplicate_threshold:
                skipped_similar += 1
                continue
            quote, _ = verbatim_quote(item, excerpts)
            verify_payload = (f"【问题】{item['q']}\n【标准答案】{item['a']}\n"
                              f"【逐字引文】\n{quote}")
            verified, vi, vo = ga.llm_json(client, QUOTE_VERIFY_PROMPT,
                                           verify_payload, max_tokens=200)
            tokens_in += vi
            tokens_out += vo
            row = candidate_from_item(item, doc, excerpts, kb[doc], questions, verified)
            accepted.append(row)
            questions.append(item["q"])
        print(f"[{doc_number}/{len(plan)}] {doc}: 累计候选 {len(accepted)}，"
              f"相似题 {skipped_similar}，无效证据 {skipped_invalid_evidence}", flush=True)
    if not accepted:
        raise ValueError("没有产生通过去重的候选；预览文件已保留")
    with output.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ga.CAND_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(accepted)
    meta = {"seed": seed, "docs": len(plan), "existing_done": sum(
                r.get("status") == "done" for r in existing),
            "candidates": len(accepted), "skipped_similar": skipped_similar,
            "skipped_invalid_evidence": skipped_invalid_evidence,
            "tokens_in": tokens_in, "tokens_out": tokens_out,
            "estimated_cost_cny": round(tokens_in / 1e6 * ga.PRICE_IN_PER_M
                                        + tokens_out / 1e6 * ga.PRICE_OUT_PER_M, 2),
            "all_status": "pending", "human_review_required": True}
    output.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return output, preview


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--per-doc", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--live", action="store_true",
                    help="将最多 limit 篇论文的片段及候选答案发往 DeepSeek API")
    ap.add_argument("--duplicate-threshold", type=float, default=0.60)
    args = ap.parse_args()
    if args.limit < 1 or args.per_doc < 1 or not 0 < args.duplicate_threshold < 1:
        ap.error("limit/per-doc 须为正，duplicate-threshold 须位于 (0,1)")
    result, preview = run(limit=args.limit, per_doc=args.per_doc, seed=args.seed,
                          output=args.output, live=args.live,
                          duplicate_threshold=args.duplicate_threshold)
    print(result)
    print(preview)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
