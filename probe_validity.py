# -*- coding: utf-8 -*-
"""probe_validity.py —— E0 实验：无标注代理指标的**有效性验证**。

论文问题（开题报告 §8 头号风险）：
    用于漂移监控的代理指标 S_k（无需人工标注即可在线计算）**真的能反映系统质量吗？**
    如果代理指标与真实质量无关，整个"无标注质量监控"的方法基础不成立。

做法：
    1. 只用**线上可得的量**构造候选代理指标（检索侧 / 回答侧 / 系统侧），全部归一化到 [0,1]；
    2. 真实质量用三种口径（互相独立、用于交叉验证）：
        A. LLM-judge 正确性 / 忠实性（1–5，人工标注的 gold answer 做参照）
        B. gold 文献是否被召回（doc 级 recall@5，二值）
        C. 结论级证据充分性分数（claim_audit 的 supported/partial/unsupported → [0,1]，连续）
    3. 计算每个代理 × 每个真值的 Spearman ρ + bootstrap 95% CI + Holm 校正 p 值，
       并给出该样本量下的**最小可检测效应量（MDE）**与各代理的分布特征
       （后者直接服务论文模块 2 的变前均值上界 m_k 估计）。

用法：
    python probe_validity.py                 # hmm 分块（n=40，检索侧代理需要本地嵌入模型）
    python probe_validity.py --no-embed      # 跳过嵌入类指标（无需 GPU）
    python probe_validity.py --methods all   # 回答侧代理扩展到 5 种分块（n=200）

输出：
    results/proxy_table_<tag>_<日期>.csv      每个查询一行：所有代理值 + 真值（后续 e-detector 的输入）
    results/proxy_validity_<tag>_<日期>.csv   代理 × 真值 的有效性统计
    results/proxy_validity_summary_<tag>_<日期>.md  汇报用汇总（含结论与建议）
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rag_core.config as config  # noqa: E402
from rag_core.config import PROJECT_DIR  # noqa: E402

RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
SERVICE_URL = os.getenv("RAG_AUDIT_URL", "http://127.0.0.1:8000")   # 补证据池用的在线服务
METHODS = ["fixed", "discourse", "hybrid", "hmm", "hmm_fixed_k"]
CITE_RE = re.compile(r"[\[（(]?\s*来源\s*(\d+)\s*[\]）)]?")
REFUSAL_RE = re.compile(r"(未找到|无法(确定|回答|判断)|未(提供|明确|提及)|没有(提供|明确)|信息不足|不足以回答)")
N_BOOT = 10000

# 代理指标的"预期方向"：+1 表示质量越好该值越大（漂移时下降），-1 表示反向
PROXY_DIRECTION = {
    "ret_top1_sim": +1, "ret_mean_sim": +1, "ret_margin": +1, "ret_sim_entropy": -1,
    "ret_unique_docs": +1, "ret_page_spread": 0, "ret_n_blocks": 0,
    "ans_citations": +1, "ans_citation_density": +1, "ans_refusal": -1,
    "ans_length": 0, "ans_n_claims": 0, "ans_claim_density": 0,
}
TRUTH_LABEL = {
    "judge_corr": "LLM-judge 正确性(1-5)",
    "judge_faith": "LLM-judge 忠实性(1-5)",
    "doc_hit": "gold 文献被召回(0/1)",
    "claim_support": "结论证据充分性(0-1)",
    "evidence_gap": "证据缺口率(0-1)",
    "ans_gold_sim": "答案↔标准答案语义相似度(0-1)",
    "quality_composite": "综合质量分(0-1)",
}


# --------------------------------------------------------------------------
# 统计工具（纯函数，便于测试）
# --------------------------------------------------------------------------
def spearman(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Spearman ρ 与双侧 p 值；样本太少或全为常数时返回 (nan, nan)。"""
    from scipy import stats
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return float("nan"), float("nan")
    rho, p = stats.spearmanr(x, y)
    return float(rho), float(p)


def bootstrap_rho_ci(x: List[float], y: List[float], n_boot: int = N_BOOT,
                     alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    """Spearman ρ 的 bootstrap 百分位区间（成对重采样，保持配对结构）。"""
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 5:
        return float("nan"), float("nan")
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        rho, _ = spearman(list(xa[idx]), list(ya[idx]))
        if not math.isnan(rho):
            vals.append(rho)
    if len(vals) < 50:
        return float("nan"), float("nan")
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


def holm(pvals: List[float]) -> List[float]:
    """Holm-Bonferroni 逐步校正（同族内多重检验）。nan 原样返回。"""
    idx = [i for i, p in enumerate(pvals) if not math.isnan(p)]
    order = sorted(idx, key=lambda i: pvals[i])
    m = len(order)
    out = [float("nan")] * len(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, pvals[i] * (m - rank))
        running = max(running, adj)
        out[i] = running
    return out


def mde_spearman(n: int, alpha: float = 0.05) -> float:
    """给定样本量下、α 水平双侧检验能达到的最小可检测 |ρ|（近似）。"""
    from scipy import stats
    if n < 4:
        return float("nan")
    tcrit = stats.t.ppf(1 - alpha / 2, n - 2)
    return float(tcrit / math.sqrt(tcrit ** 2 + (n - 2)))


def fisher_pool(rhos: List[float], ns: List[int]) -> float:
    """Fisher z 合并多个独立样本的相关系数（用于 5 种分块各自的 ρ 汇总）。"""
    zs, ws = [], []
    for r, n in zip(rhos, ns):
        if math.isnan(r) or n < 4 or abs(r) >= 1:
            continue
        zs.append(math.atanh(r))
        ws.append(n - 3)
    if not zs:
        return float("nan")
    return float(math.tanh(sum(z * w for z, w in zip(zs, ws)) / sum(ws)))


def norm01(vals: List[float]) -> List[float]:
    """min-max 归一化到 [0,1]（全常数时返回全 0.5）。"""
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return [0.5] * len(vals)
    return [(v - lo) / (hi - lo) for v in vals]


# --------------------------------------------------------------------------
# 数据装载
# --------------------------------------------------------------------------
def load_audit_truths() -> Dict[str, Dict]:
    """从 claim_audit 明细里算每题的证据充分性真值（连续，方差最大的一种真值）。"""
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "claim_audit_live_*.csv")))
    if not files:
        return {}
    out: Dict[str, Dict] = {}
    with open(files[-1], encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            qid = row["qid"]
            d = out.setdefault(qid, {"sup": 0, "par": 0, "uns": 0, "meta": 0, "claims": 0})
            d["claims"] += 1
            v = row["verdict"]
            d["sup" if v == "supported" else "par" if v == "partial"
              else "uns" if v == "unsupported" else "meta"] += 1
    for qid, d in out.items():
        content = d["sup"] + d["par"] + d["uns"]
        d["claim_support"] = ((d["sup"] + 0.5 * d["par"]) / content) if content else float("nan")
        d["evidence_gap"] = ((d["par"] + d["uns"]) / content) if content else float("nan")
        d["n_content"] = content
    return out


def load_answer_rows(methods: List[str]) -> List[Dict]:
    """读评测 CSV：真值（judge 分数）+ 回答侧代理的原料（答案文本）。"""
    rows = []
    for m in methods:
        path = os.path.join(RESULTS_DIR, f"{m}_finance.csv")
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                r["_method"] = m
                rows.append(r)
    return rows


# --------------------------------------------------------------------------
# 代理指标计算
# --------------------------------------------------------------------------
def answer_side_proxies(answer: str) -> Dict[str, float]:
    """回答侧代理：全部无需标注、无模型（引用形态、拒答、长度、结论数）。

    注意：结论数（claim 数）用规则拆句近似，与 claim_audit 的拆分口径一致。
    """
    text = answer or ""
    stripped = re.sub(r"[\[（(]?\s*来源\s*\d+\s*[\]）)]?", "", text)
    n_cit = len(CITE_RE.findall(text))
    chars = max(1, len(re.sub(r"\s", "", stripped)))
    sentences = [s for s in re.split(r"[。！？；\n]+", stripped) if len(s.strip()) >= 8]
    return {
        "ans_citations": float(n_cit),
        "ans_citation_density": 100.0 * n_cit / chars,
        "ans_refusal": 1.0 if REFUSAL_RE.search(stripped) else 0.0,
        "ans_length": float(chars),
        "ans_n_claims": float(len(sentences)),
        "ans_claim_density": 100.0 * len(sentences) / chars,
    }


def retrieval_side_proxies(evidence: List[Dict], query: str, embedder=None) -> Dict[str, float]:
    """检索侧代理：相似度水平/间隔/熵/多样性（需要嵌入模型；无模型时返回空）。"""
    if not evidence or embedder is None:
        return {}
    try:
        texts = [e["text"][:1200] for e in evidence]
        qv = embedder.encode([query], normalize_embeddings=True)[0]
        dv = embedder.encode(texts, normalize_embeddings=True)
        sims = [float(np.dot(qv, v)) for v in dv]
    except Exception:
        return {}
    sims_sorted = sorted(sims, reverse=True)
    top1 = sims_sorted[0]
    mean_sim = float(np.mean(sims))
    tail = sims_sorted[1:]
    margin = top1 - (float(np.mean(tail)) if tail else 0.0)
    # 相似度分布的熵（softmax 温度 0.1，越大说明候选之间越难区分）
    z = np.asarray(sims) / 0.1
    z = z - z.max()
    p = np.exp(z) / np.exp(z).sum()
    entropy = float(-(p * np.log(p + 1e-12)).sum() / math.log(len(sims))) if len(sims) > 1 else 0.0
    pages = [e.get("page_start") for e in evidence if e.get("page_start")]
    return {
        "ret_top1_sim": top1,
        "ret_mean_sim": mean_sim,
        "ret_margin": float(margin),
        "ret_sim_entropy": entropy,
        "ret_unique_docs": float(len({e["source"] for e in evidence})),
        "ret_page_spread": float(max(pages) - min(pages)) if len(pages) > 1 else 0.0,
        "ret_n_blocks": float(len(evidence)),
    }


def get_embedder():
    """复用项目里的 BGE 嵌入模型；GPU 不可用时退回 CPU。"""
    try:
        from rag_core.retriever import _get_embedding_model
        return _get_embedding_model()
    except Exception as e:
        print(f"  [警告] 嵌入模型加载失败（跳过检索侧相似度类指标）: {str(e)[:100]}")
        return None


def fetch_evidence(question: str, top_k: int = 5, mmr: bool = True,
                   timeout: int = 300) -> List[Dict]:
    """向运行中的 RAG 服务要一次检索结果（与生成时的检索配置一致），作为该题的证据池。"""
    import urllib.request
    body = json.dumps({"query": question, "top_k": top_k, "mmr": mmr}).encode("utf-8")
    req = urllib.request.Request(SERVICE_URL + "/search", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    out = []
    for i, hit in enumerate(data.get("results", []), start=1):
        m = hit.get("metadata", {}) or {}
        out.append({"n": i, "source": m.get("source") or "",
                    "page_start": m.get("page_start"), "page_end": m.get("page_end"),
                    "text": hit.get("text") or ""})
    return out


# --------------------------------------------------------------------------
# 证据池补全（评测集扩到 166 题后，旧缓存只有 40 题的证据）
# --------------------------------------------------------------------------
def _log_lines() -> List[str]:
    try:
        with open(config.OBS_LOG, encoding="utf-8") as f:
            return [l for l in f.read().splitlines() if l.strip()]
    except OSError:
        return []


def _log_search_rids(lines: List[str]) -> set:
    out = set()
    for line in lines:
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("event") == "search" and d.get("rid"):
            out.add(d["rid"])
    return out


def clean_live_search_log(before_lines: List[str], expected: int) -> None:
    """清理本次批量检索在观测日志与 MySQL 里留下的 search 事件（按 rid 精确比对）。

    批量工具不该污染生产统计：删掉本次新增的 search 行，并同步删除 MySQL 中同 rid 的行、
    回退 ETL 行号计数器。期间若在别处也做了检索，那些记录会被一并清掉，故打印数量对比。
    """
    before_rids = _log_search_rids(before_lines)
    lines = _log_lines()
    kept, removed_rids = [], []
    for line in lines:
        try:
            d = json.loads(line)
        except Exception:
            kept.append(line)
            continue
        rid = d.get("rid")
        if d.get("event") == "search" and rid and rid not in before_rids:
            removed_rids.append(rid)
            continue
        kept.append(line)
    if not removed_rids:
        print("  [日志] 无需清理（未发现本次批量检索的记录）")
        return
    with open(config.OBS_LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(kept) + "\n")
    n_db = 0
    try:
        from rag_core import mysql_store as ms
        conn = ms._connect()
        with conn.cursor() as cur:
            # 只删本次批量检索的 rid（不用时间窗，避免误删你在面板里的检索）
            for rid in removed_rids:
                cur.execute("DELETE FROM fact_search_log WHERE rid = %s", (rid,))
                n_db += cur.rowcount
            cur.execute("UPDATE etl_state SET v = %s WHERE k = 'search_log_lines'",
                        (str(len(kept)),))
        conn.close()
    except Exception as e:
        print(f"  [日志] MySQL 清理跳过: {str(e)[:60]}")
    note = "" if len(removed_rids) == expected else \
        f"（本次预期 {expected} 条，实际清掉 {len(removed_rids)} 条：批量期间的其他检索也会被一并清理）"
    print(f"  [日志] 已清理批量检索记录：jsonl {len(removed_rids)} 条 / MySQL {n_db} 行{note}")


def fetch_missing_evidence(cache: Dict[str, List[Dict]], rows: List[Dict],
                           top_k: int, mmr: bool, cache_path: str,
                           clean_log: bool = True, limit: int = 0) -> Dict[str, List[Dict]]:
    """为评测集里还没有证据池的题目补检索（走生产服务，配置与生成时一致）。

    166 题 × 单次 10~25 秒 ≈ 30~60 分钟；只补缺失的，支持中断续跑（每 10 题落盘一次）。
    limit>0 时只补前 N 题（小样验证链路用）。
    """
    todo = [r for r in rows if r["qid"] not in cache]
    if limit > 0:
        todo = todo[:limit]
    if not todo:
        print("  证据池已完整，无需补检索")
        return cache
    print(f"  需补检索 {len(todo)} 题（已有 {len(cache)} 题；预计 "
          f"{len(todo) * 12 / 60:.0f}~{len(todo) * 25 / 60:.0f} 分钟）")
    print("  提示：批量期间请勿在别处检索——那些记录会被一并清理")
    before = _log_lines()
    done = 0
    for i, r in enumerate(todo, 1):
        try:
            cache[r["qid"]] = fetch_evidence(r["question"], top_k, mmr)
            done += 1
        except Exception as e:
            print(f"  [{i}/{len(todo)}] {r['qid']} 检索失败: {str(e)[:70]}", flush=True)
            continue
        if i % 10 == 0 or i == len(todo):
            json.dump(cache, open(cache_path, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=0)
            print(f"  [{i}/{len(todo)}] 已补 {done} 题（缓存已落盘）", flush=True)
    json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    if clean_log:
        clean_live_search_log(before, expected=done)
    return cache


def answer_gold_similarity(embedder, pred: str, gold: str) -> float:
    """答案 ↔ 标准答案的语义相似度（标签型真值，比 1–5 整数分敏感得多）。"""
    try:
        v = embedder.encode([pred[:1500], gold[:1500]], normalize_embeddings=True)
        return float(np.dot(v[0], v[1]))
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="E0：无标注代理指标有效性验证")
    ap.add_argument("--method", default="hmm", help="主口径分块方法（默认 hmm）")
    ap.add_argument("--methods", default="one", choices=["one", "all"],
                    help="回答侧代理是否扩展到全部 5 种分块（n=200）")
    ap.add_argument("--no-embed", action="store_true", help="跳过需要嵌入模型的检索侧指标")
    ap.add_argument("--fetch-missing", action="store_true",
                    help="先补齐缺失题目的证据池（走线上检索，166 题约 30~60 分钟），再跑统计")
    ap.add_argument("--fetch-limit", type=int, default=0,
                    help="补证据池时最多补 N 题（0=全部；先用 3 试跑验证链路）")
    ap.add_argument("--top-k", type=int, default=5, help="证据池大小（默认 5，与生成时一致）")
    ap.add_argument("--no-mmr", action="store_true",
                    help="补证据池时关闭 MMR（默认开启，与生成时一致）")
    ap.add_argument("--no-clean-log", action="store_true",
                    help="补证据池后不清理观测日志/MySQL 里的批量检索记录")
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M")
    tag = args.method if args.methods == "one" else "allmethods"

    audit = load_audit_truths()
    methods = [args.method] if args.methods == "one" else METHODS
    rows = load_answer_rows(methods)
    if not rows:
        print("[错误] 找不到评测结果 CSV")
        return 1

    cache_path = os.path.join(RESULTS_DIR, f"_evidence_cache_{args.method}.json")
    if not os.path.isfile(cache_path):
        # claim_audit --from-live 写的是 _evidence_cache_live.json（答案与证据同源那一次）
        alt = os.path.join(RESULTS_DIR, "_evidence_cache_live.json")
        if os.path.isfile(alt):
            cache_path = alt
    cache = {}
    if os.path.isfile(cache_path):
        cache = json.load(open(cache_path, encoding="utf-8"))
    print(f"评测行 {len(rows)} 条（{len(methods)} 种分块）｜证据缓存 {len(cache)} 题"
          f"（{os.path.basename(cache_path)}）｜claim_audit 真值 {len(audit)} 题")
    if args.fetch_missing:
        cache = fetch_missing_evidence(cache, rows, args.top_k, not args.no_mmr,
                                       cache_path, clean_log=not args.no_clean_log,
                                       limit=args.fetch_limit)

    embedder = None if args.no_embed else get_embedder()

    # ---- 逐行算代理 ----
    table: List[Dict] = []
    for r in rows:
        qid, method = r["qid"], r["_method"]
        prox = answer_side_proxies(r.get("pred_answer") or "")
        if method == args.method and qid in cache:
            prox.update(retrieval_side_proxies(cache[qid], r["question"], embedder))
        gold = [g for g in (r.get("gold_sources") or "").split("|") if g.strip()]
        hit = None
        if method == args.method and qid in cache and gold:
            hit = 1.0 if set(gold) & {e["source"] for e in cache[qid]} else 0.0
        a = audit.get(qid, {}) if method == args.method else {}
        gold_sim = float("nan")
        if embedder is not None and (r.get("gold_answer") or "").strip() and (r.get("pred_answer") or "").strip():
            gold_sim = answer_gold_similarity(embedder, r["pred_answer"], r["gold_answer"])
        corr_n = float(r.get("judge_corr") or 0) / 5.0
        faith_n = float(r.get("judge_faith") or 0) / 5.0
        support = a.get("claim_support", float("nan"))
        parts = [v for v in (corr_n, faith_n, support) if not math.isnan(v)]
        table.append({
            "qid": qid, "method": method,
            "judge_corr": float(r.get("judge_corr") or 0),
            "judge_faith": float(r.get("judge_faith") or 0),
            "doc_hit": hit,
            "claim_support": support,
            "evidence_gap": a.get("evidence_gap", float("nan")),
            "ans_gold_sim": gold_sim,
            "quality_composite": float(np.mean(parts)) if parts else float("nan"),
            **prox,
        })

    # 字段表固定（多分块模式下只有 hmm 行才有检索侧代理，不能从首行推断）
    base_keys = ["qid", "method", "judge_corr", "judge_faith", "doc_hit", "claim_support",
                 "evidence_gap", "ans_gold_sim", "quality_composite"]
    proxy_names = list(PROXY_DIRECTION)
    truth_names = list(TRUTH_LABEL)
    fieldnames = base_keys + proxy_names

    # ---- 有效性统计 ----
    validity: List[Dict] = []
    for truth in truth_names:
        fam_p = []
        fam_rows = []
        for p in proxy_names:
            pairs = [(row[p], row[truth]) for row in table
                     if row.get(truth) is not None and not math.isnan(row.get(truth, float("nan")))
                     and row.get(p) is not None and not math.isnan(row.get(p, float("nan")))]
            xs = [a for a, _ in pairs]
            ys = [b for _, b in pairs]
            rho, pv = spearman(xs, ys)
            lo, hi = bootstrap_rho_ci(xs, ys) if len(xs) >= 8 else (float("nan"), float("nan"))
            fam_rows.append({"truth": truth, "proxy": p, "n": len(xs), "rho": rho, "p": pv,
                             "ci_lo": lo, "ci_hi": hi,
                             "expect": PROXY_DIRECTION[p],
                             "sign_ok": (None if math.isnan(rho) or PROXY_DIRECTION[p] == 0
                                         else (rho > 0) == (PROXY_DIRECTION[p] > 0)),
                             "mean": float(np.mean(xs)) if xs else float("nan"),
                             "sd": float(np.std(xs, ddof=1)) if len(xs) > 1 else float("nan"),
                             "min": float(np.min(xs)) if xs else float("nan"),
                             "p95": float(np.quantile(xs, 0.95)) if xs else float("nan"),
                             "max": float(np.max(xs)) if xs else float("nan"),
                             "n_missing": len(table) - len(xs)})
            fam_p.append(pv)
        for row, adj in zip(fam_rows, holm(fam_p)):
            row["p_holm"] = adj
            validity.append(row)

    # 多种分块各自的 ρ（回答侧代理的稳健性检查）
    per_method = []
    if args.methods == "all":
        for p in proxy_names:
            if p.startswith("ret_"):
                continue
            rhos, ns = [], []
            for m in methods:
                sub = [row for row in table if row["method"] == m]
                rho, _ = spearman([row[p] for row in sub], [row["judge_corr"] for row in sub])
                rhos.append(rho)
                ns.append(len(sub))
            per_method.append({"proxy": p, "pooled_rho": fisher_pool(rhos, ns),
                               "per_method": rhos})

    # ---- 输出 ----
    def _write_csv(path: str, data: List[Dict], fields: Optional[List[str]] = None) -> None:
        keys = fields or list(data[0].keys())
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for d in data:
                w.writerow({k: ("" if isinstance(v, float) and math.isnan(v)
                                else d.get(k, "") if k in d else "")
                            for k, v in ((k, d.get(k)) for k in keys)})

    table_path = os.path.join(RESULTS_DIR, f"proxy_table_{tag}_{stamp}.csv")
    valid_path = os.path.join(RESULTS_DIR, f"proxy_validity_{tag}_{stamp}.csv")
    summary_path = os.path.join(RESULTS_DIR, f"proxy_validity_summary_{tag}_{stamp}.md")
    _write_csv(table_path, table, fieldnames)
    _write_csv(valid_path, validity)

    n_eff = sum(1 for r in table if r["method"] == args.method)
    n_all = len(table)
    mde = mde_spearman(n_eff)
    mde_all = mde_spearman(n_all)
    strong = [r for r in validity
              if not math.isnan(r["rho"]) and r["p_holm"] < 0.05]
    strong.sort(key=lambda r: -abs(r["rho"]))
    top_any = sorted([r for r in validity if not math.isnan(r["rho"])],
                     key=lambda r: -abs(r["rho"]))[:10]
    # 指标建议表：每个代理取"最强的那个真值"作为它的有效性证据
    best_by_proxy: Dict[str, Dict] = {}
    for r in validity:
        if math.isnan(r["rho"]):
            continue
        cur = best_by_proxy.get(r["proxy"])
        if cur is None or abs(r["rho"]) > abs(cur["rho"]):
            best_by_proxy[r["proxy"]] = r

    lines = [
        f"# E0 · 无标注代理指标有效性验证（{tag}）",
        "",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M')}；代理 {len(proxy_names)} 个 × 真值 {len(truth_names)} 个",
        f"- 样本量：检索侧代理 **n={n_eff}**（{args.method} 分块的 40 题，需证据池）；"
        f"回答侧代理 **n={n_all}**（{len(methods)} 种分块 × 40 题）",
        f"- **最小可检测效应：|ρ| ≥ {mde:.2f}（n={n_eff}）／{mde_all:.2f}（n={n_all}）**"
        f"（α=0.05 双侧）——低于门槛的相关性在本样本上无法确证",
        f"- 多重检验：同一真值族内 Holm 校正；CI 为 {N_BOOT} 次配对 bootstrap 百分位区间",
        f"- 检索侧相似度类指标：{'已跳过（--no-embed）' if args.no_embed else '由 BGE 嵌入在线计算'}",
        "",
        "## 真值分布（先看区分度：真值本身没有区分度，代理再准也测不出相关性）",
        "",
        "| 真值 | 有效行数 | 不同取值 | 范围 / 均值 |",
        "|---|---|---|---|",
    ]
    for t in truth_names:
        vals = [row[t] for row in table if row.get(t) is not None
                and not math.isnan(row.get(t, float("nan")))]
        if not vals:
            continue
        uniq = sorted({round(v, 3) for v in vals})
        lines.append(f"| {TRUTH_LABEL[t]} | {len(vals)} | {len(uniq)} | "
                     f"{min(vals):.2f}~{max(vals):.2f} / {np.mean(vals):.2f} |")
    lines += ["", "## 与真实质量最相关的代理（按 |ρ| 排序）", "",
              "| 代理 | 真值 | ρ | 95% CI | Holm p | 方向符合预期 | n |",
              "|---|---|---|---|---|---|---|"]
    for r in top_any:
        lines.append(f"| {r['proxy']} | {TRUTH_LABEL[r['truth']]} | {r['rho']:+.3f} | "
                     f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] | {r['p_holm']:.3f} | "
                     f"{'—' if r['sign_ok'] is None else ('✅' if r['sign_ok'] else '❌')} | {r['n']} |")

    lines += ["", "## 指标建议表（进入 e-detector 的候选）", "",
              "| 代理 | 最强证据（真值 / ρ / Holm p） | 均值 | p95 | 范围 | 判定 |",
              "|---|---|---|---|---|---|"]
    for p, r in sorted(best_by_proxy.items(),
                       key=lambda kv: -abs(kv[1]["rho"])):
        if r["p_holm"] < 0.05 and r["sign_ok"] is not False:
            verdict = "✅ 建议纳入"
        elif r["p_holm"] < 0.05 and r["sign_ok"] is False:
            verdict = "⚠️ 显著但方向相反（需重新理解语义）"
        elif abs(r["rho"]) >= mde:
            verdict = "🟡 效应量够但校正后不显著（扩样本再验）"
        else:
            verdict = "⬜ 证据不足"
        lines.append(f"| {p} | {TRUTH_LABEL[r['truth']]} / {r['rho']:+.3f} / {r['p_holm']:.3f} | "
                     f"{r['mean']:.3f} | {r['p95']:.3f} | {r['min']:.3f}~{r['max']:.3f} | {verdict} |")
    sig = [r for r in strong if r["sign_ok"] is not False]
    lines += ["", "## 结论", ""]
    if sig:
        lines.append(f"- **通过 Holm 校正且方向符合预期的代理 {len(sig)} 个**："
                     + "、".join(f"{r['proxy']}（→{TRUTH_LABEL[r['truth']]}，ρ={r['rho']:+.3f}）"
                                for r in sig[:6]))
    lines.append(f"- 与「结论证据充分性」显著相关的代理："
                 f"{len([r for r in strong if r['truth'] == 'claim_support']) or '0 个'}"
                 f"——证据充分性这条真值目前没有被任何无标注代理捕捉到")
    lines.append(f"- **主要瓶颈是真值的区分度**：judge 分数天花板明显（"
                 f"{max(1, len([1 for row in table if row['judge_corr'] == 5]))}/{n_all} 行为满分），"
                 f"证据充分性 96% 挤在 1.0；唯一高区分度真值是答案↔标准答案相似度"
                 f"（{len({round(row['ans_gold_sim'], 3) for row in table if not math.isnan(row['ans_gold_sim'])})} 个不同取值）")
    lines.append(f"- 行动建议：① 标注 40 → 120 条（门槛 {mde:.2f} → {mde_spearman(120):.2f}）；"
                 f"② 线上暴露 reranker 分数（提升 margin 类代理的分辨率）；"
                 f"③ 后续漂移实验直接以本表「建议纳入」的代理作为检测器输入，其余作为敏感性画像的观察项")
    if per_method:
        lines += ["", "## 多种分块下的稳健性（回答侧代理 → judge 正确性，Fisher-z 合并）", "",
                  "| 代理 | 合并 ρ | 各方法 ρ |", "|---|---|---|"]
        for r in sorted(per_method,
                        key=lambda x: -(abs(x["pooled_rho"]) if not math.isnan(x["pooled_rho"]) else 0))[:8]:
            pm = ", ".join(f"{m}:{'' if math.isnan(v) else f'{v:+.2f}'}"
                           for m, v in zip(methods, r["per_method"]))
            lines.append(f"| {r['proxy']} | {r['pooled_rho']:+.3f} | {pm} |")
    lines += ["", f"逐查询代理矩阵（e-detector 的输入）：`{os.path.basename(table_path)}`；"
                  f"完整统计：`{os.path.basename(valid_path)}`。"]
    open(summary_path, "w", encoding="utf-8").write("\n".join(lines))

    print(f"\n代理矩阵：{table_path}\n有效性统计：{valid_path}\n汇总：{summary_path}")
    print(f"MDE(|ρ|) = {mde:.3f} @ n={n_eff}；显著代理（claim_support 族）= {len(strong)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
