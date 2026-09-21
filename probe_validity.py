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

# 长度派生指标：与"答案长度"强相关，而真值口径（短标准答案 + 精确匹配 + judge）
# 本身就会惩罚冗长 → 这类指标无法通过"控制长度"自证，不能当独立代理使用。
LENGTH_DERIVED = {"ans_length", "ans_n_claims", "ans_claim_density", "ans_citations"}


def variance_level(values: List[float], min_class: int = 3, discrete_max: int = 5) -> str:
    """分布可用性三档：ok / rare / degenerate。

    - degenerate：批次内近似常数（如 ret_n_blocks 新批次恒为 5）→ 相关性无定义，不可用；
    - rare：离散量里稀有类别不足 min_class 例（如拒答在某批次只有 2 例）→ 可用但估计不稳，
      必须在报告里标注；对漂移监控而言"稀有事件"往往正是最敏感的报警信号，不能直接丢弃；
    - ok：正常。
    连续量（取值很多）只要求确实有变异。
    """
    from collections import Counter
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    distinct = set(vals)
    if len(distinct) < 2:
        return "degenerate"
    if len(distinct) <= discrete_max:
        return "ok" if min(Counter(vals).values()) >= min_class else "rare"
    return "ok" if len(distinct) >= 3 else "rare"


def variance_ok(values: List[float], min_class: int = 3, discrete_max: int = 5) -> bool:
    """分布是否可用于相关性分析（rare 也算可用，只是不稳）。"""
    return variance_level(values, min_class, discrete_max) != "degenerate"
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


def fisher_pool_test(rhos: List[float], ns: List[int]) -> Tuple[float, float]:
    """Fisher-z 合并多个独立样本的 ρ，并对"合并后 ρ = 0"做双侧检验。

    注意：|ρ| = 1 时 atanh 发散，做法是**截断到 ±0.999999**而不是丢弃样本
    （丢弃会把"数学上等于 1"的合法结果变成 NaN——偏相关里很常见）。
    """
    from scipy import stats
    zs, ws = [], []
    for r, n in zip(rhos, ns):
        if r is None or (isinstance(r, float) and math.isnan(r)) or n < 5:
            continue
        r = max(-0.999999, min(0.999999, float(r)))
        zs.append(math.atanh(r))
        ws.append(n - 3)
    if not zs or sum(ws) <= 0:
        return float("nan"), float("nan")
    z_bar = sum(z * w for z, w in zip(zs, ws)) / sum(ws)
    se = math.sqrt(1.0 / sum(ws))
    p = 2 * (1 - stats.norm.cdf(abs(z_bar) / se)) if se > 0 else float("nan")
    return float(math.tanh(z_bar)), float(p)


def rho_diff_p(r1: float, n1: int, r2: float, n2: int) -> float:
    """两个独立样本的 ρ 是否有显著差异（Fisher z 检验，双侧）。"""
    from scipy import stats
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in (r1, r2)):
        return float("nan")
    if n1 < 5 or n2 < 5:
        return float("nan")
    r1 = max(-0.999999, min(0.999999, float(r1)))
    r2 = max(-0.999999, min(0.999999, float(r2)))
    se = math.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    if se <= 0:
        return float("nan")
    z = (math.atanh(r1) - math.atanh(r2)) / se
    return float(2 * (1 - stats.norm.cdf(abs(z))))


def partial_rho_rank_from_rows(rows: List[Dict], proxy: str, truth: str,
                               control: str = "ans_length",
                               batch_key: Optional[str] = None) -> Tuple[float, int]:
    """一阶偏相关（在秩上计算）：同时消除"长度混淆"与"批次差异"。

    公式：ρ_xy|z = (ρ_xy − ρ_xz·ρ_yz) / sqrt((1−ρ_xz²)(1−ρ_yz²))，
    在**每个批次内**分别计算，再按 Fisher-z 合并（因此也控制了批次差异）。
    比"先回归取残差再求相关"稳健：完全共线时残差恒等会让后者算出 ρ=1 的假强相关。
    """
    groups: Dict[str, List[Tuple[float, float, float]]] = {}
    for r in rows:
        vals = [r.get(proxy), r.get(truth), r.get(control)]
        if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in vals):
            continue
        key = str(r.get(batch_key, "")) if batch_key else "all"
        groups.setdefault(key, []).append((float(vals[0]), float(vals[1]), float(vals[2])))
    parts, ns = [], []
    total = 0
    for key, triples in groups.items():
        if len(triples) < 8:
            continue
        x = [t[0] for t in triples]
        y = [t[1] for t in triples]
        z = [t[2] for t in triples]
        total += len(triples)
        rho_xy, _ = spearman(x, y)
        rho_xz, _ = spearman(x, z)
        rho_yz, _ = spearman(y, z)
        if any(math.isnan(v) for v in (rho_xy, rho_xz, rho_yz)):
            continue
        den = math.sqrt(max(0.0, (1 - rho_xz ** 2)) * max(0.0, (1 - rho_yz ** 2)))
        if den < 1e-8:                      # 控制变量完全解释了某一侧 → 偏相关无定义
            continue
        parts.append((rho_xy - rho_xz * rho_yz) / den)
        ns.append(len(triples))
    if not parts:
        return float("nan"), total
    pooled, _ = fisher_pool_test(parts, ns)
    return pooled, total


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


ANNOTATIONS_CSV = os.path.join(PROJECT_DIR, "data", "annotations", "finance_annotations.csv")


def load_batch_map(path: str = ANNOTATIONS_CSV) -> Dict[str, str]:
    """读标注集里的 batch 列：{qid: 批次标签}。

    批次标签记录每题的构造方式（v1_人工出题 / v2_块锚定）——两批的答案长度与难度系统性不同，
    分析时必须分批或把批次作为控制变量，否则会把批次差异读成"长度↔质量"。
    """
    out: Dict[str, str] = {}
    if not os.path.isfile(path):
        return out
    try:
        with open(path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                qid = (r.get("id") or "").strip()
                batch = (r.get("batch") or "").strip()
                if qid and batch:
                    out[qid] = batch
    except Exception:
        return {}
    return out


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
    batch_map = load_batch_map()
    if batch_map:
        print(f"批次信息：{len(batch_map)} 题带 batch 标签"
              f"（{'、'.join(sorted(set(batch_map.values())))}）")
    else:
        print("  [提示] 标注集没有 batch 列，按单批次分析（建议补列以区分构造方式）")

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
            "batch": batch_map.get(qid, ""),   # 批次是"题目"的属性，与分块方法无关
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
    batch_labels = sorted({b for b in (row.get("batch") for row in table) if b})
    if len(batch_labels) < 2:
        batch_labels = batch_labels or ["all"]
    base_keys = ["qid", "method", "batch", "judge_corr", "judge_faith", "doc_hit",
                 "claim_support", "evidence_gap", "ans_gold_sim", "quality_composite"]
    proxy_names = list(PROXY_DIRECTION)
    truth_names = list(TRUTH_LABEL)
    fieldnames = base_keys + proxy_names

    # ---- 有效性统计（批次内合并为准，池化值仅作对照）----
    validity: List[Dict] = []
    for truth in truth_names:
        fam_p = []
        fam_rows = []
        for p in proxy_names:
            def _pairs(rows_sub):
                out = []
                for row in rows_sub:
                    a, b = row.get(p), row.get(truth)
                    if a is None or b is None:
                        continue
                    if isinstance(a, float) and math.isnan(a):
                        continue
                    if isinstance(b, float) and math.isnan(b):
                        continue
                    out.append((a, b))
                return out

            pairs = _pairs(table)
            xs = [a for a, _ in pairs]
            ys = [b for _, b in pairs]
            rho_all, p_all = spearman(xs, ys)
            lo, hi = bootstrap_rho_ci(xs, ys) if len(xs) >= 8 else (float("nan"), float("nan"))
            # 分批次
            part = {}
            for b_label in batch_labels:
                sub = [row for row in table if row.get("batch") == b_label]
                bxs = [a for a, _ in _pairs(sub)]
                bys = [b for _, b in _pairs(sub)]
                b_rho, b_p = spearman(bxs, bys)
                part[b_label] = {"n": len(bxs), "rho": b_rho, "p": b_p,
                                 "level": variance_level(bxs)}
            rho_within, p_within = fisher_pool_test([part[b]["rho"] for b in batch_labels],
                                                    [part[b]["n"] for b in batch_labels])
            diff_p = rho_diff_p(part[batch_labels[0]]["rho"], part[batch_labels[0]]["n"],
                                part[batch_labels[1]]["rho"], part[batch_labels[1]]["n"]) \
                if len(batch_labels) == 2 else float("nan")
            # 长度控制 + 批次内残差化后的偏相关（消除"长度混淆"与"批次差异"）
            prho, pn = partial_rho_rank_from_rows(table, p, truth, "ans_length",
                                                 batch_key="batch")
            signs = [part[b]["rho"] for b in batch_labels
                     if not math.isnan(part[b]["rho"]) and part[b]["n"] >= 8]
            sign_consistent = bool(signs) and all(
                (r > 0) == (signs[0] > 0) and abs(r) >= 0.05 for r in signs)
            has_variance = all(part[b]["level"] != "degenerate" for b in batch_labels) \
                if batch_labels else True
            is_rare = any(part[b]["level"] == "rare" for b in batch_labels) \
                if batch_labels else False
            fam_rows.append({
                "truth": truth, "proxy": p, "n": len(xs),
                "rho": rho_all, "p": p_all, "ci_lo": lo, "ci_hi": hi,
                "rho_within": rho_within, "p_within": p_within,
                "batch_diff_p": diff_p,
                "rho_partial_len": prho, "n_partial": pn,
                "sign_consistent": sign_consistent, "has_variance": has_variance,
                "is_rare": is_rare,
                "expect": PROXY_DIRECTION[p],
                "sign_ok": (None if math.isnan(rho_within) or PROXY_DIRECTION[p] == 0
                            else (rho_within > 0) == (PROXY_DIRECTION[p] > 0)),
                "mean": float(np.mean(xs)) if xs else float("nan"),
                "sd": float(np.std(xs, ddof=1)) if len(xs) > 1 else float("nan"),
                "min": float(np.min(xs)) if xs else float("nan"),
                "p95": float(np.quantile(xs, 0.95)) if xs else float("nan"),
                "max": float(np.max(xs)) if xs else float("nan"),
                "n_missing": len(table) - len(xs),
                **{f"rho_{b}": part[b]["rho"] for b in batch_labels},
                **{f"n_{b}": part[b]["n"] for b in batch_labels},
            })
            fam_p.append(p_within)
        for row, adj in zip(fam_rows, holm(fam_p)):
            row["p_within_holm"] = adj
            row["p_holm"] = adj          # 兼容旧列名：显著性一律基于批次内合并
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
    # 判定一律基于"批次内合并"（池化值只作对照，避免把批次差异读成代理-质量关系）
    strong = [r for r in validity
              if not math.isnan(r["rho_within"]) and r["p_within_holm"] < 0.05]
    strong.sort(key=lambda r: -abs(r["rho_within"]))
    top_any = sorted([r for r in validity if not math.isnan(r["rho_within"])],
                     key=lambda r: -abs(r["rho_within"]))[:10]

    def proxy_verdict(r: Dict) -> str:
        """判定：批次内显著 + 两批同号 + 控制长度后不塌 + 分布非退化。

        注意长度派生指标：它们与真值口径同源（标准答案很短、精确匹配与 judge 都惩罚冗长），
        用"控制长度"无法自证，因此单列一类，不当作独立代理。
        """
        rho_w, p_w = r.get("rho_within", float("nan")), r.get("p_within_holm", float("nan"))
        prho = r.get("rho_partial_len", float("nan"))
        if not r.get("has_variance", True):
            return "⬜ 分布退化（批次内近乎常数，不可用）"
        if r["proxy"] in LENGTH_DERIVED:
            return "⚠️ 长度派生指标（与真值口径同源，不可独立使用）"
        rare_note = "（稀有事件，估计不稳）" if r.get("is_rare") else ""
        if not math.isnan(p_w) and p_w < 0.05:
            if not r.get("sign_consistent", False):
                return "⚠️ 批次内不一致（伪信号）"
            if math.isnan(prho):
                return "🟡 控制长度后退化（无法评估长度混淆）" + rare_note
            if abs(prho) < 0.10:
                return "⚠️ 控制长度后塌陷（长度混淆）"
            if math.copysign(1, prho) != math.copysign(1, rho_w):
                return "⚠️ 控制长度后符号翻转（长度混淆）"
            if r.get("sign_ok") is False:
                return "✅ 稳健但方向与直觉相反（按反向解读纳入）" + rare_note
            return "✅ 建议纳入" + rare_note
        if not math.isnan(r.get("p_holm", float("nan"))) and r["p_holm"] < 0.05:
            return "⚠️ 池化显著但批次内不成立（批次伪信号）"
        if not math.isnan(rho_w) and abs(rho_w) >= mde:
            return "🟡 效应量够但校正后不显著（扩样本再验）"
        return "⬜ 证据不足"

    best_by_proxy: Dict[str, Dict] = {}
    for r in validity:
        if math.isnan(r.get("rho_within", float("nan"))):
            continue
        cur = best_by_proxy.get(r["proxy"])
        if cur is None or abs(r["rho_within"]) > abs(cur["rho_within"]):
            best_by_proxy[r["proxy"]] = r

    # 批次构成（说明为什么必须分批看）
    batch_profile = []
    for b in batch_labels:
        sub = [row for row in table if row.get("batch") == b and row["method"] == args.method]
        if not sub:
            continue
        lens = [row["ans_length"] for row in sub if not math.isnan(row.get("ans_length", float("nan")))]
        corrs = [row["judge_corr"] for row in sub]
        batch_profile.append({
            "batch": b, "n": len(sub),
            "len_median": float(np.median(lens)) if lens else float("nan"),
            "corr_mean": float(np.mean(corrs)) if corrs else float("nan"),
            "ceiling": float(np.mean([c == 5 for c in corrs])) if corrs else float("nan"),
        })

    lines = [
        f"# E0 · 无标注代理指标有效性验证（{tag}）",
        "",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M')}；代理 {len(proxy_names)} 个 × 真值 {len(truth_names)} 个",
        f"- 样本量：**n={n_eff}**（{args.method} 分块的 {n_eff} 题）"
        + (f"；回答侧另有 {n_all - n_eff} 行来自其他分块方法" if n_all != n_eff else ""),
        f"- **判定口径：批次内 Fisher 合并 ρ + 同族 Holm 校正**；"
        f"池化 ρ 仅作对照（两个标注批次构造不同，池化会引入批次混淆）",
        f"- **最小可检测效应：|ρ| ≥ {mde:.2f}**（α=0.05 双侧）；"
        f"CI 为 {N_BOOT} 次配对 bootstrap 百分位区间",
        f"- 检索侧相似度类指标：{'已跳过（--no-embed）' if args.no_embed else '由 BGE 嵌入在线计算'}",
        "",
        "## 批次构成（为什么必须分批分析）",
        "",
        "| 批次 | 题数 | 答案长度中位数 | judge 正确性均值 | 满分占比 |",
        "|---|---|---|---|---|",
    ]
    for b in batch_profile:
        lines.append(f"| {b['batch']} | {b['n']} | {b['len_median']:.0f} | "
                     f"{b['corr_mean']:.2f} | {b['ceiling']:.1%} |")
    lines += [
        "",
        "> 两批次的答案长度与难度系统性不同 ⇒ 直接池化会把**批次差异**读成「长度↔质量」这类伪关系。",
        "",
        "## 真值分布（真值本身没有区分度，代理再准也测不出相关性）",
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
    lines += ["", "## 批次内一致性（判定依据）", "",
              "| 代理 | 真值 | 批次内合并 ρ | Holm p | " + " | ".join(batch_labels)
              + " | 批次差异 p | 池化 ρ（对照） | 控制长度后 ρ |",
              "|---|---|---|---|" + "---|" * len(batch_labels) + "---|---|---|"]
    for r in top_any:
        per = " | ".join(f"{r.get('rho_' + b, float('nan')):+.2f}" for b in batch_labels)
        dp = "" if math.isnan(r["batch_diff_p"]) else f"{r['batch_diff_p']:.3f}"
        pl = "" if math.isnan(r["rho_partial_len"]) else f"{r['rho_partial_len']:+.3f}"
        lines.append(f"| {r['proxy']} | {TRUTH_LABEL[r['truth']]} | {r['rho_within']:+.3f} | "
                     f"{r['p_within_holm']:.3f} | {per} | {dp} | {r['rho']:+.3f} | {pl} |")
    lines += ["", "## 指标建议表（进入 e-detector 的候选）", "",
              "| 代理 | 最强证据（真值 / 批次内 ρ / Holm p） | 控制长度后 | 均值 | p95 | 范围 | 判定 |",
              "|---|---|---|---|---|---|---|"]
    for p, r in sorted(best_by_proxy.items(),
                       key=lambda kv: -abs(kv[1]["rho_within"])):
        pj = "" if math.isnan(r.get("rho_partial_len", float("nan"))) \
            else f"{r['rho_partial_len']:+.3f}"
        lines.append(f"| {p} | {TRUTH_LABEL[r['truth']]} / {r['rho_within']:+.3f} / "
                     f"{r['p_within_holm']:.3f} | {pj} | "
                     f"{r['mean']:.3f} | {r['p95']:.3f} | {r['min']:.3f}~{r['max']:.3f} | "
                     f"{proxy_verdict(r)} |")
    sig = [r for r in strong if proxy_verdict(r).startswith("✅")]
    sig_proxies = sorted({r["proxy"] for r in sig})
    lines += ["", "## 结论", ""]
    if sig:
        lines.append(f"- **批次内稳健、且非长度派生的代理×真值对 {len(sig)} 个"
                     f"（涉及 {len(sig_proxies)} 个代理：{'、'.join(sig_proxies)}）**："
                     + "、".join(f"{r['proxy']}→{TRUTH_LABEL[r['truth']]}（ρ={r['rho_within']:+.3f}）"
                                for r in sig[:8]))
    else:
        lines.append("- 批次内没有任何代理通过 Holm 校正（且非长度派生/非退化）")
    lines.append(f"- 与「结论证据充分性」显著相关的代理："
                 f"{len([r for r in strong if r['truth'] == 'claim_support']) or '0 个'}"
                 f"（注意该真值只有 {len(audit)} 题，欠功效，不能当定论）")
    gold_sim_uniq = len({round(row['ans_gold_sim'], 3) for row in table
                         if not math.isnan(row.get('ans_gold_sim', float('nan')))})
    lines.append(f"- **真值分辨率仍是瓶颈**：judge 满分占比 "
                 f"{np.mean([row['judge_corr'] == 5 for row in table]):.1%}；"
                 + (f"唯一高区分度真值是答案↔标准答案相似度（{gold_sim_uniq} 个不同取值）"
                    if gold_sim_uniq else "本次未计算答案相似度（--no-embed）"))
    lines.append("- 行动建议：① 下一批标注**按难度分层**（简单/中等/困难各 1/3），"
                 "不要只用块锚定生成，以拉开真值区分度；"
                 "② 线上暴露 reranker 分数（提升 margin 类代理的分辨率）；"
                 "③ 区分「指标有效性（per-query）」与「窗口级敏感性」——"
                 "单条相关性弱不等于窗口均值对漂移不敏感，后者才是 e-detector 依赖的性质")
    if per_method:
        lines += ["", "## 多种分块下的稳健性（回答侧代理 → judge 正确性，Fisher-z 合并）", "",
                  "| 代理 | 合并 ρ | 各方法 ρ |", "|---|---|---|"]
        for r in sorted(per_method,
                        key=lambda x: -(abs(x["pooled_rho"]) if not math.isnan(x["pooled_rho"]) else 0))[:8]:
            pm = ", ".join(f"{m}:{'' if math.isnan(v) else f'{v:+.2f}'}"
                           for m, v in zip(methods, r["per_method"]))
            lines.append(f"| {r['proxy']} | {r['pooled_rho']:+.3f} | {pm} |")
    lines += ["", f"逐查询代理矩阵（e-detector 的输入）：`{os.path.basename(table_path)}`；"
                  f"完整统计：`{os.path.basename(valid_path)}`。",
              "",
              f"> 说明：本报告的所有显著性判定基于**批次内合并**的 ρ（列 `rho_within`）；"
              f"`rho` 列为池化值，仅用于展示批次混淆的幅度。"]
    open(summary_path, "w", encoding="utf-8").write("\n".join(lines))

    print(f"\n代理矩阵：{table_path}\n有效性统计：{valid_path}\n汇总：{summary_path}")
    print(f"MDE(|ρ|) = {mde:.3f} @ n={n_eff}；批次内稳健代理 = {len(sig)}"
          f"（池化显著 = {len([r for r in validity if r.get('p_holm', 1) < 0.05])}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
