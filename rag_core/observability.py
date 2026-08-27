# -*- coding: utf-8 -*-
"""可观测性：事件日志（JSONL）+ 聚合统计（延迟分解 / 缓存命中 / 成本估算）。

用法：
    from rag_core.observability import log_event, Timer, summarize
    t = Timer()
    ... 业务 ...
    log_event("ask", retrieve_ms=..., generate_ms=...)
    summarize()  # -> {asks, avg_ms, cache, cost_cny, ...}

日志文件：环境变量 RAG_OBS_LOG 可覆盖（默认项目内 data/observability.jsonl）。
"""

import json
import os
import threading
import time
from typing import Dict, Optional

from rag_core.config import OBS_LOG

OBS_LOG_FILE = OBS_LOG

# 成本估算（元 / 百万 token，DeepSeek 谷价近似；缓存命中未细分，输入统一按 miss 计）
_PRICE_IN_PER_M = 1.5
_PRICE_OUT_PER_M = 4.5

_lock = threading.Lock()


def log_event(event: str, **fields) -> None:
    """追加一条事件（JSON 行）。失败静默（日志不可用不影响主流程）。"""
    entry: Dict = {"t": time.time(), "event": event}
    entry.update(fields)
    try:
        os.makedirs(os.path.dirname(OBS_LOG_FILE), exist_ok=True)
        line = json.dumps(entry, ensure_ascii=False)
        with _lock:
            with open(OBS_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except OSError:
        pass


class Timer:
    """毫秒级计时器。"""

    def __init__(self):
        self.t0 = time.perf_counter()

    def ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000.0


def _avg(total: float, n: int) -> Optional[float]:
    return round(total / n, 1) if n else None


def summarize(limit: int = 5000) -> Dict:
    """聚合最近 limit 条日志：问答/检索延迟分解、缓存命中、成本估算。"""
    try:
        with open(OBS_LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()[-limit:]
    except OSError:
        lines = []

    asks = searches = builds = ingests = 0
    s_qp = s_retr = s_gen = 0.0
    s_bm25 = s_vec = s_rerank = 0.0
    s_total_ask = 0.0
    tok_in = tok_out = 0
    chunk_hit = chunk_miss = 0
    embed_hit = embed_miss = 0
    ask_err = 0

    for line in lines:
        try:
            e = json.loads(line)
        except Exception:
            continue
        ev = e.get("event")
        if ev == "ask":
            asks += 1
            if not e.get("ok"):
                ask_err += 1
            s_qp += float(e.get("qp_ms") or 0)
            s_retr += float(e.get("retrieve_ms") or 0)
            s_gen += float(e.get("generate_ms") or 0)
            s_bm25 += float(e.get("bm25_ms") or 0)
            s_vec += float(e.get("vector_ms") or 0)
            s_rerank += float(e.get("rerank_ms") or 0)
            s_total_ask += float(e.get("total_ms") or 0)
            tok_in += int(e.get("tokens_in") or 0)
            tok_out += int(e.get("tokens_out") or 0)
        elif ev == "search":
            searches += 1
            s_retr += float(e.get("retrieve_ms") or 0)
            s_bm25 += float(e.get("bm25_ms") or 0)
            s_vec += float(e.get("vector_ms") or 0)
            s_rerank += float(e.get("rerank_ms") or 0)
        elif ev == "hmm_cache":
            hit = bool(e.get("hit"))
            if e.get("kind") == "chunk":
                chunk_hit += 1 if hit else 0
                chunk_miss += 0 if hit else 1
            else:
                embed_hit += 1 if hit else 0
                embed_miss += 0 if hit else 1
        elif ev == "build":
            builds += 1
        elif ev == "ingest":
            ingests += 1

    n_retr = asks + searches
    chunk_total = chunk_hit + chunk_miss
    embed_total = embed_hit + embed_miss
    cost = tok_in / 1e6 * _PRICE_IN_PER_M + tok_out / 1e6 * _PRICE_OUT_PER_M

    return {
        "events_logged": len(lines),
        "asks": asks,
        "ask_errors": ask_err,
        "searches": searches,
        "builds": builds,
        "ingests": ingests,
        "avg_ms": {
            "ask_total": _avg(s_total_ask, asks),
            "query_rewrite": _avg(s_qp, asks),
            "retrieve": _avg(s_retr, n_retr),
            "bm25": _avg(s_bm25, n_retr),
            "vector": _avg(s_vec, n_retr),
            "rerank": _avg(s_rerank, n_retr),
            "generate": _avg(s_gen, asks),
        },
        "tokens": {"in": tok_in, "out": tok_out},
        "cost_cny": round(cost, 4),
        "cache": {
            "chunk_hit": chunk_hit,
            "chunk_miss": chunk_miss,
            "chunk_hit_rate": round(chunk_hit / chunk_total, 3) if chunk_total else None,
            "embed_hit": embed_hit,
            "embed_miss": embed_miss,
            "embed_hit_rate": round(embed_hit / embed_total, 3) if embed_total else None,
        },
    }
