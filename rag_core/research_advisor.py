# -*- coding: utf-8 -*-
"""研究方向工作台：方向辅助（单方向分析）+ 方向选取（多方向对比排序）。

能力：
  analyze_direction(direction)   —— 检索本地文献证据 + LLM 生成方向卡片
                                    （创新点/可行性/风险/空白/切入建议）
  compare_directions(directions) —— 2~5 个方向的量化对比：
                                    文献支撑度 / 近年热度 / 方法成熟度 / 创新空间（本地统计）
                                    + 数据可得性 / 总体可行性（LLM 评分）
                                    → 加权总分排序（权重可自定义）

依赖：混合检索（rag_server.get_retriever）+ 文档元数据（doc_metadata.json，年份/方法/任务）。
"""

import json
import os
import re
from typing import Dict, List, Optional

# 默认权重：gap（创新空间）与 feasibility（LLM 总体可行性）占大头
DEFAULT_WEIGHTS = {
    "literature": 0.15,   # 文献支撑度（相关文档数，越多越成熟）
    "recency": 0.10,      # 近年热度（2023 及以后文献占比）
    "maturity": 0.10,     # 方法成熟度（相关文献覆盖的方法种类）
    "gap": 0.25,          # 创新空间（文献越少越开放）
    "data": 0.15,         # 数据可得性（LLM 评分）
    "feasibility": 0.25,  # 总体可行性（LLM 评分）
}

_ANALYSIS_PROMPT = """你是金融机器学习领域的研究方向顾问。给定候选研究方向与本地文献库的检索证据（含每篇的年份/方法/任务与摘要片段），输出 JSON：

{
  "summary": "一句话概述该方向",
  "innovations": ["可能的创新点1", "可能的创新点2", ...],
  "feasibility": 5,
  "data_score": 5,
  "risks": ["风险1", "风险2", ...],
  "gap": "文献库中可见的空白或可借鉴点（1~2 句）",
  "suggested_plan": "切入建议（2~3 句，含可用方法与评价指标）"
}

要求：
- innovations 2~4 个，具体可检验（方法/场景/指标）；
- feasibility / data_score 为 1~5 整数（5 最高）；
- risks 2~4 条；
- 基于给定证据作答，证据不足时如实说明。
只输出 JSON，不要任何解释文字。"""


def _llm_json(system: str, user: str, timeout: int = 90) -> Dict:
    from openai import OpenAI
    api_key = os.getenv("deepseek_api", "")
    if not api_key:
        raise RuntimeError("未配置 deepseek_api")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.2,
        stream=False,
        timeout=timeout,
    )
    raw = (resp.choices[0].message.content or "").strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if m:
        raw = m.group(1)
    return json.loads(raw)


# --------------------------------------------------------------------------
# 检索统计：方向 → 相关文献证据（复用混合检索 + 文档元数据）
# --------------------------------------------------------------------------
def _retrieve_stats(direction: str, top_k: int = 12) -> Dict:
    import rag_server as core
    from rag_core.doc_metadata import load_meta

    retriever = core.get_retriever()
    results = retriever.retrieve(direction, bm25_k=20, vector_k=20, rerank_k=top_k)
    meta = load_meta()

    docs_map: Dict[str, Dict] = {}
    excerpts: List[str] = []
    for r in results:
        m = r.get("metadata") or {}
        src = m.get("source", "未知")
        d = docs_map.setdefault(src, {"source": src, "n_chunks": 0, "pages": []})
        d["n_chunks"] += 1
        ps = m.get("page_start")
        if ps is not None:
            d["pages"].append(ps)
        dm = meta.get(src, {})
        d["year"] = dm.get("year")
        d["author"] = dm.get("author")
        d["methods"] = dm.get("methods", [])
        d["tasks"] = dm.get("tasks", [])
        d["excerpt"] = (r.get("text") or "")[:180]
        excerpts.append((src, (r.get("text") or "")[:220]))

    docs = sorted(docs_map.values(), key=lambda d: -d["n_chunks"])
    for d in docs:
        d["pages"] = sorted(set(d["pages"]))[:5]
    method_counter: Dict[str, int] = {}
    task_counter: Dict[str, int] = {}
    for d in docs:
        for m in d["methods"]:
            method_counter[m] = method_counter.get(m, 0) + 1
        for t in d["tasks"]:
            task_counter[t] = task_counter.get(t, 0) + 1
    return {
        "doc_count": len(docs),
        "docs": docs,
        "methods": sorted(method_counter.items(), key=lambda x: -x[1]),
        "tasks": sorted(task_counter.items(), key=lambda x: -x[1]),
        "excerpts": excerpts[:6],
    }


def _features(stats: Dict) -> Dict:
    """本地统计特征（不依赖 LLM，纯函数便于测试）。"""
    docs = stats.get("docs", [])
    n = len(docs)
    years = [d.get("year") for d in docs if d.get("year")]
    recency = sum(1 for y in years if y and y >= 2023) / len(years) if years else 0.0
    methods = {m for d in docs for m in d.get("methods", [])}
    maturity = min(1.0, len(methods) / 6.0)
    return {
        "docs": n,
        "recency": round(recency, 3),
        "maturity": round(maturity, 3),
        "distinct_methods": len(methods),
    }


def _evidence_text(stats: Dict) -> str:
    methods = stats.get("methods") or []
    tasks = stats.get("tasks") or []
    lines = [f"相关文献 {stats.get('doc_count', 0)} 篇；"
             f"方法分布 {methods[:6]}；任务分布 {tasks[:6]}"]
    for d in (stats.get("docs") or [])[:8]:
        year = d.get("year") or "?"
        lines.append(
            f"- [{year}] {d['source']}（作者 {d.get('author') or '?'}；"
            f"方法 {'/'.join((d.get('methods') or [])[:3]) or '?'}；任务 {'/'.join((d.get('tasks') or [])[:2]) or '?'}）"
        )
    for src, ex in (stats.get("excerpts") or [])[:4]:
        lines.append(f"  片段[{src[:24]}...]: {ex}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 对外接口
# --------------------------------------------------------------------------
def analyze_direction(direction: str, top_k: int = 12) -> Dict:
    """单个研究方向：本地文献证据 + LLM 方向卡片。"""
    stats = _retrieve_stats(direction, top_k)
    if not stats["docs"]:
        return {
            "ok": True, "direction": direction,
            "stats": stats, "analysis": None,
            "note": "本地文献库未检索到强相关文献（可考虑导入更多文献或换关键词）。",
        }
    analysis = _llm_json(_ANALYSIS_PROMPT, f"候选研究方向：{direction}\n\n检索证据：\n{_evidence_text(stats)}")
    # 数值字段兜底
    analysis["feasibility"] = int(analysis.get("feasibility", 3) or 3)
    analysis["data_score"] = int(analysis.get("data_score", 3) or 3)
    return {"ok": True, "direction": direction, "stats": stats, "analysis": analysis}


def _score_breakdown(features: Dict, lit_norm: float, llm: Dict, weights: Dict) -> Dict:
    gap = round(1 - lit_norm, 3)
    parts = {
        "literature": round(lit_norm * weights["literature"], 4),
        "recency": round(features["recency"] * weights["recency"], 4),
        "maturity": round(features["maturity"] * weights["maturity"], 4),
        "gap": round(gap * weights["gap"], 4),
        "data": round((int(llm.get("data_score", 3) or 3) / 5) * weights["data"], 4),
        "feasibility": round((int(llm.get("feasibility", 3) or 3) / 5) * weights["feasibility"], 4),
    }
    return {"parts": parts, "total": round(sum(parts.values()), 4)}


def compare_directions(directions: List[str], weights: Optional[Dict] = None,
                       top_k: int = 12) -> Dict:
    """2~5 个候选方向的量化对比与排序。"""
    directions = [d.strip() for d in directions if d and d.strip()]
    if len(directions) < 2:
        return {"ok": False, "error": "至少提供 2 个候选方向（用列表传入）"}
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        for k in list(w):
            if k in weights:
                w[k] = float(weights[k])
    total_w = sum(w.values())
    w = {k: v / total_w for k, v in w.items()}  # 归一化，容忍用户权重和不为 1

    per = []
    for d in directions:
        stats = _retrieve_stats(d, top_k)
        if not stats["docs"]:
            llm = {"summary": "未检索到相关文献", "innovations": [], "feasibility": 1,
                   "data_score": 1, "risks": ["文献支撑不足"],
                   "gap": "本地库缺乏相关文献", "suggested_plan": "建议先导入相关文献或更换表述"}
        else:
            llm = _llm_json(_ANALYSIS_PROMPT,
                            f"候选研究方向：{d}\n\n检索证据：\n{_evidence_text(stats)}")
        llm["feasibility"] = int(llm.get("feasibility", 3) or 3)
        llm["data_score"] = int(llm.get("data_score", 3) or 3)
        per.append({"direction": d, "stats": stats, "analysis": llm})

    max_docs = max(p["stats"]["doc_count"] for p in per) or 1
    for p in per:
        f = _features(p["stats"])
        lit_norm = round(p["stats"]["doc_count"] / max_docs, 3)
        s = _score_breakdown(f, lit_norm, p["analysis"], w)
        p["features"] = {**f, "literature": lit_norm, "gap": round(1 - lit_norm, 3)}
        p["scores"] = s
    per.sort(key=lambda p: -p["scores"]["total"])
    return {
        "ok": True,
        "weights": w,
        "ranking": [
            {"rank": i + 1, "direction": p["direction"],
             "score": p["scores"]["total"], "summary": p["analysis"].get("summary", "")}
            for i, p in enumerate(per)
        ],
        "results": per,
        "recommendation": {
            "direction": per[0]["direction"],
            "reason": per[0]["analysis"].get("gap", "") + " " + per[0]["analysis"].get("suggested_plan", ""),
        },
    }
