# -*- coding: utf-8 -*-
"""rag_server 纯函数测试：上下文扩展 / 链接化 / 增量状态分类 / 可观测聚合。"""
import json
import os
import types

import rag_server as core


def _mock_retriever():
    return types.SimpleNamespace(
        chunks=["块0-第5页", "块1-第6页(命中)", "块2-第7页(也命中)", "块3-第10页(太远)"],
        metadatas=[
            {"source": "论文.pdf", "page_start": 5, "page_end": 5},
            {"source": "论文.pdf", "page_start": 6, "page_end": 6},
            {"source": "论文.pdf", "page_start": 7, "page_end": 7},
            {"source": "论文.pdf", "page_start": 10, "page_end": 10},
        ],
    )


def test_build_context_expansion_dedup_page_guard():
    core._state["retriever"] = _mock_retriever()
    results = [
        {"index": 1, "text": "块1-第6页(命中)", "metadata": {"source": "论文.pdf", "page_start": 6, "page_end": 6}},
        {"index": 2, "text": "块2-第7页(也命中)", "metadata": {"source": "论文.pdf", "page_start": 7, "page_end": 7}},
    ]
    ctx, cites, used = core._build_context(results, max_chars=20000)
    assert "同篇相邻段落" in ctx
    assert "块0-第5页" in ctx            # prev 邻居
    assert "块3-第10页" not in ctx        # 页码间距 >1 排除
    assert ctx.count("块2-第7页(也命中)") == 1  # 命中块不作相邻段重复
    assert cites == ["[来源1] 论文.pdf，第6页"]


def test_linkify_answer_six_tuple():
    a = core._linkify_answer("见[来源1]。", [("[来源1]", "论文.pdf", "，第6页", 6, 6, 1)])
    assert a == "见[[来源1]](http://127.0.0.1:3080/dsh-rag/open?doc=%E8%AE%BA%E6%96%87.pdf&page=6)。"


def test_classify_inputs():
    state = {"mode": "hmm", "docs": {
        "a.pdf": {"hash": "old_a", "mode": "hmm"},
        "b.pdf": {"hash": "old_b", "mode": "hmm"},
    }}
    hashes = {"a.pdf": "new_a", "c.pdf": "new_c"}
    new, changed, unchanged, removed = core._classify_inputs(state, hashes, "hmm")
    assert new == ["c.pdf"] and changed == ["a.pdf"] and removed == ["b.pdf"] and unchanged == []


def test_classify_inputs_all_unchanged():
    state = {"mode": "hmm", "docs": {"a.pdf": {"hash": "h1", "mode": "hmm"}}}
    hashes = {"a.pdf": "h1"}
    new, changed, unchanged, removed = core._classify_inputs(state, hashes, "hmm")
    assert new == [] and changed == [] and unchanged == ["a.pdf"] and removed == []


def test_ingest_state_roundtrip(work_tmp):
    core.INGEST_STATE_FILE = os.path.join(work_tmp, "kb.json.ingest.json")
    core._save_ingest_state({"mode": "hmm", "docs": {"a.pdf": {"hash": "x", "mode": "hmm"}}})
    st = core._load_ingest_state()
    assert st["mode"] == "hmm" and st["docs"]["a.pdf"]["hash"] == "x"


def test_file_hash_stable(work_tmp):
    p = os.path.join(work_tmp, "f.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write("内容")
    assert core._file_hash(p) == core._file_hash(p)
    assert len(core._file_hash(p)) == 40


def test_observability_summarize(work_tmp):
    from rag_core import observability
    observability.OBS_LOG_FILE = os.path.join(work_tmp, "obs.jsonl")
    observability.log_event("ask", ok=True, qp_ms=10, retrieve_ms=100, generate_ms=500,
                            bm25_ms=30, vector_ms=50, rerank_ms=20,
                            total_ms=610, tokens_in=1000, tokens_out=200)
    observability.log_event("search", ok=True, retrieve_ms=80, bm25_ms=30, vector_ms=40, rerank_ms=10)
    observability.log_event("hmm_cache", kind="chunk", hit=True, doc="a.pdf")
    observability.log_event("hmm_cache", kind="embed", hit=False, doc="a.pdf")
    s = observability.summarize()
    assert s["asks"] == 1 and s["searches"] == 1
    assert s["avg_ms"]["generate"] == 500.0
    assert s["cache"]["chunk_hit"] == 1 and s["cache"]["embed_miss"] == 1
    assert s["cache"]["chunk_hit_rate"] == 1.0
    assert s["tokens"] == {"in": 1000, "out": 200}
    assert s["cost_cny"] > 0
    # 文件确实落盘
    with open(observability.OBS_LOG_FILE, encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 4
    assert json.loads(lines[0])["event"] == "ask"


def test_unified_mcp_mounted():
    # 一体化服务：rag_server 同时持有 FastAPI app 与 FastMCP 实例，lifespan 已交接
    assert core.mcp is not None
    assert core._mcp_app is not None
    assert core.app.router.lifespan_context is not None
    assert any(getattr(r, "path", "") == "/health" for r in core.app.routes)
