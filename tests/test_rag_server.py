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
    assert a == "见[来源1](http://127.0.0.1:3080/dsh-rag/open?doc=%E8%AE%BA%E6%96%87.pdf&page=6)。"


def test_linkify_answer_clean_label_no_extra_brackets():
    """回归：链接文字必须是干净的 [来源1]，不能是 [[来源1]]——
    旧实现在 label 外层多套一层方括号，导致只有第一个链接可点。"""
    used = [("[来源1]", "论文.pdf", "，第6页", 6, 6, 1)]
    a = core._linkify_answer("见[来源1]。", used)
    assert "[[来源1]]" not in a
    assert "[来源1](http" in a


def test_linkify_answer_model_double_brackets_normalized():
    """模型若输出 [[来源1]] / 【来源2】，也归一化成干净链接。"""
    used = [("[来源1]", "论文.pdf", "，第6页", 6, 6, 1), ("[来源2]", "论文.pdf", "，第7页", 7, 7, 2)]
    a = core._linkify_answer("[[来源1]]与【来源2】。", used)
    assert a.count("dsh-rag/open") == 2
    assert "[[来源" not in a and "【来源" not in a
    assert "[来源1](http" in a and "[来源2](http" in a


def test_linkify_answer_repeated_label_all_linked():
    """同一 [来源1] 出现多次时，每一处都必须变成链接（历史 bug：只点得动第一个）。"""
    used = [("[来源1]", "论文.pdf", "，第6页", 6, 6, 1)]
    a = core._linkify_answer("第1处[来源1]，第2处[来源1]。", used)
    assert a.count("dsh-rag/open") == 2


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


def test_editor_routes_registered_and_serve():
    """综述编辑器（第一步）：/editor 页面路由 + /survey/editor_data 数据路由。"""
    paths = [getattr(r, "path", "") for r in core.app.routes]
    assert "/editor" in paths and "/survey/editor_data" in paths
    resp = core.editor_page()
    assert resp.status_code == 200
    assert resp.path == core.EDITOR_HTML
    assert os.path.isfile(core.EDITOR_HTML)
    html = open(core.EDITOR_HTML, encoding="utf-8").read()
    assert 'id="ta"' in html and "renderMD" in html


def test_survey_editor_data_missing_topic():
    r = core.survey_editor_data_http("不存在主题-xyz-123")
    assert r["ok"] is False


def test_rewrite_selection_route_registered_and_validation():
    paths = [getattr(r, "path", "") for r in core.app.routes]
    assert "/survey/rewrite_selection" in paths
    req = core.RewriteSelectionReq(topic="不存在主题-xyz-123", selected_text="足够长的一段选中文字内容")
    r = core.survey_rewrite_selection_http(req)
    assert r["ok"] is False  # 无草稿：不调 LLM 直接报错


def test_editor_save_route_registered_and_validation():
    paths = [getattr(r, "path", "") for r in core.app.routes]
    assert "/survey/editor_save" in paths
    req = core.EditorSaveReq(topic="不存在主题-xyz-123", text="x")
    r = core.survey_editor_save_http(req)
    assert r["ok"] is False  # 无草稿：不落盘直接报错


def test_docx_routes_registered_and_validation():
    paths = [getattr(r, "path", "") for r in core.app.routes]
    assert "/survey/export_docx" in paths and "/survey/download" in paths
    req = core.EditorDocxReq(topic="不存在主题-xyz-123", text="正文")
    r = core.survey_export_docx_http(req)
    assert r["ok"] is False  # 无草稿：不生成文件直接报错


def test_open_doc_kb_falls_back_when_pdf_path_stale(monkeypatch, work_tmp):
    """回归：KB 里 pdf_path 过期（目录迁移后）时，按文件名重新定位，
    而不是直接拿着旧路径去打开报"文件不存在"。"""
    import rag_core.pdf_open as po
    kb_file = os.path.join(work_tmp, "kb_stale.json")
    with open(kb_file, "w", encoding="utf-8") as f:
        json.dump([{"source": "论文A.pdf",
                    "pdf_path": os.path.join(work_tmp, "old", "论文A.pdf"),
                    "text": "x"}], f, ensure_ascii=False)
    real_pdf = os.path.join(work_tmp, "docs", "论文A.pdf")
    os.makedirs(os.path.dirname(real_pdf), exist_ok=True)
    with open(real_pdf, "w", encoding="utf-8") as f:
        f.write("x")
    monkeypatch.setattr(core, "KB_FILE", kb_file)
    monkeypatch.setattr(core, "PDF_SOURCE_DIRS", [os.path.dirname(real_pdf)])
    opened = {}
    monkeypatch.setattr(po, "open_pdf_page", lambda p, page=1: opened.update(path=p) or {"ok": True})
    r = core.open_doc_kb("论文A.pdf", 3)
    assert r["ok"] and opened["path"] == real_pdf


def test_build_jobs_persist_and_interrupt(monkeypatch, work_tmp):
    """构建任务状态落盘：保存后重载，残留的 running 标记为 interrupted。"""
    f = os.path.join(work_tmp, "bj.json")
    monkeypatch.setattr(core, "_BUILD_JOBS_FILE", f)
    monkeypatch.setattr(core, "_BUILD_JOBS", {})
    core._BUILD_JOBS["语料A"] = {"status": "running", "msg": "构建中…"}
    core._save_build_jobs()
    monkeypatch.setattr(core, "_BUILD_JOBS", {})
    core._load_build_jobs()
    assert core._BUILD_JOBS["语料A"]["status"] == "interrupted"
