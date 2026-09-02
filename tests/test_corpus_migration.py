# -*- coding: utf-8 -*-
"""rag_core/corpus.py 迁移逻辑测试（全部在 work_tmp 假布局上，不碰真实数据）。"""
import json
import os

import rag_core.config as cfg
import rag_core.corpus as cp


def _fake_layout(monkeypatch, work_tmp):
    """构造旧布局（data/ 根）并让 config/corpus 指向它。每次调用独立子目录。"""
    import uuid
    data = os.path.join(work_tmp, "data_" + uuid.uuid4().hex[:8])
    kb = os.path.join(data, "knowledge_base.json")
    meta = os.path.join(data, "doc_metadata.json")
    vdb = os.path.join(data, "vector_db")
    docs = os.path.join(data, "docs")
    mineru = os.path.join(data, "mineru_out")
    surveys = os.path.join(data, "surveys")
    for d in (vdb, docs, os.path.join(mineru, "batch"), surveys):
        os.makedirs(d, exist_ok=True)
    with open(os.path.join(vdb, "st.txt"), "w") as f:
        f.write("x")
    with open(os.path.join(docs, "a.pdf"), "w") as f:
        f.write("x")
    with open(os.path.join(surveys, "s.txt"), "w") as f:
        f.write("x")
    with open(kb, "w", encoding="utf-8") as f:
        json.dump([{"source": "a.pdf", "text": "t",
                    "pdf_path": os.path.join(docs, "a.pdf")}], f, ensure_ascii=False)
    with open(meta, "w", encoding="utf-8") as f:
        json.dump({}, f)
    monkeypatch.setattr(cfg, "DATA_DIR", data)
    monkeypatch.setattr(cfg, "CORPORA_DIR", os.path.join(data, "corpora"))
    monkeypatch.setattr(cfg, "KB_FILE", kb)
    monkeypatch.setattr(cfg, "VECTOR_DB_PATH", vdb)
    monkeypatch.setattr(cfg, "DOCS_DIR", docs)
    monkeypatch.setattr(cfg, "MINERU_OUT", os.path.join(mineru, "batch"))
    monkeypatch.setattr(cfg, "DOC_META_FILE", meta)
    monkeypatch.setattr(cfg, "SURVEY_DIR", surveys)
    monkeypatch.setattr(cp, "_health_reachable", lambda: False)
    return {"data": data, "kb": kb, "docs": docs}


def test_plan_migration_lists_moves(monkeypatch, work_tmp):
    fake = _fake_layout(monkeypatch, work_tmp)
    plan = cp.plan_migration()
    assert len(plan) == 6   # kb/ingest(缺失会跳过) → kb、meta、vdb、docs、mineru、surveys
    assert any(w == "知识库" for _, _, w in plan)
    assert fake["kb"] in [s for s, _, _ in plan]


def test_migrate_moves_writes_current_patches_pdf(monkeypatch, work_tmp):
    fake = _fake_layout(monkeypatch, work_tmp)
    r = cp.migrate()
    assert r["ok"], r.get("error")
    assert len(r["moved"]) == 6
    assert cp.read_current() == cp.DEFAULT_NAME
    p = cp.paths(cp.DEFAULT_NAME)
    assert os.path.isfile(p["kb"]) and not os.path.exists(fake["kb"])
    assert os.path.isdir(p["vector_db"]) and os.path.isdir(p["docs"])
    assert os.path.isdir(p["mineru_out"]) and os.path.isdir(p["surveys"])
    assert os.path.isfile(p["meta"])
    # KB 内 pdf_path 已刷新为语料 docs 目录
    kb = json.load(open(p["kb"], encoding="utf-8"))
    assert kb[0]["pdf_path"] == os.path.join(p["docs"], "a.pdf")


def test_migrate_conflict_aborts(monkeypatch, work_tmp):
    fake = _fake_layout(monkeypatch, work_tmp)
    os.makedirs(os.path.join(fake["data"], "corpora", cp.DEFAULT_NAME, "docs"))
    r = cp.migrate()
    assert r["ok"] is False and "已存在" in r["error"]
    assert os.path.isfile(fake["kb"])  # 源未被移动


def test_migrate_refuses_while_service_running(monkeypatch, work_tmp):
    _fake_layout(monkeypatch, work_tmp)
    monkeypatch.setattr(cp, "_health_reachable", lambda: True)
    r = cp.migrate()
    assert r["ok"] is False and "8000" in r["error"]
    r2 = cp.migrate(force=True)   # force 跳过检测
    assert r2["ok"]


def test_migrate_idempotent(monkeypatch, work_tmp):
    _fake_layout(monkeypatch, work_tmp)
    assert cp.migrate()["ok"]
    r2 = cp.migrate()
    assert r2["ok"] and r2.get("moved") == 0   # 已迁移：无待搬项
