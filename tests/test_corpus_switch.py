# -*- coding: utf-8 -*-
"""多语料切换（第二步）测试：全部在 work_tmp 假布局上，不碰真实数据。"""
import json
import os

import rag_core.config as cfg
import rag_core.corpus as cp
import rag_server as core
import rag_core.survey as sv
import rag_core.doc_metadata as dm
import rag_core.retriever as rt


def _fake_corpora(monkeypatch, work_tmp):
    """monkeypatch 语料总目录到 work_tmp 下的独立子目录并返回该目录。"""
    import uuid
    root = os.path.join(work_tmp, "corpora_" + uuid.uuid4().hex[:8])
    monkeypatch.setattr(cfg, "CORPORA_DIR", root)
    return root


class TestCorpusApi:
    def test_validate_name(self):
        assert cp.validate_name("金融论文") is None
        assert cp.validate_name("语料 A2-下划线") is None
        assert cp.validate_name("") is not None
        assert cp.validate_name("../穿越") is not None
        assert cp.validate_name("a/b") is not None
        assert cp.validate_name("x" * 41) is not None

    def test_create_and_switch(self, monkeypatch, work_tmp):
        root = _fake_corpora(monkeypatch, work_tmp)
        r = cp.create("新语料A")
        assert r["ok"] and os.path.isdir(os.path.join(root, "新语料A"))
        assert cp.create("新语料A")["ok"] is False          # 重复创建
        assert cp.create("../穿越")["ok"] is False          # 非法名
        assert cp.switch("新语料A")["ok"]
        assert cp.read_current() == "新语料A"
        assert cp.switch("不存在的语料")["ok"] is False

    def test_runtime_paths_follow_current(self, monkeypatch, work_tmp):
        root = _fake_corpora(monkeypatch, work_tmp)
        cp.create("语料B")
        cp.switch("语料B")
        p = cp.runtime_paths()
        assert p["kb"] == os.path.join(root, "语料B", "knowledge_base.json")
        assert p["active"] == "语料B"
        assert p["ingest"] == p["kb"] + ".ingest.json"

    def test_runtime_paths_env_override_wins(self, monkeypatch, work_tmp):
        _fake_corpora(monkeypatch, work_tmp)
        monkeypatch.setenv("RAG_KB_FILE", os.path.join(work_tmp, "override.json"))
        p = cp.runtime_paths()
        assert p["kb"] == os.path.join(work_tmp, "override.json")

    def test_list_corpora(self, monkeypatch, work_tmp):
        _fake_corpora(monkeypatch, work_tmp)
        cp.create("语料C")
        p = cp.paths("语料C")
        with open(p["kb"], "w", encoding="utf-8") as f:
            json.dump([{"source": "a.pdf", "text": "x"}], f)
        items = cp.list_corpora()
        assert any(it["name"] == "语料C" and it["chunks"] == 1 for it in items)

    def test_runtime_paths_legacy_fallback(self, monkeypatch, work_tmp):
        """兼容模式：无 current.json 且无默认语料目录 → 沿用 config 默认路径。"""
        _fake_corpora(monkeypatch, work_tmp)
        cfg_kb = os.path.join(work_tmp, "cfg_kb.json")
        monkeypatch.setattr(cfg, "KB_FILE", cfg_kb)
        monkeypatch.setattr(cfg, "VECTOR_DB_PATH", os.path.join(work_tmp, "cfg_vdb"))
        monkeypatch.setattr(cfg, "DOCS_DIR", os.path.join(work_tmp, "cfg_docs"))
        monkeypatch.setattr(cfg, "MINERU_OUT", os.path.join(work_tmp, "cfg_mineru"))
        monkeypatch.setattr(cfg, "DOC_META_FILE", os.path.join(work_tmp, "cfg_meta.json"))
        monkeypatch.setattr(cfg, "SURVEY_DIR", os.path.join(work_tmp, "cfg_surveys"))
        p = cp.runtime_paths()
        assert p["kb"] == cfg_kb
        assert p["docs"] == os.path.join(work_tmp, "cfg_docs")


class TestServerCorpusIntegration:
    def test_switch_refreshes_globals_and_state(self, monkeypatch, work_tmp):
        root = _fake_corpora(monkeypatch, work_tmp)
        cp.create("语料D")
        p = cp.paths("语料D")
        with open(p["kb"], "w", encoding="utf-8") as f:
            json.dump([{"source": "a.pdf", "text": "x"}], f, ensure_ascii=False)
        core._state["retriever"] = None
        r = core.switch_corpus_kb("语料D")
        assert r["ok"] and r["active"] == "语料D"
        assert core.KB_FILE == p["kb"]
        assert core.VECTOR_DB_PATH == p["vector_db"]
        assert sv.SURVEY_DIR == p["surveys"]
        assert dm.DOC_META_FILE == p["meta"]
        assert core._state["kb_fp"] is None
        # 收尾：恢复真实语料路径（monkeypatch 撤销后按真实 current.json 重解析）
        monkeypatch.undo()
        core._refresh_paths()

    def test_switch_invalid_name(self, monkeypatch, work_tmp):
        _fake_corpora(monkeypatch, work_tmp)
        r = core.switch_corpus_kb("不存在的语料")
        assert r["ok"] is False

    def test_list_and_create_endpoints(self, monkeypatch, work_tmp):
        root = _fake_corpora(monkeypatch, work_tmp)
        r = core.create_corpus_kb("语料E")
        assert r["ok"] and not r["building"]
        lst = core.list_corpora_kb()
        assert any(it["name"] == "语料E" and it["build"]["status"] == "empty" for it in lst["corpora"])
        # 不带 mineru_out 的 create 不触发后台建库
        assert core._BUILD_JOBS.get("语料E") is None
        monkeypatch.undo()
        core._refresh_paths()


class TestVocabAndSurveys:
    def test_meta_vocab(self, monkeypatch, work_tmp):
        meta_file = os.path.join(work_tmp, "meta.json")
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump({
                "a.pdf": {"year": 2022, "methods": ["机器学习", "集成学习"], "tasks": ["信贷风控"]},
                "b.pdf": {"year": 2020, "methods": ["机器学习"], "tasks": ["交易策略"]},
            }, f, ensure_ascii=False)
        monkeypatch.setattr(dm, "DOC_META_FILE", meta_file)
        v = core.meta_vocab_kb()
        assert v["ok"] and v["docs"] == 2
        labels = {x["label"] for x in v["methods"]}
        assert labels == {"机器学习", "集成学习"}
        assert v["methods"][0]["count"] == 2          # 按频次降序：机器学习在前
        assert v["years"] == [2020, 2022]

    def test_survey_list(self, monkeypatch, work_tmp):
        sdir = os.path.join(work_tmp, "surveys")
        os.makedirs(os.path.join(sdir, "主题甲"), exist_ok=True)
        with open(os.path.join(sdir, "主题甲", "draft.md"), "w", encoding="utf-8") as f:
            f.write("x")
        os.makedirs(os.path.join(sdir, "主题乙"), exist_ok=True)
        monkeypatch.setattr(sv, "SURVEY_DIR", sdir)
        r = core.survey_list_kb()
        assert r["ok"] and {t["topic"] for t in r["topics"]} == {"主题甲", "主题乙"}
        assert any(t["topic"] == "主题甲" and t["has_draft"] for t in r["topics"])


class TestDictPerCorpus:
    def test_dict_fallback_and_corpus_dict(self, monkeypatch, work_tmp):
        root = _fake_corpora(monkeypatch, work_tmp)
        os.makedirs(os.path.join(root, cp.DEFAULT_NAME), exist_ok=True)
        # 无 current.json、无语料 dict.txt → 回退内置金融词典
        assert rt._dict_file().endswith(os.path.join("rag_core", "finance_dict.txt"))
        # 语料里有 dict.txt → 用语料词典
        d = os.path.join(root, cp.DEFAULT_NAME, "dict.txt")
        with open(d, "w", encoding="utf-8") as f:
            f.write("自定义术语 50000\n")
        assert rt._dict_file() == d
        # 切换后路径跟随（当前激活变化）
        cp.create("语料F")
        cp.switch("语料F")
        assert rt._dict_file().endswith(os.path.join("rag_core", "finance_dict.txt"))
        # 复原：切回默认并清理 current.json（在 work_tmp 内，无真实影响）
        cp.switch(cp.DEFAULT_NAME)
