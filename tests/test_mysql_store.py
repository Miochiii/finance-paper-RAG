# -*- coding: utf-8 -*-
"""MySQL 结构化分析层测试。

分四层：
  1. 纯函数：SQL 编译器（不连库）
  2. 降级：MySQL 不可用时所有入口返回安全值，不影响主流程
  3. 双引擎与双写：检索筛选走 SQL 白名单 / 内存掩码的选路，日志 jsonl+MySQL 双写
  4. 集成：连真实 MySQL，用独立测试库（rag_analytics_test）跑一遍完整 ETL 与筛选，
     并与 Python 内存掩码做一致性交叉验证 —— 结束后 DROP 测试库
"""
import json
import os
import time

import pytest

import rag_core.config as cfg
import rag_core.mysql_store as ms
from rag_core.retriever import HybridRetriever, match_meta_filter, normalize_filters


# --------------------------------------------------------------------------
# 1. SQL 编译器（纯函数）
# --------------------------------------------------------------------------
class TestSqlBuilder:
    def test_empty_filters_returns_base_query(self):
        sql, params = ms.build_doc_filter_sql({}, "金融论文")
        assert "WHERE c.name = %s" in sql
        assert params == ["金融论文"]

    def test_year_range(self):
        sql, params = ms.build_doc_filter_sql({"year_min": 2020, "year_max": 2023}, "X")
        assert "d.year >= %s" in sql and "d.year <= %s" in sql
        assert params == ["X", 2020, 2023]

    def test_tag_containment_semantics(self):
        """回归：标签筛选必须与 Python 侧一致（双向包含），不能退化为精确相等。"""
        sql, params = ms.build_doc_filter_sql({"methods": ["机器学习"]}, "X")
        assert "t.label LIKE %s" in sql           # w in v
        assert "LIKE CONCAT" in sql               # v in w（反向包含）
        assert "t.kind = %s" in sql
        assert params == ["X", "method", "%机器学习%", "机器学习"]

    def test_author_containment(self):
        sql, params = ms.build_doc_filter_sql({"authors": ["李"]}, "X")
        assert "d.author LIKE %s" in sql and "LIKE CONCAT" in sql
        assert params == ["X", "%李%", "李"]

    def test_multiple_values_or_joined(self):
        sql, params = ms.build_doc_filter_sql({"tasks": ["信贷风控", "交易策略"]}, "X")
        assert sql.count("t.label LIKE %s") == 2
        assert params == ["X", "task", "%信贷风控%", "信贷风控", "%交易策略%", "交易策略"]

    def test_tasks_use_task_kind(self):
        _, params = ms.build_doc_filter_sql({"tasks": ["债券"]}, "X")
        assert "task" in params and "method" not in params


# --------------------------------------------------------------------------
# 2. 降级（MySQL 不可用）
# --------------------------------------------------------------------------
class TestDegradation:
    def test_all_entries_safe_when_db_down(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("mysql down")
        monkeypatch.setattr(ms, "_connect", boom)
        assert ms.available() is False
        assert ms.stats()["ok"] is False
        assert ms.filter_docs_by_sql({"methods": ["机器学习"]}) is None
        assert ms.sync_corpus_meta("X", "no.json", "no.json")["ok"] is False
        assert ms.sync_search_logs("no_such.log")["ok"] is False
        assert ms.report_latency() == []
        assert ms.report_eval_compare() == []

    def test_empty_filters_skips_db(self, monkeypatch):
        """条件为空 → 直接返回 None（不筛选），不该连库。"""
        def boom(*a, **k):
            raise AssertionError("不应连接数据库")
        monkeypatch.setattr(ms, "_connect", boom)
        assert ms.filter_docs_by_sql({}) is None
        assert ms.filter_docs_by_sql(None) is None

    def test_insert_search_log_silent_when_db_down(self, monkeypatch):
        """实时落库失败必须静默（返回 False），不能把异常抛进检索主流程。"""
        def boom(*a, **k):
            raise RuntimeError("mysql down")
        monkeypatch.setattr(ms, "_connect", boom)
        assert ms.insert_search_log({"t": time.time(), "event": "search"}) is False

    def test_insert_search_log_ignores_non_search_events(self):
        """build/ingest 等事件不属于检索日志表，不该落库。"""
        assert ms.insert_search_log({"t": time.time(), "event": "build"}) is False
        assert ms.insert_search_log({}) is False
        assert ms.insert_search_log(None) is False


# --------------------------------------------------------------------------
# 3. 双引擎选路 + 日志双写（不连真库）
# --------------------------------------------------------------------------
class TestFilterEngine:
    """筛选掩码的两条路径：SQL 白名单优先，MySQL 不可用/镜像过期回退内存掩码。"""

    @staticmethod
    def _retriever():
        # 只测 _filter_mask：绕开 __init__ 的模型/索引加载
        r = HybridRetriever.__new__(HybridRetriever)
        r.chunks = ["c1", "c2", "c3", "c4"]
        r.metadatas = [{"source": "甲.pdf"}, {"source": "甲.pdf"},
                       {"source": "乙.pdf"}, {"source": "丙.pdf"}]
        r._doc_total = None
        assert r._doc_count() == 3
        return r

    def test_mysql_whitelist_becomes_chunk_mask(self, monkeypatch):
        monkeypatch.setattr(ms, "filter_docs_by_sql", lambda f, **k: ["甲.pdf", "丙.pdf"])
        mask, engine = self._retriever()._filter_mask({"methods": ["深度学习"]})
        assert engine == "mysql"
        assert mask == [True, True, False, True]

    def test_sql_empty_result_is_respected(self, monkeypatch):
        """SQL 明确查不到 → 空掩码（不是降级），检索层应直接返回空。"""
        monkeypatch.setattr(ms, "filter_docs_by_sql", lambda f, **k: [])
        mask, engine = self._retriever()._filter_mask({"tasks": ["不存在的任务"]})
        assert engine == "mysql" and not any(mask)

    def test_fallback_to_memory_mask_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(ms, "filter_docs_by_sql", lambda f, **k: None)
        r = self._retriever()
        f = normalize_filters({"authors": ["张"]})
        mask, engine = r._filter_mask({"authors": ["张"]})
        assert engine == "memory"
        assert mask == [match_meta_filter(m, f) for m in r.metadatas]

    def test_fallback_when_store_raises(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("boom")
        monkeypatch.setattr(ms, "filter_docs_by_sql", boom)
        mask, engine = self._retriever()._filter_mask({"year_min": 2020})
        assert engine == "memory" and mask is not None

    def test_no_filters_uses_no_engine(self, monkeypatch):
        def boom(*a, **k):
            raise AssertionError("无筛选条件时不该查库")
        monkeypatch.setattr(ms, "filter_docs_by_sql", boom)
        mask, engine = self._retriever()._filter_mask(None)
        assert mask is None and engine == "none"

    def test_expected_docs_passed_for_mirror_check(self, monkeypatch):
        """必须把当前 KB 的文献数带过去，让 SQL 侧能判断镜像是否过期。"""
        seen = {}

        def fake(filters, **kw):
            seen.update(kw)
            return ["甲.pdf"]

        monkeypatch.setattr(ms, "filter_docs_by_sql", fake)
        self._retriever()._filter_mask({"year_min": 2020})
        assert seen.get("expected_docs") == 3
        assert seen.get("corpus_name")      # 语料名必传（多语料库不能串台）


class TestLogDualWrite:
    """检索日志：jsonl 事实日志 + MySQL 实时落库，两者必须是同一条记录。"""

    def test_dual_write_same_timestamp(self, monkeypatch, work_tmp):
        import rag_server as core
        import rag_core.observability as obs
        log_file = os.path.join(work_tmp, "obs_dual.jsonl")
        monkeypatch.setattr(obs, "OBS_LOG_FILE", log_file)
        captured = []
        monkeypatch.setattr(ms, "insert_search_log",
                            lambda entry: captured.append(entry) or True)
        core._log_search_event("search", ok=True, total_ms=12.5, hits=3)
        lines = open(log_file, encoding="utf-8").read().strip().splitlines()
        assert len(lines) == 1
        written = json.loads(lines[0])
        assert written["event"] == "search" and written["total_ms"] == 12.5
        # 同一条记录（rid 相同）：ETL 补录时才能认出"这条已经实时写过了"
        assert len(captured) == 1
        assert captured[0]["rid"] == written["rid"] and captured[0]["t"] == written["t"]

    def test_mysql_failure_still_writes_jsonl(self, monkeypatch, work_tmp):
        import rag_server as core
        import rag_core.observability as obs
        log_file = os.path.join(work_tmp, "obs_dual_fail.jsonl")
        monkeypatch.setattr(obs, "OBS_LOG_FILE", log_file)

        def boom(entry):
            raise RuntimeError("mysql down")

        monkeypatch.setattr(ms, "insert_search_log", boom)
        core._log_search_event("ask", ok=True, total_ms=900.0, hits=5)
        assert os.path.isfile(log_file)
        assert json.loads(open(log_file, encoding="utf-8").read().strip())["event"] == "ask"

    def test_log_row_maps_engine_and_corpus(self):
        """行映射要带上筛选引擎、语料与 rid（面板报表按前两列自检/分组，rid 用于幂等）。"""
        entry = {"t": 1787721143.9, "rid": "abc123def456", "event": "search",
                 "total_ms": 100.0, "hits": 5,
                 "filters": True, "filter_engine": "mysql", "corpus": "金融论文",
                 "top_k": 5, "hyde": True, "mmr": False}
        row = dict(zip(ms._LOG_COLS, ms._log_row(entry, src_line=7, source="etl")))
        assert row["event_type"] == "search"
        assert row["filter_engine"] == "mysql" and row["corpus_name"] == "金融论文"
        assert row["use_hyde"] == 1 and row["use_mmr"] == 0 and row["has_filters"] == 1
        assert row["src_line"] == 7 and row["source"] == "etl"
        assert row["rid"] == "abc123def456"
        # 实时行没有 jsonl 行号，但 rid 完全相同（幂等键，与时间戳精度无关）
        live = dict(zip(ms._LOG_COLS, ms._log_row(entry)))
        assert live["src_line"] is None and live["source"] == "live"
        assert live["rid"] == row["rid"] and live["ts"] == row["ts"]

    def test_log_event_generates_unique_rid(self, monkeypatch, work_tmp):
        """每条日志自带唯一 rid：它才是幂等键（毫秒时间戳不足以区分同秒同事件）。"""
        import rag_core.observability as obs
        monkeypatch.setattr(obs, "OBS_LOG_FILE", os.path.join(work_tmp, "obs_rid.jsonl"))
        a = obs.log_event("search", total_ms=0.0, hits=1)
        b = obs.log_event("search", total_ms=0.0, hits=1)
        assert a["rid"] and b["rid"] and a["rid"] != b["rid"]
        assert a["t"] <= b["t"]


# --------------------------------------------------------------------------
# 3. 集成（真实 MySQL，独立测试库）
# --------------------------------------------------------------------------
def _live_ok() -> bool:
    try:
        conn = ms._connect(with_db=False)
        conn.close()
        return True
    except Exception:
        return False


requires_mysql = pytest.mark.skipif(not _live_ok(), reason="本机 MySQL 不可用")


def _make_corpus(work_tmp):
    """构造一份小语料（3 篇）+ 日志 + 评测 CSV，用于集成测试。"""
    d = os.path.join(work_tmp, "test_corpus")
    os.makedirs(d, exist_ok=True)
    meta = {
        "甲.pdf": {"year": 2021, "author": "张三", "methods": ["传统机器学习"], "tasks": ["信贷风控"]},
        "乙.pdf": {"year": 2023, "author": "李四", "methods": ["深度学习"], "tasks": ["交易策略"]},
        "丙.pdf": {"year": 2019, "author": "王五", "methods": ["统计方法"], "tasks": ["信贷风控"]},
    }
    kb = []
    for i, (src, m) in enumerate(meta.items()):
        for pg in (1, 2):
            kb.append({"source": src, "text": "x" * 10, "page_start": pg, "page_end": pg,
                       "source_type": "hmm", "pdf_path": f"/tmp/{src}",
                       "year": m["year"], "author": m["author"],
                       "methods": m["methods"], "tasks": m["tasks"]})
    meta_path = os.path.join(d, "doc_metadata.json")
    kb_path = os.path.join(d, "kb.json")
    obs_path = os.path.join(d, "obs.jsonl")
    res_dir = os.path.join(d, "results")
    os.makedirs(res_dir, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False)
    with open(obs_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"t": 1787721143.9, "event": "search", "ok": True,
                            "total_ms": 100.0, "hits": 5, "hyde": False, "mmr": True}) + "\n")
        f.write(json.dumps({"t": 1787721200.0, "event": "ask", "ok": True,
                            "total_ms": 900.0, "hits": 5, "hyde": True, "mmr": True,
                            "tokens_in": 100, "tokens_out": 50}) + "\n")
        f.write(json.dumps({"t": 1787721201.0, "event": "hmm_cache", "hit": True}) + "\n")  # 应被忽略
    with open(os.path.join(res_dir, "hmm_finance.csv"), "w", encoding="utf-8-sig", newline="") as f:
        f.write("qid,question,chunk_method,recall@5,mrr,ndcg@5,recall@5_c,mrr_c,ndcg@5_c,"
                "em,f1,gold_answer,pred_answer,gold_sources,judge_corr,judge_faith,judge_reason,error\n")
        f.write("fin_001,Q1,hmm,1.0,1.0,1.0,0.5,0.5,0.5,0,0.2,A,P,a.pdf,4,5,ok,\n")
    return {"meta": meta_path, "kb": kb_path, "obs": obs_path, "results": res_dir}


@requires_mysql
class TestIntegration:
    TEST_DB = "rag_analytics_test"

    @pytest.fixture(autouse=True)
    def _use_test_db(self, monkeypatch):
        monkeypatch.setattr(cfg, "MYSQL_DB", self.TEST_DB)
        yield
        try:  # 清理：删除测试库
            conn = ms._connect(with_db=False)
            with conn.cursor() as cur:
                cur.execute(f"DROP DATABASE IF EXISTS `{self.TEST_DB}`")
            conn.close()
        except Exception:
            pass

    def test_full_etl_and_filter_parity(self, work_tmp):
        paths = _make_corpus(work_tmp)
        schema = ms.ensure_schema()
        assert schema["ok"], schema.get("error")

        r1 = ms.sync_corpus_meta("测试语料", paths["meta"], paths["kb"])
        assert r1["ok"] and r1["documents"] == 3 and r1["tag_links"] == 6
        r2 = ms.sync_chunks("测试语料", paths["kb"])
        assert r2["ok"] and r2["chunks"] == 6
        r3 = ms.sync_eval_results(paths["results"], run_tag="test")
        assert r3["ok"] and r3["rows"] == 1
        r4 = ms.sync_search_logs(paths["obs"])
        assert r4["ok"] and r4["inserted"] == 2          # hmm_cache 被忽略
        r5 = ms.sync_search_logs(paths["obs"])           # 幂等：第二次不重复插
        assert r5["inserted"] == 0

        st = ms.stats()
        assert st["ok"]
        assert st["rows"]["dim_document"] == 3
        assert st["rows"]["fact_chunk"] == 6
        assert st["rows"]["fact_eval"] == 1
        assert st["rows"]["fact_search_log"] == 2

        # 结构化筛选：SQL 与 Python 掩码结果一致
        kb = json.load(open(paths["kb"], encoding="utf-8"))
        for case in ({"year_min": 2020},
                     {"methods": ["机器学习"]},       # 双向包含：应命中"传统机器学习"
                     {"tasks": ["信贷风控"], "year_max": 2020},
                     {"authors": ["李"]}):
            sql_docs = set(ms.filter_docs_by_sql(case, "测试语料") or [])
            f = normalize_filters(case)
            py_docs = {c["source"] for c in kb if match_meta_filter(c, f)}
            assert sql_docs == py_docs, f"筛选不一致: {case} SQL={sql_docs} Py={py_docs}"

        # 报表入口可跑通，且返回值必须可 JSON 序列化（回归：MySQL 的 ROUND/AVG 返回 Decimal）
        lat = ms.report_latency()
        assert isinstance(lat, list)
        assert isinstance(ms.report_tag_distribution(), list)
        cmp_rows = ms.report_eval_compare("test")
        assert isinstance(cmp_rows, list)
        json.dumps(lat, ensure_ascii=False)
        json.dumps(cmp_rows, ensure_ascii=False)
        json.dumps(ms.stats(), ensure_ascii=False)
        if cmp_rows:
            assert isinstance(cmp_rows[0]["recall5"], float)

    def test_live_insert_then_etl_dedupe(self, work_tmp):
        """实时落库 + jsonl 补录：同一条记录只能算一次（否则报表翻倍）。"""
        paths = _make_corpus(work_tmp)
        assert ms.ensure_schema()["ok"]

        entry = {"t": 1787722000.5, "rid": "live0001probe000", "event": "search", "ok": True,
                 "total_ms": 123.4, "hits": 4, "hyde": False, "mmr": True, "top_k": 5,
                 "filters": True, "filter_engine": "mysql", "corpus": "测试语料"}
        assert ms.insert_search_log(entry) is True
        st = ms.stats()
        assert st["rows"]["fact_search_log"] == 1
        assert st["log_source"].get("live") == 1
        assert st["filter_engine"].get("mysql") == 1

        # 同一条记录也躺在 jsonl 里（实时写成功后的正常状态）→ 补录必须跳过
        with open(paths["obs"], "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        r = ms.sync_search_logs(paths["obs"])
        assert r["ok"] and r["skipped_dup"] == 1
        assert r["inserted"] == 2                       # jsonl 里另外两条正常事件
        st = ms.stats()
        assert st["rows"]["fact_search_log"] == 3       # 不是 4
        assert st["log_source"] == {"live": 1, "etl": 2}

        # 再来一次补录：行号幂等（该行已扫过），什么都不插
        again = ms.sync_search_logs(paths["obs"])
        assert again["inserted"] == 0 and again["skipped_dup"] == 0

        # 带 rid 的行只有那一行（jsonl 里的老记录没有 rid，靠 uk_src_line 幂等）
        conn = ms._connect()
        with conn.cursor() as cur:
            cur.execute("""SELECT COUNT(*), COUNT(DISTINCT rid) FROM fact_search_log
                           WHERE rid IS NOT NULL""")
            total, distinct = cur.fetchone()
            assert total == distinct == 1

    def test_filter_guard_degrades_when_mirror_stale(self, work_tmp):
        """镜像过期必须降级（返回 None 让调用方走内存掩码），不能当成"查不到"。"""
        paths = _make_corpus(work_tmp)
        assert ms.ensure_schema()["ok"]
        ms.sync_corpus_meta("测试语料", paths["meta"], paths["kb"])

        case = {"methods": ["深度学习"]}
        assert ms.filter_docs_by_sql(case, "不存在的语料") is None      # 未同步 → 降级
        assert ms.filter_docs_by_sql(case, "测试语料", expected_docs=99) is None  # 文库数不符 → 降级
        got = ms.filter_docs_by_sql(case, "测试语料", expected_docs=3)
        assert got == ["乙.pdf"]                                        # 一致 → 正常出结果
