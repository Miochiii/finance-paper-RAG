# -*- coding: utf-8 -*-
"""MySQL 结构化分析层测试。

分三层：
  1. 纯函数：SQL 编译器（不连库）
  2. 降级：MySQL 不可用时所有入口返回安全值，不影响主流程
  3. 集成：连真实 MySQL，用独立测试库（rag_analytics_test）跑一遍完整 ETL 与筛选，
     并与 Python 内存掩码做一致性交叉验证 —— 结束后 DROP 测试库
"""
import json
import os

import pytest

import rag_core.config as cfg
import rag_core.mysql_store as ms
from rag_core.retriever import match_meta_filter, normalize_filters


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

        # 报表入口可跑通
        assert isinstance(ms.report_latency(), list)
        assert isinstance(ms.report_tag_distribution(), list)
        assert isinstance(ms.report_eval_compare("test"), list)
