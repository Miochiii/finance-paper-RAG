# -*- coding: utf-8 -*-
"""rag_core/mysql_store.py —— MySQL 结构化分析层（"双引擎"里的结构化一半）。

职责：
  1. **建库建表**：星型模型
       dim_corpus / dim_document / dim_tag（维表）
       rel_doc_tag（文献↔标签桥表）
       fact_chunk / fact_search_log / fact_eval（事实表）
       etl_state（ETL 增量状态）
       两个分析视图 v_tag_year / v_hyde_mmr_latency
  2. **ETL**：doc_metadata.json / knowledge_base.json / observability.jsonl /
     results/*.csv → MySQL（幂等，可重复执行）
  3. **结构化筛选**：filter_docs_by_sql() 用 SQL 查"符合条件（年份/作者/方法/任务）
     的文献白名单"，交给检索层做掩码 —— 结构化筛选下沉数据库
  4. **统计与报表**：stats() / report_*() 供面板与面试演示

设计原则：
  - **JSON 仍是知识库真源**，MySQL 是分析镜像（日志则以 MySQL 为真源）；
  - **全链路降级**：MySQL 不可用时所有函数返回安全默认值（available()/None/空），
    绝不影响检索问答主流程；
  - 连接不缓存长连接（本地场景调用频率低），每次操作短连接 + autocommit。
"""

import json
import os
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from rag_core import config


def _num(v):
    """Decimal → float：MySQL 的 ROUND/AVG 返回 Decimal，json 需要原始类型。"""
    if isinstance(v, Decimal):
        return float(v)
    return v

# --------------------------------------------------------------------------
# 连接
# --------------------------------------------------------------------------
def _connect(with_db: bool = True, db: Optional[str] = None):
    """建立 pymysql 短连接；调用方负责关闭。"""
    import pymysql
    kwargs = dict(
        host=config.MYSQL_HOST,
        port=config.MYSQL_PORT,
        user=config.MYSQL_USER,
        password=config.MYSQL_PASSWORD,
        charset="utf8mb4",
        autocommit=True,
        connect_timeout=5,
    )
    if with_db:
        kwargs["database"] = db or config.MYSQL_DB
    return pymysql.connect(**kwargs)


def available(db: Optional[str] = None) -> bool:
    """MySQL 是否可用（含目标库是否存在）。失败返回 False（调用方降级）。"""
    try:
        conn = _connect(with_db=True, db=db)
        conn.close()
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# 建库建表（DDL）
# --------------------------------------------------------------------------
_DDL_TABLES = [
    # ---- 维表 ----
    """CREATE TABLE IF NOT EXISTS dim_corpus (
        corpus_id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(100) NOT NULL UNIQUE,
        is_active TINYINT(1) NOT NULL DEFAULT 0,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",

    """CREATE TABLE IF NOT EXISTS dim_document (
        doc_id INT AUTO_INCREMENT PRIMARY KEY,
        corpus_id INT NOT NULL,
        source VARCHAR(255) NOT NULL,
        title VARCHAR(255),
        author VARCHAR(100),
        year SMALLINT,
        pdf_path VARCHAR(500),
        pages SMALLINT,
        chunk_count INT DEFAULT 0,
        UNIQUE KEY uk_corpus_source (corpus_id, source),
        KEY idx_year (year),
        KEY idx_author (author),
        CONSTRAINT fk_doc_corpus FOREIGN KEY (corpus_id)
            REFERENCES dim_corpus (corpus_id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",

    """CREATE TABLE IF NOT EXISTS dim_tag (
        tag_id INT AUTO_INCREMENT PRIMARY KEY,
        kind ENUM('method','task') NOT NULL,
        label VARCHAR(100) NOT NULL,
        UNIQUE KEY uk_kind_label (kind, label)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",

    """CREATE TABLE IF NOT EXISTS rel_doc_tag (
        doc_id INT NOT NULL,
        tag_id INT NOT NULL,
        PRIMARY KEY (doc_id, tag_id),
        KEY idx_tag (tag_id),
        CONSTRAINT fk_rel_doc FOREIGN KEY (doc_id)
            REFERENCES dim_document (doc_id) ON DELETE CASCADE,
        CONSTRAINT fk_rel_tag FOREIGN KEY (tag_id)
            REFERENCES dim_tag (tag_id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",

    # ---- 事实表 ----
    """CREATE TABLE IF NOT EXISTS fact_chunk (
        chunk_id INT PRIMARY KEY,
        doc_id INT NOT NULL,
        page_start SMALLINT,
        page_end SMALLINT,
        source_type VARCHAR(20),
        char_len INT,
        KEY idx_doc (doc_id),
        KEY idx_page (page_start),
        CONSTRAINT fk_chunk_doc FOREIGN KEY (doc_id)
            REFERENCES dim_document (doc_id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",

    """CREATE TABLE IF NOT EXISTS fact_search_log (
        log_id BIGINT AUTO_INCREMENT PRIMARY KEY,
        rid VARCHAR(16),
        ts DATETIME(3) NOT NULL,
        event_type VARCHAR(20) NOT NULL,
        corpus_name VARCHAR(100),
        top_k SMALLINT,
        use_hyde TINYINT(1) DEFAULT 0,
        use_mmr TINYINT(1) DEFAULT 1,
        has_filters TINYINT(1) DEFAULT 0,
        filter_engine VARCHAR(8),
        total_ms FLOAT, qp_ms FLOAT, retrieve_ms FLOAT,
        bm25_ms FLOAT, vector_ms FLOAT, rerank_ms FLOAT,
        mmr_ms FLOAT, hyde_ms FLOAT, generate_ms FLOAT,
        hits SMALLINT, ok TINYINT(1) DEFAULT 1,
        tokens_in INT, tokens_out INT,
        src_line INT NULL,
        source VARCHAR(8) NOT NULL DEFAULT 'etl',
        UNIQUE KEY uk_rid (rid),
        UNIQUE KEY uk_src_line (src_line),
        KEY idx_ts_event (ts, event_type),
        KEY idx_ts (ts),
        KEY idx_event (event_type),
        KEY idx_hyde_mmr (use_hyde, use_mmr)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",

    """CREATE TABLE IF NOT EXISTS fact_eval (
        eval_id BIGINT AUTO_INCREMENT PRIMARY KEY,
        run_tag VARCHAR(50) NOT NULL,
        qid VARCHAR(30) NOT NULL,
        method VARCHAR(20) NOT NULL,
        question TEXT,
        recall5 FLOAT, mrr FLOAT, ndcg5 FLOAT,
        recall5_c FLOAT, mrr_c FLOAT, ndcg5_c FLOAT,
        em FLOAT, f1 FLOAT, judge_corr FLOAT, judge_faith FLOAT,
        UNIQUE KEY uk_run_qid_method (run_tag, qid, method),
        KEY idx_method (method)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",

    """CREATE TABLE IF NOT EXISTS etl_state (
        k VARCHAR(50) PRIMARY KEY,
        v VARCHAR(200),
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
]

_DDL_VIEWS = [
    # 标签 × 年份的文献分布（JOIN + GROUP BY 演示）
    """CREATE OR REPLACE VIEW v_tag_year AS
       SELECT t.kind, t.label, d.year, COUNT(*) AS doc_cnt
       FROM rel_doc_tag r
       JOIN dim_tag t ON t.tag_id = r.tag_id
       JOIN dim_document d ON d.doc_id = r.doc_id
       GROUP BY t.kind, t.label, d.year""",

    # 检索延迟分位 + 开关对比（窗口函数演示）
    """CREATE OR REPLACE VIEW v_hyde_mmr_latency AS
       SELECT use_hyde, use_mmr, event_type,
              COUNT(*) AS n,
              ROUND(AVG(total_ms), 1) AS avg_ms,
              ROUND(MAX(CASE WHEN pct <= 0.95 THEN total_ms END), 1) AS p95_ms,
              ROUND(AVG(hits), 2) AS avg_hits
       FROM (SELECT *, PERCENT_RANK() OVER (
                     PARTITION BY use_hyde, use_mmr, event_type
                     ORDER BY total_ms) AS pct
             FROM fact_search_log
             WHERE total_ms IS NOT NULL) x
       GROUP BY use_hyde, use_mmr, event_type""",
]


def _has_column(cur, table: str, column: str) -> bool:
    cur.execute("""SELECT COUNT(*) FROM information_schema.columns
                   WHERE table_schema = DATABASE() AND table_name = %s
                     AND column_name = %s""", (table, column))
    return bool(cur.fetchone()[0])


def _has_index(cur, table: str, index: str) -> bool:
    cur.execute("""SELECT COUNT(*) FROM information_schema.statistics
                   WHERE table_schema = DATABASE() AND table_name = %s
                     AND index_name = %s""", (table, index))
    return bool(cur.fetchone()[0])


def _migrate(cur) -> List[str]:
    """老库结构升级（幂等）：P2 起支持"实时落库"，需要区分来源并与 ETL 补录去重。

    P1 的 fact_search_log 只服务 ETL 补录（src_line 非空且唯一）；
    P2 检索时直接写库：这种行没有 jsonl 行号，所以 src_line 放开为可空
    （MySQL 唯一索引允许多个 NULL，ETL 的幂等性不受影响）。
    两条路径的去重靠 rid（日志记录自带唯一 id）：实时落库与补录写的是同一条记录，
    补录时按 rid 反查即可精确跳过——时间戳只有毫秒精度，当幂等键会误判
    （实测历史数据里就有同秒同耗时的记录，唯一键根本建不起来）。
    """
    applied = []
    stmts = []
    if not _has_column(cur, "fact_search_log", "source"):
        stmts.append(("source 列",
                      "ALTER TABLE fact_search_log ADD COLUMN source VARCHAR(8) "
                      "NOT NULL DEFAULT 'etl'"))
    if not _has_column(cur, "fact_search_log", "filter_engine"):
        stmts.append(("filter_engine 列",
                      "ALTER TABLE fact_search_log ADD COLUMN filter_engine VARCHAR(8)"))
    if not _has_column(cur, "fact_search_log", "rid"):
        stmts.append(("rid 幂等键",
                      "ALTER TABLE fact_search_log ADD COLUMN rid VARCHAR(16)"))
    if not _has_index(cur, "fact_search_log", "uk_rid"):
        stmts.append(("uk_rid 唯一索引",
                      "ALTER TABLE fact_search_log ADD UNIQUE KEY uk_rid (rid)"))
    if _has_index(cur, "fact_search_log", "uk_src_line"):
        stmts.append(("src_line 放开为可空",
                      "ALTER TABLE fact_search_log MODIFY COLUMN src_line INT NULL"))
    if not _has_index(cur, "fact_search_log", "idx_ts_event"):
        stmts.append(("补录去重索引 idx_ts_event",
                      "ALTER TABLE fact_search_log ADD KEY idx_ts_event (ts, event_type)"))
    # 毫秒精度：实时行与 ETL 行要在同一秒内也能精确对齐（P1 建的库是秒精度）
    cur.execute("""SELECT DATETIME_PRECISION FROM information_schema.columns
                   WHERE table_schema = DATABASE() AND table_name = 'fact_search_log'
                     AND column_name = 'ts'""")
    row = cur.fetchone()
    if row and (row[0] or 0) < 3:
        stmts.append(("ts 毫秒精度",
                      "ALTER TABLE fact_search_log MODIFY COLUMN ts DATETIME(3) NOT NULL"))
    for label, ddl in stmts:
        try:
            cur.execute(ddl)
            applied.append(label)
        except Exception as e:          # 单项失败不阻塞其余迁移
            applied.append(f"{label}(跳过: {str(e)[:60]})")
    return applied


def ensure_schema(db: Optional[str] = None) -> Dict:
    """建库 + 建表 + 建视图（幂等）。返回 {ok, db, tables, views, error}。"""
    dbname = db or config.MYSQL_DB
    try:
        conn = _connect(with_db=False)
        with conn.cursor() as cur:
            cur.execute(f"CREATE DATABASE IF NOT EXISTS `{dbname}` "
                        f"DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        conn.close()
        conn = _connect(with_db=True, db=dbname)
        with conn.cursor() as cur:
            for ddl in _DDL_TABLES:
                cur.execute(ddl)
            migrations = _migrate(cur)
            errors = []
            for ddl in _DDL_VIEWS:
                try:
                    cur.execute(ddl)
                except Exception as e:      # 视图失败不阻塞（老版本 MySQL 兼容）
                    errors.append(str(e)[:120])
        conn.close()
        return {"ok": True, "db": dbname,
                "tables": len(_DDL_TABLES), "views": len(_DDL_VIEWS),
                "migrations": migrations, "view_errors": errors}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


# --------------------------------------------------------------------------
# ETL
# --------------------------------------------------------------------------
def _load_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _f(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _i(v, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _title_of(source: str, author: Optional[str]) -> str:
    t = source[:-4] if source.lower().endswith(".pdf") else source
    if author and t.endswith("_" + author):
        t = t[: -(len(author) + 1)]
    return t


def _state_get(cur, key: str, default: int = 0) -> int:
    cur.execute("SELECT v FROM etl_state WHERE k=%s", (key,))
    row = cur.fetchone()
    return _i(row[0], default) if row else default


def _state_set(cur, key: str, value) -> None:
    cur.execute("INSERT INTO etl_state (k, v) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE v=VALUES(v)", (key, str(value)))


# --------------------------------------------------------------------------
# 检索日志：行映射（实时落库与 ETL 补录共用）
# --------------------------------------------------------------------------
_LOG_COLS = ("rid", "ts", "event_type", "corpus_name", "top_k", "use_hyde", "use_mmr",
             "has_filters", "filter_engine", "total_ms", "qp_ms", "retrieve_ms",
             "bm25_ms", "vector_ms", "rerank_ms", "mmr_ms", "hyde_ms",
             "generate_ms", "hits", "ok", "tokens_in", "tokens_out",
             "src_line", "source")

_LOG_IDX = {c: i for i, c in enumerate(_LOG_COLS)}

_INSERT_LOG_SQL = ("INSERT IGNORE INTO fact_search_log ("
                   + ", ".join(_LOG_COLS) + ") VALUES ("
                   + ", ".join(["%s"] * len(_LOG_COLS)) + ")")


def _log_row(entry: Dict, src_line: Optional[int] = None,
             source: str = "live") -> Tuple:
    """observability 事件 → fact_search_log 一行。

    实时落库和 ETL 补录都走这里，字段映射严格一致（尤其是 rid 与 t）：
    补录时按 rid 反查就能知道"这条是不是已经实时写过了"（见 _rid_exists）。
    """
    ts = entry.get("t")
    try:
        ts_dt = datetime.fromtimestamp(float(ts)) if ts else datetime.now()
    except (TypeError, ValueError):
        ts_dt = datetime.now()
    return (
        (entry.get("rid") or None),
        ts_dt, str(entry.get("event") or "")[:20],
        (entry.get("corpus") or None),
        _i(entry.get("top_k")),
        1 if entry.get("hyde") else 0,
        0 if entry.get("mmr") is False else 1,
        1 if entry.get("filters") else 0,
        (entry.get("filter_engine") or None),
        _f(entry.get("total_ms")), _f(entry.get("qp_ms")), _f(entry.get("retrieve_ms")),
        _f(entry.get("bm25_ms")), _f(entry.get("vector_ms")), _f(entry.get("rerank_ms")),
        _f(entry.get("mmr_ms")), _f(entry.get("hyde_ms")), _f(entry.get("generate_ms")),
        _i(entry.get("hits")), 1 if entry.get("ok", True) else 0,
        _i(entry.get("tokens_in")), _i(entry.get("tokens_out")),
        src_line, source,
    )


def _rid_exists(cur, rid: Optional[str]) -> bool:
    """该日志记录（rid）是否已经落库？实时落库与 ETL 补录共用这一幂等键。"""
    if not rid:
        return False                      # 老日志没有 rid → 交给 uk_src_line 幂等
    cur.execute("SELECT 1 FROM fact_search_log WHERE rid = %s LIMIT 1", (rid,))
    return cur.fetchone() is not None


def insert_search_log(entry: Dict) -> bool:
    """检索/问答日志实时落库（结构化分析层），失败静默返回 False。

    设计取舍：jsonl 仍是"事实日志"（进程崩了也不丢），MySQL 是分析镜像。
    实时写入让面板报表即时反映行为；写失败（MySQL 没起/网络断）不影响检索主流程，
    之后用 sync_search_logs 补录即可——两条路径互为兜底并按 rid 去重。
    """
    if not entry or entry.get("event") not in ("search", "ask"):
        return False
    conn = None
    try:
        conn = _connect()
        with conn.cursor() as cur:
            cur.execute(_INSERT_LOG_SQL, _log_row(entry, src_line=None, source="live"))
        conn.close()
        return True
    except Exception:
        if conn:
            try:
                conn.close()
            except Exception:
                pass
        return False


def sync_corpus_meta(corpus_name: str, meta_path: str, kb_path: str) -> Dict:
    """同步语料/文献/标签维表 + 桥表（幂等 upsert）。"""
    meta = _load_json(meta_path, {})
    kb = _load_json(kb_path, [])
    if not meta:
        return {"ok": False, "error": f"元数据为空或不存在: {meta_path}"}

    # 从 KB 汇总每篇的块数与最大页码
    stats: Dict[str, Dict] = {}
    for c in kb:
        s = c.get("source") or ""
        d = stats.setdefault(s, {"chunks": 0, "pages": 0, "pdf_path": None})
        d["chunks"] += 1
        pe = _i(c.get("page_end"), 0) or 0
        d["pages"] = max(d["pages"], pe)
        d["pdf_path"] = d["pdf_path"] or c.get("pdf_path")

    try:
        conn = _connect()
        with conn.cursor() as cur:
            cur.execute("INSERT INTO dim_corpus (name) VALUES (%s) "
                        "ON DUPLICATE KEY UPDATE name=VALUES(name)", (corpus_name,))
            cur.execute("SELECT corpus_id FROM dim_corpus WHERE name=%s", (corpus_name,))
            corpus_id = cur.fetchone()[0]

            n_doc = n_rel = 0
            for source, m in meta.items():
                st = stats.get(source, {})
                author = (m.get("author") or "").strip() or None
                cur.execute(
                    """INSERT INTO dim_document
                       (corpus_id, source, title, author, year, pdf_path, pages, chunk_count)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                       ON DUPLICATE KEY UPDATE
                         title=VALUES(title), author=VALUES(author), year=VALUES(year),
                         pdf_path=VALUES(pdf_path), pages=VALUES(pages),
                         chunk_count=VALUES(chunk_count)""",
                    (corpus_id, source, _title_of(source, author), author,
                     _i(m.get("year")), st.get("pdf_path"),
                     st.get("pages") or None, st.get("chunks", 0)))
                n_doc += 1
                cur.execute("SELECT doc_id FROM dim_document WHERE corpus_id=%s AND source=%s",
                            (corpus_id, source))
                doc_id = cur.fetchone()[0]

                # 标签：方法/任务 → dim_tag + rel_doc_tag
                cur.execute("DELETE FROM rel_doc_tag WHERE doc_id=%s", (doc_id,))
                for kind, key in (("method", "methods"), ("task", "tasks")):
                    for label in (m.get(key) or []):
                        label = str(label).strip()
                        if not label:
                            continue
                        cur.execute("INSERT INTO dim_tag (kind, label) VALUES (%s,%s) "
                                    "ON DUPLICATE KEY UPDATE label=VALUES(label)",
                                    (kind, label))
                        cur.execute("SELECT tag_id FROM dim_tag WHERE kind=%s AND label=%s",
                                    (kind, label))
                        tag_id = cur.fetchone()[0]
                        cur.execute("INSERT IGNORE INTO rel_doc_tag (doc_id, tag_id) VALUES (%s,%s)",
                                    (doc_id, tag_id))
                        n_rel += 1
        conn.close()
        return {"ok": True, "corpus": corpus_name, "documents": n_doc, "tag_links": n_rel}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


def sync_chunks(corpus_name: str, kb_path: str) -> Dict:
    """同步块级事实表 fact_chunk（chunk_id = KB 下标）。"""
    kb = _load_json(kb_path, [])
    if not kb:
        return {"ok": False, "error": f"知识库为空或不存在: {kb_path}"}
    try:
        conn = _connect()
        with conn.cursor() as cur:
            cur.execute("SELECT corpus_id FROM dim_corpus WHERE name=%s", (corpus_name,))
            row = cur.fetchone()
            if not row:
                conn.close()
                return {"ok": False, "error": f"语料未同步: {corpus_name}（先 sync_corpus_meta）"}
            corpus_id = row[0]
            cur.execute("""SELECT doc_id, source FROM dim_document WHERE corpus_id=%s""",
                        (corpus_id,))
            doc_map = {src: did for did, src in cur.fetchall()}

            cur.execute("""SELECT d.doc_id FROM dim_document d
                           WHERE d.corpus_id=%s""", (corpus_id,))
            cur.execute("""DELETE FROM fact_chunk WHERE doc_id IN
                           (SELECT doc_id FROM dim_document WHERE corpus_id=%s)""",
                        (corpus_id,))
            n = 0
            for idx, c in enumerate(kb):
                did = doc_map.get(c.get("source") or "")
                if not did:
                    continue
                cur.execute("""INSERT INTO fact_chunk
                               (chunk_id, doc_id, page_start, page_end, source_type, char_len)
                               VALUES (%s,%s,%s,%s,%s,%s)""",
                            (idx, did, _i(c.get("page_start")), _i(c.get("page_end")),
                             (c.get("source_type") or "")[:20], len(c.get("text") or "")))
                n += 1
        conn.close()
        return {"ok": True, "corpus": corpus_name, "chunks": n}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


def sync_search_logs(log_path: str, batch: int = 2000) -> Dict:
    """增量同步检索/问答日志（按 jsonl 行号幂等；文件被截断则从头重来）。

    与实时落库（insert_search_log）共用同一行映射；补录前先按 rid 反查该记录是否已落库，
    已落库就跳过，所以"实时写一份 + 事后补录一份"不会重复计数。
    MySQL 当时不可用而漏写的行，这里正好补上——两条路径互为兜底。
    """
    if not os.path.isfile(log_path):
        return {"ok": False, "error": f"日志不存在: {log_path}"}
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.read().splitlines()
    except OSError as e:
        return {"ok": False, "error": str(e)[:200]}

    conn = None
    try:
        conn = _connect()
        with conn.cursor() as cur:
            done = _state_get(cur, "search_log_lines", 0)
            if done > len(lines):
                done = 0                      # 文件被清空/截断 → 重来
            todo = lines[done:done + batch]
            inserted = skipped_dup = 0
            for offset, ln in enumerate(todo):
                src_line = done + offset + 1
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    d = json.loads(ln)
                except Exception:
                    continue
                if d.get("event") not in ("search", "ask"):
                    continue
                row = _log_row(d, src_line=src_line, source="etl")
                if _rid_exists(cur, row[_LOG_IDX["rid"]]):
                    skipped_dup += 1          # 实时落库已写过这条记录
                    continue
                cur.execute(_INSERT_LOG_SQL, row)
                inserted += cur.rowcount
            _state_set(cur, "search_log_lines", done + len(todo))
        conn.close()
        return {"ok": True, "scanned": len(todo), "inserted": inserted,
                "skipped_dup": skipped_dup, "total_lines": len(lines)}
    except Exception as e:
        if conn:
            try:
                conn.close()
            except Exception:
                pass
        return {"ok": False, "error": str(e)[:200]}


def sync_eval_results(results_dir: str, run_tag: Optional[str] = None) -> Dict:
    """同步评测结果 CSV → fact_eval（按 run_tag + qid + method 幂等）。"""
    import csv
    run_tag = run_tag or datetime.now().strftime("%Y-%m-%d")
    files = []
    if os.path.isdir(results_dir):
        files = [os.path.join(results_dir, f) for f in sorted(os.listdir(results_dir))
                 if f.endswith(".csv")]
    if not files:
        return {"ok": False, "error": f"未找到评测 CSV: {results_dir}"}
    cols = {
        "recall@5": "recall5", "mrr": "mrr", "ndcg@5": "ndcg5",
        "recall@5_c": "recall5_c", "mrr_c": "mrr_c", "ndcg@5_c": "ndcg5_c",
        "em": "em", "f1": "f1", "judge_corr": "judge_corr", "judge_faith": "judge_faith",
    }
    n = 0
    try:
        conn = _connect()
        with conn.cursor() as cur:
            for path in files:
                with open(path, encoding="utf-8-sig") as f:
                    for row in csv.DictReader(f):
                        vals = [_f(row.get(k)) for k in cols]
                        cur.execute(
                            """INSERT INTO fact_eval
                               (run_tag, qid, method, question, recall5, mrr, ndcg5,
                                recall5_c, mrr_c, ndcg5_c, em, f1, judge_corr, judge_faith)
                               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                               ON DUPLICATE KEY UPDATE
                                 recall5=VALUES(recall5), mrr=VALUES(mrr), ndcg5=VALUES(ndcg5),
                                 recall5_c=VALUES(recall5_c), mrr_c=VALUES(mrr_c),
                                 ndcg5_c=VALUES(ndcg5_c), em=VALUES(em), f1=VALUES(f1),
                                 judge_corr=VALUES(judge_corr), judge_faith=VALUES(judge_faith)""",
                            (run_tag, (row.get("qid") or "")[:30],
                             (row.get("chunk_method") or "")[:20],
                             row.get("question"),
                             *vals))
                        n += 1
        conn.close()
        return {"ok": True, "run_tag": run_tag, "files": len(files), "rows": n}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


def sync_all(corpus_name: Optional[str] = None, results_dir: Optional[str] = None) -> Dict:
    """一键同步：建表 → 维表 → 块表 → 评测 → 日志增量。"""
    from rag_core import corpus as corpus_mod
    paths = corpus_mod.runtime_paths()
    name = corpus_name or paths.get("active") or corpus_mod.DEFAULT_NAME
    results_dir = results_dir or os.path.join(config.PROJECT_DIR, "results")
    out = {"ok": True, "corpus": name, "steps": {}}
    schema = ensure_schema()
    out["steps"]["schema"] = schema
    if not schema.get("ok"):
        out["ok"] = False
        return out
    out["steps"]["meta"] = sync_corpus_meta(name, paths["meta"], paths["kb"])
    out["steps"]["chunks"] = sync_chunks(name, paths["kb"])
    out["steps"]["eval"] = sync_eval_results(results_dir)
    out["steps"]["logs"] = sync_search_logs(config.OBS_LOG)
    out["ok"] = all(s.get("ok") for s in out["steps"].values())
    return out


# --------------------------------------------------------------------------
# 结构化筛选（双引擎：SQL 出文献白名单 → 交给向量/BM25 做语义召回）
# --------------------------------------------------------------------------
def build_doc_filter_sql(filters: Dict, corpus_name: str = None) -> Tuple[str, List]:
    """纯函数：把结构化筛选条件编译成 (SQL, 参数)。便于测试与讲解。"""
    filters = filters or {}
    params: List = []
    sql = ["SELECT d.source FROM dim_document d",
           "JOIN dim_corpus c ON c.corpus_id = d.corpus_id"]
    if corpus_name:
        sql.append("WHERE c.name = %s")
        params.append(corpus_name)
    else:
        sql.append("WHERE 1=1")
    if filters.get("year_min") is not None:
        sql.append("AND d.year >= %s")
        params.append(int(filters["year_min"]))
    if filters.get("year_max") is not None:
        sql.append("AND d.year <= %s")
        params.append(int(filters["year_max"]))
    for field, key in (("author", "authors"), ("methods", "methods"), ("tasks", "tasks")):
        wanted = [str(x).strip() for x in (filters.get(key) or []) if str(x).strip()]
        if not wanted:
            continue
        if field == "author":
            # 与 Python match_meta_filter 保持一致：双向包含（作者名可能带后缀或只写姓）
            ors = " OR ".join(["d.author LIKE %s OR %s LIKE CONCAT('%%', d.author, '%%')"]
                              * len(wanted))
            sql.append(f"AND ({ors})")
            for w in wanted:
                params.extend([f"%{w}%", w])
        else:
            kind = "method" if field == "methods" else "task"
            # 与 Python 侧语义严格对齐：标签双向包含匹配
            #   w in v  →  t.label LIKE '%w%'
            #   v in w  →  'w' LIKE '%t.label%'
            ors = " OR ".join(["t.label LIKE %s OR %s LIKE CONCAT('%%', t.label, '%%')"]
                              * len(wanted))
            sql.append(
                "AND EXISTS (SELECT 1 FROM rel_doc_tag r "
                "JOIN dim_tag t ON t.tag_id = r.tag_id "
                f"WHERE r.doc_id = d.doc_id AND t.kind = %s AND ({ors}))")
            params.append(kind)
            for w in wanted:
                params.extend([f"%{w}%", w])
    return " ".join(sql), params


def count_docs(corpus_name: str) -> Optional[int]:
    """MySQL 里某语料的文献数；语料不存在或 MySQL 不可用 → None。"""
    try:
        conn = _connect()
        with conn.cursor() as cur:
            cur.execute("""SELECT COUNT(*) FROM dim_document d
                           JOIN dim_corpus c ON c.corpus_id = d.corpus_id
                           WHERE c.name = %s""", (corpus_name,))
            n = cur.fetchone()[0]
        conn.close()
        return int(n)
    except Exception:
        return None


def filter_docs_by_sql(filters: Dict, corpus_name: Optional[str] = None,
                       expected_docs: Optional[int] = None) -> Optional[List[str]]:
    """用 SQL 查符合结构化条件的文献白名单（双引擎的"结构化"那一半）。
    - 条件为空 → None（不筛选）；
    - MySQL 不可用 / 查询失败 → None（**降级**：调用方回退 Python 内存掩码）；
    - 查询成功但无匹配 → []（明确的空结果）。

    一致性守卫（很重要）：只有 MySQL 镜像与当前 KB 文档集合一致时才敢用 SQL 结果。
    语料没同步过（0 篇）或文库数与 expected_docs 不符（同步后又新增/删除了文献），
    都返回 None 触发降级——否则白名单会漏掉新文献，变成静默的召回缺失。
    宁可慢一点走内存掩码，也不能因为镜像过期而答漏。
    """
    filters = {k: v for k, v in (filters or {}).items() if v not in (None, [], "")}
    if not filters:
        return None
    if corpus_name:
        n_db = count_docs(corpus_name)
        if not n_db:                             # None（不可用）或 0（未同步）
            return None
        if expected_docs is not None and n_db != expected_docs:
            return None
    sql, params = build_doc_filter_sql(filters, corpus_name)
    conn = None
    try:
        conn = _connect()
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = [r[0] for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception:
        if conn:
            try:
                conn.close()
            except Exception:
                pass
        return None


# --------------------------------------------------------------------------
# 统计与报表
# --------------------------------------------------------------------------
def stats() -> Dict:
    """面板用：各表行数 + 语料概览。MySQL 不可用返回 {ok: False}。"""
    tables = ["dim_corpus", "dim_document", "dim_tag", "rel_doc_tag",
              "fact_chunk", "fact_search_log", "fact_eval"]
    out = {"ok": True, "host": f"{config.MYSQL_HOST}:{config.MYSQL_PORT}",
           "db": config.MYSQL_DB, "rows": {}}
    try:
        conn = _connect()
        with conn.cursor() as cur:
            for t in tables:
                cur.execute(f"SELECT COUNT(*) FROM {t}")
                out["rows"][t] = cur.fetchone()[0]
            cur.execute("""SELECT c.name, d.year, COUNT(*) n
                           FROM dim_document d JOIN dim_corpus c USING(corpus_id)
                           WHERE d.year IS NOT NULL
                           GROUP BY c.name, d.year ORDER BY d.year""")
            out["by_year"] = [{"corpus": r[0], "year": r[1], "docs": _num(r[2])} for r in cur.fetchall()]
            # 日志来源：live=检索时实时落库，etl=jsonl 补录（用于自检双写是否生效）
            cur.execute("SELECT source, COUNT(*) FROM fact_search_log GROUP BY source")
            out["log_source"] = {(r[0] or "etl"): _num(r[1]) for r in cur.fetchall()}
            # 结构化筛选实际走的是哪条引擎（mysql=SQL 白名单，memory=内存掩码降级）
            cur.execute("""SELECT COALESCE(filter_engine, '-'), COUNT(*) FROM fact_search_log
                           WHERE has_filters = 1 GROUP BY 1 ORDER BY 2 DESC""")
            out["filter_engine"] = {(r[0]): _num(r[1]) for r in cur.fetchall()}
        conn.close()
        return out
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


def report_latency() -> List[Dict]:
    """检索延迟分位 + HyDE/MMR 开关对比（视图查询）。"""
    try:
        conn = _connect()
        with conn.cursor() as cur:
            cur.execute("""SELECT use_hyde, use_mmr, event_type, n, avg_ms, p95_ms, avg_hits
                           FROM v_hyde_mmr_latency ORDER BY event_type, use_hyde, use_mmr""")
            rows = [{"use_hyde": r[0], "use_mmr": r[1], "event_type": r[2],
                     "n": r[3], "avg_ms": _num(r[4]), "p95_ms": _num(r[5]),
                     "avg_hits": _num(r[6])}
                    for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception:
        return []


def report_tag_distribution() -> List[Dict]:
    """标签 × 年份 文献分布（视图查询）。"""
    try:
        conn = _connect()
        with conn.cursor() as cur:
            cur.execute("""SELECT kind, label, year, doc_cnt FROM v_tag_year
                           ORDER BY kind, doc_cnt DESC, label, year""")
            rows = [{"kind": r[0], "label": r[1], "year": r[2], "doc_cnt": r[3]}
                    for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception:
        return []


def report_eval_compare(run_tag: Optional[str] = None) -> List[Dict]:
    """评测方法对比（SQL 聚合，替代手看 CSV）。"""
    try:
        conn = _connect()
        with conn.cursor() as cur:
            if run_tag:
                cur.execute("""SELECT method, COUNT(*) n,
                                      ROUND(AVG(recall5),4), ROUND(AVG(mrr),4), ROUND(AVG(ndcg5),4),
                                      ROUND(AVG(recall5_c),4), ROUND(AVG(mrr_c),4), ROUND(AVG(ndcg5_c),4),
                                      ROUND(AVG(f1),4), ROUND(AVG(judge_corr),2), ROUND(AVG(judge_faith),2)
                               FROM fact_eval WHERE run_tag=%s GROUP BY method ORDER BY method""",
                            (run_tag,))
            else:
                cur.execute("""SELECT method, COUNT(*) n,
                                      ROUND(AVG(recall5),4), ROUND(AVG(mrr),4), ROUND(AVG(ndcg5),4),
                                      ROUND(AVG(recall5_c),4), ROUND(AVG(mrr_c),4), ROUND(AVG(ndcg5_c),4),
                                      ROUND(AVG(f1),4), ROUND(AVG(judge_corr),2), ROUND(AVG(judge_faith),2)
                               FROM fact_eval GROUP BY method ORDER BY method""")
            keys = ["method", "n", "recall5", "mrr", "ndcg5", "recall5_c",
                    "mrr_c", "ndcg5_c", "f1", "judge_corr", "judge_faith"]
            rows = [dict(zip(keys, [_num(x) for x in r])) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception:
        return []
