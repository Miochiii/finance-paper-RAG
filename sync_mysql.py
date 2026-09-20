# -*- coding: utf-8 -*-
"""sync_mysql.py —— MySQL 结构化分析层命令行入口（双引擎的"结构化"一半）。

用法：
    python sync_mysql.py --init       # 建库 + 建表 + 建视图（幂等）
    python sync_mysql.py --sync       # 全量同步：维表/块表/评测结果/检索日志（增量）
    python sync_mysql.py --stats      # 各表行数 + 语料年份分布
    python sync_mysql.py --report     # 报表：延迟分位 + 标签分布 + 评测对比
    python sync_mysql.py --filter "year_min=2020;methods=机器学习,深度学习;tasks=信贷风控"
                                      # 试跑"结构化筛选 → 文献白名单"（双引擎演示）

日志：检索/问答在服务运行时就会实时写进 fact_search_log（source=live）；
     --sync 只做补录（source=etl），按 rid 幂等，不会与实时行重复计数。
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_core import mysql_store as ms  # noqa: E402


def _parse_filters(text: str) -> dict:
    """解析 "year_min=2020;methods=机器学习,深度学习;authors=张三" 形式的筛选条件。"""
    out = {}
    for part in (text or "").split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, _, v = part.partition("=")
        k, v = k.strip(), v.strip()
        if not v:
            continue
        if k in ("year_min", "year_max"):
            out[k] = int(v)
        elif k == "authors":
            out[k] = [x.strip() for x in v.split(",") if x.strip()]
        elif k in ("methods", "tasks"):
            out[k] = [x.strip() for x in v.split(",") if x.strip()]
        else:
            print(f"  [忽略] 未知筛选键: {k}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="MySQL 结构化分析层：建表 / 同步 / 报表 / 筛选")
    ap.add_argument("--init", action="store_true", help="建库建表建视图")
    ap.add_argument("--sync", action="store_true", help="全量同步（幂等）")
    ap.add_argument("--stats", action="store_true", help="表行数统计")
    ap.add_argument("--report", action="store_true", help="报表查询")
    ap.add_argument("--filter", default=None, metavar="条件",
                    help="结构化筛选试跑，如 'year_min=2020;methods=机器学习'")
    ap.add_argument("--report-kind", default=None,
                    choices=["latency", "eval", "tags"],
                    help="只跑某一张报表")
    args = ap.parse_args()

    if not any([args.init, args.sync, args.stats, args.report, args.filter,
                args.report_kind]):
        ap.print_help()
        return 0

    if args.init:
        print(json.dumps(ms.ensure_schema(), ensure_ascii=False, indent=2))

    if args.sync:
        r = ms.sync_all()
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return 0 if r.get("ok") else 1

    if args.stats:
        print(json.dumps(ms.stats(), ensure_ascii=False, indent=2))

    if args.report or args.report_kind:
        kinds = [args.report_kind] if args.report_kind else ["latency", "eval", "tags"]
        if "latency" in kinds:
            print("=== 检索延迟/HyDE-MMR 对比（视图 v_hyde_mmr_latency）===")
            for row in ms.report_latency():
                print("  " + json.dumps(row, ensure_ascii=False))
        if "eval" in kinds:
            print("\n=== 评测方法对比（SQL 聚合）===")
            for row in ms.report_eval_compare():
                print("  " + json.dumps(row, ensure_ascii=False))
        if "tags" in kinds:
            tags = ms.report_tag_distribution()
            print(f"\n=== 标签 × 年份分布（视图 v_tag_year，共 {len(tags)} 行，前 10）===")
            for row in tags[:10]:
                print("  " + json.dumps(row, ensure_ascii=False))

    if args.filter:
        f = _parse_filters(args.filter)
        name, expected = _active_corpus()
        print(f"筛选条件: {json.dumps(f, ensure_ascii=False)}")
        print(f"激活语料: {name}（KB 内 {expected} 篇文献；镜像不一致会自动降级内存掩码）")
        docs = ms.filter_docs_by_sql(f, corpus_name=name, expected_docs=expected)
        if docs is None:
            print("  → 未走 SQL 白名单：条件为空 / MySQL 不可用 / 镜像与 KB 不一致（降级内存掩码）")
        else:
            print(f"  → 命中 {len(docs)} 篇文献（SQL 引擎）：")
            for d in docs:
                print("    - " + d)
    return 0


def _active_corpus():
    """激活语料名 + KB 内文献数（用于让 --filter 复现检索时的双引擎判断）。"""
    try:
        from rag_core import corpus as corpus_mod
        paths = corpus_mod.runtime_paths()
        name = paths.get("active") or corpus_mod.DEFAULT_NAME
        with open(paths["kb"], encoding="utf-8") as fp:
            kb = json.load(fp)
        return name, len({str(c.get("source") or "") for c in kb})
    except Exception:
        return None, None


if __name__ == "__main__":
    sys.exit(main())
