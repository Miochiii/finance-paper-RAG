# -*- coding: utf-8 -*-
"""run_rag.py —— 命令行入口（无需桌面端/DSH 也能完整使用）

用法：
    python run_rag.py build [--chunker hmm] [--clear]   # 构建/重建知识库
    python run_rag.py ingest                             # 增量入库（只处理新增文档）
    python run_rag.py stats                              # 知识库统计（含运行统计）
    python run_rag.py search "问题" [--top-k 5]          # 纯检索（返回证据块）
    python run_rag.py ask "问题" [--top-k 5]             # 检索 + 生成（带引用）
    python run_rag.py open 文档名.pdf [--page 12]        # 打开原始 PDF 指定页
    python run_rag.py health                             # 环境自检
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rag_server as core  # noqa: E402


def _print(obj, limit=None):
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    print(text[:limit] if limit else text)


def _cmd_health(_):
    checks = {}
    for mod in ("numpy", "sklearn", "hmmlearn", "tiktoken", "jieba", "rank_bm25",
                "qdrant_client", "sentence_transformers", "transformers", "torch",
                "openai", "docx"):
        try:
            __import__(mod)
            checks[mod] = "ok"
        except Exception as e:
            checks[mod] = f"missing: {e}"
    checks["kb_exists"] = os.path.exists(core.KB_FILE)
    checks["mineru_out"] = core.MINERU_OUT
    checks["docs_dir"] = core.DOCS_DIR
    checks["deepseek_api"] = bool(os.getenv("deepseek_api"))
    _print(checks)


def main():
    ap = argparse.ArgumentParser(description="RAG 命令行工具")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build", help="构建/重建知识库")
    p.add_argument("--chunker", default="hmm", help="fixed/discourse/hybrid/hmm")
    p.add_argument("--clear", action="store_true", help="先清空再重建")

    sub.add_parser("ingest", help="增量入库（只处理新增文档）")
    sub.add_parser("stats", help="知识库统计")

    p = sub.add_parser("search", help="纯检索")
    p.add_argument("query")
    p.add_argument("--top-k", type=int, default=5)

    p = sub.add_parser("ask", help="检索+生成")
    p.add_argument("question")
    p.add_argument("--top-k", type=int, default=5)

    p = sub.add_parser("open", help="打开原始 PDF 指定页")
    p.add_argument("doc")
    p.add_argument("--page", type=int, default=1)

    sub.add_parser("health", help="环境自检")

    args = ap.parse_args()
    if args.cmd == "build":
        _print(core.build_kb(args.chunker, args.clear))
    elif args.cmd == "ingest":
        _print(core.ingest_kb())
    elif args.cmd == "stats":
        _print(core.stats_kb(), limit=6000)
    elif args.cmd == "search":
        _print(core.search_kb(args.query, args.top_k), limit=4000)
    elif args.cmd == "ask":
        _print(core.ask_kb(args.question, args.top_k), limit=6000)
    elif args.cmd == "open":
        _print(core.open_doc_kb(args.doc, args.page))
    elif args.cmd == "health":
        _cmd_health(args)


if __name__ == "__main__":
    main()
