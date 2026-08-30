# -*- coding: utf-8 -*-
"""
build_docs_cache_v2.py — 用 MinerU 输出生成 docs_cache_v2，并与旧缓存对比

用法：
    python build_docs_cache_v2.py                          # 只打印对比，不落盘
    python build_docs_cache_v2.py --save data/docs_cache_v2.json
"""
import argparse
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_core.mineru_loader import build_docs_cache_v2, TABLE_START
from evaluate import _norm_text, _norm_text_strip_pages, load_finance

OLD_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "docs_cache.json")


def _stem(name: str) -> str:
    return os.path.splitext(name)[0]


def compare_and_report(new_docs: dict, old_docs: dict, qbank: list) -> None:
    print("=" * 70)
    print("新旧提取对比（按文档）")
    print("=" * 70)
    for name, text in sorted(new_docs.items()):
        old_text = old_docs.get(name) or old_docs.get(name + ".pdf") or ""
        n_tables = text.count(TABLE_START)
        n_eq = text.count("$$") // 2
        n_imgs = text.count("[图片")
        print(f"\n[{name}]")
        print(f"  旧提取 {len(old_text):>8} 字 | 新提取 {len(text):>8} 字")
        print(f"  新提取: 表格块 {n_tables} 个 | 公式块 {n_eq} 个 | 图片引用 {n_imgs} 个 | 页眉/页脚已按块丢弃")

    print("\n" + "=" * 70)
    print("标注证据命中对比（只统计新缓存已覆盖的文档）")
    print("=" * 70)
    old_hit = new_hit = total = 0
    for q in qbank:
        doc = q["gold_sources"][0]
        if doc not in new_docs:
            continue
        golds = q["gold_texts"]
        new_text = _norm_text_strip_pages(new_docs[doc])
        old_text = _norm_text_strip_pages(old_docs.get(doc, ""))
        for gi, g in enumerate(golds):
            total += 1
            ng = _norm_text(g)
            if old_text and ng in old_text:
                old_hit += 1
            if ng in new_text:
                new_hit += 1
            if (not old_text or ng not in old_text) and ng in new_text:
                print(f"  ✓ {q['qid']}#{gi}: 旧提取缺失 → 新提取命中")
            elif ng not in new_text and old_text and ng in old_text:
                print(f"  ✗ {q['qid']}#{gi}: 旧提取命中 → 新提取丢失（需复查）")
    print(f"\n命中率: 旧 {old_hit}/{total}（{old_hit / total:.1%}）→ 新 {new_hit}/{total}（{new_hit / total:.1%}）")


def main():
    ap = argparse.ArgumentParser(description="MinerU 输出 → docs_cache_v2 + 新旧对比")
    ap.add_argument("--mineru-out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mineru_out"))
    ap.add_argument("--save", default=None, help="保存为 JSON（如 data/docs_cache_v2.json）")
    args = ap.parse_args()

    new_docs = build_docs_cache_v2(args.mineru_out)
    if not new_docs:
        print("未找到任何 MinerU 输出（确认 --mineru-out 指向 output 目录）")
        return 1

    old_docs = {}
    if os.path.exists(OLD_CACHE):
        old_docs = json.load(open(OLD_CACHE, encoding="utf-8"))

    compare_and_report(new_docs, old_docs, load_finance())

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(new_docs, f, ensure_ascii=False, indent=2)
        print(f"\n已保存 → {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
