# -*- coding: utf-8 -*-
"""migrate_corpus.py —— 一次性迁移：data/ 根下的运行数据 → data/corpora/<语料>/ 统一布局。

用法：
    python migrate_corpus.py --dry-run     # 只看计划，不动手
    python migrate_corpus.py               # 执行（要求 8000 服务已停止）
    python migrate_corpus.py --force       # 跳过"服务运行中"检测
    python migrate_corpus.py --name 我的语料  # 指定语料名（默认"金融论文"）
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_core import corpus  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="data/ 根 → data/corpora/<语料>/ 一次性迁移")
    ap.add_argument("--dry-run", action="store_true", help="只打印迁移计划")
    ap.add_argument("--force", action="store_true", help="跳过服务运行检测")
    ap.add_argument("--name", default=corpus.DEFAULT_NAME, help="语料名（默认 %(default)s）")
    args = ap.parse_args()

    if args.dry_run:
        plan = corpus.plan_migration()
        print("=== 迁移计划（dry-run，不执行） ===")
        if not plan:
            print("未发现需要迁移的旧布局文件（可能已迁移）")
            return 0
        for src, dst, what in plan:
            print(f"  [{what}]")
            print(f"    {src}")
            print(f"    -> {dst}")
        print(f"\n共 {len(plan)} 项。执行前请先停止 8000 服务。")
        return 0

    r = corpus.migrate(name=args.name, force=args.force)
    if not r.get("ok"):
        print("[失败] " + r.get("error", "未知错误"))
        return 1
    print("=== 迁移完成 ===")
    for line in r.get("moved", []):
        print("  " + line)
    print(f"pdf_path 已刷新: {r.get('pdf_path_fixed', 0)} 条")
    print(f"激活语料: {r.get('name')}（{corpus.current_file()}）")
    print("\n下一步：重启 8000 服务（启动RAG服务.bat），验证 stats 与点击打开。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
