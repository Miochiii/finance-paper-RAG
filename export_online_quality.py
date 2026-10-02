# -*- coding: utf-8 -*-
"""从在线 JSONL 日志导出按到达顺序排列的无标注 RAG 质量代理。

只收集已启用 explicit_v1 逐请求记录、生成成功的 ask 事件。通道和时间
可用于筛选正常运行期；脚本不把这些请求自动宣布为独立或无故障样本。
"""

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from rag_core.config import OBS_LOG


FIELDS = ["rid", "timestamp_utc", "origin", "corpus", "question_hash",
          "question_chars", "answer_chars", "ans_refusal_explicit", "no_evidence",
          "hits", "top_k"]


def parse_events(lines, origins=("mcp", "http")):
    counts = Counter()
    rows = []
    for line in lines:
        try:
            event = json.loads(line)
        except (TypeError, ValueError):
            counts["bad_json"] += 1
            continue
        if not isinstance(event, dict):
            counts["bad_json"] += 1
            continue
        if event.get("event") != "ask":
            continue
        counts["all_asks"] += 1
        if event.get("quality_proxy_version") != "explicit_v1":
            counts["legacy_no_proxy"] += 1
            continue
        if event.get("origin") not in origins:
            counts["other_origin"] += 1
            continue
        if not event.get("ok") or event.get("ans_refusal_explicit") not in (0, 1):
            counts["failed_or_missing"] += 1
            continue
        t = event.get("t")
        if not isinstance(t, (float, int)):
            counts["bad_timestamp"] += 1
            continue
        row = {key: event.get(key, "") for key in FIELDS}
        row["timestamp_utc"] = datetime.fromtimestamp(t, timezone.utc).isoformat()
        rows.append((float(t), row))
    rows.sort(key=lambda pair: (pair[0], str(pair[1]["rid"])))
    counts["usable"] = len(rows)
    return [row for _, row in rows], counts


def export(log_path: Path, output_dir: Path, origins=("mcp", "http")):
    with log_path.open(encoding="utf-8") as f:
        rows, counts = parse_events(f, origins)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "online_quality_sequence.csv"
    report_path = output_dir / "online_quality_sequence.md"
    if csv_path.exists() or report_path.exists():
        raise FileExistsError("在线序列输出已存在，拒绝覆盖")
    with csv_path.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    unique_hashes = len({r["question_hash"] for r in rows if r["question_hash"]})
    by_origin = Counter(r["origin"] for r in rows)
    lines = ["# 在线 RAG 质量代理序列", "",
             f"- 日志 ask 事件 {counts['all_asks']} 条；所选通道 {', '.join(origins)}。",
             f"- 有效 explicit_v1 事件 {counts['usable']} 条；旧事件缺少该代理 {counts['legacy_no_proxy']} 条；"
             f"错误或代理缺失 {counts['failed_or_missing']} 条。",
             f"- 有效事件的不同问题哈希 {unique_hashes} 个；哈希重复只表示问题文本重复，"
             "不同哈希也不能证明请求独立。",
             f"- 通道分布：{dict(by_origin)}。"]
    if rows:
        lines.append(f"- 时间范围（UTC）：{rows[0]['timestamp_utc']} 至 {rows[-1]['timestamp_utc']}。")
        lines.append(f"- 显式拒答 {sum(int(r['ans_refusal_explicit']) for r in rows)} 条；"
                     f"无检索证据 {sum(bool(r['no_evidence']) for r in rows)} 条。")
    lines += ["", "仅在已知正常运行的时间段内，才能把此序列作为变前样本。"
              "请求可能来自脚本或重复测试；需要依据运行记录审计来源。",
              "此导出不含问题和答案原文，也不推断真实误报率。", ""]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, report_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", type=Path, default=Path(OBS_LOG))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--origins", default="mcp,http",
                    help="逗号分隔，默认仅收集 MCP/HTTP 对话入口")
    args = ap.parse_args()
    for path in export(args.log, args.output_dir,
                       tuple(s.strip() for s in args.origins.split(",") if s.strip())):
        print(path)


if __name__ == "__main__":
    main()
