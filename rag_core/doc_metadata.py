# -*- coding: utf-8 -*-
"""文档元数据补齐：年份（封面正则）、作者（封面/文件名）、方法/任务（LLM 打标）。

产物：
  1. doc_metadata.json 侧车文件（默认与知识库同目录；环境变量 RAG_DOC_META 可覆盖）；
  2. 把 author/year/methods/tasks 合并进 KB 块元数据（不改块文本 → 不触发重嵌入）。

用法：
    python build_doc_metadata.py          # 只补缺失项（已分类的跳过）
    python build_doc_metadata.py --force  # 全部重跑（含重新 LLM 打标）
    python build_doc_metadata.py --skip-classify   # 只做免费的年/作者提取
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple

from rag_core.config import KB_FILE as KB_FILE_DEFAULT

DOC_META_FILE = os.getenv(
    "RAG_DOC_META",
    os.path.join(os.path.dirname(KB_FILE_DEFAULT), "doc_metadata.json"),
)

# ---- 年份提取 ----
_YEAR_AR_RE = re.compile(r"(19|20)\d{2}\s*年")
_YEAR_CN_RE = re.compile(r"[二〇○零一二三四五六七八九十]{4}\s*年")
_CN_DIGITS = {
    "〇": 0, "○": 0, "零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}


def _cn_year(s: str) -> Optional[int]:
    n = 0
    for ch in s[:4]:
        if ch not in _CN_DIGITS:
            return None
        n = n * 10 + _CN_DIGITS[ch]
    return n if 2000 <= n <= 2030 else None


def extract_year(head: str) -> Optional[int]:
    """从封面文本提取年份。按可信度优先：
    1) 收稿/提交/答辩等日期前缀 + 年份；2) 文章编号 (YYYY)；3) XXXX年（阿拉伯）；
    4) 中文数字年份；5) 裸 YYYY 日期格式（YYYY-MM / (YYYY) 等）。"""
    # 1) 日期前缀
    for m in re.finditer(r"(?:收稿|网络首发|提交|答辩|出版|录用)日期[：:\s]*(19|20)\d{2}", head):
        y = int(m.group(1))
        if 2000 <= y <= 2030:
            return y
    # 2) 文章编号 (YYYY)
    m = re.search(r"文章编号[：:]?[\d\-—]+\((\d{4})\)", head)
    if m:
        y = int(m.group(1))
        if 2000 <= y <= 2030:
            return y
    # 3) XXXX年（取第一个有效值，跳过 1980年-2020年 这类区间中的旧年份）
    for m in _YEAR_AR_RE.finditer(head):
        y = int(m.group(0)[:4])
        if 2000 <= y <= 2030:
            return y
    # 4) 中文数字年份
    for m in _YEAR_CN_RE.finditer(head):
        y = _cn_year(m.group(0))
        if y:
            return y
    # 5) 裸年份（YYYY- / YYYY. / (YYYY)）
    for m in re.finditer(r"(19|20)\d{2}(?=[-/.)]|$)", head):
        y = int(m.group(0))
        if 2000 <= y <= 2030:
            return y
    return None


# ---- 作者提取 ----
_AUTHOR_RES = [
    re.compile(r"学位申请人姓名[：:\s]*([\u4e00-\u9fa5]{2,4})"),
    re.compile(r"学生姓名[：:\s_＿]*([\u4e00-\u9fa5]{2,4})"),
    re.compile(r"硕士研究生[：:]\s*([\u4e00-\u9fa5]{2,4})"),
    re.compile(r"作者[：:]\s*([\u4e00-\u9fa5]{2,4})"),
]


def _is_cn_name(s: str) -> bool:
    return 2 <= len(s) <= 4 and all("\u4e00" <= ch <= "\u9fa5" for ch in s)


def extract_author(filename: str, head: str) -> Optional[str]:
    """封面正则优先，回退文件名尾段（<标题>_<作者>.pdf）。"""
    for rx in _AUTHOR_RES:
        m = rx.search(head)
        if m and _is_cn_name(m.group(1)):
            return m.group(1)
    base = os.path.basename(filename)
    if base.lower().endswith(".pdf"):
        base = base[:-4]
    if "_" in base:
        cand = base.rsplit("_", 1)[1].strip()
        if _is_cn_name(cand):
            return cand
    return None


# ---- LLM 打标（方法/任务/关键词）----
_METHODS = ["深度学习", "传统机器学习", "集成学习", "时间序列", "文本挖掘",
            "优化方法", "统计方法", "图方法", "其他"]
_TASKS = ["信贷风控", "股价预测", "交易策略", "资产定价", "投资组合",
          "期货衍生品", "债券", "基金", "方法比较/综述", "其他"]

_CLASSIFY_PROMPT = """你是金融机器学习论文分类助手。根据论文标题与摘要片段，输出 JSON：

{
  "methods": ["方法1", ...],
  "tasks": ["任务1", ...],
  "keywords": ["词1", ...]
}

要求：
- methods：从备选选 1~3 个（可补充新词）：{methods}
- tasks：从备选选 1~2 个（可补充新词）：{tasks}
- keywords：3~6 个核心术语（模型名、指标名、领域词）
只输出 JSON，不要任何解释文字。"""


def classify_doc(title: str, head: str, api_key: str, timeout: int = 60) -> Dict:
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    prompt = (_CLASSIFY_PROMPT
              .replace("{methods}", "、".join(_METHODS))
              .replace("{tasks}", "、".join(_TASKS)))
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"标题：{title}\n摘要片段：\n{head[:1500]}"},
        ],
        temperature=0.1,
        stream=False,
        timeout=timeout,
    )
    raw = (resp.choices[0].message.content or "").strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if m:
        raw = m.group(1)
    data = json.loads(raw)
    return {
        "methods": [str(x).strip() for x in data.get("methods", []) if str(x).strip()][:5],
        "tasks": [str(x).strip() for x in data.get("tasks", []) if str(x).strip()][:5],
        "keywords": [str(x).strip() for x in data.get("keywords", []) if str(x).strip()][:8],
    }


# ---- 侧车读写 ----
def load_meta() -> Dict:
    if not os.path.isfile(DOC_META_FILE):
        return {}
    try:
        with open(DOC_META_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_meta(meta: Dict):
    os.makedirs(os.path.dirname(DOC_META_FILE), exist_ok=True)
    tmp = DOC_META_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    os.replace(tmp, DOC_META_FILE)


# ---- 批量构建 ----
def build_doc_metadata(docs: Dict[str, str], force: bool = False,
                       skip_classify: bool = False) -> Tuple[Dict, Dict]:
    """docs: {文档名: 全文}。返回 (meta, report)。"""
    meta = {} if force else load_meta()
    api_key = os.getenv("deepseek_api", "")
    report = {
        "total": len(docs), "year_ok": 0, "author_ok": 0,
        "classified": 0, "skipped": 0, "cls_errors": 0,
    }
    for name in sorted(docs):
        entry = meta.get(name, {})
        complete = bool(entry.get("year") and entry.get("author")
                        and entry.get("methods")) or (
            skip_classify and entry.get("year") and entry.get("author"))
        if complete and not force:
            report["skipped"] += 1
            continue
        head = docs[name][:800]
        year = extract_year(head)
        author = extract_author(name, head)
        report["year_ok"] += 1 if year else 0
        report["author_ok"] += 1 if author else 0
        if year is not None:
            entry["year"] = year
        if author:
            entry["author"] = author
        if api_key and not skip_classify and (force or not entry.get("methods")):
            try:
                cls = classify_doc(name, docs[name][:2000], api_key)
                entry.update(cls)
                report["classified"] += 1
                print(f"  [CLS] {name[:40]}... -> {cls.get('methods')} | {cls.get('tasks')}")
            except Exception as e:
                report["cls_errors"] += 1
                print(f"  [CLS-ERR] {name}: {str(e)[:120]}")
        meta[name] = entry
    save_meta(meta)
    return meta, report


def patch_kb_with_meta(kb_file: str = None) -> int:
    """把 author/year/methods/tasks 合并进 KB 块元数据（不改文本 → 不触发重嵌入）。
    返回被更新的块数。"""
    kb_file = kb_file or KB_FILE_DEFAULT
    meta = load_meta()
    if not meta or not os.path.isfile(kb_file):
        return 0
    with open(kb_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    n = 0
    for c in chunks:
        m = meta.get(c.get("source", ""))
        if not m:
            continue
        changed = False
        for k in ("author", "year", "methods", "tasks"):
            v = m.get(k)
            if v not in (None, "", []) and k not in c:
                c[k] = v
                changed = True
        if changed:
            n += 1
    with open(kb_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    return n


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import rag_server as core  # noqa: E402
    from rag_core.mineru_loader import find_mineru_outputs, load_mineru_doc  # noqa: E402
    from rag_core.document_loader import _extract_docx_content  # noqa: E402

    ap = argparse.ArgumentParser(description="补齐文档元数据（年份/作者/方法/任务）")
    ap.add_argument("--force", action="store_true", help="全部重跑（含重新 LLM 打标）")
    ap.add_argument("--skip-classify", action="store_true", help="只做免费的年/作者提取，不调 LLM")
    args = ap.parse_args()

    docs = {}
    if os.path.isdir(core.MINERU_OUT):
        for doc, path in find_mineru_outputs(core.MINERU_OUT).items():
            try:
                t = load_mineru_doc(path)
                if t.strip():
                    docs[doc + ".pdf"] = t
            except Exception as e:
                print(f"  [ERR] {doc}: {e}")
    if os.path.isdir(core.DOCS_DIR):
        for f in sorted(os.listdir(core.DOCS_DIR)):
            if f.lower().endswith(".docx"):
                try:
                    t = _extract_docx_content(os.path.join(core.DOCS_DIR, f))
                    if t.strip():
                        docs[f] = t
                except Exception as e:
                    print(f"  [ERR] {f}: {e}")
    if not docs:
        print("未找到文档（检查 MINERU_OUT / DOCS_DIR）")
        sys.exit(1)

    meta, report = build_doc_metadata(docs, force=args.force, skip_classify=args.skip_classify)
    patched = patch_kb_with_meta(core.KB_FILE)
    print("=" * 50)
    print("元数据报告:", json.dumps(report, ensure_ascii=False))
    print(f"KB 块元数据合并: {patched} 块更新（文本未动，无需重嵌入）")
    print(f"侧车文件: {DOC_META_FILE}")
    for name in sorted(meta)[:5]:
        print(f"  样例: {name[:36]}... -> {json.dumps(meta[name], ensure_ascii=False)}")
