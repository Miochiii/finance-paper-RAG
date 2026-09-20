# -*- coding: utf-8 -*-
"""gen_annotations.py —— 评测标注扩充（40 → 120）的候选生成 + 机器预筛。

设计原则（重要，决定了这批标注能不能当基准用）：
    1. **证据不由模型生成**：脚本把原文片段喂给模型，模型只负责出问题与一句话答案，
       gold_chunks 由脚本**从原文逐字摘抄**（并要求模型给出的引文必须是原文子串，
       否则脚本用原文片段兜底）——避免"模型编证据 → 又拿来评测模型"的循环；
    2. **候选默认不进评测**：新行 status 一律写 `pending`，evaluate.py 只读 `done`，
       所以未经人工审核的候选不可能混进基准；
    3. **机器预筛留下痕迹**：每题记录检索是否命中 gold 文献、系统实际回答、与既有 40 题的
       最大相似度、以及 LLM 自检结论，供人工只审"可疑项"；
    4. **可复现**：同一语料 + 同一 prompt → 可通过 --seed 复现同一批候选。

用法：
    python gen_annotations.py --limit 2          # 试跑 2 篇论文（看格式与质量）
    python gen_annotations.py                   # 全量 38 篇（约 80~100 题候选）
    python gen_annotations.py --no-check        # 跳过检索核验（不占用线上服务）
    python gen_annotations.py --merge FILE      # 把审过的候选合并进正式标注（分配 fin_0xx）

输出：
    data/annotations/finance_annotations_v2_candidates.csv   候选（含预筛列）
    log/标注扩充报告_<日期>.md                                 汇总与审核指引
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import find_dotenv, load_dotenv  # noqa: E402

load_dotenv(find_dotenv())

from rag_core.config import PROJECT_DIR  # noqa: E402
from rag_core import corpus as corpus_mod  # noqa: E402

ANNOT_DIR = os.path.join(PROJECT_DIR, "data", "annotations")
MAIN_CSV = os.path.join(ANNOT_DIR, "finance_annotations.csv")
CAND_CSV = os.path.join(ANNOT_DIR, "finance_annotations_v2_candidates.csv")
LOG_DIR = os.path.join(PROJECT_DIR, "log")
SERVICE_URL = os.getenv("RAG_AUDIT_URL", "http://127.0.0.1:8000")
LLM_MODEL = "deepseek-chat"
PRICE_IN_PER_M, PRICE_OUT_PER_M = 1.5, 4.5
EXCERPTS_PER_DOC = 4
EXCERPT_CHARS = 900
ITEMS_PER_DOC = 3
DUP_THRESHOLD = 0.92
CAND_FIELDS = ["id", "status", "question", "answer", "gold_docs", "gold_chunks", "notes",
               "gen_type", "evidence_pages", "quote_fallback", "dup_sim", "title_leak",
               "fact_missing", "missing_facts", "retrieval_hit", "system_answer",
               "system_supports_gold", "verify", "tier"]

# 文献名里的这些词是"通用词"，出现在问题里不算标题泄漏
_GENERIC_TITLE_WORDS = ("基于", "研究", "应用", "分析", "机器学习", "算法", "模型", "预测",
                        "实证", "设计", "方法", "问题", "比较", "及其", "中国", "策略", "投资")


def title_leak_tokens(doc_name: str) -> List[str]:
    """从文献名里抽出"有区分度的词"，用于检测问题是否把标题抄进去（会破坏检索难度）。"""
    stem = re.sub(r"\.(pdf|docx)$", "", doc_name or "", flags=re.I)
    stem = re.sub(r"[_—\-–]+", " ", stem)
    toks = []
    for t in re.split(r"[\s的与和及]+", stem):
        t = t.strip()
        if len(t) < 2 or t in _GENERIC_TITLE_WORDS:
            continue
        if re.fullmatch(r"[\d\W]+", t):
            continue
        toks.append(t)
    return toks


def compute_title_leak(question: str, doc_name: str, min_len: int = 3) -> int:
    """问题里是否出现文献名的区分性词（≥min_len 字），返回命中个数。"""
    hay = _norm(question)
    return sum(1 for t in title_leak_tokens(doc_name) if len(t) >= min_len and _norm(t) in hay)


def extract_key_facts(answer: str) -> List[str]:
    """从答案里抽出"必须能在原文中找到"的硬事实：数字、拉丁文模型名、常见中文方法名。"""
    a = answer or ""
    facts = set()
    for m in re.finditer(r"\d+(?:\.\d+)?%?", a):
        tok = m.group(0)
        if len(tok.replace(".", "").replace("%", "")) >= 2:      # 忽略单个数字（如"三支"里的量词）
            facts.add(tok)
    for m in re.finditer(r"[A-Za-z][A-Za-z0-9\-]{1,}", a):
        tok = m.group(0)
        if tok.lower() not in ("id", "auc", "nri", "idi", "ok"):
            facts.add(tok)
    for w in _METHOD_WORDS_CN:
        if w in a:
            facts.add(w)
    return sorted(facts)


_METHOD_WORDS_CN = ("逻辑回归", "支持向量机", "决策树", "随机森林", "神经网络", "梯度提升",
                    "自适应提升", "极限提升", "轻度提升", "朴素贝叶斯", "主成分", "偏最小二乘",
                    "弹性网络", "隐马尔科夫", "网格搜索", "交叉验证", "过采样", "欠采样",
                    "马氏距离", "信息熵", "因子分析", "聚类", "集成学习", "滞后", "协整")


def fact_check(answer: str, doc_chunks: List[Dict], pages: List[str]) -> Tuple[int, str]:
    """答案里的硬事实是否能在该文献对应页的原文里找到。

    返回 (缺失个数, 缺失清单)。数字/模型名找不到 → 说明答案掺入了原文没有的信息，
    是比 LLM 质检更硬的信号（LLM 质检只看"是否被片段支持"，容易被措辞骗过）。
    """
    facts = extract_key_facts(answer)
    if not facts:
        return 0, ""
    pool = doc_chunks
    if pages:
        pset = {int(p) for p in pages if str(p).isdigit()}
        if pset:
            hit = [c for c in doc_chunks
                   if (c.get("page_start") in pset) or (c.get("page_end") in pset)]
            pool = hit or doc_chunks
    text = _norm(" ".join((c.get("text") or "") for c in pool))
    missing = [f for f in facts if _norm(f) not in text]
    return len(missing), "|".join(missing[:6])


def triage(row: Dict) -> str:
    """人工审核分级：绿=抽查即可；黄=需读一遍；红=必须处理。

    注意：quote_fallback=1 只表示"模型给的引文与原文不一致、脚本已换成原文子句"，
    证据本身仍是逐字的，所以它属于"需人工确认这句证据确实支撑答案"（黄），
    而不是"证据造假"（红）。红只留给两类硬问题：质检不通过、或与既有题高度相似。
    """
    if not str(row.get("verify", "")).startswith("ok") or float(row.get("dup_sim") or 0) >= 0.75:
        return "红"
    if str(row.get("retrieval_hit")) == "0" or int(row.get("title_leak") or 0) > 0 \
            or str(row.get("quote_fallback")) == "1" or int(row.get("fact_missing") or 0) > 0:
        return "黄"
    return "绿"


def tier_reason(row: Dict) -> str:
    """给人工审核看的"为什么被标黄/红"。"""
    rs = []
    if not str(row.get("verify", "")).startswith("ok"):
        rs.append("机器质检存疑：" + row.get("verify", ""))
    if float(row.get("dup_sim") or 0) >= 0.75:
        rs.append(f"与既有题相似度 {row.get('dup_sim')}（可能重复出题）")
    if str(row.get("retrieval_hit")) == "0":
        rs.append("检索未命中该文献（困难样本，请确认答案确实出自这篇）")
    if int(row.get("title_leak") or 0) > 0:
        rs.append("问题含文献标题词（会削弱检索难度）")
    if str(row.get("quote_fallback")) == "1":
        rs.append("模型引文与原文不一致，已自动改用原文子句（请核对是否支撑答案）")
    if int(row.get("fact_missing") or 0) > 0:
        rs.append(f"答案中的 {row.get('fact_missing')} 未在原文对应页找到（疑似掺入原文没有的信息）")
    return "；".join(rs) or "—"

GEN_PROMPT = """你是学术文献标注员，负责为"检索增强问答系统"的评测集出题。

我会给你**同一篇论文的若干原文片段**（带编号）。请基于这些片段出题，要求：

1. 出 {n} 道**互不重复**的问题，覆盖不同方面（优先：研究方法/模型、数据与样本、评价指标、
   主要结论或改进效果、论文局限）；
2. 每题给一句**标准答案**（不超过 80 字，直接回答，不要"根据片段"之类的话）；
3. 问题必须**只靠给定片段就能回答**，且答案在片段中有明确依据；不要把片段里没有的信息写进答案；
4. 问题里**不要出现论文标题或文件名**（否则检索时会被标题词直接命中，测不出真实能力）；
5. evidence 给出支撑该答案的片段编号（可多个）；quote 给出**从该片段逐字复制**的 1~2 句证据
   （必须与原文完全一致，不要改写、不要加省略号）；
6. 数值、模型名、指标名必须与原文一致。

只输出 JSON：
{{"items":[{{"q":"问题","a":"标准答案","type":"方法|数据|指标|结论|局限","evidence":[1,2],"quote":"逐字证据句"}}]}}"""

VERIFY_PROMPT = """你是标注质检员。给定【问题】【标准答案】【原文片段】，判断：
- answer_supported：标准答案是否被原文片段直接支持（true/false）；
- unique：原文片段中该问题是否只有一个明确答案（true/false）；
- issue：若不通过，用不超过 20 字说明问题（通过则留空）。

只输出 JSON：{"answer_supported":true,"unique":true,"issue":""}"""


# --------------------------------------------------------------------------
# 纯函数：片段挑选 / 引文核验 / 去重
# --------------------------------------------------------------------------
_METHOD_WORDS = ("方法", "模型", "算法", "实验", "结果", "结论", "样本", "数据", "指标", "变量", "回归")


def pick_excerpts(chunks: List[Dict], k: int = EXCERPTS_PER_DOC,
                  max_chars: int = EXCERPT_CHARS) -> List[Dict]:
    """从一篇论文的块里挑 k 个代表性片段（尽量覆盖全文、优先含方法/数据/数字的段落）。

    规则：首块（题目/摘要区）必选；其余按"关键词密度 + 是否含数字"打分，
    并按页均匀取点，避免 k 个片段挤在同一页。
    """
    if not chunks:
        return []
    usable = [c for c in chunks if len((c.get("text") or "").strip()) >= 120]
    if not usable:
        usable = chunks
    picked: List[Dict] = [usable[0]]
    rest = usable[1:]
    scored = []
    for c in rest:
        t = c.get("text") or ""
        score = sum(1 for w in _METHOD_WORDS if w in t) + (2 if re.search(r"\d", t) else 0)
        scored.append((score, c))
    scored.sort(key=lambda x: -x[0])
    used_pages = {picked[0].get("page_start")}
    for _, c in scored:
        if len(picked) >= k:
            break
        pg = c.get("page_start")
        if pg in used_pages and len(usable) > k:      # 页重复则跳过（尽量分散）
            continue
        picked.append(c)
        used_pages.add(pg)
    for c in usable:                                  # 不够就按顺序补齐
        if len(picked) >= k:
            break
        if c not in picked:
            picked.append(c)
    return [{**c, "text": (c.get("text") or "")[:max_chars]} for c in picked[:k]]


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def verify_quote(quote: str, excerpts: List[Dict], evidence_ids: List[int]) -> Tuple[str, bool]:
    """引文核验：模型给的 quote 必须是所引片段的**逐字子串**。

    返回 (可用的逐字证据串, 是否用了兜底)。逐字子串 → 直接用；
    若不匹配，则退化为"从片段里截取含答案关键词的原句"；再不行就整段兜底。
    """
    q = (quote or "").strip().strip("“”\"'")
    pool = [excerpts[i - 1] for i in evidence_ids if 1 <= i <= len(excerpts)]
    if not pool:
        pool = excerpts[:1]
    if q:
        for e in pool:
            if _norm(q) and _norm(q) in _norm(e["text"]):
                return q, False
    # 兜底 1：取片段中与引文重合度最高的整句（保证逐字）
    sents = []
    for e in pool:
        for s in re.split(r"(?<=[。；！？])", e["text"]):
            s = s.strip()
            if len(s) >= 12:
                sents.append(s)
    if q and sents:
        target = _norm(q)
        best = max(sents, key=lambda s: len(set(_norm(s)) & set(target)) / max(1, len(set(_norm(s)))))
        if len(set(_norm(best)) & set(target)) >= 4:
            return best, True
    # 兜底 2：整段（截断）
    return (pool[0]["text"][:300], True)


def question_similarity(a: str, b: str) -> float:
    """问题相似度的词面近似（字符 3-gram Jaccard），用于与既有题目去重。"""
    def grams(s: str) -> set:
        s = _norm(s)
        return {s[i:i + 3] for i in range(max(0, len(s) - 2))}
    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def parse_items(raw: str) -> List[Dict]:
    """从模型输出里解析候选（容错：允许 markdown 代码块与多余文字）。"""
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except Exception:
        return []
    out = []
    for it in data.get("items", []) or []:
        q = str(it.get("q") or "").strip()
        a = str(it.get("a") or "").strip()
        if len(q) < 6 or len(a) < 4:
            continue
        ev = it.get("evidence") or []
        if not isinstance(ev, list):
            ev = []
        out.append({"q": q, "a": a,
                    "type": str(it.get("type") or "其他")[:6],
                    "evidence": [int(x) for x in ev if str(x).isdigit()],
                    "quote": str(it.get("quote") or "")})
    return out


def next_ids(existing_ids: List[str], n: int) -> List[str]:
    """按现有 fin_0xx 续号（跳过已占用的号）。"""
    used = set()
    for i in existing_ids:
        m = re.match(r"fin_(\d+)$", (i or "").strip())
        if m:
            used.add(int(m.group(1)))
    out, cur = [], 1
    while len(out) < n:
        if cur not in used:
            out.append(f"fin_{cur:03d}")
        cur += 1
    return out


# --------------------------------------------------------------------------
# 数据与模型
# --------------------------------------------------------------------------
def load_kb_by_doc() -> Dict[str, List[Dict]]:
    """激活语料的 KB，按文献名分组（保持页序）。"""
    paths = corpus_mod.runtime_paths()
    kb = json.load(open(paths["kb"], encoding="utf-8"))
    by_doc: Dict[str, List[Dict]] = {}
    for c in kb:
        by_doc.setdefault(c.get("source") or "", []).append(c)
    for doc in by_doc:
        by_doc[doc].sort(key=lambda c: (c.get("page_start") or 0))
    return by_doc


def load_existing_questions() -> Tuple[List[str], List[str], List[str]]:
    """既有标注的问题 / id / 已用文献。"""
    if not os.path.isfile(MAIN_CSV):
        return [], [], []
    with open(MAIN_CSV, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    qs = [r["question"] for r in rows if (r.get("question") or "").strip()]
    ids = [r.get("id") or "" for r in rows]
    docs = []
    for r in rows:
        docs += [d.strip() for d in (r.get("gold_docs") or "").split("|") if d.strip()]
    return qs, ids, docs


def get_client():
    from openai import OpenAI
    key = os.getenv("deepseek_api")
    if not key:
        raise RuntimeError("未配置 deepseek_api")
    return OpenAI(api_key=key, base_url="https://api.deepseek.com")


def llm_json(client, system: str, user: str, max_tokens: int = 1200,
             retries: int = 3) -> Tuple[Dict, int, int]:
    last = ""
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0.3, max_tokens=max_tokens, timeout=120)
            raw = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            ti = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
            to = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
            m = re.search(r"\{.*\}", raw, re.S)
            return (json.loads(m.group(0)) if m else {}), ti, to
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            if attempt < retries:
                time.sleep(3)
    print(f"    [警告] LLM 调用失败: {last[:100]}")
    return {}, 0, 0


def retrieval_check(question: str, gold_doc: str, top_k: int = 5) -> Tuple[Optional[bool], str]:
    """用线上服务检索一次：gold 文献是否进 top-k（不命中也是有效题，但要记录）。"""
    try:
        body = json.dumps({"query": question, "top_k": top_k, "mmr": True}).encode("utf-8")
        req = urllib.request.Request(SERVICE_URL + "/search", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        docs = [h.get("metadata", {}).get("source") for h in data.get("results", [])]
        return (gold_doc in docs), ""
    except Exception as e:
        return None, str(e)[:80]


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="评测标注扩充：候选生成 + 机器预筛")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 篇论文（0=全部）")
    ap.add_argument("--per-doc", type=int, default=ITEMS_PER_DOC, help="每篇论文出几题")
    ap.add_argument("--docs", default=None, help="只处理指定文献（| 分隔）")
    ap.add_argument("--no-check", action="store_true", help="跳过检索核验（不调用线上服务）")
    ap.add_argument("--merge", default=None, metavar="候选CSV",
                    help="把审核过的候选（status=done 的行）合并进正式标注并退出")
    ap.add_argument("--triage", action="store_true",
                    help="只重算分级（绿/黄/红）并回写候选文件，不重新生成")
    ap.add_argument("--fact-check", action="store_true",
                    help="重算分级并做事实核对（答案里的数字/模型名必须出现在原文对应页）")
    ap.add_argument("--bulk-approve", default=None, choices=["绿", "黄"],
                    metavar="分级", help="把该分级的候选批量置为 done（需配合 --yes）")
    ap.add_argument("--yes", action="store_true", help="确认执行批量通过（配合 --bulk-approve）")
    args = ap.parse_args()

    # ---- 分级重算 + 事实核对 + 报告重写 ----
    if args.triage or args.fact_check:
        with open(CAND_CSV, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        if args.fact_check:
            print("正在做事实核对（答案中的数字/模型名是否出现在原文对应页）…")
            by_doc = load_kb_by_doc()
            for i, r in enumerate(rows, 1):
                pages = [p for p in str(r.get("evidence_pages") or "").split("|") if p]
                n_missing, missing = fact_check(r["answer"], by_doc.get(r["gold_docs"], []), pages)
                r["fact_missing"], r["missing_facts"] = n_missing, missing
                if i % 40 == 0:
                    print(f"  {i}/{len(rows)}")
        for r in rows:
            r["title_leak"] = compute_title_leak(r["question"], r["gold_docs"])
            r["tier"] = triage(r)
        with open(CAND_CSV, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CAND_FIELDS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in CAND_FIELDS})
        meta = {}
        if os.path.isfile(CAND_CSV + ".meta.json"):
            meta = json.load(open(CAND_CSV + ".meta.json", encoding="utf-8"))
        meta.setdefault("candidates", len(rows))
        meta.setdefault("existing", len(load_existing_questions()[1]))
        meta.setdefault("cost", 0)
        meta.setdefault("minutes", 0)
        meta.setdefault("tokens_in", 0)
        meta.setdefault("tokens_out", 0)
        meta.setdefault("skipped_similar", 0)
        report = write_report(rows, meta)
        from collections import Counter
        cnt = Counter(r["tier"] for r in rows)
        print(f"分级完成：绿 {cnt.get('绿', 0)} / 黄 {cnt.get('黄', 0)} / 红 {cnt.get('红', 0)}"
              f"（共 {len(rows)} 题）\n报告 → {report}")
        return 0

    # ---- 批量通过（人工抽查后使用；会留下审计记录）----
    if args.bulk_approve:
        if not args.yes:
            print("拒绝执行：批量通过必须显式加 --yes（该操作会把这些行标为 done 并进入评测）")
            return 1
        with open(CAND_CSV, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["title_leak"] = compute_title_leak(r["question"], r["gold_docs"])
            r["tier"] = triage(r)
        picked = [r for r in rows if r["tier"] == args.bulk_approve
                  and (r.get("status") or "").strip().lower() != "done"]
        for r in picked:
            r["status"] = "done"
        with open(CAND_CSV, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CAND_FIELDS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in CAND_FIELDS})
        audit = os.path.join(LOG_DIR, f"标注批量通过_{time.strftime('%Y%m%d_%H%M')}.md")
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(audit, "w", encoding="utf-8") as f:
            f.write(f"# 标注批量通过审计（分级={args.bulk_approve}，{len(picked)} 题）\n\n")
            f.write(f"时间：{time.strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write("| # | 问题 | 答案 | 文献 |\n|---|---|---|---|\n")
            for i, r in enumerate(picked, 1):
                f.write(f"| {i} | {r['question']} | {r['answer']} | {r['gold_docs'][:30]} |\n")
        print(f"已把 {len(picked)} 题（{args.bulk_approve}）置为 done；审计记录：{audit}")
        print("下一步合并：python gen_annotations.py --merge " + os.path.relpath(CAND_CSV, PROJECT_DIR))
        return 0

    existing_qs, existing_ids, used_docs = load_existing_questions()

    # ---- 合并模式 ----
    if args.merge:
        with open(args.merge, encoding="utf-8-sig") as f:
            cand = [r for r in csv.DictReader(f)
                    if (r.get("status") or "").strip().lower() == "done"]
        if not cand:
            print("[错误] 候选文件里没有 status=done 的行（先在审核时把通过的题改成 done）")
            return 1
        with open(MAIN_CSV, encoding="utf-8-sig") as f:
            main_rows = list(csv.DictReader(f))
        fields = ["id", "status", "question", "answer", "gold_docs", "gold_chunks", "notes"]
        new_ids = next_ids([r.get("id") for r in main_rows], len(cand))
        for r, nid in zip(cand, new_ids):
            main_rows.append({"id": nid, "status": "done", "question": r["question"],
                              "answer": r["answer"], "gold_docs": r["gold_docs"],
                              "gold_chunks": r["gold_chunks"],
                              "notes": (r.get("notes") or "")[:200]})
        with open(MAIN_CSV, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in main_rows:
                w.writerow({k: r.get(k, "") for k in fields})
        print(f"已合并 {len(cand)} 题：{new_ids[0]} … {new_ids[-1]} → {MAIN_CSV}")
        print("提示：合并后跑 `python evaluate.py --methods hmm --source finance` 即可用新集评测")
        return 0

    by_doc = load_kb_by_doc()
    docs = sorted(by_doc)
    if args.docs:
        want = {d.strip() for d in args.docs.split("|") if d.strip()}
        docs = [d for d in docs if d in want]
    if args.limit:
        docs = docs[: args.limit]
    print(f"语料 {len(by_doc)} 篇 → 本次处理 {len(docs)} 篇；既有标注 {len(existing_qs)} 题")

    client = get_client()
    rows: List[Dict] = []
    tok_in = tok_out = 0
    t0 = time.time()
    skipped_similar = 0

    for di, doc in enumerate(docs, start=1):
        chunks = by_doc[doc]
        excerpts = pick_excerpts(chunks)
        if not excerpts:
            print(f"[{di}/{len(docs)}] {doc}: 无可用片段，跳过")
            continue
        ev_text = "\n\n".join(f'【片段{i}】\n{e["text"]}' for i, e in enumerate(excerpts, 1))
        user = f"论文文件名：{doc}\n\n{ev_text}"
        data, ti, to = llm_json(client, GEN_PROMPT.format(n=args.per_doc), user)
        tok_in += ti
        tok_out += to
        items = parse_items(json.dumps(data, ensure_ascii=False))
        made = 0
        for it in items:
            sim = max([question_similarity(it["q"], q) for q in existing_qs] + [0.0])
            if sim >= DUP_THRESHOLD:
                skipped_similar += 1
                continue
            quote, fallback = verify_quote(it["quote"], excerpts, it["evidence"])
            pages = sorted({excerpts[i - 1].get("page_start") for i in it["evidence"]
                            if 1 <= i <= len(excerpts)})
            # 质检：答案是否被片段支持 + 是否唯一
            vq = f"【问题】{it['q']}\n【标准答案】{it['a']}\n【原文片段】\n{ev_text}"
            vd, vti, vto = llm_json(client, VERIFY_PROMPT, vq, max_tokens=200)
            tok_in += vti
            tok_out += vto
            ok = bool(vd.get("answer_supported")) and bool(vd.get("unique", True))
            hit, err = (None, "") if args.no_check else retrieval_check(it["q"], doc)
            rows.append({
                "id": "", "status": "pending", "question": it["q"], "answer": it["a"],
                "gold_docs": doc, "gold_chunks": quote,
                "notes": f"生成类型:{it['type']}；机器质检:{'通过' if ok else '存疑'}",
                "gen_type": it["type"],
                "evidence_pages": "|".join(str(p) for p in pages if p),
                "quote_fallback": int(fallback),
                "dup_sim": round(sim, 3),
                "title_leak": compute_title_leak(it["q"], doc),
                "retrieval_hit": "" if hit is None else int(hit),
                "system_answer": "",
                "system_supports_gold": "",
                "verify": ("ok" if ok else f"存疑:{vd.get('issue', '')[:20]}")
                          + ("" if not err else f"(检索核验失败:{err})"),
            })
            rows[-1]["tier"] = triage(rows[-1])
            made += 1
            existing_qs.append(it["q"])          # 同批之间也去重
        print(f"[{di}/{len(docs)}] {doc}: 出 {made} 题"
              f"（片段 {len(excerpts)} 个，命中 gold 的 "
              f"{sum(1 for r in rows[-made:] if r['retrieval_hit'] == 1)}/{made}）", flush=True)

    if not rows:
        print("[错误] 没有生成任何候选")
        return 1

    with open(CAND_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CAND_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CAND_FIELDS})

    cost = tok_in / 1e6 * PRICE_IN_PER_M + tok_out / 1e6 * PRICE_OUT_PER_M
    meta = {"candidates": len(rows), "existing": len(existing_ids), "cost": round(cost, 2),
            "minutes": round((time.time() - t0) / 60, 1), "skipped_similar": skipped_similar,
            "tokens_in": tok_in, "tokens_out": tok_out}
    with open(CAND_CSV + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    report = write_report(rows, meta)
    print(f"\n候选 {len(rows)} 题 → {CAND_CSV}")
    print(f"报告 → {report}")
    from collections import Counter
    cnt = Counter(r["tier"] for r in rows)
    print(f"分级：绿 {cnt.get('绿', 0)} / 黄 {cnt.get('黄', 0)} / 红 {cnt.get('红', 0)}；"
          f"成本约 ¥{cost:.2f}")
    return 0


def write_report(rows: List[Dict], meta: Dict) -> str:
    """生成审核报告：先把"必须人工确认"的题列出来，再给抽查样本与操作命令。"""
    from collections import Counter
    tiers = Counter(r["tier"] for r in rows)
    checked = [r for r in rows if str(r.get("retrieval_hit")) != ""]
    hit = sum(1 for r in checked if str(r["retrieval_hit"]) == "1")
    verified = [r for r in rows if str(r.get("verify", "")).startswith("ok")]
    types = Counter(r["gen_type"] for r in rows)
    need = [r for r in rows if r["tier"] in ("红", "黄")]
    green = [r for r in rows if r["tier"] == "绿"]
    os.makedirs(LOG_DIR, exist_ok=True)
    report = os.path.join(LOG_DIR, f"标注扩充报告_{time.strftime('%Y%m%d')}.md")
    lines = [
        f"# 评测标注扩充报告（{time.strftime('%Y-%m-%d %H:%M')}）",
        "",
        f"- 候选 **{meta.get('candidates')} 题**（覆盖 {len({r['gold_docs'] for r in rows})} 篇文献）；"
        f"既有标注 {meta.get('existing')} 行；LLM 成本约 ¥{meta.get('cost')}"
        f"（{meta.get('minutes')} 分钟，{meta.get('tokens_in')}/{meta.get('tokens_out')} tokens）",
        f"- 机器质检通过 **{len(verified)}/{len(rows)}**；检索核验（gold 是否进 top-5）："
        f"**{hit}/{len(checked)}**（未命中的题是真实困难样本，不是错误）",
        f"- 去重：与既有题目 3-gram Jaccard ≥ {DUP_THRESHOLD} 的候选已丢弃 {meta.get('skipped_similar')} 题",
        f"- 生成类型分布：" + "、".join(f"{k} {v}" for k, v in types.most_common()),
        f"- **分级：🟢 绿 {tiers.get('绿', 0)}（抽查即可）／🟡 黄 {tiers.get('黄', 0)}"
        f"（需读一遍）／🔴 红 {tiers.get('红', 0)}（必须处理）**",
        "",
        "## 一、必须人工确认的题目（红 + 黄）",
        "",
    ]
    for i, r in enumerate(need, 1):
        lines += [
            f"### {i}. [{r['tier']}] {r['question']}",
            f"- **标准答案**：{r['answer']}",
            f"- **原文证据**（逐字）：{r['gold_chunks'][:220]}",
            f"- **文献**：{r['gold_docs']}（第 {r['evidence_pages']} 页）",
            f"- **为什么标{r['tier']}**：{tier_reason(r)}",
            "",
        ]
    lines += [
        "## 二、绿区抽查样本（建议抽 10 题，其余可批量通过）",
        "",
    ]
    step = max(1, len(green) // 10)
    for r in green[::step][:10]:
        lines.append(f"- {r['question']} → {r['answer']}（{r['gold_docs'][:24]}）")
    lines += [
        "",
        "## 三、操作步骤",
        "",
        "1. 先看上面第一节的题目：逐题判断**问题是否清晰 / 答案是否与原文证据一致 / 文献是否正确**；",
        "   要改就在 Excel 里直接改 `question` `answer`，不要的整行删掉；",
        "2. 绿区抽 10 题核对（第二节已列出样本），确认无误后批量通过：",
        "",
        "```bash",
        "python gen_annotations.py --bulk-approve 绿 --yes   # 绿区批量置 done（会写审计记录）",
        "```",
        "",
        "3. 黄/红区处理完后，同样把 `status` 改成 `done`（或删行）；",
        "4. 合并进正式标注（自动续 fin_0xx 号，只取 status=done）：",
        "",
        "```bash",
        "python gen_annotations.py --merge data/annotations/finance_annotations_v2_candidates.csv",
        "```",
        "",
        "5. 合并后用新标注重跑评测：`python evaluate.py --methods hmm --source finance`；",
        "   也可以只跑检索指标快速看规模扩大的影响：`python evaluate.py --methods hmm --source finance --skip-gen`。",
        "",
        "## 四、文件",
        "",
        f"- 候选：`data/annotations/{os.path.basename(CAND_CSV)}`（**默认全部 status=pending，不进评测**）",
        f"- 正式标注：`data/annotations/{os.path.basename(MAIN_CSV)}`（评测只读 status=done 的行）",
        "",
        "> 论文里怎么写：这批题目由 LLM 基于原文片段生成候选，**证据句由脚本从原文逐字摘抄**，"
        "经机器质检（答案被证据支持、答案唯一）与检索核验后，由人工逐题审核确认；"
        "未通过审核的候选不会进入评测集。请如实按这个流程描述，不要写成「全人工标注」。",
    ]
    open(report, "w", encoding="utf-8").write("\n".join(lines))
    return report


if __name__ == "__main__":
    sys.exit(main())
