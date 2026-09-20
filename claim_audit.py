# -*- coding: utf-8 -*-
"""claim_audit.py —— 引用/证据审计：把生成答案拆成"结论"，逐条核查是否有证据支撑。

这是"可信生成"这条创新线的 baseline 工具：回答一个问题——**当前系统给出的结论里，
有多大比例其实没有证据支撑？引用的编号有没有指对证据？**

流程：
    1. 读评测 CSV（results/<method>_finance.csv）拿 question / pred_answer / gold_sources；
    2. 用运行中的 RAG 服务按**生产配置**重新取证据池（top_k=5, MMR 开），与生成时一致；
       证据落缓存 results/_evidence_cache_<method>.json，重复跑不再打扰服务；
    3. 规则拆句 → 结论列表（去 markdown、去掉"📚 参考来源"尾巴、过滤过渡句）；
    4. 每题一次 LLM 调用，让模型逐条判定 supported / partial / unsupported，
       并判断该结论自带的 [来源N] 是否指向真正支撑它的证据块（cite_ok）；
    5. 汇总：无支撑率（结论级 + 题目级，后者带 95% t 区间，避免同题内相关性高估显著性）、
       引用指对率、逐句引用率，并挑出典型无支撑样例供汇报使用。

用法：
    python claim_audit.py --limit 3            # 先小样试跑（看提示词与解析是否正常）
    python claim_audit.py                      # 全量 40 题（hmm = 生产默认分块）
    python claim_audit.py --method discourse   # 换一个分块方法的答案（证据池仍是激活语料）
    python claim_audit.py --no-cache           # 忽略证据缓存，重新检索

输出：
    results/claim_audit_<method>_<日期>.csv     逐条结论的判定
    results/claim_audit_summary_<method>_<日期>.md  汇报用汇总（含区间与样例）
"""

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import find_dotenv, load_dotenv  # noqa: E402

load_dotenv(find_dotenv())

from rag_core.config import PROJECT_DIR  # noqa: E402

RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
SERVICE_URL = os.getenv("RAG_AUDIT_URL", "http://127.0.0.1:8000")
LLM_MODEL = "deepseek-chat"
MAX_CONTEXT_CHARS = 3000        # 与 evaluate.py / paper_qa 一致（拼上下文时的字符上限）
CLAIM_MIN_LEN = 8
MAX_CLAIMS = 12                 # 单题最多核查多少条（防止长答案把提示词撑爆）
EVIDENCE_CHARS = 1500           # 每个证据块截断长度：KB 块中位数约 1.3k 字符，
                                # 截太短会把"上下文提到第三章…"这类描述误判成无支撑（实测踩过）
PRICE_IN_PER_M = 1.5            # 与 rag_core.observability 保持一致
PRICE_OUT_PER_M = 4.5

# 评测答案与线上答案的引用语法不同：evaluate.py 提示词用「（来源2、来源4）」，
# 线上问答用「[来源1]」。两种都要认，否则引用指对率恒为空。
CITE_RE = re.compile(r"[\[（(]?\s*来源\s*(\d+)\s*[\]）)]?")
DROP_LINE_RE = re.compile(r"^(📚|参考来源|来源[:：]|以上|综上|总之|注[:：])")
NUM_BULLET_RE = re.compile(r"^\s*\d+[.、)]\s*")


# --------------------------------------------------------------------------
# 结论拆分
# --------------------------------------------------------------------------
def clean_answer(answer: str) -> str:
    """去掉 markdown 痕迹与末尾"参考来源"清单，只留论断正文。"""
    text = answer or ""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"^\s*[-*•]\s*", "", text, flags=re.M)
    lines = [ln.strip() for ln in text.splitlines()]
    kept = []
    for ln in lines:
        if not ln:
            continue
        if DROP_LINE_RE.match(ln):
            continue
        # "📚 参考来源: [来源1][来源2]" 这类尾行整体丢弃
        if len(CITE_RE.findall(ln)) >= 1 and len(CITE_RE.sub("", ln).strip()) <= 6:
            continue
        kept.append(ln)
    return "\n".join(kept)


def split_claims(answer: str) -> List[Dict]:
    """规则拆句：中文句末标点切分，过滤过短与纯过渡句。"""
    body = clean_answer(answer)
    raw = re.split(r"[。！？；\n]+", body)
    claims = []
    for piece in raw:
        s = piece.strip().strip("，,、 ")
        s = NUM_BULLET_RE.sub("", s)
        if len(s) < CLAIM_MIN_LEN:
            continue
        if not re.search(r"[\u4e00-\u9fa5]", s):
            continue
        cited = [int(x) for x in CITE_RE.findall(piece)]
        claims.append({"text": s, "cited": cited})
    return claims[:MAX_CLAIMS]


# --------------------------------------------------------------------------
# 证据池（走生产服务，保证与生成时同一套检索）
# --------------------------------------------------------------------------
def fetch_evidence(question: str, top_k: int, mmr: bool, timeout: int = 300) -> List[Dict]:
    body = json.dumps({"query": question, "top_k": top_k, "mmr": mmr}).encode("utf-8")
    req = urllib.request.Request(SERVICE_URL + "/search", data=body,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return normalize_hits(data.get("results", []))


def normalize_hits(results: List[Dict]) -> List[Dict]:
    out = []
    for i, hit in enumerate(results, start=1):
        m = hit.get("metadata", {}) or {}
        out.append({
            "n": i,
            "source": m.get("source") or "",
            "page_start": m.get("page_start"),
            "page_end": m.get("page_end"),
            "text": (hit.get("text") or ""),
        })
    return out


# --------------------------------------------------------------------------
# 自洽口径：用线上检索到的证据拼上下文 → 重新生成答案 → 审计同一份证据
# 必要性：CSV 里没有存生成时的上下文，而线上 KB 与评测时的 KB 可能不同
# （实测 fin_003 的评测答案描述了「来源2=股债市场联动研究」，线上 top5 里却没有它，
#  按 CSV 审计会把这条真实引用误判成无支撑）。
# --------------------------------------------------------------------------
def select_context_blocks(evidence: List[Dict],
                          max_chars: int = MAX_CONTEXT_CHARS) -> Tuple[str, List[Dict]]:
    """复刻 evaluate.py::make_context 的拼装规则（去重 + 字符预算），
    返回 (上下文文本, 真正进入上下文的证据块)。编号沿用检索序号，保证引用号可对齐。"""
    seen, parts, used, total = set(), [], [], 0
    for e in evidence:
        txt = (e["text"] or "").strip()
        if not txt:
            continue
        key = (e["source"], txt)
        if key in seen:
            continue
        seen.add(key)
        block = f'[来源{e["n"]}]（来源: {e["source"]}）\n{txt}'
        if parts and total + len(block) > max_chars:
            break
        parts.append(block)
        used.append({**e, "text": txt})
        total += len(block)
    return "\n\n".join(parts), used


def generate_answer(client, question: str, context: str, retries: int = 3) -> str:
    """用与 evaluate.py 相同的提示词生成答案（保证与评测口径一致）。
    DeepSeek 偶发 Connection error（实测首个请求就会抖），所以必须重试。"""
    try:
        from evaluate import EVAL_PROMPT
    except Exception:                       # 极端情况下退回内置副本
        EVAL_PROMPT = ("你是一位基于文档知识库的问答助手。\n"
                       "回答必须严格基于\"参考上下文\"，不编造信息。\n"
                       "回答末尾列出引用来源，格式：📚 参考来源: [来源N]\n"
                       "使用中文，精准简洁。")
    last = ""
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "system", "content": EVAL_PROMPT},
                          {"role": "user",
                           "content": f"问题：{question}\n\n参考上下文：\n{context}"}],
                temperature=0.3, max_tokens=512, timeout=120)
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            if attempt < retries:
                time.sleep(3)
    raise RuntimeError(last[:120])


# --------------------------------------------------------------------------
# LLM 逐条核查
# --------------------------------------------------------------------------
VERIFIER_PROMPT = """你是严格的证据核查员。给定【结论列表】与【证据列表】，逐条判断结论是否被证据支持。

判定标准：
- supported：证据直接给出了该结论所需的全部关键信息（模型名、数值、方向、对象等一致）；
- partial：证据只支持结论的一部分，有关键要素在证据里查不到（例如只提到某模型，但没说它效果最好）；
- unsupported：证据里找不到支撑，或与证据矛盾；
- meta：这条不是对文献内容的断言，而是对上下文/来源本身的描述、免责说明或拒答
  （例如"上下文中未明确列出具体算法""参考资料为某文档"）——不计入无支撑率。

引用核查：若结论自带引用标记（来源N 或 [来源N]），判断 N 指向的证据块是否**就是**支撑该结论的证据
（是→true；引用了但指错块或该块并不支撑→false；结论没有引用标记→null）。

严格要求：
- 只依据给定证据判断，不使用你的先验知识；证据不足就是 unsupported；
- 证据是按块截断的片段，只有确实矛盾才判 unsupported，看不到不等于矛盾（看不到关键要素判 partial）；
- 数字、模型名、结论方向必须对得上，措辞相近但含义不同的算 unsupported；
- 只输出 JSON，不要解释文字。格式：
{"claims":[{"id":"C1","verdict":"supported|partial|unsupported|meta","evidence":[1,3],"cite_ok":true,"note":"不超过25字"}]}
其中 evidence 是支撑该结论的证据编号（没有则空数组），note 用简短中文说明理由或缺失要素。"""


def build_user_prompt(question: str, claims: List[Dict], evidence: List[Dict]) -> str:
    ev_lines = []
    for e in evidence:
        page = f'第{e["page_start"]}-{e["page_end"]}页' if e.get("page_start") else ""
        ev_lines.append(f'[证据{e["n"]}]（{e["source"]} {page}）\n{e["text"]}')
    cl_lines = []
    for i, c in enumerate(claims, start=1):
        tag = "".join(f"[来源{n}]" for n in c["cited"]) or "（无引用标记）"
        cl_lines.append(f'C{i}. {c["text"]} {tag}')
    return (f"【问题】{question}\n\n【结论列表】\n" + "\n".join(cl_lines)
            + "\n\n【证据列表】\n" + "\n\n".join(ev_lines))


def verify_claims(client, question: str, claims: List[Dict], evidence: List[Dict]) -> Tuple[List[Dict], int, int]:
    """返回 (每题判定列表, tokens_in, tokens_out)。解析失败则全部标记 error。"""
    if not claims:
        return [], 0, 0
    messages = [{"role": "system", "content": VERIFIER_PROMPT},
                {"role": "user", "content": build_user_prompt(question, claims, evidence)}]
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(model=LLM_MODEL, messages=messages,
                                                 temperature=0.0, max_tokens=1200, timeout=90)
            raw = (resp.choices[0].message.content or "").strip()
            usage = getattr(resp, "usage", None)
            t_in = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
            t_out = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
            m = re.search(r"\{.*\}", raw, re.S)
            if not m:
                raise ValueError(f"未找到 JSON: {raw[:80]}")
            data = json.loads(m.group(0))
            by_id = {str(x.get("id", "")).upper(): x for x in data.get("claims", [])}
            out = []
            for i, c in enumerate(claims, start=1):
                got = by_id.get(f"C{i}", {})
                verdict = str(got.get("verdict", "error")).lower()
                if verdict not in ("supported", "partial", "unsupported", "meta"):
                    verdict = "error"
                ev = got.get("evidence") or []
                if not isinstance(ev, list):
                    ev = []
                cite_ok = got.get("cite_ok", None)
                out.append({**c, "verdict": verdict,
                            "evidence": [int(x) for x in ev if str(x).isdigit()],
                            "cite_ok": cite_ok if isinstance(cite_ok, bool) else None,
                            "note": str(got.get("note", ""))[:60]})
            return out, t_in, t_out
        except Exception as e:
            if attempt == 2:
                return [{**c, "verdict": "error", "evidence": [], "cite_ok": None,
                         "note": f"核查失败: {str(e)[:40]}"} for c in claims], 0, 0
            time.sleep(2)


# --------------------------------------------------------------------------
# 统计汇总
# --------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """比例指标的 Wilson 置信区间（比正态近似稳，n 小的时候也能用）。"""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, center - half), min(1.0, center + half))


def mean_ci(values: List[float]) -> Tuple[float, float, float]:
    """题目级均值的 95% t 区间（同题内结论相关，按题聚合后再算区间更保守）。"""
    n = len(values)
    if n == 0:
        return (0.0, 0.0, 0.0)
    mean = sum(values) / n
    if n == 1:
        return (mean, mean, mean)
    sd = statistics.stdev(values)
    t = 2.023 if n >= 40 else (2.093 if n >= 25 else 2.262)   # 粗略 t 临界值（df≈n-1）
    half = t * sd / (n ** 0.5)
    return (mean, mean - half, mean + half)


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="引用/证据审计：量化无证据支撑的结论占比")
    ap.add_argument("--method", default="hmm", help="评测 CSV 里的分块方法（默认 hmm）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0=全部）")
    ap.add_argument("--top-k", type=int, default=5, help="证据池大小（生产配置为 5）")
    ap.add_argument("--no-mmr", action="store_true", help="关掉 MMR（默认与生产一致：开）")
    ap.add_argument("--no-cache", action="store_true", help="忽略证据缓存，重新检索")
    ap.add_argument("--from-live", action="store_true",
                    help="自洽口径：用线上检索的证据重新生成答案，再审计同一份证据"
                         "（默认审计 CSV 里的历史答案，需注意其上下文可能已不可复现）")
    args = ap.parse_args()

    csv_path = os.path.join(RESULTS_DIR, f"{args.method}_finance.csv")
    if not os.path.isfile(csv_path):
        print(f"[错误] 找不到评测结果: {csv_path}")
        return 1
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig")))
    if args.limit:
        rows = rows[: args.limit]

    from openai import OpenAI
    key = os.getenv("deepseek_api")
    if not key:
        print("[错误] 未配置 deepseek_api")
        return 1
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com")

    stamp = time.strftime("%Y%m%d_%H%M")
    tag = "live" if args.from_live else args.method
    detail_path = os.path.join(RESULTS_DIR, f"claim_audit_{tag}_{stamp}.csv")
    summary_path = os.path.join(RESULTS_DIR, f"claim_audit_summary_{tag}_{stamp}.md")
    answers_path = os.path.join(RESULTS_DIR, f"claim_audit_answers_{tag}_{stamp}.csv")

    cache_path = os.path.join(RESULTS_DIR, f"_evidence_cache_{tag}.json")
    cache = {}
    if os.path.isfile(cache_path) and not args.no_cache:
        try:
            cache = json.load(open(cache_path, encoding="utf-8"))
        except Exception:
            cache = {}

    all_claims: List[Dict] = []
    per_question: List[Dict] = []
    live_answers: List[Dict] = []
    tokens_in = tokens_out = 0
    t_start = time.time()

    for i, r in enumerate(rows, start=1):
        qid, question = r["qid"], r["question"]
        gold_docs = [d for d in (r.get("gold_sources") or "").split("|") if d.strip()]
        answer = r.get("pred_answer") or ""
        if qid in cache and not args.no_cache:
            evidence = cache[qid]
            src = "缓存"
        else:
            try:
                evidence = fetch_evidence(question, args.top_k, not args.no_mmr)
                cache[qid] = evidence
                src = "在线"
            except Exception as e:
                print(f"[{i}/{len(rows)}] {qid} 检索失败: {e}")
                continue

        if args.from_live:
            # 自洽口径：证据 → 上下文 → 现场生成 → 审计同一份证据
            context, evidence = select_context_blocks(evidence)
            try:
                answer = generate_answer(client, question, context)
            except Exception as e:
                print(f"[{i}/{len(rows)}] {qid} 生成失败: {e}")
                continue
            cache[qid] = evidence          # 缓存里只留真正进入上下文的块

        claims = split_claims(answer)
        if not claims and (answer or "").strip():
            # 短拒答（如"上下文未提供该信息。"）也要送审：拒答率本身就是可信度指标之一
            claims = [{"text": clean_answer(answer).strip()[:150] or answer.strip()[:150],
                       "cited": [int(x) for x in CITE_RE.findall(answer)]}]
        if not claims:
            print(f"[{i}/{len(rows)}] {qid} 无可核查结论（答案为空），跳过")
            continue
        verdicts, t_in, t_out = verify_claims(client, question, claims, evidence)
        tokens_in += t_in
        tokens_out += t_out

        ev_docs = {e["source"] for e in evidence}
        ev_nums = {e["n"] for e in evidence}
        cited_nums = [int(x) for x in CITE_RE.findall(answer)]
        invalid_cites = sorted({n for n in cited_nums if n not in ev_nums})
        gold_in_pool = bool(set(gold_docs) & ev_docs) if gold_docs else None
        per_question.append({
            "qid": qid,
            "n_claims": len(verdicts),
            "unsupported": sum(1 for v in verdicts if v["verdict"] == "unsupported"),
            "partial": sum(1 for v in verdicts if v["verdict"] == "partial"),
            "supported": sum(1 for v in verdicts if v["verdict"] == "supported"),
            "meta": sum(1 for v in verdicts if v["verdict"] == "meta"),
            "recall5_doc": float(r.get("recall@5") or 0),
            "gold_in_pool": gold_in_pool,
            "judge_faith": float(r.get("judge_faith") or 0),
            "n_citations": len(cited_nums),
            "invalid_citations": "|".join(str(x) for x in invalid_cites),
        })
        live_answers.append({"qid": qid, "question": question, "answer": answer,
                             "n_evidence": len(evidence), "gold_in_pool": gold_in_pool,
                             "citations": len(cited_nums),
                             "invalid_citations": "|".join(str(x) for x in invalid_cites)})
        for j, v in enumerate(verdicts, start=1):
            all_claims.append({
                "qid": qid, "claim_id": f"C{j}", "claim": v["text"],
                "verdict": v["verdict"], "note": v["note"],
                "cited": "|".join(str(x) for x in v["cited"]),
                "cite_ok": "" if v["cite_ok"] is None else str(v["cite_ok"]).lower(),
                "evidence": "|".join(str(x) for x in v["evidence"]),
                "gold_sources": r.get("gold_sources") or "",
                "gold_in_pool": per_question[-1]["gold_in_pool"],
            })
        uns = per_question[-1]["unsupported"]
        print(f"[{i}/{len(rows)}] {qid} 结论 {len(verdicts)} 条，"
              f"无支撑 {uns} 条（证据{src}，gold在池内={per_question[-1]['gold_in_pool']}）",
              flush=True)

    json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=0)

    # ---- 逐条明细 ----
    if not all_claims:
        print("[错误] 没有产出任何判定结果（全部题目检索或生成失败），请检查服务与网络")
        return 1
    with open(detail_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_claims[0].keys()))
        w.writeheader()
        w.writerows(all_claims)
    if live_answers:
        with open(answers_path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(live_answers[0].keys()))
            w.writeheader()
            w.writerows(live_answers)

    # ---- 汇总 ----
    n_claims = len(all_claims)
    n_meta = sum(1 for c in all_claims if c["verdict"] == "meta")
    n_err = sum(1 for c in all_claims if c["verdict"] == "error")
    # 内容性结论 = 排除 meta（对上下文的描述/拒答）与判定失败
    content = [c for c in all_claims if c["verdict"] in ("supported", "partial", "unsupported")]
    n_content = len(content)
    n_uns = sum(1 for c in content if c["verdict"] == "unsupported")
    n_par = sum(1 for c in content if c["verdict"] == "partial")
    n_sup = sum(1 for c in content if c["verdict"] == "supported")
    cited = [c for c in content if c["cited"]]
    cite_ok = [c for c in cited if c["cite_ok"] == "true"]
    q_rates = [(q["unsupported"] / (q["n_claims"] - q["meta"]))
               for q in per_question if q["n_claims"] > q["meta"]]
    q_mean, q_lo, q_hi = mean_ci(q_rates)
    w_lo, w_hi = wilson(n_uns, max(1, n_content))
    cost = tokens_in / 1e6 * PRICE_IN_PER_M + tokens_out / 1e6 * PRICE_OUT_PER_M
    elapsed = time.time() - t_start

    examples = [c for c in content if c["verdict"] == "unsupported"][:5]
    pool_hit = [q for q in per_question if q["gold_in_pool"] is not None]
    pool_rate = (sum(1 for q in pool_hit if q["gold_in_pool"]) / len(pool_hit)) if pool_hit else 0.0
    mode_note = ("答案与证据**同源**（用线上检索结果重新生成答案，再审计同一份上下文）"
                 if args.from_live else
                 "审计的是历史评测 CSV 里的答案——其生成时的上下文未存档，"
                 "若线上 KB 与评测 KB 有差异，部分「无支撑」可能是证据池不一致造成的伪阳性")
    lines = [
        f"# 引用/证据审计基线（分块={args.method}，{len(per_question)} 题 / {n_content} 条内容性结论）",
        "",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M')}，耗时 {elapsed / 60:.1f} 分钟，"
        f"LLM 成本约 ¥{cost:.2f}（in {tokens_in} / out {tokens_out} tokens）",
        f"- 口径：{mode_note}",
        f"- 证据池：生产配置 top_k={args.top_k}，MMR={'关' if args.no_mmr else '开'}，"
        f"拼上下文上限 {MAX_CONTEXT_CHARS} 字符（与 evaluate.py 一致）；"
        f"缓存于 `{os.path.basename(cache_path)}`",
        f"- 检索侧对照：题目级 gold 文档命中率 {pct(pool_rate)}（{sum(1 for q in pool_hit if q['gold_in_pool'])}/{len(pool_hit)} 题）",
        f"- 引用越界（引用了上下文里不存在的来源号，即凭空编造来源）："
        f"{sum(1 for q in per_question if q.get('invalid_citations'))} 题 / "
        f"{sum(len(q.get('invalid_citations', '').split('|')) if q.get('invalid_citations') else 0 for q in per_question)} 处",
        "",
        "## headline",
        "",
        f"- **无证据支撑的结论占比：{pct(q_mean)}（题目级均值，95% CI {pct(q_lo)} ~ {pct(q_hi)}，n={len(q_rates)} 题）**",
        f"- 结论级口径（池化）：无支撑 {n_uns}/{n_content} = {pct(n_uns / max(1, n_content))}"
        f"（Wilson 95% CI {pct(w_lo)} ~ {pct(w_hi)}）；部分支持 {n_par}，有支撑 {n_sup}",
        f"- 另有 {n_meta} 条为「对上下文的描述 / 拒答」（如「上下文未明确列出」），不计入分母"
        + (f"；判定失败 {n_err} 条" if n_err else ""),
        f"- 引用规范：逐句带引用标记的内容性结论 {len(cited)}/{n_content} = {pct(len(cited) / max(1, n_content))}；"
        f"其中引用指对证据的占 {pct(len(cite_ok) / max(1, len(cited)))}（{len(cite_ok)}/{len(cited)}）",
        f"- 对照：历史 LLM-judge 忠实性均分 "
        f"{sum(q['judge_faith'] for q in per_question) / max(1, len(per_question)):.2f}/5"
        f"（结论级审计能暴露它看不到的问题）",
        "",
        "## 典型无支撑样例（汇报可直接用）",
        "",
    ]
    for c in examples:
        lines.append(f"- **[{c['qid']}]** {c['claim'][:110]}  \n"
                     f"  判定：{c['verdict']}（{c['note']}）；证据编号：{c['evidence'] or '无'}")
    lines += ["", "## 逐题明细", "",
              "| 题号 | 结论数 | 无支撑 | 部分支持 | 有支撑 | 描述/拒答 | 引用数 | 越界引用 | gold 在证据池 | judge 忠实性 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for q in per_question:
        lines.append(f"| {q['qid']} | {q['n_claims']} | {q['unsupported']} | {q['partial']} | "
                     f"{q['supported']} | {q['meta']} | {q.get('n_citations', 0)} | "
                     f"{q.get('invalid_citations') or '-'} | "
                     f"{'是' if q['gold_in_pool'] else '否'} | {q['judge_faith']} |")
    lines += ["", f"逐条判定明细见 `{os.path.basename(detail_path)}`。"]
    open(summary_path, "w", encoding="utf-8").write("\n".join(lines))

    print(f"\n完成：{n_content} 条内容性结论，无支撑 {n_uns} 条（题目级 {pct(q_mean)}）")
    print(f"明细：{detail_path}\n汇总：{summary_path}\n成本约 ¥{cost:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
