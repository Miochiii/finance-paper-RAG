# -*- coding: utf-8 -*-
"""
evaluate.py — RAG 分块方法消融评测框架

设计原则：唯一变量是"分块方式"，其余（文档加载、检索参数、生成配置、Prompt）全部固定。

评测流程：
  文档 -> 分块(方法M) -> 建索引 -> 对每个问题检索 topK
    -> 检索指标 Recall@5 / MRR / nDCG@5（用 gold 证据）
    -> 相同 Prompt 生成答案 -> QA 指标 EM / F1
    -> 逐条写入 results/{method}_{source}.csv  （配对 t 检验的前提）

数据源：
  - finance:  金融论文语料 + 人工标注 data/annotations/finance_annotations.csv（格式见同目录 .csv.template）
  - hotpotqa: HotpotQA 子集 data/hotpotqa_subset.json（官方格式）

用法：
  python evaluate.py --methods fixed,discourse,hybrid --source finance --limit 5
  python evaluate.py --methods all --source both --skip-gen
  python evaluate.py --methods fixed,hybrid --source both --ttest
  python evaluate.py --ttest-only --pairs hmm,fixed hmm,hybrid --source finance   # 只做检验，不重跑评测
  python evaluate.py --methods all --ttest-only --source finance                 # 全部两两组合
"""
import os
# ── 环境变量加载（.env 或系统环境变量） ──
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except ImportError:
    pass

import argparse
import csv
import json
import math
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from typing import Dict, List, Optional, Tuple

# ====================================================================
# 一、固定配置（写死。所有分块方法共用完全相同的下游配置）
# ====================================================================
CHUNK_SIZE = 800          # 分块 token 上限（对 fixed/discourse/hybrid 统一）
OVERLAP_TOKENS = 50       # 相邻块重叠 token
RERANK_K = 5              # 检索返回 topK（= 评测里的 k）
BM25_K = 20               # BM25 召回数（写死）
VECTOR_K = 20             # 向量召回数（写死）
GENERATION_MODEL = "deepseek-chat"
GENERATION_TEMPERATURE = 0.0
MAX_CONTEXT_CHARS = 3000  # 与 paper_qa._MAX_CONTEXT_CHARS 一致
MAX_GEN_RETRIES = 2

# 路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ANNOTATIONS_CSV = os.path.join(BASE_DIR, "data", "annotations", "finance_annotations.csv")
HOTPOTQA_JSON = os.path.join(BASE_DIR, "data", "hotpotqa_subset.json")
DOCS_DIR = os.path.join(BASE_DIR, "data", "pdfs")   # 原始 PDF 目录（仅未指定 --docs-cache 时的慢速加载路径；推荐直接用 MinerU 缓存）
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
VECTOR_DB_ROOT = os.path.join(OUTPUT_DIR, "vector_db")
DOCS_CACHE = os.path.join(BASE_DIR, "data", "docs_cache.json")   # 文档加载缓存（避免每次重跑都 OCR）
HMM_CHUNK_CACHE = os.path.join(BASE_DIR, "data", "hmm_chunk_cache")  # HMM 块级缓存（同文本同参数只算一次）

# ====================================================================
# 二、Prompt（与 paper_qa.py 完全一致，保证生成配置相同）
# ====================================================================
PROMPT_FACTUAL = """你是一位基于文档知识库的问答助手。
回答必须严格基于"参考上下文"，不编造信息。
回答末尾列出引用来源，格式：📚 参考来源: [来源N]
使用中文，精准简洁。"""

PROMPT_SUMMARY = """你是一位基于文档知识库的学术综述助手。
请综合"参考上下文"中的多个信息片段，给出系统性的概述。
如果信息不完整，指出缺失部分，不编造。
回答末尾列出引用来源，格式：📚 参考来源: [来源N]
使用中文，结构清晰。"""

PROMPT_MIXED = """你是一位基于文档知识库的问答助手。
先精准回答事实性问题，再补充相关背景或延伸。
所有判断严格基于"参考上下文"，不编造。
回答末尾列出引用来源，格式：📚 参考来源: [来源N]
使用中文，条理分明。"""

# 评测统一用 FACTUAL prompt（保持完全一致；可按需要改这里）
EVAL_PROMPT = PROMPT_FACTUAL

# LLM-as-judge：对生成答案做 正确性+忠实性 双维打分（1-5 整数）
JUDGE_PROMPT = (
    "你是严谨的 RAG 答案评审员。请对【模型答案】在两个维度打分（1-5 整数）：\n"
    "1. correctness 正确性：对照【标准答案】，模型答案信息是否准确、是否答非所问；\n"
    "2. faithfulness 忠实性：模型答案是否完全基于【参考上下文】，有无编造/幻觉。\n"
    "严格输出 JSON 对象，例如：\"{\\\"correctness\":5,\\\"faithfulness\":4,\\\"reason\":\"简短理由\"}\"\n"
    "只输出 JSON，不要多余文字。"
)

# ====================================================================
# 三、分块方法注册表（唯一变量）。
#    5 组对比：fixed / discourse / hybrid / HMM-BIC自适应 / HMM-固定K
#    （后两者构成"K 如何确定"的消融，对应方法论文档）
# ====================================================================
def _chunk_fixed(text: str) -> List[str]:
    """baseline：按 token 数固定长度切分 + 重叠（单位与 discourse/hybrid 一致）。"""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        enc = None
    if enc is None:
        # 降级：字符级切分
        step = max(CHUNK_SIZE * 2, 1)
        return [text[i:i + step] for i in range(0, len(text), step) if text[i:i + step].strip()]
    tokens = enc.encode(text)
    if not tokens:
        return []
    n = len(tokens)
    out, i = [], 0
    while i < n:
        j = min(i + CHUNK_SIZE, n)
        out.append(enc.decode(tokens[i:j]))
        if j >= n:
            break
        nxt = j - OVERLAP_TOKENS
        if nxt <= i:
            nxt = i + 1  # 防死循环
        i = nxt
    return out


def _chunk_discourse(text: str) -> List[str]:
    """章节感知分块（现有实现，DISCOURSE）。"""
    from rag_core.chunk_splitter import dispatch_chunk
    return dispatch_chunk(text, "DISCOURSE", CHUNK_SIZE, OVERLAP_TOKENS)


def _chunk_hybrid(text: str) -> List[str]:
    """混合分块（现有实现，HYBRID，含表格识别）。"""
    from rag_core.chunk_splitter import dispatch_chunk
    return dispatch_chunk(text, "HYBRID", CHUNK_SIZE, OVERLAP_TOKENS)


def _chunk_hmm(text: str) -> List[str]:
    """HMM 最优文本分块（无监督话题分割，BIC 自适应选 K）。"""
    from rag_core.hmm_chunker import hmm_chunk
    return hmm_chunk(
        text,
        chunk_size=CHUNK_SIZE,
        overlap_tokens=OVERLAP_TOKENS,
        cache_dir=os.path.join(BASE_DIR, "data", "hmm_embed_cache"),
        chunk_cache_dir=HMM_CHUNK_CACHE,
        bic_coef=HMM_BIC_COEF,
        verbose=False,
    )


HMM_FIXED_K = 12  # HMM 固定 K 对比组（方法论文档：回答"K 如何确定"，与 BIC 自适应构成消融）
HMM_BIC_COEF = 2.0  # HMM 正式配置的 BIC 惩罚系数（与 CLI 验证一致，见 log/HMM建模优化_2026-08-16.txt 第七节）


def _chunk_hmm_fixed_k(text: str) -> List[str]:
    """HMM 固定 K=HMM_FIXED_K 对比组：不做 BIC 选 K，其余（种子择优、后处理）与 hmm 完全一致。"""
    from rag_core.hmm_chunker import hmm_chunk
    return hmm_chunk(
        text,
        chunk_size=CHUNK_SIZE,
        overlap_tokens=OVERLAP_TOKENS,
        cache_dir=os.path.join(BASE_DIR, "data", "hmm_embed_cache"),
        chunk_cache_dir=HMM_CHUNK_CACHE,
        k_min=HMM_FIXED_K,
        k_max=HMM_FIXED_K,
        bic_coef=HMM_BIC_COEF,
        verbose=False,
    )


CHUNKERS = {
    "fixed": _chunk_fixed,              # baseline：固定长度
    "discourse": _chunk_discourse,      # 章节感知
    "hybrid": _chunk_hybrid,            # 混合（含表格）
    "hmm": _chunk_hmm,                  # 创新点：HMM BIC 自适应话题分块
    "hmm_fixed_k": _chunk_hmm_fixed_k,  # ablation：HMM 固定 K=12
}


def _available_methods(methods_arg: str) -> List[str]:
    ms = [m.strip() for m in methods_arg.split(",") if m.strip()]
    if "all" in ms:
        ms = list(CHUNKERS.keys())
    avail = []
    for m in ms:
        if m in CHUNKERS:
            avail.append(m)
        else:
            print(f"  [SKIP] 未知分块方法: {m}")
    return avail


# ====================================================================
# 四、数据加载
# ====================================================================
def load_finance(csv_path: str = ANNOTATIONS_CSV) -> List[Dict]:
    """读取人工标注 CSV（只取 status==done 的行）。"""
    qbank = []
    if not os.path.exists(csv_path):
        print(f"  [WARN] 找不到标注文件: {csv_path}（先填写 finance_annotations.csv）")
        return qbank
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if (row.get("status") or "").strip().lower() != "done":
                continue
            q = (row.get("question") or "").strip()
            a = (row.get("answer") or "").strip()
            if not q or not a:
                continue
            gold_docs = [g.strip() for g in re.split(r"[|,;]", row.get("gold_docs") or "") if g.strip()]
            # 标注常从 JSON 原文复制，可能带字面量 "\n" 转义 → 还原为真实换行（归一化会再删除，但避免把 \n 当成字母 n）
            gold_chunks = [g.strip().replace("\\n", "\n") for g in re.split(r"[|,;]", row.get("gold_chunks") or "") if g.strip()]
            qbank.append({
                "qid": (row.get("id") or f"fin_{i:04d}").strip(),
                "question": q, "answer": a,
                "gold_sources": gold_docs, "gold_texts": gold_chunks,
                "source": "finance",
            })
    return qbank


def load_hotpotqa(json_path: str = HOTPOTQA_JSON) -> List[Dict]:
    """读取 HotpotQA 子集（官方 dev 格式）。
    gold_sources = supporting_facts 的标题；gold_texts = 对应支撑句。"""
    qbank = []
    if not os.path.exists(json_path):
        print(f"  [WARN] 找不到 HotpotQA: {json_path}（可跳过）")
        return qbank
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    for i, d in enumerate(data):
        ctx = d.get("context") or []
        ctx_map = {t: sents for t, sents in ctx}
        sf = d.get("supporting_facts") or []
        gold_sources = [t for t, _ in sf]
        gold_texts = []
        for t, sid in sf:
            sents = ctx_map.get(t, [])
            if isinstance(sid, int) and sid < len(sents):
                gold_texts.append(sents[sid])
        qbank.append({
            "qid": f"hotpot_{i:04d}",
            "question": (d.get("question") or "").strip(),
            "answer": (d.get("answer") or "").strip(),
            "gold_sources": gold_sources, "gold_texts": gold_texts,
            "source": "hotpotqa",
        })
    return qbank


# 排除缓存/非文档文件（DOCS_DIR 里混有 knowledge_base.json 等）
_EXCLUDE = {"knowledge_base.json", "retriever_state.json", "docx_chunks.json"}
_DOC_EXTS = (".pdf", ".docx", ".doc", ".txt")


def _doc_dir_manifest(docs_dir: str) -> Dict[str, list]:
    """文档目录清单：{文件名: [mtime_ns, size]}，用于检测目录增删改。"""
    files = sorted(f for f in os.listdir(docs_dir)
                   if f.lower().endswith(_DOC_EXTS) and f not in _EXCLUDE)
    man: Dict[str, list] = {}
    for f in files:
        st = os.stat(os.path.join(docs_dir, f))
        man[f] = [st.st_mtime_ns, st.st_size]
    return man


def _save_docs_cache(docs: Dict[str, str], docs_dir: str) -> None:
    """保存文档加载缓存 + 目录清单（清单不一致时缓存自动失效）。"""
    os.makedirs(os.path.dirname(DOCS_CACHE), exist_ok=True)
    with open(DOCS_CACHE, "w", encoding="utf-8") as fp:
        json.dump(docs, fp, ensure_ascii=False)
    with open(DOCS_CACHE + ".manifest", "w", encoding="utf-8") as fp:
        json.dump(_doc_dir_manifest(docs_dir), fp)
    print(f"  [CACHE] 已保存到 docs_cache.json（含目录清单校验），下次评测将秒开")


def load_docs(docs_dir: str = DOCS_DIR, use_cache: bool = True) -> Dict[str, str]:
    """加载文档目录下的所有 PDF/DOCX -> {filename: raw_text}。
    所有分块方法共用同一份原文（加载只做一次）。
    use_cache=True 时优先读 docs_cache.json，并校验目录清单：
    目录里文件增/删/改（mtime/大小）任一变化 → 缓存失效、重新加载。"""
    if use_cache and os.path.exists(DOCS_CACHE):
        try:
            with open(DOCS_CACHE + ".manifest", "r", encoding="utf-8") as fp:
                manifest = json.load(fp)
            if manifest != _doc_dir_manifest(docs_dir):
                print("  [CACHE] 目录内容变化（增/删/改），缓存失效，重新加载")
            else:
                with open(DOCS_CACHE, "r", encoding="utf-8") as fp:
                    cache = json.load(fp)
                print(f"  [CACHE] 命中缓存 docs_cache.json（{len(cache)} 篇），跳过 PDF 解析/OCR")
                return cache
        except Exception as e:
            print(f"  [WARN] 缓存校验失败，重新加载: {e}")
    from rag_core.document_loader import load_document
    from rag_core.classify_file import DocumentClassifier
    analyzer = DocumentClassifier()
    docs: Dict[str, str] = {}
    files = sorted(f for f in os.listdir(docs_dir)
                   if f.lower().endswith(_DOC_EXTS) and f not in _EXCLUDE)
    if not files:
        print(f"  [ERR] {docs_dir} 下没有 PDF/DOCX 文档")
        return docs
    for f in files:
        p = os.path.join(docs_dir, f)
        try:
            page_results = None
            if f.lower().endswith(".pdf"):
                from rag_core.classify_file import PDFPageAnalyzer
                page_results = PDFPageAnalyzer.analyze_pdf_fast(p)
            info = analyzer.classify(p, page_results=page_results)
            text = load_document(p, info["extractor"], page_results=page_results)
            if text and text.strip():
                docs[f] = text
                print(f"  [OK] {f}  ({len(text)}字, {info.get('extractor')}, {info.get('structure')})")
            else:
                print(f"  [SKIP] {f}: 空文本")
        except Exception as e:
            print(f"  [ERR] {f}: {e}")
    print(f"  共加载 {len(docs)} 篇文档")
    try:
        _save_docs_cache(docs, docs_dir)
    except Exception as e:
        print(f"  [WARN] 缓存保存失败: {e}")
    return docs


# ====================================================================
# 五、指标计算（检索指标 + QA 指标）
# ====================================================================
def _relevant(meta_source, gold_sources: List[str]) -> bool:
    s = (meta_source or "").strip()
    return any(s and g and s == g.strip() for g in gold_sources if g.strip())


def retrieval_metrics(results: List[Dict], gold_sources: List[str], k: int = RERANK_K):
    """从检索结果算 Recall@k / MRR / nDCG@k。
    相关判定：块 metadata.source 命中任一 gold 文档。
    nDCG 按文档粒度计算：每个相关文档只取它在 top-k 中"首次命中块"的增益。
    若按块累积增益（同一文档的多个块都加分），而分母 idcg 又是按文档数算的
    理想 DCG，nDCG 会超过 1——2026-08-28 修正，回归测试见 tests/test_evaluate_metrics.py。"""
    gs = [g for g in (gold_sources or []) if g.strip()]
    if not gs:
        return 0.0, 0.0, 0.0
    top = results[:k]
    hit_ranks = []
    for g in gs:
        rank = None
        for i, r in enumerate(top, 1):
            if _relevant(r.get("metadata", {}).get("source", ""), [g]):
                rank = i
                break
        if rank is not None:
            hit_ranks.append(rank)
    recall = len(hit_ranks) / len(gs)
    mrr = 1.0 / min(hit_ranks) if hit_ranks else 0.0
    # 文档粒度 nDCG：每篇相关文档仅在首次命中位置贡献一次增益
    dcg = sum(1.0 / math.log2(r + 1) for r in hit_ranks)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gs), k) + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return recall, mrr, ndcg


def retrieval_metrics_chunk(results: List[Dict], gold_texts: List[str], k: int = RERANK_K):
    """块级证据指标（用 gold_chunks / gold_texts，文档级指标无法区分分块方法）。
    相关判定：块文本（归一化后）包含任一 gold 证据文本。
    - finance：gold_texts = 标注 gold_chunks（目前是章节标题；升级为证据句后自动更准）；
    - hotpotqa：gold_texts = supporting_facts 支撑句（句级，直接可用）。
    返回 (recall@k, mrr, ndcg@k)：recall = 命中的证据条数 / 证据总数。
    nDCG：块"相关"= 覆盖至少一条"尚未被更靠前块覆盖"的 gold 证据（每条证据最多
    计一次增益，保证 nDCG <= 1）；同一块覆盖多条证据也只计该块一次增益。"""
    gs = [g for g in (gold_texts or []) if g and g.strip()]
    if not gs:
        return 0.0, 0.0, 0.0
    top = results[:k]
    covered = set()
    rel_ranks = []
    for i, r in enumerate(top, 1):
        ct = _norm_text_strip_pages(r.get("text") or "")
        if not ct:
            continue
        hit_here = [j for j, g in enumerate(gs)
                    if j not in covered and _norm_text(g) and _norm_text(g) in ct]
        if hit_here:
            rel_ranks.append(i)
            covered.update(hit_here)
    recall = len(covered) / len(gs)
    mrr = 1.0 / rel_ranks[0] if rel_ranks else 0.0
    dcg = sum(1.0 / math.log2(i + 1) for i in rel_ranks)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gs), k) + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return recall, mrr, ndcg


def _norm_text(s: str) -> str:
    """归一化：全角→半角（NFKC）、去空白与标点、转小写（中英文通用）。
    NFKC 让 ｐｒｅｃｉｓｉｏｎ/２０２３ 与 precision/2023 等价——PDF 提取常见全角字符。"""
    s = str(s or "")
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[\s\W_]+", "", s, flags=re.UNICODE)
    return s.lower()


_PAGE_MARKER_RE = re.compile(r"【第\s*\d+\s*页[^】]*】")


def _norm_text_strip_pages(s: str) -> str:
    """归一化前先剥掉页码标记（【第56页】/【第56页-文本】等）。
    证据句跨页时页码标记会插在词中间（"进行产【第56页】品回测"），
    不剥掉则永远无法逐字命中——匹配用此函数，展示/分块仍保留标记。"""
    return _norm_text(_PAGE_MARKER_RE.sub("", str(s or "")))


def em_f1(pred: str, gold: str):
    """EM 与 F1。中文按字符，英文按词。"""
    pn, gn = _norm_text(pred), _norm_text(gold)
    em = 1.0 if pn and pn == gn else 0.0
    if not pn or not gn:
        return em, 0.0
    if re.search(r"[\u4e00-\u9fff]", pn + gn):
        pt, gt = list(pn), list(gn)
    else:
        pt, gt = pn.split(), gn.split()
    pc, gc = Counter(pt), Counter(gt)
    overlap = sum((pc & gc).values())
    if overlap == 0:
        return em, 0.0
    prec = overlap / len(pt)
    rec = overlap / len(gt)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return em, f1


def make_context(results: List[Dict], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """把检索结果组装成 LLM 上下文（与 paper_qa._build_context 同思路）。"""
    seen, parts, total = set(), [], 0
    for i, r in enumerate(results, 1):
        src = r.get("metadata", {}).get("source", "未知")
        txt = (r.get("text") or "").strip()
        if not txt:
            continue
        key = (src, txt)
        if key in seen:
            continue
        seen.add(key)
        block = f"[来源{i}]（来源: {src}）\n{txt}"
        if parts and total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


# ====================================================================
# 六、生成答案（固定配置）
# ====================================================================
def _get_client():
    from openai import OpenAI
    key = os.getenv("deepseek_api")
    if not key:
        raise RuntimeError("未设置环境变量 deepseek_api（或 .env 未加载）")
    return OpenAI(api_key=key, base_url="https://api.deepseek.com")


def generate(question: str, context: str) -> str:
    client = _get_client()
    messages = [
        {"role": "system", "content": EVAL_PROMPT},
        {"role": "user", "content": f"问题：{question}\n\n参考上下文：\n{context}"},
    ]
    last = ""
    for attempt in range(MAX_GEN_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=GENERATION_MODEL, messages=messages,
                temperature=GENERATION_TEMPERATURE, max_tokens=512,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last = str(e)
            if attempt < MAX_GEN_RETRIES:
                time.sleep(2)
    raise RuntimeError(f"生成失败: {last}")


def llm_judge(question: str, gold_answer: str, pred_answer: str, context: str) -> Tuple[float, float, str]:
    """LLM-as-judge：对生成答案做 正确性+忠实性 打分（1-5）。失败返回 (0,0,错误)。"""
    if not pred_answer:
        return 0.0, 0.0, ""
    client = _get_client()
    user_content = (f"问题：{question}\n\n标准答案：{gold_answer}\n\n"
                    f"模型答案：{pred_answer}\n\n参考上下文：\n{context}")
    messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": user_content},
    ]
    try:
        resp = client.chat.completions.create(
            model=GENERATION_MODEL, messages=messages,
            temperature=0.0, max_tokens=200,
        )
        raw = (resp.choices[0].message.content or "").strip()
        m = re.search(r"\{[^{}]*\}", raw, re.S)
        if m:
            d = json.loads(m.group(0))
            corr = float(d.get("correctness", 0))
            faith = float(d.get("faithfulness", 0))
            reason = str(d.get("reason", ""))[:120]
            return corr, faith, reason
        return 0.0, 0.0, f"judge解析失败: {raw[:80]}"
    except Exception as e:
        return 0.0, 0.0, f"judge失败: {str(e)[:80]}"


# ====================================================================
# 七、主评测流程
# ====================================================================
_CSV_HEADER = ["qid", "question", "chunk_method", "recall@5", "mrr", "ndcg@5",
               "recall@5_c", "mrr_c", "ndcg@5_c",
               "em", "f1", "gold_answer", "pred_answer", "gold_sources",
               "judge_corr", "judge_faith", "judge_reason", "error"]


def build_index(docs: Dict[str, str], method: str):
    """对某分块方法：全文分块 -> 建 BM25+向量双索引（独立 vector_db 目录）。"""
    from rag_core.retriever import HybridRetriever
    chunks, metadatas = [], []
    for i, (src, text) in enumerate(docs.items(), 1):
        t0 = time.time()
        print(f"  [{i}/{len(docs)}] {method}: 分块 {src[:40]} ...", flush=True)
        parts = CHUNKERS[method](text)
        print(f"       → {len(parts)} 块（{time.time() - t0:.1f}s）", flush=True)
        for ci, c in enumerate(parts):
            chunks.append(c)
            metadatas.append({"source": src, "chunk_idx": ci, "chunk_method": method})
    if not chunks:
        raise RuntimeError(f"{method}: 分块结果为空")
    vdb = os.path.join(VECTOR_DB_ROOT, method)
    os.makedirs(vdb, exist_ok=True)
    rtr = HybridRetriever(vector_db_path=vdb)
    n = rtr.index(chunks, metadatas)
    print(f"  [INDEX] {method}: {n} 块")
    return rtr


def run_method(method: str, qbank: List[Dict], docs: Dict[str, str], skip_gen: bool = False) -> str:
    if not qbank:
        print(f"  [SKIP] {method}: 该数据源无样本")
        return ""
    out_path = os.path.join(OUTPUT_DIR, f"{method}_{qbank[0]['source']}.csv")
    rtr = build_index(docs, method)
    rows = []
    for qi, q in enumerate(qbank):
        try:
            results = rtr.retrieve(q["question"], bm25_k=BM25_K, vector_k=VECTOR_K,
                                   rerank_k=RERANK_K, keywords=None)
            recall, mrr, ndcg = retrieval_metrics(results, q["gold_sources"], k=RERANK_K)
            recall_c, mrr_c, ndcg_c = retrieval_metrics_chunk(results, q.get("gold_texts") or [], k=RERANK_K)
            em, f1, pred = 0.0, 0.0, ""
            jc, jf, jr = 0.0, 0.0, ""
            if not skip_gen:
                ctx = make_context(results)
                pred = generate(q["question"], ctx)
                em, f1 = em_f1(pred, q["answer"])
                jc, jf, jr = llm_judge(q["question"], q["answer"], pred, ctx)
            rows.append({
                "qid": q["qid"], "question": q["question"], "chunk_method": method,
                "recall@5": round(recall, 4), "mrr": round(mrr, 4), "ndcg@5": round(ndcg, 4),
                "recall@5_c": round(recall_c, 4), "mrr_c": round(mrr_c, 4), "ndcg@5_c": round(ndcg_c, 4),
                "em": em, "f1": round(f1, 4), "gold_answer": q["answer"],
                "pred_answer": pred, "gold_sources": "|".join(q["gold_sources"]),
                "judge_corr": jc, "judge_faith": jf, "judge_reason": jr, "error": "",
            })
            print(f"    [{qi+1}/{len(qbank)}] recall@5={recall:.2f} recall@5_c={recall_c:.2f} mrr={mrr:.2f} em={em} f1={f1:.2f}  {q['qid']}")
        except Exception as e:
            rows.append({
                "qid": q["qid"], "question": q["question"], "chunk_method": method,
                "recall@5": "", "mrr": "", "ndcg@5": "", "recall@5_c": "", "mrr_c": "", "ndcg@5_c": "",
                "em": "", "f1": "",
                "gold_answer": q["answer"], "pred_answer": "", "gold_sources": "|".join(q["gold_sources"]),
                "error": str(e)[:200],
            })
            print(f"    [ERR] {q['qid']}: {e}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [DONE] {method} -> {out_path}（{len(rows)} 条）")
    return out_path


# ====================================================================
# 八、配对显著性检验（逐条 CSV 是前提）
# ====================================================================
def load_csv_rows(path: str) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def paired_ttest(method_a: str, method_b: str, source: str):
    """对两种分块方法在同一批问题上的逐条指标做配对检验。"""
    try:
        from scipy import stats
    except ImportError:
        print("  需要 scipy：pip install scipy")
        return
    pa = os.path.join(OUTPUT_DIR, f"{method_a}_{source}.csv")
    pb = os.path.join(OUTPUT_DIR, f"{method_b}_{source}.csv")
    if not (os.path.exists(pa) and os.path.exists(pb)):
        print(f"  缺少结果文件: {pa} 或 {pb}（先跑评测）")
        return
    ra, rb = load_csv_rows(pa), load_csv_rows(pb)
    if len(ra) != len(rb):
        print("  两方法样本数不一致，无法配对")
        return
    for metric in ["recall@5", "mrr", "ndcg@5", "recall@5_c", "mrr_c", "ndcg@5_c", "f1", "judge_corr", "judge_faith"]:
        va = [float(r.get(metric) or 0) for r in ra]
        vb = [float(r.get(metric) or 0) for r in rb]
        try:
            ma, mb = sum(va) / len(va), sum(vb) / len(vb)
            diff_nonzero = sum(1 for x, y in zip(va, vb) if x != y)
            if diff_nonzero == 0:
                print(f"  {metric}: {method_a} mean={ma:.4f} vs {method_b} mean={mb:.4f}  identical, test not applicable")
                continue
            t, p = stats.ttest_rel(va, vb)
            w, wp = stats.wilcoxon(va, vb)
            print(f"  {metric}: {method_a} mean={ma:.4f} vs {method_b} mean={mb:.4f}  "
                  f"t={t:.3f} p={p:.4f}  Wilcoxon p={wp:.4f}")
        except Exception as e:
            print(f"  {metric}: 检验失败 {e}")


def _parse_ttest_pairs(pairs_arg, methods_arg):
    """解析检验对：--pairs 'A,B C,D' 指定；省略则对 --methods 全部两两组合。"""
    pairs = []
    if pairs_arg:
        for p in pairs_arg:
            parts = [x.strip() for x in p.split(",") if x.strip()]
            if len(parts) != 2:
                print(f"  [SKIP] 无法解析方法对: {p!r}（应为 A,B）")
                continue
            pairs.append((parts[0], parts[1]))
    else:
        ms = _available_methods(methods_arg)
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                pairs.append((ms[i], ms[j]))
    return pairs


def run_ttest_only(pairs, sources):
    """不做评测，直接对 results/ 下已有 CSV 做配对检验。"""
    for src in sources:
        for a, b in pairs:
            print(f"\n[{src}] {a} vs {b}")
            paired_ttest(a, b, src)


def main():
    parser = argparse.ArgumentParser(description="RAG 分块方法消融评测")
    parser.add_argument("--methods", default="fixed,discourse,hybrid", help="逗号分隔；all=全部")
    parser.add_argument("--source", default="finance", choices=["finance", "hotpotqa", "both"])
    parser.add_argument("--limit", type=int, default=0, help="每个数据源最多评测前 N 题（0=全部，测试用）")
    parser.add_argument("--skip-gen", action="store_true", help="只算检索指标，不调生成 API")
    parser.add_argument("--docs-dir", default=DOCS_DIR)
    parser.add_argument("--ttest", action="store_true", help="评测后对前两种方法做配对检验")
    parser.add_argument("--ttest-only", action="store_true", help="不做评测，直接对已有 CSV 做配对检验（秒级）")
    parser.add_argument("--pairs", nargs="+", default=None, metavar="A,B",
                        help="指定检验对，如 'hmm,fixed hmm,hybrid'；省略则对 --methods 全部两两组合")
    parser.add_argument("--docs-cache", default=None, metavar="JSON",
                        help="直接用预生成文档缓存（如 MinerU 的 data/docs_cache_v2.json），跳过目录加载与清单校验")
    args = parser.parse_args()

    methods = _available_methods(args.methods)
    if any(m.startswith("hmm") for m in methods):
        try:
            import hmmlearn  # noqa: F401
            import sklearn  # noqa: F401
        except ImportError:
            print("  [WARN] hmmlearn/sklearn 未安装，跳过 hmm 系列方法（pip install hmmlearn scikit-learn）")
            methods = [m for m in methods if not m.startswith("hmm")]
    if not methods:
        print("没有可用的分块方法")
        sys.exit(1)

    sources = ["finance", "hotpotqa"] if args.source == "both" else [args.source]

    if args.ttest_only:
        # 只做检验：不加载文档、不重跑评测，直接读 results/ 下已有 CSV
        print("=" * 60)
        print("仅配对显著性检验（读取已有 CSV，不重跑评测）")
        print("=" * 60)
        pairs = _parse_ttest_pairs(args.pairs, args.methods)
        if not pairs:
            print("没有可检验的方法对（--pairs 或 --methods 配置有误）")
            sys.exit(1)
        run_ttest_only(pairs, sources)
        return

    print("=" * 60)
    print(f"评测方法: {methods}  文档目录: {args.docs_dir}")
    print("=" * 60)

    if args.docs_cache:
        # 预生成缓存（MinerU 提取基座）：直接读 JSON，跳过目录扫描/OCR/清单校验
        print(f"使用预生成文档缓存: {args.docs_cache}")
        try:
            with open(args.docs_cache, "r", encoding="utf-8") as f:
                docs = json.load(f)
        except Exception as e:
            print(f"  [ERR] 缓存读取失败: {e}")
            sys.exit(1)
        if not docs:
            print("  [ERR] 缓存为空")
            sys.exit(1)
        print(f"  共 {len(docs)} 篇文档")
    else:
        print("加载文档（一次，所有方法共用原文）...")
        docs = load_docs(args.docs_dir)
        if not docs:
            sys.exit(1)

    for src in sources:
        qbank = load_finance() if src == "finance" else load_hotpotqa()
        if args.limit > 0:
            qbank = qbank[:args.limit]
        print(f"\n数据源: {src}  （{len(qbank)} 题）")
        if not qbank:
            print("  无样本，跳过")
            continue
        for m in methods:
            try:
                run_method(m, qbank, docs, skip_gen=args.skip_gen)
            except NotImplementedError as e:
                print(f"  [SKIP] {m}: {e}")
            except Exception as e:
                print(f"  [FAIL] {m}: {e}")

    if args.ttest and len(methods) >= 2:
        print("\n" + "=" * 60)
        print("配对显著性检验")
        print("=" * 60)
        for src in sources:
            print(f"\n[{src}] {methods[0]} vs {methods[1]}")
            paired_ttest(methods[0], methods[1], src)
    print("\n全部完成。结果在 results/ 目录，逐条 CSV 可用于后续配对 t 检验。")


if __name__ == "__main__":
    main()
