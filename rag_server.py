# -*- coding: utf-8 -*-
"""
rag_server.py —— 一体化本地 RAG 服务（HTTP API + MCP 双协议，单进程）

一个进程同时提供：
  - HTTP API（脚本 / 桌面面板调用）：GET /health /stats；POST /build /search /ask /open /ingest
                              POST /direction/analyze /direction/compare
  - MCP 端点（DeepSeek Harness 注册工具）：/mcp
    （工具：search / stats / build / ingest / ask / open_doc / direction_analyze / direction_compare，
     DSH 中为 mcp__rag__*）

单进程设计保证 qdrant 本地存储锁唯一持有者——不要再同时启动其他会加载
检索器的进程（原 8001 的 rag_mcp_server.py 已退役，见该文件说明）。

启动：python -m uvicorn rag_server:app --host 127.0.0.1 --port 8000
      （或双击 启动RAG服务.bat）

接口：
  HTTP  GET  /health                         环境自检
  HTTP  GET  /stats                          知识库统计（含运行统计 obs）
  HTTP  POST /build   {chunker, clear}       重建知识库（MinerU 输出 + docx 文档）
  HTTP  POST /search  {query, top_k}         纯检索（返回块 + 来源，不生成）
  HTTP  POST /ask     {question, top_k, deep} 检索 + 生成（带引用）
  HTTP  POST /open    {doc, page}            打开原始 PDF 指定页
  HTTP  POST /ingest  {}                     增量入库（只处理新增文档）
  MCP   /mcp                                 同一批功能的 MCP 工具（供 DSH 注册）
"""

import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastmcp import FastMCP
from pydantic import BaseModel

# ---- 路径（集中配置见 rag_core/config.py，可用环境变量 / .env 覆盖）----
from rag_core import config  # noqa: E402
KB_FILE = config.KB_FILE
VECTOR_DB_PATH = config.VECTOR_DB_PATH
MINERU_OUT = config.MINERU_OUT
DOCS_DIR = config.DOCS_DIR
# 原始 PDF/docx 搜索目录（按文件名定位；环境变量 PDF_SOURCE_DIRS 用 ; 分隔追加）
PDF_SOURCE_DIRS = [d for d in (
    [x.strip() for x in os.getenv("PDF_SOURCE_DIRS", "").split(";") if x.strip()]
    + [DOCS_DIR, os.path.join(os.path.dirname(MINERU_OUT), "input")]
) if d]
# 引用链接基地址（DSH 前端只渲染 http/https 链接，自定义协议会被丢弃；
# 环境变量 RAG_LINK_BASE 可在 DSH 端口变化时覆盖）
RAG_LINK_BASE = os.getenv("RAG_LINK_BASE", "http://127.0.0.1:3080/dsh-rag/open")
# 综述编辑器页面地址（浏览器新标签页；环境变量 RAG_EDITOR_BASE 可在端口变化时覆盖）
EDITOR_BASE = os.getenv("RAG_EDITOR_BASE", "http://127.0.0.1:8000/editor")
EDITOR_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "editor.html")

# ---- 增量入库状态（侧车文件：记录每篇输入的哈希/模式，用于三分类）----
INGEST_STATE_FILE = KB_FILE + ".ingest.json"


def _file_hash(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_ingest_state() -> dict:
    try:
        with open(INGEST_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("docs"), dict):
            return data
    except Exception:
        pass
    return {"mode": None, "docs": {}}


def _save_ingest_state(state: dict):
    os.makedirs(os.path.dirname(INGEST_STATE_FILE), exist_ok=True)
    tmp = INGEST_STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, INGEST_STATE_FILE)

app = FastAPI(title="rag-server", version="1.0")

# ---- 请求模型 ----
class BuildReq(BaseModel):
    chunker: str = "hmm"     # fixed / discourse / hybrid / hmm
    clear: bool = False
    mineru_out: str = MINERU_OUT
    docs_dir: str = DOCS_DIR


class SearchReq(BaseModel):
    query: str
    top_k: int = 5
    filters: dict = None   # 元数据过滤：{year_min,year_max,authors,methods,tasks}


class AskReq(BaseModel):
    question: str
    top_k: int = 5
    deep: bool = False
    filters: dict = None


class OpenReq(BaseModel):
    doc: str
    page: int = 1


class IngestReq(BaseModel):
    mineru_out: str = MINERU_OUT
    docs_dir: str = DOCS_DIR


class DirectionReq(BaseModel):
    direction: str
    top_k: int = 12


class DirectionsReq(BaseModel):
    directions: list
    weights: dict = None
    top_k: int = 12


class SurveyOutlineReq(BaseModel):
    topic: str
    constraints: str = ""
    outline: list = None


class SurveyDraftReq(BaseModel):
    topic: str
    outline: list = None
    force: bool = False


class SurveySectionReq(BaseModel):
    topic: str
    section: str


class SurveyRewriteReq(BaseModel):
    topic: str
    section: str
    instruction: str = "更学术化、更深入"


class SurveyEditReq(BaseModel):
    topic: str
    section: str
    text: str


class SurveyExportReq(BaseModel):
    topic: str
    format: str = "markdown"


class RewriteSelectionReq(BaseModel):
    topic: str
    selected_text: str
    instruction: str = ""          # 重写指令（默认"改写得更学术化"）
    evidence: bool = False         # True = v2 带知识库证据
    context_scope: str = "section"  # section=本节上下文；full=全文上下文


class EditorSaveReq(BaseModel):
    topic: str
    text: str
    filename: str = ""             # 空=覆盖默认导出文件；非空=另存为（白名单校验）


class EditorDocxReq(BaseModel):
    topic: str
    text: str = None               # 编辑器当前文本；缺省加载已导出 Markdown
    citation_format: str = "author_year"   # author_year=著者-年份；superscript=上标编号
    include_refs: bool = False     # 是否附生成的参考文献列表（默认不附，正式引用走知网）
    fmt_options: dict = None       # 排版覆盖：正文/标题字体字号、段落、页边距、页眉页码


# ---- 全局状态（懒加载：health 秒回，不碰 GPU）----
_state = {"retriever": None, "kb_fp": None}


def _fingerprint(texts, metadatas):
    h = hashlib.md5()
    for t, m in zip(texts, metadatas):
        h.update((m.get("source", "") + "\n").encode("utf-8", "ignore"))
        h.update(t.encode("utf-8", "ignore"))
    return h.hexdigest()


def get_retriever():
    """懒加载知识库与索引；内容未变化时复用向量索引（指纹校验）。"""
    from rag_core.knowledge_base import KnowledgeBase
    from rag_core.retriever import HybridRetriever

    kb = KnowledgeBase(KB_FILE)
    texts, metadatas = kb.to_texts_metadatas()
    if not texts:
        raise RuntimeError(f"知识库为空: {KB_FILE}（请先调用 build）")

    fp = _fingerprint(texts, metadatas)
    if _state["retriever"] is not None and _state["kb_fp"] == fp:
        return _state["retriever"]

    # 知识库变化：先关闭旧检索器（释放 qdrant 本地存储文件锁），否则新建实例会锁冲突
    if _state["retriever"] is not None:
        _state["retriever"].close()

    retriever = HybridRetriever(vector_db_path=VECTOR_DB_PATH)
    retriever.chunks = texts
    retriever.metadatas = metadatas
    retriever._build_bm25()
    # 向量索引：存在且点数一致才复用，否则重建（重建会自动清空旧存储段）
    vector_ok = False
    try:
        from qdrant_client import QdrantClient
        if os.path.exists(VECTOR_DB_PATH):
            c = QdrantClient(path=VECTOR_DB_PATH)
            try:
                names = [x.name for x in c.get_collections().collections]
                if retriever._collection_name in names:
                    vector_ok = c.count(retriever._collection_name, exact=True).count == len(texts)
            finally:
                c.close()  # 临时检查客户端必须显式关闭，否则锁不释放
    except Exception:
        vector_ok = False
    if not vector_ok:
        retriever.index(texts, metadatas)
    _state["retriever"] = retriever
    _state["kb_fp"] = fp
    return retriever


def _page_label(meta: dict) -> str:
    ps, pe = meta.get("page_start"), meta.get("page_end")
    if ps is None:
        return ""
    if pe and pe != ps:
        return f"，第{ps}-{pe}页"
    return f"，第{ps}页"


def _neighbor_texts(idx, source, ps, pe):
    """取同文档相邻块（±1）作为上下文扩展素材；页码间距 >1 视为跨章节跳过。
    返回 [(文本, 页码标注), ...]。"""
    retriever = _state.get("retriever")
    if retriever is None or idx is None:
        return []
    out = []
    for d in (-1, 1):
        j = idx + d
        if not (0 <= j < len(retriever.chunks)):
            continue
        m2 = retriever.metadatas[j] if j < len(retriever.metadatas) else {}
        if m2.get("source") != source:
            continue
        p2s, p2e = m2.get("page_start"), m2.get("page_end")
        if ps is not None and p2s is not None:
            hi = pe if pe is not None else ps
            lo = p2e if p2e is not None else p2s
            if p2s > hi + 1 or lo < ps - 1:
                continue
        t2 = (retriever.chunks[j] or "").strip()
        if t2:
            out.append((t2, _page_label(m2)))
    return out


def _build_context(results, max_chars: int = 4500):
    seen, items = set(), []
    for r in results:
        meta = r.get("metadata") or {}
        source = meta.get("source", "未知来源")
        text = (r.get("text") or "").strip()
        if not text or (source, text) in seen:
            continue
        seen.add((source, text))
        items.append((source, text, _page_label(meta),
                      meta.get("page_start"), meta.get("page_end"),
                      r.get("index")))
    # 第一遍：命中块优先占预算
    parts, total, used = [], 0, []
    for i, (source, text, pages, ps, pe, idx) in enumerate(items, start=1):
        label = f"[来源{i}]"
        block = f"{label}（来源: {source}{pages}）\n{text}"
        if parts and total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
        used.append((label, source, pages, ps, pe, idx))
    # 第二遍：同篇相邻段落补充（解决表格/实证段落跨块；不占引用编号）
    for label, source, pages, ps, pe, idx in used:
        for nt, nlabel in _neighbor_texts(idx, source, ps, pe):
            if (source, nt) in seen:
                continue
            seen.add((source, nt))
            nb = f"（同篇相邻段落{nlabel}）\n{nt}"
            if total + len(nb) > max_chars:
                break
            parts.append(nb)
            total += len(nb)
    citations, seen_src = [], set()
    for label, source, pages, ps, pe, idx in used:
        if source not in seen_src:
            seen_src.add(source)
            citations.append(f"{label} {source}{pages}")
    return "\n\n".join(parts), citations, used


def _linkify_answer(answer: str, used) -> str:
    """把回答里的 [来源N] 替换为指向 DSH /dsh-rag/open 的 http 链接
    （点击由 dsh-rag-citation 插件拦截并打开 PDF 对应页）。
    链接文字必须是干净的 [来源N]——旧实现在 label 外层又套了一层方括号，
    生成 [[来源N]](url)，部分渲染器会解析失败导致只有第一个链接可点。
    另归一化模型可能输出的多余括号写法（[[来源N]] / 【来源N】）。"""
    import urllib.parse
    answer = re.sub(r"\[\[(来源\s*\d+)\]\]", r"[\1]", answer)
    answer = re.sub(r"【(来源\s*\d+)】", r"[\1]", answer)
    for label, source, _pages, ps, _pe, _idx in used:
        page = ps if ps is not None else 1
        doc_q = urllib.parse.quote(source, safe="")
        answer = answer.replace(label, f"{label}({RAG_LINK_BASE}?doc={doc_q}&page={page})")
    return answer


# ================= 核心函数（HTTP 与 MCP 共用） =================

def stats_kb() -> dict:
    from rag_core.knowledge_base import KnowledgeBase
    kb = KnowledgeBase(KB_FILE)
    chunks = kb.load()
    by_source = {}
    for c in chunks:
        by_source[c.get("source", "未知")] = by_source.get(c.get("source", "未知"), 0) + 1
    from rag_core.observability import summarize
    return {"total_chunks": len(chunks), "by_source": by_source, "obs": summarize()}


def search_kb(query: str, top_k: int = 5, filters: dict = None) -> dict:
    """纯检索：返回块与来源，不生成。filters 为元数据过滤（见 retriever.match_meta_filter）。"""
    from rag_core.observability import Timer, log_event
    from rag_core.retriever import normalize_filters
    t_all = Timer()
    filters = normalize_filters(filters)
    try:
        retriever = get_retriever()
    except RuntimeError as e:
        log_event("search", ok=False, total_ms=round(t_all.ms(), 1), error=str(e)[:120])
        return {"ok": False, "error": str(e)}
    results = retriever.retrieve(query, bm25_k=20, vector_k=20, rerank_k=top_k, filters=filters)
    rt = getattr(retriever, "last_timing", {}) or {}
    log_event(
        "search", ok=True,
        total_ms=round(t_all.ms(), 1),
        retrieve_ms=round(t_all.ms(), 1),
        bm25_ms=rt.get("bm25_ms"), vector_ms=rt.get("vector_ms"),
        rerank_ms=rt.get("rerank_ms"),
        hits=len(results),
        filters=bool(filters),
    )
    return {"ok": True, "results": results, "filters": filters}


def build_kb(chunker: str = "hmm", clear: bool = False,
             mineru_out: str = MINERU_OUT, docs_dir: str = DOCS_DIR) -> dict:
    """MinerU 输出 + docx 文档 → 分块 → 知识库 + 索引。"""
    from rag_core.knowledge_base import KnowledgeBase
    from rag_core.chunk_splitter import dispatch_chunk
    from rag_core.mineru_loader import find_mineru_outputs, load_mineru_doc
    from rag_core.document_loader import _extract_docx_content

    mode = chunker.strip().lower()
    if mode not in ("fixed", "discourse", "hybrid", "hmm"):
        return {"ok": False, "error": f"未知分块模式 {chunker}"}

    from rag_core.observability import Timer, log_event
    t_all = Timer()

    if clear and os.path.exists(KB_FILE):
        os.remove(KB_FILE)

    docs = {}
    doc_hashes = {}
    if os.path.isdir(mineru_out):
        for doc, path in find_mineru_outputs(mineru_out).items():
            try:
                t = load_mineru_doc(path)
                if t.strip():
                    docs[doc + ".pdf"] = t
                    doc_hashes[doc + ".pdf"] = _file_hash(path)
            except Exception as e:
                print(f"  [ERR] {doc}: {e}")
    if os.path.isdir(docs_dir):
        for f in sorted(os.listdir(docs_dir)):
            if f.lower().endswith(".docx"):
                try:
                    t = _extract_docx_content(os.path.join(docs_dir, f))
                    if t.strip():
                        docs[f] = t
                        doc_hashes[f] = _file_hash(os.path.join(docs_dir, f))
                except Exception as e:
                    print(f"  [ERR] {f}: {e}")
    if not docs:
        log_event("build", ok=False, mode=mode, total_ms=round(t_all.ms(), 1),
                  error="未找到文档")
        return {"ok": False, "error": f"未找到文档（mineru_out={mineru_out}, docs_dir={docs_dir}）"}

    kb = KnowledgeBase(KB_FILE)
    total_chunks = 0
    per_doc_chunks = {}
    doc_names = sorted(docs)
    print(f"全量重建开始：模式={mode}，共 {len(doc_names)} 篇（clear={clear}）", flush=True)
    for di, name in enumerate(doc_names, start=1):
        print(f"  [{di}/{len(doc_names)}] {name} …", flush=True)
        if mode == "hmm":
            base = os.path.dirname(os.path.abspath(__file__))
            from rag_core.hmm_chunker import hmm_chunk
            parts = hmm_chunk(
                docs[name], chunk_size=800, overlap_tokens=50,
                cache_dir=os.path.join(base, "data", "hmm_embed_cache"),
                chunk_cache_dir=os.path.join(base, "data", "hmm_chunk_cache"),
                bic_coef=2.0,  # 与评测验证配置一致——保证命中已有块缓存，重建秒级
                obs_doc=name,
            )
        else:
            parts = dispatch_chunk(docs[name], mode, 800, 50)
        # 溯源元数据：原始文件路径 + 每块页码归属（对齐失败时页码为 None）
        from rag_core.pdf_open import resolve_pdf_path
        pdf_path = resolve_pdf_path(name, PDF_SOURCE_DIRS)
        page_ranges = None
        if name.lower().endswith(".pdf"):
            from rag_core.chunk_splitter import attribute_pages
            page_ranges = attribute_pages(docs[name], parts)
        kb.add_chunks(parts, source=name, source_type=mode,
                      pdf_path=pdf_path, page_ranges=page_ranges)
        total_chunks += len(parts)
        per_doc_chunks[name] = len(parts)
        print(f"      完成：{len(parts)} 块", flush=True)
    kb.save()
    print(f"知识库已保存：{len(doc_names)} 篇 / {total_chunks} 块；开始重建向量索引…", flush=True)
    # 同步增量状态：之后 ingest 按哈希三分类
    _save_ingest_state({"mode": mode, "docs": {
        n: {"hash": doc_hashes.get(n), "chunks": per_doc_chunks.get(n, 0), "mode": mode}
        for n in docs
    }})

    # 重建索引：先关闭旧检索器释放 qdrant 锁，再让 get_retriever 重建
    if _state["retriever"] is not None:
        _state["retriever"].close()
    _state["retriever"] = None
    _state["kb_fp"] = None
    get_retriever()
    log_event("build", ok=True, mode=mode, clear=clear, docs=len(docs),
              chunks=total_chunks, total_ms=round(t_all.ms(), 1))
    return {"ok": True, "chunker": mode, "docs": len(docs), "chunks": total_chunks}


def _classify_inputs(state: dict, hashes: dict, mode: str):
    """按状态侧车三分类输入：返回 (new, changed, unchanged, removed)。"""
    new, changed, unchanged = [], [], []
    for name in sorted(hashes):
        st = state["docs"].get(name)
        if st and st.get("hash") == hashes[name] and st.get("mode") == mode:
            unchanged.append(name)
        elif st:
            changed.append(name)
        else:
            new.append(name)
    removed = [d for d in state["docs"] if d not in hashes]
    return new, changed, unchanged, removed


def ingest_kb(mineru_out: str = MINERU_OUT, docs_dir: str = DOCS_DIR) -> dict:
    """增量入库（方案 A）：
    - 只处理**新增**文档：分块 + 页码归属 + 追加块 + 只嵌入新增块（已有向量不动）；
    - 检测到删除/变更（文件哈希与状态侧车不一致）→ 自动全量重建；
    - 无变化 → 秒级空跑并同步状态。
    返回报告 {ok, added, added_chunks, unchanged, mode, msg}。"""
    from rag_core.knowledge_base import KnowledgeBase, _chunk_entries
    from rag_core.mineru_loader import find_mineru_outputs, load_mineru_doc
    from rag_core.document_loader import _extract_docx_content
    from rag_core.pdf_open import resolve_pdf_path

    # 1) 扫描输入（只取路径，不加载全文）
    inputs = {}
    if os.path.isdir(mineru_out):
        for doc, path in find_mineru_outputs(mineru_out).items():
            inputs[doc + ".pdf"] = ("mineru", path)
    if os.path.isdir(docs_dir):
        for f in sorted(os.listdir(docs_dir)):
            if f.lower().endswith(".docx"):
                inputs[f] = ("docx", os.path.join(docs_dir, f))
    if not inputs:
        return {"ok": False, "error": f"未找到输入文档（mineru_out={mineru_out}, docs_dir={docs_dir}）"}

    state = _load_ingest_state()
    mode = state.get("mode") or "hmm"

    # 2) 三分类（文件哈希比对，秒级）
    hashes = {name: _file_hash(info[1]) for name, info in inputs.items()}
    new_docs, changed_docs, unchanged, removed = _classify_inputs(state, hashes, mode)

    # 3) 删除/变更 → 全量重建（方案 A 规则：不动点 id，避免空洞）
    if removed or changed_docs:
        r = build_kb(mode, False, mineru_out, docs_dir)
        if r.get("ok"):
            r["note"] = f"检测到删除 {len(removed)} 篇 / 变更 {len(changed_docs)} 篇，已自动全量重建"
        return r

    # 4) 与现有知识库去重：首次启用状态文件时（state 为空但库里有货），
    #    只补状态、不重复入库，防止索引与 KB 双重追加
    kb = KnowledgeBase(KB_FILE)
    try:
        existing_sources = {c.get("source") for c in kb.load()}
    except Exception:
        existing_sources = set()
    truly_new = [n for n in new_docs if n not in existing_sources]
    unchanged.extend(n for n in new_docs if n in existing_sources)

    if not truly_new:
        _save_ingest_state({"mode": mode, "docs": {
            n: {"hash": hashes[n], "mode": mode} for n in inputs
        }})
        return {"ok": True, "added": 0, "added_chunks": 0,
                "unchanged": len(unchanged), "mode": mode, "msg": "无新文档（状态已同步）"}

    # 5) 纯新增 → 增量追加
    try:
        retriever = get_retriever()
    except RuntimeError:
        return build_kb(mode, False, mineru_out, docs_dir)  # 知识库为空 → 全量构建

    from rag_core.chunk_splitter import attribute_pages
    from rag_core.hmm_chunker import hmm_chunk

    new_state = {"mode": mode, "docs": {
        n: {"hash": hashes[n], "mode": mode} for n in inputs
    }}
    added_docs, added_chunks = 0, 0
    for name in sorted(truly_new):
        kind, path = inputs[name]
        try:
            raw = load_mineru_doc(path) if kind == "mineru" else _extract_docx_content(path)
        except Exception as e:
            print(f"  [ERR] {name}: {e}")
            continue
        if not raw or not raw.strip():
            continue
        if mode == "hmm":
            base = os.path.dirname(os.path.abspath(__file__))
            parts = hmm_chunk(
                raw, chunk_size=800, overlap_tokens=50,
                cache_dir=os.path.join(base, "data", "hmm_embed_cache"),
                chunk_cache_dir=os.path.join(base, "data", "hmm_chunk_cache"),
                bic_coef=2.0,  # 与全量构建一致——保证命中已有块缓存
                obs_doc=name,
            )
        else:
            from rag_core.chunk_splitter import dispatch_chunk
            parts = dispatch_chunk(raw, mode, 800, 50)
        pdf_path = resolve_pdf_path(name, PDF_SOURCE_DIRS)
        page_ranges = attribute_pages(raw, parts) if name.lower().endswith(".pdf") else None
        entries = _chunk_entries(parts, name, mode, pdf_path, page_ranges)
        if not entries:
            continue
        kb.add_entries(entries)
        retriever.add_chunks(
            [e["text"] for e in entries],
            [{k: v for k, v in e.items() if k != "text"} for e in entries],
        )
        new_state["docs"][name]["chunks"] = len(entries)
        added_docs += 1
        added_chunks += len(entries)
        print(f"  [INGEST] {name}: +{len(entries)} 块")

    kb.save()
    _save_ingest_state(new_state)
    _state["kb_fp"] = _fingerprint(retriever.chunks, retriever.metadatas)
    from rag_core.observability import log_event
    log_event("ingest", ok=True, added=added_docs, added_chunks=added_chunks,
              unchanged=len(unchanged), mode=mode)
    return {"ok": True, "added": added_docs, "added_chunks": added_chunks,
            "unchanged": len(unchanged), "mode": mode,
            "msg": f"新增 {added_docs} 篇 / {added_chunks} 块，已入库并增量索引"}


def ask_kb(question: str, top_k: int = 5, filters: dict = None) -> dict:
    """检索 + 生成（DeepSeek），返回 {answer, citations}。filters 为元数据过滤。"""
    from rag_core.observability import Timer, log_event
    from rag_core.retriever import normalize_filters
    t_all = Timer()
    filters = normalize_filters(filters)
    if not os.getenv("deepseek_api"):
        log_event("ask", ok=False, total_ms=round(t_all.ms(), 1), error="未配置 deepseek_api")
        return {"ok": False, "error": "未配置环境变量 deepseek_api"}
    try:
        retriever = get_retriever()
    except RuntimeError as e:
        log_event("ask", ok=False, total_ms=round(t_all.ms(), 1), error=str(e)[:120])
        return {"ok": False, "error": str(e)}

    from openai import OpenAI
    from rag_core.query_processor import QueryProcessor

    t_qp = Timer()
    qp = QueryProcessor()
    qr = qp.process(question)
    qp_ms = t_qp.ms()

    t_ret = Timer()
    results = retriever.retrieve(
        qr["expanded_query"], bm25_k=20, vector_k=20, rerank_k=top_k,
        keywords=qr["keywords"], filters=filters,
    )
    retrieve_ms = t_ret.ms()
    rt = getattr(retriever, "last_timing", {}) or {}
    if not results:
        log_event("ask", ok=True, total_ms=round(t_all.ms(), 1), qp_ms=round(qp_ms, 1),
                  retrieve_ms=round(retrieve_ms, 1), bm25_ms=rt.get("bm25_ms"),
                  vector_ms=rt.get("vector_ms"), rerank_ms=rt.get("rerank_ms"),
                  generate_ms=0, tokens_in=0, tokens_out=0, hits=0)
        return {"ok": True, "answer": "未检索到相关内容。", "citations": []}

    context, citations, used = _build_context(results)
    client = OpenAI(api_key=os.getenv("deepseek_api"), base_url="https://api.deepseek.com")
    system_prompt = (
        "你是一位基于文档知识库的问答助手。回答必须严格基于\"参考上下文\"，不编造信息。"
        "引用规范：每写完一个使用了上下文中某块内容的段落，就在该段落末尾标注对应来源编号，"
        "形如 [来源1]（只写单层方括号；编号必须与上下文块的编号一致；一个段落引用多块则连续列出，如 [来源1][来源2]）。"
        "不要在文末单列来源清单。使用中文，精准简洁，结构清晰。"
    )
    user_prompt = f"### 参考上下文\n{context}\n\n### 用户问题\n{question}\n"
    tokens_in = tokens_out = 0
    t_gen = Timer()
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
            timeout=60,
        )
        answer = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        if usage:
            tokens_in = int(getattr(usage, "prompt_tokens", 0) or 0)
            tokens_out = int(getattr(usage, "completion_tokens", 0) or 0)
    except Exception as e:
        answer = f"调用 DeepSeek API 出错: {e}"
    generate_ms = t_gen.ms()
    log_event(
        "ask", ok=True, total_ms=round(t_all.ms(), 1),
        qp_ms=round(qp_ms, 1), retrieve_ms=round(retrieve_ms, 1),
        bm25_ms=rt.get("bm25_ms"), vector_ms=rt.get("vector_ms"),
        rerank_ms=rt.get("rerank_ms"), generate_ms=round(generate_ms, 1),
        tokens_in=tokens_in, tokens_out=tokens_out, hits=len(results),
    )
    return {"ok": True, "answer": _linkify_answer(answer or "", used), "citations": citations}


def open_doc_kb(doc: str, page: int = 1) -> dict:
    """打开知识库中某文档的原始文件并跳到指定页。"""
    from rag_core.pdf_open import resolve_pdf_path, open_pdf_page

    pdf_path = None
    try:
        from rag_core.knowledge_base import KnowledgeBase
        for c in KnowledgeBase(KB_FILE).load():
            if c.get("source") == doc and c.get("pdf_path"):
                pdf_path = c["pdf_path"]
                break
    except Exception:
        pass
    if not pdf_path:
        pdf_path = resolve_pdf_path(doc, PDF_SOURCE_DIRS)
    if not pdf_path:
        return {"ok": False, "error": f"未找到原始文件: {doc}（已搜索: {PDF_SOURCE_DIRS}）"}
    return open_pdf_page(pdf_path, page)


# ================= HTTP 路由 =================

@app.get("/health")
def health():
    checks = {}
    for mod in ("numpy", "sklearn", "hmmlearn", "tiktoken", "jieba", "rank_bm25",
                "qdrant_client", "sentence_transformers", "transformers", "torch", "openai"):
        try:
            __import__(mod)
            checks[mod] = "ok"
        except Exception as e:
            checks[mod] = f"missing: {e}"
    checks["kb_exists"] = os.path.exists(KB_FILE)
    checks["mineru_out"] = MINERU_OUT
    checks["deepseek_api"] = bool(os.getenv("deepseek_api"))
    return {"status": "ok", "checks": checks}


@app.get("/stats")
def stats():
    return stats_kb()


@app.post("/build")
def build(req: BuildReq):
    return build_kb(req.chunker, req.clear, req.mineru_out, req.docs_dir)


@app.post("/search")
def search(req: SearchReq):
    return search_kb(req.query, req.top_k, req.filters)


@app.post("/ask")
def ask(req: AskReq):
    return ask_kb(req.question, req.top_k, req.filters)


@app.post("/open")
def open_doc(req: OpenReq):
    return open_doc_kb(req.doc, req.page)


@app.post("/ingest")
def ingest(req: IngestReq):
    return ingest_kb(req.mineru_out, req.docs_dir)


@app.post("/direction/analyze")
def direction_analyze(req: DirectionReq):
    from rag_core.research_advisor import analyze_direction
    try:
        return analyze_direction(req.direction, req.top_k)
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/direction/compare")
def direction_compare(req: DirectionsReq):
    from rag_core.research_advisor import compare_directions
    try:
        return compare_directions(req.directions, req.weights, req.top_k)
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ================= 交互式综述（survey）路由 =================

@app.post("/survey/outline")
def survey_outline_http(req: SurveyOutlineReq):
    from rag_core.survey import survey_outline
    try:
        return survey_outline(req.topic, req.constraints, req.outline)
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/survey/draft")
def survey_draft_http(req: SurveyDraftReq):
    from rag_core.survey import survey_draft
    try:
        return survey_draft(req.topic, req.outline, req.force)
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/survey/status")
def survey_status_http(topic: str):
    from rag_core.survey import survey_status
    try:
        return survey_status(topic)
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/survey/section")
def survey_section_http(topic: str, section: str):
    from rag_core.survey import survey_section
    try:
        return survey_section(topic, section)
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/survey/rewrite")
def survey_rewrite_http(req: SurveyRewriteReq):
    from rag_core.survey import survey_rewrite
    try:
        return survey_rewrite(req.topic, req.section, req.instruction)
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/survey/edit")
def survey_edit_http(req: SurveyEditReq):
    from rag_core.survey import survey_edit
    try:
        return survey_edit(req.topic, req.section, req.text)
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/survey/export")
def survey_export_http(req: SurveyExportReq):
    from rag_core.survey import survey_export
    try:
        r = survey_export(req.topic, req.format)
        if r.get("ok"):
            import urllib.parse
            r["editor_url"] = f"{EDITOR_BASE}?topic={urllib.parse.quote(str(req.topic))}"
        return r
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/editor")
def editor_page():
    """综述编辑器页面（浏览器新标签页）：加载 /survey/editor_data 的导出稿，
    支持手动编辑、预览与（后续步骤）选中段落让 AI 重写。"""
    from fastapi.responses import FileResponse
    if not os.path.isfile(EDITOR_HTML):
        return {"ok": False, "error": f"编辑器页面文件缺失: {EDITOR_HTML}"}
    return FileResponse(EDITOR_HTML, media_type="text/html")


@app.get("/survey/editor_data")
def survey_editor_data_http(topic: str):
    from rag_core.survey import survey_editor_data
    return survey_editor_data(topic)


@app.post("/survey/rewrite_selection")
def survey_rewrite_selection_http(req: RewriteSelectionReq):
    from rag_core.survey import survey_rewrite_selection
    try:
        return survey_rewrite_selection(
            req.topic, req.selected_text, req.instruction,
            evidence=req.evidence, context_scope=req.context_scope,
        )
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/survey/editor_save")
def survey_editor_save_http(req: EditorSaveReq):
    from rag_core.survey import survey_editor_save
    try:
        return survey_editor_save(req.topic, req.text, req.filename)
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/survey/export_docx")
def survey_export_docx_http(req: EditorDocxReq):
    from rag_core.survey import survey_export_docx
    try:
        return survey_export_docx(req.topic, req.text,
                                  citation_format=req.citation_format,
                                  include_refs=req.include_refs,
                                  fmt_options=req.fmt_options)
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/survey/download")
def survey_download_http(topic: str, filename: str):
    """下载导出文件（md/markdown/docx；文件名白名单防路径穿越）。"""
    from fastapi.responses import FileResponse
    from rag_core.survey import survey_download_path
    p = survey_download_path(topic, filename)
    if not p:
        return {"ok": False, "error": "文件名不合法或文件不存在"}
    return FileResponse(p, media_type="application/octet-stream",
                        filename=os.path.basename(p))


# ================= MCP 工具（挂载到 /mcp，供 DSH 注册） =================

mcp = FastMCP("rag")


@mcp.tool(name="search")
def mcp_search(query: str, top_k: int = 5, year_min: int = None, year_max: int = None,
               authors: list = None, methods: list = None, tasks: list = None) -> dict:
    """在本地知识库中检索：只返回证据块与来源（含页码），不生成答案。
    可选元数据过滤：year_min/year_max（年份闭区间）、authors（作者，任一命中）、
    methods（方法标签，如 "随机森林"/"机器学习"）、tasks（任务标签，如 "信贷风控"）。
    适合需要自行组织材料的场景（文献分析/综述/方向对比等）。
    若用户要的是基于知识库的直接问答成品，请改用 ask 工具——它返回带可点击引用链接的
    标准答案（每个引用段落末尾挂 [来源N](http链接)，点击直开本地 PDF 对应页），
    直接把 ask 的 answer 字段呈现给用户即可，不要用 search 的证据自己改写。"""
    filters = {"year_min": year_min, "year_max": year_max,
               "authors": authors, "methods": methods, "tasks": tasks}
    return search_kb(query, top_k, filters)


@mcp.tool(name="stats")
def mcp_stats() -> dict:
    """本地知识库统计：总块数、各来源文档块数与运行统计。用于判断库内是否包含相关领域资料。"""
    return stats_kb()


@mcp.tool(name="build")
def mcp_build(chunker: str = "hmm", clear: bool = False) -> dict:
    """重建本地知识库：MinerU 输出 + Word 文档 → 分块 → 建索引。
    chunker 可选 fixed / discourse / hybrid / hmm（默认 hmm 无监督话题分割）。"""
    return build_kb(chunker, clear)


@mcp.tool(name="ingest")
def mcp_ingest() -> dict:
    """增量更新知识库：按文件哈希只处理新增文档，已有块向量不动（几十秒级）；
    检测到文档删除或内容变更时自动全量重建。新文档放好后调用本工具即可。"""
    return ingest_kb()


@mcp.tool(name="ask")
def mcp_ask(question: str, top_k: int = 5, year_min: int = None, year_max: int = None,
            authors: list = None, methods: list = None, tasks: list = None) -> dict:
    """检索 + 生成（走 DeepSeek API）：返回基于知识库的成品答案，
    每个引用段落的末尾带可点击引用链接 [来源N](http://127.0.0.1:3080/dsh-rag/open?doc=...&page=...)，
    点击直开本地 PDF 对应页（标准引用样式）。用户需要基于本地库回答问题/解释概念时
    优先调用本工具，并把 answer 字段直接作为回复呈现，不要改写。
    可选元数据过滤：year_min/year_max（年份闭区间）、authors（作者，任一命中）、
    methods（方法标签）、tasks（任务标签），如"只看2022年以后机器学习方法的信贷风控论文"。"""
    filters = {"year_min": year_min, "year_max": year_max,
               "authors": authors, "methods": methods, "tasks": tasks}
    return ask_kb(question, top_k, filters)


@mcp.tool(name="open_doc")
def mcp_open_doc(doc: str, page: int = 1) -> dict:
    """打开知识库中某文档的原始 PDF 并跳到指定页。
    doc 为文档名（如 论文名.pdf）；page 为页码，默认第 1 页。"""
    return open_doc_kb(doc, page)


@mcp.tool(name="direction_analyze")
def mcp_direction_analyze(direction: str, top_k: int = 12) -> dict:
    """分析单个研究方向：基于本地文献检索证据，给出创新点、可行性/数据可得性评分、
    风险、文献空白与切入建议。适合用户有模糊研究想法时逐步明确。"""
    from rag_core.research_advisor import analyze_direction
    try:
        return analyze_direction(direction, top_k)
    except Exception as e:
        return {"ok": False, "error": str(e)}


@mcp.tool(name="direction_compare")
def mcp_direction_compare(directions: list, weights: dict = None, top_k: int = 12) -> dict:
    """对比 2~5 个候选研究方向并排序推荐：多维评分（文献支撑度/近年热度/方法成熟度/
    创新空间/数据可得性/总体可行性），weights 可自定义各维度权重（键：
    literature/recency/maturity/gap/data/feasibility）。返回排序与推荐理由。"""
    from rag_core.research_advisor import compare_directions
    try:
        return compare_directions(directions, weights, top_k)
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ================= 交互式综述（survey）MCP 工具 =================

@mcp.tool(name="survey_outline")
def mcp_survey_outline(topic: str, constraints: str = "", outline: list = None) -> dict:
    """生成或保存综述大纲。传入 outline（手动编辑后的节列表 [{title, keywords}]）
    则直接校验保存；不传则基于本地文献检索 + LLM 自动生成 4~7 节大纲。"""
    from rag_core.survey import survey_outline
    try:
        return survey_outline(topic, constraints, outline)
    except Exception as e:
        return {"ok": False, "error": str(e)}


@mcp.tool(name="survey_draft")
def mcp_survey_draft(topic: str, outline: list = None, force: bool = False) -> dict:
    """按大纲逐节生成综述草稿：每节先检索本地证据（先检索后写作），段落内用 [来源N]
    标注引用（可溯源到文献与页码）。已生成的节默认跳过（断点续写）；force=True 全量重写。"""
    from rag_core.survey import survey_draft
    try:
        return survey_draft(topic, outline, force)
    except Exception as e:
        return {"ok": False, "error": str(e)}


@mcp.tool(name="survey_rewrite")
def mcp_survey_rewrite(topic: str, section: str, instruction: str = "更学术化、更深入") -> dict:
    """只重写综述的指定小节（section 为节标题或序号），其余节不变。
    instruction 为修改指令，如"深入对比深度学习与传统方法的优缺点""缩短到200字"。"""
    from rag_core.survey import survey_rewrite
    try:
        return survey_rewrite(topic, section, instruction)
    except Exception as e:
        return {"ok": False, "error": str(e)}


@mcp.tool(name="survey_edit")
def mcp_survey_edit(topic: str, section: str, text: str) -> dict:
    """手动编辑：用给定文本直接覆盖综述的指定小节（不经 LLM，用户手改）。"""
    from rag_core.survey import survey_edit
    try:
        return survey_edit(topic, section, text)
    except Exception as e:
        return {"ok": False, "error": str(e)}


@mcp.tool(name="survey_section")
def mcp_survey_section(topic: str, section: str) -> dict:
    """读取综述草稿中指定小节的当前内容（section 为节标题或序号）。"""
    from rag_core.survey import survey_section
    try:
        return survey_section(topic, section)
    except Exception as e:
        return {"ok": False, "error": str(e)}


@mcp.tool(name="survey_status")
def mcp_survey_status(topic: str) -> dict:
    """查看综述草稿状态：大纲、各节字数、总字数、引用数与草稿文件路径。"""
    from rag_core.survey import survey_status
    try:
        return survey_status(topic)
    except Exception as e:
        return {"ok": False, "error": str(e)}


@mcp.tool(name="survey_export")
def mcp_survey_export(topic: str, format: str = "markdown") -> dict:
    """导出综述（当前支持 markdown）：引用编号重排为 [N] 并生成参考文献列表，返回导出文件路径。
    结果含 editor_url——一个浏览器新标签页编辑器，可手动修改文字（保存/另存为在第4步接入），
    也可选中某一段让 AI 重写（第2步接入）。用户要"打开编辑器窗口"时，把 editor_url 作为可点击链接呈现即可。"""
    from rag_core.survey import survey_export
    try:
        r = survey_export(topic, format)
        if r.get("ok"):
            import urllib.parse
            r["editor_url"] = f"{EDITOR_BASE}?topic={urllib.parse.quote(str(topic))}"
        return r
    except Exception as e:
        return {"ok": False, "error": str(e)}


# 挂载 MCP（streamable-http 协议，路径 /mcp）。
# 注意三点：1) http_app 内部已自带 /mcp 路由，故挂载在根路径，否则变成 /mcp/mcp（404）；
# 2) http_app 的 lifespan 必须交给父应用，否则报 "lifespan was not passed"；
# 3) 挂载放在最后，父应用自身的路由优先匹配，其余请求落入 MCP 应用。
_mcp_app = mcp.http_app(transport="streamable-http", path="/mcp")
app.router.lifespan_context = _mcp_app.lifespan
app.mount("/", _mcp_app)


# ================= 浏览器友好页 =================
# MCP 规范要求 GET /mcp 必须带 Accept: text/event-stream（SSE 通知流），
# 否则服务器返回 406 "Not Acceptable"。浏览器直接打开这个地址会看到 406，
# 容易被误认为服务没连上。这里把这类请求转成友好状态页（DSH 等正规
# MCP 客户端带正确 Accept，不受影响，仍走真实 SSE 流）。
_MCP_FRIENDLY_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>RAG MCP 服务运行中</title>
<style>
body{font-family:"Microsoft YaHei",sans-serif;margin:40px;background:#f6f7f9;color:#333}
.ok{color:#2ecc71;font-weight:bold}
h1{font-size:22px}
code{background:#eef1f6;padding:2px 6px;border-radius:4px}
li{margin:4px 0}
</style></head>
<body>
<h1><span class="ok">&#9989;</span> RAG 一体化服务运行中（HTTP + MCP）</h1>
<p>这个地址是 <b>MCP（streamable-http）端点</b>，专供 DeepSeek Harness 等 MCP 客户端调用，不是给人浏览的网页。</p>
<p>已注册工具（DSH 中显示为 <code>mcp__rag__*</code>）：</p>
<ul>
<li><code>search(query, top_k)</code> —— 检索本地知识库</li>
<li><code>stats()</code> —— 知识库统计（含运行统计）</li>
<li><code>build(chunker, clear)</code> —— 重建知识库</li>
<li><code>ingest()</code> —— 增量更新（新文档入库）</li>
<li><code>ask(question, top_k)</code> —— 检索 + 生成问答</li>
<li><code>open_doc(doc, page)</code> —— 打开 PDF 指定页</li>
<li><code>direction_analyze(direction)</code> —— 单研究方向分析（创新点/可行性/空白）</li>
<li><code>direction_compare(directions, weights)</code> —— 多方向对比排序推荐</li>
<li><code>survey_outline / survey_draft / survey_rewrite / survey_edit</code> —— 交互式综述（大纲→草稿→改稿）</li>
<li><code>survey_section / survey_status / survey_export</code> —— 综述读取/状态/导出</li>
</ul>
<p>看到本页即说明服务正常。若 DSH 里工具未出现，请检查补丁配置的 <code>url</code> 与本服务端口是否一致。</p>
</body></html>"""


class McpFriendlyMiddleware:
    """纯 ASGI 中间件：GET /mcp 且 Accept 不含 text/event-stream 时返回友好状态页。"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] == "http"
            and scope.get("method") == "GET"
            and scope.get("path", "").rstrip("/") == "/mcp"
        ):
            accept = ""
            for k, v in scope.get("headers", []):
                if k.lower() == b"accept":
                    accept = v.decode("latin-1", "ignore").lower()
                    break
            if "text/event-stream" not in accept:
                body = _MCP_FRIENDLY_PAGE.encode("utf-8")
                await send({
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"content-type", b"text/html; charset=utf-8"),
                        (b"content-length", str(len(body)).encode()),
                        (b"cache-control", b"no-store"),
                    ],
                })
                await send({"type": "http.response.body", "body": body})
                return
        await self.app(scope, receive, send)


app.add_middleware(McpFriendlyMiddleware)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
