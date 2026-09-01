# -*- coding: utf-8 -*-
"""交互式综述生成（第一期）：大纲协商 → 分段生成（先检索后写作、引用可溯源）
→ 局部重写 / 手动编辑 → 状态查询 → Markdown 导出。

状态落盘：data/surveys/<slug>/outline.json + draft.md + refs.json。
交互由 DSH agent 会话或桌面面板承担；工具全部无状态读写文件。

用法：
    from rag_core.survey import survey_outline, survey_draft, survey_rewrite
    survey_outline("机器学习在信贷风控中的应用")
    survey_draft("机器学习在信贷风控中的应用")
"""

import json
import os
import re
from typing import Dict, List, Optional

from rag_core.config import KB_FILE as KB_FILE_DEFAULT

SURVEY_DIR = os.getenv(
    "RAG_SURVEY_DIR",
    os.path.join(os.path.dirname(KB_FILE_DEFAULT), "surveys"),
)

_SLUG_RE = re.compile(r"[^\w\u4e00-\u9fa5]+")


def _slugify(topic: str) -> str:
    slug = _SLUG_RE.sub("_", topic.strip()).strip("_")[:40]
    return slug or "survey"


def _survey_dir(topic: str) -> str:
    return os.path.join(SURVEY_DIR, _slugify(topic))


def _load_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _llm_text(system: str, user: str, timeout: int = 120) -> str:
    from openai import OpenAI
    api_key = os.getenv("deepseek_api", "")
    if not api_key:
        raise RuntimeError("未配置 deepseek_api")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.4,
        stream=False,
        timeout=timeout,
    )
    return resp.choices[0].message.content or ""


_OUTLINE_PROMPT = """你是学术综述的结构设计专家。根据给定主题与本地文献证据，生成综述大纲 JSON：

{
  "sections": [
    {"title": "1 引言", "keywords": ["关键词1", "关键词2"]},
    {"title": "2 文献综述", "keywords": ["..."]}
  ]
}

要求：
- 4~7 节，标题简洁（可含编号）；
- 每节给 2~4 个检索关键词（用于后续逐节检索证据）；
- 结构要覆盖：背景意义、方法综述、应用场景、评价指标、挑战与展望；
- 只输出 JSON，不要任何解释文字。"""

_SECTION_PROMPT = """你是学术综述撰写助手。请撰写综述的某一节。要求：
- 只依据给定的证据片段写作，不得编造证据之外的事实；
- 引用方式：在相关句末用 [来源N] 标注（N 为证据编号），每段至少 1 处引用；
- 学术化、连贯、简洁，中文，250~450 字；
- 若提供"上一节结尾"，可自然承接，但不要重复其内容；
- 直接输出本节正文，不要输出标题，不要解释。"""

_REWRITE_PROMPT = """你是学术综述修改助手。根据用户指令改写给定小节。要求：
- 保持该节的主题与位置不变；
- 只依据给定的证据片段补充内容，不得编造；
- 引用方式：相关句末用 [来源N] 标注（N 为证据编号）；
- 直接输出改写后的正文（不含标题），中文，学术化。"""

_REWRITE_SEL_PROMPT = """你是学术综述修改助手。根据用户指令改写选中的段落。要求：
- 保持段落的主题与所在位置不变；
- 只输出改写后的段落正文，不要输出标题、不要解释、不要输出引用编号；
- 学术化、连贯，中文；
- 若指令要求扩写或补充论据，只能基于给出的"所在小节上下文"，不得编造。"""

_REWRITE_SEL_PROMPT_EVIDENCE = """你是学术综述修改助手。根据用户指令改写选中的段落，允许基于"可参考证据"补充论据。要求：
- 保持段落的主题与所在位置不变；
- 补充的论据必须来自证据片段，不得编造证据之外的事实；
- 只输出改写后的段落正文，不要输出标题、不要解释、不要输出引用编号（引用由用户自行管理）。"""


def _collect_evidence(query: str, top_k: int = 15) -> List[Dict]:
    """检索证据：[(source, page, text)]，按文档去重。"""
    import rag_server as core
    from rag_core.doc_metadata import load_meta

    retriever = core.get_retriever()
    results = retriever.retrieve(query, bm25_k=20, vector_k=20, rerank_k=top_k)
    meta = load_meta()
    seen = set()
    out = []
    for r in results:
        m = r.get("metadata") or {}
        src = m.get("source", "未知")
        text = (r.get("text") or "").strip()
        if not text or (src, text) in seen:
            continue
        seen.add((src, text))
        dm = meta.get(src, {})
        out.append({
            "source": src,
            "page": m.get("page_start"),
            "author": dm.get("author"),
            "year": dm.get("year"),
            "text": text,
        })
        if len(out) >= 8:
            break
    return out


def _refs_path(d):
    return os.path.join(d, "refs.json")


def _load_refs(d) -> List[Dict]:
    return _load_json(_refs_path(d), [])


def _save_refs(d, refs):
    _save_json(_refs_path(d), refs)
    # 同步写人读版对照表 refs.md
    try:
        lines = ["# 引用对照表", ""]
        for i, r in enumerate(refs, start=1):
            page = f"，第 {r['page']} 页" if r.get("page") else ""
            lines.append(f"- [来源{i}] {r.get('author') or '佚名'}. {_title_of(r)}（{r.get('year') or '?'}）{page}")
        with open(os.path.join(d, "refs.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except OSError:
        pass


_NOTE_RE = re.compile(r"(?m)^> 【本节引用】[^\n]*\n?")


def _title_of(r: Dict) -> str:
    """从引用条目取干净标题：去 .pdf 后缀，并剥离文件名尾部的作者段（标题_作者.pdf）。"""
    title = r.get("source", "?")
    if title.lower().endswith(".pdf"):
        title = title[:-4]
    author = r.get("author")
    if author and title.endswith("_" + author):
        title = title[: -len("_" + author)]
    return title


def _refs_note(refs: List[Dict], ids: List[int]) -> str:
    """把引用编号解析成人读注记：> 【本节引用】[来源N] 作者. 标题（年份），第X页；..."""
    parts = []
    for n in ids:
        if 1 <= n <= len(refs):
            r = refs[n - 1]
            page = f"，第{r['page']}页" if r.get("page") else ""
            parts.append(f"[来源{n}] {r.get('author') or '佚名'}. {_title_of(r)}（{r.get('year') or '?'}）{page}")
    return "> 【本节引用】" + "；".join(parts) if parts else ""


def _append_note(body: str, refs: List[Dict]) -> str:
    """给节正文追加本节引用注（若正文含引用且尚无注记）。"""
    if "【本节引用】" in body:
        return body
    ids = sorted({int(x) for x in re.findall(r"\[来源(\d+)\]", body)
                  if 1 <= int(x) <= len(refs)})
    note = _refs_note(refs, ids)
    return body + "\n\n" + note if note else body


# --------------------------------------------------------------------------
# 大纲
# --------------------------------------------------------------------------
def _normalize_outline(outline) -> List[Dict]:
    sections = []
    for i, s in enumerate(outline or [], start=1):
        if isinstance(s, str):
            s = {"title": s, "keywords": []}
        title = str(s.get("title", "")).strip()
        if not title:
            continue
        sections.append({
            "title": title if re.match(r"^\d", title) else f"{i} {title}",
            "keywords": [str(k).strip() for k in s.get("keywords", []) if str(k).strip()][:4],
        })
    return sections


def survey_outline(topic: str, constraints: str = "", outline: Optional[List] = None) -> Dict:
    """生成或保存大纲。传入 outline（手动编辑）则直接校验保存；否则检索+LLM 生成。"""
    d = _survey_dir(topic)
    if outline:
        sections = _normalize_outline(outline)
        if not sections:
            return {"ok": False, "error": "大纲为空或格式不正确（需含 title 字段的列表）"}
        data = {"topic": topic.strip(), "constraints": constraints or "", "sections": sections}
        _save_json(os.path.join(d, "outline.json"), data)
        return {"ok": True, "topic": topic, "sections": sections, "mode": "manual"}

    evidence = _collect_evidence(topic, top_k=12)
    evidence_txt = "\n".join(
        f"- [{i + 1}] {e['source']}（{e.get('year') or '?'}，作者 {e.get('author') or '?'}）：{e['text'][:150]}"
        for i, e in enumerate(evidence)
    )
    raw = _llm_text(_OUTLINE_PROMPT,
                    f"主题：{topic}\n约束：{constraints or '无'}\n\n本地文献证据：\n{evidence_txt}")
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if m:
        raw = m.group(1)
    data = json.loads(raw)
    sections = _normalize_outline(data.get("sections", []))
    if not sections:
        return {"ok": False, "error": "大纲生成失败（LLM 返回无有效 sections）"}
    saved = {"topic": topic.strip(), "constraints": constraints or "", "sections": sections}
    _save_json(os.path.join(d, "outline.json"), saved)
    return {"ok": True, "topic": topic, "sections": sections,
            "mode": "generated", "evidence_docs": len(evidence)}


def _load_outline(d) -> Optional[Dict]:
    data = _load_json(os.path.join(d, "outline.json"), None)
    return data


# --------------------------------------------------------------------------
# 草稿（逐节生成，先检索后写作）
# --------------------------------------------------------------------------
def _draft_parts(text: str) -> List[Dict]:
    """把 draft.md 拆成节列表 [{title, body}]；无标题内容归入前言。"""
    parts: List[Dict] = []
    for block in re.split(r"(?m)^(?=##\s)", text or ""):
        block = block.strip()
        if not block:
            continue
        m = re.match(r"^##\s+(.+)$", block, flags=re.M)
        if m:
            title = m.group(1).strip()
            body = block[m.end():].strip()
            parts.append({"title": title, "body": body})
        else:
            parts.append({"title": "", "body": block})
    return parts


def _dump_draft(parts: List[Dict], topic: str) -> str:
    head = parts[0] if parts and not parts[0]["title"] else {"title": "", "body": ""}
    out = []
    if head["body"]:
        out.append(head["body"])
    for p in parts:
        if p["title"]:
            out.append(f"## {p['title']}\n\n{p['body']}".strip())
    return "\n\n".join(out) + "\n"


def survey_draft(topic: str, outline: Optional[List] = None, force: bool = False) -> Dict:
    """按大纲逐节生成草稿（已存在的节跳过，可断点续写；force=True 全量重写）。"""
    d = _survey_dir(topic)
    data = _load_outline(d)
    sections = _normalize_outline(outline) if outline else (
        data.get("sections", []) if data else [])
    if not sections:
        return {"ok": False, "error": f"未找到大纲（{_slugify(topic)}），请先 survey_outline"}

    draft_file = os.path.join(d, "draft.md")
    parts = _draft_parts(open(draft_file, encoding="utf-8").read()) if os.path.isfile(draft_file) else []
    if force:
        # 强制重写：丢弃已有节（仅保留无标题的前言，如有）
        parts = [p for p in parts if not p["title"]]
    existing_titles = {p["title"] for p in parts if p["title"]}
    refs = _load_refs(d)

    written = []
    prev_tail = ""
    for i, sec in enumerate(sections, start=1):
        if not force and sec["title"] in existing_titles:
            continue
        query = sec["title"] + " " + " ".join(sec["keywords"])
        evidence = _collect_evidence(query, top_k=15) or _collect_evidence(topic, top_k=15)
        evidence_txt = []
        for ev in evidence:
            refs.append({"source": ev["source"], "page": ev["page"],
                         "author": ev["author"], "year": ev["year"], "text": ev["text"][:300]})
            label = f"[来源{len(refs)}]"
            evidence_txt.append(
                f"{label}（{ev['source']}，第 {ev['page'] or '?'} 页）: {ev['text'][:250]}")
        user = (
            f"主题：{topic}\n本节：{sec['title']}\n关键词：{'、'.join(sec['keywords'])}\n\n"
            f"证据片段：\n" + "\n".join(evidence_txt) +
            (f"\n\n上一节结尾（用于衔接）：\n{prev_tail[-150:]}" if prev_tail else "")
        )
        body = _llm_text(_SECTION_PROMPT, user).strip()
        prev_tail = body
        body = _append_note(body, refs)  # 节末附“本节引用”注（人读文献名+年份+页码）
        parts.append({"title": sec["title"], "body": body})
        existing_titles.add(sec["title"])
        written.append(sec["title"])

    with open(draft_file, "w", encoding="utf-8") as f:
        f.write(_dump_draft(parts, topic))
    _save_refs(d, refs)
    chars = sum(len(p["body"]) for p in parts)
    return {"ok": True, "topic": topic, "written": written,
            "sections": len(sections), "total_chars": chars,
            "refs": len(refs), "draft_path": draft_file}


def _find_section(parts: List[Dict], section) -> Optional[int]:
    for idx, p in enumerate(parts):
        if not p["title"]:
            continue
        if str(section) == str(p["title"]) or str(section) in str(p["title"]):
            return idx
    try:
        i = int(section)
        titled = [j for j, p in enumerate(parts) if p["title"]]
        if 1 <= i <= len(titled):
            return titled[i - 1]
    except (TypeError, ValueError):
        pass
    return None


def survey_section(topic: str, section) -> Dict:
    d = _survey_dir(topic)
    draft_file = os.path.join(d, "draft.md")
    if not os.path.isfile(draft_file):
        return {"ok": False, "error": "草稿不存在，请先 survey_draft"}
    parts = _draft_parts(open(draft_file, encoding="utf-8").read())
    idx = _find_section(parts, section)
    if idx is None:
        return {"ok": False, "error": f"未找到小节: {section}"}
    return {"ok": True, "topic": topic, "title": parts[idx]["title"],
            "body": parts[idx]["body"]}


def survey_rewrite(topic: str, section, instruction: str = "更学术化、更深入") -> Dict:
    d = _survey_dir(topic)
    draft_file = os.path.join(d, "draft.md")
    if not os.path.isfile(draft_file):
        return {"ok": False, "error": "草稿不存在，请先 survey_draft"}
    parts = _draft_parts(open(draft_file, encoding="utf-8").read())
    idx = _find_section(parts, section)
    if idx is None:
        return {"ok": False, "error": f"未找到小节: {section}"}
    sec = parts[idx]

    query = sec["title"] + " " + instruction
    evidence = _collect_evidence(query, top_k=15)
    refs = _load_refs(d)
    evidence_txt = []
    for ev in evidence:
        refs.append({"source": ev["source"], "page": ev["page"],
                     "author": ev["author"], "year": ev["year"], "text": ev["text"][:300]})
        evidence_txt.append(f"[来源{len(refs)}]（{ev['source']}，第 {ev['page'] or '?'} 页）: {ev['text'][:250]}")

    # 剥离旧引用注后再交给 LLM（避免注记干扰重写）
    old_body = _NOTE_RE.sub("", sec["body"]).strip()
    user = (
        f"主题：{topic}\n小节：{sec['title']}\n修改指令：{instruction}\n\n"
        f"当前正文：\n{old_body}\n\n证据片段：\n" + "\n".join(evidence_txt)
    )
    new_body = _llm_text(_REWRITE_PROMPT, user).strip()
    new_body = _append_note(new_body, refs)
    parts[idx]["body"] = new_body
    with open(draft_file, "w", encoding="utf-8") as f:
        f.write(_dump_draft(parts, topic))
    _save_refs(d, refs)
    return {"ok": True, "topic": topic, "title": sec["title"], "body": new_body}


def survey_edit(topic: str, section, text: str) -> Dict:
    """手动编辑：直接用给定文本覆盖某节（不经 LLM）。"""
    d = _survey_dir(topic)
    draft_file = os.path.join(d, "draft.md")
    if not os.path.isfile(draft_file):
        return {"ok": False, "error": "草稿不存在，请先 survey_draft"}
    parts = _draft_parts(open(draft_file, encoding="utf-8").read())
    idx = _find_section(parts, section)
    if idx is None:
        return {"ok": False, "error": f"未找到小节: {section}"}
    body = (text or "").strip()
    # 手改后若正文仍有 [来源N] 且无注记 → 自动补“本节引用”注
    body = _append_note(body, _load_refs(d))
    parts[idx]["body"] = body
    with open(draft_file, "w", encoding="utf-8") as f:
        f.write(_dump_draft(parts, topic))
    return {"ok": True, "topic": topic, "title": parts[idx]["title"]}


def survey_status(topic: str) -> Dict:
    d = _survey_dir(topic)
    data = _load_outline(d)
    draft_file = os.path.join(d, "draft.md")
    parts = _draft_parts(open(draft_file, encoding="utf-8").read()) if os.path.isfile(draft_file) else []
    refs = _load_refs(d)
    return {
        "ok": True, "topic": topic, "slug": _slugify(topic),
        "outline": data.get("sections", []) if data else [],
        "sections": [
            {"title": p["title"] or "(前言)", "chars": len(p["body"]),
             "has_body": bool(p["body"])}
            for p in parts if p["title"]
        ],
        "total_chars": sum(len(p["body"]) for p in parts),
        "refs": len(refs),
        "draft_path": draft_file,
    }


def survey_export(topic: str, format: str = "markdown") -> Dict:
    """导出 Markdown（参考文献编号对齐，[来源N] → [N]）。"""
    d = _survey_dir(topic)
    draft_file = os.path.join(d, "draft.md")
    if not os.path.isfile(draft_file):
        return {"ok": False, "error": "草稿不存在，请先 survey_draft"}
    text = open(draft_file, encoding="utf-8").read()
    refs = _load_refs(d)

    # 导出前剥离“本节引用”注记（正式引用由文末参考文献列表承担）
    text = _NOTE_RE.sub("", text)

    # 全局引用号 → 参考文献序号：按首次出现顺序重编号
    order = []
    for m in re.finditer(r"\[来源(\d+)\]", text):
        n = int(m.group(1))
        if n not in order:
            order.append(n)
    mapping = {n: i + 1 for i, n in enumerate(order)}

    def _sub(m):
        return f"[{mapping[int(m.group(1))]}]"

    body = re.sub(r"\[来源(\d+)\]", _sub, text)
    lines = [body.rstrip(), "", "## 参考文献", ""]
    for n in order:
        r = refs[n - 1] if n - 1 < len(refs) else {}
        author = r.get("author") or "佚名"
        year = r.get("year") or "?"
        page = r.get("page")
        page_txt = f"，第 {page} 页" if page else ""
        lines.append(f"[{mapping[n]}] {author}. {_title_of(r)}（{year}）{page_txt}")
    out_dir = os.path.join(d, "exports")
    os.makedirs(out_dir, exist_ok=True)

    # 落盘编号顺序侧车（docx 著者-年份转换的兜底映射：N → 来源序号）
    try:
        _save_json(os.path.join(out_dir, f"{_slugify(topic)}.map.json"), {"order": order})
    except OSError:
        pass

    if format == "docx":
        from rag_core.md_docx import md_to_docx, extract_ref_map
        full_text = "\n".join(lines) + "\n"
        out_path = os.path.join(out_dir, f"{_slugify(topic)}.docx")
        md_to_docx(full_text, out_path, topic.strip(),
                   citation_format="author_year", include_refs=False,
                   ref_map=extract_ref_map(full_text))
        return {"ok": True, "topic": topic, "format": format, "path": out_path,
                "refs": len(order), "citation_format": "author_year"}

    out_path = os.path.join(out_dir, f"{_slugify(topic)}.{format}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return {"ok": True, "topic": topic, "format": format, "path": out_path,
            "refs": len(order)}


def survey_editor_data(topic: str) -> Dict:
    """编辑器数据源（综述导出升级·第一步）：返回导出 Markdown 的文本与路径，
    供浏览器编辑器（/editor?topic=...）加载；导出文件不存在时先自动导出
    （导出只读草稿、不调 LLM，秒级）。编辑保存逻辑由编辑器端点负责，
    本函数只读、不落盘（自动导出除外）。"""
    d = _survey_dir(topic)
    if not os.path.isfile(os.path.join(d, "draft.md")):
        return {"ok": False, "error": "草稿不存在，请先 survey_draft"}
    out_path = os.path.join(d, "exports", f"{_slugify(topic)}.markdown")
    if not os.path.isfile(out_path):
        r = survey_export(topic, "markdown")
        if not r.get("ok"):
            return {"ok": False, "error": r.get("error", "导出失败")}
        out_path = r["path"]
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        return {"ok": False, "error": f"读取失败: {e}"}
    return {"ok": True, "topic": topic.strip(), "text": text, "path": out_path}


def _clean_rewrite(text: str) -> str:
    """清理 LLM 重写输出：剥代码围栏/首尾引号与空白、去可能的"改写后："前缀。"""
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()
    t = re.sub(r"^[\s\"'“”「」]+|[\s\"'“”「」]+$", "", t)
    t = re.sub(r"^改写后[：:]\s*", "", t)
    return t.strip()


def survey_rewrite_selection(topic: str, selected_text: str,
                             instruction: str = "改写得更学术化",
                             evidence: bool = False,
                             context_scope: str = "section",
                             top_k: int = 5) -> Dict:
    """选中段落重写（综述导出升级·第二步）：
    - v1 纯文字：基于指令 + 所在小节上下文（context_scope="section"，默认）；
    - v2 带证据：evidence=True 时先检索知识库，允许基于证据补充论据，
      返回 evidence 列表（source/page/excerpt），但不在输出里插引用编号（引用由用户管理）；
    - context_scope="full"：用整篇导出稿做上下文（保留全文重写接口）。
    只读草稿与导出稿，不落盘（保存是第四步的事）。"""
    d = _survey_dir(topic)
    if not os.path.isfile(os.path.join(d, "draft.md")):
        return {"ok": False, "error": "草稿不存在，请先 survey_draft"}

    sel = (selected_text or "").strip()
    if len(sel) < 10:
        return {"ok": False, "error": "选中文字太短（至少 10 字）"}
    if len(sel) > 4000:
        return {"ok": False, "error": "选中文字过长（上限 4000 字）"}
    instr = (instruction or "").strip() or "改写得更学术化"

    # 上下文定位：用导出稿（引用注记已剥离）
    ed = survey_editor_data(topic)
    full_text = ed["text"] if ed.get("ok") else ""

    ctx_text, used_scope = "", context_scope
    if context_scope == "full":
        used_scope = "full"
        ctx_text = full_text.replace(sel, "「此处为待改写段落」", 1) if sel in full_text else full_text
    else:
        idx = full_text.find(sel)
        if idx >= 0:
            head = full_text[:idx]
            m = None
            for mm in re.finditer(r"^##[ \t]+.+$", head, re.M):
                m = mm
            if m:
                sect_start = m.start()
                tail = full_text[idx + len(sel):]
                nm = re.search(r"^##[ \t]+", tail, re.M)
                sect_end = idx + len(sel) + (nm.start() if nm else len(tail))
                ctx_text = full_text[sect_start:sect_end].replace(sel, "", 1)
        if not ctx_text.strip():
            # 选中内容不在导出稿（编辑器有未保存改动）或该节只有选中段：
            # 退回全文上下文，保证重写仍有语境
            ctx_text, used_scope = full_text, "full(fallback)"
    ctx_text = ctx_text.strip()[:2000]

    evidence_list, ev_block = [], ""
    if evidence:
        for e in (_collect_evidence(sel[:120], top_k=top_k) or [])[:top_k]:
            page = f"，第{e['page']}页" if e.get("page") else ""
            evidence_list.append({
                "source": e.get("source"), "page": e.get("page"),
                "excerpt": (e.get("text") or "")[:200],
            })
            ev_block += f"- {e.get('source')}{page}：{(e.get('text') or '')[:260]}\n"
        if not ev_block:
            return {"ok": False, "error": "未检索到相关证据（可关闭\"带知识库证据\"后重试）"}

    system = _REWRITE_SEL_PROMPT_EVIDENCE if evidence else _REWRITE_SEL_PROMPT
    user = (
        f"指令：{instr}\n\n"
        f"所在小节上下文（不含选中段落）：\n{ctx_text or '（无）'}\n\n"
        + (f"可参考证据：\n{ev_block}\n\n" if ev_block else "")
        + f"待改写段落：\n{sel}"
    )
    try:
        out = _clean_rewrite(_llm_text(system, user))
    except Exception as e:
        return {"ok": False, "error": f"重写失败: {e}"}
    if not out:
        return {"ok": False, "error": "重写结果为空（LLM 无输出）"}
    return {"ok": True, "rewritten_text": out, "evidence": evidence_list,
            "context_scope": used_scope}


_SAVE_NAME_RE = re.compile(r"[\w\u4e00-\u9fa5.\- ]{1,80}\.(md|markdown)")


def survey_editor_save(topic: str, text: str, filename: str = "") -> Dict:
    """编辑器保存（综述导出升级·第四步）：写到 <survey>/exports/ 下，
    不触碰 draft.md（工作稿与编辑稿分离，导出于编辑相互独立）。
    - filename 为空 → 覆盖默认导出文件 <slug>.markdown；
    - filename 非空 → 另存为（白名单：仅中英文/数字/点/短横/空格，
      且必须以 .md / .markdown 结尾；落点限定在本主题 exports 目录内，防路径穿越）。"""
    d = _survey_dir(topic)
    if not os.path.isfile(os.path.join(d, "draft.md")):
        return {"ok": False, "error": "草稿不存在，请先 survey_draft"}
    out_dir = os.path.join(d, "exports")
    if filename and str(filename).strip():
        name = str(filename).strip()
        if not _SAVE_NAME_RE.fullmatch(name):
            return {"ok": False, "error": "文件名不合法：只允许中英文/数字/点/短横/空格，且以 .md 或 .markdown 结尾"}
        out_path = os.path.join(out_dir, name)
        if os.path.abspath(out_path) != os.path.join(os.path.abspath(out_dir), name):
            return {"ok": False, "error": "文件名不合法"}  # 双保险
    else:
        out_path = os.path.join(out_dir, f"{_slugify(topic)}.markdown")
    os.makedirs(out_dir, exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, out_path)
    return {"ok": True, "topic": topic.strip(), "path": out_path,
            "filename": os.path.basename(out_path), "chars": len(text)}


_DOWNLOAD_NAME_RE = re.compile(r"[\w\u4e00-\u9fa5.\- ]{1,80}\.(md|markdown|docx)")


def survey_export_docx(topic: str, text: str = None,
                       citation_format: str = "author_year",
                       include_refs: bool = False) -> Dict:
    """导出 Word（引用格式由用户选择）：
    - citation_format: "author_year"（[1] → （作者，年份），默认）/
                       "superscript"（[1] 保持数字引用并渲染为 Word 上标）；
    - include_refs: 是否在文末附生成的参考文献列表（默认不附，正式引用走知网）；
    - text: 编辑器当前文本（未保存的改动也生效）；缺省加载已导出的 Markdown。
    产物：exports/<slug>_著者年份.docx / <slug>_上标编号.docx。"""
    from rag_core.md_docx import md_to_docx, extract_ref_map

    if citation_format not in ("superscript", "author_year"):
        return {"ok": False, "error": "未知引用格式（可选 superscript / author_year）"}
    d = _survey_dir(topic)
    if not os.path.isfile(os.path.join(d, "draft.md")):
        return {"ok": False, "error": "草稿不存在，请先 survey_draft"}

    if text is None or not str(text).strip():
        ed = survey_editor_data(topic)
        if not ed.get("ok"):
            return {"ok": False, "error": ed.get("error", "加载草稿失败")}
        text = ed["text"]
    text = str(text)

    # 编号 → (作者, 年份)：优先从文末参考文献列表解析（编辑稿编号也能对上）
    ref_map = extract_ref_map(text)
    if not ref_map:
        # 兜底：导出侧车 map.json + refs.json
        sidecar = os.path.join(d, "exports", f"{_slugify(topic)}.map.json")
        order = _load_json(sidecar, {}).get("order", []) if os.path.isfile(sidecar) else []
        refs = _load_refs(d)
        for n, idx in enumerate(order, start=1):
            if 0 < idx <= len(refs):
                r = refs[idx - 1]
                ref_map[n] = (r.get("author") or "佚名", str(r.get("year") or ""))

    suffix = {"superscript": "上标编号", "author_year": "著者年份"}[citation_format]
    out_dir = os.path.join(d, "exports")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{_slugify(topic)}_{suffix}.docx")
    md_to_docx(text, out_path, topic.strip(), citation_format=citation_format,
               include_refs=include_refs, ref_map=ref_map)
    return {"ok": True, "topic": topic.strip(), "format": "docx",
            "citation_format": citation_format, "include_refs": include_refs,
            "path": out_path, "filename": os.path.basename(out_path)}


def survey_download_path(topic: str, filename: str) -> Optional[str]:
    """下载白名单：返回 exports 目录内合法文件的绝对路径；非法/不存在返回 None。"""
    if not filename or not _DOWNLOAD_NAME_RE.fullmatch(str(filename).strip()):
        return None
    d = _survey_dir(topic)
    out_dir = os.path.join(d, "exports")
    name = str(filename).strip()
    p = os.path.join(out_dir, name)
    if os.path.abspath(p) != os.path.join(os.path.abspath(out_dir), name):
        return None
    return p if os.path.isfile(p) else None
