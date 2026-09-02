# -*- coding: utf-8 -*-
"""rag_core/corpus.py —— 多语料管理（布局约定 + 迁移 + 激活/切换/新建 + 运行时路径）。

统一布局（模型与日志全局共享，数据按语料隔离）：
    data/
    ├── corpora/
    │   ├── <语料名>/
    │   │   ├── knowledge_base.json
    │   │   ├── knowledge_base.json.ingest.json
    │   │   ├── vector_db/
    │   │   ├── doc_metadata.json
    │   │   ├── docs/            # 原始 PDF/Word
    │   │   ├── mineru_out/      # MinerU 解析产物（内含 batch/）
    │   │   ├── surveys/         # 综述工作台
    │   │   └── dict.txt         # 本语料 jieba 用户词典（可选）
    │   └── current.json         # 激活语料
    ├── models/                  # 全局共享
    └── observability.jsonl      # 全局共享
"""

import json
import os
import re
import shutil
from typing import Dict, List, Optional, Tuple

from rag_core import config

DEFAULT_NAME = config.DEFAULT_CORPUS  # 默认语料名（迁移目标）
CURRENT_FILE = "current.json"          # 语料总目录下的激活状态文件
SUB_KB = "knowledge_base.json"
SUB_INGEST = "knowledge_base.json.ingest.json"
SUB_META = "doc_metadata.json"
SUB_VDB = "vector_db"
SUB_DOCS = "docs"
SUB_MINERU = "mineru_out"
SUB_SURVEYS = "surveys"
SUB_DICT = "dict.txt"

_NAME_RE = re.compile(r"^[\w\u4e00-\u9fa5\- ]{1,40}$")
SUB_INGEST = "knowledge_base.json.ingest.json"
SUB_META = "doc_metadata.json"
SUB_VDB = "vector_db"
SUB_DOCS = "docs"
SUB_MINERU = "mineru_out"
SUB_SURVEYS = "surveys"
SUB_DICT = "dict.txt"


def _corpora_dir() -> str:
    return config.CORPORA_DIR


def corpus_dir(name: str) -> str:
    return os.path.join(_corpora_dir(), str(name).strip())


def current_file() -> str:
    return os.path.join(_corpora_dir(), CURRENT_FILE)


def read_current() -> Optional[str]:
    """读激活语料名；无状态文件返回 None。"""
    try:
        with open(current_file(), "r", encoding="utf-8") as f:
            data = json.load(f)
        name = str(data.get("name") or "").strip()
        return name if name else None
    except Exception:
        return None


def paths(name: str) -> Dict[str, str]:
    """某语料的全部运行时路径（按布局约定）。"""
    d = corpus_dir(name)
    return {
        "name": str(name).strip(),
        "kb": os.path.join(d, SUB_KB),
        "ingest": os.path.join(d, SUB_INGEST),
        "meta": os.path.join(d, SUB_META),
        "vector_db": os.path.join(d, SUB_VDB),
        "docs": os.path.join(d, SUB_DOCS),
        "mineru_out": os.path.join(d, SUB_MINERU),
        "surveys": os.path.join(d, SUB_SURVEYS),
        "dict": os.path.join(d, SUB_DICT),
    }


def _old_sources() -> List[Tuple[str, str]]:
    """旧布局（data/ 根）→ 新布局（corpora/<name>/）的迁移计划。
    旧位置按迁移前的固定布局取（data/ 根下的固定文件名），
    与新布局目标相同或不存在则跳过。"""
    data_dir = config.DATA_DIR
    p = paths(DEFAULT_NAME)
    plan = [
        (os.path.join(data_dir, "knowledge_base.json"), p["kb"], "知识库"),
        (os.path.join(data_dir, "knowledge_base.json.ingest.json"), p["ingest"], "增量入库状态"),
        (os.path.join(data_dir, "doc_metadata.json"), p["meta"], "文档元数据"),
        (os.path.join(data_dir, "vector_db"), p["vector_db"], "向量索引"),
        (os.path.join(data_dir, "docs"), p["docs"], "原始文档"),
        (os.path.join(data_dir, "mineru_out"), p["mineru_out"], "MinerU 解析产物"),
        (os.path.join(data_dir, "surveys"), p["surveys"], "综述工作台"),
    ]
    out = []
    for src, dst, what in plan:
        if src == dst:
            continue
        if os.path.exists(src):
            out.append((src, dst, what))
    return out


def _health_reachable() -> bool:
    """8000 服务是否在运行（迁移前必须停止，否则 qdrant 锁着 vector_db）。"""
    try:
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def _patch_kb_pdf_path(kb_path: str, docs_dir: str) -> int:
    """把 KB 里过期/失效的 pdf_path 刷新为语料 docs 目录；返回修复条数。"""
    if not os.path.isfile(kb_path):
        return 0
    with open(kb_path, "r", encoding="utf-8") as f:
        kb = json.load(f)
    fixed = 0
    for x in kb:
        p = x.get("pdf_path") or ""
        if p and os.path.isfile(p):
            continue
        base = os.path.basename(p) or (x.get("source") or "")
        newp = os.path.join(docs_dir, base)
        if os.path.isfile(newp):
            x["pdf_path"] = newp
            fixed += 1
    if fixed:
        tmp = kb_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(kb, f, ensure_ascii=False, indent=2)
        os.replace(tmp, kb_path)
    return fixed


def plan_migration() -> List[Tuple[str, str, str]]:
    return _old_sources()


def migrate(name: str = DEFAULT_NAME, dry_run: bool = False, force: bool = False) -> Dict:
    """执行一次性迁移：data/ 根 → data/corpora/<name>/。
    - 目标已存在 → 整体中止（先检查，避免搬一半）；
    - 8000 服务运行中 → 拒绝执行（force 可跳过检测）；
    - 迁移完成后写 current.json，并刷新 KB 内 pdf_path。"""
    target_dir = corpus_dir(name)
    plan = _old_sources()

    if dry_run:
        return {"ok": True, "dry_run": True, "plan": [(s, d, w) for s, d, w in plan]}

    # 冲突预检：目标已存在
    conflicts = [d for _, d, _ in plan if os.path.exists(d)]
    if conflicts:
        return {"ok": False, "error": "迁移目标已存在，中止：\n  " + "\n  ".join(conflicts)}

    if _health_reachable() and not force:
        return {"ok": False, "error": "检测到 8000 服务仍在运行——请先停止服务（迁移需移动 vector_db，qdrant 锁会阻止）"}

    if not plan:
        return {"ok": True, "moved": 0, "msg": "未发现需要迁移的旧布局文件（可能已迁移）"}

    os.makedirs(target_dir, exist_ok=True)
    moved = []
    for src, dst, what in plan:
        shutil.move(src, dst)
        moved.append(f"{what}: {src} -> {dst}")

    # 写激活状态
    os.makedirs(_corpora_dir(), exist_ok=True)
    tmp = current_file() + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"name": name}, f, ensure_ascii=False, indent=2)
    os.replace(tmp, current_file())

    # 刷新 KB 内 pdf_path（指向语料 docs）
    fixed = _patch_kb_pdf_path(paths(name)["kb"], paths(name)["docs"])

    return {"ok": True, "name": name, "moved": moved, "pdf_path_fixed": fixed}


# --------------------------------------------------------------------------
# 二期：激活/切换/新建 + 运行时路径
# --------------------------------------------------------------------------
def validate_name(name: str) -> Optional[str]:
    """校验语料名；非法返回错误信息，合法返回 None。"""
    n = str(name or "").strip()
    if not n:
        return "语料名不能为空"
    if not _NAME_RE.fullmatch(n):
        return "语料名只能包含中英文/数字/下划线/短横/空格（1~40 字符）"
    if n in (".", "..") or os.sep in n:
        return "语料名不合法"
    return None


def list_corpora() -> List[Dict]:
    """枚举语料目录（子目录视为语料），返回 [{name, kb, chunks, has_dict, has_docs}]。"""
    root = _corpora_dir()
    items = []
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            d = os.path.join(root, name)
            if not os.path.isdir(d) or name == "." or name.startswith("."):
                continue
            p = paths(name)
            chunks = 0
            if os.path.isfile(p["kb"]):
                try:
                    with open(p["kb"], "r", encoding="utf-8") as f:
                        chunks = len(json.load(f))
                except Exception:
                    pass
            items.append({
                "name": name,
                "kb": os.path.isfile(p["kb"]),
                "chunks": chunks,
                "has_dict": os.path.isfile(p["dict"]),
                "has_docs": os.path.isdir(p["docs"]),
            })
    return items


def switch(name: str) -> Dict:
    """切换激活语料：校验 + 原子写 current.json。"""
    err = validate_name(name)
    if err:
        return {"ok": False, "error": err}
    target = corpus_dir(name)
    if not os.path.isdir(target):
        return {"ok": False, "error": f"语料不存在: {name}（先 /corpus/create 或放入目录）"}
    os.makedirs(_corpora_dir(), exist_ok=True)
    tmp = current_file() + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"name": str(name).strip()}, f, ensure_ascii=False, indent=2)
    os.replace(tmp, current_file())
    return {"ok": True, "name": str(name).strip()}


def create(name: str) -> Dict:
    """新建语料目录（不建库——建库由服务端 build 以 target 方式执行）。"""
    err = validate_name(name)
    if err:
        return {"ok": False, "error": err}
    target = corpus_dir(name)
    if os.path.exists(target):
        return {"ok": False, "error": f"语料已存在: {name}"}
    try:
        os.makedirs(target, exist_ok=False)
    except OSError as e:
        return {"ok": False, "error": f"创建目录失败: {e}"}
    return {"ok": True, "name": str(name).strip(), "paths": paths(name)}


def runtime_paths(name: Optional[str] = None) -> Dict[str, str]:
    """激活语料的运行时路径（调用时解析，随 current.json 变化）。
    显式环境变量覆盖优先（全局锁定，不随语料切换）。
    兼容模式：语料布局尚未建立（无 current.json 且默认语料目录不存在）时，
    沿用 config 的默认路径（如仓库自带样例数据），保证旧部署零迁移可用。"""
    active = name or read_current() or DEFAULT_NAME
    p = paths(active)
    if not (os.path.isfile(current_file()) or os.path.isdir(corpus_dir(active))):
        p.update(
            kb=config.KB_FILE,
            vector_db=config.VECTOR_DB_PATH,
            mineru_out=config.MINERU_OUT,
            docs=config.DOCS_DIR,
            meta=config.DOC_META_FILE,
            surveys=config.SURVEY_DIR,
        )
    env_map = (
        ("RAG_KB_FILE", "kb"),
        ("RAG_VECTOR_DB", "vector_db"),
        ("MINERU_OUT", "mineru_out"),
        ("DOCS_DIR", "docs"),
        ("RAG_DOC_META", "meta"),
        ("RAG_SURVEY_DIR", "surveys"),
    )
    for env_key, key in env_map:
        v = os.getenv(env_key)
        if v:
            p[key] = v
    p["ingest"] = p["kb"] + ".ingest.json"
    p["pdf_source_dirs"] = [
        d for d in (
            [x.strip() for x in os.getenv("PDF_SOURCE_DIRS", "").split(";") if x.strip()]
            + [p["docs"]]
        ) if d
    ]
    p["active"] = active
    return p
