# -*- coding: utf-8 -*-
"""全局配置：所有路径与超参集中在此，可用环境变量 / 项目根目录 .env 覆盖。

优先级：系统环境变量 > .env > 默认值（默认值全部解析到项目内目录，
换机器零迁移）。
"""

import os


def _load_dotenv():
    """加载项目根目录 .env（若存在）；系统环境变量优先，不会被覆盖。"""
    env_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
    )
    if not os.path.isfile(env_file):
        return
    try:
        with open(env_file, "r", encoding="utf-8-sig") as f:
            lines = f.read().splitlines()
    except UnicodeDecodeError:
        with open(env_file, "r", encoding="gbk", errors="ignore") as f:
            lines = f.read().splitlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


_load_dotenv()

# ---- 目录 ----
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../rag_core
PROJECT_DIR = os.path.dirname(PACKAGE_DIR)                  # 项目根目录
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# ---- 路径（默认全部在项目内；环境变量可覆盖）----
# 知识库与向量索引
KB_FILE = os.getenv("RAG_KB_FILE", os.path.join(DATA_DIR, "knowledge_base.json"))
VECTOR_DB_PATH = os.getenv("RAG_VECTOR_DB", os.path.join(DATA_DIR, "vector_db"))
# MinerU 输出目录（content_list.json 所在根；默认指向仓库自带样例，便于 3 分钟跑通）
MINERU_OUT = os.getenv(
    "MINERU_OUT",
    os.path.join(PROJECT_DIR, "examples", "mineru_output", "batch"),
)
# Word 文档目录（默认仓库自带样例目录）
DOCS_DIR = os.getenv("DOCS_DIR", os.path.join(PROJECT_DIR, "examples", "input_docx"))
# 嵌入/重排模型缓存目录（首次构建自动从 ModelScope 下载，约 2GB）
MODEL_DIR = os.getenv("MODELSCOPE_CACHE", os.path.join(PROJECT_DIR, "models"))
# 可观测日志
OBS_LOG = os.getenv("RAG_OBS_LOG", os.path.join(DATA_DIR, "observability.jsonl"))
# 原始 PDF 搜索目录（打开引用时按文件名定位；; 分隔追加）
PDF_SOURCE_DIRS = [d for d in (
    [x.strip() for x in os.getenv("PDF_SOURCE_DIRS", "").split(";") if x.strip()]
    + [DOCS_DIR]
) if d]
# 引用链接基地址（配合 dsh-rag-citation 插件点击翻页；DSH 端口变化时覆盖）
RAG_LINK_BASE = os.getenv("RAG_LINK_BASE", "http://127.0.0.1:3080/dsh-rag/open")

# ---- 模型 ----
# 中文语料默认 bge-base-zh-v1.5；英文语料设环境变量 BGE_EMBED_MODEL=bge-base-en-v1.5
EMBED_MODEL_ID = os.getenv("BGE_EMBED_MODEL", "bge-base-zh-v1.5")
RERANK_MODEL_ID = "bge-reranker-v2-m3"   # 多语重排模型，无需切换

# ---- 问答 ----
DEEPSEEK_API_KEY = os.getenv("deepseek_api", "")

# ---- 分块 ----
CHUNK_SIZE = 800
OVERLAP_TOKENS = 50
DEFAULT_CHUNKER = "hmm"   # 可选 fixed / discourse / hybrid / hmm

# ---- HMM 分块超参（与验证配置一致；改动会使块缓存失效）----
HMM_BIC_COEF = 2.0
