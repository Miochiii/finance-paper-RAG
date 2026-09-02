# retriever.py
# 混合检索 + 重排序：BM25 关键词 + BGE 向量 + BGE-Reranker 精排
import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple

from rag_core.config import MODEL_DIR as _MODEL_DIR, EMBED_MODEL_ID, RERANK_MODEL_ID
os.environ.setdefault("MODELSCOPE_CACHE", _MODEL_DIR)

# BGE 系列强制前缀（不遵守则效果暴跌）
_QUERY_PREFIX = "为这个问题检索相关文档："
_DOC_PREFIX = "检索文档："

# ---------- 懒加载单例 ----------

_EMBEDDING_MODEL = None
_RERANKER_MODEL = None


def _download_from_modelscope(model_id: str) -> str:
    """从 ModelScope 下载模型到 _MODEL_DIR，返回本地路径"""
    from modelscope import snapshot_download
    # ModelScope 上的 BAAI 模型通常以 AI-ModelScope 前缀镜像
    ms_id = f"AI-ModelScope/{model_id}" if not model_id.startswith("AI-ModelScope/") else model_id
    return snapshot_download(ms_id, cache_dir=_MODEL_DIR)


def _get_embedding_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer
        local_path = _download_from_modelscope(EMBED_MODEL_ID)
        _EMBEDDING_MODEL = SentenceTransformer(local_path, device="cuda")
    return _EMBEDDING_MODEL


def _get_reranker():
    """
    返回 (tokenizer, model) 元组。
    使用原生 transformers 接口，不依赖 FlagEmbedding，
    彻底避开 prepare_for_model 等旧 API 兼容问题。
    """
    global _RERANKER_MODEL
    if _RERANKER_MODEL is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        local_path = _download_from_modelscope(RERANK_MODEL_ID)
        tokenizer = AutoTokenizer.from_pretrained(local_path)
        model = AutoModelForSequenceClassification.from_pretrained(local_path)
        model.eval()
        model.to("cuda")
        _RERANKER_MODEL = (tokenizer, model)
    return _RERANKER_MODEL


# ---------- 中文分词（BM25 需要） ----------

_jieba_dict_loaded = False
_jieba_dict_path = None


def _dict_file() -> str:
    """当前语料的 jieba 用户词典；语料无 dict.txt 时回退内置金融词典。"""
    try:
        from rag_core import corpus
        p = corpus.runtime_paths()
        cand = p.get("dict") or ""
        if cand and os.path.isfile(cand):
            return cand
    except Exception:
        pass
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "finance_dict.txt")


def _ensure_jieba_dict():
    """懒加载用户词典；语料切换（词典路径变化）后自动重置重载，旧语料术语不残留。"""
    global _jieba_dict_loaded, _jieba_dict_path
    path = _dict_file()
    if _jieba_dict_loaded and _jieba_dict_path == path:
        return
    try:
        import jieba
        jieba.dt = jieba.Tokenizer()   # 重置分词器：清掉旧用户词典
        if os.path.isfile(path):
            try:
                jieba.load_userdict(path)
            except Exception:
                pass
        _jieba_dict_loaded = True
        _jieba_dict_path = path
    except ImportError:
        pass


def _tokenize(text: str) -> List[str]:
    """jieba 分词（含金融词典），用于 BM25 索引和检索"""
    try:
        _ensure_jieba_dict()
        import jieba
        return list(jieba.cut(text))
    except ImportError:
        # 无 jieba 时按字符切分（效果折扣但可用）
        return list(text)


# ---------- 元数据过滤（年份/作者/方法/任务） ----------

def normalize_filters(filters: Optional[Dict]) -> Optional[Dict]:
    """清洗过滤条件：丢弃空值；全部为空返回 None（= 不过滤）。
    支持键：year_min / year_max（整数年份，闭区间）、
    authors / methods / tasks（字符串或字符串列表）。"""
    if not filters:
        return None
    out: Dict = {}
    for k in ("year_min", "year_max"):
        v = filters.get(k)
        if v is None or isinstance(v, bool):
            continue
        try:
            out[k] = int(v)
        except (TypeError, ValueError):
            continue
    for k in ("authors", "methods", "tasks"):
        v = filters.get(k)
        if isinstance(v, str):
            v = [v]
        if isinstance(v, (list, tuple)):
            items = [str(x).strip() for x in v if x and str(x).strip()]
            if items:
                out[k] = items
    return out or None


def match_meta_filter(meta: Dict, filters: Optional[Dict]) -> bool:
    """块元数据是否满足过滤条件（供检索时对候选块做掩码）。
    - year_min / year_max：年份闭区间（块缺少年份则不通过）；
    - authors / methods / tasks：任一命中即通过，双向包含匹配
      （"随机森林"可命中"随机森林、XGBoost"这类写法差异）。"""
    if not filters:
        return True
    m = meta or {}
    for bound in ("year_min", "year_max"):
        if filters.get(bound) is not None:
            y = m.get("year")
            if not isinstance(y, int):
                return False
            if bound == "year_min" and y < filters[bound]:
                return False
            if bound == "year_max" and y > filters[bound]:
                return False

    def _hit(field: str, wanted: list) -> bool:
        vals = m.get(field)
        if vals is None:
            vals = []
        if isinstance(vals, str):
            vals = [vals]
        for v in vals:
            v = str(v)
            for w in wanted:
                w = str(w)
                if w and (w in v or v in w):
                    return True
        return False

    for field in ("author", "methods", "tasks"):
        wanted = filters.get(field + "s" if field == "author" else field)
        if wanted:
            if not _hit(field, wanted):
                return False
    return True


# ---------- RRF 融合 ----------

def _rrf_fusion(
    bm25_hits: List[Tuple[int, float]],
    vector_hits: List[Tuple[int, float]],
    k: int = 60,
) -> List[Tuple[int, float]]:
    """Reciprocal Rank Fusion：合并两路召回结果"""
    scores: Dict[int, float] = {}
    for rank, (doc_id, _) in enumerate(bm25_hits):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, (doc_id, _) in enumerate(vector_hits):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ========== 主类 ==========

class HybridRetriever:
    """
    混合检索器：
    1. BM25 关键词检索（专业术语精准匹配）
    2. BGE 向量语义检索（同义表达召回）
    3. RRF 融合两路结果
    4. BGE-Reranker 精排
    5. 返回 Top-K 文本块
    """

    def __init__(self, vector_db_path: str = None):
        self.chunks: List[str] = []
        self.metadatas: List[Dict] = []
        self._bm25_index = None
        self._bm25_docs: List[List[str]] = []
        self._vector_client = None
        self._collection_name = "rag_chunks"
        self._vector_db_path = vector_db_path or os.path.join(
            _MODEL_DIR, "vector_db"
        )

    def close(self):
        """关闭底层 Qdrant 本地存储句柄并释放文件锁。
        重建索引前必须调用：qdrant 本地模式同一存储目录只允许一个客户端实例，
        不关闭旧实例就直接新建会抛 'already accessed by another instance'。"""
        if self._vector_client is not None:
            try:
                self._vector_client.close()
            except Exception:
                pass
            self._vector_client = None

    # ===== 索引构建 =====

    def index(self, chunks: List[str], metadatas: List[Dict] = None) -> int:
        """构建 BM25 + 向量双索引，返回已索引块数"""
        if not chunks:
            return 0
        self.chunks = list(chunks)
        self.metadatas = list(metadatas) if metadatas else [{}] * len(chunks)
        self._build_bm25()
        self._build_vector()
        return len(self.chunks)

    def _build_bm25(self):
        from rank_bm25 import BM25Okapi
        self._bm25_docs = [_tokenize(ch) for ch in self.chunks]
        self._bm25_index = BM25Okapi(self._bm25_docs)

    def _build_vector(self):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct

        model = _get_embedding_model()
        doc_texts = [_DOC_PREFIX + ch for ch in self.chunks]
        embeddings = model.encode(
            doc_texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32,
        )

        os.makedirs(self._vector_db_path, exist_ok=True)
        # qdrant 本地模式同名集合 delete+create 会残留旧存储段（孤儿点，id 越界），
        # 曾导致检索时 chunks[cid] IndexError（fin_035 及换 MinerU 基座后四方法全中）。
        # 根治：每次重建前清空整个存储目录，保证点集与当前 chunks 完全一致。
        import shutil
        for entry in os.listdir(self._vector_db_path):
            p = os.path.join(self._vector_db_path, entry)
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    os.remove(p)
            except OSError:
                pass
        self._vector_client = QdrantClient(path=self._vector_db_path)

        collections = [
            c.name
            for c in self._vector_client.get_collections().collections
        ]
        if self._collection_name in collections:
            self._vector_client.delete_collection(self._collection_name)

        dim = embeddings.shape[1]
        self._vector_client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

        points = [
            PointStruct(
                id=i,
                vector=embeddings[i].tolist(),
                payload={"text": self.chunks[i][:200]},
            )
            for i in range(len(embeddings))
        ]
        batch_size = 500
        for start in range(0, len(points), batch_size):
            self._vector_client.upsert(
                collection_name=self._collection_name,
                points=points[start : start + batch_size],
            )

    def add_chunks(self, new_chunks: List[str], new_metadatas: List[Dict] = None) -> int:
        """增量追加块（方案 A）：只嵌入新增块并 upsert，已有块向量原地不动。
        点 id 从 len(chunks) 顺延（无删除则永远连续，检索下标安全）。
        嵌入维度与现有集合不一致（换了嵌入模型）时回退全量重建。"""
        if not new_chunks:
            return 0
        metas = list(new_metadatas) if new_metadatas else [{}] * len(new_chunks)
        self.chunks.extend(new_chunks)
        self.metadatas.extend(metas)

        if self._vector_client is None:
            # 尚无向量索引：全量构建兜底
            return self.index(self.chunks, self.metadatas)

        model = _get_embedding_model()
        doc_texts = [_DOC_PREFIX + ch for ch in new_chunks]
        embeddings = model.encode(
            doc_texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32,
        )

        # 维度守卫：嵌入模型更换后旧集合维度不匹配，upsert 必炸 → 全量重建
        try:
            col = self._vector_client.get_collection(self._collection_name)
            dim = int(col.config.params.vectors.size)
        except Exception:
            dim = None
        if dim is not None and int(embeddings.shape[1]) != dim:
            return self.index(self.chunks, self.metadatas)

        from qdrant_client.models import PointStruct
        start_id = len(self.chunks) - len(new_chunks)
        points = [
            PointStruct(
                id=start_id + i,
                vector=embeddings[i].tolist(),
                payload={"text": new_chunks[i][:200]},
            )
            for i in range(len(embeddings))
        ]
        for s in range(0, len(points), 500):
            self._vector_client.upsert(
                collection_name=self._collection_name,
                points=points[s : s + 500],
            )
        self._build_bm25()
        return len(new_chunks)

    # ===== 检索 =====

    def retrieve(
        self,
        query: str,
        bm25_k: int = 20,
        vector_k: int = 20,
        rerank_k: int = 5,
        keywords: List[str] = None,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        混合检索主入口。

        参数:
            query: 用户问题（或用扩展后的查询字串）
            bm25_k: BM25 召回数
            vector_k: 向量召回数
            rerank_k: 重排序后返回数
            keywords: 预提取的关键词列表，增强 BM25 召回
            filters: 元数据过滤（可选），见 match_meta_filter：
                {"year_min": 2020, "year_max": 2024,
                 "authors": ["张三"], "methods": ["随机森林"], "tasks": ["信贷风控"]}
                过滤在候选层生效（BM25 掩码 + 向量过采样后过滤），不影响索引。

        返回:
            [{index, text, metadata}, ...]
        """
        if not self.chunks:
            return []

        # 阶段计时（供可观测性打点读取，见 self.last_timing）
        import time as _time
        self.last_timing = {}

        # 元数据过滤：候选层掩码（过滤后为空直接返回，不碰 GPU）
        mask = None
        if filters:
            mask = [match_meta_filter(m, filters) for m in self.metadatas]
            if not any(mask):
                self.last_timing.update({"bm25_ms": 0.0, "vector_ms": 0.0, "rerank_ms": 0.0})
                return []

        # 合并用户问题 + 关键词作为 BM25 查询
        bm25_input = query
        if keywords:
            bm25_input = query + " " + " ".join(keywords)

        t0 = _time.perf_counter()
        bm25_hits = self._bm25_search(bm25_input, bm25_k, mask)
        t1 = _time.perf_counter()
        vector_hits = self._vector_search(query, vector_k, mask)
        t2 = _time.perf_counter()
        self.last_timing["bm25_ms"] = round((t1 - t0) * 1000, 1)
        self.last_timing["vector_ms"] = round((t2 - t1) * 1000, 1)

        fused = _rrf_fusion(bm25_hits, vector_hits)
        candidate_ids = [cid for cid, _ in fused[: rerank_k * 4]]
        if len(candidate_ids) < rerank_k:
            candidate_ids = [cid for cid, _ in fused]

        if len(candidate_ids) <= rerank_k:
            self.last_timing["rerank_ms"] = 0.0
            return self._format_results(candidate_ids[:rerank_k])

        reranked_ids = self._rerank(query, candidate_ids, rerank_k)
        t3 = _time.perf_counter()
        self.last_timing["rerank_ms"] = round((t3 - t2) * 1000, 1)
        return self._format_results(reranked_ids)

    def _bm25_search(self, query: str, top_k: int, mask: Optional[List[bool]] = None) -> List[Tuple[int, float]]:
        if self._bm25_index is None:
            return []
        tokenized_query = _tokenize(query)
        scores = self._bm25_index.get_scores(tokenized_query)
        if mask is not None:
            # 掩码：被过滤块分数置 -inf，必然排在末尾
            mask_arr = np.asarray(mask, dtype=bool)
            if mask_arr.shape != scores.shape:
                return []
            scores = np.where(mask_arr, scores, -np.inf)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]

    def _vector_search(
        self, query: str, top_k: int, mask: Optional[List[bool]] = None
    ) -> List[Tuple[int, float]]:
        if self._vector_client is None:
            from qdrant_client import QdrantClient
            if not os.path.exists(self._vector_db_path):
                return []
            self._vector_client = QdrantClient(path=self._vector_db_path)

        collections = [
            c.name
            for c in self._vector_client.get_collections().collections
        ]
        if self._collection_name not in collections:
            return []

        model = _get_embedding_model()
        query_vec = model.encode(
            _QUERY_PREFIX + query,
            normalize_embeddings=True,
        )

        # 有掩码时过采样（向量库里没有元数据载荷，过滤在 Python 侧做）
        limit = top_k if mask is None else max(top_k * 3, 20)
        results = self._vector_client.query_points(
            collection_name=self._collection_name,
            query=query_vec.tolist(),
            limit=limit,
        )
        hits: List[Tuple[int, float]] = []
        for hit in results.points:
            # 防御：过滤越界 id（旧存储段残留的孤儿点），保证下游 chunks[cid] 安全
            if not (isinstance(hit.id, int) and 0 <= hit.id < len(self.chunks)):
                continue
            if mask is not None and not mask[hit.id]:
                continue
            hits.append((hit.id, hit.score))
            if len(hits) >= top_k:
                break
        return hits

    def _rerank(
        self, query: str, candidate_ids: List[int], top_k: int
    ) -> List[int]:
        import torch
        tokenizer, model = _get_reranker()

        # 防御：丢弃越界候选 id（不应出现——_vector_search 已过滤，双保险）
        candidate_ids = [cid for cid in candidate_ids
                         if isinstance(cid, int) and 0 <= cid < len(self.chunks)]
        if not candidate_ids:
            return []
        pairs = [(query, self.chunks[cid]) for cid in candidate_ids]
        with torch.no_grad():
            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to("cuda")
            scores = model(**inputs).logits.squeeze(-1).cpu().tolist()

        if isinstance(scores, float):
            scores = [scores]
        ranked = sorted(
            zip(candidate_ids, scores), key=lambda x: x[1], reverse=True
        )
        return [cid for cid, _ in ranked[:top_k]]

    def _format_results(self, chunk_ids: List[int]) -> List[Dict]:
        results = []
        for cid in chunk_ids:
            if cid < len(self.chunks):
                results.append({
                    "index": cid,
                    "text": self.chunks[cid],
                    "metadata": (
                        self.metadatas[cid]
                        if cid < len(self.metadatas)
                        else {}
                    ),
                })
        return results

    # ===== 持久化 =====

    def save_state(self, path: str):
        state = {"chunks": self.chunks, "metadatas": self.metadatas}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def load_state(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.index(state["chunks"], state.get("metadatas", []))


# ========== 便捷函数 ==========

def build_index_from_chunks(
    chunks: List[str],
    metadatas: List[Dict] = None,
    vector_db_path: str = None,
) -> HybridRetriever:
    """便捷函数：直接用分块列表构建检索器"""
    retriever = HybridRetriever(vector_db_path=vector_db_path)
    retriever.index(chunks, metadatas)
    return retriever


# ========== 测试 ==========

if __name__ == "__main__":
    test_chunks = [
        "格林公式揭示了平面闭区域上二重积分与其边界曲线上曲线积分之间的关系。",
        "斯托克斯公式是格林公式在三维空间中的推广。",
        "高等数学下册第10章主要讲述曲线积分与曲面积分。",
        "机器学习中的梯度下降算法通过迭代优化损失函数来更新模型参数。",
        "高斯公式将三重积分转化为闭合曲面上的曲面积分。",
    ]

    print("=== 构建索引 ===")
    retriever = HybridRetriever()
    retriever.index(test_chunks)
    print(f"已索引 {len(test_chunks)} 个块")

    queries = [
        "格林公式是什么",
        "什么是梯度下降",
        "曲线积分和曲面积分的关系",
    ]

    for q in queries:
        print(f"\n=== 查询: {q} ===")
        results = retriever.retrieve(q, bm25_k=5, vector_k=5, rerank_k=3)
        for r in results:
            print(f"  [{r['index']}] {r['text'][:80]}...")
