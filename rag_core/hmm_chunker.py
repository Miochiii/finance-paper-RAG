# -*- coding: utf-8 -*-
"""
hmm_chunker.py — HMM 无监督话题分割文本分块

核心思想：
  把一篇文档视为"若干个话题状态按顺序切换"的序列，
  每个句子是该状态下产生的观测。用无监督 HMM 推断状态序列，
  状态切换处 = 话题边界 → 由此得到语义连贯的分块。

设计要点（对应方法论文档）：
  1. 每篇文档独立建模：独立 PCA、独立 HMM、独立 BIC 选 K；
  2. 句子先过滤表格/数字碎片（只建模携带话题信息的自然语言句），再做完全重复句去重，
     复用 retriever 的 bge-base-zh-v1.5 嵌入（GPU 批量），结果磁盘缓存（带分句版本号与句子指纹校验，防静默错位）；
  3. PCA 降至 d=24 维（规避 768 维协方差数值崩溃）；
  4. BIC 自适应确定状态数 K ∈ [K_MIN, K_MAX]，每 K 多随机种子取最优；
     BIC 按"Viterbi 实际用到的状态数"计参，K 上限随建模句子数自适应封顶（每状态平均 ≥ MIN_SENTENCES_PER_STATE 句）；
  5. GaussianHMM，covariance_type="tied"（状态共享协方差，结构上禁止"背诵"重复句的方差塌缩；
     "diag" 保留为消融对比），min_covar 兜底，Baum-Welch 训练；
  6. Viterbi 解码 → 标签映射回全部原始句子 → 话题段 → 过碎段并入相邻较短一侧 → 超限递归切 → 相邻块加重叠；
  7. 与 DISCOURSE 相同先做页眉/页码清洗（_clean_text），保证消融实验唯一变量是分块方式；
  8. 建模句子过少(< MIN_SENTENCES)或依赖缺失时，回退到 DISCOURSE 分块。

依赖：hmmlearn, scikit-learn, numpy（sentence-transformers 由 retriever 提供）
用法：
    from rag_core.hmm_chunker import hmm_chunk
    chunks = hmm_chunk(text, chunk_size=800, overlap_tokens=50)
命令行演示：
    python rag_core/hmm_chunker.py 某文档.txt --k-max 20 --pca-dim 24
"""

import hashlib
import json
import logging
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

# 保证无论从哪个目录运行都能找到 rag_core 包（把项目根目录加入 sys.path）
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import numpy as np
import warnings

# 过滤 hmmlearn 训练过程的已知无害警告（收敛噪声 / 吸收态转移零行），避免刷屏
warnings.filterwarnings("ignore", message="Model is not converging.*")
warnings.filterwarnings("ignore", message="Some rows of transmat_ have zero sum.*")
# hmmlearn 的"Model is not converging"实际走 logging.warning（不是 warnings.warn），单独压制
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

# 默认超参（对应方法论文档：K∈[4,20]，3 种子，diag，n_iter=100）
PCA_DIM = 24
K_MIN = 4
K_MAX = 20
N_SEEDS = 3
N_ITER = 100
MIN_SENTENCES = 40
MIN_SENTENCES_PER_STATE = 10  # K 自适应封顶：每个话题状态平均至少覆盖的句子数（防短文档过参数化）
TINY_SEG_RATIO = 0.25         # 过碎段判定：token < chunk_size*0.25（对应设计文档 chunk_size/4）时并入相邻较短一侧
_MIN_OVERLAP_TINY = 50        # 过碎段合并阈值的下限（防极小 chunk_size 时阈值过低）


# --------------------------------------------------------------------------
# 工具：句子切分（修复版：英文句点不再无条件当句界）
# --------------------------------------------------------------------------
_SPLIT_RE = re.compile(
    r"(?<=[。！？；\n])\s*"                                                        # 中文句末/换行：总是边界
    r"|(?<=[.!?])(?<![\dA-Z][.!?])\s+(?=[A-Z\u4e00-\u9fff])(?![A-Z][A-Za-z]{0,4}\.\s)"  # 英文句末：后接空格+大写/CJK，
                                                                                  # 且前非数字/单大写缩写，且下一个词非缩写（防 Financ. Econ. 链被切）
)


def _split_sentences(text: str) -> List[str]:
    """句子切分。与 chunk_splitter 的旧正则不同：英文句点只在"后接空格 + 大写字母/CJK"时才作句界，
    且排除 数字、单大写字母缩写（J. / U.S.）、下一个词为缩写（Financ. Econ.）的情形，
    避免把 et al. / 2019. / J. Financ. Econ. 等缩写与参考文献炸成碎片
    （本语料 57%~73% 的重复句正是旧正则把英文句点全当句界产生的）。
    已知局限：英文缩写后紧跟中文仍会切一次（如 "Econ. 是期刊"），但配合去重不产生重复点团，影响可控。"""
    return [s.strip() for s in _SPLIT_RE.split(text) if s.strip()]


def _dedupe_sentences(sents: List[str]) -> Tuple[List[str], List[int]]:
    """完全重复句去重：返回 (unique 句列表, 原始句 → unique 下标 映射)。
    重复句（参考文献、页眉、表注等）构成零方差点团 → 协方差塌缩、logL 变正、BIC 失效；
    只建模 unique 句子，解码后再把标签映射回原始序列（重复句共享首次出现的标签，语义一致）。"""
    uniq: List[str] = []
    first: Dict[str, int] = {}
    uniq_of: List[int] = []
    for s in sents:
        idx = first.get(s)
        if idx is None:
            idx = len(uniq)
            first[s] = idx
            uniq.append(s)
        uniq_of.append(idx)
    return uniq, uniq_of


# --------------------------------------------------------------------------
# 表格/数字碎片过滤：只建模携带话题信息的自然语言句
# --------------------------------------------------------------------------
_PROSE_NOISE_RE = re.compile(r"^[0-9.,%\-–—+*/()\[\]{}<>=:：;；×xX±\s'\"“”‘’【】]*$")  # 纯数字/符号行（表格单元格/残留括号）
_PROSE_WORD_RE = re.compile(r"^[A-Za-z]{1,15}$")  # 孤立英文单词（表头：German/Score/rank/LR）
_PROSE_DOTS_RE = re.compile(r"[.…·]{6,}")  # 目录点线行（"1.3 研究思路 ......"），无话题信息


def _is_prose(s: str) -> bool:
    """是否为携带话题信息的自然语言句。
    论文提取文本里结果表/参考文献把大量"0.7442、German、rank"之类的碎片当句子，
    它们在嵌入空间挤成点团（实测 PCA 维方差仅 ~0.01），是协方差塌缩与过分割的另一主因；
    这些碎片不建模，解码后归入前一个建模句所在的话题段。"""
    if len(s) < 2:
        return False
    if _PROSE_NOISE_RE.fullmatch(s):
        return False
    if _PROSE_WORD_RE.fullmatch(s):
        return False
    if _PROSE_DOTS_RE.search(s):
        return False
    return True


# --------------------------------------------------------------------------
# 工具：token 计数 / 重叠 / 递归切分（复用 chunk_splitter，保证同一套配置）
# --------------------------------------------------------------------------
def _count_tokens(text: str) -> int:
    from rag_core.chunk_splitter import _count_tokens as c
    return c(text)


def _token_overlap(text: str, overlap_tokens: int) -> str:
    from rag_core.chunk_splitter import _token_overlap as t
    return t(text, overlap_tokens)


def _recursive_split(text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    from rag_core.chunk_splitter import _recursive_split as r
    return r(text, max_tokens, overlap_tokens)


# --------------------------------------------------------------------------
# 句子嵌入（GPU 批量 + 磁盘缓存：同一文本只嵌入一次，新增文档只算增量）
# --------------------------------------------------------------------------
_SPLIT_VERSION = "hmm-split-v2"  # 分句正则版本号：改动分句逻辑必须递增，否则旧缓存与句子静默错位（v2=英文句点防缩写误切）


def _sents_digest(sents: List[str]) -> str:
    """句子列表指纹（防分句规则变化后旧缓存与当前句子列表静默错位）。"""
    h = hashlib.sha1()
    for s in sents:
        h.update(s.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _embed_sentences(
    sents: List[str],
    cache_key: str,
    cache_dir: Optional[str] = None,
    batch_size: int = 128,
    obs_doc: Optional[str] = None,
) -> np.ndarray:
    """把句子列表批量嵌入为 (n, 768) 归一化向量。
    缓存命中条件：行数与当前句子数一致 且 句子指纹匹配；任一不符视为失效重算。"""
    cache_file = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{cache_key}.npz")
        if os.path.exists(cache_file):
            try:
                with np.load(cache_file) as data:
                    emb = data["embeddings"]
                    if (
                        emb.ndim == 2
                        and emb.shape[0] == len(sents)
                        and str(data["sents_digest"]) == _sents_digest(sents)
                    ):
                        from rag_core.observability import log_event
                        log_event("hmm_cache", kind="embed", hit=True, doc=obs_doc)
                        return emb.astype(np.float32)
            except Exception:
                pass  # 缓存损坏/版本不符/指纹不一致 → 重算

    from rag_core.retriever import _get_embedding_model
    model = _get_embedding_model()  # GPU
    emb = model.encode(
        sents,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    emb = np.asarray(emb, dtype=np.float32)
    if cache_dir and cache_file:
        # 原子落盘：先写临时文件再 rename，避免中断产生半写缓存
        tmp = cache_file + ".tmp.npz"
        np.savez(tmp, embeddings=emb, sents_digest=_sents_digest(sents))
        os.replace(tmp, cache_file)
    from rag_core.observability import log_event
    log_event("hmm_cache", kind="embed", hit=False, doc=obs_doc)
    return emb


# --------------------------------------------------------------------------
# 每篇独立 PCA
# --------------------------------------------------------------------------
def _pca_fit_transform(emb: np.ndarray, pca_dim: int) -> np.ndarray:
    from sklearn.decomposition import PCA
    n, d = emb.shape
    n_comp = max(1, min(pca_dim, n - 1, d))
    pca = PCA(n_components=n_comp, random_state=0)
    return pca.fit_transform(emb).astype(np.float32)


# --------------------------------------------------------------------------
# BIC 自适应选 K + 训练
# --------------------------------------------------------------------------
def _state_usage(model, obs: np.ndarray) -> Tuple[int, int]:
    """Viterbi 解码诊断：返回 (实际用到的状态数, 话题段数)。"""
    labels = model.predict(obs)
    if labels is None or len(labels) == 0:
        return 0, 0
    n_used = int(len(np.unique(labels)))
    n_segments = int(np.sum(labels[1:] != labels[:-1]) + 1)
    return n_used, n_segments


def _select_k_fit(
    obs: np.ndarray,
    k_min: int,
    k_max: int,
    n_seeds: int,
    n_iter: int,
    bic_coef: float = 1.0,
    switch_coef: float = 1.0,
    min_covar: float = 0.01,
    covariance_type: str = "tied",
):
    """
    扫描 K∈[k_min,k_max]，每 K 跑 n_seeds 个随机种子取最优 logL，
    用 BIC = -2*logL + n_params*ln(n) + switch_coef*边界数*ln(n) 选最小 K。
    修复点：
    - covariance_type 默认 "tied"（状态共享协方差）：结构上禁止单状态把重复句
      点团的方差塌缩成 0（diag 下 logL 可被抬到正的大值），BIC 更可信；
      "diag" 保留为消融对比；
    - n_params 按"Viterbi 实际用到的状态数 n_used"计参（名义 K 大但存在死状态/
      吸收态时不虚增惩罚，也不为无效状态漏惩），配合调用方的 K 上限自适应封顶；
      计参公式随协方差类型变化：tied = n_used(n_used-1)+(n_used-1)+d·n_used+d（共享协方差只计一次）；
    - switch_coef：MDL 两段式编码视角——每个话题边界都要编码其位置（代价 ln(n)），
      防止"每句切换到更近状态"的标签抖动模型靠发射似然优势压低 BIC 胜出
      （实测：K=5 抖动模型自转移 0.44 vs K=4 稳定模型 0.92，后者才是合理分割）；
    - min_covar 抬高（默认 0.01，hmmlearn 默认 0.001）兜底防数值退化；
    - 训练出现 NaN（状态后验为 0 的死状态）的种子直接丢弃。
    返回 (best_model, best_k, best_bic, k_scan)，k_scan 为逐 K 诊断信息（打印 BIC 曲线用）。
    观测 obs: (n, d) 已 PCA 降维。
    """
    from hmmlearn.hmm import GaussianHMM

    n, d = obs.shape
    best_k, best_bic, best_model = None, float("inf"), None
    k_scan: List[dict] = []
    for K in range(k_min, k_max + 1):
        best_logl = -float("inf")
        best_m = None
        for seed in range(n_seeds):
            model = GaussianHMM(
                n_components=K,
                covariance_type=covariance_type,
                min_covar=min_covar,
                n_iter=n_iter,
                random_state=seed,
                init_params="stmc",
            )
            try:
                model.fit(obs)
                logl = model.score(obs)
                if not np.isfinite(logl):
                    continue  # 死状态导致 NaN，该种子无效
                if logl > best_logl:
                    best_logl, best_m = logl, model
            except Exception:
                continue  # 该种子失败则跳过
        if best_m is None:
            continue  # K 完全无法训练
        n_used, n_segments = _state_usage(best_m, obs)
        if n_used <= 0:
            continue  # 解码失败，该 K 不可用
        if covariance_type == "tied":
            n_params = n_used * (n_used - 1) + (n_used - 1) + d * n_used + d
        else:
            n_params = n_used * (n_used - 1) + (n_used - 1) + 2 * d * n_used
        bic = -2.0 * best_logl + bic_coef * n_params * np.log(n) \
            + switch_coef * max(n_segments - 1, 0) * np.log(n)
        self_trans = float(np.nanmean(np.diag(best_m.transmat_))) if best_m.transmat_.size else 0.0
        if not np.isfinite(self_trans):
            self_trans = 0.0
        k_scan.append({
            "k": K,
            "logl": float(best_logl),
            "bic": float(bic),
            "n_params": n_params,
            "n_used": n_used,
            "n_segments": n_segments,
            "self_trans": self_trans,
        })
        if bic < best_bic:
            best_bic, best_k, best_model = bic, K, best_m
    return best_model, best_k, best_bic, k_scan


# --------------------------------------------------------------------------
# 后处理：话题段 → 最终块
# --------------------------------------------------------------------------
def _smooth_labels(labels: np.ndarray, min_run: int) -> np.ndarray:
    """最小段长约束（消除标签抖动）：把长度 < min_run 的标签游程并入相邻较长一侧。
    GaussianHMM 的发射似然收益常常覆盖切换代价，Viterbi 输出在相邻状态间逐句抖动
    （实测自转移仅 0.33~0.45）；话题段必须有一定篇幅才有意义，故对解码结果做
    run-length 平滑——这是话题分割文献的标准后处理（最小段长/时长约束的近似）。"""
    if min_run <= 1 or len(labels) == 0:
        return labels
    n = len(labels)
    runs = []  # [start, end, label]
    i = 0
    while i < n:
        j = i + 1
        while j < n and labels[j] == labels[i]:
            j += 1
        runs.append([i, j, labels[i]])
        i = j
    changed = True
    while changed and len(runs) > 1:
        changed = False
        new_runs: List[list] = []
        m = len(runs)
        for idx in range(m):
            s, e, lab = runs[idx]
            if e - s >= min_run:
                new_runs.append([s, e, lab])
                continue
            left = new_runs[-1] if new_runs else None
            right = runs[idx + 1] if idx + 1 < m else None
            if left is None and right is None:
                new_runs.append([s, e, lab])  # 唯一游程
            elif right is not None and (left is None or (left[1] - left[0]) > (right[1] - right[0])):
                runs[idx + 1] = [s, right[1], right[2]]  # 并入右侧较长游程
                changed = True
            else:
                new_runs[-1] = [left[0], e, left[2]]  # 并入左侧
                changed = True
        runs = new_runs
    out = labels.copy()
    for s, e, lab in runs:
        out[s:e] = lab
    return out


def _chunk_cache_key(text: str, params: Dict) -> str:
    """块级缓存 key = 分句版本 + 全部超参 + 清洗后文本 的 sha1。
    同文本同参数 → 同结果（训练种子固定，确定性），可安全复用。"""
    h = hashlib.sha1()
    h.update(_SPLIT_VERSION.encode("utf-8"))
    for k in sorted(params):
        h.update(f"{k}={params[k]}\n".encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def _merge_tiny_segments(
    seg_texts: List[str], chunk_size: int, min_block_tokens: Optional[int] = None
) -> List[str]:
    """token 数 < 阈值 的段并入【相邻较短一侧】（避免'一句一块'）。
    默认阈值 chunk_size*TINY_SEG_RATIO（对应设计文档 chunk_size/4），可显式覆盖。"""
    if min_block_tokens is not None:
        min_tokens = max(_MIN_OVERLAP_TINY, min_block_tokens)
    else:
        min_tokens = max(_MIN_OVERLAP_TINY, int(chunk_size * TINY_SEG_RATIO))
    segs = [s.strip() for s in seg_texts if s.strip()]
    merged: List[str] = []
    i, n = 0, len(segs)
    while i < n:
        seg = segs[i]
        if _count_tokens(seg) >= min_tokens:
            merged.append(seg)
            i += 1
            continue
        if not merged and i + 1 >= n:
            # 全文仅这一段：保留（孤段无可并入对象）
            merged.append(seg)
            i += 1
            continue
        left = merged[-1] if merged else None
        right = segs[i + 1] if i + 1 < n else None
        if right is not None and (left is None or _count_tokens(left) > _count_tokens(right)):
            # 并入右侧较短邻居；合并块放回原地，下一轮继续检查（可能仍为碎段，链式合并）
            segs[i + 1] = seg + "\n" + right
            i += 1
        else:
            # 并入左侧（左侧较短或右侧不存在）
            merged[-1] = left + "\n" + seg
            i += 1
    return merged


def _segments_to_chunks(
    seg_texts: List[str],
    chunk_size: int,
    overlap_tokens: int,
    min_block_tokens: Optional[int] = None,
) -> List[str]:
    """合并过碎段 → 超限段递归切 → 相邻块之间加重叠（重叠在块开头，承接上一块）。"""
    merged = _merge_tiny_segments(seg_texts, chunk_size, min_block_tokens)

    # 超长话题段用 _recursive_split 递归切（与 discourse/hybrid 同一函数、同一配置）
    base: List[str] = []
    for seg in merged:
        if _count_tokens(seg) > chunk_size:
            base.extend(_recursive_split(seg, chunk_size, overlap_tokens))
        else:
            base.append(seg)

    # 相邻块之间补重叠（覆盖话题边界处）
    final: List[str] = []
    prev = ""
    for c in base:
        if not final:
            final.append(c)
        else:
            ov = _token_overlap(prev, overlap_tokens) if overlap_tokens > 0 else ""
            final.append((ov + "\n" + c) if ov else c)
        prev = c
    return final


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------
def hmm_chunk(
    text: str,
    chunk_size: int = 800,
    overlap_tokens: int = 50,
    cache_dir: Optional[str] = None,
    pca_dim: int = PCA_DIM,
    k_min: int = K_MIN,
    k_max: int = K_MAX,
    n_seeds: int = N_SEEDS,
    n_iter: int = N_ITER,
    min_sentences: int = MIN_SENTENCES,
    batch_size: int = 128,
    min_block_tokens: Optional[int] = None,
    bic_coef: float = 1.0,
    switch_coef: float = 1.0,
    min_covar: float = 0.01,
    covariance_type: str = "tied",
    dedupe: bool = True,
    prose_filter: bool = True,
    min_run: int = 5,
    chunk_cache_dir: Optional[str] = None,
    verbose: bool = False,
    obs_doc: Optional[str] = None,
) -> List[str]:
    """
    HMM 无监督话题分割分块。返回 List[str]。
    - 与 DISCOURSE 一致先做 _clean_text 清洗（保证消融公平）；
    - 分句只把"后接空格+大写/CJK"的英文句点当句界（不炸参考文献缩写）；
    - prose_filter=True 时过滤表格/数字碎片，只建模自然语言句（碎片归入前一个建模句所在段）；
    - dedupe=True 时完全重复句去重后建模，标签映射回原始序列（重复句共享标签）；
    - covariance_type 默认 "tied"（共享协方差，结构上防状态级方差塌缩；"diag" 为消融）；
    - K 上限按建模句数自适应封顶（每状态平均 ≥ MIN_SENTENCES_PER_STATE 句），
      k_min==k_max 时视为"固定 K 对比组"不封顶；
    - BIC 按 Viterbi 实际用到的状态数计参；switch_coef 给每个话题边界加编码成本
      （MDL 两段式编码：每个边界要记录位置），抑制"发射增益覆盖切换代价"的标签抖动模型；
    - min_run：解码后做最小段长平滑（< min_run 句的标签游程并入相邻较长一侧），
      消除 HMM 在相邻状态间的逐句抖动（话题段必须有一定篇幅）；
    - chunk_cache_dir：块级缓存目录。同文本同参数只算一次，命中直接返回
      （首次全量后，evaluate.py 重跑 hmm 与 fixed 一样快）；
    - 依赖 hmmlearn/sklearn 缺失，或建模句数 < min_sentences 时，回退 DISCOURSE 分块。
    """
    if not text or not text.strip():
        return []

    # 粘贴/导出文本常见问题：换行被转义成字面量 "\n"（test.txt 即如此，全文无真实换行）。
    # 先还原为真实换行，否则表格单元格全部黏成一整句，分句/碎片过滤/话题建模全部失效。
    if "\\n" in text:
        text = text.replace("\\r\\n", "\n").replace("\\n", "\n")

    # 与 DISCOURSE baseline 对齐：先做同样的页眉/页码清洗（split_discourse_advanced 内部也会 _clean_text），
    # 保证消融实验里 HMM 与 discourse 的输入处理完全一致（唯一变量 = 分块方式）
    from rag_core.chunk_splitter import _clean_text
    text = _clean_text(text)
    if not text.strip():
        return []

    # 块级缓存：同文本同参数只算一次（训练种子固定 → 确定性）。
    # 命中直接返回，跳过 嵌入 + BIC 扫描 + 解码 全部重活。
    _chunk_file = None
    if chunk_cache_dir:
        os.makedirs(chunk_cache_dir, exist_ok=True)
        _cache_params = {
            "chunk_size": chunk_size, "overlap_tokens": overlap_tokens,
            "pca_dim": pca_dim, "k_min": k_min, "k_max": k_max,
            "n_seeds": n_seeds, "n_iter": n_iter, "min_sentences": min_sentences,
            "min_block_tokens": min_block_tokens, "bic_coef": bic_coef,
            "switch_coef": switch_coef, "min_covar": min_covar,
            "covariance_type": covariance_type, "dedupe": dedupe,
            "prose_filter": prose_filter, "min_run": min_run,
            # 表格感知切块开关：改动组块逻辑时必须递增，防止旧缓存静默复用
            "table_safe": "v1",
        }
        _chunk_key = _chunk_cache_key(text, _cache_params)
        _chunk_file = os.path.join(chunk_cache_dir, _chunk_key + ".json")
        if os.path.exists(_chunk_file):
            try:
                with open(_chunk_file, "r", encoding="utf-8") as f:
                    _cached = json.load(f)
                if isinstance(_cached, list) and all(isinstance(c, str) for c in _cached):
                    if verbose:
                        print(f"  [HMM-CACHE] 命中块缓存 {_chunk_key[:12]}（{len(_cached)} 块）")
                    from rag_core.observability import log_event
                    log_event("hmm_cache", kind="chunk", hit=True, doc=obs_doc)
                    return _cached
            except Exception:
                pass  # 缓存损坏 → 重算

    # 依赖检查 → 回退 DISCOURSE（保证与 baseline 同配置）
    try:
        import hmmlearn  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError:
        from rag_core.chunk_splitter import dispatch_chunk
        return dispatch_chunk(text, "DISCOURSE", chunk_size, overlap_tokens)

    sents = _split_sentences(text)

    # 表格/数字碎片过滤：只对携带话题信息的自然语言句建模（"0.7442 / German / rank"
    # 之类碎片在嵌入空间挤成点团，是协方差塌缩与过分割的根源）；
    # 解码后碎片归入前一个建模句所在的话题段
    if prose_filter:
        modeled = [i for i, s in enumerate(sents) if _is_prose(s)]
        if not modeled:
            if verbose:
                print("  [HMM-FALLBACK] 过滤后无自然语言句，回退 DISCOURSE")
            from rag_core.chunk_splitter import dispatch_chunk
            return dispatch_chunk(text, "DISCOURSE", chunk_size, overlap_tokens)
        if verbose and len(modeled) < len(sents):
            print(f"  [HMM] 句子过滤: {len(sents)} → {len(modeled)} 建模句（过滤 {len(sents) - len(modeled)} 个表格/数字碎片）")
    else:
        modeled = list(range(len(sents)))
    model_sents = [sents[i] for i in modeled]

    # 完全重复句去重：只嵌入/建模 unique 句子（重复句 = 零方差点团），
    # 解码后把标签映射回建模句，再携带式映射回全部原始句子
    if dedupe:
        uniq_sents, uniq_of = _dedupe_sentences(model_sents)
        if verbose and len(uniq_sents) < len(model_sents):
            dup_rate = (1.0 - len(uniq_sents) / len(model_sents)) * 100
            print(f"  [HMM] 句子去重: {len(model_sents)} → {len(uniq_sents)}（重复 {dup_rate:.1f}%）")
    else:
        uniq_sents, uniq_of = model_sents, list(range(len(model_sents)))

    if len(uniq_sents) < min_sentences:
        if verbose:
            print(f"  [HMM-FALLBACK] 建模句数 {len(uniq_sents)} < {min_sentences}，回退 DISCOURSE")
        from rag_core.chunk_splitter import dispatch_chunk
        return dispatch_chunk(text, "DISCOURSE", chunk_size, overlap_tokens)

    # 嵌入（磁盘缓存：同一文本只嵌入一次；缓存 key = 分句版本号 + 清洗后全文，防旧缓存错位）
    cache_key = hashlib.sha1((_SPLIT_VERSION + "\x00" + text).encode("utf-8")).hexdigest()
    emb = _embed_sentences(uniq_sents, cache_key, cache_dir, batch_size=batch_size, obs_doc=obs_doc)
    if len(emb) != len(uniq_sents):
        # 缓存加载已校验行数一致，走到这里只可能是编码器异常返回 → 显式回退，绝不静默截断错位
        if verbose:
            print(f"  [HMM-FALLBACK] 嵌入行数 {len(emb)} 与句子数 {len(uniq_sents)} 不一致，回退 DISCOURSE")
        from rag_core.chunk_splitter import dispatch_chunk
        return dispatch_chunk(text, "DISCOURSE", chunk_size, overlap_tokens)

    # 每篇独立 PCA
    obs = _pca_fit_transform(emb, pca_dim)

    # K 上限随建模句数自适应封顶：每状态平均至少 MIN_SENTENCES_PER_STATE 句，
    # 防止短文档用大 K 过参数化（BIC 惩罚不足时无意义地顶到上限）。
    # 显式固定 K（k_min == k_max，如 HMM-固定K 对比组）不封顶。
    if k_max > k_min:
        k_max_eff = min(k_max, max(k_min, len(uniq_sents) // MIN_SENTENCES_PER_STATE))
    else:
        k_max_eff = k_max
    if verbose and k_max_eff != k_max:
        print(f"  [HMM] 建模句数 {len(uniq_sents)} → K 上限自适应封顶: K ∈ [{k_min}, {k_max_eff}]（原上限 {k_max}）")

    # BIC 选 K + 训练（返回逐 K 诊断信息，供打印 BIC 曲线）
    model, best_k, best_bic, k_scan = _select_k_fit(
        obs,
        k_min=k_min,
        k_max=k_max_eff,
        n_seeds=n_seeds,
        n_iter=n_iter,
        bic_coef=bic_coef,
        switch_coef=switch_coef,
        min_covar=min_covar,
        covariance_type=covariance_type,
    )
    if verbose and k_scan:
        print(f"  [HMM] BIC 扫描（K ∈ [{k_min}, {k_max_eff}]，每 K {n_seeds} 种子，{covariance_type}）：")
        for row in k_scan:
            sel = "  <== 选中" if row["k"] == best_k else ""
            avg_seg = len(uniq_sents) / max(row["n_segments"], 1)
            print(
                f"    K={row['k']:>2}  BIC={row['bic']:12.1f}  logL={row['logl']:12.1f}  "
                f"有效状态={row['n_used']:>2}/{row['k']}  段数={row['n_segments']:>4}  平均段长={avg_seg:6.1f}句  "
                f"自转移={row['self_trans']:.2f}{sel}"
            )
    if model is None:
        from rag_core.chunk_splitter import dispatch_chunk
        return dispatch_chunk(text, "DISCOURSE", chunk_size, overlap_tokens)

    # Viterbi 解码（去重序列）→ 映射回建模句 → 携带式映射回全部原始句子
    # （重复句共享首次出现句的标签；过滤掉的表格碎片归入前一个建模句）
    labels = model.predict(obs)
    labels_modeled = labels[np.asarray(uniq_of, dtype=np.int64)]
    labels_all = np.empty(len(sents), dtype=labels.dtype)
    last = labels_modeled[0]
    p = 0
    for j in range(len(sents)):
        if p < len(modeled) and j == modeled[p]:
            last = labels_modeled[p]
            p += 1
        labels_all[j] = last

    # 最小段长平滑：消除发射似然主导导致的逐句标签抖动
    labels_all = _smooth_labels(labels_all, min_run)

    # 话题段：原始序列中相邻状态切换处为边界
    segs: List[str] = []
    start = 0
    for i in range(len(labels_all) - 1):
        if labels_all[i] != labels_all[i + 1]:
            segs.append("\n".join(sents[start : i + 1]))
            start = i + 1
    segs.append("\n".join(sents[start:]))

    chunks = _segments_to_chunks(segs, chunk_size, overlap_tokens, min_block_tokens)
    if _chunk_file:
        # 原子落盘缓存（只缓存成功的最终结果；临时文件+rename 防半写）
        tmp = _chunk_file + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False)
            os.replace(tmp, _chunk_file)
        except OSError:
            pass  # 缓存写入失败不影响本次结果
    if verbose:
        print(f"  [HMM] K={best_k} (BIC={best_bic:.1f}) 段数={len(segs)} 块数={len(chunks)}")
    from rag_core.observability import log_event
    log_event("hmm_cache", kind="chunk", hit=False, doc=obs_doc)
    return chunks


# --------------------------------------------------------------------------
# 命令行演示
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="HMM 分块演示：读取文本文件并分块")
    ap.add_argument("file", help="要分块的文本文件路径")
    ap.add_argument("--chunk-size", type=int, default=800)
    ap.add_argument("--overlap", type=int, default=50)
    ap.add_argument("--k-min", type=int, default=K_MIN)
    ap.add_argument("--k-max", type=int, default=K_MAX)
    ap.add_argument("--pca-dim", type=int, default=PCA_DIM)
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--min-block-tokens", type=int, default=None, help="过碎段合并阈值（默认 chunk_size*0.25，800 时=200；并入相邻较短一侧，可调）")
    ap.add_argument("--bic-coef", type=float, default=1.0, help="BIC 惩罚系数（>1 更偏向小 K，抑制过分割）")
    ap.add_argument("--switch-coef", type=float, default=1.0, help="话题边界编码成本系数（MDL：每个边界付 ln(n) 代价，抑制标签抖动的模型靠 BIC 胜出）")
    ap.add_argument("--min-run", type=int, default=5, help="最小段长平滑：< N 句的标签游程并入相邻较长一侧（消除逐句抖动）")
    ap.add_argument("--min-covar", type=float, default=0.01, help="GaussianHMM 协方差下限（数值兜底）")
    ap.add_argument("--covariance", choices=["tied", "diag"], default="tied", help="协方差类型：tied=状态共享（默认，防塌缩）；diag=各状态独立（消融）")
    ap.add_argument("--no-dedupe", action="store_true", help="关闭完全重复句去重（对比实验用）")
    ap.add_argument("--no-prose-filter", action="store_true", help="关闭表格/数字碎片过滤（对比实验用）")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--chunk-cache-dir", default=None, help="块级缓存目录（默认 data/hmm_chunk_cache；同文本同参数只算一次）")
    ap.add_argument("--no-chunk-cache", action="store_true", help="禁用块级缓存")
    args = ap.parse_args()

    with open(args.file, encoding="utf-8") as f:
        full = f.read()

    cache = args.cache_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "hmm_embed_cache",
    )
    chunk_cache = (
        None if args.no_chunk_cache else (
            args.chunk_cache_dir or os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data",
                "hmm_chunk_cache",
            )
        )
    )
    result = hmm_chunk(
        full,
        chunk_size=args.chunk_size,
        overlap_tokens=args.overlap,
        cache_dir=cache,
        pca_dim=args.pca_dim,
        k_min=args.k_min,
        k_max=args.k_max,
        n_seeds=args.seeds,
        min_block_tokens=args.min_block_tokens,
        bic_coef=args.bic_coef,
        switch_coef=args.switch_coef,
        min_covar=args.min_covar,
        covariance_type=args.covariance,
        dedupe=not args.no_dedupe,
        prose_filter=not args.no_prose_filter,
        min_run=args.min_run,
        chunk_cache_dir=chunk_cache,
        verbose=True,
    )
    print(f"共 {len(result)} 个块：")
    for i, c in enumerate(result):
        print(f"  [{i + 1}] ({_count_tokens(c)} tokens) {c[:60].replace(chr(10), ' ')}...")
