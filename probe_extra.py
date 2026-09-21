# -*- coding: utf-8 -*-
"""probe_extra.py —— 补两类代理指标（B 步）：检索一致性 + 分布/散布类。

为什么补这两类（E0 的教训）
--------------------------
E0 只测了"检索分数水平"与"回答形态"两类代理，结论是：只有拒答（行为类）稳健，
检索相似度/间隔在 per-query 层面无证据，而**饱和型代理（文档多样性）根本没有功效**。
按开题报告的缺口与 D 组文献（Gupta 的嵌入分布、Greco 的 Fréchet/散布指标），
这里补两类**没被测过**的代理：

A. **检索一致性**（需改写 + 二次检索）
   同一问题换个问法再检索一次，两次结果的重叠程度就是"检索稳不稳"。
   与检索分数无关，是**自洽性**信号：语料/索引/查询分布漂移时最先松动的往往是它。
   - cons_doc_jaccard / cons_chunk_jaccard / cons_top1_agree / cons_overlap3

B. **散布与嵌入分布类**（只需已有证据池 + 本地嵌入）
   - ret_sim_std / ret_sim_iqr / ret_sim_range / ret_sim_skew：相似度**分布形状**（Greco 论点：
     均值与散布同时变化时更灵敏；JSD/MMD 类只看整体差异）；
   - ret_doc_hhi：证据的文档集中度（Herfindahl 指数，越小越分散）；
   - q_evi_cos：问题与证据质心的距离（查询-证据一致性）；
   - ans_evi_cos：答案与证据质心的距离（**答案是否有证据支撑**的嵌入版）；
   - ans_q_cos：答案与问题的距离（答非所问的粗筛）；
   - ans_centroid_dist：答案与"其余答案质心"的距离（离群度，留一法，与顺序无关）。

用法（手动运行）
----------------
    python probe_extra.py --gen-paraphrases      # 1) LLM 生成每题 2 个改写（约 ¥0.5，几分钟）
    python probe_extra.py --fetch                # 2) 用改写再检索一遍（走线上服务，30~60 分钟，可续跑）
    python probe_extra.py                        # 3) 计算并写出新代理
    python probe_validity.py                     # 4) 自动合并新代理，跑批次内统计

输出：`results/proxy_extra_<时间戳>.csv`（qid + 新代理列），probe_validity 会自动读取最新一份。
缓存：改写 `results/_paraphrases_<method>.json`、二次证据 `results/_evidence_cache_para_<method>.json`。
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import find_dotenv, load_dotenv  # noqa: E402

load_dotenv(find_dotenv())

import probe_validity as pv  # noqa: E402  复用证据取用与日志清理

RESULTS_DIR = pv.RESULTS_DIR
LLM_MODEL = "deepseek-chat"
PARAPHRASE_PROMPT = """把下面这个问题改写成 {n} 个**语义相同但措辞不同**的问法。

要求：
1. 保持问的信息点完全一致（同样的对象、同样的问点），只换表达方式；
2. 不要照抄原句，不要添加原文没有的信息，不要给出答案；
3. 每个改写不超过 40 字，风格上覆盖：同义替换、语序调整、口语化提问；
4. 只输出 JSON：{{"items":["改写1","改写2"]}}"""


# --------------------------------------------------------------------------
# 纯函数：一致性 / 散布
# --------------------------------------------------------------------------
def jaccard(a: Sequence, b: Sequence) -> float:
    """集合 Jaccard 相似度（两边都空时返回 1.0，表示"都没召回"也算一致）。"""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def overlap_at_k(a: Sequence, b: Sequence, k: int = 3) -> float:
    """top-k 的召回重叠比例（分母为 k）。"""
    return len(set(a[:k]) & set(b[:k])) / float(max(1, k))


def rank_corr_common(a: Sequence, b: Sequence) -> float:
    """两次检索在**共同命中**上的排名一致性（Spearman；共同项 <3 时返回 nan）。"""
    pos_b = {x: i for i, x in enumerate(b)}
    pairs = [(i, pos_b[x]) for i, x in enumerate(a) if x in pos_b]
    if len(pairs) < 3:
        return float("nan")
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    return pv.spearman(x, y)[0]


def spread_stats(sims: Sequence[float]) -> Dict[str, float]:
    """相似度分布的散布特征（Greco 的"散布类指标"在单查询上的对应物）。"""
    a = np.asarray([s for s in sims if s is not None and not math.isnan(s)], dtype=float)
    if a.size < 2:
        return {k: float("nan") for k in
                ("ret_sim_std", "ret_sim_iqr", "ret_sim_range", "ret_sim_skew")}
    q25, q75 = np.quantile(a, [0.25, 0.75])
    std = float(a.std(ddof=1))
    return {
        "ret_sim_std": std,
        "ret_sim_iqr": float(q75 - q25),
        "ret_sim_range": float(a.max() - a.min()),
        "ret_sim_skew": float((a.mean() - np.median(a)) / std) if std > 1e-9 else 0.0,
    }


def herfindahl(counts: Sequence[int]) -> float:
    """文档集中度（Herfindahl 指数，1=全部来自同一文档，越小越分散）。"""
    c = np.asarray([x for x in counts if x > 0], dtype=float)
    if c.size == 0:
        return float("nan")
    p = c / c.sum()
    return float((p ** 2).sum())


def centroid_distance(vecs: Sequence[np.ndarray], target: np.ndarray,
                      leave_one_out: bool = False) -> float:
    """1 − cos(target, 其余向量的质心)；leave_one_out 时把 target 自身排除（离群度）。"""
    arr = np.asarray([v for v in vecs if v is not None], dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return float("nan")
    if leave_one_out and arr.shape[0] > 1:
        # arr 的第一行视为 target（调用方保证），质心用其余行
        center = arr[1:].mean(axis=0)
        tgt = arr[0]
    else:
        center = arr.mean(axis=0)
        tgt = np.asarray(target, dtype=float)
    cn = np.linalg.norm(center)
    tn = np.linalg.norm(tgt)
    if cn < 1e-9 or tn < 1e-9:
        return float("nan")
    return float(1.0 - float(np.dot(tgt, center) / (cn * tn)))


def chunk_key(text: str, n: int = 80) -> str:
    """块的稳定指纹（检索返回的文本前 n 个非空白字符）。"""
    return re.sub(r"\s+", "", text or "")[:n]


# --------------------------------------------------------------------------
# 步骤 1：生成改写
# --------------------------------------------------------------------------
def _client():
    from openai import OpenAI
    key = os.getenv("deepseek_api")
    if not key:
        raise RuntimeError("未配置 deepseek_api")
    return OpenAI(api_key=key, base_url="https://api.deepseek.com")


def _llm_json(client, system: str, user: str, max_tokens: int = 400,
              retries: int = 3) -> Dict:
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0.4, max_tokens=max_tokens, timeout=90)
            raw = resp.choices[0].message.content or ""
            m = re.search(r"\{.*\}", raw, re.S)
            return json.loads(m.group(0)) if m else {}
        except Exception as e:
            if attempt == retries:
                print(f"    [警告] 改写失败: {str(e)[:70]}")
            time.sleep(2)
    return {}


def build_paraphrases(rows: List[Dict], cache_path: str, n: int = 2, limit: int = 0) -> Dict[str, List[str]]:
    """为评测集里每题生成 n 个改写（缓存续跑；只处理还没有的题）。"""
    cache: Dict[str, List[str]] = {}
    if os.path.isfile(cache_path):
        try:
            cache = json.load(open(cache_path, encoding="utf-8"))
        except Exception:
            cache = {}
    todo = [r for r in rows if r["qid"] not in cache]
    if limit > 0:
        todo = todo[:limit]
    if not todo:
        print("  改写已完整，无需生成")
        return cache
    print(f"  需生成改写 {len(todo)} 题（每题 {n} 个）")
    client = _client()
    for i, r in enumerate(todo, 1):
        d = _llm_json(client, PARAPHRASE_PROMPT.format(n=n), r["question"])
        items = [str(x).strip() for x in (d.get("items") or []) if str(x).strip()]
        if items:
            cache[r["qid"]] = items[:n]
        if i % 20 == 0 or i == len(todo):
            json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
            print(f"  [{i}/{len(todo)}] 已生成 {len(cache)} 题（缓存已落盘）", flush=True)
    json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    return cache


# --------------------------------------------------------------------------
# 步骤 2：改写检索（走线上服务）
# --------------------------------------------------------------------------
def fetch_paraphrase_evidence(rows: List[Dict], para: Dict[str, List[str]], cache_path: str,
                              top_k: int, mmr: bool, limit: int = 0) -> Dict[str, List[List[Dict]]]:
    """对每题的各改写各检索一次；缓存 {qid: [改写1的命中, 改写2的命中]}。"""
    cache: Dict[str, List[List[Dict]]] = {}
    if os.path.isfile(cache_path):
        try:
            cache = json.load(open(cache_path, encoding="utf-8"))
        except Exception:
            cache = {}
    todo = [r for r in rows if r["qid"] in para and r["qid"] not in cache]
    if limit > 0:
        todo = todo[:limit]
    if not todo:
        print("  改写证据已完整，无需再检索")
        return cache
    n_calls = sum(len(para[r["qid"]]) for r in todo)
    print(f"  需检索 {len(todo)} 题 × 改写 = {n_calls} 次（预计 {n_calls * 12 / 60:.0f}~"
          f"{n_calls * 25 / 60:.0f} 分钟）")
    print("  提示：批量期间请勿在别处检索——那些记录会被一并清理")
    before = pv._log_lines()
    done = 0
    for i, r in enumerate(todo, 1):
        hits_list = []
        ok = True
        for q in para[r["qid"]]:
            try:
                hits_list.append(pv.fetch_evidence(q, top_k, mmr))
            except Exception as e:
                print(f"  [{i}/{len(todo)}] {r['qid']} 检索失败: {str(e)[:60]}")
                ok = False
                break
        if ok:
            cache[r["qid"]] = hits_list
            done += 1
        if i % 10 == 0 or i == len(todo):
            json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
            print(f"  [{i}/{len(todo)}] 已补 {done} 题", flush=True)
    json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    pv.clean_live_search_log(before, expected=n_calls)
    return cache


# --------------------------------------------------------------------------
# 步骤 3：计算新代理
# --------------------------------------------------------------------------
def compute_proxies(rows: List[Dict], evi: Dict[str, List[Dict]],
                    para_evi: Dict[str, List[List[Dict]]], embedder) -> List[Dict]:
    """逐题算两类新代理；缺失部分留空（probe_validity 会按 NaN 跳过）。"""
    # 答案/问题的嵌入（用于分布类代理）
    q_texts = [r["question"] for r in rows]
    a_texts = [(r.get("pred_answer") or "")[:1200] for r in rows]
    q_vec = a_vec = None
    if embedder is not None:
        try:
            q_vec = embedder.encode(q_texts, normalize_embeddings=True)
            a_vec = embedder.encode(a_texts, normalize_embeddings=True)
        except Exception as e:
            print(f"  [警告] 嵌入失败，跳过嵌入类代理: {str(e)[:80]}")
    ans_center = np.mean(a_vec, axis=0) if a_vec is not None and len(a_vec) > 1 else None

    out: List[Dict] = []
    for i, r in enumerate(rows):
        qid = r["qid"]
        rec: Dict[str, float] = {"qid": qid}
        hits0 = evi.get(qid) or []
        docs0 = [h["source"] for h in hits0]
        chunks0 = [chunk_key(h.get("text", "")) for h in hits0]

        # A. 检索一致性（改写 vs 原问题）
        ph = para_evi.get(qid) or []
        if ph:
            h1 = ph[0]
            docs1 = [h["source"] for h in h1]
            chunks1 = [chunk_key(h.get("text", "")) for h in h1]
            rec["cons_doc_jaccard"] = jaccard(docs0, docs1)
            rec["cons_chunk_jaccard"] = jaccard(chunks0, chunks1)
            rec["cons_top1_agree"] = 1.0 if (docs0 and docs1 and docs0[0] == docs1[0]) else 0.0
            rec["cons_overlap3"] = overlap_at_k(docs0, docs1, 3)
            rec["cons_rank_corr"] = rank_corr_common(docs0, docs1)
            if len(ph) > 1:
                docs2 = [h["source"] for h in ph[1]]
                rec["cons_doc_jaccard_p2"] = jaccard(docs0, docs2)

        # B. 散布 / 分布类
        if embedder is not None and hits0:
            try:
                dvec = embedder.encode([h.get("text", "")[:1200] for h in hits0],
                                       normalize_embeddings=True)
                sims = [float(np.dot(q_vec[i], v)) for v in dvec]
                rec.update(spread_stats(sims))
                rec["q_evi_cos"] = centroid_distance(dvec, q_vec[i])
                if a_vec is not None:
                    rec["ans_evi_cos"] = centroid_distance(dvec, a_vec[i])
                    rec["ans_q_cos"] = float(np.dot(a_vec[i], q_vec[i]))
                    if ans_center is not None:
                        rec["ans_centroid_dist"] = float(
                            1.0 - float(np.dot(a_vec[i], ans_center) /
                                        (np.linalg.norm(a_vec[i]) * np.linalg.norm(ans_center) + 1e-9)))
            except Exception:
                pass
        from collections import Counter
        rec["ret_doc_hhi"] = herfindahl(list(Counter(docs0).values())) if docs0 else float("nan")
        out.append(rec)
    return out


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="补两类代理：检索一致性 + 分布/散布类")
    ap.add_argument("--method", default="hmm")
    ap.add_argument("--gen-paraphrases", action="store_true", help="第一步：LLM 生成改写")
    ap.add_argument("--fetch", action="store_true", help="第二步：用改写再检索一遍")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 题（0=全部）")
    ap.add_argument("--n-para", type=int, default=2, help="每题几个改写（默认 2）")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--no-mmr", action="store_true")
    ap.add_argument("--no-embed", action="store_true", help="跳过嵌入类代理（无需加载模型）")
    args = ap.parse_args()

    rows = pv.load_answer_rows([args.method])
    if not rows:
        print("[错误] 找不到评测结果 CSV（先跑 evaluate.py）")
        return 1
    if args.limit:
        rows = rows[: args.limit]
    para_path = os.path.join(RESULTS_DIR, f"_paraphrases_{args.method}.json")
    para_evi_path = os.path.join(RESULTS_DIR, f"_evidence_cache_para_{args.method}.json")
    evi_cache = os.path.join(RESULTS_DIR, f"_evidence_cache_{args.method}.json")
    if not os.path.isfile(evi_cache):
        alt = os.path.join(RESULTS_DIR, "_evidence_cache_live.json")
        evi_cache = alt if os.path.isfile(alt) else evi_cache
    evi = json.load(open(evi_cache, encoding="utf-8")) if os.path.isfile(evi_cache) else {}
    print(f"评测 {len(rows)} 题｜原问题证据池 {len(evi)} 题（{os.path.basename(evi_cache)}）")

    para: Dict[str, List[str]] = {}
    if args.gen_paraphrases or args.fetch:
        para = build_paraphrases(rows, para_path, args.n_para, args.limit)
    if args.fetch:
        para_evi = fetch_paraphrase_evidence(rows, para, para_evi_path, args.top_k,
                                            not args.no_mmr, args.limit)
    else:
        para_evi = json.load(open(para_evi_path, encoding="utf-8")) if os.path.isfile(para_evi_path) else {}

    if not para_evi and not evi:
        print("[提示] 既没有改写证据也没有原证据池，只能算不出东西；先跑 --gen-paraphrases --fetch")
    embedder = None if args.no_embed else pv.get_embedder()
    records = compute_proxies(rows, evi, para_evi, embedder)
    stamp = time.strftime("%Y%m%d_%H%M")
    out_path = os.path.join(RESULTS_DIR, f"proxy_extra_{stamp}.csv")
    keys: List[str] = []
    for r in records:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in records:
            row = {}
            for k in keys:
                v = r.get(k)
                row[k] = "" if isinstance(v, float) and math.isnan(v) else v
            w.writerow(row)
    filled = {k: sum(1 for r in records if r.get(k) not in (None, "")) for k in keys if k != "qid"}
    print(f"\n新代理矩阵 → {out_path}")
    for k, v in filled.items():
        print(f"  {k:<24} 有效 {v}/{len(records)}")
    print("\n下一步：跑 probe_validity.py（会自动合并这份新代理并出批次内统计）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
