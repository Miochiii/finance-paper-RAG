# 评测说明（分块消融 EVAL）

`evaluate.py` 回答一个问题：**在检索栈（BM25 + BGE 向量 + RRF + 重排）完全固定的前提下，分块方式对检索质量影响多大？** 设计原则：唯一变量 = 分块方式，其余全部固定。

## 1. 数据

| 项 | 说明 |
|---|---|
| 语料 | 38 篇金融/机器学习方向硕博论文 PDF，经 MinerU 解析（`--docs-cache` 指向 `build_docs_cache_v2.py` 生成的缓存）。原始论文非公开，不在本仓库 |
| 标注 | 40 条人工标注问答（`data/annotations/finance_annotations.csv`，非公开；格式见同目录 `.csv.template`） |
| gold_docs | 相关文档名（可多项，`\|` 分隔） |
| gold_chunks | 证据句，**逐字从原文复制**（可多项，`\|` 分隔；允许含字面 `\n`） |
| status | 填 `done` 的行才参与评测，其余跳过 |

## 2. 指标定义

检索固定 top-5（`--skip-gen` 只算检索指标；不带该参数还会算生成指标 EM/F1 + LLM-as-judge 正确性/忠实性，需 API Key）。

**文档级**（相关判定：块的 `metadata.source` 命中任一 gold 文档）

- `recall@5`：top-5 命中 gold 文档的比例；
- `mrr`：首个相关文档命中位次的倒数均值；
- `ndcg@5`：文档粒度 nDCG——每篇相关文档只取它在 top-5 中**首次命中块**的增益（2026-08-28 修正：旧实现按块累积增益导致 nDCG 可 >1，回归测试见 `tests/test_evaluate_metrics.py`）。

**块级**（相关判定：归一化后块文本包含任一 gold 证据句；用于区分分块方法，文档级指标分不出差异）

- `recall@5_c`：top-5 覆盖的证据句比例；
- `mrr_c`：首个含证据块位次的倒数均值；
- `ndcg@5_c`：证据粒度 nDCG——每条证据只计一次增益（同块覆盖多条证据/同证据被多块覆盖均去重）。

## 3. 流程

```bash
# 1) MinerU 输出 → 评测用文档缓存（顺带打印新旧提取的标注证据命中对比）
python build_docs_cache_v2.py --mineru-out <MinerU输出目录> --save data/docs_cache_v2.json

# 2) 评测：5 种分块方法 × 40 题（hmm 会命中共享块缓存，后续重跑很快）
python evaluate.py --methods all --source finance --skip-gen --docs-cache data/docs_cache_v2.json

# 3) 两两配对显著性检验（读 results/ 下 CSV，秒级）
python evaluate.py --ttest-only --methods all --source finance
```

结果逐条写入 `results/{method}_finance.csv`（配对检验的前提）；检验输出为纯 ASCII（避免 Windows 控制台编码问题）。

## 4. 参考结果（38 篇 / 40 题，2026-08-28，`--skip-gen` 均值）

| 指标 | fixed | discourse | hybrid | **hmm（默认）** | hmm_fixed_k |
|---|---|---|---|---|---|
| recall@5 | 0.9250 | **0.9500** | 0.9250 | 0.9250 | **0.9500** |
| mrr | 0.8938 | **0.9375** | 0.9062 | 0.9062 | 0.9300 |
| ndcg@5 | 0.9015 | **0.9408** | 0.9108 | 0.9108 | 0.9347 |
| recall@5_c | 0.4164 | 0.4811 | **0.5168** | 0.4942 | 0.4621 |
| mrr_c | 0.3937 | **0.4946** | 0.4904 | 0.4842 | 0.4479 |
| ndcg@5_c | 0.3120 | 0.3860 | 0.3916 | **0.3930** | 0.3769 |

### 结论

1. **文档级 recall@5 ≥ 92.5%**：检索栈对不同分块方式都稳健，"找对论文"的能力是系统质量的来源；
2. **块级证据召回（0.31–0.52）是主要短板**：由"引用带页码 + 点击直开 PDF 对应页"兜底；
3. **分块方式无统计显著差异**：5 方法两两配对 t 检验 / Wilcoxon 全部 p > 0.05（最接近的 fixed vs hybrid 块级 p≈0.057）。分块方式不是质量瓶颈，默认 hmm（BIC 自适应 + 块缓存加速）可以放心保留。

## 5. 复现注意

- **控制台编码**：Windows 下如乱码，`$env:PYTHONIOENCODING = "utf-8"` 后再跑；`--ttest-only` 输出已做 ASCII 化不受影响；
- **qdrant 单进程锁**：评测用独立的 `results/vector_db`，与运行中的知识库服务互不争锁，但 GPU 共用；
- **hmm 块缓存**：`data/hmm_chunk_cache` 与知识库构建共享（同文本同参数），跑过一次后 hmm 几乎零重算；
- **私有数据勿入库**：`data/` 与 `results/` 已在 `.gitignore`（仅标注格式模板 `.csv.template` 例外）。
