# -*- coding: utf-8 -*-
"""引用/证据审计工具的纯函数测试（不连服务、不调 LLM）。

覆盖三个最容易出错的环节：
  1. 结论拆分：markdown 清理、逗号编号、末尾"参考来源"清单不能当成结论；
  2. 引用识别：线上是 [来源N]，评测答案是（来源N、来源M），两种都要认；
  3. 上下文拼装：去重与字符预算，且编号必须沿用检索序号（否则引用号对不上）；
  4. 统计口径：Wilson 区间与题目级 t 区间的边界行为。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import claim_audit as ca


class TestClaimSplit:
    ANSWER = """根据参考上下文，该论文使用 SMOTE 过采样解决不平衡问题。

1. **采用SMOTE过采样技术**：在预处理阶段对少数类样本过采样（来源2、来源4）。

2. 改进SMOTE算法以提升合成数据质量，引入加权马氏距离与ENN过滤[来源3]。

📚 参考来源: [来源2]、[来源3]、[来源4]
"""

    def test_drops_tail_source_list(self):
        claims = ca.split_claims(self.ANSWER)
        assert all("参考来源" not in c["text"] for c in claims)
        assert all(not c["text"].startswith("📚") for c in claims)

    def test_strips_numbering_and_markdown(self):
        texts = [c["text"] for c in ca.split_claims(self.ANSWER)]
        assert texts, "应至少拆出一条结论"
        assert all(not t.startswith("1.") for t in texts)
        assert all("**" not in t for t in texts)

    def test_reads_both_citation_syntaxes(self):
        claims = ca.split_claims(self.ANSWER)
        cited = [n for c in claims for n in c["cited"]]
        assert 2 in cited and 4 in cited          # （来源2、来源4）
        assert 3 in cited                         # [来源3]

    def test_skips_too_short_and_non_chinese(self):
        claims = ca.split_claims("好的。OK。这是一条足够长的中文结论内容。")
        assert [c["text"] for c in claims] == ["这是一条足够长的中文结论内容"]

    def test_caps_claim_count(self):
        long_answer = "。".join(f"这是第{i}条足够长的中文结论内容示例" for i in range(30))
        assert len(ca.split_claims(long_answer)) == ca.MAX_CLAIMS


class TestContextBlocks:
    @staticmethod
    def _ev(n, source, text, page=1):
        return {"n": n, "source": source, "text": text, "page_start": page, "page_end": page}

    def test_dedup_and_keep_original_numbering(self):
        ev = [self._ev(1, "a.pdf", "内容一"), self._ev(2, "a.pdf", "内容一"),
              self._ev(3, "b.pdf", "内容三")]
        ctx, used = ca.select_context_blocks(ev)
        assert [u["n"] for u in used] == [1, 3]     # 重复块被去掉，编号不改写
        assert "[来源1]" in ctx and "[来源3]" in ctx and "[来源2]" not in ctx

    def test_respects_char_budget(self):
        ev = [self._ev(i, f"{i}.pdf", "长" * 800) for i in range(1, 6)]
        _, used = ca.select_context_blocks(ev, max_chars=1000)
        assert 0 < len(used) < 5


class TestStats:
    def test_wilson_bounds(self):
        lo, hi = ca.wilson(0, 40)
        assert lo == 0.0 and 0.05 < hi < 0.15       # 0/40 的上界不该是 0
        lo, hi = ca.wilson(50, 100)
        assert 0.39 < lo < 0.41 and 0.59 < hi < 0.61

    def test_wilson_empty(self):
        assert ca.wilson(0, 0) == (0.0, 0.0)

    def test_mean_ci_constant_and_spread(self):
        m, lo, hi = ca.mean_ci([0.2, 0.2, 0.2, 0.2])
        assert m == pytest.approx(0.2) and lo == pytest.approx(0.2) and hi == pytest.approx(0.2)
        m, lo, hi = ca.mean_ci([0.0, 0.5, 1.0])
        assert m == pytest.approx(0.5) and lo < m < hi

    def test_mean_ci_single_value(self):
        assert ca.mean_ci([0.3]) == (0.3, 0.3, 0.3)


class TestCleanAnswer:
    def test_clean_answer_keeps_content(self):
        out = ca.clean_answer("**重点**：这是内容。\n\n📚 参考来源: [来源1]")
        assert "**" not in out and "这是内容" in out and "参考来源" not in out
