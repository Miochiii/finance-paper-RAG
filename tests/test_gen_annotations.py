# -*- coding: utf-8 -*-
"""标注扩充工具的纯函数测试（不调 LLM、不连服务）。

标注是评测基准的根，这里守住三件事：
  1. **证据逐字性**：模型引文与原文不一致时必须兜底成原文片段，绝不能把改写过的句子当证据；
  2. **不重复编号**：合并进正式标注时续号不能覆盖既有 fin_0xx；
  3. **抽取容错**：模型输出带 markdown / 多余文字时仍能解析，缺字段的行要丢掉。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gen_annotations as ga


class TestPickExcerpts:
    @staticmethod
    def _chunk(text, page):
        return {"text": text, "page_start": page, "source": "a.pdf"}

    def test_keeps_first_chunk(self):
        chunks = [self._chunk("首块" * 100, 1), self._chunk("方法模型算法" * 50 + "123", 5)]
        out = ga.pick_excerpts(chunks, k=2)
        assert out[0]["page_start"] == 1

    def test_prefers_method_and_number_chunks(self):
        chunks = [self._chunk("摘要" * 100, 1),
                  self._chunk("致谢" * 100, 2),
                  self._chunk("本文方法使用随机森林模型，样本量 1200 条。" * 40, 3)]
        out = ga.pick_excerpts(chunks, k=2)
        assert any("随机森林" in c["text"] for c in out)

    def test_truncates_long_text(self):
        chunks = [self._chunk("长" * 5000, 1)]
        assert len(ga.pick_excerpts(chunks, k=1, max_chars=300)[0]["text"]) == 300

    def test_empty(self):
        assert ga.pick_excerpts([], k=3) == []


class TestVerifyQuote:
    EX = [{"text": "本文使用随机森林模型进行预测。实验样本为2015至2020年数据。", "page_start": 3},
          {"text": "结果显示 AUC 达到 0.91，优于逻辑回归。", "page_start": 9}]

    def test_verbatim_quote_passes(self):
        q, fb = ga.verify_quote("本文使用随机森林模型进行预测。", self.EX, [1])
        assert q == "本文使用随机森林模型进行预测。" and fb is False

    def test_rewritten_quote_falls_back_to_verbatim_sentence(self):
        q, fb = ga.verify_quote("文章采用了随机森林这个模型来做预测工作", self.EX, [1])
        assert fb is True
        assert q in self.EX[0]["text"]           # 兜底后必须是原文子串

    def test_bad_evidence_id_uses_first_excerpt(self):
        q, fb = ga.verify_quote("不存在的引文", self.EX, [99])
        assert fb is True and q.strip()

    def test_whitespace_differences_tolerated(self):
        q, fb = ga.verify_quote("本文使用 随机森林 模型进行预测。", self.EX, [1])
        assert fb is False and "随机森林" in q


class TestSimilarityAndDedup:
    def test_identical_questions_score_high(self):
        assert ga.question_similarity("这篇论文用了什么模型？", "这篇论文用了什么模型？") > 0.99

    def test_different_questions_score_low(self):
        assert ga.question_similarity("这篇论文用了什么模型？", "样本区间是哪几年？") < 0.2

    def test_empty_safe(self):
        assert ga.question_similarity("", "x") == 0.0


class TestParseItems:
    def test_parses_plain_json(self):
        raw = ('{"items":[{"q":"该论文使用了什么模型？","a":"随机森林与逻辑回归。",'
               '"type":"方法","evidence":[1],"quote":"原文句"}]}')
        items = ga.parse_items(raw)
        assert len(items) == 1 and items[0]["type"] == "方法" and items[0]["evidence"] == [1]

    def test_parses_with_markdown_fence(self):
        raw = '```json\n{"items":[{"q":"该论文使用了什么模型？","a":"随机森林。"}]}\n```'
        assert len(ga.parse_items(raw)) == 1

    def test_drops_incomplete_and_bad_evidence(self):
        raw = ('{"items":[{"q":"","a":"随机森林与逻辑回归。"},'
               '{"q":"该论文使用了什么模型？","a":"随机森林。","evidence":"1"}]}')
        items = ga.parse_items(raw)
        assert len(items) == 1 and items[0]["evidence"] == []

    def test_drops_too_short_answer(self):
        """过短的答案（如"是。"）没有评测价值，直接丢。"""
        raw = '{"items":[{"q":"该论文使用了什么模型？","a":"是。"}]}'
        assert ga.parse_items(raw) == []

    def test_garbage_returns_empty(self):
        assert ga.parse_items("模型今天心情不错") == []


class TestNextIds:
    def test_continues_after_existing(self):
        ids = ga.next_ids([f"fin_{i:03d}" for i in range(1, 41)], 3)
        assert ids == ["fin_041", "fin_042", "fin_043"]

    def test_skips_gaps_and_occupied(self):
        assert ga.next_ids(["fin_001", "fin_003"], 3) == ["fin_002", "fin_004", "fin_005"]

    def test_ignores_malformed_ids(self):
        assert ga.next_ids(["", "hotpot_1", "fin_002"], 2) == ["fin_001", "fin_003"]
