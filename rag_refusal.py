# -*- coding: utf-8 -*-
"""回答侧的保守拒答代理；保留旧 ans_refusal 指标以便历史结果可比。"""

import re


_REFUSAL = re.compile(
    r"(无法.{0,12}(?:回答|确定|判断|找到|得出)|未(?:找到|提供|明确|提及)|"
    r"没有(?:找到|提供|明确)|信息不足|不足以回答)"
)
_CONTEXT = re.compile(r"(上下文|参考资料|现有文档|文档中|根据提供的|所给材料)")
_LEADING = re.compile(r"^\s*(抱歉|无法|未找到|没有找到|信息不足|根据提供的)")


def explicit_refusal(answer: str) -> float:
    """只在答案明确表示当前证据不足时记 1，避免把一般性“无法判断”算作拒答。"""
    text = re.sub(r"[*_`]", "", answer or "").split("📚", 1)[0]
    head = text[:350]
    cue = _REFUSAL.search(head)
    if not cue:
        return 0.0
    context = _CONTEXT.search(head[:max(180, cue.start() + 30)])
    return 1.0 if context or _LEADING.search(head[:60]) else 0.0
