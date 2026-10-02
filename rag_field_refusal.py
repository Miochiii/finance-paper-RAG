"""Development refusal rule for explicit missing requested fields.

Historical rag_refusal and frozen experiment labels stay unchanged. This rule
uses only public field labels and the answer, never expected values or scores.
"""
import re

from rag_refusal import explicit_refusal
from rag_answer_quote_v4 import presentation_map

_FIELD_ABSENT=re.compile(r"(?:未(?:能|在|从).{0,30}(?:找到|查到)|未找到|没有找到)")
_OVERALL_ABSENT=re.compile(r"无法.{0,35}(?:参考上下文|所给材料|参考资料).{0,12}(?:提供|给出).{0,15}(?:所要求|所需|字段)")


def field_refusal(answer,labels):
    if explicit_refusal(answer):return dict(refusal=1,reason="legacy_contextual_rule")
    body=(answer or "").split("📚",1)[0]
    text,_=presentation_map(body)
    if _OVERALL_ABSENT.search(text):return dict(refusal=1,reason="cannot_supply_requested_fields_from_context")
    if not labels:return dict(refusal=0,reason="no_public_fields")
    missing=set()
    for line in body.splitlines():
        visible,_=presentation_map(line)
        if "：" not in line and ":" not in line:continue
        # NFKC normalizes the colon, while code identifiers keep underscores.
        left,_,right=visible.partition(":")
        for label in labels:
            normalized,_=presentation_map(label)
            if not left.endswith(normalized):continue
            match=_FIELD_ABSENT.search(right)
            if match and not re.search(r"(?:并非|不是|不再|并不是)",right[:match.start()]):
                missing.add(label)
    if set(labels)<=missing:return dict(refusal=1,reason="all_requested_fields_explicitly_not_found")
    return dict(refusal=0,reason="not_all_requested_fields_missing")
