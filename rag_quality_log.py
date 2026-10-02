# -*- coding: utf-8 -*-
"""从在线问答生成不含原始问题或答案的逐请求质量日志字段。"""

import hashlib
import re

from rag_refusal import explicit_refusal


def quality_log_fields(question: str, answer: str = None, *,
                       origin: str = "internal", no_evidence: bool = False) -> dict:
    normalized = re.sub(r"\s+", "", question or "").casefold()
    fingerprint = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    response = answer or ""
    valid = answer is not None
    return {
        "question_hash": fingerprint,
        "question_chars": len(normalized),
        "answer_chars": len(re.sub(r"\s+", "", response)) if valid else None,
        "ans_refusal_explicit": (1 if no_evidence else int(explicit_refusal(response)))
          if valid else None,
        "no_evidence": bool(no_evidence),
        "origin": origin,
        "quality_proxy_version": "explicit_v1",
    }
