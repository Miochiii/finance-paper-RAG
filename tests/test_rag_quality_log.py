from rag_quality_log import quality_log_fields


def test_quality_log_contains_proxy_without_raw_text():
    row = quality_log_fields("  什么是 XGBoost？ ", "该论文采用 XGBoost。", origin="mcp")
    assert row["ans_refusal_explicit"] == 0
    assert row["origin"] == "mcp"
    assert row["question_hash"] == quality_log_fields("什么是XGBoost？", "答")["question_hash"]
    assert "什么是" not in str(row)
    assert "该论文" not in str(row)


def test_no_evidence_and_error_are_distinct():
    empty = quality_log_fields("问题", "未检索到相关内容。", no_evidence=True)
    failed = quality_log_fields("问题", None)
    assert empty["ans_refusal_explicit"] == 1 and empty["no_evidence"]
    assert failed["ans_refusal_explicit"] is None and not failed["no_evidence"]
