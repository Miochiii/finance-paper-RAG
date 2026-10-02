from rag_quote_span_repair import enrich_short_quotes, agreement_score, answer_payload


def test_short_existing_quote_expands_to_exact_original_span():
    text = "3、模型构建。\n在全部训练集上拟合算法并构建模型。"
    raw = dict(points=[dict(point="模型构建", eid="C1", quote="3、模型构建。")])
    enriched, changes = enrich_short_quotes(raw, [dict(eid="C1", text=text)])
    assert changes
    assert enriched["points"][0]["point"] == raw["points"][0]["point"]
    assert raw["points"][0]["quote"] == "3、模型构建。"
    c = changes[0]
    assert text[c["start"]:c["end"]] == c["quote"]


def test_missing_quote_is_never_repaired_into_a_real_fact():
    raw = dict(points=[dict(point="事实", eid="C1", quote="完全虚构句子。")])
    enriched, changes = enrich_short_quotes(raw, [dict(eid="C1", text="实际存在的另一个句子。")])
    assert not changes
    assert enriched == raw


def test_source_support_is_not_evidence_of_answer_coverage():
    answer = "模型包括XGBoost、神经网络和SVM。"
    raw = dict(point_results=[dict(index=1, status="covered", answer_quote="随机森林、支持向量机、XGBoost、神经网络", missing_elements=[])])
    assert agreement_score([raw, raw], 1, answer, True)["status"] == "U"


def test_two_round_disagreement_keeps_u_and_actual_answer_quotes_pass():
    answer = "模型包括XGBoost、神经网络和SVM。"
    partial = dict(point_results=[dict(index=1, status="partial", answer_quote="XGBoost、神经网络和SVM", missing_elements=["随机森林"])])
    covered = dict(point_results=[dict(index=1, status="covered", answer_quote="XGBoost、神经网络和SVM", missing_elements=[])])
    assert agreement_score([partial, partial], 1, answer, True)["score"] == .5
    assert agreement_score([partial, covered], 1, answer, True)["score"] is None
    payload = answer_payload("q", answer, dict(points=[dict(point="四种模型", quote="原文四种模型", gold="private")]))
    assert set(payload) == {"question", "answer_to_check", "retrieved_facts"}
    assert "private" not in str(payload)


def test_quote_does_not_cross_source_headers_and_long_quotes_stay_unchanged():
    raw = dict(points=[dict(point="模型构建", eid="C1", quote="3、模型构建。")])
    enriched, changes = enrich_short_quotes(raw, [dict(eid="C1", text="[来源1]\n3、模型构建。\n[来源2]\n别的文献。")])
    assert not changes
    assert enriched == raw
