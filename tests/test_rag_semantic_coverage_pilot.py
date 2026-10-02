import pytest

from rag_semantic_coverage_pilot import (sentence_units, mapped_ids, delete_units,
    qualified, retrieval_candidates, validated_plan, direct_score, retrieval_score, proxy_payloads)


def test_partition_and_semantic_support_mapping_remove_repeated_paraphrases():
    text = "[来源1]（来源: a.pdf）\n先调参。调整超参数可提高效果；最终训练模型。"
    units = sentence_units(text)
    assert "".join(u["text"] for u in units) == text
    ids = mapped_ids({"support_units": [dict(unit_id=u["unit_id"], quote=u["text"])
                      for u in units if "超参数" in u["text"]]}, units)
    # Too-short quotes cannot authorize deletion. Whole spans and headers survive exactly.
    assert "调整超参数" not in delete_units(units, ids)
    assert delete_units(units, ids).startswith("[来源1]")
    assert "最终训练模型。" in delete_units(units, ids)


def test_invalid_mapping_and_protected_header():
    units = sentence_units("[来源1]（来源: a.pdf）\n真实证据超过八个字符。")
    assert not mapped_ids({"support_units": [dict(unit_id="S0002", quote="不存在的证据内容abcdefgh")]}, units)
    with pytest.raises(ValueError):
        delete_units(units, {units[0]["unit_id"]})


def test_strict_semantic_gate_keeps_other_points_and_requires_absent_twice():
    assert qualified([["supported", "absent"], ["supported", "absent"]], 2)
    assert not qualified([["supported", "partial"], ["supported", "absent"]], 2)
    assert not qualified([["absent", "absent"], ["supported", "absent"]], 2)
    assert not qualified([["supported", "U"], ["supported", "absent"]], 2)
    assert not qualified([], 2)


def test_retrieval_uses_visible_metadata_only():
    corpus = {"C1": dict(sources=["visible.pdf"], text="some text"),
              "C2": dict(sources=["other.pdf"], text="other text")}
    assert [i for i, _ in retrieval_candidates(corpus, "[来源2]（来源: visible.pdf）\n正文")] == ["C1"]
    assert retrieval_candidates(corpus, "no known source") == []


def test_extracted_basis_needs_literal_quotes_and_scope_verification():
    evidence = [dict(eid="C1", text="这是准确存在于补充证据的完整句子。")]
    raw = dict(basis_complete="1", points=[dict(point="事实", eid="C1", quote="存在于补充证据的完整句子")])
    assert validated_plan(raw, evidence, dict(basis_complete="1", points_valid="1"))["basis_complete"]
    assert not validated_plan(raw, evidence, dict(basis_complete="U", points_valid="1"))["basis_complete"]
    raw["points"][0]["quote"] = "这是不在证据里面的虚构摘录"
    assert not validated_plan(raw, evidence, dict(basis_complete="1", points_valid="1"))["basis_complete"]


def test_u_is_not_imputed_as_normal_and_missing_has_bounded_score():
    assert retrieval_score({}, 2, False)["score"] is None
    assert retrieval_score(dict(point_results=[dict(index=1, status="covered"), dict(index=2, status="missing")]), 2, True)["score"] == .5
    assert retrieval_score(dict(point_results=[dict(index=1, status="U")]), 1, True)["score"] is None


def test_direct_check_validates_quotes_and_status_consistency():
    q, a = "甲和乙的准确率分别多少？", "甲的准确率是90%。"
    raw = dict(status="partial", checks=[dict(demand_quote="乙的准确率", answer_quote="", status="missing")])
    assert direct_score(raw, q, a)["score"] == .5
    raw["checks"][0]["demand_quote"] = "虚构要求"
    assert direct_score(raw, q, a)["score"] is None


def test_proxy_payloads_exclude_private_gold_and_condition():
    case = dict(question="question", context="context", necessary_points=["secret gold"], condition="fault")
    direct, comparison = proxy_payloads(case, "answer", dict(points=[dict(point="retrieved fact")]))
    assert direct == dict(question="question", model_answer="answer")
    assert comparison == dict(question="question", model_answer="answer", temporary_points=["retrieved fact"])
    assert "secret gold" not in str((direct, comparison))
