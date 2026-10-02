from rag_auto_evaluation import (bounds, cluster_interval, paired_bounds, quality_consensus,
                                 select_snippets, validate_faith, validate_reference)


def _quality(quality="2", refusal="0", status="covered"):
    return {"answer_quality": quality, "explicit_refusal": refusal,
            "point_results": [{"index": 1, "status": status}]}


def test_reference_quote_must_exist_in_original_excerpt():
    draft = {"question_clear": "1", "old_reference_adequate": "1", "basis_complete": "1",
             "source_status": "supported", "points": [{"point": "使用模型A", "eid": "E1", "quote": "研究使用随机森林模型"}]}
    verifier = {"question_clear": "1", "old_reference_adequate": "1", "basis_complete": "1", "points_valid": "1"}
    snippets = [{"eid": "E1", "text": "研究使用随机森林模型。"}]
    assert validate_reference(draft, verifier, snippets)["basis_status"] == "supported"
    snippets[0]["text"] = "研究没有说明模型。"
    result = validate_reference(draft, verifier, snippets)
    assert result["basis_status"] == "uncertain"
    assert "quote_not_verified" in result["errors"]


def test_refusal_cannot_be_high_quality_success_and_conflict_is_unknown():
    assert quality_consensus([_quality(refusal="1")]*2, 1, True, 1)["success"] is None
    result = quality_consensus([_quality()]*2, 1, True, 1)
    assert result["explicit_refusal"] == "U" and result["success"] is None
    result = quality_consensus([_quality("0", "1", "missing")]*2, 1, True, 1)
    assert result["success"] == 0 and result["answer_quality"] == "0"
    assert quality_consensus([_quality()]*2, 0, False, 1)["success"] is None


def test_quality_round_disagreement_is_not_forced_to_consensus():
    result = quality_consensus([_quality(), _quality("1", status="partial")], 0, True, 1)
    assert result["answer_quality"] == "U"
    assert result["explicit_refusal"] == "0"
    assert result["success"] is None


def test_refusal_label_is_independent_of_quality_validation():
    invalid = _quality("2", "0", "missing")
    result = quality_consensus([invalid, invalid], 0, True, 1)
    assert result["valid_rounds"] == 0
    assert result["answer_quality"] == "U"
    assert result["explicit_refusal"] == "0"
    assert result["success"] is None


def test_support_requires_a_matching_context_quote():
    row = {"support_status": "supported", "claims": [{"status": "supported", "quote": "模型使用时间加权方法"}]}
    assert validate_faith(row, "模型使用时间加权方法。") == "supported"
    assert validate_faith(row, "仅介绍普通SVM。") == "uncertain"


def test_unknown_bounds_use_all_pairs_not_only_complete_pairs():
    assert bounds([1, 0, None]) == {"n": 3, "known_n": 2, "successes": 1, "unknown_n": 1, "lower": 1/3, "upper": 2/3}
    assert paired_bounds([(1, 0), (None, None)]) == (-1.0, 0.0)
    assert paired_bounds([(1, 0), (1, None)]) == (-1.0, -0.5)


def test_snippet_selection_includes_final_window_of_long_page():
    target = "研究结果包括净重新分类指数和综合判别改善指数。" * 5
    pages = ["甲" * 1600 + target]
    snippets = select_snippets(pages, "净重新分类指数综合判别改善指数", "净重新分类指数", target)
    assert target in snippets[0]["text"]
    assert snippets[0]["pdf_page_1based"] == 1


def test_cluster_interval_keeps_questions_of_same_paper_together():
    items = [("paper1", -1), ("paper1", -1), ("paper2", 0)]
    result = cluster_interval(items, repeats=1000)
    assert result["lower"] == -1 and result["upper"] == 0
    assert result["clusters"] == 2 and result["pairs"] == 3
    assert cluster_interval([("same", -1), ("same", 0)]) is None
