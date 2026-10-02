from rag_coverage_refinement import minimum_plan, quoted_payload


def example():
    evidence = [dict(eid="C1", text="模型样本外准确率为89.49%，可选参数为3。")]
    raw = dict(basis_complete="1", points=[dict(point="样本外准确率89.49%", eid="C1", quote="模型样本外准确率为89.49%"),
               dict(point="参数为3", eid="C1", quote="可选参数为3。")])
    check = dict(basis_complete="1", mandatory_indices=[1], point_checks=[
                 dict(index=1, necessary="1", supported="1"), dict(index=2, necessary="0", supported="1")])
    return raw, check, evidence


def test_optional_points_are_excluded_without_using_a_gold_answer():
    raw, check, evidence = example()
    plan = minimum_plan(raw, check, evidence)
    assert plan["basis_complete"]
    assert len(plan["points"]) == 1
    assert "89.49%" in quoted_payload("q", "a", plan)["temporary_points"][0]["quote"]


def test_u_or_inconsistent_minimum_mask_does_not_become_normal():
    raw, check, evidence = example()
    check["point_checks"][1]["necessary"] = "U"
    assert not minimum_plan(raw, check, evidence)["basis_complete"]
    check["point_checks"][1]["necessary"] = "1"
    assert not minimum_plan(raw, check, evidence)["basis_complete"]


def test_bogus_quote_or_unsupported_selected_fact_invalidates_plan():
    raw, check, evidence = example()
    raw["points"][0]["quote"] = "证据中没有这个数值92.00%"
    assert not minimum_plan(raw, check, evidence)["basis_complete"]
    raw, check, evidence = example()
    check["point_checks"][0]["supported"] = "U"
    assert not minimum_plan(raw, check, evidence)["basis_complete"]


def test_comparison_schema_contains_quote_but_no_private_fields():
    payload = quoted_payload("q", "a", dict(points=[dict(point="fact", quote="source quote", eid="C1", gold="private")]))
    assert payload == dict(question="q", model_answer="a", temporary_points=[dict(point="fact", quote="source quote")])
