from rag_subtle_fault_pilot import (benign_prefix, coverage_result, mechanism,
                                   normalized_map, remove_fact, sources)


def test_mapping_preserves_fullwidth_and_subscripts():
    normalized, positions = normalized_map("Ａ B₂\n3")
    assert normalized == "AB23"
    assert positions == [0, 2, 3, 5]


def test_removal_deletes_repeated_target_and_preserves_other_fact():
    context = "[来源1]（来源: a.pdf）\n第一步骤收集数据。第二步骤训练模型。第二步骤训练模型。结束说明。" * 20
    points = [dict(quote="第一步骤收集数据"), dict(quote="第二步骤训练模型")]
    result = remove_fact(context, points)
    assert result is not None
    assert result["target_index"] == 1  # Least total deletion.
    assert "第一步骤收集数据" not in result["context"]
    assert "第二步骤训练模型" in result["context"]
    assert result["removed_texts"]


def test_overlapping_quotes_cannot_be_used_for_partial_deletion():
    context = "同一句话包含第一事实和第二事实。" * 30
    assert remove_fact(context, [dict(quote="第一事实"), dict(quote="第二事实")]) is None


def test_retention_control_keeps_content_and_source_ids():
    base = "[来源1]（来源: original.pdf）\n必要内容。"
    donor = "[来源1]（来源: donor.pdf）\n无关内容。"
    result = benign_prefix(base, donor)
    assert "必要内容。" in result
    assert "[来源2]（来源: original.pdf）" in result
    assert sources(result) == {"donor.pdf", "original.pdf"}


def test_supported_evidence_requires_actual_quote():
    case = dict(necessary_points=["fact"], context="上下文足以支持这一事实")
    raw = dict(point_results=[dict(index=1, status="supported", quote="不存在的足够长引文")])
    assert coverage_result(raw, case) == ["U"]
    raw["point_results"][0]["quote"] = "上下文足以支持这一事实"
    assert coverage_result(raw, case) == ["supported"]


def test_plan_audit_does_not_assume_intervention_success():
    assert mechanism("partial_evidence", ["supported", "absent"], 2) == "yes"
    assert mechanism("partial_evidence", ["supported", "supported"], 2) == "no"
    assert mechanism("retained_evidence_prefix", ["supported", "absent"], None) == "no"
    assert mechanism("near_topic", ["supported", "U"], None) == "U"
