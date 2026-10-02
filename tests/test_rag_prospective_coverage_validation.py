import pytest
from rag_prospective_coverage_validation import locate_spans, delete_spans, sentence_units, literal_failure, assess_gate


def test_local_parameter_removal_keeps_neighbor_parameter_and_exact_offsets():
    text="[来源1]（来源: a.pdf）\n最佳参数：max_depth=3 和 n_estimators=100。"
    units=sentence_units(text)
    unit=next(u for u in units if "max_depth" in u['text'])
    spans=locate_spans(dict(support_spans=[dict(unit_id=unit['unit_id'],quote='max_depth=3')]),units)
    assert [text[a:b] for a,b in spans]==['max_depth=3']
    reduced=delete_spans(text,spans)
    assert 'n_estimators=100' in reduced and 'max_depth=3' not in reduced
    assert reduced.startswith('[来源1]')


def test_repeated_normalized_spans_and_invalid_quotes():
    text='[来源1]\n目标ＡＢＣＤ出现，重复目标ABCD出现。'
    unit=sentence_units(text)[1]
    spans=locate_spans(dict(support_spans=[dict(unit_id=unit['unit_id'],quote='目标ABCD')]),[unit])
    assert len(spans)==2
    assert '目标' not in delete_spans(text,spans)
    assert not locate_spans(dict(support_spans=[dict(unit_id=unit['unit_id'],quote='凭空虚构')]),[unit])


def test_headers_cannot_be_removed_and_bad_spans_fail():
    text='[来源1]\n一段原文。'
    assert not locate_spans(dict(support_spans=[dict(unit_id='S0001',quote='[来源1]')]),sentence_units(text))
    with pytest.raises(ValueError):delete_spans(text,[(0,5)])
    with pytest.raises(ValueError):delete_spans(text,[(3,300)])


def test_only_literal_supported_failure_triggers_bounded_retry():
    assert literal_failure(dict(point_results=[dict(index=1,status='supported')]),['U'])
    assert not literal_failure(dict(point_results=[dict(index=1,status='U')]),['U'])


def test_gate_counts_u_against_detection_and_requires_new_sample_size():
    gate=dict(minimum_paired_questions=5,minimum_complete_answers=10,minimum_degraded_pairs=5,
        minimum_non_refusal_degraded_pairs=3,maximum_complete_flags=0,maximum_u_fraction=.2,minimum_incomplete_flag_fraction=.8)
    rows=[dict(qid=str(i),quality='2',signal_status='complete') for i in range(10)]
    rows+=[dict(qid=str(i),quality='1',signal_status='incomplete' if i<3 else 'U') for i in range(5)]
    pairs=[dict(condition='semantic_partial',quality_declined=True,variant_refusal=0,variant_signal='incomplete' if i<3 else 'U') for i in range(5)]
    result=assess_gate(rows,pairs,gate)
    assert result['status']=='not_passed'
    assert not result['checks']['incomplete_flag_fraction']
    assert not result['checks']['all_quiet_degradations_flagged']
