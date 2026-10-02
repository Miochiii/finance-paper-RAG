import numpy as np
import pytest

from rag_fresh_change_experiment import (DEFAULTS, empirical_threshold, schedule,
                                         summarize_alarms, messages, valid_record, sha)
import json


def test_schedule_freezes_disjoint_requests_and_fault_position():
    config = dict(DEFAULTS)
    plan = schedule(["a", "b", "c"], config)
    assert len(plan) == 1300
    assert len({r["request_id"] for r in plan}) == 1300
    assert plan == schedule(["a", "b", "c"], config)
    assert all((r["condition"] == "remove_gold") ==
               (r["phase"] == "fault_test" and r["step"] > 10) for r in plan)
    for phase in ("threshold", "normal_test", "fault_test"):
        assert len([r for r in plan if r["phase"] == phase and r["run"] == 0]) == 30
    streams = [[r["qid"] for r in plan if r["phase"] == p and r["run"] == 0]
               for p in ("threshold", "normal_test", "fault_test")]
    assert streams[0] != streams[1] != streams[2]


def test_schedule_rejects_mixed_disjoint_window():
    with pytest.raises(ValueError):
        schedule(["a"], dict(DEFAULTS, change_at=11))


def test_empirical_rank_uses_calibration_only_and_inclusive_alarm_ties():
    data = np.arange(20).reshape(10, 2)
    threshold = empirical_threshold(data, 0.1)
    assert threshold > 19
    assert threshold < 20
    with pytest.raises(ValueError):
        empirical_threshold(data, 0.01)


def test_no_pre_alarm_is_counted_as_detection_and_censoring_is_reported():
    result = summarize_alarms([0, 0, 3, 0], [5, 12, 0, 30], 30, 10)
    assert result["false_alarms"] == 1
    assert result["pre_alarms"] == 1
    assert result["detected"] == 2
    assert result["undetected"] == 1
    assert result["mean_detected_delay"] == 11
    assert result["restricted_post_delay"] == pytest.approx((2 + 20 + 21) / 3)
    assert result["detect_given_no_pre"] == pytest.approx(2 / 3)


def test_cache_is_per_request_even_for_same_question():
    case = dict(question="q", context="c")
    config = dict(DEFAULTS, system_prompt="s")
    slot = dict(request_id="one", qid="q")
    record = dict(slot=slot, payload_sha256=sha(json.dumps(messages(case, config),
                  ensure_ascii=False, sort_keys=True)), status="ok", response_id="resp1")
    assert valid_record(record, slot, case, config)
    assert not valid_record(record, dict(slot, request_id="two"), case, config)
    assert not valid_record(record, slot, dict(case, context="changed"), config)


def test_payload_carries_no_gold_or_historical_answer():
    payload = messages(dict(question="new question", context="visible evidence",
                            gold_answer="SECRET_GOLD", model_answer="SECRET_OLD"),
                       dict(DEFAULTS, system_prompt="s"))
    assert "SECRET" not in json.dumps(payload)
