import numpy as np
import pytest

import change_point_protocol as protocol


def test_query_alignment_for_three_aggregation_modes():
    raw = np.array([[0.0, 1.0, 0.0, 1.0, 1.0, 1.0]])
    values, steps = protocol.aggregate_queries(raw, "raw", 2)
    assert values.shape == (1, 6)
    assert steps.tolist() == [1, 2, 3, 4, 5, 6]
    values, steps = protocol.aggregate_queries(raw, "overlap", 2)
    assert np.allclose(values, [[0.5, 0.5, 0.5, 1.0, 1.0]])
    assert steps.tolist() == [2, 3, 4, 5, 6]
    values, steps = protocol.aggregate_queries(raw, "disjoint", 2)
    assert np.allclose(values, [[0.5, 0.5, 1.0]])
    assert steps.tolist() == [2, 4, 6]


def test_alarm_and_censoring_are_in_query_units():
    scores = np.array([[1.0, 2.0, 10.0], [1.0, 2.0, 3.0]])
    alarm = protocol.alarm_queries(scores, 5.0, np.array([2, 4, 6]))
    assert alarm.tolist() == [6, 0]
    summary = protocol.null_summary(alarm, horizon=6)
    assert summary["far_h"] == 0.5
    assert summary["restricted_mean_queries"] == 6.5


def test_threshold_uses_calibration_maxima_only():
    scores = np.array([[1.0, 2.0], [1.0, 3.0], [2.0, 4.0]])
    threshold = protocol.choose_threshold(scores, 0.1)
    assert threshold > 4.0
    assert protocol.alarm_queries(scores, threshold, np.array([1, 2])).sum() == 0


def test_overlap_conditional_mean_can_exceed_valid_raw_bound():
    raw = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])
    assert protocol.overlapping_conditional_exceed(raw, p=0.03, m=0.05,
                                                   window=2) == pytest.approx(0.25)


def test_drift_summary_excludes_pre_alarm_from_detection():
    stats = protocol.drift_summary(np.array([50, 110, 150, 0]), change_at=100,
                                   post_horizon=20)
    assert stats["pre_alarm_rate"] == 0.25
    assert stats["detect_rate"] == 0.25
    assert stats["detect_rate_given_no_pre"] == pytest.approx(1 / 3)
    assert stats["edd_queries"] == 10.0
