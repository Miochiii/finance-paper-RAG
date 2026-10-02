import numpy as np
import pytest
from compare_rag_fresh_detectors import (bernoulli_cusum, calibration_threshold,
    consecutive_score, page_hinkley, window_count)


def test_short_rules_do_not_look_ahead():
    x = [[0, 1, 1, 0, 1, 1]]
    assert consecutive_score(x).tolist() == [[0, 0, 1, 0, 0, 1]]
    assert window_count(x).tolist() == [[0, 0, 0, 0, 3, 4]]


def test_cusum_matches_one_step_likelihood_and_resets():
    score = bernoulli_cusum([[0, 1, 0, 0, 0]], .2, .6)[0]
    assert score[0] == 0
    assert score[1] == pytest.approx(np.log(3))
    assert score[-1] == 0
    with pytest.raises(ValueError):
        bernoulli_cusum([[.5]], .2, .6)


def test_ph_running_mean_and_future_invariance():
    a = page_hinkley([[0, 0, 1, 1]])
    b = page_hinkley([[0, 0, 1, 0]])
    assert np.all(a[:, :3] == b[:, :3])
    assert a[0, 2] == pytest.approx(1 - 1/3 - .05)
    assert not np.allclose(a, bernoulli_cusum([[0, 0, 1, 1]], .1, .6))


def test_calibration_ties_and_degenerate_null():
    threshold = calibration_threshold(np.zeros((10, 30)))
    assert 0 < threshold < 1e-9
    data = np.arange(20).reshape(10, 2)
    assert calibration_threshold(data) > data.max()
    with pytest.raises(ValueError):
        calibration_threshold(data, .01)
