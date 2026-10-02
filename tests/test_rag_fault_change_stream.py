import numpy as np

import rag_fault_change_stream as stream


def test_stream_splices_only_after_change_at():
    pairs = [
        ({"ans_refusal": "0"}, {"ans_refusal": "1"}),
        ({"ans_refusal": "1"}, {"ans_refusal": "0"}),
        ({"ans_refusal": "0"}, {"ans_refusal": "1"}),
        ({"ans_refusal": "0"}, {"ans_refusal": "1"}),
    ]
    orders = np.array([[2, 1, 0, 3], [3, 0, 2, 1]])
    normal, changed = stream.build_streams(pairs, orders, change_at=2, proxy="legacy")
    np.testing.assert_array_equal(changed[:, :2], normal[:, :2])
    np.testing.assert_array_equal(changed[0, 2:], [1, 1])
    np.testing.assert_array_equal(changed[1, 2:], [1, 0])
