import csv
import io
from pathlib import Path
from unittest.mock import patch

import pytest

import analyze_rag_fault_pairs as analysis


def _csv_text(fault_hit=0):
    rows = [
        {"qid": "q1", "condition": "baseline", "gold_hit": 1,
         "evidence_count": 4, "f1": 0.8, "judge_corr": 5,
         "ans_refusal": 0, "pred_answer": "正常答案", "error": ""},
        {"qid": "q1", "condition": "remove_gold", "gold_hit": fault_hit,
         "evidence_count": 4, "f1": 0.2, "judge_corr": 2,
         "ans_refusal": 1, "pred_answer": "无法确定", "error": ""},
    ]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def test_load_pairs_checks_fidelity():
    with patch.object(Path, "open", return_value=io.StringIO(_csv_text())):
        pairs = analysis.load_pairs(Path("pairs.csv"))
    assert len(pairs) == 1
    assert pairs[0][0]["qid"] == "q1"

    with patch.object(Path, "open", return_value=io.StringIO(_csv_text(fault_hit=1))):
        with pytest.raises(ValueError, match="证据移除"):
            analysis.load_pairs(Path("pairs.csv"))


def test_document_cluster_bootstrap_uses_paired_differences():
    pairs = [
        ({"qid": "q1", "f1": "0.8", "error": ""},
         {"qid": "q1", "f1": "0.2", "error": ""}),
        ({"qid": "q2", "f1": "0.7", "error": ""},
         {"qid": "q2", "f1": "0.5", "error": ""}),
    ]
    mean, interval = analysis.cluster_bootstrap_mean(
        pairs, {"q1": "a.pdf", "q2": "b.pdf"}, "f1", reps=500)
    assert mean == pytest.approx(-0.4)
    assert -0.61 <= interval[0] <= interval[1] <= -0.19
