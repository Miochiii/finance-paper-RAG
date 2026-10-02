import csv
from pathlib import Path

import pytest

from analyze_rag_blind_review import _kappa, _load_key, analyze


FIELDS = ["review_id", "question", "reference_answer", "model_answer",
          "reference_adequate_0_1_U", "explicit_refusal_0_1_U",
          "answer_quality_0_1_2_U", "review_notes"]


def _write(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _review_rows():
    return [
        {"review_id": "R0001", "question": "问一", "reference_answer": "答一",
         "model_answer": "答一", "reference_adequate_0_1_U": "1",
         "explicit_refusal_0_1_U": "0", "answer_quality_0_1_2_U": "2", "review_notes": ""},
        {"review_id": "R0002", "question": "问一", "reference_answer": "答一",
         "model_answer": "无法确定", "reference_adequate_0_1_U": "1",
         "explicit_refusal_0_1_U": "1", "answer_quality_0_1_2_U": "0", "review_notes": ""},
    ]


def test_incomplete_review_stays_blind(work_tmp):
    root = Path(work_tmp)
    left, right = root / "r1.csv", root / "r2.csv"
    rows = _review_rows()
    rows[0]["answer_quality_0_1_2_U"] = ""
    _write(left, FIELDS, rows)
    _write(right, FIELDS, rows)
    outputs = analyze(left, right, root / "key-does-not-exist.csv", root / "out")
    assert [p.name for p in outputs] == ["review_progress.md"]
    assert "完成 1/2" in outputs[0].read_text(encoding="utf-8")


def test_agreement_and_adjudicated_condition_summary(work_tmp):
    root = Path(work_tmp)
    left, right, key = root / "r1.csv", root / "r2.csv", root / "key.csv"
    rows1 = _review_rows()
    rows2 = _review_rows()
    rows2[1]["explicit_refusal_0_1_U"] = "0"
    _write(left, FIELDS, rows1)
    _write(right, FIELDS, rows2)
    _write(key, ["review_id", "qid", "condition"], [
        {"review_id": "R0001", "qid": "q1", "condition": "baseline"},
        {"review_id": "R0002", "qid": "q1", "condition": "remove_gold"},
    ])
    out = root / "out"
    outputs = analyze(left, right, key, out)
    assert {p.name for p in outputs} == {
        "review_progress.md", "blind_agreement.md", "blind_disagreements.csv"}
    assert "分歧或 U 的条目 1" in (out / "blind_agreement.md").read_text(encoding="utf-8")
    disagreements = out / "blind_disagreements.csv"
    with disagreements.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1 and rows[0]["review_id"] == "R0002"
    assert "condition" not in rows[0] and "qid" not in rows[0]
    rows[0]["final_reference_adequate"] = "1"
    rows[0]["final_explicit_refusal"] = "1"
    rows[0]["final_answer_quality"] = "0"
    adjudication = root / "adjudicated.csv"
    _write(adjudication, list(rows[0]), rows)
    outputs = analyze(left, right, key, out, adjudication)
    assert outputs[-1].name == "blind_condition_summary.md"
    summary = outputs[-1].read_text(encoding="utf-8")
    assert "baseline：可判 1/1；成功 1/1" in summary
    assert "remove_gold：可判 1/1；成功 0/1" in summary


def test_frozen_template_detects_matching_modified_reviews(work_tmp):
    root = Path(work_tmp)
    left, right, frozen = [root / name for name in ("modified1.csv", "modified2.csv", "frozen.csv")]
    rows = _review_rows()
    _write(frozen, FIELDS, rows)
    rows[0]["reference_answer"] = "另一个答案"
    _write(left, FIELDS, rows)
    _write(right, FIELDS, rows)
    with pytest.raises(ValueError, match="冻结模板"):
        analyze(left, right, root / "missing-key.csv", root / "modified-out", template=frozen)


def test_disagreement_output_does_not_read_key(work_tmp):
    root = Path(work_tmp)
    left, right = root / "blind1.csv", root / "blind2.csv"
    rows = _review_rows()
    _write(left, FIELDS, rows)
    rows[0]["answer_quality_0_1_2_U"] = "1"
    _write(right, FIELDS, rows)
    outputs = analyze(left, right, root / "missing-key.csv", root / "still-blind-out")
    assert outputs[-1].name == "blind_disagreements.csv"


def test_final_reference_adequacy_must_be_question_level(work_tmp):
    root = Path(work_tmp)
    left, right, key = [root / name for name in ("same1.csv", "same2.csv", "paired-key.csv")]
    rows = _review_rows()
    rows[1]["reference_adequate_0_1_U"] = "0"
    _write(left, FIELDS, rows)
    _write(right, FIELDS, rows)
    _write(key, ["review_id", "qid", "condition"], [
        {"review_id": "R0001", "qid": "q1", "condition": "baseline"},
        {"review_id": "R0002", "qid": "q1", "condition": "remove_gold"}])
    with pytest.raises(ValueError, match="充分性不一致"):
        analyze(left, right, key, root / "inconsistent-reference-out")
    assert not (root / "inconsistent-reference-out" / "blind_condition_summary.md").exists()


def test_key_rejects_extra_answer_in_one_condition(work_tmp):
    root = Path(work_tmp)
    key = root / "duplicate-condition.csv"
    _write(key, ["review_id", "qid", "condition"], [
        {"review_id": "R0001", "qid": "q1", "condition": "baseline"},
        {"review_id": "R0002", "qid": "q1", "condition": "baseline"},
        {"review_id": "R0003", "qid": "q1", "condition": "remove_gold"}])
    with pytest.raises(ValueError, match="配对条件"):
        _load_key(key, {"R0001", "R0002", "R0003"})


def test_constant_labels_have_undefined_kappa():
    row = {"explicit_refusal_0_1_U": "0"}
    assert _kappa([(row, row)], "explicit_refusal_0_1_U") == (None, 1)
