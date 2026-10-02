import csv
from pathlib import Path

import pytest

from audit_rag_blind_review import FIELDS, agreement, audit, read_review


def _row(rid="R0001"):
    return dict(zip(FIELDS, [rid, "公式是什么？", "x₁", "作答。📚", "1", "0", "2", ""]))


def _write(path, rows, encoding="utf-8-sig"):
    with path.open("w", encoding=encoding, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_lossy_import_preserves_labels_and_blocks_changed_text(work_tmp):
    root = Path(work_tmp)
    template, left, right = [root / name for name in ("template.csv", "left.csv", "right.csv")]
    rows = [_row(), _row("R0002")]
    _write(template, rows)
    _write(right, rows)
    changed = [dict(row) for row in rows]
    changed[0]["model_answer"] = "作答。??"
    changed[0]["reference_answer"] = "x?"
    changed[1]["model_answer"] = "作答。??"
    changed[1]["reference_answer"] = ""
    changed[1]["explicit_refusal_0_1_U"] = "2"
    _write(left, changed, "gbk")
    original = left.read_bytes()
    output = root / "lossy_audit_output"
    result = audit(template, [left, right], output, True, True)
    assert left.read_bytes() == original
    assert result["reviewers"][0]["encoding"] == "gb18030"
    assert result["reference_inconsistent_groups"] == [0, 0]
    assert result["issue_counts"] == {
        "decorative_encoding_loss": 2, "symbol_encoding_loss": 1,
        "content_changed": 1, "invalid_label": 1}
    repair, _ = read_review(output / "reviewer_1_recheck.csv")
    assert repair["R0002"]["reference_answer"] == "x₁"
    assert repair["R0002"]["previous_explicit_refusal"] == "2"
    assert all(repair["R0002"][name] == "" for name in FIELDS[4:7])
    assert "condition" not in repair["R0001"] and "qid" not in repair["R0001"]
    with pytest.raises(FileExistsError):
        audit(template, [left, right], output, True)


def test_agreement_excludes_invalid_missing_and_U():
    values = [("0", "0"), ("1", "0"), ("U", "U"), ("1", "2"), ("", "1")]
    left = {str(i): {"explicit_refusal_0_1_U": a} for i, (a, _) in enumerate(values)}
    right = {str(i): {"explicit_refusal_0_1_U": b} for i, (_, b) in enumerate(values)}
    stat = agreement(left, right, "explicit_refusal_0_1_U")
    assert stat["joint_valid_n"] == 3
    assert stat["exact_including_U"] == 2
    assert stat["joint_certain_n"] == 2
    assert stat["exact_certain"] == 1
    assert stat["cohen_kappa"] == 0
    constant = {"a": {"explicit_refusal_0_1_U": "0"}}
    assert agreement(constant, constant, "explicit_refusal_0_1_U")["cohen_kappa"] is None


def test_audit_rejects_missing_id_before_writing(work_tmp):
    root = Path(work_tmp)
    template, left, right = [root / name for name in ("template.csv", "left.csv", "right.csv")]
    _write(template, [_row(), _row("R0002")])
    _write(left, [_row()])
    _write(right, [_row()])
    with pytest.raises(ValueError, match="ID"):
        audit(template, [left, right], root / "missing_id_output")
    assert not (root / "missing_id_output").exists()
