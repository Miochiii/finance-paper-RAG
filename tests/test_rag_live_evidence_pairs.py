import pytest
from pathlib import Path

import rag_live_evidence_pairs as v9


@pytest.mark.parametrize("value,wrong", [("3564", "3565"), ("0.0032", "0.0033"),
                                        ("11.07%", "11.08%"), ("-14.5448", "-14.5447"),
                                        ("0.9940", "0.9941")])
def test_display_precision_and_unit_preserved(value, wrong):
    assert v9.display_tick(value) == wrong


def setup(texts, quote, value="45"):
    rows = [dict(text=t, source="paper.pdf", start=100, end=100+len(t)) for t in texts]
    slot = dict(index=1, type="number", expected=value, unit="", quote=quote)
    return dict(retrieved=rows), dict(slots=[slot])


def test_all_duplicate_copies_changed_and_table_coordinates_preserved():
    row = "| Score | 38 | 51 | 45 | 59 |"
    ret, plan = setup([row, row], row)
    fault = v9.corrupt_evidence(ret, plan)
    assert fault["structurally_valid"]
    assert len(fault["edits"]) == 2
    assert all(e["text"] == "| Score | 38 | 51 | 46 | 59 |" for e in fault["evidence"])
    assert ret["retrieved"][0]["text"] == row


def test_redundant_prose_rejected_before_generation():
    row = "| Score | 38 | 51 | 45 | 59 |"
    ret, plan = setup([row + "\n第3个Score为45。"], row)
    fault = v9.corrupt_evidence(ret, plan)
    assert not fault["structurally_valid"]
    assert fault["reason"] == "residual_original_value"


def test_coincidental_other_table_value_rejected_not_globally_replaced():
    row = "| Score | 38 | 51 | 45 | 59 |"
    ret, plan = setup([row + "\n| other | 45 |"], row)
    fault = v9.corrupt_evidence(ret, plan)
    assert not fault["structurally_valid"]
    assert "| other | 45 |" in fault["context"]


def test_no_numeric_substring_edit_and_preserve_percent():
    row = "| error | 11.07% | 11.0701% |"
    ret, plan = setup([row], row, "11.07%")
    fault = v9.corrupt_evidence(ret, plan)
    assert fault["structurally_valid"]
    assert fault["evidence"][0]["text"] == "| error | 11.08% | 11.0701% |"


def test_ambiguous_same_value_inside_cited_row_is_rejected():
    row = "| Score | 45 | 45 | 45 |"
    ret, plan = setup([row], row)
    assert not v9.corrupt_evidence(ret, plan)["structurally_valid"]


def test_generate_cannot_call_client_on_preflight_failure(monkeypatch):
    called = []
    def fail(_):
        raise ValueError("mandatory preflight failed")
    monkeypatch.setattr(v9, "verify_ready", fail)
    monkeypatch.setattr(v9.experiment, "generate", lambda p: called.append(p))
    with pytest.raises(ValueError, match="mandatory preflight"):
        v9.generate(Path("unused_preflight_test_output"))
    assert called == []


def test_unknown_detection_stays_in_bad_denominator():
    rows = [dict(quality="0", proxy=p, refusal=0) for p in ("incomplete", "complete", "U")]
    m = v9.metrics(rows)
    assert (m["bad"], m["bad_flagged"], m["bad_missed"], m["bad_unresolved"]) == (3, 1, 1, 1)
