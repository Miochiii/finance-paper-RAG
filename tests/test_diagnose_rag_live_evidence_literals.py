from diagnose_rag_live_evidence_literals import labeled_number


def test_item_index_on_next_line_never_joins_previous_numeric_value():
    assert labeled_number("1）样本数：130482  \n2）样本数B：2346", "样本数") == "130482"


def test_signed_primary_coefficient_and_statistics_stay_separate():
    assert labeled_number("1）α值：-2.18[2.36]\n2）其他：1", "α值") == "-2.18"
