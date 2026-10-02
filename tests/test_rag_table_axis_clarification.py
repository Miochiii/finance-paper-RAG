import pytest
from rag_table_axis_clarification import table_cell


def test_data_column_excludes_row_label_and_preserves_literal():
    table="| loan_status | 0 | 0 | 1 |\n| --- | --- | --- | --- |\n| Score | 38 | 51 | 45 |\n| 评级 | B | BBB | BB |"
    assert table_cell(table,"Score",3)["value"]=="45"
    assert table_cell(table,"评级",3)["value"]=="BB"


def test_duplicate_row_or_invalid_data_column_is_not_guessed():
    with pytest.raises(ValueError):table_cell("| Score | 45 |\n| Score | 76 |","Score",1)
    with pytest.raises(ValueError):table_cell("| Score | 45 |","Score",0)
