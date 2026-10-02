from diagnose_rag_table_calculation_literals import result_literal


def test_calc_field_uses_result_not_operand_or_source_index():
    answer='1. A值: 11.2%\n2. B值: 11.08%\n3. 差值: 11.2-11.08=0.1200百分点\n📚 参考来源: [来源1]'
    r=result_literal(answer,'difference')
    assert r['literal']=='0.1200' and r['unit']=='百分点'


def test_negative_sign_math_format_and_units():
    r=result_literal('**3）比值（A除以B）：** \\(−0.3333\\) 倍','ratio')
    assert r['literal']=='-0.3333' and r['unit']=='倍'


def test_duplicate_result_field_kept_unknown():
    assert result_literal('3. 比值: 2.0\n3. 比值: 3.0','ratio')['status']=='U'


def test_scientific_notation_is_not_misread_as_exponent():
    assert result_literal('3. 比值: 1e-3','ratio')['status']=='U'


def test_unlabelled_value_and_prose_tail_kept_unknown():
    assert result_literal('0.3333','ratio')['status']=='U'
    assert result_literal('3. 比值: 0.3333，保留4位小数','ratio')['status']=='U'


def test_numbered_third_field_can_omit_repeated_label():
    r=result_literal('1. 1\n2. 3\n3. 0.3333倍','ratio')
    assert r['status']=='readable' and r['literal']=='0.3333'
