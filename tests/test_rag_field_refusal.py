from rag_field_refusal import field_refusal


def test_field_style_total_missing_and_long_explicit_inability():
    answer="1）n_heads：未在参考上下文中找到\n2）input_size：未在参考上下文中找到\n📚参考来源：[来源1]"
    assert field_refusal(answer,["n_heads","input_size"])["refusal"]==1
    answer="1）max_depth最优参数：未能从参考上下文中找到该信息。\n2）n_estimators最优参数：未能从参考上下文中找到该信息。"
    assert field_refusal(answer,["max_depth最优参数","n_estimators最优参数"])["refusal"]==1
    assert field_refusal("因此，无法依据参考上下文提供所要求的字段信息。",["起始日期","结束日期"])["refusal"]==1


def test_partial_answer_and_incidental_uncertainty_are_not_total_field_refusal():
    answer="n_heads：2\ninput_size：未在参考上下文中找到"
    assert field_refusal(answer,["n_heads","input_size"])["refusal"]==0
    assert field_refusal("市场变化无法预料，n_heads为2，input_size为512。",["n_heads","input_size"])["refusal"]==0
    assert field_refusal("n_heads：并非未能找到，它为2。",["n_heads"])["refusal"]==0
    assert field_refusal("n_heads：2\ninput_size：512\n📚来源中未找到资料",["n_heads","input_size"])["refusal"]==0
